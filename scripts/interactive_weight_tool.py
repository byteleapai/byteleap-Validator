#!/usr/bin/env python3
"""
Interactive Weight Setting Tool

A command-line tool for manually setting weights for specific miners using validator wallet.
Supports continuous loop scoring with configurable intervals.
"""

import asyncio
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
from bittensor.utils.weight_utils import (convert_weights_and_uids_for_emit,
                                          process_weights_for_netuid)

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class YumaConsensus:
    """
    Python implementation of Bittensor's Yuma Consensus algorithm.

    Based on the Rust implementation in:
    - subtensor/pallets/subtensor/src/epoch/run_epoch.rs
    - subtensor/pallets/subtensor/src/epoch/math.rs

    Reference: https://github.com/opentensor/subtensor/blob/main/docs/consensus.md
    """

    def __init__(self, kappa: float = 0.5):
        """
        Initialize Yuma Consensus calculator.

        Args:
            kappa: Majority ratio threshold (default 0.5 = 50% stake majority)
        """
        self.kappa = kappa

    def weighted_median(
        self,
        stake: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Calculate stake-weighted median for a single column (miner).

        The weighted median finds the maximum weight w such that at least
        kappa fraction of total stake supports weight >= w.

        Args:
            stake: Normalized stake array for each validator
            weights: Weight array from each validator to this miner

        Returns:
            Stake-weighted median weight
        """
        if len(stake) == 0 or len(weights) == 0:
            return 0.0

        # Create pairs of (weight, stake) and sort by weight descending
        pairs = [
            (float(weights[i]), float(stake[i]))
            for i in range(len(stake))
            if stake[i] > 0
        ]
        if not pairs:
            return 0.0

        pairs.sort(key=lambda x: x[0], reverse=True)

        # Find the weight where cumulative stake reaches kappa threshold
        total_stake = sum(s for _, s in pairs)
        if total_stake <= 0:
            return 0.0

        threshold = self.kappa * total_stake
        cumulative_stake = 0.0

        for weight, s in pairs:
            cumulative_stake += s
            if cumulative_stake >= threshold:
                return weight

        # If we didn't reach threshold, return the minimum weight
        return pairs[-1][0] if pairs else 0.0

    def weighted_median_col(
        self,
        stake: np.ndarray,
        weight_matrix: Dict[int, Dict[int, float]],
        miner_uids: List[int],
    ) -> Dict[int, float]:
        """
        Calculate stake-weighted median for each miner (column).

        Args:
            stake: {validator_uid: stake} normalized stake
            weight_matrix: {validator_uid: {miner_uid: weight}}
            miner_uids: List of all miner UIDs

        Returns:
            {miner_uid: consensus_weight} consensus weights
        """
        validator_uids = list(weight_matrix.keys())
        if not validator_uids:
            return {}

        # Normalize stakes
        stake_array = np.array([stake.get(vid, 0.0) for vid in validator_uids])
        total_stake = stake_array.sum()
        if total_stake > 0:
            stake_array = stake_array / total_stake

        consensus = {}
        for miner_uid in miner_uids:
            # Get weights from all validators for this miner
            weights_array = np.array(
                [
                    weight_matrix.get(vid, {}).get(miner_uid, 0.0)
                    for vid in validator_uids
                ]
            )
            consensus[miner_uid] = self.weighted_median(stake_array, weights_array)

        return consensus

    def col_clip(
        self,
        weight_matrix: Dict[int, Dict[int, float]],
        consensus: Dict[int, float],
    ) -> Dict[int, Dict[int, float]]:
        """
        Clip weights to consensus threshold.

        W̄_ij = min(W_ij, W̄_j)

        Args:
            weight_matrix: {validator_uid: {miner_uid: weight}}
            consensus: {miner_uid: consensus_weight}

        Returns:
            Clipped weight matrix
        """
        clipped = {}
        for vid, weights in weight_matrix.items():
            clipped[vid] = {}
            for miner_uid, weight in weights.items():
                threshold = consensus.get(miner_uid, 0.0)
                clipped[vid][miner_uid] = min(weight, threshold)
        return clipped

    def calculate_validator_trust(
        self,
        clipped_weights: Dict[int, Dict[int, float]],
    ) -> Dict[int, float]:
        """
        Calculate validator trust (vtrust) as sum of clipped weights.

        T_vi = Σ_j W̄_ij

        Since original weights sum to 1, vtrust represents the fraction
        of weights that survived clipping.

        Args:
            clipped_weights: {validator_uid: {miner_uid: clipped_weight}}

        Returns:
            {validator_uid: vtrust}
        """
        vtrust = {}
        for vid, weights in clipped_weights.items():
            vtrust[vid] = sum(weights.values())
        return vtrust

    def calculate_ranks(
        self,
        stake: Dict[int, float],
        clipped_weights: Dict[int, Dict[int, float]],
    ) -> Dict[int, float]:
        """
        Calculate miner ranks from stake-weighted clipped weights.

        R_j = Σ_i (S_i * W̄_ij)

        Args:
            stake: {validator_uid: stake}
            clipped_weights: {validator_uid: {miner_uid: clipped_weight}}

        Returns:
            {miner_uid: rank}
        """
        # Normalize stakes
        total_stake = sum(stake.values())
        if total_stake <= 0:
            return {}

        norm_stake = {vid: s / total_stake for vid, s in stake.items()}

        # Calculate ranks
        ranks: Dict[int, float] = {}
        for vid, weights in clipped_weights.items():
            s = norm_stake.get(vid, 0.0)
            for miner_uid, weight in weights.items():
                if miner_uid not in ranks:
                    ranks[miner_uid] = 0.0
                ranks[miner_uid] += s * weight

        return ranks

    def calculate_incentive(
        self,
        ranks: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Calculate miner incentive (normalized ranks).

        I_j = R_j / Σ_k R_k

        Args:
            ranks: {miner_uid: rank}

        Returns:
            {miner_uid: incentive}
        """
        total_rank = sum(ranks.values())
        if total_rank <= 0:
            return {uid: 0.0 for uid in ranks}
        return {uid: r / total_rank for uid, r in ranks.items()}

    def run(
        self,
        stake: Dict[int, float],
        weight_matrix: Dict[int, Dict[int, float]],
    ) -> Dict[str, Any]:
        """
        Run complete Yuma Consensus calculation.

        Args:
            stake: {validator_uid: stake}
            weight_matrix: {validator_uid: {miner_uid: weight}} (normalized per validator)

        Returns:
            Dict containing:
            - consensus: {miner_uid: consensus_weight}
            - clipped_weights: {validator_uid: {miner_uid: clipped_weight}}
            - validator_trust: {validator_uid: vtrust}
            - ranks: {miner_uid: rank}
            - incentive: {miner_uid: incentive}
        """
        if not weight_matrix or not stake:
            return {
                "consensus": {},
                "clipped_weights": {},
                "validator_trust": {},
                "ranks": {},
                "incentive": {},
            }

        # Get all miner UIDs
        miner_uids: set = set()
        for weights in weight_matrix.values():
            miner_uids.update(weights.keys())
        miner_uids_list = list(miner_uids)

        # Step 1: Calculate consensus (stake-weighted median per miner)
        consensus = self.weighted_median_col(stake, weight_matrix, miner_uids_list)

        # Step 2: Clip weights to consensus
        clipped_weights = self.col_clip(weight_matrix, consensus)

        # Step 3: Calculate validator trust
        validator_trust = self.calculate_validator_trust(clipped_weights)

        # Step 4: Calculate ranks
        ranks = self.calculate_ranks(stake, clipped_weights)

        # Step 5: Calculate incentive
        incentive = self.calculate_incentive(ranks)

        return {
            "consensus": consensus,
            "clipped_weights": clipped_weights,
            "validator_trust": validator_trust,
            "ranks": ranks,
            "incentive": incentive,
        }

    def estimate_vtrust(
        self,
        candidate_weights: Dict[int, float],
        consensus: Dict[int, float],
    ) -> float:
        """
        Estimate vtrust for a candidate weight distribution.

        Args:
            candidate_weights: {miner_uid: weight} (should sum to 1)
            consensus: {miner_uid: consensus_weight}

        Returns:
            Estimated vtrust (0 to 1)
        """
        if not candidate_weights:
            return 0.0

        clipped_sum = 0.0
        for miner_uid, weight in candidate_weights.items():
            threshold = consensus.get(miner_uid, 0.0)
            clipped_sum += min(weight, threshold)

        total_weight = sum(candidate_weights.values())
        if total_weight <= 0:
            return 0.0

        return clipped_sum / total_weight


class InteractiveWeightTool:
    """Interactive tool for setting miner weights"""

    # Background sync interval in seconds
    SYNC_INTERVAL = 30
    # Minimum blocks required between weight submissions (chain rate limit)
    MIN_BLOCKS_BETWEEN_SUBMISSIONS = 100
    # Kappa: 50% stake majority threshold for Yuma Consensus
    KAPPA = 0.5

    def __init__(
        self, wallet_name: str, hotkey_name: str, netuid: int, network: str = "finney"
    ):
        """Initialize the weight tool

        Args:
            wallet_name: Validator wallet name
            hotkey_name: Validator hotkey name
            netuid: Network UID
            network: Bittensor network (finney, test, local)
        """
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = int(netuid)

        bt.logging.info(
            f"🔑 Loading wallet | wallet={wallet_name} hotkey={hotkey_name}"
        )
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

        bt.logging.info(
            f"🌐 Connecting to subtensor | network={network} netuid={netuid}"
        )
        self.subtensor = bt.subtensor(network=network)

        bt.logging.info(f"Loading metagraph | network={network} netuid={netuid}")
        self.metagraph = bt.metagraph(netuid=netuid, network=network)
        self.metagraph.sync(subtensor=self.subtensor)

        self._subtensor_lock = (
            threading.Lock()
        )  # For all subtensor operations (read/write)
        self._metagraph_lock = threading.Lock()  # For metagraph access
        self._last_sync_block: int = 0

        # Start background sync thread
        self._sync_stop_event = threading.Event()
        self._sync_thread = threading.Thread(
            target=self._background_sync_loop, daemon=True
        )
        self._sync_thread.start()

        bt.logging.info(f"✅ Initialized | miners={len(self.metagraph.hotkeys)}")
        bt.logging.info(f"🔄 Background sync started | interval={self.SYNC_INTERVAL}s")

    def _background_sync_loop(self) -> None:
        """Background thread that periodically syncs metagraph and collects data."""
        while not self._sync_stop_event.is_set():
            try:
                self._do_sync_and_collect()
            except Exception as e:
                bt.logging.debug(f"Background sync error: {e}")

            # Wait for next sync interval or stop event
            self._sync_stop_event.wait(timeout=self.SYNC_INTERVAL)

    def _do_sync_and_collect(self) -> None:
        """Perform metagraph sync."""
        try:
            # Use subtensor_sync_lock to prevent concurrent subtensor access
            with self._subtensor_lock:
                with self._metagraph_lock:
                    self.metagraph.sync(subtensor=self.subtensor)
                current_block = self.subtensor.get_current_block()

            if current_block > self._last_sync_block:
                self._last_sync_block = current_block
                bt.logging.debug(f"Sync complete | block={current_block}")

        except Exception as e:
            bt.logging.debug(f"Sync error: {e}")

    def stop_background_sync(self) -> None:
        """Stop the background sync thread."""
        self._sync_stop_event.set()
        if self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)

    def get_real_vtrust(self, uid: Optional[int] = None) -> Dict[int, float]:
        """Get real vtrust values from metagraph.

        Args:
            uid: Specific validator UID, or None for all validators

        Returns:
            {uid: vtrust} dict
        """
        with self._metagraph_lock:
            permits = getattr(self.metagraph, "validator_permit", None)
            validator_trust = getattr(self.metagraph, "validator_trust", None)

            if permits is None or validator_trust is None:
                return {}

            result = {}
            for i, permit in enumerate(permits):
                if bool(permit):
                    if uid is None or i == uid:
                        result[i] = float(validator_trust[i])

            return result

    def get_miner_uid(self, identifier: str) -> Optional[int]:
        """Get miner UID from hotkey or UID string

        Args:
            identifier: Either hotkey address or UID number

        Returns:
            UID if found, None otherwise
        """
        # Try as UID first
        try:
            uid = int(identifier)
            if 0 <= uid < len(self.metagraph.hotkeys):
                return uid
        except ValueError:
            pass

        # Try as hotkey
        try:
            if identifier in self.metagraph.hotkeys:
                return self.metagraph.hotkeys.index(identifier)
        except (ValueError, AttributeError):
            pass

        return None

    def get_miner_info(self, uid: int) -> Dict[str, Any]:
        """Get miner information by UID"""
        if uid >= len(self.metagraph.hotkeys):
            return {}

        return {
            "uid": uid,
            "hotkey": self.metagraph.hotkeys[uid],
            "coldkey": self.metagraph.coldkeys[uid],
            "stake": float(self.metagraph.stake[uid]),
            "trust": float(self.metagraph.trust[uid]),
            "consensus": float(self.metagraph.consensus[uid]),
            "incentive": float(self.metagraph.incentive[uid]),
        }

    def list_miners(self, limit: int = 10) -> None:
        """List first N miners with their UIDs and hotkeys"""
        print(f"\n📋 First {limit} miners:")
        print("-" * 80)
        print(f"{'UID':<5} {'Hotkey':<50} {'Stake':<12}")
        print("-" * 80)

        for i in range(min(limit, len(self.metagraph.hotkeys))):
            info = self.get_miner_info(i)
            print(f"{info['uid']:<5} {info['hotkey']:<50} {info['stake']:<12.2f}")

    def search_miners(self, query: str) -> List[Dict[str, Any]]:
        """Search miners by partial hotkey match"""
        matches = []
        query_lower = query.lower()

        for i, hotkey in enumerate(self.metagraph.hotkeys):
            if query_lower in hotkey.lower():
                matches.append(self.get_miner_info(i))

        return matches

    async def set_weight_for_miner(self, uid: int, weight: float) -> bool:
        """Set weight for a specific miner using weight_manager's exact method

        Args:
            uid: Miner UID
            weight: Weight value (0.0 to 1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            bt.logging.info(f"Setting weight | uid={uid} weight={weight:.6f}")

            if uid >= len(self.metagraph.axons):
                bt.logging.error(
                    f"❌ Invalid UID {uid} | max={len(self.metagraph.axons)-1}"
                )
                return False

            hotkey = self.metagraph.axons[uid].hotkey
            weights_dict = {hotkey: weight}

            return await self._apply_weights_to_network(weights_dict)

        except Exception as e:
            bt.logging.error(f"❌ Error setting weight | uid={uid} error={e}")
            return False

    async def _apply_weights_to_network(self, weights: Dict[str, float]) -> bool:
        """Apply hotkey-based weights to network."""
        try:
            weight_array = np.array(list(weights.values()))
            if np.isnan(weight_array).any():
                bt.logging.warning("⚠️ Weights contain NaN | action=replace_zeros")
                weights = {
                    hotkey: 0.0 if np.isnan(weight) else weight
                    for hotkey, weight in weights.items()
                }

            raw_weights = np.zeros(len(self.metagraph.axons))
            for uid, axon in enumerate(self.metagraph.axons):
                if axon.hotkey in weights:
                    raw_weights[uid] = weights[axon.hotkey]

            processed_weight_uids, processed_weights = process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            bt.logging.debug(
                f"Processed weights for UIDs: {processed_weight_uids} -> {processed_weights}"
            )

            uint_uids, uint_weights = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )

            bt.logging.debug(
                f"Converted weights for UIDs: {uint_uids} -> {uint_weights}"
            )

            if len(uint_uids) == 0:
                bt.logging.warning("⚠️ No valid weights to set")
                return False

            bt.logging.info(f"Setting weights | miners={len(uint_uids)}")

            def _set_weights():
                with self._subtensor_lock:
                    return self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.netuid,
                        uids=uint_uids,
                        weights=uint_weights,
                        wait_for_inclusion=False,
                        wait_for_finalization=False,
                        version_key=0,
                    )

            loop = asyncio.get_event_loop()
            result, msg = await loop.run_in_executor(None, _set_weights)

            if result:
                bt.logging.info(f"✅ Weights set | miners={len(uint_uids)}")
                return True
            else:
                bt.logging.error(f"❌ Weight submission failed | msg={msg}")
                return False

        except Exception as e:
            bt.logging.error(f"❌ Weight submission error | error={e}")
            return False

    async def set_weights_for_miners(self, target_miners: List[Dict[str, Any]]) -> bool:
        try:
            weights_dict = {}

            for miner in target_miners:
                uid = miner["uid"]
                weight = miner["weight"]
                if 0 <= uid < len(self.metagraph.axons):
                    hotkey = self.metagraph.axons[uid].hotkey
                    weights_dict[hotkey] = weight

            if not weights_dict:
                bt.logging.warning("⚠️ No valid miners to set weights for")
                return False

            bt.logging.info(f"Setting weights | miners={len(weights_dict)}")

            return await self._apply_weights_to_network(weights_dict)
        except Exception as e:
            bt.logging.error(f"❌ Error setting weights | error={e}")
            return False

    async def continuous_scoring_loop(
        self, target_miners: List[Dict[str, Any]], interval_seconds: int
    ) -> None:
        """Run continuous scoring loop for target miners

        Args:
            target_miners: List of {"uid": int, "weight": float} dicts
            interval_seconds: Interval between weight updates in seconds
        """
        if not target_miners:
            print("❌ No target miners specified")
            return

        print(f"\n🔄 Starting continuous scoring loop:")
        print(f"   Interval: {interval_seconds} seconds")
        print(f"   Target miners: {len(target_miners)}")
        for miner in target_miners:
            info = self.get_miner_info(miner["uid"])
            print(
                f"   - UID {miner['uid']}: {info.get('hotkey', 'Unknown')[:20]}... -> {miner['weight']:.6f}"
            )

        print(f"\n⏰ Press Ctrl+C to stop the loop\n")

        attempt = 0
        try:
            while True:
                attempt += 1
                print(f"🎯 Attempt {attempt} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Set weights for all target miners in a single transaction
                success = await self.set_weights_for_miners(target_miners)

                if success:
                    print(
                        f"✅ Attempt {attempt} complete | weights set for {len(target_miners)} miners"
                    )
                else:
                    print(f"❌ Attempt {attempt} failed | weights not set")

                # Wait for next attempt
                print(f"⏱️ Waiting {interval_seconds} seconds...")
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping continuous scoring loop after {attempt} attempts")

    def run_interactive_mode(self) -> None:
        """Run interactive command-line interface"""
        print(f"Wallet: {self.wallet_name}")
        print(f"Hotkey: {self.hotkey_name}")
        print(f"Network: {self.netuid}")
        print(f"Total miners: {len(self.metagraph.hotkeys)}")
        print()

        while True:
            try:
                print("\n📋 Available commands:")
                print("  - list [N] - List first N miners (default 10)")
                print("  - search <query> - Search miners by hotkey")
                print("  - info <uid|hotkey> - Get miner information")
                print("  - weight <uid|hotkey> <score> - Set single weight")
                print("  - loop - Start continuous scoring loop")
                print("  - backtest - Run vtrust estimation backtest")
                print("  - axon <ip> <port> - Update on-chain axon address")
                print("  - quit - Exit tool")

                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue

                action = cmd[0].lower()

                if action == "quit" or action == "q":
                    break

                elif action == "list":
                    limit = int(cmd[1]) if len(cmd) > 1 else 10
                    self.list_miners(limit)

                elif action == "search":
                    if len(cmd) < 2:
                        print("❌ Usage: search <query>")
                        continue
                    query = " ".join(cmd[1:])
                    matches = self.search_miners(query)
                    if matches:
                        print(f"\n🔍 Found {len(matches)} matches:")
                        for match in matches:
                            print(
                                f"  UID {match['uid']}: {match['hotkey']} (stake: {match['stake']:.2f})"
                            )
                    else:
                        print("❌ No matches found")

                elif action == "info":
                    if len(cmd) < 2:
                        print("❌ Usage: info <uid|hotkey>")
                        continue
                    identifier = cmd[1]
                    uid = self.get_miner_uid(identifier)
                    if uid is not None:
                        info = self.get_miner_info(uid)
                        print(f"\n📊 Miner Information:")
                        for key, value in info.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"❌ Miner not found: {identifier}")

                elif action == "weight":
                    if len(cmd) < 3:
                        print("❌ Usage: weight <uid|hotkey> <score>")
                        continue
                    identifier = cmd[1]
                    try:
                        weight = float(cmd[2])
                        if not 0.0 <= weight <= 1.0:
                            print("❌ Weight must be between 0.0 and 1.0")
                            continue
                    except ValueError:
                        print("❌ Invalid weight value")
                        continue

                    uid = self.get_miner_uid(identifier)
                    if uid is not None:
                        asyncio.run(self.set_weight_for_miner(uid, weight))
                    else:
                        print(f"❌ Miner not found: {identifier}")

                elif action == "axon":
                    if len(cmd) < 2:
                        print("❌ Usage: axon <ip> [port]")
                        continue
                    ip = cmd[1]

                    port: Optional[int] = None
                    if len(cmd) >= 3:
                        try:
                            port = int(cmd[2])
                            # Allow 0..65535 inclusive. Port 0 is allowed for on-chain broadcast.
                            if not (0 <= port <= 65535):
                                print("❌ Port must be in 0-65535")
                                continue
                        except ValueError:
                            print("❌ Invalid port value")
                            continue

                    try:
                        asyncio.run(self.update_axon_onchain(ip, port))
                    except Exception as e:
                        print(f"❌ Failed to update axon: {e}")

                elif action == "loop":
                    self._run_loop_setup()

                elif action == "backtest":
                    asyncio.run(self.run_backtest())

                else:
                    print(f"❌ Unknown command: {action}")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        self.stop_background_sync()
        print("\n👋 Goodbye!")

    async def update_axon_onchain(self, ip: str, port: int) -> None:
        """Update on-chain axon endpoint for current hotkey"""
        try:
            bt.logging.info(
                f"🛰️ Updating axon | ip={ip} port={port} netuid={self.netuid}"
            )

            # Set desired external address and push to chain
            # Important: chain uses external_ip for on-chain endpoint, not bind ip.
            # Passing external_ip ensures on-chain value matches provided ip (e.g., 0.0.0.0).
            ax = bt.axon(wallet=self.wallet, port=port, external_ip=ip)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: ax.serve(netuid=self.netuid, subtensor=self.subtensor)
            )

            # Show result (metagraph refreshed by background sync)
            try:
                my_hotkey = getattr(
                    self.wallet.hotkey, "ss58_address", None
                ) or getattr(self.wallet.hotkey, "address", None)
                uid = (
                    self.metagraph.hotkeys.index(my_hotkey)
                    if my_hotkey in self.metagraph.hotkeys
                    else None
                )
                if uid is not None and 0 <= uid < len(self.metagraph.axons):
                    ep = self.metagraph.axons[uid]
                    ep_ip = getattr(ep, "ip", ip)
                    ep_port = getattr(ep, "port", port)
                    print(f"✅ Axon updated on-chain → {ep_ip}:{ep_port} (uid {uid})")
                else:
                    print("✅ Axon update extrinsic submitted")
            except Exception:
                print("✅ Axon update extrinsic submitted")

        except Exception as e:
            bt.logging.error(f"❌ Axon update failed | error={e}")
            raise

    def _run_loop_setup(self) -> None:
        """Interactive setup for continuous scoring loop"""
        print("\n🔄 Continuous Scoring Loop Setup")
        print("-" * 40)
        print("Available modes:")
        print("  manual    - Set weights for specific miners manually")
        print("  mimic     - Follow a validator (submit when they submit)")
        print("  consensus - Calculate stake-weighted median (Yuma Consensus)")
        print("  emission  - Use current network emission/incentive ratios")
        print()

        # Choose mode
        try:
            mode = (
                input("Mode [manual/mimic/consensus/emission] (default manual): ")
                .strip()
                .lower()
                or "manual"
            )
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Setup cancelled")
            return

        if mode not in ("manual", "mimic", "consensus", "emission"):
            print("❌ Invalid mode. Use 'manual', 'mimic', 'consensus', or 'emission'.")
            return

        if mode == "manual":
            target_miners = []

            print("Enter target miners (one per line). Format: <uid|hotkey> <weight>")
            print("Type 'done' when finished:")

            while True:
                try:
                    line = input("Miner > ").strip()
                    if line.lower() == "done":
                        break

                    parts = line.split()
                    if len(parts) != 2:
                        print("❌ Format: <uid|hotkey> <weight>")
                        continue

                    identifier = parts[0]
                    try:
                        weight = float(parts[1])
                        if not 0.0 <= weight <= 1.0:
                            print("❌ Weight must be between 0.0 and 1.0")
                            continue
                    except ValueError:
                        print("❌ Invalid weight value")
                        continue

                    uid = self.get_miner_uid(identifier)
                    if uid is not None:
                        target_miners.append({"uid": uid, "weight": weight})
                        info = self.get_miner_info(uid)
                        print(
                            f"✅ Added UID {uid}: {info.get('hotkey', 'Unknown')[:20]}... -> {weight:.6f}"
                        )
                    else:
                        print(f"❌ Miner not found: {identifier}")

                except (KeyboardInterrupt, EOFError):
                    print("\n❌ Setup cancelled")
                    return

            if not target_miners:
                print("❌ No target miners specified")
                return

            try:
                interval = int(input("Interval (seconds): "))
                if interval <= 0:
                    print("❌ Interval must be positive")
                    return
            except (ValueError, KeyboardInterrupt, EOFError):
                print("❌ Invalid interval")
                return

            asyncio.run(self.continuous_scoring_loop(target_miners, interval))
            return

        if mode == "consensus":
            self._setup_consensus_mode()
            return

        if mode == "emission":
            self._setup_emission_mode()
            return

        # Mimic mode (mode == "mimic")
        print("\n🪞 Mimic Mode")
        print("This mode follows a target validator's weight submissions:")
        print("  - Monitors when target validator submits new weights")
        print("  - Copies their weights immediately after they submit")
        print("  - Automatically follows their submission rhythm")
        print()

        try:
            validator_identifier = input("Validator to mimic (uid or hotkey): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Setup cancelled")
            return

        validator_uid = self.get_miner_uid(validator_identifier)
        if validator_uid is None:
            print(f"❌ Validator not found: {validator_identifier}")
            return

        # Show target validator info
        target_info = self.get_miner_info(validator_uid)
        print(f"📋 Target validator: uid={validator_uid}")
        print(f"   Hotkey: {target_info.get('hotkey', 'Unknown')[:40]}...")
        print(f"   Stake: {target_info.get('stake', 0):.2f}")
        print()

        try:
            polling_interval = int(
                input("Polling interval (seconds) [default 30]: ").strip() or "30"
            )
            if polling_interval <= 0:
                print("❌ Polling interval must be positive")
                return
        except (ValueError, KeyboardInterrupt, EOFError):
            print("❌ Invalid polling interval")
            return

        asyncio.run(self.continuous_mimic_follow_loop(validator_uid, polling_interval))
        return

    def _setup_consensus_mode(self) -> None:
        """Setup and run consensus mode loop."""
        print("\n📊 Consensus Mode (Stake-Weighted Median)")
        print("This mode calculates weights using Yuma Consensus algorithm:")
        print("  - Waits until N blocks before tempo end")
        print("  - Fetches all validators' current weights from chain")
        print("  - Calculates stake-weighted median for each miner")
        print("  - Submits the consensus weights")
        print()

        try:
            polling_interval = int(
                input("Polling interval (seconds) [default 30]: ").strip() or "30"
            )
            if polling_interval <= 0:
                print("❌ Polling interval must be positive")
                return
        except (ValueError, KeyboardInterrupt, EOFError):
            print("❌ Invalid polling interval")
            return

        try:
            blocks_before_end = int(
                input("Blocks before tempo end to submit [default 5]: ").strip() or "5"
            )
            if blocks_before_end <= 0:
                print("❌ Must be positive")
                return
        except (ValueError, KeyboardInterrupt, EOFError):
            print("❌ Invalid value")
            return

        asyncio.run(
            self.continuous_consensus_tempo_loop(polling_interval, blocks_before_end)
        )

    def _setup_emission_mode(self) -> None:
        """Setup and run emission mode loop."""
        print("\n💰 Emission Mode")
        print("This mode uses current network incentive/emission ratios:")
        print("  - Waits until N blocks before tempo end")
        print("  - Reads metagraph incentive values (consensus result)")
        print("  - Submits these as weights directly")
        print()

        try:
            polling_interval = int(
                input("Polling interval (seconds) [default 30]: ").strip() or "30"
            )
            if polling_interval <= 0:
                print("❌ Polling interval must be positive")
                return
        except (ValueError, KeyboardInterrupt, EOFError):
            print("❌ Invalid polling interval")
            return

        try:
            blocks_before_end = int(
                input("Blocks before tempo end to submit [default 5]: ").strip() or "5"
            )
            if blocks_before_end <= 0:
                print("❌ Must be positive")
                return
        except (ValueError, KeyboardInterrupt, EOFError):
            print("❌ Invalid value")
            return

        asyncio.run(
            self.continuous_emission_tempo_loop(polling_interval, blocks_before_end)
        )

    async def _fetch_validator_uint_pairs(
        self, validator_uid: int
    ) -> Tuple[List[int], List[int]]:
        """Fetch the exact on-chain integer weight pairs for a validator (uids, uint16 weights)."""

        def _get_weights():
            with self._subtensor_lock:
                return self.subtensor.weights(netuid=self.netuid, block=None)

        try:
            loop = asyncio.get_event_loop()
            all_weights = await loop.run_in_executor(None, _get_weights)
        except Exception as e:
            bt.logging.error(f"❌ Failed to read on-chain weights | error={e}")
            return [], []

        entry: Optional[Tuple[int, List[Tuple[int, int]]]] = None
        for vid, pairs in all_weights:
            if int(vid) == int(validator_uid):
                entry = (vid, pairs or [])
                break

        if not entry:
            return [], []

        _, to_pairs = entry
        if not to_pairs:
            return [], []

        uids: List[int] = []
        weights: List[int] = []
        for to_uid, w in to_pairs:
            uids.append(int(to_uid))
            weights.append(int(w))
        return uids, weights

    async def _apply_uint_weights_exact(
        self, uids: List[int], uint_weights: List[int]
    ) -> bool:
        """Submit exact uint weights for perfect mimic.

        Converts chain uint weights to float ratios, then uses standard
        convert_weights_and_uids_for_emit to minimize quantization drift.
        """
        try:
            if not uids or not uint_weights or len(uids) != len(uint_weights):
                bt.logging.warning("⚠️ No valid uint weights to set")
                return False

            # Step 1: Convert uint to float ratios (preserve exact proportions)
            total = sum(uint_weights)
            if total <= 0:
                bt.logging.warning("⚠️ Total uint weights is zero")
                return False

            float_weights = [float(w) / float(total) for w in uint_weights]

            # Step 2: Use convert_weights_and_uids_for_emit for proper format
            uids_array = np.array(uids)
            weights_array = np.array(float_weights)
            uint_uids, uint_wts = convert_weights_and_uids_for_emit(
                uids_array, weights_array
            )

            # Step 3: Submit using the standard format (Python lists)
            def _set_weights():
                with self._subtensor_lock:
                    return self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.netuid,
                        uids=uint_uids,
                        weights=uint_wts,
                        wait_for_inclusion=False,
                        wait_for_finalization=False,
                        version_key=0,
                    )

            loop = asyncio.get_event_loop()
            result, msg = await loop.run_in_executor(None, _set_weights)

            if result:
                bt.logging.info(f"✅ Weights set (mimic) | miners={len(uids)}")
                return True
            else:
                bt.logging.error(f"❌ Weight submission failed | msg={msg}")
                return False
        except Exception as e:
            bt.logging.error(f"❌ Mimic weights submit error | error={e}")
            return False

    async def continuous_mimic_follow_loop(
        self,
        validator_uid: int,
        polling_interval: int,
    ) -> None:
        """Follow target validator's weight submissions.

        Monitors when the target validator submits new weights and copies them
        immediately. Submission timing automatically follows the target's rhythm.

        Args:
            validator_uid: UID of the validator to follow
            polling_interval: Seconds between checks for target updates
        """
        print(f"\n🪞 Mimic follow mode | target_uid={validator_uid}")
        print(f"   Polling interval: {polling_interval}s")
        print(f"⏰ Press Ctrl+C to stop\n")

        # Get initial blocks_since for target
        last_target_blocks = await self._get_uid_blocks_since_update(validator_uid)
        if last_target_blocks is None:
            print("❌ Failed to get initial target status")
            return

        print(
            f"📊 Target uid={validator_uid} last updated {last_target_blocks} blocks ago"
        )

        poll_count = 0

        try:
            while True:
                poll_count += 1
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # Get current target blocks_since
                target_blocks = await self._get_uid_blocks_since_update(validator_uid)
                if target_blocks is None:
                    print(f"⚠️ [{current_time}] Failed to check target | will retry")
                    await asyncio.sleep(polling_interval)
                    continue

                # Detect if target has submitted new weights
                # If blocks_since decreased, target submitted new weights
                target_submitted = target_blocks < last_target_blocks

                if target_submitted:
                    print(
                        f"🔔 [{current_time}] Target submitted! "
                        f"blocks_since: {last_target_blocks} -> {target_blocks}"
                    )

                    # Check our own tempo gate before submitting
                    if not await self._is_own_tempo_ready():
                        last_target_blocks = target_blocks
                        await asyncio.sleep(polling_interval)
                        continue

                    # Fetch and submit target's weights
                    uids_exact, wts_exact = await self._fetch_validator_uint_pairs(
                        validator_uid
                    )
                    if not uids_exact or not wts_exact:
                        print("⚠️ Failed to fetch target weights | will retry")
                    else:
                        success = await self._apply_uint_weights_exact(
                            uids_exact, wts_exact
                        )
                        if success:
                            print(
                                f"✅ Mimicked {len(uids_exact)} miners | "
                                f"target_blocks_since={target_blocks}"
                            )
                        else:
                            print("⚠️ Weight submission failed | will retry")

                    last_target_blocks = target_blocks
                else:
                    # No new submission from target, just update baseline
                    # Log periodically (every 10 polls)
                    if poll_count % 10 == 0:
                        print(
                            f"👀 [{current_time}] Monitoring | "
                            f"target_blocks_since={target_blocks}"
                        )
                    last_target_blocks = target_blocks

                await asyncio.sleep(polling_interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping mimic follow loop after {poll_count} polls")

    def _get_own_uid(self) -> Optional[int]:
        """Get our own UID from metagraph."""
        try:
            my_hotkey = getattr(self.wallet.hotkey, "ss58_address", None) or getattr(
                self.wallet.hotkey, "address", None
            )
            if my_hotkey and my_hotkey in self.metagraph.hotkeys:
                return self.metagraph.hotkeys.index(my_hotkey)
        except (ValueError, AttributeError):
            pass
        return None

    async def _is_own_tempo_ready(self) -> bool:
        """Check if own tempo allows weight submission.

        Returns:
            True if ready to submit, False if rate limited
        """
        own_uid = self._get_own_uid()
        if own_uid is None:
            return True  # Can't check, assume ready

        blocks_since = await self._get_uid_blocks_since_update(own_uid)
        if blocks_since is None:
            return True  # Can't check, assume ready

        if blocks_since < self.MIN_BLOCKS_BETWEEN_SUBMISSIONS:
            print(f"⏭️ Own tempo not ready | blocks_since={blocks_since} | skip")
            return False

        return True

    async def _fetch_all_validator_weights(
        self,
    ) -> Tuple[Dict[int, Dict[int, int]], Dict[int, float]]:
        """Fetch all validators' weights and their stakes.

        Always excludes own weights to avoid self-reference bias in consensus
        calculation and vtrust estimation.

        Returns:
            Tuple of:
            - validator_weights: {validator_uid: {miner_uid: uint_weight}}
            - validator_stakes: {validator_uid: stake}
        """
        # Always exclude own weights to avoid self-reference bias
        own_uid = self._get_own_uid()

        # Get validator permits
        permits = getattr(self.metagraph, "validator_permit", None)
        if permits is None:
            bt.logging.error("❌ Metagraph missing validator_permit")
            return {}, {}

        validator_uids = [uid for uid, permit in enumerate(permits) if bool(permit)]
        if not validator_uids:
            bt.logging.warning("⚠️ No validators with permits found")
            return {}, {}

        # Fetch all weights from chain
        def _get_weights():
            with self._subtensor_lock:
                return self.subtensor.weights(netuid=self.netuid, block=None)

        try:
            loop = asyncio.get_event_loop()
            all_weights = await loop.run_in_executor(None, _get_weights)
        except Exception as e:
            bt.logging.error(f"❌ Failed to read on-chain weights | error={e}")
            return {}, {}

        validator_uid_set = set(validator_uids)
        validator_weights: Dict[int, Dict[int, int]] = {}
        validator_stakes: Dict[int, float] = {}

        for vid, pairs in all_weights:
            vid = int(vid)
            if vid not in validator_uid_set:
                continue
            if not pairs:
                continue
            # Always exclude own weights to avoid self-reference bias
            if vid == own_uid:
                continue

            # Store raw uint weights
            validator_weights[vid] = {int(uid): int(w) for uid, w in pairs}
            validator_stakes[vid] = float(self.metagraph.stake[vid])

        return validator_weights, validator_stakes

    def _calculate_stake_weighted_median(
        self,
        validator_weights: Dict[int, Dict[int, int]],
        validator_stakes: Dict[int, float],
        kappa: float = 0.5,
    ) -> Dict[int, float]:
        """Calculate stake-weighted median weights (Yuma Consensus approximation).

        For each miner, finds the maximum weight w such that at least kappa (50%)
        of total stake supports that weight or higher.

        Args:
            validator_weights: {validator_uid: {miner_uid: uint_weight}}
            validator_stakes: {validator_uid: stake}
            kappa: Stake fraction threshold (default 0.5 for majority)

        Returns:
            {miner_uid: consensus_weight} normalized to sum=1
        """
        if not validator_weights or not validator_stakes:
            return {}

        # Normalize each validator's weights to 0-1 range
        validator_float_weights: Dict[int, Dict[int, float]] = {}
        for vid, weights in validator_weights.items():
            total = sum(weights.values()) or 1
            validator_float_weights[vid] = {
                uid: float(w) / float(total) for uid, w in weights.items()
            }

        # Get all miner UIDs
        all_miner_uids: set = set()
        for weights in validator_float_weights.values():
            all_miner_uids.update(weights.keys())

        total_stake = sum(validator_stakes.values())
        if total_stake <= 0:
            return {}

        consensus: Dict[int, float] = {}

        for miner_uid in all_miner_uids:
            # Collect (weight, stake) pairs for this miner
            weight_stake_pairs: List[Tuple[float, float]] = []
            for vid, weights in validator_float_weights.items():
                w = weights.get(miner_uid, 0.0)
                s = validator_stakes.get(vid, 0.0)
                if s > 0:
                    weight_stake_pairs.append((w, s))

            if not weight_stake_pairs:
                consensus[miner_uid] = 0.0
                continue

            # Sort by weight descending
            weight_stake_pairs.sort(key=lambda x: x[0], reverse=True)

            # Find stake-weighted median (kappa quantile)
            cumulative_stake = 0.0
            threshold = kappa * total_stake
            median_weight = 0.0

            for w, s in weight_stake_pairs:
                cumulative_stake += s
                if cumulative_stake >= threshold:
                    median_weight = w
                    break

            consensus[miner_uid] = median_weight

        # Normalize to sum=1
        total_weight = sum(consensus.values())
        if total_weight > 0:
            consensus = {uid: w / total_weight for uid, w in consensus.items()}

        return consensus

    async def run_backtest(self) -> None:
        """Run vtrust estimation backtest for different strategies."""
        print("\n🔬 VTrust Estimation Backtest (Yuma Consensus)")
        print("=" * 70)
        print("Fetching on-chain data...")

        # Get real vtrust values from metagraph
        real_vtrust = self.get_real_vtrust()
        if real_vtrust:
            print(f"✅ Loaded real vtrust for {len(real_vtrust)} validators")
        else:
            print("⚠️ Could not load real vtrust values")

        # Fetch all validator weights (excludes self to avoid self-reference)
        validator_weights_uint, validator_stakes = (
            await self._fetch_all_validator_weights()
        )

        if not validator_weights_uint:
            print("❌ No validator weights found on chain")
            return

        # Get active validators (those who set weights)
        active_validators = list(validator_weights_uint.keys())
        print(f"✅ Found {len(active_validators)} active validators with weights")

        # Get kappa from chain for accurate consensus calculation
        # Use fixed kappa (0.5 is the design intent for 50% stake majority)
        kappa = self.KAPPA
        print(f"✅ Using kappa: {kappa}")

        # Normalize validator weights to float (sum=1 per validator)
        validator_weights: Dict[int, Dict[int, float]] = {}
        for vid, weights in validator_weights_uint.items():
            total = sum(weights.values()) or 1
            validator_weights[vid] = {
                uid: float(w) / float(total) for uid, w in weights.items()
            }

        # Initialize Yuma Consensus calculator
        yuma = YumaConsensus(kappa=kappa)

        # Run full Yuma Consensus to get consensus and calculated vtrust
        yuma_result = yuma.run(validator_stakes, validator_weights)
        true_consensus = yuma_result["consensus"]
        calculated_vtrust = yuma_result["validator_trust"]

        if not true_consensus:
            print("❌ Failed to calculate consensus weights")
            return

        print(f"✅ Calculated consensus weights for {len(true_consensus)} miners")

        # Show top consensus weights
        sorted_consensus = sorted(
            true_consensus.items(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"\n📊 Top 5 Consensus Weights:")
        for uid, w in sorted_consensus:
            print(f"   uid={uid}: {w:.4f}")

        # Show real vtrust vs calculated vtrust comparison
        if real_vtrust and calculated_vtrust:
            print(f"\n📈 VTrust Comparison (Real vs Calculated, top 5 by real):")
            sorted_real = sorted(real_vtrust.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            for uid, real_vt in sorted_real:
                calc_vt = calculated_vtrust.get(uid, 0.0)
                diff = calc_vt - real_vt
                stake = validator_stakes.get(uid, 0.0)
                print(
                    f"   uid={uid}: real={real_vt:.6f} calc={calc_vt:.6f} diff={diff:+.6f} stake={stake:.2f}"
                )

        # Prepare results
        results: List[Dict[str, Any]] = []

        # 1. Test emission mode (using incentive)
        print("\n🔄 Testing emission mode...")
        try:
            incentives = self.metagraph.incentive
            if incentives is not None and len(incentives) > 0:
                total_inc = sum(float(i) for i in incentives)
                if total_inc > 0:
                    emission_weights = {
                        uid: float(incentives[uid]) / total_inc
                        for uid in range(len(incentives))
                        if float(incentives[uid]) > 0
                    }
                    est_vtrust = yuma.estimate_vtrust(emission_weights, true_consensus)
                    results.append(
                        {
                            "mode": "emission",
                            "uid": None,
                            "est_vtrust": est_vtrust,
                            "real_vtrust": None,
                            "stake": None,
                            "description": "Use network incentive ratios",
                        }
                    )
        except Exception as e:
            print(f"   ⚠️ Emission mode test failed: {e}")

        # 2. Test consensus mode (submit the consensus weights directly)
        print("🔄 Testing consensus mode...")
        try:
            # If we submit exact consensus weights, our vtrust should be ~1.0
            est_vtrust = yuma.estimate_vtrust(true_consensus, true_consensus)
            results.append(
                {
                    "mode": "consensus",
                    "uid": None,
                    "est_vtrust": est_vtrust,
                    "real_vtrust": None,
                    "stake": None,
                    "description": f"Submit consensus weights (kappa={kappa:.4f})",
                }
            )
        except Exception as e:
            print(f"   ⚠️ Consensus mode test failed: {e}")

        # 3. Test mimic mode for each active validator
        print(f"🔄 Testing mimic mode for {len(active_validators)} validators...")
        for vid in active_validators:
            try:
                # Get validator's normalized weights (already normalized above)
                vid_weights = validator_weights[vid]

                # Use Yuma's estimate_vtrust for prediction
                est_vtrust = yuma.estimate_vtrust(vid_weights, true_consensus)
                # Also get the calculated vtrust from full Yuma run
                calc_vtrust = calculated_vtrust.get(vid, 0.0)
                stake = validator_stakes.get(vid, 0.0)
                rv = real_vtrust.get(vid) if real_vtrust else None

                results.append(
                    {
                        "mode": "mimic",
                        "uid": vid,
                        "est_vtrust": est_vtrust,
                        "calc_vtrust": calc_vtrust,
                        "real_vtrust": rv,
                        "stake": stake,
                        "description": f"Mimic validator uid={vid}",
                    }
                )
            except Exception as e:
                print(f"   ⚠️ Mimic test for uid={vid} failed: {e}")

        # Sort results by estimated vtrust descending
        results.sort(key=lambda x: x["est_vtrust"], reverse=True)

        # Display results with real vtrust comparison
        print("\n" + "=" * 110)
        print("📊 BACKTEST RESULTS (sorted by estimated vtrust)")
        print("=" * 110)
        print(
            f"{'Rank':<5} {'Mode':<10} {'UID':<6} {'Est VT':<10} "
            f"{'Calc VT':<10} {'Real VT':<10} {'Diff':<10} {'Stake':<12}"
        )
        print("-" * 110)

        for i, result in enumerate(results[:25], 1):  # Show top 25
            uid_str = str(result["uid"]) if result["uid"] is not None else "-"
            stake_str = f"{result['stake']:.1f}" if result["stake"] is not None else "-"
            est_str = f"{result['est_vtrust']:.6f}"
            calc_str = (
                f"{result.get('calc_vtrust', 0.0):.6f}"
                if result.get("calc_vtrust")
                else "-"
            )

            if result["real_vtrust"] is not None:
                real_str = f"{result['real_vtrust']:.6f}"
                # Use calc_vtrust for diff if available, otherwise est_vtrust
                calc_vt = result.get("calc_vtrust", result["est_vtrust"])
                diff = calc_vt - result["real_vtrust"]
                diff_str = f"{diff:+.6f}"
            else:
                real_str = "-"
                diff_str = "-"

            print(
                f"{i:<5} {result['mode']:<10} {uid_str:<6} {est_str:<10} "
                f"{calc_str:<10} {real_str:<10} {diff_str:<10} {stake_str:<12}"
            )

        # Calculate accuracy metrics for mimic mode using calc_vtrust
        mimic_results = [
            r for r in results if r["mode"] == "mimic" and r["real_vtrust"] is not None
        ]
        if mimic_results:
            # Use calc_vtrust (from full Yuma run) for accuracy measurement
            diffs = [
                abs(r.get("calc_vtrust", r["est_vtrust"]) - r["real_vtrust"])
                for r in mimic_results
            ]
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)

            print("\n" + "=" * 110)
            print("📏 YUMA CONSENSUS ACCURACY (calculated vtrust vs real vtrust)")
            print("=" * 110)
            print(f"   Samples: {len(mimic_results)}")
            print(f"   Average absolute difference: {avg_diff:.6f}")
            print(f"   Max absolute difference: {max_diff:.6f}")

            # Check correlation
            if len(mimic_results) > 2:
                calc_vals = [
                    r.get("calc_vtrust", r["est_vtrust"]) for r in mimic_results
                ]
                real_vals = [r["real_vtrust"] for r in mimic_results]
                correlation = np.corrcoef(calc_vals, real_vals)[0, 1]
                print(f"   Correlation: {correlation:.4f}")

        # Show recommendations
        print("\n" + "=" * 90)
        print("💡 RECOMMENDATIONS")
        print("=" * 90)

        if results:
            best = results[0]
            print(f"\n🏆 Best strategy by estimated vtrust: {best['mode']}")
            if best["mode"] == "mimic":
                print(
                    f"   Mimic validator uid={best['uid']} (stake={best['stake']:.2f})"
                )
                if best["real_vtrust"] is not None:
                    print(f"   Their real vtrust: {best['real_vtrust']:.6f}")
            print(f"   Estimated vtrust: {best['est_vtrust']:.6f}")

            # Find best mimic target by real vtrust
            mimic_by_real = [
                r
                for r in results
                if r["mode"] == "mimic" and r["real_vtrust"] is not None
            ]
            if mimic_by_real:
                best_real = max(mimic_by_real, key=lambda x: x["real_vtrust"])
                print(f"\n🎯 Best mimic target by REAL vtrust: uid={best_real['uid']}")
                print(f"   Real vtrust: {best_real['real_vtrust']:.6f}")
                print(f"   Estimated vtrust: {best_real['est_vtrust']:.6f}")
                print(f"   Stake: {best_real['stake']:.2f}")

            # Compare modes
            mode_best = {}
            for r in results:
                if r["mode"] not in mode_best:
                    mode_best[r["mode"]] = r

            print("\n📈 Mode comparison (by estimated vtrust):")
            for mode in ["emission", "consensus", "mimic"]:
                if mode in mode_best:
                    r = mode_best[mode]
                    uid_info = f" (uid={r['uid']})" if r["uid"] is not None else ""
                    real_info = (
                        f" real={r['real_vtrust']:.4f}" if r["real_vtrust"] else ""
                    )
                    print(
                        f"   {mode:<12}: est={r['est_vtrust']:.6f}{real_info}{uid_info}"
                    )

    async def _get_uid_blocks_since_update(self, uid: int) -> Optional[int]:
        """Get blocks_since_last_update for a specific UID.

        Args:
            uid: The UID to check

        Returns:
            Number of blocks since last update, or None if failed
        """
        try:

            def _get_blocks():
                with self._subtensor_lock:
                    return self.subtensor.blocks_since_last_update(self.netuid, uid)

            loop = asyncio.get_event_loop()
            blocks_since = await loop.run_in_executor(None, _get_blocks)
            return int(blocks_since)
        except Exception as e:
            bt.logging.debug(f"Failed to get blocks_since for uid={uid} | error={e}")
            return None

    async def _get_tempo_info(self) -> Optional[Dict[str, int]]:
        """Get tempo information for this subnet.

        Uses blocks_since_last_step for accurate subnet-specific epoch tracking.
        Each subnet has independent epoch rhythm, not simply current_block % tempo.

        Returns:
            Dict with keys: tempo, current_block, blocks_since_last_step, blocks_until_end
            or None if failed
        """
        try:

            def _fetch_tempo():
                with self._subtensor_lock:
                    tempo = self.subtensor.tempo(self.netuid)
                    current_block = self.subtensor.get_current_block()
                    blocks_since_last_step = self.subtensor.blocks_since_last_step(
                        self.netuid
                    )
                    return tempo, current_block, blocks_since_last_step

            loop = asyncio.get_event_loop()
            tempo, current_block, blocks_since_last_step = await loop.run_in_executor(
                None, _fetch_tempo
            )

            tempo = int(tempo)
            current_block = int(current_block)
            blocks_since_last_step = int(blocks_since_last_step)

            # Calculate blocks until epoch end using subnet-specific offset
            blocks_until_end = tempo - blocks_since_last_step

            return {
                "tempo": tempo,
                "current_block": current_block,
                "blocks_since_last_step": blocks_since_last_step,
                "blocks_until_end": blocks_until_end,
            }
        except Exception as e:
            bt.logging.error(f"❌ Failed to get tempo info | error={e}")
            return None

    async def _submit_float_weights(self, weights: Dict[int, float]) -> bool:
        """Submit normalized float weights to chain."""
        if not weights:
            return False

        uids = list(weights.keys())
        float_weights = [weights[uid] for uid in uids]

        # Use convert_weights_and_uids_for_emit for proper format
        uids_array = np.array(uids)
        weights_array = np.array(float_weights)
        uint_uids, uint_wts = convert_weights_and_uids_for_emit(
            uids_array, weights_array
        )

        def _set_weights():
            with self._subtensor_lock:
                return self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uint_uids,
                    weights=uint_wts,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                    version_key=0,
                )

        loop = asyncio.get_event_loop()
        result, msg = await loop.run_in_executor(None, _set_weights)

        if result:
            bt.logging.info(f"✅ Weights submitted | miners={len(uids)}")
            return True
        else:
            bt.logging.error(f"❌ Weight submission failed | msg={msg}")
            return False

    async def continuous_consensus_tempo_loop(
        self,
        polling_interval: int,
        blocks_before_end: int,
    ) -> None:
        """Submit stake-weighted median consensus at tempo boundary.

        Waits until the specified blocks before tempo end, then fetches
        all validators' current weights and calculates stake-weighted
        median (Yuma Consensus algorithm).

        Uses blocks_since_last_step for accurate subnet-specific epoch tracking.

        Args:
            polling_interval: Seconds between tempo checks
            blocks_before_end: Blocks before tempo end to submit (e.g. 5)
        """

        print(f"\n📊 Consensus Tempo Mode (Stake-Weighted Median)")
        print(f"   Polling interval: {polling_interval}s")
        print(f"   Submit at: {blocks_before_end} blocks before tempo end")
        print(f"⏰ Press Ctrl+C to stop\n")

        # Get initial tempo info
        tempo_info = await self._get_tempo_info()
        if tempo_info is None:
            print("❌ Failed to get tempo info")
            return

        tempo = tempo_info["tempo"]
        target_offset = tempo - blocks_before_end  # e.g., 360 - 5 = 355

        print(
            f"📋 Tempo info | tempo={tempo} "
            f"current={tempo_info['current_block']} "
            f"blocks_since_step={tempo_info['blocks_since_last_step']} "
            f"target_offset={target_offset}"
        )

        # Track submission state: last blocks_since_last_step when submitted
        # When blocks_since_last_step resets (becomes smaller), new epoch started
        last_submitted_step: Optional[int] = None
        poll_count = 0

        try:
            while True:
                poll_count += 1
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # Get current tempo info
                tempo_info = await self._get_tempo_info()
                if tempo_info is None:
                    print(f"⚠️ [{current_time}] Failed to get tempo info | will retry")
                    await asyncio.sleep(polling_interval)
                    continue

                current_block = tempo_info["current_block"]
                blocks_since_step = tempo_info["blocks_since_last_step"]
                blocks_until_end = tempo_info["blocks_until_end"]

                # Detect new epoch: blocks_since_step reset (became smaller than last submission)
                new_epoch = (
                    last_submitted_step is not None
                    and blocks_since_step < last_submitted_step
                )
                if new_epoch:
                    last_submitted_step = None  # Reset for new epoch

                # Check if we should submit
                # Submit when: at or past target offset AND haven't submitted this epoch
                should_submit = (
                    blocks_since_step >= target_offset and last_submitted_step is None
                )

                if should_submit:
                    print(
                        f"🔔 [{current_time}] Submit time! "
                        f"block={current_block} blocks_since_step={blocks_since_step} "
                        f"target_offset={target_offset}"
                    )

                    # Fetch all validators' current weights (excludes own)
                    validator_weights, validator_stakes = (
                        await self._fetch_all_validator_weights()
                    )

                    if not validator_weights:
                        print("⚠️ No validator weights found | skip this tempo")
                    else:
                        # Calculate stake-weighted median (Yuma Consensus)
                        consensus = self._calculate_stake_weighted_median(
                            validator_weights, validator_stakes, kappa=self.KAPPA
                        )

                        if not consensus:
                            print("⚠️ Consensus calculation failed | skip this tempo")
                        else:
                            # Show top weights
                            sorted_weights = sorted(
                                consensus.items(), key=lambda x: x[1], reverse=True
                            )[:5]
                            print(
                                f"📊 Consensus weights (top 5): "
                                + ", ".join(
                                    f"uid{uid}={w:.4f}" for uid, w in sorted_weights
                                )
                            )

                            # Check own tempo gate before submitting
                            if not await self._is_own_tempo_ready():
                                pass  # Skip this tempo
                            else:
                                success = await self._submit_float_weights(consensus)
                                if success:
                                    last_submitted_step = blocks_since_step
                                    print(
                                        f"✅ Submitted | validators={len(validator_weights)} "
                                        f"miners={len(consensus)}"
                                    )
                                else:
                                    print("⚠️ Submission failed | will retry")

                else:
                    # Log periodically (every 10 polls)
                    if poll_count % 10 == 0:
                        print(
                            f"👀 [{current_time}] Waiting | "
                            f"block={current_block} blocks_since_step={blocks_since_step} "
                            f"until_end={blocks_until_end} target={target_offset}"
                        )

                await asyncio.sleep(polling_interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping consensus tempo loop after {poll_count} polls")

    async def continuous_emission_tempo_loop(
        self,
        polling_interval: int,
        blocks_before_end: int,
    ) -> None:
        """Submit emission weights at tempo boundary.

        Monitors tempo and submits current emission/incentive weights
        at a specified blocks before tempo end.

        Uses blocks_since_last_step for accurate subnet-specific epoch tracking.

        Args:
            polling_interval: Seconds between tempo checks
            blocks_before_end: Blocks before tempo end to submit (e.g. 5)
        """

        print(f"\n💰 Emission Tempo Mode")
        print(f"   Polling interval: {polling_interval}s")
        print(f"   Submit at: {blocks_before_end} blocks before tempo end")
        print(f"⏰ Press Ctrl+C to stop\n")

        # Get initial tempo info
        tempo_info = await self._get_tempo_info()
        if tempo_info is None:
            print("❌ Failed to get tempo info")
            return

        tempo = tempo_info["tempo"]
        target_offset = tempo - blocks_before_end  # e.g., 360 - 5 = 355

        print(
            f"📋 Tempo info | tempo={tempo} "
            f"current={tempo_info['current_block']} "
            f"blocks_since_step={tempo_info['blocks_since_last_step']} "
            f"target_offset={target_offset}"
        )

        # Track submission state: last blocks_since_last_step when submitted
        # When blocks_since_last_step resets (becomes smaller), new epoch started
        last_submitted_step: Optional[int] = None
        poll_count = 0

        try:
            while True:
                poll_count += 1
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")

                # Get current tempo info
                tempo_info = await self._get_tempo_info()
                if tempo_info is None:
                    print(f"⚠️ [{current_time}] Failed to get tempo info | will retry")
                    await asyncio.sleep(polling_interval)
                    continue

                current_block = tempo_info["current_block"]
                blocks_since_step = tempo_info["blocks_since_last_step"]
                blocks_until_end = tempo_info["blocks_until_end"]

                # Detect new epoch: blocks_since_step reset (became smaller than last submission)
                new_epoch = (
                    last_submitted_step is not None
                    and blocks_since_step < last_submitted_step
                )
                if new_epoch:
                    last_submitted_step = None  # Reset for new epoch

                # Check if we should submit
                # Submit when: at or past target offset AND haven't submitted this epoch
                should_submit = (
                    blocks_since_step >= target_offset and last_submitted_step is None
                )

                if should_submit:
                    print(
                        f"🔔 [{current_time}] Submit time! "
                        f"block={current_block} blocks_since_step={blocks_since_step} "
                        f"target_offset={target_offset}"
                    )

                    # Sync metagraph to get latest incentive values
                    with self._metagraph_lock:
                        self.metagraph.sync(subtensor=self.subtensor)

                    # Get incentive values
                    incentives = self.metagraph.incentive
                    if incentives is None or len(incentives) == 0:
                        print("⚠️ No incentive data available | skip this tempo")
                    else:
                        # Build weights from incentive values
                        weights: Dict[int, float] = {}
                        total_incentive = sum(float(i) for i in incentives)

                        if total_incentive <= 0:
                            print("⚠️ Total incentive is zero | skip this tempo")
                        else:
                            for uid in range(len(incentives)):
                                inc = float(incentives[uid])
                                if inc > 0:
                                    weights[uid] = inc / total_incentive

                            if not weights:
                                print("⚠️ No positive incentives | skip this tempo")
                            else:
                                # Show top weights
                                sorted_weights = sorted(
                                    weights.items(), key=lambda x: x[1], reverse=True
                                )[:5]
                                print(
                                    f"💰 Emission weights (top 5): "
                                    + ", ".join(
                                        f"uid{uid}={w:.4f}" for uid, w in sorted_weights
                                    )
                                )

                                # Check own tempo gate before submitting
                                if not await self._is_own_tempo_ready():
                                    pass  # Skip this tempo
                                else:
                                    success = await self._submit_float_weights(weights)
                                    if success:
                                        last_submitted_step = blocks_since_step
                                        print(f"✅ Submitted | miners={len(weights)}")
                                    else:
                                        print("⚠️ Submission failed | will retry")

                else:
                    # Log periodically (every 10 polls)
                    if poll_count % 10 == 0:
                        print(
                            f"👀 [{current_time}] Waiting | "
                            f"block={current_block} blocks_since_step={blocks_since_step} "
                            f"until_end={blocks_until_end} target={target_offset}"
                        )

                await asyncio.sleep(polling_interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping emission tempo loop after {poll_count} polls")


def main():
    """Main entry point"""
    print("🎯 Interactive Weight Setting Tool")
    print("=" * 50)

    # Get parameters interactively
    try:
        wallet_name = input("Wallet name: ").strip()
        if not wallet_name:
            print("❌ Wallet name is required")
            return

        hotkey_name = input("Hotkey name: ").strip()
        if not hotkey_name:
            print("❌ Hotkey name is required")
            return

        netuid_str = input("Network UID: ").strip()
        try:
            netuid = int(netuid_str)
        except ValueError:
            print("❌ Network UID must be a number")
            return

        network = input("Network (finney/test/local) [finney]: ").strip() or "finney"
        if network not in ["finney", "test", "local"]:
            print("❌ Invalid network. Use: finney, test, or local")
            return

        # debug_input = input("Enable debug logging? (y/n): ").strip().lower()
        # debug = debug_input in ["y", "yes", "1", "true"]

        # if debug:
        bt.logging.set_debug(True)

        print()
        tool = InteractiveWeightTool(wallet_name, hotkey_name, netuid, network)
        tool.run_interactive_mode()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
