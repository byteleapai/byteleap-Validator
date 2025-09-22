#!/usr/bin/env python3
"""
Interactive Weight Setting Tool

A command-line tool for manually setting weights for specific miners using validator wallet.
Supports continuous loop scoring with configurable intervals.
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
from bittensor.utils.weight_utils import (convert_weights_and_uids_for_emit,
                                          process_weights_for_netuid)

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.shared.config.config_manager import ConfigManager


class InteractiveWeightTool:
    """Interactive tool for setting miner weights"""

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

        bt.logging.info(f"📊 Loading metagraph | network={network} netuid={netuid}")
        self.metagraph = bt.metagraph(netuid=netuid, network=network)
        self.metagraph.sync(subtensor=self.subtensor)

        self._subtensor_lock = asyncio.Lock()

        bt.logging.info(f"✅ Initialized | miners={len(self.metagraph.hotkeys)}")

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
            bt.logging.info(f"📤 Setting weight | uid={uid} weight={weight:.6f}")

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
        async with self._subtensor_lock:
            success, error_msg = await self._do_apply_weights_to_network(weights)
            return success

    async def _do_apply_weights_to_network(
        self, weights: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
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
                return False, "No valid weights to set"

            bt.logging.info(f"Setting weights | miners={len(uint_uids)}")

            loop = asyncio.get_event_loop()
            result, msg = await loop.run_in_executor(
                None,
                lambda: self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uint_uids,
                    weights=uint_weights,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                    version_key=0,
                ),
            )

            success = result

            if success:
                bt.logging.info(f"✅ Weights set | miners={len(uint_uids)}")
                return True, None
            else:
                bt.logging.error(f"❌ Weight submission failed | msg={msg}")
                return False, msg

        except Exception as e:
            error_msg = str(e)
            return False, error_msg

    async def _get_current_block(self) -> int:
        async with self._subtensor_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.subtensor.get_current_block()
            )

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

            bt.logging.info(f"📤 Setting weights | miners={len(weights_dict)}")

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

        epoch = 0
        try:
            while True:
                epoch += 1
                print(f"🎯 Epoch {epoch} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Set weights for all target miners in a single transaction
                success = await self.set_weights_for_miners(target_miners)
                success_count = 1 if success else 0

                if success:
                    print(
                        f"✅ Epoch {epoch} complete | weights set for {len(target_miners)} miners"
                    )
                else:
                    print(f"❌ Epoch {epoch} failed | weights not set")

                # Wait for next epoch
                print(f"⏱️ Waiting {interval_seconds} seconds until next epoch...")
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping continuous scoring loop after {epoch} epochs")

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
                print("  1. list [N] - List first N miners (default 10)")
                print("  2. search <query> - Search miners by hotkey")
                print("  3. info <uid|hotkey> - Get miner information")
                print("  4. weight <uid|hotkey> <score> - Set single weight")
                print("  5. loop - Start continuous scoring loop")
                print("  6. quit - Exit tool")

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

                elif action == "loop":
                    self._run_loop_setup()

                else:
                    print(f"❌ Unknown command: {action}")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n👋 Goodbye!")

    def _run_loop_setup(self) -> None:
        """Interactive setup for continuous scoring loop"""
        print("\n🔄 Continuous Scoring Loop Setup")
        print("-" * 40)

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

        # Run the continuous loop
        asyncio.run(self.continuous_scoring_loop(target_miners, interval))


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

        debug_input = input("Enable debug logging? (y/n): ").strip().lower()
        debug = debug_input in ["y", "yes", "1", "true"]

        if debug:
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
