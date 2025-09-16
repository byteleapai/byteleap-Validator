"""
Validator weight management service
Responsible for calculating and setting miner network weights
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from sqlalchemy import or_

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.models.database import (
    DatabaseManager,
    MinerInfo,
    NetworkWeight,
    WorkerInfo,
)
from neurons.validator.services.worker_performance_ranker import WorkerPerformanceRanker

# Weight management timing constants
WEIGHT_CALCULATION_INTERVAL = 300
WEIGHT_SUBMISSION_CHECK_INTERVAL = 30

CHALLENGE_SCORE_CAP = 100


class WeightManager:
    """
    Weight management service for Bittensor subnet

    Manages the calculation and distribution of network weights based on miner performance,
    including lease status, challenge success rates, hardware capabilities, and online presence.

    Key responsibilities:
    - Calculate miner scores across multiple dimensions (CPU, GPU, memory, challenges)
    - Apply lease status weighting (65% weight for leased miners)
    - Manage online presence multipliers based on heartbeat history
    - Distribute weights to the Bittensor network via subtensor
    - Record weight history for audit and analysis

    Weight Distribution:
    - Lease Status: 70% (worker-level lease scores)
    - CPU Challenge Score: 30% (CPU matrix performance ranking across all workers)
    - Online Multiplier: Based on 169-hour heartbeat window with adaptive adjustment
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        wallet: bt.wallet,
        subtensor: bt.subtensor,
        metagraph: bt.metagraph,
        config: ConfigManager,
    ):
        """
        Initialize weight manager

        Args:
            database_manager: Database management service
            wallet: Bittensor wallet
            subtensor: Bittensor subtensor
            metagraph: Bittensor metagraph
            config: Configuration manager
        """
        self.db_manager = database_manager
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.config = config

        # Validate configuration
        self.netuid = config.get_positive_number("netuid", int)
        self.weight_update_tempo = config.get_positive_number(
            "metagraph.weight_update_tempo", int
        )

        # Score weight configuration - use fail-fast config access
        lease_weight = config.get("weight_management.score_weights.lease_weight")
        challenge_weight = config.get(
            "weight_management.score_weights.challenge_weight"
        )

        total_weight = lease_weight + challenge_weight
        if (
            abs(total_weight - 1.0) > 0.001
        ):  # Allow minimal floating point precision errors
            bt.logging.error(
                f"Score weights sum to {total_weight:.3f}, not 1.0. Configuration error."
            )
            raise ValueError(
                f"Invalid weight configuration: weights must sum to 1.0, got {total_weight:.3f}"
            )

        self.score_weights = {
            "lease_weight": lease_weight,
            "challenge_weight": challenge_weight,
        }
        bt.logging.info(
            f"⚖️ Score weights | lease={lease_weight:.2%} challenge={challenge_weight:.2%}"
        )

        self.ranking_window_minutes = config.get_positive_number(
            "validation.ranking_window_minutes", int
        )
        self.challenge_interval = config.get_positive_number(
            "validation.challenge_interval", int
        )
        self.participation_rate_threshold = config.get_range(
            "validation.participation_rate_threshold", 0.1, 1.0, float
        )

        # Initialize worker performance ranker for CPU matrix challenges
        self.performance_ranker = WorkerPerformanceRanker(
            database_manager, self.challenge_interval, self.participation_rate_threshold
        )

        # Running state
        self.is_running = False
        self._scoring_task: Optional[asyncio.Task] = None
        self._setting_task: Optional[asyncio.Task] = None
        self._last_weight_update = 0.0

        # Thread safety lock for subtensor operations
        self._subtensor_lock = asyncio.Lock()

        bt.logging.info("🚀 Weight manager initialized")

    async def start(self) -> None:
        """Start weight management service"""
        if self.is_running:
            bt.logging.warning("⚠️ Weight manager already running")
            return

        self.is_running = True
        self._scoring_task = asyncio.create_task(self._scoring_loop())
        self._setting_task = asyncio.create_task(self._setting_loop())
        bt.logging.info("🚀 Weight manager started")

    async def stop(self) -> None:
        """Stop weight management service"""
        if not self.is_running:
            return

        self.is_running = False

        for task in [self._scoring_task, self._setting_task]:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._scoring_task = None
        self._setting_task = None

    async def _scoring_loop(self) -> None:
        """Weight calculation loop - runs every 5 minutes, calculates weights independently of network."""
        while self.is_running:
            try:
                bt.logging.info("🧮 Weight calc cycle start")

                # Update metagraph with thread safety lock to prevent recv conflicts
                try:
                    async with self._subtensor_lock:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, lambda: self.metagraph.sync(subtensor=self.subtensor)
                        )
                        bt.logging.debug("Metagraph synced")
                except Exception as e:
                    bt.logging.error(
                        f"⚠️ Metagraph sync error | error={e} using_cached_metagraph"
                    )

                # Update worker online status first
                await self._update_worker_online_status()

                await self._calculate_all_weights()

                bt.logging.info("✅ Weight calc cycle done")

                # Wait for next calculation cycle
                for _ in range(WEIGHT_CALCULATION_INTERVAL):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"❌ Scoring loop error | error={e}")
                # Shorter sleep on error, check shutdown more frequently
                for _ in range(60):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

    async def _setting_loop(self) -> None:
        """Weight submission loop - checks block conditions and submits weights based on tempo."""
        while self.is_running:
            try:
                bt.logging.debug("Weight submission check")

                # Check weight submission condition
                try:
                    should_submit = await self._should_set_weights()
                except Exception as e:
                    bt.logging.warning(
                        f"⚠️ Weight submission check error | error={e} skip"
                    )
                    await asyncio.sleep(WEIGHT_SUBMISSION_CHECK_INTERVAL)
                    continue

                if should_submit:
                    bt.logging.info("📦 Weight submission condition met")

                    miners_to_submit = self._get_miners_for_submission()

                    if miners_to_submit:
                        bt.logging.info(
                            f"📤 Submitting weights | miners={len(miners_to_submit)}"
                        )
                        current_block = await self._get_current_block()
                        await self._submit_miners_weights(
                            miners_to_submit, current_block
                        )
                    else:
                        bt.logging.warning(
                            "⚠️ No miners in metagraph for weight submission"
                        )
                else:
                    bt.logging.debug("Weight submission not met")

                # Wait for next check
                for _ in range(WEIGHT_SUBMISSION_CHECK_INTERVAL):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"❌ Submission loop error | error={e}")
                await asyncio.sleep(WEIGHT_SUBMISSION_CHECK_INTERVAL)

    async def _get_current_block(self) -> int:
        """Get current block number from subtensor."""
        async with self._subtensor_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.subtensor.get_current_block()
            )

    async def get_last_update_blocks(self) -> int:
        """Get blocks since last weight update for this validator from chain."""
        try:
            async with self._subtensor_lock:
                loop = asyncio.get_event_loop()

                # Get validator UID
                def _get_validator_info():
                    current_block = self.subtensor.get_current_block()
                    validator_hotkey = self.wallet.hotkey.ss58_address

                    # Find validator UID in metagraph
                    if validator_hotkey in self.metagraph.hotkeys:
                        validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
                    else:
                        raise ValueError(
                            f"Validator {validator_hotkey} not found in metagraph"
                        )

                    # Get blocks since last update
                    last_update_blocks = self.subtensor.blocks_since_last_update(
                        self.netuid, validator_uid
                    )

                    return last_update_blocks

                return await loop.run_in_executor(None, _get_validator_info)

        except Exception as e:
            bt.logging.error(f"❌ Last update blocks error | error={e}")
            # Return a large number to avoid premature weight setting
            return 0

    async def _should_set_weights(self) -> bool:
        """
        Check if we should set weights based on chain conditions.
        Similar to Lium's implementation but using our dual-loop approach.
        """
        try:
            # Get chain information
            tempo = self.weight_update_tempo
            last_update_blocks = await self.get_last_update_blocks()

            # Should set weights if enough blocks have passed since last update
            should_submit = last_update_blocks >= tempo

            bt.logging.debug(
                f"⏱️ Weights check | tempo={tempo} last={last_update_blocks} submit={should_submit}"
            )

            return should_submit

        except Exception as e:
            bt.logging.error(f"❌ Weights check error | error={e}")
            return False

    def _get_miners_for_submission(self) -> List[MinerInfo]:
        """
        Get all miners in metagraph for weight submission.
        Includes both online and offline miners to ensure weights are updated properly.
        """
        with self.db_manager.get_session() as session:
            # Get all miners that appear in metagraph
            active_hotkeys = self.metagraph.hotkeys
            if not active_hotkeys:
                return []

            return (
                session.query(MinerInfo)
                .filter(MinerInfo.hotkey.in_(active_hotkeys))
                .all()
            )

    async def _submit_miners_weights(self, miners: List[MinerInfo], current_block: int):
        """Submit weights for the given miners."""
        with self.db_manager.get_session() as session:
            uids = []
            weights = []
            weight_records_to_update = []

            for miner in miners:
                # Get latest weight for this miner
                latest_weight = (
                    session.query(NetworkWeight)
                    .filter(
                        NetworkWeight.hotkey == miner.hotkey,
                        NetworkWeight.deleted_at.is_(None),
                    )
                    .order_by(NetworkWeight.created_at.desc())
                    .first()
                )

                if latest_weight:
                    uid = self._get_uid_for_hotkey(miner.hotkey)
                    if uid is not None:
                        uids.append(uid)
                        weights.append(latest_weight.weight_value)
                        weight_records_to_update.append(latest_weight)

            if uids:
                # Submit to blockchain
                success = await self._submit_weights_to_chain(uids, weights)

                if success:
                    # Update database records to mark as applied
                    with self.db_manager.get_session() as update_session:
                        for weight_record in weight_records_to_update:
                            self.db_manager.mark_weight_applied(
                                session=update_session,
                                weight_record_id=weight_record.id,
                                apply_remark="success",
                            )

                    bt.logging.info(
                        f"✅ Weights submitted | miners={len(weight_records_to_update)} block={current_block}"
                    )
                else:
                    bt.logging.warning("⚠️ Weight submission failed | retry_next_cycle")
            else:
                bt.logging.warning("⚠️ No valid weights to submit")

    def _get_uid_for_hotkey(self, hotkey: str) -> Optional[int]:
        """Get UID for a given hotkey from metagraph."""
        try:
            if hotkey in self.metagraph.hotkeys:
                return self.metagraph.hotkeys.index(hotkey)
        except (ValueError, AttributeError):
            pass
        return None

    async def _submit_weights_to_chain(
        self, uids: List[int], weights: List[float]
    ) -> bool:
        """Submit weights to the blockchain."""
        try:
            import numpy as np

            # Convert to proper format - ensure numpy arrays
            uids_array = np.array(uids)
            weights_array = np.array(weights)
            uint_uids, uint_weights = convert_weights_and_uids_for_emit(
                uids_array, weights_array
            )

            # Submit to chain
            async with self._subtensor_lock:
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
                    ),
                )

            if result:
                bt.logging.info(
                    f"Successfully submitted weights for {len(uids)} miners"
                )
            else:
                bt.logging.warning(f"⚠️ Weight submission failed | msg={msg}")

            return result

        except Exception as e:
            bt.logging.error(f"❌ Weight submit error | error={e}")
            return False

    async def _update_worker_online_status(self) -> None:
        """Update worker and miner online status based on heartbeat timeout"""
        if not self.is_running:
            return

        try:
            with self.db_manager.get_session() as session:
                # Update worker online status
                offline_worker_count = self.db_manager.update_worker_online_status(
                    session=session, offline_threshold_minutes=30
                )

                # Update miner online status
                offline_miner_count = self.db_manager.update_miner_online_status(
                    session=session, offline_threshold_minutes=30
                )

                if offline_worker_count > 0:
                    bt.logging.info(
                        f"Marked {offline_worker_count} workers as offline due to heartbeat timeout"
                    )

                if offline_miner_count > 0:
                    bt.logging.info(
                        f"Marked {offline_miner_count} miners as offline due to heartbeat timeout"
                    )

        except Exception as e:
            bt.logging.error(f"❌ Online status update error | error={e}")

    async def _calculate_all_weights(self) -> None:
        """Calculates and stores pending weights for all active miners."""
        if not self.is_running:
            return

        with self.db_manager.get_session() as session:
            active_hotkeys = self.metagraph.hotkeys
            if not active_hotkeys:
                bt.logging.warning("⚠️ No active miners for scoring")
                return

            # Get miner information only for active hotkeys
            active_miners = (
                session.query(MinerInfo)
                .filter(MinerInfo.hotkey.in_(active_hotkeys))
                .all()
            )

            bt.logging.debug(f"🧮 Weight calc | miners={len(active_miners)}")

            # Pre-calculate global worker rankings once for all miners
            bt.logging.debug("Pre-calculating global rankings")
            worker_rankings = self.performance_ranker.calculate_global_worker_rankings(
                evaluation_window_minutes=self.ranking_window_minutes
            )

            # Calculate miner-level challenge scores from rankings
            miner_challenge_scores = (
                self.performance_ranker.calculate_miner_challenge_scores(
                    worker_rankings
                )
            )
            max_raw_challenge = (
                max(miner_challenge_scores.values()) if miner_challenge_scores else 0.0
            )

            miner_scores = []
            availability_scores = []

            for miner in active_miners:
                score, score_details = self._calculate_miner_score(
                    miner, miner_challenge_scores, max_raw_challenge
                )
                miner_scores.append(
                    {"miner": miner, "score": score, "score_details": score_details}
                )
                availability_scores.append(score_details["availability_score"])

            miner_scores = self._apply_adaptive_availability_adjustment(
                miner_scores, availability_scores
            )

            weights = self._calculate_weights_from_scores(miner_scores)

            bt.logging.debug(f"💾 Save pending weights | count={len(weights)}")
            for miner_data in miner_scores:
                miner = miner_data["miner"]
                weight = weights.get(miner.hotkey, 0.0)

                # Save as a pending weight update
                self.db_manager.record_weight_update(
                    session=session,
                    hotkey=miner.hotkey,
                    weight_value=weight,
                    scores=miner_data["score_details"],
                    calculation_remark=f"Composite score: {miner_data['score']:.4f}",
                    is_applied=False,
                )

    def _calculate_miner_score(
        self,
        miner: MinerInfo,
        raw_miner_challenge_scores: Dict[str, float],
        max_raw_challenge: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate miner score.

        - Challenge score: normalized to 0..1 by dividing the current-round max raw score.
        - Lease score: aggregated and normalized to 0..1.
        """
        scores: Dict[str, float] = {}

        # Challenge score: normalized to 0..1 based on current max
        raw_challenge = raw_miner_challenge_scores.get(miner.hotkey, 0.0)
        challenge_norm = (
            min(1.0, max(0.0, raw_challenge / max_raw_challenge))
            if max_raw_challenge > 0
            else 0.0
        )
        # Store normalized value in score_details for DB visibility
        scores["challenge_score"] = challenge_norm
        bt.logging.debug(
            f"Miner {miner.hotkey} challenge raw={raw_challenge:.4f} normalized={challenge_norm:.4f} max={max_raw_challenge:.4f}"
        )

        # Lease score: aggregated and normalized to 0..1
        scores["lease_score"] = self._calculate_worker_lease_score(miner)

        # Composite 7:3 using normalized components
        composite_score = (
            challenge_norm * self.score_weights["challenge_weight"]
            + scores["lease_score"] * self.score_weights["lease_weight"]
        )

        # Availability multiplier based on 169h window
        availability_score = self._calculate_online_weight_from_heartbeats(miner)
        scores["availability_score"] = availability_score

        total_score = composite_score * availability_score

        return total_score, scores

    # Legacy helper retained for compatibility if needed
    def _calculate_cpu_matrix_challenge_score(
        self, miner: MinerInfo, miner_challenge_scores: Dict[str, float]
    ) -> float:
        try:
            challenge_score = miner_challenge_scores.get(miner.hotkey, 0.0)
            bt.logging.debug(
                f"Miner {miner.hotkey} CPU matrix challenge score (raw): {challenge_score:.4f}"
            )
            return challenge_score
        except Exception as e:
            bt.logging.error(
                f"Failed to get CPU matrix challenge score for miner {miner.hotkey}: {e}"
            )
            return 0.0

    def _calculate_worker_lease_score(self, miner: MinerInfo) -> float:
        """Calculate aggregated lease score from all workers of this miner"""
        try:
            with self.db_manager.get_session() as session:
                workers = (
                    session.query(WorkerInfo)
                    .filter(
                        WorkerInfo.hotkey == miner.hotkey,
                        WorkerInfo.deleted_at.is_(None),
                    )
                    .order_by(WorkerInfo.lease_score.desc())
                    .limit(CHALLENGE_SCORE_CAP)
                    .all()
                )

                if not workers:
                    return 0.0

                # Calculate normalized lease score
                # Method: sum of all worker lease scores, normalized by max possible
                total_lease_score = sum(worker.lease_score or 0.0 for worker in workers)
                worker_count = len(workers)

                # Normalize based on the assumption that max lease score per worker could be high
                # We'll use a reasonable normalization factor
                if total_lease_score > 0:
                    # Simple normalization: if any worker has lease_score > 0, miner gets proportional score
                    max_workers = min(CHALLENGE_SCORE_CAP, worker_count)
                    # Normalize assuming average lease score of 1.0 per worker would be "full score"
                    if max_workers > 0:
                        normalized_score = min(1.0, total_lease_score / max_workers)
                    else:
                        normalized_score = 0.0
                else:
                    normalized_score = 0.0

                bt.logging.debug(
                    f"Miner {miner.hotkey} worker lease score: {normalized_score:.4f} "
                    f"(from {worker_count} workers, total: {total_lease_score:.2f})"
                )

                return normalized_score

        except Exception as e:
            bt.logging.error(
                f"Failed to calculate worker lease score for miner {miner.hotkey}: {e}"
            )
            return 0.0

    def _calculate_online_weight_from_heartbeats(self, miner: MinerInfo) -> float:
        """Calculate online weight based on 169h window from heartbeat_records table"""
        if miner.last_heartbeat is None:
            return 0.0

        with self.db_manager.get_session() as session:
            from neurons.validator.models.database import HeartbeatRecord

            window_start = datetime.utcnow() - timedelta(hours=169)

            # Query heartbeat records within the 169h window
            heartbeat_records = (
                session.query(HeartbeatRecord)
                .filter(HeartbeatRecord.hotkey == miner.hotkey)
                .filter(HeartbeatRecord.created_at >= window_start)
                .order_by(HeartbeatRecord.created_at.asc())
                .all()
            )

            if not heartbeat_records:
                return 0.0

            # Calculate online ratio within the window
            # Each 5-minute interval with at least one heartbeat counts as online
            expected_intervals = (
                169 * 12
            )  # 169 hours * 12 five-minute intervals per hour

            # Group heartbeats by 5-minute intervals
            online_intervals = set()
            for record in heartbeat_records:
                # Convert timestamp to 5-minute interval index
                interval_index = int(
                    record.created_at.timestamp() // 300
                )  # 300 seconds = 5 minutes
                online_intervals.add(interval_index)

            actual_online_intervals = len(online_intervals)

            # Calculate online weight as ratio of online intervals to expected intervals
            online_ratio = min(1.0, actual_online_intervals / expected_intervals)

            return online_ratio

    def _apply_adaptive_availability_adjustment(
        self, miner_scores: List[Dict[str, Any]], availability_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Apply adaptive availability adjustment based on network average (Method 1)"""
        if not availability_scores or len(availability_scores) == 0:
            return miner_scores

        # Calculate network average availability
        avg_network_availability = sum(availability_scores) / len(availability_scores)

        bt.logging.debug(
            f"Network average availability: {avg_network_availability:.6f}"
        )

        # Apply adaptive adjustment if network average is low
        if avg_network_availability < 0.3:
            bt.logging.info(
                f"Applying adaptive availability adjustment - network avg: {avg_network_availability:.6f}"
            )

            for miner_data in miner_scores:
                original_availability = miner_data["score_details"][
                    "availability_score"
                ]

                if avg_network_availability > 0:
                    adjusted_availability = min(
                        1.0, original_availability / avg_network_availability * 0.5
                    )
                else:
                    adjusted_availability = 0.0

                # Preserve composite sum scale; only adjust availability
                if original_availability > 0:
                    composite_sum = miner_data["score"] / original_availability
                else:
                    composite_sum = 0.0

                # Update final score with adjusted availability
                miner_data["score"] = composite_sum * adjusted_availability
                miner_data["score_details"][
                    "adjusted_availability_score"
                ] = adjusted_availability

                bt.logging.debug(
                    f"Miner {miner_data['miner'].hotkey}: "
                    f"original_avail: {original_availability:.6f} -> "
                    f"adjusted_avail: {adjusted_availability:.6f}, "
                    f"final_score: {miner_data['score']:.6f}"
                )
        else:
            bt.logging.debug("Network availability sufficient")
            # Add the original availability as adjusted for consistency
            for miner_data in miner_scores:
                miner_data["score_details"]["adjusted_availability_score"] = miner_data[
                    "score_details"
                ]["availability_score"]

        return miner_scores

    def _calculate_weights_from_scores(
        self, miner_scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate weights from scores"""
        if not miner_scores:
            return {}

        # Extract all scores
        scores = [data["score"] for data in miner_scores]

        if max(scores) == 0:
            # If all scores are 0, distribute weights evenly
            uniform_weight = 1.0 / len(miner_scores)
            return {data["miner"].hotkey: uniform_weight for data in miner_scores}

        # Use proportional allocation based on scores
        total_score = sum(scores)

        weights = {}
        for i, data in enumerate(miner_scores):
            if total_score > 0:
                weight = scores[i] / total_score
            else:
                weight = 1.0 / len(
                    miner_scores
                )  # Equal distribution if all scores are 0

            weight = max(0, min(1, weight))
            weights[data["miner"].hotkey] = weight

        # Normalize weights to ensure they sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for hotkey in weights:
                weights[hotkey] /= total_weight

        return weights

    async def _apply_weights_to_network(
        self, weights: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Apply weights to Bittensor network with thread safety"""
        async with self._subtensor_lock:
            return await self._do_apply_weights_to_network(weights)

    async def _do_apply_weights_to_network(
        self, weights: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Internal method to apply weights to Bittensor network"""
        try:
            import numpy as np
            from bittensor.utils.weight_utils import (
                convert_weights_and_uids_for_emit,
                process_weights_for_netuid,
            )

            # Check for NaN values in weights
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

            # Use template weight processing logic
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

            # Convert to uint16 format
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

            # Set weights to network
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
                    version_key=0,  # Can be adjusted as needed
                ),
            )

            success = result

            if success:
                bt.logging.info(f"✅ Weights set | miners={len(uint_uids)}")
                return True, None
            else:
                return False, msg

        except Exception as e:
            error_msg = str(e)
            return False, error_msg

    def get_weight_status(self) -> Dict[str, Any]:
        """Get weight management status"""
        return {
            "is_running": self.is_running,
            "last_weight_update": self._last_weight_update,
            "calculation_interval": WEIGHT_CALCULATION_INTERVAL,
            "submission_check_interval": WEIGHT_SUBMISSION_CHECK_INTERVAL,
            "score_weights": self.score_weights,
            "netuid": self.netuid,
        }

    def get_miner_weight_info(self, hotkey: str) -> Dict[str, Any]:
        """Get specific miner's weight information using global rankings"""
        with self.db_manager.get_session() as session:
            miner = session.query(MinerInfo).filter(MinerInfo.hotkey == hotkey).first()

            if not miner:
                return {}

            # Calculate global rankings for consistent scoring
            worker_rankings = self.performance_ranker.calculate_global_worker_rankings(
                evaluation_window_minutes=self.ranking_window_minutes
            )
            raw_miner_challenge_scores = (
                self.performance_ranker.calculate_miner_challenge_scores(
                    worker_rankings
                )
            )
            max_raw = (
                max(raw_miner_challenge_scores.values())
                if raw_miner_challenge_scores
                else 0.0
            )

            score, score_details = self._calculate_miner_score(
                miner, raw_miner_challenge_scores, max_raw
            )

            return {
                "hotkey": hotkey,
                "current_weight": miner.current_weight or 0.0,
                "calculated_score": score,
                "score_details": score_details,
                "last_weight_update": (
                    miner.last_weight_update.isoformat()
                    if miner.last_weight_update
                    else None
                ),
            }
