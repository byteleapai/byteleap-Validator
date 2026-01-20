"""Validator weight management service"""

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from sqlalchemy import func

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.models.database import (DatabaseManager,
                                               HeartbeatRecord, MinerInfo,
                                               NetworkWeight, WorkerInfo)
from neurons.validator.services.subtensor_access import SubtensorAccessGuard

# Timing constants
WEIGHT_CALCULATION_INTERVAL = 420
WEIGHT_SUBMISSION_CHECK_INTERVAL = 30


class WeightManager:
    """Calculate and set miner network weights"""

    def __init__(
        self,
        database_manager: DatabaseManager,
        wallet: bt.wallet,
        subtensor_guard: SubtensorAccessGuard,
        metagraph: bt.metagraph,
        config: ConfigManager,
        meshhub_client=None,
        metagraph_cache=None,
    ):
        """Initialize weight manager"""
        self.db_manager = database_manager
        self.wallet = wallet
        self._subtensor_guard = subtensor_guard
        self.subtensor = subtensor_guard.subtensor
        self.metagraph = metagraph
        self.metagraph_cache = metagraph_cache
        self.config = config
        self.meshhub_client = meshhub_client

        self.netuid = config.get_positive_number("netuid", int)

        lease_weight = config.get("weight_management.score_weights.lease_weight")
        bt.logging.info(
            f"Weight mode: lease-only | lease_weight={float(lease_weight):.2%}"
        )

        self.is_running = False
        self._scoring_task: Optional[asyncio.Task] = None
        self._setting_task: Optional[asyncio.Task] = None
        self._last_weight_update = 0.0

        self._start_time: Optional[datetime] = None

        bt.logging.info("🚀 Weight manager initialized")

    async def start(self) -> None:
        """Start weight management service"""
        if self.is_running:
            bt.logging.warning("⚠️ Weight manager already running")
            return

        self.is_running = True
        self._start_time = datetime.utcnow()
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
                cycle_start = time.monotonic()
                bt.logging.info("🚀 Weight calc cycle start")

                await self._update_worker_online_status()

                await self._calculate_all_weights()

                elapsed = time.monotonic() - cycle_start
                bt.logging.info(f"✅ Weight calc cycle done | elapsed={elapsed:.2f}s")

                for _ in range(WEIGHT_CALCULATION_INTERVAL):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"❌ Scoring loop error | error={e}")

                for _ in range(60):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

    async def _setting_loop(self) -> None:
        """Weight submission loop - checks block conditions and submits weights based on tempo."""
        while self.is_running:
            try:
                bt.logging.debug("Weight submission check")

                try:
                    should_submit = await self._should_set_weights()
                except Exception as e:
                    bt.logging.warning(
                        f"⚠️ Weight submission check error | error={e} skip"
                    )
                    await asyncio.sleep(WEIGHT_SUBMISSION_CHECK_INTERVAL)
                    continue

                if should_submit:
                    bt.logging.info("🚀 Weight submission condition met")

                    miners_to_submit = self._get_miners_for_submission()

                    if miners_to_submit:
                        bt.logging.info(
                            f"🚀 Submitting weights | miners={len(miners_to_submit)}"
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
        return await self._subtensor_guard.run(
            lambda: self.subtensor.get_current_block()
        )

    async def get_last_update_blocks(self) -> int:
        """Get blocks since last weight update for this validator from chain."""
        try:

            def _get_validator_info():
                current_block = self.subtensor.get_current_block()
                validator_hotkey = self.wallet.hotkey.ss58_address

                if validator_hotkey in self.metagraph.hotkeys:
                    validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
                else:
                    raise ValueError(
                        f"Validator {validator_hotkey} not found in metagraph"
                    )

                last_update_blocks = self.subtensor.blocks_since_last_update(
                    self.netuid, validator_uid
                )

                bt.logging.debug(
                    f"Subtensor head={current_block} validator_uid={validator_uid}"
                )
                return last_update_blocks

            return await self._subtensor_guard.run(_get_validator_info)

        except Exception as e:
            bt.logging.error(f"❌ Last update blocks error | error={e}")

            return 0

    async def _should_set_weights(self) -> bool:
        """Check if weight submission conditions are met based on chain tempo.

        Behavior:
        - metagraph.weight_update_tempo > 0: submit when blocks_since_last_update >= tempo
        - metagraph.weight_update_tempo == 0: submission disabled
        """
        try:
            # Use fail-fast method with type validation (allow 0 for disabled)
            tempo_val = self.config.get("metagraph.weight_update_tempo")
            if not isinstance(tempo_val, int) or tempo_val < 0:
                raise ValueError(
                    f"metagraph.weight_update_tempo must be non-negative int, got {type(tempo_val).__name__}: {tempo_val}"
                )

            if tempo_val == 0:
                bt.logging.info("Weight submission disabled | tempo=0")
                return False

            if tempo_val < 0:
                raise ValueError(
                    f"metagraph.weight_update_tempo must be >= 0, got {tempo_val}"
                )

            last_update_blocks = await self.get_last_update_blocks()
            should_submit = last_update_blocks >= int(tempo_val)

            bt.logging.debug(
                f"Weights check | tempo={tempo_val} last={last_update_blocks} submit={should_submit}"
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

            # Read active hotkeys from cache snapshot
            if self.metagraph_cache is not None:
                active_hotkeys = self.metagraph_cache.get_hotkeys()
            else:
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

            # Batch load all latest weights for all miners in one query
            # Find latest weight record for each hotkey
            subq = (
                session.query(
                    NetworkWeight.hotkey,
                    func.max(NetworkWeight.created_at).label("max_created"),
                )
                .filter(NetworkWeight.deleted_at.is_(None))
                .group_by(NetworkWeight.hotkey)
                .subquery()
            )

            # Join to get the full records
            all_latest_weights = (
                session.query(NetworkWeight)
                .join(
                    subq,
                    (NetworkWeight.hotkey == subq.c.hotkey)
                    & (NetworkWeight.created_at == subq.c.max_created),
                )
                .filter(NetworkWeight.deleted_at.is_(None))
                .all()
            )

            # Create lookup map
            latest_weight_map = {w.hotkey: w for w in all_latest_weights}

            # Track skipped miners for debugging
            skipped_no_weight = []
            skipped_no_uid = []

            for miner in miners:
                latest_weight = latest_weight_map.get(miner.hotkey)

                if latest_weight:
                    uid = self._get_uid_for_hotkey(miner.hotkey)
                    if uid is not None:
                        uids.append(uid)
                        weights.append(
                            0.0 if not miner.is_online else latest_weight.weight_value
                        )
                        weight_records_to_update.append(latest_weight)
                    else:
                        skipped_no_uid.append(miner.hotkey)
                else:
                    skipped_no_weight.append(miner.hotkey)

            # Log skipped miners for debugging
            if skipped_no_weight or skipped_no_uid:
                bt.logging.warning(
                    f"⚠️ Skipped miners in submission | "
                    f"no_weight={len(skipped_no_weight)} no_uid={len(skipped_no_uid)}"
                )
                if skipped_no_uid:
                    bt.logging.debug(f"Skipped (no UID): {skipped_no_uid[:3]}...")  # Show first 3
                if skipped_no_weight:
                    bt.logging.debug(f"Skipped (no weight): {skipped_no_weight[:3]}...")

            if uids:

                # Skip SDK call when all weights are zero
                has_positive_weight = any((w or 0.0) > 0.0 for w in weights)
                if not has_positive_weight:
                    bt.logging.warning(
                        "⚠️ Skip weight submission | reason=all_zero_weights"
                    )
                    return

                success = await self._submit_weights_to_chain(uids, weights)

                if success:

                    with self.db_manager.get_session() as update_session:
                        effective_iso = datetime.now(timezone.utc).isoformat()
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
        """Get UID for a given hotkey from cached metagraph snapshot."""
        if self.metagraph_cache is not None:
            return self.metagraph_cache.get_uid(hotkey)
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
            uids_array = np.array(uids)
            weights_array = np.array(weights)
            uint_uids, uint_weights = convert_weights_and_uids_for_emit(
                uids_array, weights_array
            )

            result, msg = await self._subtensor_guard.run(
                lambda: self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uint_uids,
                    weights=uint_weights,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                )
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
        if (datetime.utcnow() - self._start_time) < timedelta(
            seconds=WEIGHT_CALCULATION_INTERVAL
        ):
            bt.logging.debug("Skip online status update | reason=warmup_window")
            return

        try:
            with self.db_manager.get_session() as session:

                offline_worker_count = self.db_manager.update_worker_online_status(
                    session=session, offline_threshold_minutes=30
                )

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
        """Calculates and stores pending weights in a worker thread; event loop only publishes."""
        if not self.is_running:
            return

        loop = asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(None, self._calculate_all_weights_sync)
        except Exception as e:
            bt.logging.error(f"❌ Weight calc (thread) error | error={e}")
            return

        # Send to MeshHub once per cycle (non-blocking)
        try:
            if self.meshhub_client and (
                payload.get("workerScores") or payload.get("minerScores")
            ):
                await self.meshhub_client.publish_score_report(
                    worker_scores=payload.get("workerScores"),
                    miner_scores=payload.get("minerScores"),
                    global_stats=payload.get("globalStats"),
                )
        except Exception as e:
            bt.logging.warning(f"⚠️ Score report publish failed | error={e}")

    def _calculate_all_weights_sync(self) -> Dict[str, Any]:
        """Synchronous heavy path: DB scan, weight calc, DB write, payload build (lease-only mode)."""
        out: Dict[str, Any] = {
            "workerScores": [],
            "minerScores": [],
            "globalStats": None,
        }

        with self.db_manager.get_session() as session:
            active_hotkeys = self.metagraph.hotkeys
            if not active_hotkeys:
                bt.logging.warning("⚠️ No active miners for scoring")
                return out

            active_miners = (
                session.query(MinerInfo)
                .filter(MinerInfo.hotkey.in_(active_hotkeys))
                .all()
            )

            bt.logging.debug(f"Weight calc (lease-only) | miners={len(active_miners)}")

            # Batch load all workers for all miners to avoid N+1 queries
            all_workers = (
                session.query(WorkerInfo)
                .filter(
                    WorkerInfo.hotkey.in_([m.hotkey for m in active_miners]),
                    WorkerInfo.deleted_at.is_(None),
                )
                .order_by(WorkerInfo.hotkey, WorkerInfo.lease_score.desc())
                .all()
            )

            # Group workers by miner hotkey
            workers_by_hotkey: Dict[str, List[WorkerInfo]] = defaultdict(list)
            for worker in all_workers:
                workers_by_hotkey[worker.hotkey].append(worker)

            bt.logging.debug(
                f"Batch loaded workers | miners={len(workers_by_hotkey)} workers={len(all_workers)}"
            )

            miner_scores = []

            for miner in active_miners:
                score, score_details = self._calculate_miner_score(
                    miner,
                    workers_by_hotkey.get(miner.hotkey, []),
                )
                # Store only primitive hotkey to avoid detached ORM access outside session
                miner_scores.append(
                    {
                        "hotkey": miner.hotkey,
                        "score": score,
                        "score_details": score_details,
                    }
                )

            weights = self._calculate_weights_from_scores(miner_scores)

            # Build MeshHub payload (simplified for lease-only mode)
            miner_scores_payload: List[Dict[str, Any]] = []
            try:
                timestamp_iso = datetime.utcnow().isoformat() + "Z"

                # Build minerScores payload for all miners
                for ms in miner_scores:
                    hk = ms["hotkey"]
                    miner_scores_payload.append(
                        {
                            "hotkey": hk,
                            "weight": round(float(weights.get(hk, 0.0) or 0.0), 6),
                        }
                    )
                
                # Build simplified global stats
                global_stats_payload = {
                    "totalMiners": len(active_miners),
                    "totalWorkers": len(all_workers),
                    "calculatedAt": timestamp_iso,
                    "mode": "lease-only",
                }
            except Exception as e:
                bt.logging.warning(f"⚠️ Build MeshHub score payload failed | error={e}")
                miner_scores_payload = []
                global_stats_payload = None

            # Count only miners with non-zero weights (online miners)
            online_count = sum(1 for w in weights.values() if w > 0)
            offline_count = len(weights) - online_count

            bt.logging.debug(
                f"Save pending weights | total_in_metagraph={len(weights)} "
                f"online_with_weights={online_count} offline_zero_weight={offline_count}"
            )
            for miner_data in miner_scores:
                m_hotkey = miner_data["hotkey"]
                weight = weights.get(m_hotkey, 0.0)

                # Store lease-only score details
                detail_json = json.dumps({"mode": "lease-only"})

                self.db_manager.record_weight_update(
                    session=session,
                    hotkey=m_hotkey,
                    weight_value=weight,
                    scores=miner_data["score_details"],
                    calculation_remark=detail_json,
                    is_applied=False,
                )

            out["workerScores"] = []  # No worker performance scores in lease-only mode
            out["minerScores"] = miner_scores_payload
            out["globalStats"] = global_stats_payload

        return out

    def _calculate_miner_score(
        self,
        miner: MinerInfo,
        preloaded_workers: Optional[List[WorkerInfo]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate miner score (lease-only mode)

        Total lease score (sum of all worker lease scores) is used directly
        for weight calculation without penalties or normalization.

        Args:
            miner: Miner to score
            preloaded_workers: Pre-loaded worker list

        Returns:
            (composite_score, score_details_dict)
        """
        scores: Dict[str, float] = {}

        # If miner is offline, force score to 0
        if not miner.is_online:
            scores["lease_score"] = 0.0
            scores["composite_score"] = 0.0
            return 0.0, scores

        # Calculate lease score (aggregated from workers)
        lease_score = self._calculate_worker_lease_score(miner, preloaded_workers)
        scores["lease_score"] = lease_score

        # Composite score = lease score (no penalty mechanism)
        composite_score = lease_score
        
        scores["composite_score"] = composite_score
        bt.logging.debug(
            f"Miner {miner.hotkey} total_lease={lease_score:.2f} final={composite_score:.2f}"
        )
        
        return composite_score, scores

    def _calculate_worker_lease_score(
        self, miner: MinerInfo, preloaded_workers: Optional[List[WorkerInfo]] = None
    ) -> float:
        """Calculate aggregated lease score from all workers of this miner

        Args:
            miner: Miner to calculate lease score for
            preloaded_workers: Optional pre-loaded workers list (avoids DB query)

        Returns:
            Total lease score (sum of all worker lease scores)
        """
        try:
            # Use preloaded workers if available (batch optimization)
            if preloaded_workers is not None:
                workers = preloaded_workers
            else:
                # Fallback to DB query for backward compatibility
                with self.db_manager.get_session() as session:
                    workers = (
                        session.query(WorkerInfo)
                        .filter(
                            WorkerInfo.hotkey == miner.hotkey,
                            WorkerInfo.deleted_at.is_(None),
                        )
                        .order_by(WorkerInfo.lease_score.desc())
                        .all()
                    )

            if not workers:
                return 0.0

            # Sum all worker lease scores (no limit, no normalization)
            total_lease_score = sum(
                worker.lease_score or 0.0 for worker in workers
            )
            worker_count = len(workers)

            bt.logging.debug(
                f"Miner {miner.hotkey} total lease score: {total_lease_score:.2f} "
                f"from {worker_count} workers"
            )

            return total_lease_score

        except Exception as e:
            bt.logging.error(
                f"Failed to calculate worker lease score for miner {miner.hotkey}: {e}"
            )
            return 0.0

    def _calculate_weights_from_scores(
        self, miner_scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate weights from scores"""
        if not miner_scores:
            return {}

        scores = [float(data.get("score") or 0.0) for data in miner_scores]

        max_score = max(scores) if scores else 0.0
        if max_score <= 0.0:
            bt.logging.warning("⚠️ Skip weight calc | reason=zero_scores")
            return {}

        total_score = sum(scores)

        weights = {}
        for i, data in enumerate(miner_scores):
            if total_score > 0:
                weight = scores[i] / total_score
            else:
                weight = 1.0 / len(miner_scores)

            weight = max(0, min(1, weight))
            weights[data["hotkey"]] = weight

        total_weight = sum(weights.values())
        if total_weight > 0:
            for hotkey in weights:
                weights[hotkey] /= total_weight

        return weights

    def get_weight_status(self) -> Dict[str, Any]:
        """Get weight management status"""
        return {
            "is_running": self.is_running,
            "last_weight_update": self._last_weight_update,
            "calculation_interval": WEIGHT_CALCULATION_INTERVAL,
            "submission_check_interval": WEIGHT_SUBMISSION_CHECK_INTERVAL,
            "mode": "lease-only",
            "lease_weight": self.config.get_range(
                "weight_management.score_weights.lease_weight", 0.0, 1.0, float
            ),
            "netuid": self.netuid,
        }
