"""Validator weight management service"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.models.database import (DatabaseManager, MinerInfo,
                                               NetworkWeight, WorkerInfo)
from neurons.validator.services.worker_performance_ranker import \
    WorkerPerformanceRanker

# Timing constants
WEIGHT_CALCULATION_INTERVAL = 420
WEIGHT_SUBMISSION_CHECK_INTERVAL = 30

CHALLENGE_SCORE_CAP = 100


class WeightManager:
    """Calculate and set miner network weights"""

    def __init__(
        self,
        database_manager: DatabaseManager,
        wallet: bt.wallet,
        subtensor: bt.subtensor,
        metagraph: bt.metagraph,
        config: ConfigManager,
        meshhub_client=None,
    ):
        """Initialize weight manager"""
        self.db_manager = database_manager
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.config = config
        self.meshhub_client = meshhub_client

        self.netuid = config.get_positive_number("netuid", int)

        lease_weight = config.get("weight_management.score_weights.lease_weight")
        challenge_weight = config.get(
            "weight_management.score_weights.challenge_weight"
        )
        bt.logging.info(
            f"⚖️ Score weights | lease={float(lease_weight):.2%} challenge={float(challenge_weight):.2%}"
        )

        self.performance_ranker = WorkerPerformanceRanker(
            database_manager,
            config.get_positive_number("validation.challenge_interval", int),
            config.get_range(
                "validation.participation_rate_threshold", 0.1, 1.0, float
            ),
        )

        self.is_running = False
        self._scoring_task: Optional[asyncio.Task] = None
        self._setting_task: Optional[asyncio.Task] = None
        self._last_weight_update = 0.0

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

                await self._update_worker_online_status()

                await self._calculate_all_weights()

                bt.logging.info("✅ Weight calc cycle done")

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

                    return last_update_blocks

                return await loop.run_in_executor(None, _get_validator_info)

        except Exception as e:
            bt.logging.error(f"❌ Last update blocks error | error={e}")

            return 0

    async def _should_set_weights(self) -> bool:
        """
        Check if weight submission conditions are met based on chain tempo.
        """
        try:

            tempo = self.config.get_positive_number(
                "metagraph.weight_update_tempo", int
            )
            last_update_blocks = await self.get_last_update_blocks()

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
                        weights.append(
                            0.0 if not miner.is_online else latest_weight.weight_value
                        )
                        weight_records_to_update.append(latest_weight)

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
                        from datetime import datetime, timezone

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

            uids_array = np.array(uids)
            weights_array = np.array(weights)
            uint_uids, uint_weights = convert_weights_and_uids_for_emit(
                uids_array, weights_array
            )

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
        """Calculates and stores pending weights for all active miners."""
        if not self.is_running:
            return

        with self.db_manager.get_session() as session:
            active_hotkeys = self.metagraph.hotkeys
            if not active_hotkeys:
                bt.logging.warning("⚠️ No active miners for scoring")
                return

            active_miners = (
                session.query(MinerInfo)
                .filter(MinerInfo.hotkey.in_(active_hotkeys))
                .all()
            )

            bt.logging.debug(f"Weight calc | miners={len(active_miners)}")

            bt.logging.debug("Pre-calculating worker performance")

            self.performance_ranker.challenge_interval = (
                self.config.get_positive_number("validation.challenge_interval", int)
            )
            self.performance_ranker.participation_rate_threshold = (
                self.config.get_range(
                    "validation.participation_rate_threshold", 0.1, 1.0, float
                )
            )

            eval_window = self.config.get_positive_number(
                "validation.ranking_window_minutes", int
            )

            worker_rankings = self.performance_ranker.calculate_worker_performance(
                evaluation_window_minutes=eval_window
            )

            miner_challenge_scores = (
                self.performance_ranker.calculate_miner_challenge_scores(
                    worker_rankings
                )
            )
            max_raw_challenge = (
                max(miner_challenge_scores.values()) if miner_challenge_scores else 0.0
            )

            # Build normalized challenge map for all active miners and compute global average
            challenge_norm_map: Dict[str, float] = {}
            for miner in active_miners:
                raw = miner_challenge_scores.get(miner.hotkey, 0.0)
                if max_raw_challenge > 0:
                    challenge_norm = min(1.0, max(0.0, raw / max_raw_challenge))
                else:
                    challenge_norm = 0.0
                challenge_norm_map[miner.hotkey] = challenge_norm

            if challenge_norm_map:
                avg_challenge_norm = sum(challenge_norm_map.values()) / float(
                    len(challenge_norm_map)
                )
            else:
                avg_challenge_norm = 0.0

            miner_scores = []

            for miner in active_miners:
                score, score_details = self._calculate_miner_score(
                    miner, challenge_norm_map, avg_challenge_norm
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

            # Build MeshHub workerScores (worker-level only) and global stats for this cycle

            worker_scores_payload: List[Dict[str, Any]] = []
            global_stats_payload: Optional[Dict[str, Any]] = None
            try:
                try:
                    max_worker_perf = (
                        max(
                            float(ws.performance_score or 0.0)
                            for ws in worker_rankings.values()
                        )
                        if worker_rankings
                        else 0.0
                    )
                except Exception:
                    max_worker_perf = 0.0

                timestamp_iso = datetime.utcnow().isoformat() + "Z"

                for ws in worker_rankings.values():
                    be = float(ws.baseline_expected or 0.0)
                    ap = int(ws.actual_participation or 0)
                    perf = float(ws.performance_score or 0.0)
                    norm = (
                        float(perf) / float(max_worker_perf)
                        if max_worker_perf > 0
                        else 0.0
                    )
                    worker_scores_payload.append(
                        {
                            "workerKey": f"{ws.hotkey}:{ws.worker_id}",
                            "score": round(norm, 6),
                            "factors": {
                                "participationBaselineCount": float(be),
                                "participationValidCount": int(ap),
                                "avgExecMs": float(ws.execution_time_ms or 0.0),
                                "avgExecGpus": float(
                                    getattr(ws, "success_count_avg", 0.0) or 0.0
                                ),
                                "rawPerfScore": perf,
                                "leaseScore": float(ws.lease_score or 0.0),
                            },
                            "calculatedAt": timestamp_iso,
                        }
                    )

                try:
                    avg_score = (
                        sum(w["score"] for w in worker_scores_payload)
                        / float(len(worker_scores_payload))
                        if worker_scores_payload
                        else 0.0
                    )
                except Exception:
                    avg_score = 0.0

                window_start_iso = (
                    datetime.utcnow() - timedelta(minutes=eval_window)
                ).isoformat() + "Z"
                window_end_iso = datetime.utcnow().isoformat() + "Z"
                pr_threshold = self.config.get_range(
                    "validation.participation_rate_threshold", 0.1, 1.0, float
                )
                global_stats_payload = {
                    "totalWorkers": len(worker_scores_payload),
                    "averageScore": round(avg_score, 6),
                    "windowStart": window_start_iso,
                    "windowEnd": window_end_iso,
                    "participationThreshold": float(pr_threshold),
                    "workerCap": int(CHALLENGE_SCORE_CAP),
                }
            except Exception as e:
                bt.logging.warning(f"⚠️ Build MeshHub score payload failed | error={e}")
                worker_scores_payload = []
                global_stats_payload = None

            worker_factors_by_miner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for item in worker_scores_payload or []:
                wk = item.get("workerKey")
                if not isinstance(wk, str) or ":" not in wk:
                    continue
                hk, wid = wk.split(":", 1)
                if wid == "-1":
                    continue
                factors = item.get("factors") or {}
                worker_factors_by_miner[hk].append(factors)

            bt.logging.debug(f"Save pending weights | count={len(weights)}")
            for miner_data in miner_scores:
                m_hotkey = miner_data["hotkey"]
                weight = weights.get(m_hotkey, 0.0)

                # remark stores array of worker factors for this miner
                miner_worker_factors = worker_factors_by_miner.get(m_hotkey, [])
                detail_json = json.dumps(miner_worker_factors)

                self.db_manager.record_weight_update(
                    session=session,
                    hotkey=m_hotkey,
                    weight_value=weight,
                    scores=miner_data["score_details"],
                    calculation_remark=detail_json,
                    is_applied=False,
                )

            # Send to MeshHub once per cycle
            try:
                if self.meshhub_client and worker_scores_payload:
                    await self.meshhub_client.publish_score_report(
                        worker_scores=worker_scores_payload,
                        global_stats=global_stats_payload,
                    )
            except Exception as e:
                bt.logging.warning(f"⚠️ Score report publish failed | error={e}")

    def _calculate_miner_score(
        self,
        miner: MinerInfo,
        challenge_norm_map: Dict[str, float],
        avg_challenge_norm: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate miner score.

        - Challenge score: normalized to 0..1 by dividing the current-round max raw score.
        - Lease score: aggregated and normalized to 0..1.
        """
        scores: Dict[str, float] = {}

        # If miner is offline, force zero score for the next epoch
        if not miner.is_online:
            scores["availability_score"] = 0.0
            scores["lease_score"] = 0.0
            scores["challenge_score"] = 0.0
            scores["raw_challenge_norm"] = float(
                challenge_norm_map.get(miner.hotkey, 0.0) or 0.0
            )
            scores["lease_weight"] = float(
                self.config.get("weight_management.score_weights.lease_weight")
            )
            scores["challenge_weight"] = float(
                self.config.get("weight_management.score_weights.challenge_weight")
            )
            scores["composite_score"] = 0.0
            return 0.0, scores

        # Challenge score: normalized 0..1 from precomputed map
        challenge_norm = float(challenge_norm_map.get(miner.hotkey, 0.0) or 0.0)
        scores["raw_challenge_norm"] = challenge_norm
        scores["lease_score"] = self._calculate_worker_lease_score(miner)

        # If lease_score > 0, override challenge score with global average
        effective_challenge_norm = (
            float(avg_challenge_norm)
            if (scores["lease_score"] or 0.0) > 0.0
            else challenge_norm
        )
        scores["challenge_score"] = effective_challenge_norm
        bt.logging.debug(
            f"Miner {miner.hotkey} challenge_norm={challenge_norm:.4f} avg_norm={avg_challenge_norm:.4f} lease={scores['lease_score']:.4f}"
        )

        lease_w = float(self.config.get("weight_management.score_weights.lease_weight"))
        chall_w = float(
            self.config.get("weight_management.score_weights.challenge_weight")
        )
        scores["lease_weight"] = lease_w
        scores["challenge_weight"] = chall_w

        # Short-circuit: no lease and no challenge → score 0, skip availability query
        if (scores["lease_score"] or 0.0) <= 0.0 and (
            scores["challenge_score"] or 0.0
        ) <= 0.0:
            scores["availability_score"] = 0.0
            scores["composite_score"] = 0.0
            return 0.0, scores

        composite_score = (
            scores["challenge_score"] * chall_w + scores["lease_score"] * lease_w
        )
        scores["composite_score"] = composite_score

        availability_score = self._calculate_online_weight_from_heartbeats(miner)
        scores["availability_score"] = availability_score

        total_score = composite_score * availability_score

        return total_score, scores

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

                total_lease_score = sum(worker.lease_score or 0.0 for worker in workers)
                worker_count = len(workers)

                if total_lease_score > 0:

                    max_workers = min(CHALLENGE_SCORE_CAP, worker_count)

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

            heartbeat_records = (
                session.query(HeartbeatRecord)
                .filter(HeartbeatRecord.hotkey == miner.hotkey)
                .filter(HeartbeatRecord.created_at >= window_start)
                .order_by(HeartbeatRecord.created_at.asc())
                .all()
            )

            if not heartbeat_records:
                return 0.0

            expected_intervals = 169 * 12

            online_intervals = set()
            for record in heartbeat_records:

                interval_index = int(record.created_at.timestamp() // 300)
                online_intervals.add(interval_index)

            actual_online_intervals = len(online_intervals)

            online_ratio = min(1.0, actual_online_intervals / expected_intervals)

            # 169h window: each change halves the score
            ip_changes = 0
            last_ip = None
            for record in heartbeat_records:
                ip = getattr(record, "public_ip", None)
                if not ip:
                    continue
                if last_ip is None:
                    last_ip = ip
                    continue
                if ip != last_ip:
                    ip_changes += 1
                    last_ip = ip

            if ip_changes > 0:
                # Each IP change halves the accumulated availability
                penalty = 0.5**ip_changes
                if penalty < 0.1:
                    penalty = 0.0
                online_ratio *= penalty

            return online_ratio

    def _calculate_weights_from_scores(
        self, miner_scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate weights from scores"""
        if not miner_scores:
            return {}

        scores = [data["score"] for data in miner_scores]

        if max(scores) == 0:

            uniform_weight = 1.0 / len(miner_scores)
            return {data["hotkey"]: uniform_weight for data in miner_scores}

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
                convert_weights_and_uids_for_emit, process_weights_for_netuid)

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
            "score_weights": {
                "lease_weight": self.config.get(
                    "weight_management.score_weights.lease_weight"
                ),
                "challenge_weight": self.config.get(
                    "weight_management.score_weights.challenge_weight"
                ),
            },
            "netuid": self.netuid,
        }

    def get_miner_weight_info(self, hotkey: str) -> Dict[str, Any]:
        """Get specific miner's weight information using performance scoring"""
        with self.db_manager.get_session() as session:
            miner = session.query(MinerInfo).filter(MinerInfo.hotkey == hotkey).first()

            if not miner:
                return {}

            self.performance_ranker.challenge_interval = (
                self.config.get_positive_number("validation.challenge_interval", int)
            )
            self.performance_ranker.participation_rate_threshold = (
                self.config.get_range(
                    "validation.participation_rate_threshold", 0.1, 1.0, float
                )
            )
            eval_window = self.config.get_positive_number(
                "validation.ranking_window_minutes", int
            )

            worker_rankings = self.performance_ranker.calculate_worker_performance(
                evaluation_window_minutes=eval_window
            )
            raw_miner_challenge_scores = (
                self.performance_ranker.calculate_miner_challenge_scores(
                    worker_rankings
                )
            )
            # Normalize challenges across all miners in metagraph for per-miner query
            challenge_norm_map: Dict[str, float] = {}
            all_hotkeys = list(self.metagraph.hotkeys)
            max_raw = (
                max(raw_miner_challenge_scores.values())
                if raw_miner_challenge_scores
                else 0.0
            )
            for hk in all_hotkeys:
                raw_val = raw_miner_challenge_scores.get(hk, 0.0)
                challenge_norm_map[hk] = (
                    min(1.0, max(0.0, raw_val / max_raw)) if max_raw > 0 else 0.0
                )
            avg_norm = (
                sum(challenge_norm_map.values()) / len(challenge_norm_map)
                if challenge_norm_map
                else 0.0
            )

            score, score_details = self._calculate_miner_score(
                miner, challenge_norm_map, avg_norm
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
