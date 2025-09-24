"""
Worker Performance Scoring System
Implements global worker absolute-performance scoring based on execution time per GPU
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from neurons.validator.challenge_status import ChallengeStatus
from neurons.validator.models.database import (ComputeChallenge,
                                               DatabaseManager, WorkerInfo)


class WorkerPerformanceScore:
    """Individual worker performance data"""

    def __init__(
        self,
        worker_id: str,
        hotkey: str,
        execution_time_ms: float,
        lease_score: float = 0.0,
        success_rate: float = 1.0,
        total_attempts: int = 1,
        total_compute_ms: float = 0.0,
        success_count_avg: float = 1.0,
    ):
        self.worker_id = worker_id
        self.hotkey = hotkey
        self.execution_time_ms = execution_time_ms
        self.lease_score = lease_score
        self.success_rate = success_rate
        self.total_attempts = total_attempts
        self.total_compute_ms = total_compute_ms
        self.success_count_avg = success_count_avg
        self.performance_score = 0.0  # Will be calculated by ranker
        self.order_index = 0  # Position in ordered list (0-based)

        # Participation tiering fields
        self.participation_score = 0.0  # Participation score for ordering
        self.actual_participation = 0  # Actual successful attempts
        self.baseline_expected = 0.0  # Expected baseline for participation


class WorkerPerformanceRanker:
    """
    Global worker performance scoring system

    Core logic:
    1. Collect verified worker challenge results in the evaluation period
    2. Compute average execution time per GPU for each worker
    3. Assign worker score = 1 / avg_time_ms (higher = faster)
    4. Aggregate worker scores to miner-level by summing (cap 100 workers)
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        challenge_interval: int = 180,
        participation_rate_threshold: float = 0.75,
    ):
        self.db_manager = database_manager
        self.performance_cache: Dict[str, WorkerPerformanceScore] = {}
        self.last_update_ts = 0.0

        # Participation configuration
        self.challenge_interval = challenge_interval
        self.participation_rate_threshold = participation_rate_threshold

    def calculate_worker_performance(
        self, evaluation_window_minutes: int
    ) -> Dict[str, WorkerPerformanceScore]:
        """
        Calculate global worker performance metrics using participation tiering for sorting
        and absolute-performance worker scoring.

        Args:
            evaluation_window_minutes: Time window for collecting challenge results

        Returns:
            Dictionary mapping worker_key to WorkerPerformanceScore
        """
        bt.logging.info(
            f"Calculating worker performance | window={evaluation_window_minutes}m"
        )

        with self.db_manager.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(
                minutes=evaluation_window_minutes
            )

            # Calculate unified participation baseline
            max_possible_challenges = (
                evaluation_window_minutes * 60
            ) / self.challenge_interval
            baseline_expected = (
                max_possible_challenges * self.participation_rate_threshold
            )

            bt.logging.info(
                f"📊 Baseline: {evaluation_window_minutes}min window, "
                f"{self.challenge_interval}s interval → max {max_possible_challenges:.0f} challenges, "
                f"{self.participation_rate_threshold:.0%} threshold = {baseline_expected:.1f}"
            )

            # Get all recent challenges, considering only those that passed two-phase verification
            all_challenges = (
                session.query(ComputeChallenge)
                .filter(
                    ComputeChallenge.created_at >= cutoff_time,
                    ComputeChallenge.deleted_at.is_(None),
                    ComputeChallenge.challenge_status == ChallengeStatus.VERIFIED,
                )
                .order_by(ComputeChallenge.created_at.desc())
                .all()
            )

            if not all_challenges:
                bt.logging.warning("No recent challenges found for scoring")
                return {}

            bt.logging.debug(f"Processing {len(all_challenges)} recent challenges")
            worker_stats = {}

            # Build GPU uuid -> canonical hotkey map from inventory
            uuid_owner_map: Dict[str, str] = {}
            try:
                from neurons.validator.models.database import GPUInventory

                gpu_rows = (
                    session.query(GPUInventory)
                    .filter(
                        GPUInventory.deleted_at.is_(None),
                        GPUInventory.last_seen_at >= cutoff_time,
                    )
                    .all()
                )
                uuid_owner_map = {r.gpu_uuid: r.hotkey for r in gpu_rows if r.gpu_uuid}
            except Exception as e:
                bt.logging.warning(
                    f"Failed to load GPU inventory for uuid ownership: {e}"
                )
                uuid_owner_map = {}

            # Statistics collection
            for challenge in all_challenges:
                worker_key = f"{challenge.hotkey}_{challenge.worker_id}"

                if worker_key not in worker_stats:
                    worker_stats[worker_key] = {
                        "hotkey": challenge.hotkey,
                        "worker_id": challenge.worker_id,
                        "successful_times": [],
                        "total_attempts": 0,
                        "successful_attempts": 0,
                        "total_compute_ms": 0.0,
                        "total_success_count": 0,
                    }

                # Enforce single-hotkey-per-uuid participation for GPU challenges
                try:
                    mc = challenge.merkle_commitments or {}
                    # GPU commitments use real uuids; CPU uses "-1"
                    gpu_uuids = [
                        u
                        for u in (mc.keys() if isinstance(mc, dict) else [])
                        if u and u != "-1"
                    ]
                    if gpu_uuids:
                        # If any uuid's canonical owner hotkey differs, skip counting this challenge
                        mismatch = False
                        for u in gpu_uuids:
                            owner = uuid_owner_map.get(u)
                            if owner is not None and owner != challenge.hotkey:
                                mismatch = True
                                break
                        if mismatch:
                            # Do not count towards participation or performance
                            continue
                except Exception:
                    pass

                worker_stats[worker_key]["total_attempts"] += 1

                if (
                    challenge.is_success
                    and challenge.verification_result
                    and challenge.computation_time_ms is not None
                ):
                    worker_stats[worker_key]["successful_attempts"] += 1

                    # Multi-GPU challenges need per-unit normalization
                    success_count = getattr(challenge, "success_count", None)
                    if success_count is None or success_count == 0:
                        success_count = 1

                    # Calculate normalized time per GPU/processing unit
                    normalized_time = challenge.computation_time_ms / success_count
                    worker_stats[worker_key]["successful_times"].append(normalized_time)
                    worker_stats[worker_key]["total_compute_ms"] += float(
                        challenge.computation_time_ms
                    )
                    worker_stats[worker_key]["total_success_count"] += int(
                        success_count
                    )
                    # Keep only aggregate counts; last success_count not needed

            # Calculate metrics and absolute scores for each worker
            workers_for_ordering = []
            worker_lease_scores = self._get_worker_lease_scores(session)

            for worker_key, stats in worker_stats.items():
                if stats["successful_attempts"] == 0:
                    continue

                # Participation metrics
                actual_participation = stats["successful_attempts"]
                participation_score = min(baseline_expected, actual_participation)

                # Average execution time
                average_time = sum(stats["successful_times"]) / len(
                    stats["successful_times"]
                )
                success_rate = stats["successful_attempts"] / stats["total_attempts"]
                total_compute_ms = float(stats.get("total_compute_ms", 0.0))
                tsc = int(stats.get("total_success_count", 0))
                success_count_avg = (
                    (tsc / stats["successful_attempts"])
                    if stats["successful_attempts"] > 0
                    else 0.0
                )
                lease_score = worker_lease_scores.get(worker_key, 0.0)

                workers_for_ordering.append(
                    {
                        "worker_key": worker_key,
                        "hotkey": stats["hotkey"],
                        "worker_id": stats["worker_id"],
                        "participation_score": participation_score,
                        "average_time": average_time,
                        "actual_participation": actual_participation,
                        "baseline_expected": baseline_expected,
                        "success_rate": success_rate,
                        "lease_score": lease_score,
                        "total_attempts": stats["total_attempts"],
                        "total_compute_ms": total_compute_ms,
                        "success_count_avg": success_count_avg,
                    }
                )

            # Tier sorting (participation tier, then avg execution time)
            workers_for_ordering.sort(
                key=lambda x: (
                    -x["participation_score"],  # Higher participation first
                    x["average_time"],  # Faster first
                )
            )

            # Generate final ordering and scores
            worker_scores = {}
            total_workers = len(workers_for_ordering)

            for rank, worker_data in enumerate(workers_for_ordering):
                worker_key = worker_data["worker_key"]

                worker_score = WorkerPerformanceScore(
                    worker_id=worker_data["worker_id"],
                    hotkey=worker_data["hotkey"],
                    execution_time_ms=worker_data["average_time"],
                    lease_score=worker_data["lease_score"],
                    success_rate=worker_data["success_rate"],
                    total_attempts=worker_data["total_attempts"],
                    total_compute_ms=worker_data.get("total_compute_ms", 0.0),
                    success_count_avg=worker_data.get("success_count_avg", 0.0),
                )

                # Set ordering fields
                worker_score.order_index = rank
                worker_score.participation_score = worker_data["participation_score"]
                worker_score.actual_participation = worker_data["actual_participation"]
                worker_score.baseline_expected = worker_data["baseline_expected"]

                # Absolute-performance score with participation multiplier
                base_score = self._calculate_absolute_performance_score(
                    worker_score.execution_time_ms
                )
                be = worker_score.baseline_expected
                ap = worker_score.actual_participation
                participation_coeff = (
                    1.0
                    if (be is None or be <= 0)
                    else min(1.0, (ap / be) if be > 0 else 1.0)
                )
                worker_score.performance_score = base_score * participation_coeff

                worker_scores[worker_key] = worker_score

            self.performance_cache = worker_scores
            self.last_update_ts = time.time()

            # Logging results
            self._log_performance_summary(worker_scores, evaluation_window_minutes)

            return worker_scores

    def calculate_miner_challenge_scores(
        self, ranked_workers: Dict[str, WorkerPerformanceScore]
    ) -> Dict[str, float]:
        """
        Aggregate worker performance scores to miner-level challenge scores

        Unleased workers receive scores based on actual challenge performance.
        Formula: sum(top 100 worker absolute scores)

        Args:
            ranked_workers: Worker performance map from calculate_worker_performance()

        Returns:
            Dictionary mapping miner hotkey to challenge score (raw sum, not normalized)
        """
        miner_scores: Dict[str, List[float]] = {}

        # Group worker scores by miner
        for worker_key, worker_score in ranked_workers.items():
            hotkey = worker_score.hotkey

            if hotkey not in miner_scores:
                miner_scores[hotkey] = []

            miner_scores[hotkey].append(worker_score.performance_score)

        # Calculate final miner challenge scores
        miner_challenge_scores = {}

        # Sum absolute scores per miner with a cap of 100 workers
        all_miners = set(miner_scores.keys())
        for hotkey in all_miners:
            worker_scores = miner_scores.get(hotkey, [])
            top_worker_scores = sorted(worker_scores, reverse=True)[:100]
            total_score = sum(top_worker_scores)
            miner_challenge_scores[hotkey] = total_score

        bt.logging.debug(
            f"Calculated challenge scores for {len(miner_challenge_scores)} miners"
        )

        return miner_challenge_scores

    def get_worker_performance_score(self, hotkey: str, worker_id: str) -> float:
        """Get cached performance score for a specific worker"""
        worker_key = f"{hotkey}_{worker_id}"

        if worker_key in self.performance_cache:
            return self.performance_cache[worker_key].performance_score

        return 0.0  # No recent performance data

    def get_miner_worker_count_and_scores(self, hotkey: str) -> Tuple[int, List[float]]:
        """Get worker count and scores for a specific miner"""
        worker_scores = []

        for worker_key, worker_score in self.performance_cache.items():
            if worker_score.hotkey == hotkey:
                worker_scores.append(worker_score.performance_score)

        return len(worker_scores), worker_scores

    # Removed legacy rank-based scoring function

    def _calculate_absolute_performance_score(self, average_time_ms: float) -> float:
        """
        Absolute-performance worker score based on inverse of average execution time per GPU.

        Higher score means faster worker. Zero time yields zero score.
        """
        try:
            t = float(average_time_ms)
            if t <= 0 or not (t < float("inf")):
                return 0.0
            return 1.0 / t
        except Exception:
            return 0.0

    def _log_performance_summary(
        self, worker_scores: Dict[str, WorkerPerformanceScore], window_minutes: int
    ):
        """Log summary of performance results"""
        if not worker_scores:
            return

        bt.logging.debug(
            f"Performance scoring completed | workers={len(worker_scores)}"
        )

        # Group by participation score for analysis
        participation_groups = {}
        for worker_key, score in worker_scores.items():
            ps = score.participation_score
            if ps not in participation_groups:
                participation_groups[ps] = []
            participation_groups[ps].append((worker_key, score))

        # Log top participation groups
        for ps in sorted(participation_groups.keys(), reverse=True)[:3]:
            workers = participation_groups[ps]
            workers.sort(key=lambda x: x[1].execution_time_ms)

            bt.logging.debug(f"Participation score {ps:.1f} | workers={len(workers)}")
            for i, (worker_key, score) in enumerate(workers[:3]):  # Show top 3 in group
                completion_rate = (
                    score.actual_participation / score.baseline_expected * 100
                )
                bt.logging.info(
                    f"  #{score.order_index+1} {worker_key}: "
                    f"{score.actual_participation} challenges ({completion_rate:.1f}%), "
                    f"{score.execution_time_ms:.1f}ms avg, SR:{score.success_rate:.1%}"
                )

        # Overall stats
        best_participation = max(s.participation_score for s in worker_scores.values())
        avg_participation = sum(
            s.participation_score for s in worker_scores.values()
        ) / len(worker_scores)
        bt.logging.info(
            f"📈 Participation stats - Best: {best_participation:.1f}, "
            f"Average: {avg_participation:.1f}, Baseline: {next(iter(worker_scores.values())).baseline_expected:.1f}"
        )

    def _get_worker_lease_scores(self, session) -> Dict[str, float]:
        """Get lease scores for all workers"""
        workers = (
            session.query(WorkerInfo).filter(WorkerInfo.deleted_at.is_(None)).all()
        )

        lease_scores = {}
        for worker in workers:
            worker_key = f"{worker.hotkey}_{worker.worker_id}"
            lease_scores[worker_key] = worker.lease_score or 0.0

        return lease_scores

    def _get_leased_worker_counts_by_miner(self, session) -> Dict[str, int]:
        """Get count of leased workers for each miner"""
        from sqlalchemy import func

        leased_worker_counts = (
            session.query(
                WorkerInfo.hotkey, func.count(WorkerInfo.worker_id).label("count")
            )
            .filter(WorkerInfo.deleted_at.is_(None), WorkerInfo.lease_score > 0.0)
            .group_by(WorkerInfo.hotkey)
            .all()
        )

        return {hotkey: count for hotkey, count in leased_worker_counts}

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get statistics about current performance cache"""
        if not self.performance_cache:
            return {
                "total_workers": 0,
                "last_update": self.last_update_ts,
                "score_distribution": {},
            }

        scores = [w.performance_score for w in self.performance_cache.values()]
        execution_times = [w.execution_time_ms for w in self.performance_cache.values()]

        # Calculate score distribution
        score_ranges = {
            "0.8-1.0": len([s for s in scores if s >= 0.8]),
            "0.5-0.8": len([s for s in scores if 0.5 <= s < 0.8]),
            "0.2-0.5": len([s for s in scores if 0.2 <= s < 0.5]),
            "0.0-0.2": len([s for s in scores if s < 0.2]),
        }

        return {
            "total_workers": len(self.performance_cache),
            "last_update": self.last_update_ts,
            "score_distribution": score_ranges,
            "execution_time_range_ms": {
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "avg": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
            },
        }
