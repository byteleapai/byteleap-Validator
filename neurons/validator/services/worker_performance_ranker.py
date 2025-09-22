"""
Worker Performance Ranking System
Implements global worker ranking and weighted scoring to prevent low-performance spam
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

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
    ):
        self.worker_id = worker_id
        self.hotkey = hotkey
        self.execution_time_ms = execution_time_ms
        self.lease_score = lease_score
        self.success_rate = success_rate
        self.total_attempts = total_attempts
        self.performance_score = 0.0  # Will be calculated by ranker
        self.global_rank = 0  # Position in global ranking

        # Two-stage ranking fields
        self.participation_score = 0.0  # Participation score for ranking
        self.actual_participation = 0  # Actual successful attempts
        self.baseline_expected = 0.0  # Expected baseline for participation


class WorkerPerformanceRanker:
    """
    Global worker performance ranking and scoring system

    This system implements the core challenge scoring logic:
    1. Collect all worker challenge results from current evaluation period
    2. Rank workers globally by performance (execution time)
    3. Apply weighted scoring to heavily penalize low-performance workers
    4. Aggregate worker scores to miner-level scores
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        challenge_interval: int = 180,
        participation_rate_threshold: float = 0.75,
    ):
        self.db_manager = database_manager
        self.performance_cache: Dict[str, WorkerPerformanceScore] = {}
        self.last_ranking_update = 0.0

        # Two-stage ranking configuration
        self.challenge_interval = challenge_interval
        self.participation_rate_threshold = participation_rate_threshold

    def calculate_global_worker_rankings(
        self, evaluation_window_minutes: int
    ) -> Dict[str, WorkerPerformanceScore]:
        """
        Calculate global worker performance rankings using two-stage ranking:
        1. Participation score based on min(expected_baseline, actual_participation)
        2. Average execution time within same participation tier

        Args:
            evaluation_window_minutes: Time window for collecting challenge results

        Returns:
            Dictionary mapping worker_key to WorkerPerformanceScore
        """
        bt.logging.info(
            f"Calculating two-stage worker rankings (window: {evaluation_window_minutes}m)"
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
                f"📊 Ranking baseline: {evaluation_window_minutes}min window, "
                f"{self.challenge_interval}s interval → "
                f"max {max_possible_challenges:.0f} challenges, "
                f"{self.participation_rate_threshold:.0%} baseline = {baseline_expected:.1f}"
            )

            # Get all recent challenges, considering only those that passed two-phase verification
            all_challenges = (
                session.query(ComputeChallenge)
                .filter(
                    ComputeChallenge.created_at >= cutoff_time,
                    ComputeChallenge.deleted_at.is_(None),
                    ComputeChallenge.challenge_status == "verified",
                )
                .order_by(ComputeChallenge.created_at.desc())
                .all()
            )

            if not all_challenges:
                bt.logging.warning("No recent challenges found for ranking")
                return {}

            bt.logging.debug(f"Processing {len(all_challenges)} recent challenges")
            worker_stats = {}

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
                    }

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

            # Calculate ranking metrics for each worker
            workers_for_ranking = []
            worker_lease_scores = self._get_worker_lease_scores(session)

            for worker_key, stats in worker_stats.items():
                if stats["successful_attempts"] == 0:
                    continue

                # Two-stage ranking metrics
                actual_participation = stats["successful_attempts"]
                participation_score = min(baseline_expected, actual_participation)

                # Average execution time
                average_time = sum(stats["successful_times"]) / len(
                    stats["successful_times"]
                )
                success_rate = stats["successful_attempts"] / stats["total_attempts"]
                lease_score = worker_lease_scores.get(worker_key, 0.0)

                workers_for_ranking.append(
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
                    }
                )

            # Two-stage sorting
            workers_for_ranking.sort(
                key=lambda x: (
                    -x["participation_score"],  # Participation score high to low
                    x["average_time"],  # Average time low to high
                )
            )

            # Generate final rankings and scores
            worker_scores = {}
            total_workers = len(workers_for_ranking)

            for rank, worker_data in enumerate(workers_for_ranking):
                worker_key = worker_data["worker_key"]

                worker_score = WorkerPerformanceScore(
                    worker_id=worker_data["worker_id"],
                    hotkey=worker_data["hotkey"],
                    execution_time_ms=worker_data["average_time"],
                    lease_score=worker_data["lease_score"],
                    success_rate=worker_data["success_rate"],
                    total_attempts=worker_data["total_attempts"],
                )

                # Set ranking fields
                worker_score.global_rank = rank
                worker_score.participation_score = worker_data["participation_score"]
                worker_score.actual_participation = worker_data["actual_participation"]
                worker_score.baseline_expected = worker_data["baseline_expected"]

                # Calculate final performance score based on rank
                worker_score.performance_score = (
                    self._calculate_weighted_performance_score(rank, total_workers)
                )

                worker_scores[worker_key] = worker_score

            self.performance_cache = worker_scores
            self.last_ranking_update = time.time()

            # Logging results
            self._log_ranking_summary(worker_scores, evaluation_window_minutes)

            return worker_scores

    def calculate_miner_challenge_scores(
        self, ranked_workers: Dict[str, WorkerPerformanceScore]
    ) -> Dict[str, float]:
        """
        Aggregate worker performance scores to miner-level challenge scores

        Leased workers (lease_score > 0.0) receive perfect score (1.0) without participating in challenges.
        Unleased workers receive scores based on actual challenge performance.

        Formula: sum(unleased_worker_scores) + sum(leased_worker_perfect_scores)
        More workers = higher miner score

        Args:
            ranked_workers: Global worker rankings from calculate_global_worker_rankings()

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

        with self.db_manager.get_session() as session:
            # Get leased worker counts by miner
            leased_worker_counts = self._get_leased_worker_counts_by_miner(session)

            # Ensure miners with only leased workers are included
            all_miners = set(miner_scores.keys()) | set(leased_worker_counts.keys())

            for hotkey in all_miners:
                worker_scores = miner_scores.get(hotkey, [])

                # Cap unleased contributions at 100 workers
                top_worker_scores = sorted(worker_scores, reverse=True)[:100]

                # Add perfect scores for leased workers within remaining slots
                leased_count = leased_worker_counts.get(hotkey, 0)
                available_slots = max(0, 100 - len(top_worker_scores))
                leased_workers_to_count = min(leased_count, available_slots)

                # Sum unleased performance + leased perfect scores
                unleased_score = sum(top_worker_scores)
                leased_score = leased_workers_to_count * 1.0
                total_score = unleased_score + leased_score

                miner_challenge_scores[hotkey] = total_score

                if leased_count > 0 and len(worker_scores) == 0:
                    bt.logging.debug(
                        f"Miner {hotkey}: 0 unleased + {leased_workers_to_count} leased → challenge score {total_score:.2f}"
                    )
                elif leased_count > 0:
                    bt.logging.debug(
                        f"Miner {hotkey}: {len(top_worker_scores)} unleased (score: {unleased_score:.2f}) + "
                        f"{leased_workers_to_count} leased (score: {leased_score:.2f}), total: {total_score:.2f}"
                    )

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

    def _calculate_weighted_performance_score(
        self, rank: int, total_workers: int
    ) -> float:
        """
        Calculate weighted performance score based on ranking coefficient algorithm

        Uses a continuous ranking coefficient formula to maintain score diversity
        while still heavily penalizing low-performance workers to prevent spam attacks

        Args:
            rank: Worker's rank (0 = best performer)
            total_workers: Total number of workers being ranked

        Returns:
            Performance score between 0.02 and 1.0
        """
        if total_workers <= 0:
            return 0.0

        # Calculate ranking coefficient
        ranking_coeff = rank / (total_workers - 1) if total_workers > 1 else 0.0

        # Apply exponential decay with ranking coefficient to create continuous scoring
        # Exponential decay maintains heavy penalties for low performers

        max_score = 1.0
        min_score = 0.02  # Minimal "basic income" for worst performers
        decay_factor = 3.5  # Controls how aggressively scores drop
        power = 1.8  # Controls the curve shape

        # Calculate exponential decay score
        import math

        raw_score = max_score * math.exp(-decay_factor * (ranking_coeff**power))

        # Ensure minimum score and cap at maximum
        score = max(min_score, min(max_score, raw_score))

        return score

    def _log_ranking_summary(
        self, worker_scores: Dict[str, WorkerPerformanceScore], window_minutes: int
    ):
        """Log summary of two-stage ranking results"""
        if not worker_scores:
            return

        bt.logging.debug(f"Two-stage ranking completed | workers={len(worker_scores)}")

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
                    f"  #{score.global_rank+1} {worker_key}: "
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

    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get statistics about current rankings"""
        if not self.performance_cache:
            return {
                "total_workers": 0,
                "last_update": self.last_ranking_update,
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
            "last_update": self.last_ranking_update,
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
