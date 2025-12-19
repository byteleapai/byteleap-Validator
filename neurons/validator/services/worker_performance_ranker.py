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
        self.performance_score = 0.0  # Calculated by ranker
        self.order_index = 0  # Position in ordered list (0-based)
        self.availability = 1.0  # Per-worker availability in [0,1]

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
        availability_window_hours: int = 169,
    ):
        self.db_manager = database_manager

        # Participation configuration
        self.challenge_interval = challenge_interval
        self.participation_rate_threshold = participation_rate_threshold
        # Availability configuration
        self.availability_window_hours = int(availability_window_hours)

        # Availability caching, key -> (availability, timestamp)
        self._availability_cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = 60.0  # seconds
        # Cache capacity controls periodic trimming; set higher for larger participation sets
        self._cache_capacity = 2000
        self._cache_keep = int(self._cache_capacity * 0.8)

        # Database dialect cache (None = not yet detected)
        self._db_dialect: Optional[str] = None

    def _is_postgres(self, session) -> bool:
        """Check if using PostgreSQL, with caching."""
        if self._db_dialect is None:
            self._db_dialect = session.bind.dialect.name or ""
        return self._db_dialect == "postgresql"

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
        t_perf_start = time.monotonic()

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
            t_ch_start = time.monotonic()
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
            bt.logging.debug(
                f"Challenges query | count={len(all_challenges)} elapsed={time.monotonic()-t_ch_start:.2f}s"
            )

            if not all_challenges:
                bt.logging.warning("No recent challenges found for scoring")
                return {}

            bt.logging.debug(f"Processing {len(all_challenges)} recent challenges")
            worker_stats = {}
            gil_release_counter = 0

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
                            if owner != challenge.hotkey:
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

                # Release GIL periodically to allow other threads to execute
                gil_release_counter += 1
                if gil_release_counter % 100 == 0:
                    time.sleep(0.05)

            # Compute per-worker availability over configured window
            t_avail_start = time.monotonic()
            worker_availability: Dict[str, float] = self._compute_worker_availability(
                session=session,
                hours=self.availability_window_hours,
                limit_to_worker_keys=set(worker_stats.keys()),
                consistency_minutes=evaluation_window_minutes,
            )
            bt.logging.debug(
                f"Availability calc | workers={len(worker_availability)} elapsed={time.monotonic()-t_avail_start:.2f}s"
            )

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

                # Multiply by per-worker availability
                key_for_avail = f"{worker_data['hotkey']}_{worker_data['worker_id']}"
                avail = float(worker_availability.get(key_for_avail, 0.0))
                avail = max(0.0, min(1.0, avail))
                worker_score.availability = avail

                worker_score.performance_score = (
                    base_score * participation_coeff * avail
                )
                worker_scores[worker_key] = worker_score

            # End of scoring pass

            # Logging results
            self._log_performance_summary(worker_scores, evaluation_window_minutes)

        return worker_scores

    def _compute_worker_availability(
        self,
        session,
        hours: int,
        limit_to_worker_keys: Optional[set] = None,
        consistency_minutes: int = 180,
    ) -> Dict[str, float]:
        """Compute per-worker availability using SQL aggregation.

        Returns: ("{hotkey}_{worker_id}") -> availability in [0,1]
        """
        from sqlalchemy import bindparam, text

        if hours <= 0:
            return {k: 1.0 for k in (limit_to_worker_keys or [])}

        # Check cache for improved performance
        now = time.time()
        cache_key_params = f"h{hours}_cm{consistency_minutes}"
        cached_results: Dict[str, float] = {}
        uncached_keys: set = set()

        if limit_to_worker_keys:
            for key in limit_to_worker_keys:
                cache_key = f"{key}_{cache_key_params}"
                if cache_key in self._availability_cache:
                    availability, timestamp = self._availability_cache[cache_key]
                    if now - timestamp < self._cache_ttl:
                        cached_results[key] = availability
                        continue
                uncached_keys.add(key)
        else:
            uncached_keys = set()

        # If all results are cached, return early
        if limit_to_worker_keys and not uncached_keys:
            return cached_results

        window_start = datetime.utcnow() - timedelta(hours=hours)
        expected_intervals = max(1, int(hours * 12))  # 5-min buckets

        # Extract hotkeys we need for filtering
        needed_hotkeys: Optional[List[str]] = None
        if limit_to_worker_keys:
            needed_hotkeys = list({k.split("_", 1)[0] for k in limit_to_worker_keys})

        is_postgres = self._is_postgres(session)

        # SQL aggregation: bucket counts and IP changes per worker
        t_sql_start = time.monotonic()
        worker_stats: Dict[str, Tuple[int, int]] = {}

        params: Dict[str, Any] = {"window_start": window_start}
        hotkey_filter = ""
        if needed_hotkeys:
            hotkey_filter = "AND hotkey IN :hotkeys"
            params["hotkeys"] = tuple(needed_hotkeys)

        bucket_expr = (
            "CAST(EXTRACT(EPOCH FROM created_at) / 300 AS INTEGER)"
            if is_postgres
            else "CAST(strftime('%s', created_at) / 300 AS INTEGER)"
        )

        # Match master branch behavior: skip NULL IPs when computing IP changes
        # but include all records for bucket counting
        sql = text(
            f"""
            WITH all_heartbeats AS (
                SELECT
                    hotkey,
                    worker_id,
                    public_ip,
                    ({bucket_expr}) AS bucket
                FROM heartbeat_records
                WHERE created_at >= :window_start
                    AND deleted_at IS NULL
                    {hotkey_filter}
            ),
            non_null_ips AS (
                SELECT
                    hotkey,
                    worker_id,
                    public_ip,
                    LAG(public_ip) OVER (PARTITION BY hotkey, worker_id ORDER BY created_at, id) AS prev_ip
                FROM heartbeat_records
                WHERE created_at >= :window_start
                    AND deleted_at IS NULL
                    AND public_ip IS NOT NULL
                    AND public_ip != ''
                    {hotkey_filter}
            ),
            ip_change_counts AS (
                SELECT
                    hotkey,
                    worker_id,
                    SUM(CASE WHEN prev_ip IS NOT NULL AND public_ip <> prev_ip THEN 1 ELSE 0 END) AS ip_changes
                FROM non_null_ips
                GROUP BY hotkey, worker_id
            )
            SELECT
                a.hotkey,
                a.worker_id,
                COUNT(DISTINCT a.bucket) AS online_buckets,
                COALESCE(ic.ip_changes, 0) AS ip_changes
            FROM all_heartbeats a
            LEFT JOIN ip_change_counts ic
                ON a.hotkey = ic.hotkey AND a.worker_id = ic.worker_id
            WHERE a.worker_id IS NOT NULL
            GROUP BY a.hotkey, a.worker_id, ic.ip_changes
        """
        )
        if needed_hotkeys:
            sql = sql.bindparams(bindparam("hotkeys", expanding=True))
        rows = session.execute(sql, params).fetchall()
        for row in rows:
            hotkey, worker_id, online_buckets, ip_changes = row
            if not hotkey or not worker_id:
                continue
            wkey = f"{hotkey}_{worker_id}"
            if limit_to_worker_keys and wkey not in limit_to_worker_keys:
                continue
            worker_stats[wkey] = (int(online_buckets), int(ip_changes))

        t_sql_elapsed = time.monotonic() - t_sql_start
        bt.logging.debug(
            f"Availability SQL | workers={len(worker_stats)} elapsed={t_sql_elapsed:.2f}s"
        )

        result: Dict[str, float] = {}

        # GPU worker valid bucket counts - computed entirely in SQL
        # Returns (hotkey, worker_id, valid_bucket_count) for GPU workers
        t_gpu_start = time.monotonic()
        gpu_worker_buckets: Dict[str, int] = {}  # wkey -> valid_bucket_count

        gpu_cutoff = datetime.utcnow() - timedelta(
            minutes=max(1, int(consistency_minutes))
        )
        gpu_params: Dict[str, Any] = {
            "window_start": window_start,
            "gpu_cutoff": gpu_cutoff,
        }
        gpu_hotkey_filter = ""
        if needed_hotkeys:
            gpu_hotkey_filter = "AND hotkey IN :hotkeys"
            gpu_params["hotkeys"] = tuple(needed_hotkeys)

        if is_postgres:
            # PostgreSQL: Use array containment operator for subset check
            gpu_sql = text(
                f"""
                WITH
                -- Step 1: Extract required GPU UUIDs per worker from recent challenges
                worker_required_gpus AS (
                    SELECT
                        hotkey,
                        worker_id,
                        array_agg(DISTINCT gpu_uuid) AS required_uuids
                    FROM (
                        SELECT
                            hotkey,
                            worker_id,
                            jsonb_object_keys(merkle_commitments::jsonb) AS gpu_uuid
                        FROM compute_challenges
                        WHERE created_at >= :gpu_cutoff
                            AND deleted_at IS NULL
                            AND verification_result = TRUE
                            AND merkle_commitments IS NOT NULL
                            {gpu_hotkey_filter}
                    ) sub
                    WHERE gpu_uuid IS NOT NULL AND gpu_uuid != '-1'
                    GROUP BY hotkey, worker_id
                ),
                -- Step 2: GPU UUIDs present per (hotkey, bucket)
                bucket_gpus AS (
                    SELECT
                        hotkey,
                        CAST(EXTRACT(EPOCH FROM created_at) / 300 AS INTEGER) AS bucket,
                        array_agg(DISTINCT gpu_uuid) AS present_uuids
                    FROM (
                        SELECT
                            hotkey,
                            created_at,
                            COALESCE(
                                elem->>'uuid',
                                elem->>'gpu_uuid',
                                elem->>'id'
                            ) AS gpu_uuid
                        FROM heartbeat_records
                        CROSS JOIN LATERAL jsonb_array_elements(gpu_utilization::jsonb) AS elem
                        WHERE created_at >= :window_start
                            AND deleted_at IS NULL
                            AND gpu_utilization IS NOT NULL
                            AND jsonb_array_length(gpu_utilization::jsonb) > 0
                            {gpu_hotkey_filter}
                    ) sub
                    WHERE gpu_uuid IS NOT NULL AND gpu_uuid != '-1' AND gpu_uuid != ''
                    GROUP BY hotkey, bucket
                )
                -- Step 3: Count valid buckets per worker (where required ⊆ present)
                SELECT
                    w.hotkey,
                    w.worker_id,
                    COUNT(DISTINCT b.bucket) AS valid_bucket_count
                FROM worker_required_gpus w
                LEFT JOIN bucket_gpus b
                    ON w.hotkey = b.hotkey
                    AND w.required_uuids <@ b.present_uuids
                GROUP BY w.hotkey, w.worker_id
            """
            )
        else:
            # SQLite: Use json_group_array and NOT EXISTS for robust subset check
            gpu_sql = text(
                f"""
                WITH
                -- Step 1: Extract required GPU UUIDs per worker from recent challenges
                worker_required_gpus AS (
                    SELECT
                        hotkey,
                        worker_id,
                        json_group_array(gpu_uuid) AS required_uuids
                    FROM (
                        SELECT DISTINCT
                            hotkey,
                            worker_id,
                            key AS gpu_uuid
                        FROM compute_challenges, json_each(merkle_commitments)
                        WHERE created_at >= :gpu_cutoff
                            AND deleted_at IS NULL
                            AND verification_result = 1
                            AND merkle_commitments IS NOT NULL
                            AND json_type(merkle_commitments) = 'object'
                            AND key IS NOT NULL
                            AND key != '-1'
                            {gpu_hotkey_filter}
                    )
                    GROUP BY hotkey, worker_id
                ),
                -- Step 2: GPU UUIDs present per (hotkey, bucket)
                bucket_gpus AS (
                    SELECT
                        hotkey,
                        bucket,
                        json_group_array(gpu_uuid) AS present_uuids
                    FROM (
                        SELECT DISTINCT
                            hotkey,
                            CAST(strftime('%s', created_at) / 300 AS INTEGER) AS bucket,
                            COALESCE(
                                json_extract(elem.value, '$.uuid'),
                                json_extract(elem.value, '$.gpu_uuid'),
                                json_extract(elem.value, '$.id')
                            ) AS gpu_uuid
                        FROM heartbeat_records
                        JOIN json_each(gpu_utilization) AS elem
                        WHERE created_at >= :window_start
                            AND deleted_at IS NULL
                            AND gpu_utilization IS NOT NULL
                            AND json_type(gpu_utilization) = 'array'
                            AND json_array_length(gpu_utilization) > 0
                            {gpu_hotkey_filter}
                    )
                    WHERE gpu_uuid IS NOT NULL AND gpu_uuid != '' AND gpu_uuid != '-1'
                    GROUP BY hotkey, bucket
                ),
                -- Step 3: Check subset using NOT EXISTS (all required must be in present)
                worker_bucket_valid AS (
                    SELECT
                        w.hotkey,
                        w.worker_id,
                        b.bucket
                    FROM worker_required_gpus w
                    JOIN bucket_gpus b ON w.hotkey = b.hotkey
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM json_each(w.required_uuids) req
                        WHERE req.value NOT IN (
                            SELECT p.value FROM json_each(b.present_uuids) p
                        )
                    )
                )
                -- Step 4: Count valid buckets per worker
                SELECT
                    hotkey,
                    worker_id,
                    COUNT(DISTINCT bucket) AS valid_bucket_count
                FROM worker_bucket_valid
                GROUP BY hotkey, worker_id
            """
            )

        if needed_hotkeys:
            gpu_sql = gpu_sql.bindparams(bindparam("hotkeys", expanding=True))

        gpu_rows = session.execute(gpu_sql, gpu_params).fetchall()
        t_gpu_query = time.monotonic() - t_gpu_start
        bt.logging.debug(
            f"GPU worker bucket query | rows={len(gpu_rows)} elapsed={t_gpu_query:.2f}s"
        )

        for hotkey, worker_id, valid_bucket_count in gpu_rows:
            if not hotkey or not worker_id:
                continue
            wkey = f"{hotkey}_{worker_id}"
            gpu_worker_buckets[wkey] = int(valid_bucket_count or 0)

        # Process each worker
        all_worker_keys = set(worker_stats.keys())
        if limit_to_worker_keys:
            all_worker_keys = all_worker_keys & limit_to_worker_keys

        for wkey in all_worker_keys:
            _, ip_changes = worker_stats.get(wkey, (0, 0))

            if wkey in gpu_worker_buckets:
                # GPU worker: use SQL-computed valid bucket count
                valid_buckets = gpu_worker_buckets[wkey]
                online_ratio = min(1.0, valid_buckets / expected_intervals)
            else:
                # CPU/no-GPU worker: use heartbeat bucket count from first SQL
                online_buckets, _ = worker_stats.get(wkey, (0, 0))
                online_ratio = min(1.0, online_buckets / expected_intervals)

            # Apply IP penalty
            if ip_changes > 0:
                penalty = 0.5**ip_changes
                if penalty < 0.1:
                    penalty = 0.0
                online_ratio *= penalty

            computed_availability = max(0.0, min(1.0, float(online_ratio)))
            result[wkey] = computed_availability

            cache_key = f"{wkey}_{cache_key_params}"
            self._availability_cache[cache_key] = (computed_availability, now)

        # Workers with no heartbeats → availability 0 if present in limit
        if limit_to_worker_keys:
            for k in limit_to_worker_keys:
                if k not in result:
                    result[k] = 0.0
                    cache_key = f"{k}_{cache_key_params}"
                    self._availability_cache[cache_key] = (0.0, now)

        # Merge cached and computed results
        final_result = {**cached_results, **result}

        # Periodic cache cleanup: purge expired first, then cap by capacity
        try:
            if self._availability_cache:
                expired = [
                    k
                    for k, (_val, ts) in self._availability_cache.items()
                    if now - ts >= self._cache_ttl
                ]
                if expired:
                    for k in expired:
                        self._availability_cache.pop(k, None)
        except Exception:
            pass

        if len(self._availability_cache) > self._cache_capacity:
            try:
                sorted_items = sorted(
                    self._availability_cache.items(),
                    key=lambda x: x[1][1],
                    reverse=True,
                )
                keep_n = max(1, int(self._cache_keep))
                self._availability_cache = dict(sorted_items[:keep_n])
            except Exception:
                self._availability_cache.clear()

        return final_result

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
