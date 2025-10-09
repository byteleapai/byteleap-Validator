"""
Asynchronous Challenge Verifier
Handles background verification of committed challenges using batch processing
"""

import asyncio
import concurrent.futures
import math
import multiprocessing as mp
import signal
from collections import OrderedDict

import numpy as np


def _verify_coords_worker(
    seed_hash: int,
    matrix_size: int,
    spot_check_coords: list,
    spot_values: list,
    iterations: int,
    abs_tol: float,
    rel_tol: float,
    threshold: float,
) -> bool:
    # Reduce redundant work across coordinates by caching A rows and B cols
    if not spot_check_coords or not spot_values:
        return True
    if len(spot_check_coords) != len(spot_values):
        return False

    required = math.ceil(threshold * len(spot_check_coords))
    success = 0

    unique_rows = sorted({r for r, c in spot_check_coords})
    unique_cols = sorted({c for r, c in spot_check_coords})

    a_rows = {
        r: _get_a_row_vector_cached(seed_hash, matrix_size, r) for r in unique_rows
    }
    b_cols = {
        c: _get_b_col_vector_cached(seed_hash, matrix_size, c) for c in unique_cols
    }

    for i, (row, col) in enumerate(spot_check_coords):
        expected = float(np.dot(a_rows[row], b_cols[col]))
        if iterations > 1:
            expected *= iterations
        claimed = spot_values[i]
        diff = abs(expected - claimed)
        rel = diff / (abs(expected) + 1e-10)
        if diff <= abs_tol or rel <= rel_tol:
            success += 1

        remaining = len(spot_check_coords) - (i + 1)
        # Early exit when outcome is decided
        if success >= required:
            return True
        if success + remaining < required:
            return False

    return success >= required


def _verify_row_sampling_worker(
    seed_hash: int,
    matrix_size: int,
    row_idx: int,
    coordinate_values: list,
    row_start: int,
    iterations: int,
    sampling_columns: list,
    abs_tol: float,
    rel_tol: float,
    threshold: float,
    b_cols_matrix: np.ndarray,
) -> bool:
    if not sampling_columns:
        return False

    # Batch compute expected values for sampled columns using cached B cols
    a_row = _get_a_row_vector_cached(seed_hash, matrix_size, row_idx)
    expected_vec = a_row @ b_cols_matrix
    if iterations > 1:
        expected_vec = expected_vec * float(iterations)

    claimed = np.array(
        [coordinate_values[row_start + c] for c in sampling_columns], dtype=np.float64
    )
    diff = np.abs(expected_vec - claimed)
    rel = diff / (np.abs(expected_vec) + 1e-10)
    ok = (diff <= abs_tol) | (rel <= rel_tol)
    success = int(np.count_nonzero(ok))
    required = math.ceil(threshold * len(sampling_columns))
    return success >= required


def _pool_worker_initializer() -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        # Ignore environments without SIGINT support
        pass


def _transform_gpu_seed_worker(gpu_seed_str: str) -> int:
    import hashlib

    seed_hash = hashlib.sha256(gpu_seed_str.encode()).digest()
    transformed_seed = int.from_bytes(seed_hash[:8], byteorder="little")
    return transformed_seed


from typing import Any, Dict, List, Tuple


def _compute_row_hash_from_segment_worker(
    values: List[float], start: int, length: int
) -> str:
    # GPU worker row hashing compatibility with CUDA FNV-1a on FP16
    import struct

    hash_val = 0xCBF29CE484222325
    fnv_prime = 0x100000001B3

    end = start + length
    for i in range(start, end):
        fp16_bytes = struct.pack("<e", float(values[i]))
        element_bits = struct.unpack("<H", fp16_bytes)[0]
        hash_val ^= element_bits
        hash_val = (hash_val * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    return f"{hash_val:016x}"


# Vectorized element generators with per-process LRU caches
_MAX_A_CACHE = 64
_MAX_B_CACHE = 64
_A_ROW_CACHE: "OrderedDict[Tuple[int,int,int], np.ndarray]" = OrderedDict()
_B_COL_CACHE: "OrderedDict[Tuple[int,int,int], np.ndarray]" = OrderedDict()


def _lru_get(cache: OrderedDict, key: Tuple[int, int, int]):
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _lru_set(
    cache: OrderedDict, key: Tuple[int, int, int], val: np.ndarray, max_size: int
) -> np.ndarray:
    cache[key] = val
    cache.move_to_end(key)
    if len(cache) > max_size:
        cache.popitem(last=False)
    return val


def _get_a_row_vector_cached(seed_hash: int, matrix_size: int, row: int) -> np.ndarray:
    key = (int(seed_hash), int(matrix_size), int(row))
    cached = _lru_get(_A_ROW_CACHE, key)
    if cached is not None:
        return cached
    # Generate A(row, k) for k in [0..N)
    N = int(matrix_size)
    k = np.arange(N, dtype=np.uint64)
    el = (
        np.uint64(seed_hash)
        ^ (np.uint64(row) << np.uint64(32))
        ^ (k << np.uint64(16))
        ^ np.uint64(0)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    h = (el ^ (el >> np.uint64(32))) & np.uint64(0xFFFFFFFF)
    h = (h * np.uint64(0x9E3779B9) + np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    h = (h * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    arr = (h & np.uint64(0xFFFF)).astype(np.float64) / 32768.0 - 1.0
    return _lru_set(_A_ROW_CACHE, key, arr, _MAX_A_CACHE)


def _get_b_col_vector_cached(seed_hash: int, matrix_size: int, col: int) -> np.ndarray:
    key = (int(seed_hash), int(matrix_size), int(col))
    cached = _lru_get(_B_COL_CACHE, key)
    if cached is not None:
        return cached
    # Generate B(k, col) for k in [0..N)
    N = int(matrix_size)
    k = np.arange(N, dtype=np.uint64)
    el = (
        np.uint64(seed_hash)
        ^ (k << np.uint64(32))
        ^ (np.uint64(col) << np.uint64(16))
        ^ np.uint64(1)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    h = (el ^ (el >> np.uint64(32))) & np.uint64(0xFFFFFFFF)
    h = (h * np.uint64(0x9E3779B9) + np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    h = (h * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    arr = (h & np.uint64(0xFFFF)).astype(np.float64) / 32768.0 - 1.0
    return _lru_set(_B_COL_CACHE, key, arr, _MAX_B_CACHE)


def _get_b_cols_matrix_cached(
    seed_hash: int, matrix_size: int, columns: List[int]
) -> np.ndarray:
    # Avoid large recomputation by stacking cached column vectors
    cols = [int(c) for c in columns]
    vectors = [_get_b_col_vector_cached(seed_hash, matrix_size, c) for c in cols]
    if len(vectors) == 1:
        return vectors[0].reshape(-1, 1)
    return np.column_stack(vectors)


def _compute_cpu_matrix_rows_worker(
    seed_hex: str, matrix_size: int, row_indices: List[int], iterations: int = 1
) -> List[str]:
    import hashlib

    import numpy as np

    from neurons.shared.challenges.cpu_matrix_challenge import \
        CPUMatrixChallenge

    seed = bytes.fromhex(seed_hex)
    matrix_a, matrix_b = CPUMatrixChallenge._generate_matrices_from_seed(
        seed, matrix_size
    )

    if iterations > 1:
        result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))
        for _ in range(iterations - 1):
            result = np.dot(result, matrix_b.astype(np.int64))
    else:
        result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))

    expected_hashes: List[str] = []
    for row_idx in row_indices:
        computed_row = result[row_idx]
        expected_hash = hashlib.sha256(computed_row.tobytes()).hexdigest()[:16]
        expected_hashes.append(expected_hash)

    return expected_hashes


def _verify_challenge_worker(
    challenge_payload: Dict[str, Any],
    proof_data: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    import bittensor as bt

    try:
        challenge_type = challenge_payload.get("challenge_type")

        abs_tol: float = settings["abs_tolerance"]
        rel_tol: float = settings["rel_tolerance"]
        threshold: float = settings["success_rate_threshold"]
        row_sample_rate: float = settings["row_sample_rate"]

        if challenge_type == "cpu_matrix":
            cpu_proof = proof_data.get("-1")
            if not cpu_proof:
                return False, {"error": "No CPU proof found"}

            challenge_data = challenge_payload["challenge_data"]
            seed_hex = challenge_data["seed"]
            matrix_size = challenge_data["matrix_size"]
            iterations = challenge_data.get("iterations", 1)

            trusted_rows_to_check = challenge_payload.get("verification_targets") or []
            trusted_rows = [
                item[0] for item in trusted_rows_to_check if item[1] is None
            ]
            if not trusted_rows:
                return False, {"error": "No trusted rows"}

            commitment_merkle_root = challenge_payload.get("merkle_commitments", {})
            expected_merkle_root = commitment_merkle_root.get("-1")
            if not expected_merkle_root:
                return False, {"error": "No commitment found"}
            if isinstance(expected_merkle_root, dict):
                expected_merkle_root = expected_merkle_root.get("merkle_root")

            row_hashes = cpu_proof.get("row_hashes", [])
            merkle_proofs = cpu_proof.get("merkle_proofs", [])
            if not row_hashes or not merkle_proofs:
                return False, {"error": "CPU proof missing data"}

            expected_hashes = _compute_cpu_matrix_rows_worker(
                seed_hex, matrix_size, trusted_rows, iterations
            )
            if len(row_hashes) != len(expected_hashes) or expected_hashes != row_hashes:
                return False, {"error": "Row hash mismatch"}

            from neurons.shared.utils.merkle_tree import verify_row_proofs

            merkle_valid, merkle_error = verify_row_proofs(
                row_indices=trusted_rows,
                row_hashes=row_hashes,
                merkle_proofs=merkle_proofs,
                expected_merkle_root=expected_merkle_root,
            )
            if not merkle_valid:
                return False, {"error": f"CPU Merkle proof failed: {merkle_error}"}

            total_rows = len(trusted_rows)
            total_proofs = 1
            successful = 1
            details = {
                "total_data_points": 0,
                "total_rows": total_rows,
                "successful_verifications": successful,
                "total_proofs": total_proofs,
                "success_count": successful,
                "notes": f"Processed {total_rows} rows, verified {successful}/{total_proofs} proofs",
            }
            return True, details

        elif challenge_type == "gpu_matrix":
            challenge_data = challenge_payload["challenge_data"]
            seed = challenge_data["seed"]
            matrix_size = challenge_data["matrix_size"]
            iterations = challenge_data.get("iterations", 1)

            trusted_rows_to_check = challenge_payload.get("verification_targets") or []
            commitment_merkle_root = challenge_payload.get("merkle_commitments", {})

            # Count GPUs present in proofs
            total_gpus = len([uuid for uuid in proof_data.keys() if uuid != "-1"])
            if total_gpus == 0:
                return False, {"error": "No GPU proofs"}

            successful_verifications = 0

            for gpu_uuid, gpu_proof in proof_data.items():
                if gpu_uuid == "-1":
                    continue

                expected_merkle_root = commitment_merkle_root.get(gpu_uuid)
                if not expected_merkle_root:
                    continue
                if isinstance(expected_merkle_root, dict):
                    expected_merkle_root = expected_merkle_root.get("merkle_root")

                trusted_coords = [
                    [item[0], item[1]]
                    for item in trusted_rows_to_check
                    if len(item) >= 2 and item[1] is not None
                ]
                trusted_rows = [
                    item[0]
                    for item in trusted_rows_to_check
                    if len(item) >= 2 and item[1] is None
                ]

                coordinate_values = gpu_proof.get("coordinate_values", [])
                row_hashes = gpu_proof.get("row_hashes", [])
                merkle_proofs = gpu_proof.get("merkle_proofs", [])

                # Transform seed for this GPU
                gpu_seed_str = f"{seed}|{gpu_uuid}"
                seed_hash = _transform_gpu_seed_worker(gpu_seed_str)

                # 1) Coordinate verification
                if trusted_coords:
                    if not coordinate_values or len(coordinate_values) < len(
                        trusted_coords
                    ):
                        continue
                    coord_values = coordinate_values[: len(trusted_coords)]
                    coord_ok = _verify_coords_worker(
                        seed_hash,
                        matrix_size,
                        trusted_coords,
                        coord_values,
                        iterations,
                        abs_tol,
                        rel_tol,
                        threshold,
                    )
                    if not coord_ok:
                        continue

                # 2) Row verification by sampling
                verified_rows = set()
                if trusted_rows:
                    required_len = len(trusted_coords) + len(trusted_rows) * matrix_size
                    if not coordinate_values or len(coordinate_values) < required_len:
                        continue

                    import secrets

                    crypto_random = secrets.SystemRandom()
                    sample_count = max(1, int(matrix_size * float(row_sample_rate)))
                    shared_sampling_columns = crypto_random.sample(
                        range(matrix_size), min(sample_count, matrix_size)
                    )
                    shared_sampling_columns.sort()
                    b_cols_matrix = _get_b_cols_matrix_cached(
                        seed_hash, matrix_size, shared_sampling_columns
                    )

                    row_data_start = len(trusted_coords)
                    for i, row_idx in enumerate(trusted_rows):
                        row_start = row_data_start + i * matrix_size
                        row_ok = _verify_row_sampling_worker(
                            seed_hash,
                            matrix_size,
                            row_idx,
                            coordinate_values,
                            row_start,
                            iterations,
                            shared_sampling_columns,
                            abs_tol,
                            rel_tol,
                            threshold,
                            b_cols_matrix,
                        )
                        if row_ok:
                            verified_rows.add(row_idx)

                    if len(verified_rows) == 0:
                        continue

                # 3) Merkle verification of rows
                if trusted_rows and row_hashes and merkle_proofs:
                    if len(row_hashes) != len(trusted_rows) or len(
                        merkle_proofs
                    ) != len(trusted_rows):
                        continue

                    computed_row_hashes: List[str] = []
                    for i, row_idx in enumerate(trusted_rows):
                        if row_idx not in verified_rows:
                            # Only trust rows that passed sampling
                            continue

                        row_start = len(trusted_coords) + i * matrix_size
                        computed_hash = _compute_row_hash_from_segment_worker(
                            coordinate_values, row_start, matrix_size
                        )
                        provided_hash = row_hashes[i]
                        if computed_hash != provided_hash:
                            computed_row_hashes = []
                            break
                        computed_row_hashes.append(computed_hash)

                    if not computed_row_hashes or len(computed_row_hashes) != len(
                        trusted_rows
                    ):
                        continue

                    from neurons.shared.utils.merkle_tree import \
                        verify_row_proofs

                    merkle_ok, _ = verify_row_proofs(
                        row_indices=trusted_rows,
                        row_hashes=computed_row_hashes,
                        merkle_proofs=merkle_proofs,
                        expected_merkle_root=expected_merkle_root,
                    )
                    if not merkle_ok:
                        continue

                successful_verifications += 1

            is_success = successful_verifications > 0

            total_coordinates = len(
                [item for item in trusted_rows_to_check if item[1] is not None]
            )
            total_rows = len(
                [item for item in trusted_rows_to_check if item[1] is None]
            )

            details = {
                "total_data_points": total_coordinates,
                "total_rows": total_rows,
                "successful_verifications": successful_verifications,
                "total_proofs": total_gpus,
                "success_count": successful_verifications,
                "notes": f"Processed {total_coordinates} coordinates, {total_rows} rows, verified {successful_verifications}/{total_gpus} proofs",
            }
            return is_success, details

        else:
            return False, {"error": f"Unknown challenge type: {challenge_type}"}

    except Exception as e:
        bt.logging.error(f"Worker error verifying challenge: {e}")
        return False, {"error": str(e)}


import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import bittensor as bt
from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.challenge_status import ChallengeStatus
from neurons.validator.models.database import ComputeChallenge, DatabaseManager
from neurons.validator.services.proof_cache import LRUProofCache


class AsyncChallengeVerifier:
    """
    Asynchronous verification service for background challenge verification

    Features:
    - Batch processing with CPU-1 concurrency
    - Simple state management (committed → verified/failed)
    - Background verification loop independent of request processing
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        config: ConfigManager,
        proof_cache: LRUProofCache,
    ):
        """
        Initialize async verification service

        Args:
            database_manager: Database manager instance
            config: Configuration manager instance
            proof_cache: Proof cache instance
        """
        self.database_manager = database_manager
        self.config = config
        self.proof_cache = proof_cache

        # Use CPU cores - 1 to prevent system saturation when configured as -1
        verification_concurrent = config.get("validation.verification_concurrent")
        if verification_concurrent == -1:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                bt.logging.warning(
                    "Could not detect CPU count, using single verification thread"
                )
                self.concurrent_tasks = 1
            else:
                self.concurrent_tasks = max(1, cpu_count - 1)
        else:
            self.concurrent_tasks = max(1, verification_concurrent)

        # Verification polling interval
        self.verification_interval = config.get_positive_number(
            "validation.verification_interval", int
        )

        self.abs_tolerance = config.get_range(
            "validation.gpu.verification.abs_tolerance", 0.0, 1.0, float
        )
        self.rel_tolerance = config.get_range(
            "validation.gpu.verification.rel_tolerance", 0.0, 1.0, float
        )
        self.success_rate_threshold = config.get_range(
            "validation.gpu.verification.success_rate_threshold", 0.0, 1.0, float
        )

        # Extract required verification settings only

        # Service state
        self.running = False
        self._verification_task = None
        try:
            ctx = mp.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.concurrent_tasks,
                mp_context=ctx,
                initializer=_pool_worker_initializer,
            )
        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize ProcessPoolExecutor | error={e}")
            raise

        bt.logging.info(
            f"AsyncChallengeVerifier initialized: "
            f"concurrent_tasks={self.concurrent_tasks}, "
            f"verification_interval={self.verification_interval}s"
        )

    def _get_abs_tolerance(self) -> float:
        return self.config.get_range(
            "validation.gpu.verification.abs_tolerance", 0.0, 1.0, float
        )

    def _get_rel_tolerance(self) -> float:
        return self.config.get_range(
            "validation.gpu.verification.rel_tolerance", 0.0, 1.0, float
        )

    def _get_success_rate_threshold(self) -> float:
        return self.config.get_range(
            "validation.gpu.verification.success_rate_threshold", 0.0, 1.0, float
        )

    def _get_verification_interval(self) -> int:
        return self.config.get_positive_number("validation.verification_interval", int)

    def _get_concurrent_tasks(self) -> int:
        configured = self.config.get("validation.verification_concurrent")
        if configured == -1:
            cpu_count = os.cpu_count() or 1
            return max(1, int(cpu_count) - 1)
        return max(1, int(configured))

    async def start(self) -> None:
        """Start the async verification service"""
        if self.running:
            bt.logging.warning("⚠️ AsyncChallengeVerifier already running")
            return

        self.running = True
        self._verification_task = asyncio.create_task(self._verification_loop())
        bt.logging.info("🔄 AsyncChallengeVerifier started")

    async def stop(self) -> None:
        """Stop the async verification service"""
        if not self.running:
            return

        self.running = False
        if self._verification_task:
            self._verification_task.cancel()
            try:
                await self._verification_task
            except asyncio.CancelledError:
                pass

        # Best-effort executor shutdown with forced terminate of workers
        try:
            if getattr(self, "_executor", None):
                self._shutdown_executor_forceful()
        except Exception as e:
            bt.logging.debug(f"Executor shutdown error ignored | error={e}")

        bt.logging.info("⏹️ AsyncChallengeVerifier stopped")

    def _shutdown_executor_forceful(self) -> None:
        executor = getattr(self, "_executor", None)
        if not executor:
            return
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        try:
            processes = getattr(executor, "_processes", {}) or {}
            for p in list(processes.values()):
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    continue
            # Optional short join to avoid lingering children
            for p in list(processes.values()):
                try:
                    if p.is_alive():
                        p.join(timeout=0.3)
                except Exception:
                    continue
        except Exception:
            # Private API best-effort only
            pass

    async def _verification_loop(self) -> None:
        """Main verification loop - processes committed challenges in batches"""
        bt.logging.debug("Starting verification loop")

        while self.running:
            try:
                # Get oldest pending challenges
                pending_challenges = self._get_newest_verifying_challenges()

                if pending_challenges:
                    bt.logging.debug(
                        f"Processing batch of {len(pending_challenges)} pending challenges"
                    )

                    verification_tasks = [
                        self._verify_single_challenge(challenge)
                        for challenge in pending_challenges
                    ]

                    # Wait for all verifications to complete
                    results = await asyncio.gather(
                        *verification_tasks, return_exceptions=True
                    )

                    # Log results summary
                    success_count = sum(
                        1 for r in results if isinstance(r, tuple) and r[0] is True
                    )
                    error_count = sum(1 for r in results if isinstance(r, Exception))
                    fail_count = len(results) - success_count - error_count

                    bt.logging.info(
                        f"✅ Verification batch completed: "
                        f"success={success_count}, failed={fail_count}, errors={error_count}"
                    )

                    # If there is backlog, loop again immediately to keep CPUs busy
                    await asyncio.sleep(0)
                else:
                    bt.logging.debug("No pending challenges found")
                    # Idle backoff only when queue is empty
                    await asyncio.sleep(self._get_verification_interval())

            except asyncio.CancelledError:
                bt.logging.debug("Verification loop cancelled")
                break
            except (ConnectionError, TimeoutError) as e:
                bt.logging.warning(f"⚠️ Network error in verification | error={e}")
                await asyncio.sleep(self._get_verification_interval())
            except Exception as e:
                bt.logging.error(f"❌ Unexpected error in verification loop: {e}")
                await asyncio.sleep(self._get_verification_interval())

    def _get_newest_verifying_challenges(self) -> List[ComputeChallenge]:
        """
        Get newest VERIFYING challenges, and mark stale ones (>1h) failed

        Returns latest-first list limited by concurrency
        """
        try:
            now_utc = datetime.utcnow()
            stale_cutoff = now_utc - timedelta(hours=1)

            with self.database_manager.get_session() as session:
                # Mark stale challenges as failed
                stale_list = (
                    session.query(ComputeChallenge)
                    .filter(
                        ComputeChallenge.challenge_status.in_(
                            [
                                ChallengeStatus.VERIFYING,
                                ChallengeStatus.COMMITTED,
                            ]
                        ),
                        ComputeChallenge.deleted_at.is_(None),
                        ComputeChallenge.computed_at.isnot(None),
                        ComputeChallenge.computed_at < stale_cutoff,
                    )
                    .all()
                )

                if stale_list:
                    for db_challenge in stale_list:
                        db_challenge.challenge_status = ChallengeStatus.FAILED
                        db_challenge.verification_result = False
                        db_challenge.verified_at = now_utc
                        db_challenge.verification_time_ms = 0.0
                        db_challenge.is_success = False
                        db_challenge.verification_notes = (
                            "Timeout: proof stale (>1h since computed_at)"
                        )
                        if db_challenge.worker_id:
                            try:
                                self.database_manager.update_worker_task_statistics(
                                    session=session,
                                    hotkey=db_challenge.hotkey,
                                    worker_id=db_challenge.worker_id,
                                    is_success=False,
                                    computation_time_ms=None,
                                )
                            except Exception:
                                pass
                        try:
                            self._update_challenge_gpu_activity(
                                session, db_challenge, False
                            )
                        except Exception:
                            pass

                    try:
                        session.commit()
                    except Exception:
                        session.rollback()

                    # Best-effort cache cleanup
                    for db_challenge in stale_list:
                        try:
                            cache_key = (
                                f"{db_challenge.hotkey}:{db_challenge.worker_id}"
                            )
                            self.proof_cache.remove_proof(cache_key)
                        except Exception:
                            pass

                # Fetch latest challenges (newest first)
                challenges = (
                    session.query(ComputeChallenge)
                    .filter(
                        ComputeChallenge.challenge_status == ChallengeStatus.VERIFYING,
                        ComputeChallenge.deleted_at.is_(None),
                    )
                    .order_by(
                        ComputeChallenge.computed_at.desc().nullslast(),
                        ComputeChallenge.created_at.desc(),
                    )
                    .limit(self._get_concurrent_tasks())
                    .all()
                )

                bt.logging.debug(f"Challenges pending | fetched={len(challenges)}")
                return challenges

        except Exception as e:
            bt.logging.error(f"❌ Fetch challenges error | error={e}")
            return []

    async def _verify_single_challenge(self, challenge: ComputeChallenge) -> bool:
        """
        Verify a single challenge using existing proof processor logic

        Args:
            challenge: Challenge to verify

        Returns:
            True if verification successful, False otherwise
        """
        verification_start_time = time.time()

        try:
            bt.logging.debug(
                f"start verification | challenge_id={challenge.challenge_id}"
            )

            # Timeout guard: skip stale proofs (> 1 hour since computed_at)
            try:
                if challenge.computed_at is not None:
                    now_utc = datetime.utcnow()
                    age_seconds = (now_utc - challenge.computed_at).total_seconds()
                    if age_seconds > 3600:
                        with self.database_manager.get_session() as session:
                            db_challenge = session.get(ComputeChallenge, challenge.id)
                            if db_challenge:
                                db_challenge.challenge_status = ChallengeStatus.FAILED
                                db_challenge.verification_result = False
                                db_challenge.verified_at = now_utc
                                db_challenge.verification_time_ms = (
                                    time.time() - verification_start_time
                                ) * 1000
                                db_challenge.is_success = False
                                db_challenge.verification_notes = (
                                    "Timeout: proof stale (>1h since computed_at)"
                                )
                                # Update worker task statistics on timeout
                                if db_challenge.worker_id:
                                    try:
                                        self.database_manager.update_worker_task_statistics(
                                            session=session,
                                            hotkey=db_challenge.hotkey,
                                            worker_id=db_challenge.worker_id,
                                            is_success=False,
                                            computation_time_ms=None,
                                        )
                                    except Exception:
                                        pass
                                session.commit()
                        # Best-effort cache cleanup for this worker
                        try:
                            cache_key = f"{challenge.hotkey}:{challenge.worker_id}"
                            self.proof_cache.remove_proof(cache_key)
                        except Exception:
                            pass
                        return False, {"error": "stale_timeout"}
            except Exception:
                # Non-fatal; continue to verification path
                pass

            # Prepare per-challenge payload and offload verification
            cache_key = f"{challenge.hotkey}:{challenge.worker_id}"
            cached_proof = self.proof_cache.get_proof(cache_key)

            do_offload = True
            success = False
            verification_details: Dict[str, Any] = {}

            if not cached_proof:
                bt.logging.error(
                    f"No cached proof found for challenge {challenge.challenge_id}, key={cache_key[:12]}..."
                )
                do_offload = False
                verification_details = {
                    "error": "No cached proof data",
                    "notes": "Missing cached proof for verification",
                    "success_count": 0,
                }

            if (
                do_offload
                and cached_proof.get("challenge_id") != challenge.challenge_id
            ):
                bt.logging.error(
                    f"Cached proof challenge_id mismatch: expected={challenge.challenge_id}, cached={cached_proof.get('challenge_id')}"
                )
                do_offload = False
                verification_details = {
                    "error": "Challenge ID mismatch",
                    "notes": "Cached proof belongs to different challenge_id",
                    "success_count": 0,
                }

            proof_data = cached_proof.get("proofs", {}) if cached_proof else {}
            if do_offload and not proof_data:
                bt.logging.error(
                    f"No proof data in cache for challenge {challenge.challenge_id}"
                )
                do_offload = False
                verification_details = {
                    "error": "No proof data in cache",
                    "notes": "Cached proof record missing proofs payload",
                    "success_count": 0,
                }

            challenge_payload = {
                "id": challenge.id,
                "challenge_id": challenge.challenge_id,
                "hotkey": challenge.hotkey,
                "worker_id": challenge.worker_id,
                "challenge_type": challenge.challenge_type,
                "challenge_data": challenge.challenge_data,
                "verification_targets": challenge.verification_targets,
                "merkle_commitments": challenge.merkle_commitments,
            }

            settings = {
                "abs_tolerance": self._get_abs_tolerance(),
                "rel_tolerance": self._get_rel_tolerance(),
                "success_rate_threshold": self._get_success_rate_threshold(),
                "row_sample_rate": self.config.get_range(
                    "validation.gpu.verification.row_sample_rate", 0.0, 1.0, float
                ),
            }

            if do_offload:
                loop = asyncio.get_event_loop()
                success, verification_details = await loop.run_in_executor(
                    self._executor,
                    _verify_challenge_worker,
                    challenge_payload,
                    proof_data,
                    settings,
                )

            # Update verification results in database
            verification_time_ms = (time.time() - verification_start_time) * 1000

            with self.database_manager.get_session() as session:
                # Refresh challenge in new session
                db_challenge = session.get(ComputeChallenge, challenge.id)
                if db_challenge:
                    db_challenge.challenge_status = (
                        ChallengeStatus.VERIFIED if success else ChallengeStatus.FAILED
                    )
                    db_challenge.verification_result = success
                    db_challenge.verification_time_ms = verification_time_ms
                    db_challenge.verified_at = datetime.utcnow()
                    db_challenge.is_success = success
                    if verification_details:
                        db_challenge.success_count = verification_details.get(
                            "success_count", 0
                        )
                        db_challenge.verification_notes = verification_details.get(
                            "notes", f"Verification {'passed' if success else 'failed'}"
                        )
                    else:
                        db_challenge.success_count = 1 if success else 0
                        db_challenge.verification_notes = (
                            f"Verification {'passed' if success else 'failed'}"
                        )

                    # Update worker task statistics
                    if db_challenge.worker_id:
                        self.database_manager.update_worker_task_statistics(
                            session=session,
                            hotkey=db_challenge.hotkey,
                            worker_id=db_challenge.worker_id,
                            is_success=success,
                            computation_time_ms=db_challenge.computation_time_ms,
                        )

                    session.commit()

                    # Update GPU activity statistics for each GPU involved in the challenge
                    self._update_challenge_gpu_activity(session, db_challenge, success)

                    bt.logging.debug(
                        f"verification done | challenge_id={challenge.challenge_id} "
                        f"success={success} duration_ms={verification_time_ms:.1f}"
                    )

                    # Remove cached proof for worker after processing
                    try:
                        cache_key = f"{db_challenge.hotkey}:{db_challenge.worker_id}"
                        self.proof_cache.remove_proof(cache_key)
                    except Exception:
                        pass

            return success, verification_details

        except Exception as e:
            bt.logging.error(
                f"❌ Error verifying challenge {challenge.challenge_id}: {e}"
            )

            # Mark as failed on error
            verification_time_ms = (time.time() - verification_start_time) * 1000

            try:
                with self.database_manager.get_session() as session:
                    db_challenge = session.get(ComputeChallenge, challenge.id)
                    if db_challenge:
                        db_challenge.challenge_status = ChallengeStatus.FAILED
                        db_challenge.verification_result = False
                        db_challenge.verification_time_ms = verification_time_ms
                        db_challenge.verified_at = datetime.utcnow()
                        db_challenge.verification_notes = (
                            f"Verification error: {str(e)}"
                        )
                        db_challenge.is_success = False

                        # Update worker task statistics for failed challenge
                        if db_challenge.worker_id:
                            self.database_manager.update_worker_task_statistics(
                                session=session,
                                hotkey=db_challenge.hotkey,
                                worker_id=db_challenge.worker_id,
                                is_success=False,
                                computation_time_ms=None,  # No computation time for error cases
                            )

                        # Update GPU activity statistics for failed challenge
                        self._update_challenge_gpu_activity(
                            session, db_challenge, False
                        )

                        session.commit()

                        # Best-effort cache cleanup on failure
                        try:
                            cache_key = (
                                f"{db_challenge.hotkey}:{db_challenge.worker_id}"
                            )
                            self.proof_cache.remove_proof(cache_key)
                        except Exception:
                            pass

            except (IntegrityError, OperationalError, DatabaseError) as db_error:
                bt.logging.error(
                    f"Database error updating failed challenge status: {db_error}"
                )
                session.rollback()
            except Exception as db_error:
                bt.logging.error(
                    f"Unexpected error updating failed challenge status: {db_error}"
                )
                session.rollback()

            return False, {"error": f"Verification error: {str(e)}"}

    def _update_challenge_gpu_activity(
        self, session, db_challenge, is_successful: bool
    ) -> None:
        """Update GPU activity statistics for challenge completion"""
        if not db_challenge.merkle_commitments:
            return

        computation_time = (
            db_challenge.computation_time_ms
            if db_challenge.computation_time_ms
            else None
        )
        gpu_count = 0

        for gpu_uuid, merkle_root in db_challenge.merkle_commitments.items():
            if gpu_uuid != "-1":
                try:
                    self.database_manager.update_gpu_activity(
                        session=session,
                        gpu_uuid=gpu_uuid,
                        is_successful=is_successful,
                        computation_time_ms=computation_time,
                    )
                    gpu_count += 1
                except (IntegrityError, OperationalError, DatabaseError) as e:
                    bt.logging.warning(
                        f"Database error updating GPU activity for {gpu_uuid}: {e}"
                    )
                except Exception as e:
                    bt.logging.error(
                        f"Unexpected error updating GPU activity for {gpu_uuid}: {e}"
                    )

        if gpu_count > 0:
            pass

    def validate_configuration(self) -> List[str]:
        """
        Validate service configuration and return any issues

        Returns:
            List of configuration issues (empty if all good)
        """
        issues = []

        # Validate concurrent tasks
        if self.concurrent_tasks < 1:
            issues.append("concurrent_tasks must be at least 1")

        # Validate verification interval
        if self.verification_interval < 1:
            issues.append("verification_interval must be at least 1 second")

        # Validate tolerances
        if self.abs_tolerance <= 0:
            issues.append("abs_tolerance must be positive")

        if self.rel_tolerance <= 0:
            issues.append("rel_tolerance must be positive")

        # Validate success rate
        if not (0.5 <= self.success_rate_threshold <= 1.0):
            issues.append("success_rate_threshold must be between 0.5 and 1.0")

        return issues
