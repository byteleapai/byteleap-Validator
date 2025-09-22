"""
Asynchronous Challenge Verifier
Handles background verification of committed challenges using batch processing
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import bittensor as bt
from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

from neurons.shared.config.config_manager import ConfigManager
from neurons.shared.protocols import ProofData
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

        # Extract coordinate verification settings
        self.coordinate_sample_count = config.get(
            "validation.gpu.verification.coordinate_sample_count"
        )
        self.row_verification_count = config.get(
            "validation.gpu.verification.row_verification_count"
        )

        # Service state
        self.running = False
        self._verification_task = None

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

    def _get_coordinate_sample_count(self) -> int:
        return int(
            self.config.get("validation.gpu.verification.coordinate_sample_count")
        )

    def _get_row_verification_count(self) -> int:
        return int(
            self.config.get("validation.gpu.verification.row_verification_count")
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

        bt.logging.info("⏹️ AsyncChallengeVerifier stopped")

    async def _verification_loop(self) -> None:
        """Main verification loop - processes committed challenges in batches"""
        bt.logging.debug("Starting verification loop")

        while self.running:
            try:
                # Get oldest pending challenges
                pending_challenges = self._get_oldest_verifying_challenges()

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

                else:
                    bt.logging.debug("No pending challenges found")

            except asyncio.CancelledError:
                bt.logging.debug("Verification loop cancelled")
                break
            except (ConnectionError, TimeoutError) as e:
                bt.logging.warning(f"⚠️ Network error in verification | error={e}")
            except Exception as e:
                bt.logging.error(f"❌ Unexpected error in verification loop: {e}")

            await asyncio.sleep(self._get_verification_interval())

    def _get_oldest_verifying_challenges(self) -> List[ComputeChallenge]:
        """
        Get oldest pending challenges in VERIFYING state for verification

        Returns:
            List of pending challenges ordered by creation time
        """
        try:
            with self.database_manager.get_session() as session:
                challenges = (
                    session.query(ComputeChallenge)
                    .filter(
                        ComputeChallenge.challenge_status == ChallengeStatus.VERIFYING,
                        ComputeChallenge.deleted_at.is_(None),
                    )
                    .order_by(ComputeChallenge.created_at.asc())
                    .limit(self._get_concurrent_tasks())
                    .all()
                )

                bt.logging.debug(f"Challenges pending | count={len(challenges)}")
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

            # Create mock proof data from challenge
            # The actual verification logic will be extracted from ProofProcessor
            success, verification_details = await self._perform_challenge_verification(
                challenge
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

                    # Remove cached proof for this hotkey after processing
                    try:
                        self.proof_cache.remove_proof(db_challenge.hotkey)
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
                            self.proof_cache.remove_proof(db_challenge.hotkey)
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

    async def _perform_challenge_verification(
        self, challenge: ComputeChallenge
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform the actual challenge verification logic

        Args:
            challenge: Challenge to verify

        Returns:
            Tuple of (success, verification_details)
        """
        challenge_type = challenge.challenge_type

        cached_proof = self.proof_cache.get_proof(challenge.hotkey)
        if not cached_proof:
            bt.logging.error(
                f"No cached proof data found for challenge {challenge.challenge_id}, hotkey={challenge.hotkey[:8]}..."
            )
            return False, {"error": "No cached proof data"}

        if cached_proof.get("challenge_id") != challenge.challenge_id:
            bt.logging.error(
                f"Cached proof challenge_id mismatch: expected={challenge.challenge_id}, cached={cached_proof.get('challenge_id')}"
            )
            return False, {"error": "Challenge ID mismatch"}

        proof_data = cached_proof.get("proofs", {})
        if not proof_data:
            bt.logging.error(
                f"No proof data in cache for challenge {challenge.challenge_id}"
            )
            return False, {"error": "No proof data in cache"}

        if challenge_type == "cpu_matrix":
            return await self._verify_cpu_challenge(challenge, proof_data)
        elif challenge_type == "gpu_matrix":
            return await self._verify_gpu_challenge(challenge, proof_data)
        else:
            bt.logging.error(f"❌ Unknown challenge type | type={challenge_type}")
            return False, {"error": f"Unknown challenge type: {challenge_type}"}

    async def _verify_cpu_challenge(
        self, challenge: ComputeChallenge, proof_data: Dict
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify CPU matrix challenge"""
        bt.logging.debug(f"CPU challenge verify | id={challenge.challenge_id}")

        try:

            cpu_proof = proof_data.get("-1")
            if not cpu_proof:
                bt.logging.error(
                    f"No CPU proof found (UUID '-1') for challenge {challenge.challenge_id}"
                )
                return False, {"error": "No CPU proof found"}

            challenge_data = challenge.challenge_data
            seed = bytes.fromhex(challenge_data["seed"])
            matrix_size = challenge_data["matrix_size"]
            iterations = challenge_data.get("iterations", 1)

            trusted_rows_to_check = challenge.verification_targets or []
            trusted_rows = [
                item[0] for item in trusted_rows_to_check if item[1] is None
            ]

            if not trusted_rows:
                bt.logging.error(
                    f"No trusted rows for CPU verification of challenge {challenge.challenge_id}"
                )
                return False, {"error": "No trusted rows"}

            commitment_merkle_root = challenge.merkle_commitments or {}
            expected_merkle_root = commitment_merkle_root.get("-1")
            if not expected_merkle_root:
                bt.logging.error(
                    f"No commitment found for CPU challenge {challenge.challenge_id}"
                )
                return False, {"error": "No commitment found"}

            # Extract merkle root if it's a dict
            if isinstance(expected_merkle_root, dict):
                expected_merkle_root = expected_merkle_root.get("merkle_root")

            # Verify using CPU verification logic
            verification_result = await self._verify_cpu_matrix(
                seed=seed,
                matrix_size=matrix_size,
                proof_data=cpu_proof,
                expected_merkle_root=expected_merkle_root,
                trusted_rows=trusted_rows,
                iterations=iterations,
            )

            # Create detailed verification statistics
            verification_details = {
                "total_data_points": 0,  # CPU doesn't verify coordinates
                "total_rows": len(trusted_rows),
                "successful_verifications": 1 if verification_result else 0,
                "total_proofs": 1,
                "success_count": 1 if verification_result else 0,
                "notes": f"Processed 0 coordinates, {len(trusted_rows)} rows, verified {1 if verification_result else 0}/1 proofs",
            }

            return verification_result, verification_details

        except Exception as e:
            bt.logging.error(
                f"CPU verification error for challenge {challenge.challenge_id}: {e}"
            )
            return False, {"error": f"CPU verification error: {str(e)}"}

    async def _verify_gpu_challenge(
        self, challenge: ComputeChallenge, proof_data: Dict
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify GPU matrix challenge"""
        bt.logging.debug(f"GPU challenge verify | id={challenge.challenge_id}")

        try:
            # Get challenge parameters
            challenge_data = challenge.challenge_data
            seed = challenge_data["seed"]
            matrix_size = challenge_data["matrix_size"]
            iterations = challenge_data.get("iterations", 1)

            # Get trusted verification targets
            trusted_rows_to_check = challenge.verification_targets or []
            bt.logging.debug(
                f"Retrieved {len(trusted_rows_to_check)} verification targets for GPU challenge {challenge.challenge_id}"
            )
            if trusted_rows_to_check:
                bt.logging.debug(
                    f"First few targets: {trusted_rows_to_check[:5]}, last few: {trusted_rows_to_check[-5:]}"
                )

            # Get commitment data
            commitment_merkle_root = challenge.merkle_commitments or {}

            successful_verifications = 0
            # Validation: Ensure proof_data is well-formed
            if not isinstance(proof_data, dict):
                bt.logging.error(
                    f"🚨 SECURITY: Invalid proof_data structure for challenge {challenge.challenge_id}"
                )
                return False

            # Count GPU entries, excluding CPU "-1"
            total_gpus = len([uuid for uuid in proof_data.keys() if uuid != "-1"])

            if total_gpus == 0:
                bt.logging.warning(
                    f"No valid GPU proof data found for challenge {challenge.challenge_id}"
                )
                return False

            # Verify each GPU proof
            for gpu_uuid, gpu_proof in proof_data.items():
                if gpu_uuid == "-1":  # Skip CPU proofs in GPU challenges
                    continue

                # Get commitment for this GPU
                expected_merkle_root = commitment_merkle_root.get(gpu_uuid)
                if not expected_merkle_root:
                    bt.logging.warning(
                        f"No commitment found for GPU {gpu_uuid} in challenge {challenge.challenge_id}"
                    )
                    continue

                # Extract merkle root if it's a dict
                if isinstance(expected_merkle_root, dict):
                    expected_merkle_root = expected_merkle_root.get("merkle_root")

                # Extract trusted coordinates and rows for this GPU
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

                # Verify this GPU
                gpu_result = await self._verify_gpu_matrix(
                    challenge=challenge,
                    seed=seed,
                    gpu_uuid=gpu_uuid,
                    matrix_size=matrix_size,
                    proof_data=gpu_proof,
                    expected_merkle_root=expected_merkle_root,
                    trusted_coords=trusted_coords,
                    trusted_rows=trusted_rows,
                    iterations=iterations,
                )

                if gpu_result:
                    successful_verifications += 1

            # GPU challenges pass if at least one GPU verifies successfully
            is_success = successful_verifications > 0

            # Log consolidated GPU verification results
            bt.logging.debug(
                f"GPU verification completed: {successful_verifications}/{total_gpus} GPUs passed"
            )

            # Calculate detailed statistics for GPU verification
            trusted_rows_to_check = challenge.verification_targets or []
            total_coordinates = len(
                [item for item in trusted_rows_to_check if item[1] is not None]
            )
            total_rows = len(
                [item for item in trusted_rows_to_check if item[1] is None]
            )

            verification_details = {
                "total_data_points": total_coordinates,
                "total_rows": total_rows,
                "successful_verifications": successful_verifications,
                "total_proofs": total_gpus,
                "success_count": successful_verifications,
                "notes": f"Processed {total_coordinates} coordinates, {total_rows} rows, verified {successful_verifications}/{total_gpus} proofs",
            }

            return is_success, verification_details

        except Exception as e:
            bt.logging.error(
                f"GPU verification error for challenge {challenge.challenge_id}: {e}"
            )
            return False, {"error": f"GPU verification error: {str(e)}"}

    async def _verify_cpu_matrix(
        self,
        seed: bytes,
        matrix_size: int,
        proof_data: Dict,
        expected_merkle_root: str,
        trusted_rows: List[int],
        iterations: int = 1,
    ) -> bool:
        """CPU matrix verification with full row computation"""
        try:
            # Extract row hashes and merkle proofs
            row_hashes = proof_data.get("row_hashes", [])
            merkle_proofs = proof_data.get("merkle_proofs", [])

            if not row_hashes or not merkle_proofs:
                bt.logging.error(
                    "❌ CPU proof missing data | missing=row_hashes_or_merkle_proofs"
                )
                return False

            if not trusted_rows:
                bt.logging.error("❌ No trusted rows for CPU verification")
                return False

            bt.logging.debug(
                f"CPU verification: {len(trusted_rows)} rows, {len(row_hashes)} hashes"
            )

            # Step 1: Verify row hashes against re-computed hashes
            expected_hashes = await self._compute_cpu_matrix_rows(
                seed, matrix_size, trusted_rows, iterations
            )

            if len(row_hashes) != len(expected_hashes):
                bt.logging.error(
                    f"Expected {len(expected_hashes)} row hashes, got {len(row_hashes)}"
                )
                return False

            if expected_hashes != row_hashes:
                bt.logging.error("❌ Row hash mismatch")
                return False

            # Step 2: Verify Merkle proofs
            from neurons.shared.utils.merkle_tree import verify_row_proofs

            merkle_valid, merkle_error = verify_row_proofs(
                row_indices=trusted_rows,
                row_hashes=row_hashes,
                merkle_proofs=merkle_proofs,
                expected_merkle_root=expected_merkle_root,
            )

            if not merkle_valid:
                bt.logging.error(
                    f"CPU Merkle proof verification failed: {merkle_error}"
                )
                return False

            bt.logging.debug(
                "✅ CPU verification passed: row hashes and merkle proofs verified"
            )
            return True

        except Exception as e:
            bt.logging.error(f"❌ CPU verification error | error={e}")
            return False

    async def _verify_gpu_matrix(
        self,
        challenge: Any,  # ComputeChallenge object with metadata
        seed: str,
        gpu_uuid: str,
        matrix_size: int,
        proof_data: Dict,
        expected_merkle_root: str,
        trusted_coords: List[List[int]],
        trusted_rows: List[int],
        iterations: int = 1,
    ) -> bool:
        """
        GPU matrix verification with three-step process
        """
        try:
            # Check if we have required proof data
            coordinate_values = proof_data.get("coordinate_values", [])
            row_hashes = proof_data.get("row_hashes", [])
            merkle_proofs = proof_data.get("merkle_proofs", [])

            # Separate coordinate and row verification with proper data slicing
            verified_rows = set()
            gpu_seed_str = f"{seed}|{gpu_uuid}"
            seed_hash = self._transform_gpu_seed(gpu_seed_str)

            # Coordinate verification
            coord_success = False
            if trusted_coords and coordinate_values:
                coord_values = coordinate_values[: len(trusted_coords)]
                coord_valid = await self._verify_spot_check_coordinates(
                    seed_hash, matrix_size, trusted_coords, coord_values, iterations
                )
                if not coord_valid:
                    bt.logging.debug(
                        f"❌ Coordinate verification failed for GPU {gpu_uuid}"
                    )
                    return False
                coord_success = True

            # Row verification
            row_success = False
            if trusted_rows and coordinate_values:

                # Row data starts after coordinate data
                row_data_start = len(trusted_coords)

                import secrets

                crypto_random = secrets.SystemRandom()
                row_sample_rate = self.config.get_range(
                    "validation.gpu.verification.row_sample_rate", 0.0, 1.0, float
                )
                sample_count = max(1, int(matrix_size * row_sample_rate))
                shared_sampling_columns = crypto_random.sample(
                    range(matrix_size), min(sample_count, matrix_size)
                )
                shared_sampling_columns.sort()

                for i, row_idx in enumerate(trusted_rows):
                    row_start = row_data_start + i * matrix_size
                    row_end = row_start + matrix_size

                    if row_end > len(coordinate_values):
                        bt.logging.error(
                            f"❌ Row data out of bounds: need {row_end}, have {len(coordinate_values)}"
                        )
                        return False

                    row_values = coordinate_values[row_start:row_end]

                    row_valid = await self._verify_full_row_with_sampling(
                        seed_hash,
                        matrix_size,
                        row_idx,
                        row_values,
                        iterations,
                        shared_sampling_columns,
                    )
                    if row_valid:
                        verified_rows.add(row_idx)

                # Check if enough rows passed
                if len(verified_rows) == 0:
                    bt.logging.debug(
                        f"❌ No rows passed verification for GPU {gpu_uuid}"
                    )
                    return False
                row_success = True
            elif not trusted_coords and not trusted_rows and not coordinate_values:
                pass  # No verification requested
            else:
                bt.logging.error(
                    f"❌ Coordinate data mismatch for GPU {gpu_uuid}: coords={len(trusted_coords) if trusted_coords else 0}, rows={len(trusted_rows) if trusted_rows else 0}, values={len(coordinate_values) if coordinate_values else 0}"
                )
                return False
            merkle_success = False
            if trusted_rows and row_hashes and merkle_proofs:

                if len(row_hashes) != len(trusted_rows) or len(merkle_proofs) != len(
                    trusted_rows
                ):
                    bt.logging.error(
                        f"❌ Row count mismatch for GPU {gpu_uuid}: "
                        f"expected {len(trusted_rows)} rows, got {len(row_hashes)} hashes, {len(merkle_proofs)} proofs"
                    )
                    return False

                computed_row_hashes = []
                for i, row_idx in enumerate(trusted_rows):
                    if row_idx not in verified_rows:
                        bt.logging.error(
                            f"❌ Row {row_idx} did not pass coordinate verification but has hash/proof"
                        )
                        return False

                    # Worker-provided row data verified by sampling
                    row_start = len(trusted_coords) + i * matrix_size
                    row_end = row_start + matrix_size
                    row_data = coordinate_values[row_start:row_end]

                    computed_hash = self._compute_row_hash_from_data(row_data)
                    computed_row_hashes.append(computed_hash)

                    provided_hash = row_hashes[i]
                    if computed_hash != provided_hash:
                        bt.logging.error(
                            f"❌ Row {row_idx} hash mismatch: computed {computed_hash[:16]}... vs provided {provided_hash[:16]}..."
                        )
                        return False

                merkle_success = await self._verify_row_merkle_proofs(
                    trusted_rows,
                    computed_row_hashes,
                    merkle_proofs,
                    expected_merkle_root,
                )

                if not merkle_success:
                    bt.logging.debug(
                        f"❌ Row Merkle proof verification failed for GPU {gpu_uuid}"
                    )
                    return False
            elif not trusted_rows and not row_hashes and not merkle_proofs:
                # No row verification requested - this is valid when row_verification_count = 0
                pass
            elif trusted_rows:
                bt.logging.warning(
                    f"⚠️ Trusted rows specified but missing row_hashes or merkle_proofs for GPU {gpu_uuid}"
                )
                return False

            # Special case: If both coordinate_sample_count = 0 and row_verification_count = 0, trust the result
            if not trusted_coords and not trusted_rows:
                bt.logging.info(
                    f"✅ GPU {gpu_uuid} verification bypassed - both coordinate_sample_count and row_verification_count are 0 (trust mode)"
                )
                return True

            # Log consolidated verification results with counts
            results = []
            if trusted_coords:
                results.append(
                    f"{len(trusted_coords)} coords ({'pass' if coord_success else 'fail'})"
                )
            if trusted_rows:
                results.append(
                    f"{len(trusted_rows)} rows ({'pass' if row_success else 'fail'})"
                )
                if row_hashes and merkle_proofs:
                    results.append(
                        f"{len(merkle_proofs)} merkles ({'pass' if merkle_success else 'fail'})"
                    )
            bt.logging.debug(
                f"GPU {gpu_uuid} verification | results={', '.join(results)}"
            )
            return True

        except Exception as e:
            bt.logging.error(f"❌ GPU verification error | gpu={gpu_uuid} error={e}")
            return False

    async def _verify_coordinates_batch(
        self,
        challenge: Any,
        seed: str,
        gpu_uuid: str,
        matrix_size: int,
        trusted_coords: List[List[int]],
        coordinate_values: List[float],
        iterations: int,
        proof_data: Dict,
    ) -> Tuple[bool, Set[int]]:
        """
        Verify coordinates with mixed spot checks and full row data

        Args:
            challenge: ComputeChallenge object with metadata
            seed: Challenge seed string
            gpu_uuid: GPU UUID
            matrix_size: Matrix size
            trusted_coords: Mixed list of [row,col] and [row,None] requests
            coordinate_values: Data layout [spot_check_values...][row1_full_data...][row2_full_data...]
            iterations: Matrix multiplication iterations
            proof_data: Dict containing row hashes and merkle proofs

        Returns:
            Tuple of (verification_success, verified_rows_set)
        """
        try:
            if not trusted_coords or not coordinate_values:
                bt.logging.warning(
                    "Missing verification data - trusted_coords or coordinate_values empty"
                )
                return False, set()

            gpu_seed_str = f"{seed}|{gpu_uuid}"
            seed_hash = self._transform_gpu_seed(gpu_seed_str)

            spot_check_coords = []
            row_requests = []

            for coord in trusted_coords:
                if len(coord) == 2 and coord[1] is None:
                    # Row request format: [row, None]
                    row_requests.append(coord[0])
                else:
                    # Coordinate format: [row, col]
                    spot_check_coords.append(coord)

            spot_check_count = len(spot_check_coords)
            row_data_count = len(row_requests) * matrix_size
            expected_total = spot_check_count + row_data_count

            if len(coordinate_values) != expected_total:
                bt.logging.error(
                    f"❌ Proof data mismatch: received {len(coordinate_values)} values, expected {expected_total} "
                    f"({spot_check_count} spot checks + {len(row_requests)} rows × {matrix_size} = {expected_total})"
                )
                return False, set()

            bt.logging.debug(
                f"✅ Proof data layout verified: {spot_check_count} spot checks + {len(row_requests)} full rows"
            )

            verified_rows = set()

            if spot_check_coords:
                spot_values = coordinate_values[:spot_check_count]
                spot_success = await self._verify_spot_check_coordinates(
                    seed_hash, matrix_size, spot_check_coords, spot_values, iterations
                )
                if not spot_success:
                    bt.logging.debug("Spot check coordinate verification failed")
                    return False, set()
                bt.logging.debug(
                    f"✅ Spot check verification passed ({len(spot_check_coords)} coords)"
                )

            if row_requests:
                import secrets

                crypto_random = secrets.SystemRandom()
                row_sample_rate = self.config.get_range(
                    "validation.gpu.verification.row_sample_rate", 0.0, 1.0, float
                )
                sample_count = max(1, int(matrix_size * row_sample_rate))

                shared_sampling_columns = crypto_random.sample(
                    range(matrix_size), min(sample_count, matrix_size)
                )
                shared_sampling_columns.sort()

                bt.logging.debug(
                    f"Generated {len(shared_sampling_columns)} shared sampling columns for matrix caching"
                )

                row_data_start = spot_check_count
                for i, row_idx in enumerate(row_requests):
                    row_start = row_data_start + i * matrix_size
                    row_end = row_start + matrix_size
                    row_values = coordinate_values[row_start:row_end]

                    row_success = await self._verify_full_row_with_sampling(
                        seed_hash,
                        matrix_size,
                        row_idx,
                        row_values,
                        iterations,
                        shared_sampling_columns,
                    )

                    if row_success:
                        verified_rows.add(row_idx)

                if len(verified_rows) == 0:
                    bt.logging.error("❌ No rows passed verification")
                    return False, set()

                success_rate = len(verified_rows) / len(row_requests)
                bt.logging.debug(
                    f"Row verification: {len(verified_rows)}/{len(row_requests)} rows passed ({success_rate:.1%})"
                )

            return True, verified_rows

        except Exception as e:
            bt.logging.error(f"Coordinate verification error: {e}")
            return False, set()

    async def _verify_row_hashes_with_sampling(
        self,
        seed: str,
        gpu_uuid: str,
        matrix_size: int,
        trusted_rows: List[int],
        trusted_coords: List[List[int]],
        coordinate_values: List[float],
        row_hashes: List[str],
        iterations: int,
    ) -> tuple[bool, List[int], List[str]]:
        """
        Row hash verification with coordinate sampling and selective trust

        Only trust sampling rows where coordinate verification passed 95% success rate
        """
        try:
            if not trusted_rows or not row_hashes or not trusted_coords:
                return False, [], []

            # Transform seed for GPU
            gpu_seed_str = f"{seed}|{gpu_uuid}"
            seed_hash = self._transform_gpu_seed(gpu_seed_str)

            # Group sampling coordinates by row
            sampling_coords_by_row = self._group_sampling_coords_by_row(trusted_coords)

            verified_row_indices = []
            verified_row_hashes = []

            # Check each trusted row
            for row_idx, row_id in enumerate(trusted_rows):
                if row_id not in sampling_coords_by_row:
                    bt.logging.debug(
                        f"No sampling coordinates for row {row_id}, skipping"
                    )
                    continue

                # Get sampling coordinates for this row
                row_sampling_coords = sampling_coords_by_row[row_id]

                # Row sampling requires 95% coordinate verification success
                row_coord_success = await self._verify_row_sampling_coordinates(
                    seed_hash,
                    matrix_size,
                    row_id,
                    row_sampling_coords,
                    coordinate_values,
                    iterations,
                )

                if row_coord_success:
                    # Row passes verification threshold
                    verified_row_indices.append(row_idx)
                    verified_row_hashes.append(row_hashes[row_idx])

            # Mandatory requirement: at least one sampling row must be trusted
            if not verified_row_indices:
                bt.logging.error(
                    "No sampling rows could be trusted - verification failed"
                )
                return False, [], []

            success_rate = len(verified_row_indices) / len(trusted_rows)
            bt.logging.info(
                f"Row trust analysis: {len(verified_row_indices)}/{len(trusted_rows)} rows trusted ({success_rate:.1%})"
            )

            return True, verified_row_indices, verified_row_hashes

        except Exception as e:
            bt.logging.error(f"Row hash verification error: {e}")
            return False, [], []

    def _transform_gpu_seed(self, gpu_seed_str: str) -> int:
        """Transform GPU seed string to hash format matching GPU implementation"""
        import hashlib

        seed_hash = hashlib.sha256(gpu_seed_str.encode()).digest()
        transformed_seed = int.from_bytes(seed_hash[:8], byteorder="little")
        return transformed_seed

    def _analyze_verification_pattern(
        self, challenge: Any, trusted_coords: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Analyze rows_to_check pattern to determine optimal verification caching strategy

        Parse rows_to_check to extract:
        - Coordinate verification: [row, col] format
        - Row verification: [row, null] format
        - Analyze which rows/columns appear frequently to decide caching strategy
        """
        try:
            rows_to_check = getattr(challenge, "verification_targets", [])
            if not rows_to_check:
                bt.logging.debug("No verification_targets found")
                return {
                    "coordinate_checks": [],
                    "row_checks": [],
                    "cache_rows": [],
                    "cache_cols": [],
                    "row_columns": {},
                    "row_cache_threshold": 0,
                    "col_cache_threshold": 0,
                    "reason": "no_verification_pattern",
                }

            # Parse verification patterns
            coordinate_checks = []  # [row, col] coordinate verification
            row_checks = []  # [row, null] row verification

            for item in rows_to_check:
                if len(item) >= 2:
                    row, col = item[0], item[1]
                    if col is None:
                        row_checks.append(row)  # Row hash verification
                    else:
                        coordinate_checks.append([row, col])  # Coordinate verification

            # Count occurrences for each row and column
            row_columns = {}  # {row: [col1, col2, ...]}
            col_rows = {}  # {col: [row1, row2, ...]}

            for row, col in coordinate_checks:
                if row not in row_columns:
                    row_columns[row] = []
                row_columns[row].append(col)

                if col not in col_rows:
                    col_rows[col] = []
                col_rows[col].append(row)

            # Analyze caching value
            total_coords = len(coordinate_checks)
            if total_coords == 0:
                bt.logging.debug("No coordinate checks found")
                return {
                    "coordinate_checks": [],
                    "row_checks": row_checks,
                    "cache_rows": [],
                    "cache_cols": [],
                    "row_columns": {},
                    "row_cache_threshold": 0,
                    "col_cache_threshold": 0,
                    "reason": "no_coordinates",
                }

            # Determine caching thresholds
            matrix_size = challenge.challenge_data["matrix_size"]

            # Row sample rate determines verification intensity
            row_sample_rate = self.config.get_range(
                "validation.gpu.verification.row_sample_rate", 0.0, 1.0, float
            )

            # Row threshold: based on configured sample rate
            row_cache_threshold = max(10, int(matrix_size * row_sample_rate))
            # Column threshold: each fixed column is shared by ~row_verification_count rows
            col_cache_threshold = max(5, int(self._get_row_verification_count() * 0.8))

            # Determine which rows/columns should be cached with separate thresholds
            cache_rows = [
                row
                for row, cols in row_columns.items()
                if len(cols) >= row_cache_threshold
            ]
            cache_cols = [
                col
                for col, rows in col_rows.items()
                if len(rows) >= col_cache_threshold
            ]

            # Detailed caching statistics for debugging
            row_coord_counts = {row: len(cols) for row, cols in row_columns.items()}
            col_coord_counts = {col: len(rows) for col, rows in col_rows.items()}

            analysis = {
                "coordinate_checks": coordinate_checks,
                "row_checks": row_checks,
                "cache_rows": cache_rows,
                "cache_cols": cache_cols,
                "row_columns": row_columns,  # Track which columns each row needs
                "row_cache_threshold": row_cache_threshold,
                "col_cache_threshold": col_cache_threshold,
                "reason": f"row_threshold={row_cache_threshold}, col_threshold={col_cache_threshold}, cached_rows={len(cache_rows)}, cached_cols={len(cache_cols)}",
            }

            return analysis

        except Exception as e:
            bt.logging.warning(f"Failed to analyze verification pattern: {e}")
            return {
                "coordinate_checks": [],
                "row_checks": [],
                "cache_rows": [],
                "cache_cols": [],
                "row_columns": {},
                "row_cache_threshold": 0,
                "col_cache_threshold": 0,
                "reason": f"analysis_error: {e}",
            }

    def _group_sampling_coords_by_row(
        self, trusted_coords: List[List[int]]
    ) -> Dict[int, List[List[int]]]:
        """Group sampling coordinates by their row ID"""
        coords_by_row = {}
        for coord in trusted_coords:
            if len(coord) >= 2:
                row_id = coord[0]
                if row_id not in coords_by_row:
                    coords_by_row[row_id] = []
                coords_by_row[row_id].append(coord)
        return coords_by_row

    async def _verify_random_coordinates(
        self,
        seed_hash: int,
        matrix_size: int,
        random_coords: List[List[int]],
        coordinate_values: List[float],
        iterations: int,
    ) -> bool:
        """Verify random coordinates individually"""
        if not random_coords:
            return True

        bt.logging.debug(
            f"Verifying {len(random_coords)} random coordinates individually"
        )

        if len(coordinate_values) != len(random_coords):
            bt.logging.error(
                f"Coordinate values count mismatch: {len(coordinate_values)} values vs {len(random_coords)} coordinates"
            )
            return False

        success_count = 0
        for i, coord in enumerate(random_coords):
            row, col = coord[0], coord[1]

            # Compute expected value
            expected_value = self._compute_matrix_element_gemm(
                seed_hash, matrix_size, row, col, iterations
            )
            claimed_value = coordinate_values[i]

            # Check with tolerance
            if self._values_match_with_tolerance(expected_value, claimed_value):
                success_count += 1

        success_rate = success_count / len(random_coords) if random_coords else 1.0
        return success_rate >= self._get_success_rate_threshold()

    async def _verify_coordinates(
        self,
        seed_hash: int,
        matrix_size: int,
        trusted_coords: List[List[int]],
        coordinate_values: List[float],
        iterations: int,
        cache_analysis: Dict[str, Any],
    ) -> Tuple[bool, Set[int]]:
        """
        Coordinate verification with caching

        Uses cache when coordinates benefit from it, otherwise computes directly
        """
        try:
            if not trusted_coords or not coordinate_values:
                bt.logging.warning(
                    "Missing verification data - trusted_coords or coordinate_values empty"
                )
                return False, set()

            if len(coordinate_values) != len(trusted_coords):
                bt.logging.error(
                    f"Coordinate values count mismatch: {len(coordinate_values)} values vs {len(trusted_coords)} coordinates"
                )
                return False, set()

            cache_rows = cache_analysis.get("cache_rows", [])
            cache_cols = cache_analysis.get("cache_cols", [])
            row_columns = cache_analysis.get("row_columns", {})

            bt.logging.debug(
                f"Coordinate verification: {len(trusted_coords)} coords, "
                f"caching {len(cache_rows)} rows + {len(cache_cols)} cols"
            )

            # Generate caches only for rows/columns meeting threshold
            a_matrix_cache = {}
            for row in cache_rows:
                a_matrix_cache[row] = self._compute_matrix_a_row(
                    seed_hash, matrix_size, row
                )

            b_matrix_cache = {}
            for col in cache_cols:
                b_matrix_cache[col] = self._compute_matrix_b_column(
                    seed_hash, matrix_size, col
                )

            # Verify coordinates with cache usage
            success_count = 0
            row_success_count = {}  # Track success per row

            for i, coord in enumerate(trusted_coords):
                row, col = coord[0], coord[1]

                # Use cache if available, otherwise direct computation
                if row in a_matrix_cache and col in b_matrix_cache:
                    a_row = a_matrix_cache[row]
                    b_col = b_matrix_cache[col]
                    expected_value = (
                        sum(a_row[k] * b_col[k] for k in range(matrix_size))
                        * iterations
                    )
                else:
                    # Direct computation
                    expected_value = self._compute_matrix_element_gemm(
                        seed_hash, matrix_size, row, col, iterations
                    )

                claimed_value = coordinate_values[i]

                if self._values_match_with_tolerance(expected_value, claimed_value):
                    success_count += 1

                    # Track success count per row
                    if row not in row_success_count:
                        row_success_count[row] = 0
                    row_success_count[row] += 1

            # Determine which rows are fully verified through coordinate sampling
            verified_rows = set()
            for row, success_count_for_row in row_success_count.items():
                total_coords_for_row = len(row_columns.get(row, []))
                if (
                    success_count_for_row == total_coords_for_row
                    and total_coords_for_row > 0
                ):
                    verified_rows.add(row)

            success_rate = (
                success_count / len(trusted_coords) if trusted_coords else 1.0
            )
            bt.logging.debug(
                f"Coordinate verification: {success_count}/{len(trusted_coords)} coords passed ({success_rate:.1%}), "
                f"{len(verified_rows)} rows fully verified"
            )

            return success_rate >= self._get_success_rate_threshold(), verified_rows

        except Exception as e:
            bt.logging.error(f"Coordinate verification error: {e}")
            return False, set()

    async def _verify_row_merkle_proofs(
        self,
        trusted_rows: List[int],
        row_hashes: List[str],
        merkle_proofs: List[Dict],
        expected_merkle_root: str,
    ) -> bool:
        """
        Verify row Merkle proofs for [row, None] format entries

        This verifies that miner-provided row_hashes can be proven against
        the committed merkle_root through the provided merkle_proofs.

        Args:
            trusted_rows: List of row indices to verify
            row_hashes: List of row hashes provided by miner
            merkle_proofs: List of Merkle proofs for each row
            expected_merkle_root: Committed Merkle root from Phase 1

        Returns:
            True if all row Merkle proofs verify successfully
        """
        try:
            from neurons.shared.utils.merkle_tree import verify_row_proofs

            bt.logging.debug(
                f"Verifying {len(trusted_rows)} row Merkle proofs against root: {expected_merkle_root[:16]}..."
            )

            # Use the existing Merkle tree verification utility
            is_valid, error_message = verify_row_proofs(
                row_indices=trusted_rows,
                row_hashes=row_hashes,
                merkle_proofs=merkle_proofs,
                expected_merkle_root=expected_merkle_root,
            )

            if not is_valid:
                bt.logging.debug(
                    f"❌ Row Merkle proof verification failed: {error_message}"
                )
                return False

            return True

        except Exception as e:
            bt.logging.error(f"Error in row Merkle proof verification: {e}")
            return False

    async def _verify_spot_check_coordinates(
        self,
        seed_hash: int,
        matrix_size: int,
        spot_check_coords: List[List[int]],
        spot_values: List[float],
        iterations: int,
    ) -> bool:
        """
        Verify random coordinate spot checks

        Args:
            seed_hash: Transformed GPU seed
            matrix_size: Matrix size
            spot_check_coords: List of [row, col] coordinates to verify
            spot_values: Corresponding values for each coordinate
            iterations: Matrix multiplication iterations
        """
        if not spot_check_coords or not spot_values:
            return True

        if len(spot_check_coords) != len(spot_values):
            bt.logging.error(
                f"Spot check data mismatch: {len(spot_check_coords)} coords vs {len(spot_values)} values"
            )
            return False

        success_count = 0
        for i, coord in enumerate(spot_check_coords):
            row, col = coord[0], coord[1]
            expected_value = self._compute_matrix_element_gemm(
                seed_hash, matrix_size, row, col, iterations
            )
            claimed_value = spot_values[i]

            if self._values_match_with_tolerance(expected_value, claimed_value):
                success_count += 1

        success_rate = success_count / len(spot_check_coords)
        bt.logging.debug(
            f"Spot check success rate: {success_count}/{len(spot_check_coords)} ({success_rate:.1%})"
        )
        return success_rate >= self._get_success_rate_threshold()

    async def _verify_full_row_with_sampling(
        self,
        seed_hash: int,
        matrix_size: int,
        row_idx: int,
        row_values: List[float],
        iterations: int,
        shared_sampling_columns: List[int] = None,
    ) -> bool:
        """
        Verify full row by sampling shared columns for matrix caching efficiency

        Args:
            seed_hash: Transformed GPU seed
            matrix_size: Matrix size
            row_idx: Row index to verify
            row_values: All values in the row (matrix_size elements)
            iterations: Matrix multiplication iterations
            shared_sampling_columns: Same columns used for all rows (e.g. [100, 250, 1024])
        """
        if len(row_values) != matrix_size:
            bt.logging.error(
                f"Row {row_idx} data size mismatch: {len(row_values)} vs {matrix_size}"
            )
            return False

        sampling_columns = shared_sampling_columns
        if not sampling_columns:
            bt.logging.error("No sampling columns provided for row verification")
            return False

        success_count = 0
        for col_idx in sampling_columns:
            expected_value = self._compute_matrix_element_gemm(
                seed_hash, matrix_size, row_idx, col_idx, iterations
            )
            claimed_value = row_values[col_idx]

            if self._values_match_with_tolerance(expected_value, claimed_value):
                success_count += 1

        success_rate = (
            success_count / len(sampling_columns) if sampling_columns else 1.0
        )
        bt.logging.debug(
            f"Row {row_idx} sampling: {success_count}/{len(sampling_columns)} coords passed ({success_rate:.1%})"
        )
        return success_rate >= self._get_success_rate_threshold()

    def _compute_row_hash_from_data(self, row_data: List[float]) -> str:
        """
        Compute FNV-1a hash from provided row data - matches CUDA implementation

        Args:
            row_data: List of float values for the row

        Returns:
            FNV-1a hash as 16-character lowercase hex string
        """
        import struct

        # FNV-1a hash algorithm - matches CUDA implementation
        hash_val = 0xCBF29CE484222325  # FNV offset basis
        fnv_prime = 0x100000001B3  # FNV prime

        for val in row_data:
            # Float to FP16 bit representation for hashing
            fp16_bytes = struct.pack("<e", float(val))  # IEEE 754 half precision
            element_bits = struct.unpack("<H", fp16_bytes)[0]  # Read as uint16

            # FNV-1a algorithm: XOR then multiply
            hash_val ^= element_bits
            hash_val *= fnv_prime
            hash_val &= 0xFFFFFFFFFFFFFFFF  # Keep as 64-bit

        # CUDA-compatible 16-character hex format
        return f"{hash_val:016x}"

    async def _verify_sampling_coordinates_cached(
        self,
        seed_hash: int,
        matrix_size: int,
        sampling_coords: List[List[int]],
        coordinate_values: List[float],
        iterations: int,
        cache_info: Dict[str, Any] = None,
    ) -> bool:
        """
        Verify sampling coordinates with precomputed A/B matrix caching
        """
        if not sampling_coords:
            return True

        bt.logging.debug(
            f"Verifying {len(sampling_coords)} sampling coordinates with EXPLICIT caching"
        )

        # Use explicit cache information if available, otherwise analyze patterns
        if cache_info and cache_info.get("is_precomputed", True):
            # Use pre-computed exact cache requirements
            rows_needed = set(cache_info.get("rows_needed", []))
            cols_needed = set(cache_info.get("cols_needed", []))
            efficiency_gain = cache_info.get("efficiency_gain", 1.0)

            bt.logging.debug(
                f"EXPLICIT caching: {len(rows_needed)} A-matrix rows + {len(cols_needed)} B-matrix columns "
                f"(~{efficiency_gain:.1f}x pre-calculated efficiency)"
            )
        else:
            # Analyze coordinate patterns
            rows_needed = set(coord[0] for coord in sampling_coords)
            cols_needed = set(coord[1] for coord in sampling_coords)
            estimated_old_cols = len(sampling_coords)
            efficiency_gain = (
                estimated_old_cols / len(cols_needed) if cols_needed else 1.0
            )

            bt.logging.debug(
                f"Selected caching plan: {len(rows_needed)} A-matrix rows + {len(cols_needed)} B-matrix columns "
                f"(~{efficiency_gain:.1f}x detected efficiency)"
            )

        # Precompute and cache A matrix rows
        a_matrix_cache = {}
        for row in rows_needed:
            a_matrix_cache[row] = self._compute_matrix_a_row(
                seed_hash, matrix_size, row
            )

        # Precompute and cache B matrix columns
        b_matrix_cache = {}
        for col in cols_needed:
            b_matrix_cache[col] = self._compute_matrix_b_column(
                seed_hash, matrix_size, col
            )

        # Verify coordinates using cached matrices
        if len(coordinate_values) != len(sampling_coords):
            bt.logging.error(
                f"Coordinate values count mismatch: {len(coordinate_values)} values vs {len(sampling_coords)} coordinates"
            )
            return False

        success_count = 0
        for i, coord in enumerate(sampling_coords):
            row, col = coord[0], coord[1]

            # Cached matrices enable O(1) coordinate access
            if row in a_matrix_cache and col in b_matrix_cache:
                a_row = a_matrix_cache[row]
                b_col = b_matrix_cache[col]

                # C[row,col] = sum(A[row,k] * B[k,col])
                expected_value = sum(a_row[k] * b_col[k] for k in range(matrix_size))

                # Apply iterations if needed
                if iterations > 1:
                    expected_value *= iterations

                claimed_value = coordinate_values[i]

                if self._values_match_with_tolerance(expected_value, claimed_value):
                    success_count += 1

        success_rate = success_count / len(sampling_coords) if sampling_coords else 1.0
        cache_efficiency = (
            len(rows_needed) * len(cols_needed) / (len(rows_needed) + len(cols_needed))
            if (len(rows_needed) + len(cols_needed)) > 0
            else 1.0
        )

        bt.logging.debug(
            f"Sampling verification: {success_rate:.1%} success rate, cache efficiency: {cache_efficiency:.1f}x"
        )
        return success_rate >= self._get_success_rate_threshold()

    async def _verify_row_sampling_coordinates(
        self,
        seed_hash: int,
        matrix_size: int,
        row_id: int,
        row_sampling_coords: List[List[int]],
        coordinate_values: List[float],
        iterations: int,
    ) -> bool:
        """Verify sampling coordinates for a specific row with 95% success rate"""
        if not row_sampling_coords:
            return False

        if len(coordinate_values) != len(row_sampling_coords):
            bt.logging.error(
                f"Row sampling coordinate values count mismatch: {len(coordinate_values)} values vs {len(row_sampling_coords)} coordinates"
            )
            return False

        success_count = 0
        for i, coord in enumerate(row_sampling_coords):
            row, col = coord[0], coord[1]

            # Compute expected value
            expected_value = self._compute_matrix_element_gemm(
                seed_hash, matrix_size, row, col, iterations
            )
            claimed_value = coordinate_values[i]

            if self._values_match_with_tolerance(expected_value, claimed_value):
                success_count += 1

        success_rate = (
            success_count / len(row_sampling_coords) if row_sampling_coords else 0.0
        )
        return success_rate >= self._get_success_rate_threshold()

    def _compute_matrix_element_gemm(
        self, seed_hash: int, matrix_size: int, row: int, col: int, iterations: int = 1
    ) -> float:
        """Compute single matrix element C[row,col] = sum(A[row,k] * B[k,col])"""
        if iterations > 1:
            # Multi-iteration matrix multiplication not properly implemented
            # Current system assumes iterations=1 only
            raise NotImplementedError(
                f"Multi-iteration ({iterations}) matrix multiplication not implemented correctly"
            )

        result = 0.0
        for k in range(matrix_size):
            a_element = self._generate_matrix_element(seed_hash, row, k, 0)  # Matrix A
            b_element = self._generate_matrix_element(seed_hash, k, col, 1)  # Matrix B
            result += a_element * b_element
        return result

    def _compute_matrix_a_row(
        self, seed_hash: int, matrix_size: int, row: int
    ) -> List[float]:
        """Precompute entire A matrix row for caching"""
        return [
            self._generate_matrix_element(seed_hash, row, k, 0)
            for k in range(matrix_size)
        ]

    def _compute_matrix_b_column(
        self, seed_hash: int, matrix_size: int, col: int
    ) -> List[float]:
        """Precompute entire B matrix column for caching"""
        return [
            self._generate_matrix_element(seed_hash, k, col, 1)
            for k in range(matrix_size)
        ]

    def _generate_matrix_element(
        self, seed_hash: int, row: int, col: int, matrix_type: int
    ) -> float:
        """Generate matrix element matching GPU algorithm"""
        element_id = seed_hash ^ (row << 32) ^ (col << 16) ^ matrix_type
        element_id = element_id & 0xFFFFFFFFFFFFFFFF

        # GPU cutlass_gemm.cu compatible 32-bit hash
        hash_val = (element_id ^ (element_id >> 32)) & 0xFFFFFFFF
        hash_val = (hash_val * 0x9E3779B9 + 0x85EBCA6B) & 0xFFFFFFFF
        hash_val = hash_val ^ (hash_val >> 16)
        hash_val = (hash_val * 0x85EBCA6B) & 0xFFFFFFFF

        # Convert to [-1, 1] range
        normalized = (hash_val & 0xFFFF) / 32768.0 - 1.0
        return normalized

    def _values_match_with_tolerance(self, expected: float, claimed: float) -> bool:
        """Check if values match within tolerance (FP16/FP64 precision)"""
        abs_diff = abs(expected - claimed)
        rel_diff = abs_diff / (abs(expected) + 1e-10)

        return (
            abs_diff <= self._get_abs_tolerance()
            or rel_diff <= self._get_rel_tolerance()
        )

    async def _compute_cpu_matrix_rows(
        self, seed: bytes, matrix_size: int, row_indices: List[int], iterations: int = 1
    ) -> List[str]:
        """
        Validator independently computes CPU matrix rows for verification
        Supports multiple iterations matching worker execution
        """
        try:
            import hashlib

            import numpy as np

            from neurons.shared.challenges.cpu_matrix_challenge import \
                CPUMatrixChallenge

            # Generate matrices with trusted parameters
            matrix_a, matrix_b = CPUMatrixChallenge._generate_matrices_from_seed(
                seed, matrix_size
            )

            # Compute the full result matrix using the same iteration logic as worker
            if iterations > 1:
                result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))
                for _ in range(iterations - 1):
                    result = np.dot(result, matrix_b.astype(np.int64))
            else:
                result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))

            # Compute expected hash for each requested row from the final result
            expected_hashes = []
            for row_idx in row_indices:
                computed_row = result[row_idx]
                # Match worker hashing format: SHA-256 hex truncated to 16 chars
                expected_hash = hashlib.sha256(computed_row.tobytes()).hexdigest()[:16]
                expected_hashes.append(expected_hash)

            bt.logging.debug(
                f"Computed {len(expected_hashes)} CPU matrix row hashes (iterations={iterations})"
            )
            return expected_hashes

        except Exception as e:
            bt.logging.error(f"CPU matrix computation failed: {e}")
            raise RuntimeError(f"Failed to compute expected CPU results: {str(e)}")

    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get verification service statistics

        Returns:
            Dictionary with service statistics
        """
        return {
            "running": self.running,
            "concurrent_tasks": self.concurrent_tasks,
            "verification_interval": self.verification_interval,
            "coordinate_sample_count": self.coordinate_sample_count,
            "row_verification_count": self.row_verification_count,
            "success_rate_threshold": self.success_rate_threshold,
            "abs_tolerance": self.abs_tolerance,
            "rel_tolerance": self.rel_tolerance,
            "random_coordinate_verification_enabled": self.coordinate_sample_count > 0,
        }

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

        # Validate coordinate settings
        if self.coordinate_sample_count < 0:
            issues.append(
                "coordinate_sample_count must be >= 0 (0 disables random verification)"
            )

        # Allow 0 for row-only-disabled mode; enforce the pairwise constraint below
        if self.row_verification_count < 0:
            issues.append("row_verification_count must be >= 0")

        # Validate tolerances
        if self.abs_tolerance <= 0:
            issues.append("abs_tolerance must be positive")

        if self.rel_tolerance <= 0:
            issues.append("rel_tolerance must be positive")

        # Validate success rate
        if not (0.5 <= self.success_rate_threshold <= 1.0):
            issues.append("success_rate_threshold must be between 0.5 and 1.0")

        # Check if both coordinate and row verification are disabled
        if self.coordinate_sample_count == 0 and self.row_verification_count == 0:
            issues.append(
                "coordinate_sample_count and row_verification_count cannot both be 0"
            )

        return issues

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the verification service

        Returns:
            Health status dictionary
        """
        health_status = {
            "healthy": True,
            "issues": [],
            "service_running": self.running,
            "verification_task_active": self._verification_task is not None
            and not self._verification_task.done(),
        }

        # Validate configuration
        config_issues = self.validate_configuration()
        if config_issues:
            health_status["healthy"] = False
            health_status["issues"].extend(
                [f"Config: {issue}" for issue in config_issues]
            )

        # Check database connectivity
        try:
            with self.database_manager.get_session() as session:
                # Simple query to test connectivity
                pending_challenges = self._get_oldest_verifying_challenges()
                health_status["database_accessible"] = True
                health_status["pending_challenges_pending"] = len(pending_challenges)
        except Exception as e:
            health_status["healthy"] = False
            health_status["issues"].append(f"Database: {str(e)}")
            health_status["database_accessible"] = False

        # Check verification task status
        if self.running and (
            not self._verification_task or self._verification_task.done()
        ):
            health_status["healthy"] = False
            health_status["issues"].append(
                "Verification task not running despite service being started"
            )

        return health_status
