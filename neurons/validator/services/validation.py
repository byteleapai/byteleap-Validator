"""
Validator Validation Logic Service
Responsible for verifying miner's computing power legitimacy and computational capabilities
"""

import asyncio
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from neurons.shared.challenges.cpu_matrix_challenge import CPUMatrixChallenge
from neurons.shared.challenges.gpu_matrix_challenge import GPUMatrixChallenge
from neurons.shared.crypto import CryptoManager
from neurons.shared.protocols import ComputeChallenge
from neurons.shared.utils.error_handler import ErrorHandler, ValidationError
from neurons.validator.challenge_status import ChallengeStatus
from neurons.validator.models.database import (
    ComputeChallenge,
    DatabaseManager,
    MinerInfo,
    WorkerInfo,
)
from neurons.validator.services.worker_performance_ranker import WorkerPerformanceRanker


class MinerValidationService:
    """
    Miner validation service for computational capability verification

    Responsibilities:
    - Generate and distribute CPU matrix challenges to miners
    - Verify challenge responses and calculate performance scores
    - Track worker performance for weight calculation
    - Manage challenge lifecycle and anti-replay protection
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        subtensor: bt.subtensor,
        metagraph: bt.metagraph,
        config,
        wallet: Optional[bt.wallet] = None,
    ):
        """
        Initialize validation service

        Args:
            database_manager: Database manager instance
            subtensor: Bittensor subtensor instance
            metagraph: Bittensor metagraph instance
            config: Complete validation configuration
            wallet: Validator wallet for challenge signing

        Raises:
            ValueError: If configuration values are invalid
            KeyError: If required configuration keys are missing
        """
        self.db_manager = database_manager
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.config = config

        # Initialize configuration validator
        # Validate required configuration keys
        self.netuid = config.get_positive_number("netuid", int)
        self.challenge_interval = config.get_positive_number(
            "validation.challenge_interval", int
        )
        self.challenge_timeout = config.get_positive_number(
            "validation.challenge_timeout", int
        )
        self.participation_rate_threshold = config.get_range(
            "validation.participation_rate_threshold", 0.1, 1.0, float
        )

        # Challenge configuration
        self.challenge_type = config.get_non_empty_string("validation.challenge_type")

        # CPU-specific configuration - all required
        self.matrix_size = config.get("validation.cpu.matrix_size")
        self.size_variance = config.get("validation.cpu.size_variance")

        # GPU-specific configuration - all required
        self.gpu_matrix_size = config.get("validation.gpu.matrix_size")
        self.gpu_size_variance = config.get("validation.gpu.size_variance")
        self.gpu_iterations = config.get("validation.gpu.iterations")
        self.gpu_enable_normalization = config.get(
            "validation.gpu.enable_normalization"
        )
        self.gpu_mode = config.get("validation.gpu.mode")

        # Challenge management components
        self.validator_hotkey = wallet.hotkey.ss58_address if wallet else ""

        # CPU verification configuration
        self.cpu_row_verification_count = config.get(
            "validation.cpu.verification.row_verification_count"
        )
        self.cpu_row_verification_count_variance = config.get(
            "validation.cpu.verification.row_verification_count_variance"
        )

        # GPU verification configuration
        self.coordinate_sample_count = config.get(
            "validation.gpu.verification.coordinate_sample_count"
        )
        self.coordinate_sample_count_variance = config.get(
            "validation.gpu.verification.coordinate_sample_count_variance"
        )
        self.gpu_row_verification_count = config.get(
            "validation.gpu.verification.row_verification_count"
        )
        self.gpu_row_verification_count_variance = config.get(
            "validation.gpu.verification.row_verification_count_variance"
        )
        self.row_sample_rate = config.get("validation.gpu.verification.row_sample_rate")
        self.abs_tolerance = config.get("validation.gpu.verification.abs_tolerance")
        self.rel_tolerance = config.get("validation.gpu.verification.rel_tolerance")
        self.success_rate_threshold = config.get(
            "validation.gpu.verification.success_rate_threshold"
        )

        self.verification_config = {
            "cpu_row_verification_count": self.cpu_row_verification_count,
            "cpu_row_verification_count_variance": self.cpu_row_verification_count_variance,
            "coordinate_sample_count": self.coordinate_sample_count,
            "coordinate_sample_count_variance": self.coordinate_sample_count_variance,
            "row_verification_count": self.gpu_row_verification_count,
            "row_verification_count_variance": self.gpu_row_verification_count_variance,
            "row_sample_rate": self.row_sample_rate,
            "abs_tolerance": self.abs_tolerance,
            "rel_tolerance": self.rel_tolerance,
            "success_rate_threshold": self.success_rate_threshold,
        }
        self.performance_ranker = WorkerPerformanceRanker(
            database_manager, self.challenge_interval, self.participation_rate_threshold
        )

        # Runtime status
        self.is_running = False
        self._validation_task: Optional[asyncio.Task] = None
        self._current_round_id = None
        self._startup_cleanup_done = False  # Flag to track startup cleanup completion

        # Anti-replay protection
        self._used_challenges: Dict[str, float] = {}

        bt.logging.info("🚀 Validation service initialized")

    async def start(self) -> None:
        """
        Start validation service and begin challenge cycles

        Raises:
            RuntimeError: If service is already running
        """
        if self.is_running:
            raise RuntimeError("Validation service is already running")

        self.is_running = True
        self._validation_task = asyncio.create_task(self._validation_loop())
        bt.logging.info("🚀 Validation service started")

    async def stop(self) -> None:
        """Stop validation service and cleanup resources"""
        if not self.is_running:
            return

        self.is_running = False
        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass

        bt.logging.info("⏹️ Validation service stopped")

    async def _validation_loop(self) -> None:
        """Main validation loop for continuous challenge distribution"""
        while self.is_running:
            try:
                # Startup cleanup without issuing new challenges
                if not self._startup_cleanup_done:
                    bt.logging.info("📋 Performing startup cleanup of expired data")
                    self._cleanup_expired_challenges()
                    self._startup_cleanup_done = True
                    bt.logging.info("✅ Startup cleanup completed, service ready")
                else:
                    # Normal operation: validate miners and issue challenges
                    await self._validate_online_miners()

                    # Clean up expired challenges
                    self._cleanup_expired_challenges()

                # Wait for next validation cycle
                await asyncio.sleep(self.challenge_interval)

            except Exception as e:
                ErrorHandler.log_error(
                    "validation_loop",
                    e,
                    context={"service_running": self.is_running},
                    include_traceback=True,
                )
                await asyncio.sleep(10)

    async def _validate_online_miners(self) -> None:
        """Validate all currently online miners with active workers"""
        with self.db_manager.get_session() as session:
            online_miners = self.db_manager.get_online_miners(session)

            bt.logging.info(f"🔄 Validation cycle | online_miners={len(online_miners)}")

            for miner in online_miners:
                try:
                    # Check if this miner needs validation
                    if self._should_validate_miner(miner):
                        await self._create_challenge_for_miner(miner)

                except Exception as e:
                    bt.logging.error(
                        f"❌ Miner validation error | hotkey={miner.hotkey} error={e}"
                    )

    def _should_validate_miner(self, miner: MinerInfo) -> bool:
        """
        Check if miner requires new validation challenges

        Args:
            miner: Miner information record

        Returns:
            True if miner needs validation based on time intervals
        """
        # Miners without challenge history need immediate validation
        if miner.last_challenge_time is None:
            return True

        # Check time since last validation
        time_since_last_challenge = datetime.utcnow() - miner.last_challenge_time
        return time_since_last_challenge.total_seconds() >= self.challenge_interval

    async def _create_challenge_for_miner(self, miner: MinerInfo) -> Optional[int]:
        """
        Create CPU matrix challenges for eligible miner workers

        Generates batch challenges for all unleased workers with cpu_matrix capability.
        Only workers with lease_score = 0.0 receive challenges.

        Args:
            miner: Miner information record

        Returns:
            Number of challenges created, None if creation failed
        """
        try:
            # Generate round ID and timestamp for batch challenges
            round_id = f"round_{int(time.time())}"
            challenge_timestamp = time.time()

            with self.db_manager.get_session() as session:
                # Get active workers for this miner
                workers = (
                    session.query(WorkerInfo)
                    .filter(
                        WorkerInfo.hotkey == miner.hotkey,
                        WorkerInfo.deleted_at.is_(None),
                        WorkerInfo.is_online == True,
                    )
                    .order_by(WorkerInfo.last_heartbeat.desc())
                    .limit(100)
                    .all()
                )

                if not workers:
                    bt.logging.warning(f"⚠️ No active workers | hotkey={miner.hotkey}")
                    return 0

                # Filter workers eligible for challenges
                eligible_workers = []
                for worker in workers:
                    # Skip leased workers
                    if worker.lease_score and worker.lease_score > 0.0:
                        bt.logging.debug(
                            f"Skip leased worker | id={worker.worker_id} lease_score={worker.lease_score}"
                        )
                        continue

                    # Check challenge capability support
                    worker_capabilities = worker.capabilities or []
                    has_cpu = "cpu_matrix" in worker_capabilities
                    has_gpu = "gpu_matrix" in worker_capabilities

                    if not (has_cpu or has_gpu):
                        bt.logging.debug(
                            f"Skip worker | id={worker.worker_id} reason=missing_capabilities has={worker_capabilities}"
                        )
                        continue

                    unsent_challenges = (
                        session.query(ComputeChallenge)
                        .filter(
                            ComputeChallenge.worker_id == worker.worker_id,
                            ComputeChallenge.challenge_status
                            == ChallengeStatus.CREATED,
                            ComputeChallenge.deleted_at.is_(None),
                        )
                        .count()
                    )

                    if unsent_challenges > 0:
                        bt.logging.debug(
                            f"Skipping worker {worker.worker_id} - "
                            f"{unsent_challenges} pending unsent challenges exist"
                        )
                        continue

                        # Filter workers based on validator's challenge_type configuration
                    worker_capabilities = worker.capabilities or []
                    if (
                        self.challenge_type == "gpu_matrix"
                        and "gpu_matrix" not in worker_capabilities
                    ):
                        bt.logging.debug(
                            f"Skipping worker {worker.worker_id} - no GPU capability for gpu_matrix mode"
                        )
                        continue
                    elif (
                        self.challenge_type == "cpu_matrix"
                        and "cpu_matrix" not in worker_capabilities
                    ):
                        bt.logging.debug(
                            f"Skipping worker {worker.worker_id} - no CPU capability for cpu_matrix mode"
                        )
                        continue

                    eligible_workers.append(worker)

                if not eligible_workers:
                    bt.logging.info(
                        f"No eligible workers found for miner {miner.hotkey} "
                        f"(required capability: {self.challenge_type})"
                    )
                    return 0

                bt.logging.info(
                    f"Creating batch challenges for {len(eligible_workers)} eligible workers "
                    f"from miner {miner.hotkey} (round: {round_id}, mode: {self.challenge_type})"
                )
                batch_challenges = []
                for worker in eligible_workers:
                    try:
                        # Generate challenge based on validator's configured challenge type
                        if self.challenge_type == "gpu_matrix":
                            challenge_data = GPUMatrixChallenge.generate_challenge(
                                matrix_size=self.gpu_matrix_size,
                                validator_hotkey=self.validator_hotkey,
                                enable_dynamic_size=self.gpu_size_variance > 0,
                                size_variance=self.gpu_size_variance,
                                iterations=self.gpu_iterations,
                                mode=self.gpu_mode,
                            )
                            challenge_data["challenge_type"] = "gpu_matrix"
                        else:  # cpu_matrix
                            cpu_iterations = self.config.get(
                                "validation.cpu.iterations"
                            )

                            challenge_data = CPUMatrixChallenge.generate_challenge(
                                matrix_size=self.matrix_size,
                                validator_hotkey=self.validator_hotkey,
                                enable_dynamic_size=self.size_variance > 0,
                                size_variance=self.size_variance,
                                iterations=cpu_iterations,
                            )
                            challenge_data["challenge_type"] = "cpu_matrix"

                        import uuid

                        challenge_data["challenge_id"] = str(uuid.uuid4())
                        challenge_data["worker_id"] = worker.worker_id
                        challenge_data["miner_hotkey"] = miner.hotkey
                        challenge_data["round_id"] = round_id

                        batch_challenges.append(
                            {"challenge_data": challenge_data, "worker": worker}
                        )

                    except Exception as e:
                        bt.logging.error(
                            f"Failed to create challenge for worker {worker.worker_id}: {e}"
                        )
                        continue
                for challenge_info in batch_challenges:
                    challenge_data = challenge_info["challenge_data"]
                    worker = challenge_info["worker"]

                    try:
                        self.db_manager.record_challenge(
                            session=session,
                            challenge_id=challenge_data["challenge_id"],
                            hotkey=miner.hotkey,
                            challenge_type=challenge_data.get(
                                "challenge_type", self.challenge_type
                            ),
                            challenge_data=challenge_data,
                            matrix_size=challenge_data["matrix_size"],
                            worker_id=worker.worker_id,
                            challenge_timeout=self.challenge_timeout,
                        )
                    except Exception as db_error:
                        bt.logging.warning(
                            f"⚠️ Record challenge failed | id={challenge_data['challenge_id']} error={db_error}"
                        )

                miner.last_challenge_time = datetime.utcnow()
                session.commit()

                bt.logging.info(
                    f"🧮 Challenges created | count={len(batch_challenges)} hotkey={miner.hotkey} batch={round_id}"
                )

                return len(batch_challenges)

        except Exception as e:
            bt.logging.error(
                f"❌ Create challenges failed | hotkey={miner.hotkey} error={e}"
            )
            return None

    def _cleanup_expired_challenges(self) -> None:
        """Clean up expired sent challenges while preserving unsent ones"""
        try:
            with self.db_manager.get_session() as session:
                expired_db_count = self.db_manager.mark_expired_sent_tasks(session)

                # Mark workers offline based on heartbeat deadlines
                offline_count = self.db_manager.mark_workers_offline_by_deadline(
                    session
                )

                if expired_db_count > 0 or offline_count > 0:
                    bt.logging.debug(
                        f"Cleanup | expired_tasks={expired_db_count} offline_workers={offline_count}"
                    )

                remaining_unsent = (
                    session.query(ComputeChallenge)
                    .filter(
                        ComputeChallenge.challenge_status == ChallengeStatus.CREATED,
                        ComputeChallenge.deleted_at.is_(None),
                    )
                    .count()
                )

                if remaining_unsent > 0:
                    bt.logging.debug(f"📦 Unsent challenges | count={remaining_unsent}")

        except Exception as e:
            bt.logging.error(f"❌ Cleanup error | error={e}")

        # Clean up old submission tracking records
        current_time = time.time()
        expired_used_challenges = []
        for challenge_id, used_timestamp in self._used_challenges.items():
            if (
                current_time - used_timestamp
                > CryptoManager.USED_CHALLENGE_CLEANUP_INTERVAL_SECONDS
            ):
                expired_used_challenges.append(challenge_id)

        for challenge_id in expired_used_challenges:
            del self._used_challenges[challenge_id]

        # Clean up worker submission tracking
        if hasattr(self, "_worker_submissions"):
            expired_worker_submissions = []
            for worker_key, used_timestamp in self._worker_submissions.items():
                if (
                    current_time - used_timestamp
                    > CryptoManager.USED_CHALLENGE_CLEANUP_INTERVAL_SECONDS
                ):
                    expired_worker_submissions.append(worker_key)

            for worker_key in expired_worker_submissions:
                del self._worker_submissions[worker_key]

            if expired_worker_submissions:
                bt.logging.debug(
                    f"📋 Cleaned {len(expired_worker_submissions)} worker submission records"
                )

        if expired_used_challenges:
            bt.logging.debug(
                f"📋 Cleaned {len(expired_used_challenges)} challenge tracking records"
            )

    def get_miner_validation_stats(self, hotkey: str) -> Dict[str, Any]:
        """
        Get miner validation statistics from recent challenge records

        Args:
            hotkey: Miner hotkey identifier

        Returns:
            Dictionary containing validation statistics and performance metrics
        """
        with self.db_manager.get_session() as session:
            miner = session.query(MinerInfo).filter(MinerInfo.hotkey == hotkey).first()

            if not miner:
                return {}

            # Calculate statistics from challenge records
            from neurons.validator.models.database import ComputeChallenge

            # Get challenges from the last 24 hours for current stats
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            challenges = (
                session.query(ComputeChallenge)
                .filter(
                    ComputeChallenge.hotkey == hotkey,
                    ComputeChallenge.created_at >= cutoff_time,
                    ComputeChallenge.computed_at.is_not(None),
                    ComputeChallenge.deleted_at.is_(None),
                )
                .all()
            )

            total_challenges = len(challenges)
            success_count = sum(1 for c in challenges if c.is_success)
            fail_count = total_challenges - success_count
            success_rate = (
                0.0 if total_challenges == 0 else success_count / total_challenges
            )

            # Calculate average computation time from recent challenges
            successful_challenges = [
                c for c in challenges if c.is_success and c.computation_time_ms
            ]
            avg_computation_time_ms = 0.0
            if successful_challenges:
                avg_computation_time_ms = sum(
                    c.computation_time_ms for c in successful_challenges
                ) / len(successful_challenges)

            return {
                "hotkey": hotkey,
                "total_challenges": total_challenges,
                "success_count": success_count,
                "fail_count": fail_count,
                "success_rate": success_rate,
                "avg_computation_time_ms": avg_computation_time_ms,
                "last_challenge_time": (
                    miner.last_challenge_time.isoformat()
                    if miner.last_challenge_time
                    else None
                ),
                "is_valid": True,  # Two-stage ranking replaces success rate validation
            }

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation service operational status"""
        with self.db_manager.get_session() as session:
            pending_challenges_count = (
                session.query(ComputeChallenge)
                .filter(
                    ComputeChallenge.challenge_status == ChallengeStatus.CREATED,
                    ComputeChallenge.deleted_at.is_(None),
                )
                .count()
            )

        return {
            "is_running": self.is_running,
            "pending_challenges": pending_challenges_count,
            "challenge_type": self.challenge_type,
            "challenge_interval": self.challenge_interval,
            "participation_rate_threshold": self.participation_rate_threshold,
            "cpu_matrix_stats": {"current_matrix_size": self.matrix_size},
            "gpu_matrix_stats": {"current_matrix_size": self.gpu_matrix_size},
        }

    def get_security_status(self) -> Dict[str, Any]:
        """Get validation security status and anti-gaming measures"""
        current_time = time.time()

        # Count active security measures from database
        with self.db_manager.get_session() as session:
            active_pending_challenges = (
                session.query(ComputeChallenge)
                .filter(
                    ComputeChallenge.challenge_status == ChallengeStatus.CREATED,
                    ComputeChallenge.deleted_at.is_(None),
                )
                .count()
            )
        active_used_challenges = len(self._used_challenges)

        # Calculate cleanup statistics
        old_used_challenges = sum(
            1
            for timestamp in self._used_challenges.values()
            if current_time - timestamp
            > CryptoManager.USED_CHALLENGE_CLEANUP_INTERVAL_SECONDS / 2
        )

        return {
            "is_running": self.is_running,
            "security_features": {
                "anti_replay_protection": True,
                "timeout_validation": True,
                "challenge_freshness_check": True,
                "worker_targeting": True,
            },
            "active_challenges": {
                "pending_challenges": active_pending_challenges,
                "used_challenges_tracked": active_used_challenges,
                "old_used_challenges": old_used_challenges,
            },
            "security_config": {
                "challenge_timeout": self.challenge_timeout,
                "used_challenge_cleanup_interval": CryptoManager.USED_CHALLENGE_CLEANUP_INTERVAL_SECONDS,
            },
        }
