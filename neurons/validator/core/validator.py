import asyncio
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from neurons.shared.crypto import CryptoManager
from neurons.shared.protocols import (
    ChallengeProofSynapse,
    ChallengeSynapse,
)
from neurons.shared.protocols import EncryptedSynapse as BaseSynapse
from neurons.shared.protocols import (
    HeartbeatSynapse,
    SessionInitSynapse,
    TaskSynapse,
)
from neurons.validator.models.database import (
    DatabaseManager,
    MinerInfo,
    NetworkLog,
    WorkerInfo,
)
from neurons.validator.services.communication import ValidatorCommunicationService
from neurons.validator.services.processor_factory import ValidatorProcessorFactory
from neurons.validator.services.validation import MinerValidationService
from neurons.validator.services.weight_manager import WeightManager


class Validator:
    """
    Main validator controller for compute resource validation

    Responsibilities:
    - Handle incoming synapse requests from miners
    - Validate miner computational capabilities
    - Manage network weight distribution
    - Coordinate database operations for validator state
    """

    def __init__(self, config, bt_config):
        """
        Initialize validator with required components

        Args:
            config: Complete validator configuration

        Raises:
            ValueError: If config is None or missing required keys
            KeyError: If required configuration keys are missing
        """
        if config is None:
            raise ValueError("config cannot be None")

        self.config = config

        # Initialize wallet using provided bt_config (single source of truth)
        self.wallet = bt.wallet(config=bt_config)
        bt.logging.info(f"👛 Wallet | name={self.wallet.name}")
        bt.logging.info(f"🔑 Hotkey | name={self.wallet.hotkey_str}")

        # Network configuration
        self.netuid = self.config.get_positive_number("netuid", int)
        self.port = self.config.get_positive_number("port", int)

        # Initialize subtensor using provided bt_config
        try:
            self.subtensor = bt.subtensor(config=bt_config)
        except Exception as e:
            bt.logging.error(
                f"❌ Subtensor init failed | network={self.config.get_non_empty_string('subtensor.network')} | error={e}"
            )
            raise

        # Initialize metagraph
        self.metagraph = bt.metagraph(netuid=self.netuid, subtensor=self.subtensor)

        # Database manager
        database_url = self.config.get_non_empty_string("database.url")
        self.database_manager = DatabaseManager(database_url)

        # Communication system
        self.communicator = ValidatorCommunicationService(
            self.wallet, config, self.database_manager
        )
        # Service components
        self.validation_service = MinerValidationService(
            self.database_manager, self.subtensor, self.metagraph, config, self.wallet
        )
        self.weight_manager = WeightManager(
            self.database_manager, self.wallet, self.subtensor, self.metagraph, config
        )

        # Create verification config for async verification service
        self.verification_config = {
            "cpu_row_verification_count": self.validation_service.cpu_row_verification_count,
            "cpu_row_verification_count_variance": self.validation_service.cpu_row_verification_count_variance,
            "coordinate_sample_count": self.validation_service.coordinate_sample_count,
            "coordinate_sample_count_variance": self.validation_service.coordinate_sample_count_variance,
            "row_verification_count": self.validation_service.gpu_row_verification_count,
            "row_verification_count_variance": self.validation_service.gpu_row_verification_count_variance,
            "row_sample_rate": self.validation_service.row_sample_rate,
            "abs_tolerance": self.validation_service.abs_tolerance,
            "rel_tolerance": self.validation_service.rel_tolerance,
            "success_rate_threshold": self.validation_service.success_rate_threshold,
        }

        # Async verification service for background challenge verification
        from neurons.validator.services.async_challenge_verifier import (
            AsyncChallengeVerifier,
        )

        self.async_challenge_verifier = AsyncChallengeVerifier(
            self.database_manager, config
        )

        # Register communication processors
        self._register_processors()

        # Axon server
        external_ip = (
            self.config.get_optional("external_ip")
            or bt.utils.networking.get_external_ip()
        )
        self.axon = bt.axon(wallet=self.wallet, port=self.port, ip=external_ip)

        # Register forward functions
        self._setup_axon_handlers()
        # Runtime state
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        # Setup signal handlers
        self._setup_signal_handlers()

        bt.logging.info("✅ Validator initialization complete")

    def _register_processors(self) -> None:
        """Register synapse processors for communication"""
        from neurons.shared.protocols import (
            ChallengeProofSynapse,
            ChallengeSynapse,
            HeartbeatSynapse,
            TaskSynapse,
        )
        from neurons.validator.processors.commitment_processor import (
            CommitmentProcessor,
        )
        from neurons.validator.processors.proof_processor import ProofProcessor

        # Setup communication processor factory
        processor_factory = ValidatorProcessorFactory(self.communicator)

        # Register request processors
        heartbeat_processor = processor_factory.create_heartbeat_processor(
            self._process_heartbeat_request
        )
        self.communicator.register_processor(HeartbeatSynapse, heartbeat_processor)

        task_processor = processor_factory.create_task_processor(
            self._process_task_request
        )
        self.communicator.register_processor(TaskSynapse, task_processor)

        # Two-phase challenge verification processors with verification config
        commitment_processor = CommitmentProcessor(
            self.communicator, self.database_manager, self.verification_config
        )
        self.communicator.register_processor(ChallengeSynapse, commitment_processor)

        proof_processor = ProofProcessor(
            self.communicator, self.database_manager, self.verification_config
        )
        self.communicator.register_processor(ChallengeProofSynapse, proof_processor)

        bt.logging.debug("Communication processors registered")

    async def _process_heartbeat_request(self, request_data, peer_hotkey):
        """Process decrypted heartbeat request data"""
        from neurons.shared.protocols import ErrorCodes, HeartbeatResponse

        try:
            # HeartbeatData is a Pydantic model; enforce model-based parsing
            heartbeat_dict = request_data.model_dump()

            workers_processed = 0

            with self.database_manager.get_session() as session:
                # Update miner heartbeat
                self.database_manager.update_miner_heartbeat(
                    session, peer_hotkey, heartbeat_dict
                )

                # Process individual workers
                for worker_info in heartbeat_dict.get("workers", []):
                    worker_id = worker_info.get("worker_id")
                    if worker_id:
                        worker_data = worker_info
                        self.database_manager.update_worker_heartbeat(
                            session,
                            worker_id,
                            peer_hotkey,
                            worker_data,
                            heartbeat_interval_minutes=1,  # 60 second intervals
                        )

                        # Update worker hardware info if system_info is available
                        system_info = worker_data.get("system_info")
                        if system_info and isinstance(system_info, dict):
                            self.database_manager.update_worker_hardware_info(
                                session, peer_hotkey, worker_id, system_info
                            )

                            # Update GPU inventory if GPU plugin details are present
                            gpu_plugin_details = system_info.get("gpu_plugin", [])

                            if gpu_plugin_details:
                                for gpu_detail in gpu_plugin_details:
                                    gpu_uuid = gpu_detail.get("uuid") or gpu_detail.get(
                                        "gpu_uuid"
                                    )
                                    if gpu_uuid:
                                        try:
                                            # Pass complete GPU details to database
                                            self.database_manager.upsert_gpu_inventory(
                                                session=session,
                                                gpu_uuid=gpu_uuid,
                                                hotkey=peer_hotkey,
                                                worker_id=worker_id,
                                                gpu_details=gpu_detail,
                                            )
                                        except Exception as e:
                                            bt.logging.warning(
                                                f"Failed to update GPU inventory for {gpu_uuid}: {e}"
                                            )

                        workers_processed += 1

            response = HeartbeatResponse(
                error_code=ErrorCodes.SUCCESS,
                message=f"Processed heartbeat for {workers_processed} workers",
                workers_processed=workers_processed,
            )

            bt.logging.debug(
                f"✅ Heartbeat processed | peer={peer_hotkey} workers={workers_processed}"
            )
            return response.model_dump(), 0

        except Exception as e:
            bt.logging.error(
                f"❌ Heartbeat processing failed | peer={peer_hotkey} | error={e}"
            )
            response = HeartbeatResponse(
                error_code=ErrorCodes.HEARTBEAT_PROCESSING_FAILED,
                message=f"Processing failed: {str(e)}",
                workers_processed=0,
            )
            return response.model_dump(), ErrorCodes.HEARTBEAT_PROCESSING_FAILED

    async def _process_task_request(self, request_data, peer_hotkey):
        """Process decrypted task request data"""
        from neurons.shared.protocols import TaskResponse

        try:
            # Check if there are any pending challenges for this miner
            with self.database_manager.get_session() as session:
                from datetime import datetime, timedelta

                from neurons.validator.challenge_status import ChallengeStatus
                from neurons.validator.models.database import ComputeChallenge

                # Find all pending challenges for this miner using status field
                now = datetime.utcnow()
                pending_challenges = (
                    session.query(ComputeChallenge)
                    .filter(
                        ComputeChallenge.hotkey == peer_hotkey,
                        ComputeChallenge.challenge_status == ChallengeStatus.CREATED,
                        ComputeChallenge.deleted_at.is_(None),  # Not deleted
                    )
                    .all()
                )

                if pending_challenges:
                    timeout_secs = self.config.get_positive_number(
                        "validation.challenge_timeout", int
                    )

                    # Prepare challenge data for batch processing
                    challenges_data = []
                    for challenge in pending_challenges:
                        challenge_data = {
                            "challenge_id": challenge.challenge_id,
                            "challenge_type": challenge.challenge_type,
                            "data": challenge.challenge_data,
                            "timeout": timeout_secs,
                            "target_worker_id": challenge.worker_id,
                        }
                        challenges_data.append(challenge_data)

                        # Mark challenge as sent and update expiration time with same timestamp
                        challenge.sent_at = now
                        challenge.challenge_status = ChallengeStatus.SENT
                        challenge.expires_at = now + timedelta(seconds=timeout_secs)

                    session.commit()

                    # Always use batch response for consistency
                    response = TaskResponse(
                        task_type="compute_challenge_batch",
                        task_data={"challenges": challenges_data},
                    )
                    bt.logging.debug(
                        f"📋 Sent {len(pending_challenges)} challenges in batch to {peer_hotkey}"
                    )
                else:
                    # No tasks available
                    response = TaskResponse(task_type="no_task", task_data=None)
                    bt.logging.debug(f"📋 No tasks available for {peer_hotkey}")

            return response.model_dump(), 0

        except Exception as e:
            bt.logging.error(
                f"❌ Task request processing failed for {peer_hotkey}: {e}"
            )
            return None, 1

    async def _check_expired_data_on_startup(self) -> None:
        """
        Clean up expired tasks and offline workers on startup

        Performs database cleanup to remove stale data from previous
        validator sessions that may have terminated unexpectedly.
        """
        bt.logging.info("Checking for expired tasks and offline workers...")

        try:
            with self.database_manager.get_session() as session:
                # Mark expired challenges as failed
                expired_count = self.database_manager.mark_expired_tasks(session)

                # Mark workers offline based on heartbeat deadlines
                offline_count = self.database_manager.mark_workers_offline_by_deadline(
                    session
                )

                bt.logging.info(
                    f"Startup cleanup complete - expired tasks: {expired_count}, offline workers: {offline_count}"
                )

        except Exception as e:
            bt.logging.error(f"Error during startup expired data check: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            bt.logging.info(
                f"Received signal {signum} ({signal_name}), shutting down..."
            )
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_axon_handlers(self) -> None:
        """Setup axon handler functions for incoming synapses"""
        self.axon.attach(
            forward_fn=self._handle_heartbeat, blacklist_fn=self._blacklist_heartbeat
        )
        self.axon.attach(
            forward_fn=self._handle_task_request,
            blacklist_fn=self._blacklist_task_request,
        )
        self.axon.attach(
            forward_fn=self._handle_challenge_commitment,
            blacklist_fn=self._blacklist_challenge_commitment,
        )
        self.axon.attach(
            forward_fn=self._handle_challenge_proof,
            blacklist_fn=self._blacklist_challenge_proof,
        )
        self.axon.attach(
            forward_fn=self._handle_session_init,
            blacklist_fn=self._blacklist_session_init,
        )

    def _blacklist_heartbeat(self, synapse: HeartbeatSynapse) -> Tuple[bool, str]:
        return False, ""

    def _blacklist_task_request(self, synapse: TaskSynapse) -> Tuple[bool, str]:
        return False, ""

    def _blacklist_challenge_commitment(
        self, synapse: ChallengeSynapse
    ) -> Tuple[bool, str]:
        return False, ""

    def _blacklist_challenge_proof(
        self, synapse: ChallengeProofSynapse
    ) -> Tuple[bool, str]:
        return False, ""

    def _blacklist_session_init(self, synapse: SessionInitSynapse) -> Tuple[bool, str]:
        return False, ""

    async def _handle_heartbeat(self, synapse: HeartbeatSynapse) -> HeartbeatSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_task_request(self, synapse: TaskSynapse) -> TaskSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_challenge_commitment(
        self, synapse: ChallengeSynapse
    ) -> ChallengeSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_challenge_proof(
        self, synapse: ChallengeProofSynapse
    ) -> ChallengeProofSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_session_init(
        self, synapse: SessionInitSynapse
    ) -> SessionInitSynapse:
        return await self.communicator.handle_session_init(synapse)

    async def _session_cleanup_loop(self) -> None:
        """Periodic cleanup of expired sessions"""
        while self.is_running:
            try:
                # Perform session cleanup
                expired_count = (
                    self.communicator.session_manager.cleanup_expired_sessions()
                )
                if expired_count > 0:
                    bt.logging.info(f"🧹 Session cleanup | expired={expired_count}")

                await asyncio.sleep(CryptoManager.SESSION_CLEANUP_INTERVAL_SECONDS)

            except Exception as e:
                bt.logging.error(f"❌ Session cleanup error | error={e}")
                await asyncio.sleep(CryptoManager.SESSION_CLEANUP_ERROR_RETRY_SECONDS)

    async def start(self) -> None:
        """
        Start validator and all sub-services

        Raises:
            RuntimeError: If validator is already running
            Exception: If startup fails
        """
        if self.is_running:
            bt.logging.warning("Validator is already running")
            return

        try:
            # Initialize the database.
            self.database_manager.create_tables()
            # Check for expired data after database initialization
            await self._check_expired_data_on_startup()
            # Register Axon to the network.
            self.axon.serve(netuid=self.netuid, subtensor=self.subtensor)
            bt.logging.info(f"🛰️ Axon served | netuid={self.netuid}")
            # Start the Axon HTTP server.
            self.axon.start()
            bt.logging.info(
                f"🛰️ Axon online | addr={self.axon.ip}:{self.axon.port} started={self.axon.started}"
            )
            # Start the validation service.
            await self.validation_service.start()
            # Start the weight manager.
            await self.weight_manager.start()
            # Start the async verification service.
            await self.async_challenge_verifier.start()
            # Start session cleanup task
            self._session_cleanup_task = asyncio.create_task(
                self._session_cleanup_loop()
            )
            self.is_running = True
            bt.logging.info("✅ Validator started")

        except Exception as e:
            bt.logging.error(f"Error starting validator: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop validator and cleanup resources"""
        if not self.is_running:
            return
        bt.logging.info("⏹️ Stopping validator...")
        await self._graceful_shutdown()

    async def _graceful_shutdown(self) -> None:
        """Gracefully shutdown all validator services"""
        # Stop services
        await self.async_challenge_verifier.stop()
        await self.weight_manager.stop()
        await self.validation_service.stop()

        # Cancel session cleanup task
        if hasattr(self, "_session_cleanup_task") and self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            try:
                await self._session_cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop axon server
        if hasattr(self.axon, "stop"):
            self.axon.stop()
        self.database_manager.close()
        self.is_running = False
        self._shutdown_event.set()
        bt.logging.info("Validator stopped")

    async def run(self) -> None:
        """Run validator continuously until stopped"""
        try:
            await self.start()
            bt.logging.debug("Validator running | Ctrl+C to stop")
            # Keep running until shutdown event is set
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            bt.logging.info("⏹️ Validator interrupt | stopping")
        except Exception as e:
            bt.logging.error(f"❌ Validator run error | error={e}")
        finally:
            await self.stop()
