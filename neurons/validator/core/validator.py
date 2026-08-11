import asyncio
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from neurons.shared.crypto import CryptoManager
from neurons.shared.protocols import EncryptedSynapse as BaseSynapse
from neurons.shared.protocols import (GetVmgwEnrollTokenSynapse,
                                      HeartbeatSynapse, SessionInitSynapse,
                                      TaskSynapse)
from neurons.validator.models.database import (DatabaseManager, MinerInfo,
                                               NetworkLog, WorkerInfo)
from neurons.validator.services.communication import \
    ValidatorCommunicationService
from neurons.validator.services.data_cleanup import DataCleanupService
from neurons.validator.services.meshhub_client import MeshHubClient
from neurons.validator.services.metagraph_cache import MetagraphCache
from neurons.validator.services.processor_factory import \
    ValidatorProcessorFactory
from neurons.validator.services.subtensor_access import SubtensorAccessGuard
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
        self.wallet = bt.Wallet(config=bt_config)
        bt.logging.info(f"🚀 Wallet | name={self.wallet.name}")
        bt.logging.info(f"🔐 Hotkey | name={self.wallet.hotkey_str}")

        # Network configuration
        self.netuid = self.config.get_positive_number("netuid", int)
        self.port = self.config.get_positive_number("port", int)

        # Initialize subtensor using provided bt_config
        try:
            self.subtensor = bt.Subtensor(config=bt_config)
        except Exception as e:
            bt.logging.error(
                f"❌ Subtensor init failed | network={self.config.get_non_empty_string('subtensor.network')} | error={e}"
            )
            raise

        self.subtensor_guard = SubtensorAccessGuard(self.subtensor)

        # Initialize metagraph and cache service
        self.metagraph = bt.Metagraph(netuid=self.netuid, subtensor=self.subtensor)
        self.metagraph_cache = MetagraphCache(
            self.subtensor_guard, self.netuid, self.metagraph, config
        )

        # Database manager
        database_url = self.config.get_non_empty_string("database.url")
        self.database_manager = DatabaseManager(database_url)

        # Communication system
        self.communicator = ValidatorCommunicationService(
            self.wallet, config, self.database_manager
        )
        # Service components
        # MeshHub client
        self._fatal_reason: Optional[str] = None
        self.meshhub_client = MeshHubClient(
            self.wallet, config, self.database_manager, on_fatal=self._on_meshhub_fatal
        )
        self.weight_manager = WeightManager(
            self.database_manager,
            self.wallet,
            self.subtensor_guard,
            self.metagraph,
            config,
            meshhub_client=self.meshhub_client,
            metagraph_cache=self.metagraph_cache,
        )
        # Data retention / cleanup service (startup + nightly)
        try:
            retention_days = self.config.get_positive_number(
                "database.event_retention_days", int
            )
        except Exception as e:
            # Fail-fast inside neurons/: propagate after logging for clarity
            bt.logging.error(
                f"❌ Config invalid | key=database.event_retention_days error={e}"
            )
            raise
        self.data_cleanup_service = DataCleanupService(
            self.database_manager,
            retention_days=retention_days,
        )

        # Register communication processors
        self._register_processors()

        # Axon server
        external_ip = (
            self.config.get_optional("external_ip")
            or bt.utils.networking.get_external_ip()
        )
        self.axon = bt.Axon(wallet=self.wallet, port=self.port, ip=external_ip)

        # Register forward functions
        self._setup_axon_handlers()
        # Runtime state
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        # Setup signal handlers
        self._setup_signal_handlers()

        bt.logging.info("✅ Validator initialization complete")

    def _on_meshhub_fatal(self, reason: str) -> None:
        """Gracefully request shutdown on MeshHub fatal errors (e.g., invalid token)."""
        self._fatal_reason = reason or "meshhub_fatal"
        self._shutdown_event.set()

    def _register_processors(self) -> None:
        """Register synapse processors for communication"""
        from neurons.shared.protocols import (GetVmgwEnrollTokenSynapse,
                                              HeartbeatSynapse, TaskSynapse)

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

        enroll_token_processor = processor_factory.create_vmgw_enroll_token_processor(
            self._process_enroll_token_request
        )
        self.communicator.register_processor(
            GetVmgwEnrollTokenSynapse, enroll_token_processor
        )

        bt.logging.debug("Communication processors registered")

    async def _process_heartbeat_request(self, request_data, peer_hotkey):
        """Process decrypted heartbeat request data"""
        from neurons.shared.protocols import ErrorCodes, HeartbeatResponse

        try:
            # HeartbeatData is a Pydantic model; enforce model-based parsing
            heartbeat_dict = request_data.model_dump()

            workers_processed = await asyncio.to_thread(
                self._process_heartbeat_sync, heartbeat_dict, peer_hotkey
            )

            response = HeartbeatResponse(
                error_code=ErrorCodes.SUCCESS,
                message=f"Processed heartbeat for {workers_processed} workers",
                workers_processed=workers_processed,
            )

            bt.logging.debug(
                f"Heartbeat processed | peer={peer_hotkey} workers={workers_processed}"
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

    def _process_heartbeat_sync(
        self, heartbeat_dict: Dict[str, Any], peer_hotkey: str
    ) -> int:
        """Synchronous heartbeat processing executed in worker thread."""
        workers_processed = 0

        with self.database_manager.get_session() as session:
            # Update miner heartbeat
            self.database_manager.update_miner_heartbeat(
                session, peer_hotkey, heartbeat_dict
            )

            # Process individual workers
            for worker_info in heartbeat_dict.get("workers", []):
                worker_id = worker_info.get("worker_id")
                if not worker_id:
                    continue

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
                            gpu_uuid = gpu_detail.get("uuid")
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

        return workers_processed

    async def _process_task_request(self, request_data, peer_hotkey):
        """Process decrypted task request data"""
        from neurons.shared.protocols import ErrorCodes, TaskResponse

        bt.logging.info(
            f"Task request received | peer={peer_hotkey} task=challenge_removed"
        )
        response = TaskResponse(
            task_type="no_task",
            task_data={"reason": "challenge_removed"},
        )
        return response.model_dump(), ErrorCodes.SUCCESS

    async def _process_enroll_token_request(
        self, request_data, peer_hotkey: str
    ) -> Tuple[Dict[str, Any], int]:
        """Handle miner requests for VM gateway enroll tokens."""
        from neurons.shared.protocols import ErrorCodes

        worker_id = getattr(request_data, "worker_id", "")
        bt.logging.debug(
            f"VMGW enroll token request received | peer={peer_hotkey} worker_id={worker_id}"
        )

        try:
            state, token_data = await self.meshhub_client.acquire_enroll_token(
                peer_hotkey
            )
        except Exception as e:
            bt.logging.error(
                f"❌ VMGW enroll token handling failed | peer={peer_hotkey} | error={e}"
            )
            payload = {
                "code": ErrorCodes.NETWORK_ERROR,
                "error_message": "meshhub_unavailable",
            }
            return payload, ErrorCodes.NETWORK_ERROR

        bt.logging.debug(f"VMGW enroll token state | peer={peer_hotkey} state={state}")

        if state == "ready" and token_data:
            payload = {
                "code": ErrorCodes.SUCCESS,
                "token": token_data.get("token"),
                "expires_at": token_data.get("expires_at"),
                "enrollment_url": token_data.get("enrollment_url"),
            }
            return payload, ErrorCodes.SUCCESS

        if state == "pending":
            payload = {
                "code": ErrorCodes.SUCCESS,
                "token": None,
                "try_again_in_sec": MeshHubClient.ENROLL_TOKEN_RETRY_SECONDS,
            }
            return payload, ErrorCodes.SUCCESS

        if state == "blocked":
            payload = {
                "code": ErrorCodes.INVALID_REQUEST,
                "error_message": "token_request_failed",
            }
            return payload, ErrorCodes.INVALID_REQUEST

        if state == "unavailable":
            payload = {
                "code": ErrorCodes.NETWORK_ERROR,
                "error_message": "meshhub_unavailable",
            }
            return payload, ErrorCodes.NETWORK_ERROR

        payload = {
            "code": ErrorCodes.INVALID_RESPONSE,
            "error_message": "unexpected_token_state",
        }
        return payload, ErrorCodes.INVALID_RESPONSE

    async def _check_expired_data_on_startup(self) -> None:
        """
        Clean up expired tasks and offline workers on startup

        Performs database cleanup to remove stale data from previous
        validator sessions that may have terminated unexpectedly.
        """
        bt.logging.info("Checking for expired tasks and offline workers...")

        try:
            with self.database_manager.get_session() as session:
                # Mark workers offline based on heartbeat deadlines
                offline_count = self.database_manager.mark_workers_offline_by_deadline(
                    session
                )

                bt.logging.info(
                    f"Startup cleanup complete, offline workers: {offline_count}"
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
            forward_fn=self._handle_get_enroll_token,
            blacklist_fn=self._blacklist_get_enroll_token,
        )
        self.axon.attach(
            forward_fn=self._handle_session_init,
            blacklist_fn=self._blacklist_session_init,
        )

    def _is_authorized_hotkey(self, synapse) -> Tuple[bool, str]:
        try:
            peer_hotkey = (
                synapse.dendrite.hotkey if getattr(synapse, "dendrite", None) else None
            )
        except Exception:
            peer_hotkey = None

        if not isinstance(peer_hotkey, str) or not peer_hotkey:
            return True, "missing_hotkey"

        # Authorize only miners present in current metagraph cache
        try:
            if not self.metagraph_cache.is_member(peer_hotkey):
                return True, "hotkey_not_in_metagraph"
        except Exception:
            # Fail safe by not blacklisting if cache access throws unexpectedly
            return False, ""

        return False, ""

    def _blacklist_heartbeat(self, synapse: HeartbeatSynapse) -> Tuple[bool, str]:
        return self._is_authorized_hotkey(synapse)

    def _blacklist_task_request(self, synapse: TaskSynapse) -> Tuple[bool, str]:
        return self._is_authorized_hotkey(synapse)

    def _blacklist_get_enroll_token(
        self, synapse: GetVmgwEnrollTokenSynapse
    ) -> Tuple[bool, str]:
        return self._is_authorized_hotkey(synapse)

    def _blacklist_session_init(self, synapse: SessionInitSynapse) -> Tuple[bool, str]:
        # Block session handshakes from non‑metagraph hotkeys
        return self._is_authorized_hotkey(synapse)

    async def _handle_heartbeat(self, synapse: HeartbeatSynapse) -> HeartbeatSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_task_request(self, synapse: TaskSynapse) -> TaskSynapse:
        return await self.communicator.handle_synapse(synapse)

    async def _handle_get_enroll_token(
        self, synapse: GetVmgwEnrollTokenSynapse
    ) -> GetVmgwEnrollTokenSynapse:
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
                    bt.logging.info(f"✅ Session cleanup | expired={expired_count}")

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

            await self._check_expired_data_on_startup()

            # Start metagraph cache and wait until first successful snapshot
            await self.metagraph_cache.start()
            await self.metagraph_cache.wait_until_ready()

            await self.meshhub_client.start()

            await self.data_cleanup_service.start()

            self.axon.serve(netuid=self.netuid, subtensor=self.subtensor)
            bt.logging.info(f"✅ Axon served | netuid={self.netuid}")

            self.axon.start()
            bt.logging.info(
                f"✅ Axon online | addr={self.axon.ip}:{self.axon.port} started={self.axon.started}"
            )

            await self.weight_manager.start()

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
        bt.logging.info("Stopping validator...")
        await self._graceful_shutdown()

    async def _graceful_shutdown(self) -> None:
        """Gracefully shutdown all validator services"""

        await self.weight_manager.stop()
        await self.meshhub_client.stop()
        await self.data_cleanup_service.stop()
        await self.metagraph_cache.stop()

        if hasattr(self, "_session_cleanup_task") and self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            try:
                await self._session_cleanup_task
            except asyncio.CancelledError:
                pass

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

            await self._shutdown_event.wait()

            if getattr(self, "_fatal_reason", None):
                bt.logging.error(
                    f"❌ Fatal shutdown requested | reason={self._fatal_reason}"
                )

                raise RuntimeError(self._fatal_reason)
        except KeyboardInterrupt:
            bt.logging.info("Validator interrupt | stopping")
        except Exception as e:
            bt.logging.error(f"❌ Validator run error | error={e}")
        finally:
            await self.stop()
