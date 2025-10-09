"""
Communication Logging System
Centralized logging for all communication operations
"""

from datetime import datetime
from typing import Any, Dict, Optional

import bittensor as bt

from neurons.shared.protocols import CommunicationResult, ErrorCodes


class CommunicationLogger:
    """Centralized communication logging"""

    def __init__(self, component_name: str):
        self.component_name = component_name

    def log_request_start(
        self, operation: str, synapse_type: str, peer_address: str = "unknown"
    ) -> None:
        """Log start of request processing"""
        bt.logging.debug(
            f"start | op={operation} type={synapse_type} peer={peer_address}"
        )

    def log_request_complete(
        self, operation: str, result: CommunicationResult, peer_address: str = "unknown"
    ) -> None:
        """Log completion of request processing"""
        if result.success:
            bt.logging.info(
                f"✅ done | op={operation} peer={peer_address} time={result.processing_time_ms:.1f}ms"
            )
        else:
            bt.logging.error(
                f"❌ fail | op={operation} peer={peer_address} code={result.error_code} time={result.processing_time_ms:.1f}ms err={result.error_message}"
            )

    def log_outbound_request(
        self, operation: str, target_count: int, synapse_type: str
    ) -> None:
        """Log outbound request initiation"""
        bt.logging.info(
            f"📤 send | op={operation} type={synapse_type} targets={target_count}"
        )

    def log_outbound_results(
        self, operation: str, success_count: int, total_count: int
    ) -> None:
        """Log outbound request results"""
        if success_count > 0:
            bt.logging.info(
                f"✅ result | op={operation} ok={success_count}/{total_count}"
            )
        else:
            bt.logging.error(f"❌ result | op={operation} ok=0/{total_count}")

    def log_encryption_metrics(
        self, operation: str, encryption_time: float, decryption_time: float = 0.0
    ) -> None:
        """Log encryption/decryption performance"""
        if encryption_time > 0 and decryption_time > 0:
            bt.logging.debug(
                f"crypto | op={operation} enc={encryption_time:.1f}ms dec={decryption_time:.1f}ms"
            )
        elif encryption_time > 0:
            bt.logging.debug(f"enc | op={operation} time={encryption_time:.1f}ms")
        elif decryption_time > 0:
            bt.logging.debug(f"dec | op={operation} time={decryption_time:.1f}ms")

    def log_validation_error(
        self, operation: str, error_message: str, peer_address: str = "unknown"
    ) -> None:
        """Log validation errors"""
        bt.logging.warning(
            f"⚠️ validation | op={operation} peer={peer_address} err={error_message}"
        )

    def log_security_event(
        self, event_type: str, details: str, peer_address: str = "unknown"
    ) -> None:
        """Log security-related events"""
        bt.logging.warning(
            f"🔒 security | type={event_type} peer={peer_address} detail={details}"
        )


class NetworkRecorder:
    """Records communication to database (for validators only)"""

    def __init__(
        self, database_manager, logger: CommunicationLogger, record_enabled: bool = True
    ):
        self.db_manager = database_manager
        self.logger = logger
        # When False, skip DB inserts to network_logs to control growth
        self.record_enabled = bool(record_enabled)

    def record_inbound_request(
        self,
        session,
        synapse_type: str,
        peer_hotkey: str,
        decrypted_data: Any,
        synapse: Any = None,
    ) -> int:
        """Record inbound request to network_logs"""
        # Skip recording if disabled (e.g., non-DEBUG log level)
        if not self.record_enabled:
            return 0
        try:
            # Extract worker_id from decrypted data if available
            worker_id = None
            if hasattr(decrypted_data, "worker_id"):
                worker_id = decrypted_data.worker_id
            elif isinstance(decrypted_data, dict) and "worker_id" in decrypted_data:
                worker_id = decrypted_data["worker_id"]

            client_ip = None
            client_port = None
            raw_synapse_data = None
            if synapse:
                # Extract client IP/port from dendrite
                if hasattr(synapse, "dendrite") and synapse.dendrite:
                    client_ip = getattr(synapse.dendrite, "ip", None)
                    client_port = getattr(synapse.dendrite, "port", None)

                # Extract raw synapse data
                if hasattr(synapse, "model_dump"):
                    raw_synapse_data = synapse.model_dump()
                elif hasattr(synapse, "__dict__"):
                    raw_synapse_data = vars(synapse)

            # Create endpoint identifier
            endpoint = f"{synapse_type.lower().replace('synapse', '')}/{peer_hotkey}"

            # Record to database
            network_log = self.db_manager.log_network_request(
                session=session,
                direction="inbound",
                endpoint=endpoint,
                synapse_type=synapse_type,
                hotkey=peer_hotkey,
                worker_id=worker_id,
                client_ip=client_ip,
                client_port=client_port,
                raw_synapse_data=raw_synapse_data,
                decrypted_data=decrypted_data.model_dump(),
                error_code=ErrorCodes.SUCCESS,
            )

            return network_log.id

        except Exception as e:
            self.logger.log_validation_error(
                f"database recording for {synapse_type}", str(e)
            )
            return 0

    def record_processing_result(
        self,
        session,
        log_id: int,
        result: CommunicationResult,
        response_data: Any = None,
        processing_time_ms: float = None,
    ) -> None:
        """Update network log with processing results"""
        if log_id == 0:
            return

        try:
            response_dict = {"error_code": result.error_code}
            if response_data and hasattr(response_data, "model_dump"):
                response_dict.update(response_data.model_dump())
            elif response_data:
                response_dict.update({"result": str(response_data)})

            self.db_manager.update_network_log(
                session=session,
                log_id=log_id,
                error_code=result.error_code,
                response_data=response_dict,
                processing_time_ms=processing_time_ms,
                error_message=result.error_message,
            )

        except Exception as e:
            self.logger.log_validation_error("database update", str(e))


class MinimalLogger:
    """Minimal logger for miner (no database recording)"""

    def __init__(self, logger: CommunicationLogger):
        self.logger = logger

    def log_outbound_attempt(
        self, operation: str, peer_hotkey: str, synapse_type: str
    ) -> None:
        """Log outbound request attempt"""
        self.logger.log_outbound_request(operation, 1, synapse_type)

    def log_outbound_result(
        self, operation: str, success: bool, error: Optional[str] = None
    ) -> None:
        """Log outbound request result"""
        if success:
            self.logger.log_outbound_results(operation, 1, 1)
        else:
            self.logger.log_outbound_results(operation, 0, 1)
            if error:
                self.logger.log_validation_error(operation, error)
