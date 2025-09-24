"""
Validator Communication Service
Handles incoming synapses from miners with clean logging and database recording
"""

import base64
import json
import time
from typing import Any, Dict, Type

import bittensor as bt
from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

from neurons.shared.communication_logging import (CommunicationLogger,
                                                  NetworkRecorder)
from neurons.shared.crypto import CryptoManager
from neurons.shared.protocols import (CommunicationResult, EncryptedSynapse,
                                      ErrorCodes, SessionInitRequest,
                                      SessionInitResponse, SessionInitSynapse)
from neurons.shared.synapse import SynapseHandler
from neurons.validator.services.processor_factory import \
    ValidatorProcessorFactory
from neurons.validator.services.session_manager import SessionManager
from neurons.validator.synapse_processor import SynapseProcessor


class ValidatorCommunicationService:
    """Validator communication service with network logging and database recording"""

    def __init__(self, wallet: bt.wallet, config: Dict[str, Any], database_manager):
        # Core components
        self.wallet = wallet
        self.config = config
        self.database_manager = database_manager

        # Crypto components
        self.session_crypto = CryptoManager(wallet)
        # Extract actual config dict if config is a ConfigManager object
        config_dict = config.config if hasattr(config, "config") else config
        self.session_manager = SessionManager(
            database_manager, self.session_crypto, config_dict
        )

        # Synapse handler
        self.synapse_handler = SynapseHandler()

        # Logging components
        self.logger = CommunicationLogger("validator")
        self.recorder = NetworkRecorder(database_manager, self.logger)

        # Registered processors keyed by stable protocol type string
        self._processors: Dict[str, SynapseProcessor] = {}

        bt.logging.info("🛰️ Validator communication service initialized")

    def register_processor(
        self, synapse_type: Type[EncryptedSynapse], processor: SynapseProcessor
    ) -> None:
        """Register processor for specific synapse type using stable PROTOCOL_TYPE"""
        if not hasattr(synapse_type, "PROTOCOL_TYPE"):
            raise ValueError("Attempted to register synapse without PROTOCOL_TYPE")
        # Ensure synapse is present in registry for consistency
        try:
            from neurons.shared.protocols import ProtocolRegistry

            ProtocolRegistry.register(synapse_type)
        except Exception as e:
            bt.logging.warning(f"Synapse registry consistency warning: {e}")

        protocol_key = synapse_type.PROTOCOL_TYPE
        self._processors[protocol_key] = processor
        bt.logging.debug(f"Registered processor for {protocol_key}")

    async def handle_session_init(
        self, synapse: SessionInitSynapse
    ) -> SessionInitSynapse:
        """
        Handle session initialization request from miner

        Args:
            synapse: Session initialization synapse

        Returns:
            Synapse with session response or error
        """
        peer_hotkey = synapse.dendrite.hotkey if synapse.dendrite else "unknown"
        peer_address = self.synapse_handler.get_peer_address(synapse)

        try:
            self.logger.log_request_start(
                "handle_session_init", "SessionInit", peer_address
            )

            # Parse session init request
            if not synapse.request:
                raise ValueError("Empty session init request")

            request_data = json.loads(synapse.request)
            session_request = SessionInitRequest(**request_data)

            # Decode nonces
            client_nonce = base64.b64decode(
                session_request.client_nonce16.encode("ascii")
            )

            # Accept handshake
            (
                session_id,
                validator_eph_pub_b64,
                expires_at,
                server_nonce_b64,
                error_msg,
            ) = self.session_manager.accept_handshake(
                peer_hotkey, session_request.miner_eph_pub32, client_nonce
            )

            if error_msg:
                bt.logging.error(
                    f"❌ Session handshake failed | peer={peer_hotkey} | reason={error_msg}"
                )
                # Return error in response
                synapse.response = json.dumps(
                    {"error": error_msg, "error_code": ErrorCodes.HANDSHAKE_FAILED}
                )
                return synapse

            # Create successful response
            response = SessionInitResponse(
                validator_eph_pub32=validator_eph_pub_b64,
                session_id=session_id,
                server_nonce16=server_nonce_b64,
                expires_at=expires_at,
            )

            synapse.response = json.dumps(response.model_dump())

            bt.logging.info(
                f"🔒 Session established | id={session_id} peer={peer_hotkey}"
            )

            result = CommunicationResult(success=True)
            self.logger.log_request_complete(
                "handle_session_init", result, peer_address
            )

            return synapse

        except Exception as e:
            bt.logging.error(f"❌ Session init error | peer={peer_hotkey} | error={e}")
            synapse.response = json.dumps(
                {"error": str(e), "error_code": ErrorCodes.HANDSHAKE_FAILED}
            )

            result = CommunicationResult(success=False, error_message=str(e))
            self.logger.log_request_complete(
                "handle_session_init", result, peer_address
            )

            return synapse

    async def handle_synapse(self, synapse: EncryptedSynapse) -> EncryptedSynapse:
        """
        Handle incoming synapse with clean separation of concerns

        Args:
            synapse: Incoming encrypted synapse

        Returns:
            Synapse with encrypted response (if applicable)
        """
        start_time = time.time()
        synapse_type = type(synapse)
        if not hasattr(synapse_type, "PROTOCOL_TYPE"):
            # Treat as protocol mismatch
            peer_address = self.synapse_handler.get_peer_address(synapse)
            self.logger.log_validation_error(
                "protocol_mismatch", "Missing PROTOCOL_TYPE", peer_address
            )
            synapse.response = json.dumps(
                {
                    "error": "Missing PROTOCOL_TYPE",
                    "error_code": ErrorCodes.SESSION_REQUIRED,
                }
            )
            return synapse
        synapse_type_name = synapse_type.PROTOCOL_TYPE
        peer_hotkey = synapse.dendrite.hotkey if synapse.dendrite else "unknown"
        peer_address = self.synapse_handler.get_peer_address(synapse)

        network_log_id = 0
        result = CommunicationResult(success=False)

        try:
            # Log operation start
            self.logger.log_request_start(
                "handle_synapse", synapse_type_name, peer_address
            )

            # Find processor
            # Lookup by stable protocol type string to avoid class identity mismatches
            processor = self._processors.get(synapse_type_name)
            if not processor:
                result.error_code = ErrorCodes.SESSION_REQUIRED
                result.error_message = f"No processor for {synapse_type_name}"
                self.logger.log_validation_error(
                    "processor lookup", result.error_message, peer_address
                )
                # Return plaintext error for visibility
                try:
                    synapse.response = json.dumps(
                        {"error": result.error_message, "error_code": result.error_code}
                    )
                except Exception:
                    pass
                return synapse

            # Decrypt request data using session-based decryption
            try:
                if not synapse.request:
                    raise ValueError("Empty synapse request data")

                # Use session manager for decryption
                decrypted_data, session_state, error_msg = (
                    self.session_manager.validate_and_decrypt(
                        synapse.request,
                        self.wallet.hotkey.ss58_address,
                        peer_hotkey,
                        synapse_type_name,
                    )
                )

                if error_msg:
                    # Parse structured error message format: ERROR_TYPE:code:message
                    if ":" in error_msg and error_msg.count(":") >= 2:
                        try:
                            error_parts = error_msg.split(":", 2)
                            error_code = int(error_parts[1])
                            actual_message = error_parts[2]
                            result.error_code = error_code
                            result.error_message = actual_message
                        except (ValueError, IndexError):
                            # Use original message if parsing fails
                            result.error_code = ErrorCodes.SESSION_UNKNOWN
                            result.error_message = error_msg
                    else:
                        # Handle unstructured error messages
                        if (
                            "not found" in error_msg.lower()
                            or "expired" in error_msg.lower()
                        ):
                            result.error_code = ErrorCodes.SESSION_EXPIRED
                        else:
                            result.error_code = ErrorCodes.SESSION_UNKNOWN
                        result.error_message = error_msg
                    self.logger.log_validation_error(
                        "session_decryption", error_msg, peer_address
                    )
                    # Return unencrypted error response for session issues
                    synapse.response = json.dumps(
                        {"error": error_msg, "error_code": result.error_code}
                    )
                    return synapse

                # Convert decrypted data to request object
                request_data = processor.request_class(**decrypted_data)
                decryption_time = 0.0  # Session manager handles timing internally

                bt.logging.debug(
                    f"Session decrypted for {peer_hotkey}: session={session_state.session_id}"
                )

            except (ValueError, RuntimeError) as e:
                result.error_code = ErrorCodes.BAD_AAD
                result.error_message = f"Session decryption failed: {str(e)}"
                self.logger.log_validation_error(
                    "session_decryption", result.error_message, peer_address
                )
                # Return unencrypted error response so miner can recover
                synapse.response = json.dumps(
                    {
                        "error": result.error_message,
                        "error_code": result.error_code,
                    }
                )
                return synapse
            except Exception as e:
                result.error_code = ErrorCodes.SESSION_UNKNOWN
                result.error_message = f"Unexpected session error: {str(e)}"
                self.logger.log_validation_error(
                    "session_decryption", result.error_message, peer_address
                )
                # Return unencrypted error response for unexpected errors
                synapse.response = json.dumps(
                    {
                        "error": result.error_message,
                        "error_code": result.error_code,
                    }
                )
                return synapse

            # Record to database
            with self.database_manager.get_session() as session:
                network_log_id = self.recorder.record_inbound_request(
                    session,
                    synapse_type_name,
                    peer_hotkey,
                    request_data,
                    synapse=synapse,
                )

            # Process request
            processing_start_time = time.time()
            try:
                response_data, error_code = await processor.process_request(
                    request_data, peer_hotkey
                )

                if error_code == 0:
                    result.success = True
                    result.data = response_data
                else:
                    result.error_code = error_code
                    result.error_message = f"Processing failed with code {error_code}"
                    # Ensure response_data is not None for consistency
                    if response_data is None:
                        response_data = {
                            "error": result.error_message,
                            "error_code": error_code,
                        }
                    else:
                        if (
                            response_data
                            and isinstance(response_data, dict)
                            and "error" in response_data
                        ):
                            result.error_message = response_data.get("error")

            except (IntegrityError, OperationalError, DatabaseError) as e:
                result.error_code = ErrorCodes.DB_ERROR
                result.error_message = f"Database error during processing: {str(e)}"
                self.logger.log_validation_error(
                    "database", result.error_message, peer_address
                )
            except (ValueError, KeyError, AttributeError) as e:
                result.error_code = ErrorCodes.VALIDATION_FAILED
                result.error_message = f"Processing validation error: {str(e)}"
                self.logger.log_validation_error(
                    "processing", result.error_message, peer_address
                )
            except Exception as e:
                result.error_code = ErrorCodes.INVALID_RESPONSE
                result.error_message = f"Processing failed: {str(e)}"
                bt.logging.error(f"❌ Request processing error | error={e}")

            processing_time_ms = (time.time() - processing_start_time) * 1000

            # Record processing result to database
            if network_log_id > 0:
                with self.database_manager.get_session() as session:
                    self.recorder.record_processing_result(
                        session,
                        network_log_id,
                        result,
                        response_data if result.success else None,
                        processing_time_ms=processing_time_ms,
                    )

            # Encrypt response if successful using session; otherwise send clear error
            bt.logging.debug(
                f"Response conditions: success={result.success}, response_data={bool(response_data)}, session_state={bool(session_state)}, data_type={type(response_data)}"
            )
            if session_state and response_data is not None:
                try:
                    encrypted_response, encrypt_error = (
                        self.session_manager.encrypt_response(
                            response_data,
                            session_state,
                            peer_hotkey,
                            self.wallet.hotkey.ss58_address,
                            synapse_type_name,
                        )
                    )

                    if encrypt_error:
                        bt.logging.error(
                            f"❌ Session response encryption failed | error={encrypt_error}"
                        )
                        # Set error response when encryption fails
                        error_payload = {
                            "error": f"Response encryption failed: {encrypt_error}",
                            "error_code": ErrorCodes.SESSION_UNKNOWN,
                        }
                        try:
                            synapse.response = json.dumps(error_payload)
                        except Exception:
                            synapse.response = json.dumps(
                                {
                                    "error": "encryption failed",
                                    "error_code": ErrorCodes.SESSION_UNKNOWN,
                                }
                            )
                    else:
                        synapse.response = encrypted_response
                        bt.logging.debug(
                            f"🔒 Session response encrypted for {peer_hotkey}: {len(encrypted_response)} chars"
                        )

                except Exception as e:
                    bt.logging.error(
                        f"❌ Session response encryption error | error={e}"
                    )
                    # Set error response when encryption fails
                    error_payload = {
                        "error": f"Response encryption failed: {str(e)}",
                        "error_code": ErrorCodes.SESSION_UNKNOWN,
                    }
                    try:
                        synapse.response = json.dumps(error_payload)
                    except Exception:
                        synapse.response = json.dumps(
                            {
                                "error": "encryption failed",
                                "error_code": ErrorCodes.SESSION_UNKNOWN,
                            }
                        )
            else:
                # Build error payload and send as plaintext so miner can handle uniformly
                error_payload = {
                    "error": result.error_message or "Processing failed",
                    "error_code": result.error_code or ErrorCodes.INVALID_RESPONSE,
                }
                try:
                    synapse.response = json.dumps(error_payload)
                except Exception:
                    # As a last resort, ensure a minimal string is sent
                    synapse.response = json.dumps(
                        {
                            "error": "processing failed",
                            "error_code": ErrorCodes.INVALID_RESPONSE,
                        }
                    )

            # Log completion
            self.logger.log_request_complete("handle_synapse", result, peer_address)

            return synapse

        except Exception as e:
            result.error_code = ErrorCodes.INVALID_RESPONSE
            result.error_message = f"Unexpected error: {str(e)}"
            bt.logging.error(f"❌ Synapse handling error | error={e}")

            # Log completion with error
            self.logger.log_request_complete("handle_synapse", result, peer_address)

            return synapse

        finally:
            # Always record timing
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        session_stats = self.session_manager.get_session_stats()
        return {
            "registered_processors": len(self._processors),
            "processor_types": list(self._processors.keys()),
            "session_stats": session_stats,
        }
