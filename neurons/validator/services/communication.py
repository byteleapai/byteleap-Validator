"""
Validator Communication Service
Handles incoming synapses from miners with clean logging and database recording
"""

import asyncio
import base64
import json
import threading
import time
from collections import OrderedDict
from typing import Optional


class _RLPeerWindow:
    """Per-hotkey sliding window counter using fixed second buckets.

    - buckets: size = window_seconds, index = sec % window_seconds
    - total: rolling sum over active window
    - last_cleaned_sec: last second we advanced to; used to clear aged-out buckets
    """

    def __init__(self, window_seconds: int) -> None:
        self.window_seconds = int(max(1, window_seconds))
        self.counts = [0] * self.window_seconds
        self.bucket_secs = [-1] * self.window_seconds
        self.total = 0
        self.last_cleaned_sec: Optional[int] = None

    def advance(self, now_sec: int) -> None:
        if self.last_cleaned_sec is None:
            self.last_cleaned_sec = now_sec
            return
        if now_sec <= self.last_cleaned_sec:
            return
        delta = now_sec - self.last_cleaned_sec
        # Fast-forward at most window size
        if delta > self.window_seconds:
            delta = self.window_seconds
        # For each second advanced, clear the bucket for that second index
        for s in range(self.last_cleaned_sec + 1, self.last_cleaned_sec + delta + 1):
            idx = s % self.window_seconds
            prev = self.counts[idx]
            if prev:
                self.total -= prev
                self.counts[idx] = 0
            self.bucket_secs[idx] = s
        self.last_cleaned_sec = self.last_cleaned_sec + delta

    def would_block(self, max_requests: int) -> bool:
        return self.total >= int(max_requests)

    def add_now(self, now_sec: int) -> None:
        idx = now_sec % self.window_seconds
        # Ensure bucket is aligned to now_sec
        if self.bucket_secs[idx] != now_sec:
            prev = self.counts[idx]
            if prev:
                self.total -= prev
            self.counts[idx] = 0
            self.bucket_secs[idx] = now_sec
        self.counts[idx] += 1
        self.total += 1


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
        # Only record network logs to DB when log level is DEBUG to avoid growth
        lvl = self.config.get("logging.log_level")
        record_enabled = str(lvl).upper() == "DEBUG"
        self.recorder = NetworkRecorder(
            database_manager, self.logger, record_enabled=record_enabled
        )

        # Registered processors keyed by stable protocol type string
        self._processors: Dict[str, SynapseProcessor] = {}

        # Global per-hotkey rate limiter
        self._rl_window_seconds = self.config.get_positive_number(
            "security.rate_limit.window_seconds", int
        )
        self._rl_max_requests = self.config.get_positive_number(
            "security.rate_limit.max_requests", int
        )
        # Per-hotkey sliding windows: hotkey -> _RLPeerWindow
        self._rl_overall: Dict[str, _RLPeerWindow] = {}
        # RL housekeeping constants (no config dependency)
        self.RL_MAX_PEERS = 5000
        # LRU implemented with OrderedDict for O(1) move/evict
        # Key: peer_hotkey, Value: None (placeholder)
        self._rl_lru: "OrderedDict[str, None]" = OrderedDict()
        # Async lock to coordinate coroutines on the event loop
        self._rl_lock = asyncio.Lock()
        # Monotonic clock for RL to avoid wall-clock jumps
        self._rl_clock = time.monotonic
        # Simple counters
        self._rl_blocked_total: int = 0

        bt.logging.info("🛰️ Validator communication service initialized")
        bt.logging.info(
            f"🧱 RL config | window={self._rl_window_seconds}s max={self._rl_max_requests} max_peers={self.RL_MAX_PEERS}"
        )

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

            # Centralized per-hotkey rate limiting before decryption
            try:
                if peer_hotkey and synapse_type_name:
                    async with self._rl_lock:
                        now = self._rl_clock()
                        now_sec = int(now)
                        # Evict oldest peers if new peer and capacity exceeded
                        is_new_peer = peer_hotkey not in self._rl_overall
                        if is_new_peer:
                            self._rl_evict_if_needed()
                            if len(self._rl_overall) < self.RL_MAX_PEERS:
                                self._rl_overall[peer_hotkey] = _RLPeerWindow(
                                    self._rl_window_seconds
                                )
                            else:
                                bt.logging.warning(
                                    "⚠️ RL peer create blocked | reason=capacity"
                                )
                        wnd = self._rl_overall.get(peer_hotkey)
                        if wnd is not None:
                            wnd.advance(now_sec)
                            if wnd.would_block(self._rl_max_requests):
                                # Return plaintext error so miner can back off
                                err = {
                                    "error": "rate_limited",
                                    "error_code": ErrorCodes.NETWORK_ERROR,
                                }
                                try:
                                    synapse.response = json.dumps(err)
                                except Exception:
                                    synapse.response = json.dumps(
                                        {"error": "rate_limited"}
                                    )
                                bt.logging.warning(
                                    f"⚠️ Rate limited | type={synapse_type_name} hotkey={peer_hotkey} window={self._rl_window_seconds}s max={self._rl_max_requests}"
                                )
                                # Structured completion log & counters
                                result.error_code = ErrorCodes.NETWORK_ERROR
                                result.error_message = "rate_limited"
                                self._rl_blocked_total += 1
                                self.logger.log_request_complete(
                                    "handle_synapse", result, peer_address
                                )
                                return synapse
                            # Record current event
                            wnd.add_now(now_sec)
                            self._rl_touch_peer(peer_hotkey)
            except Exception:
                # Fail-open on limiter errors with warning
                bt.logging.warning(
                    "⚠️ RL error | type=unknown detail=exception_swallowed"
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

    # --- Rate limiter housekeeping helpers ---
    def _rl_touch_peer(self, peer_hotkey: str) -> None:
        """Move peer to MRU position in LRU OrderedDict."""
        try:
            if peer_hotkey in self._rl_lru:
                self._rl_lru.move_to_end(peer_hotkey, last=True)
            else:
                self._rl_lru[peer_hotkey] = None
        except Exception:
            # Best-effort only
            pass

    def _rl_evict_if_needed(self) -> None:
        """Evict least-recently-used peers if capacity exceeded."""
        try:
            evicted = 0
            while len(self._rl_overall) >= self.RL_MAX_PEERS and self._rl_lru:
                try:
                    oldest, _ = self._rl_lru.popitem(last=False)
                except Exception:
                    break
                if oldest in self._rl_overall:
                    try:
                        del self._rl_overall[oldest]
                        evicted += 1
                    except Exception:
                        pass
            if evicted:
                bt.logging.info(
                    f"🧹 RL evict | count={evicted} max_peers={self.RL_MAX_PEERS}"
                )
        except Exception:
            # Do not block request path
            pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        session_stats = self.session_manager.get_session_stats()
        return {
            "registered_processors": len(self._processors),
            "processor_types": list(self._processors.keys()),
            "session_stats": session_stats,
            "rate_limit": {
                "window_seconds": self._rl_window_seconds,
                "max_requests": self._rl_max_requests,
                "tracked_peers": len(self._rl_overall),
                "max_peers": self.RL_MAX_PEERS,
                "blocked_total": self._rl_blocked_total,
            },
        }
