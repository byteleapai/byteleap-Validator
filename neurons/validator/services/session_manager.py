"""
Session Manager for Validator
Manages session lifecycle, replay protection, and cleanup with in-memory sessions for forward secrecy
"""

import base64
import json
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import bittensor as bt

from neurons.shared.crypto import CryptoManager
from neurons.shared.protocols import ErrorCodes
from neurons.shared.session_replay_protection import (MAX_SEQ,
                                                      SlidingWindowValidator)


class SessionState:
    """Represents active session state with replay protection"""

    def __init__(
        self,
        session_id: str,
        peer_hotkey: str,
        k_cs: bytes,
        k_sc: bytes,
        created_at: float,
        expires_at: float,
        replay_window_size: int = 1024,
    ):
        self.session_id = session_id
        self.peer_hotkey = peer_hotkey
        self.k_cs = k_cs  # Client to server key
        self.k_sc = k_sc  # Server to client key
        self.created_at = created_at
        self.expires_at = expires_at
        self.last_seen = created_at

        # Replay protection - sliding window validators for both directions
        self.replay_validator_cs = SlidingWindowValidator(
            replay_window_size
        )  # Client to server
        self.replay_validator_sc = SlidingWindowValidator(
            replay_window_size
        )  # Server to client
        self.seq_sc_out = 0  # Outgoing sequence for server to client messages
        self._seq_lock = threading.Lock()  # Protect sequence increment

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return time.time() > self.expires_at

    def update_last_seen(self):
        """Update last activity timestamp"""
        self.last_seen = time.time()


class SessionManager:
    """Manages session lifecycle for validator"""

    def __init__(
        self, database_manager, crypto_manager: CryptoManager, config: Dict[str, Any]
    ):
        self.database_manager = database_manager
        self.crypto_manager = crypto_manager

        # Flood protection configuration
        self.enable_lru_eviction = True

        # In-memory session cache
        self._sessions: Dict[str, SessionState] = {}

        # Per-peer session tracking
        self._peer_session_counts: Dict[str, int] = defaultdict(
            int
        )  # peer_hotkey -> session_count
        self._peer_session_lists: Dict[str, deque] = defaultdict(
            deque
        )  # peer_hotkey -> session_ids (LRU order)

        # Flood protection tracking
        self._handshake_attempts: Dict[str, deque] = defaultdict(
            lambda: deque()
        )  # peer_hotkey -> timestamps

        # Cleanup tracking
        self._last_cleanup = time.time()

        bt.logging.info(
            f"🧭 Session manager initialized | ttl={CryptoManager.SESSION_TTL_SECONDS}s, window={CryptoManager.REPLAY_WINDOW_SIZE}"
        )

        # Sessions are managed in memory for forward secrecy

    def _check_flood_protection(self, peer_hotkey: str) -> Optional[str]:
        """
        Check if peer is allowed to create a new session (flood protection)

        Args:
            peer_hotkey: Peer's hotkey

        Returns:
            Error message if blocked, None if allowed
        """
        current_time = time.time()

        # Check session quota per peer with LRU eviction
        if (
            self._peer_session_counts[peer_hotkey]
            >= CryptoManager.MAX_SESSIONS_PER_PEER
        ):
            # Evict least recently used session for this peer
            peer_sessions = self._peer_session_lists[peer_hotkey]
            if peer_sessions:
                # Remove oldest session (LRU)
                oldest_session_id = peer_sessions.popleft()
                self.invalidate_session(oldest_session_id)
            bt.logging.info(
                f"🧹 LRU evicted session {oldest_session_id} | peer={peer_hotkey}"
            )

        # Check handshake rate limiting
        attempts = self._handshake_attempts[peer_hotkey]

        # Clean up expired attempts outside rate limit window
        while (
            attempts
            and attempts[0] < current_time - CryptoManager.RATE_LIMIT_WINDOW_SECONDS
        ):
            attempts.popleft()

        # Check if rate limit exceeded
        if len(attempts) >= CryptoManager.HANDSHAKE_RATE_LIMIT:
            return f"Handshake rate limit exceeded: {len(attempts)}/{CryptoManager.HANDSHAKE_RATE_LIMIT} in {CryptoManager.RATE_LIMIT_WINDOW_SECONDS}s"

        # Record this attempt
        attempts.append(current_time)

        return None  # Allowed

    def accept_handshake(
        self, peer_hotkey: str, miner_eph_pub_b64: str, client_nonce: bytes
    ) -> Tuple[str, str, float, str, Optional[str]]:
        """
        Accept handshake from miner and create new session

        Args:
            peer_hotkey: Miner's hotkey
            miner_eph_pub_b64: Miner's ephemeral public key
            client_nonce: Client nonce

        Returns:
            Tuple of (session_id, validator_eph_pub_b64, expires_at, server_nonce_b64, error_message)
        """
        try:
            # Check flood protection
            flood_error = self._check_flood_protection(peer_hotkey)
            if flood_error:
                bt.logging.warning(
                    f"🚫 Handshake blocked | peer={peer_hotkey} | reason={flood_error}"
                )
                return "", "", 0.0, "", flood_error

            current_time = time.time()

            # Validate parameter lengths
            miner_pub_bytes = base64.b64decode(miner_eph_pub_b64.encode("ascii"))
            if len(miner_pub_bytes) != 32:
                return (
                    "",
                    "",
                    0.0,
                    "",
                    f"Invalid miner public key length: {len(miner_pub_bytes)} != 32",
                )
            if len(client_nonce) != 16:
                return (
                    "",
                    "",
                    0.0,
                    "",
                    f"Invalid client nonce length: {len(client_nonce)} != 16",
                )

            # Generate server nonce
            server_nonce = CryptoManager.generate_nonce()
            server_nonce_b64 = base64.b64encode(server_nonce).decode("ascii")

            # Perform handshake
            session_id, k_cs, k_sc, validator_eph_pub_b64 = (
                self.crypto_manager.accept_handshake(
                    miner_eph_pub_b64, client_nonce, server_nonce, peer_hotkey
                )
            )

            expires_at = current_time + CryptoManager.SESSION_TTL_SECONDS

            # Create session state
            session_state = SessionState(
                session_id=session_id,
                peer_hotkey=peer_hotkey,
                k_cs=k_cs,
                k_sc=k_sc,
                created_at=current_time,
                expires_at=expires_at,
                replay_window_size=CryptoManager.REPLAY_WINDOW_SIZE,
            )

            # Store in memory cache
            self._sessions[session_id] = session_state
            self._peer_session_counts[peer_hotkey] += 1

            # Add to LRU tracking (newest at end)
            self._peer_session_lists[peer_hotkey].append(session_id)

            # Session keys remain in memory only for forward secrecy

            bt.logging.info(
                f"🔒 Session created | id={session_id} peer={peer_hotkey} exp={int(expires_at)}"
            )

            return session_id, validator_eph_pub_b64, expires_at, server_nonce_b64, None

        except Exception as e:
            bt.logging.error(f"❌ Handshake failed | peer={peer_hotkey} | error={e}")
            return "", "", 0.0, "", f"Handshake failed: {str(e)}"

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session by ID

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found and valid, None otherwise
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if session.is_expired():
            bt.logging.info(f"Session expired: {session_id}")
            self.invalidate_session(session_id)
            return None

        return session

    def _update_session_lru(self, session_id: str, peer_hotkey: str) -> None:
        """Update LRU order for a session (move to end as most recently used)"""
        peer_sessions = self._peer_session_lists[peer_hotkey]
        try:
            # Remove from current position and add to end
            peer_sessions.remove(session_id)
            peer_sessions.append(session_id)
        except ValueError:
            peer_sessions.append(session_id)

    def validate_and_decrypt(
        self,
        encrypted_package_json: str,
        expected_recipient: str,
        expected_sender: str,
        synapse_type: str,
    ) -> Tuple[Any, Optional[SessionState], Optional[str]]:
        """
        Validate and decrypt session-encrypted message

        Args:
            encrypted_package_json: JSON encrypted message package
            expected_recipient: Expected recipient hotkey
            expected_sender: Expected sender hotkey (from synapse.dendrite.hotkey)
            synapse_type: Synapse type for AAD

        Returns:
            Tuple of (decrypted_data, session_state, error_message)
        """
        try:
            # First parse JSON to get session_id for session lookup
            package = json.loads(encrypted_package_json)
            session_id = package["session_id"]
            seq = package["seq"]

            # Get session
            session = self.get_session(session_id)
            if not session:
                return None, None, f"Session not found or expired: {session_id}"

            # Decrypt and authenticate using GCM - this validates sender/recipient/synapse_type via AAD
            decrypted_data, _, decrypted_seq = self.crypto_manager.decrypt_with_session(
                encrypted_package_json,
                session.k_cs,  # Client to server key
                expected_sender,
                expected_recipient,
                synapse_type,
            )

            # Validate sequence for replay protection after decryption
            # Prevents DoS attacks from advancing sequence window with invalid packages
            if not session.replay_validator_cs.validate_sequence(seq, "client"):
                from neurons.shared.protocols import ErrorCodes

                error_msg = f"SEQUENCE_ERROR:{ErrorCodes.SEQUENCE_ERROR}:Replay detected or sequence out of window"
                return None, session, error_msg

            # Update last seen and LRU order
            session.update_last_seen()
            self._update_session_lru(session_id, session.peer_hotkey)

            return decrypted_data, session, None

        except json.JSONDecodeError as e:
            return None, None, f"Malformed encrypted package: {e}"
        except KeyError as e:
            return None, None, f"Missing field in encrypted package: {e}"
        except ValueError as e:
            return None, None, f"Decryption failed: {str(e)}"
        except Exception as e:
            bt.logging.error(f"Unexpected error in decrypt: {e}")
            return None, None, f"Decrypt error: {str(e)}"

    def encrypt_response(
        self,
        data: Any,
        session_state: SessionState,
        recipient_hotkey: str,
        sender_hotkey: str,
        synapse_type: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Encrypt response data for session

        Args:
            data: Data to encrypt
            session_state: Active session
            recipient_hotkey: Recipient's hotkey
            sender_hotkey: Sender's hotkey (validator)
            synapse_type: Synapse type for AAD

        Returns:
            Tuple of (encrypted_package_b64, error_message)
        """
        try:
            # Check sequence overflow before incrementing
            if session_state.seq_sc_out >= MAX_SEQ:
                bt.logging.warning(
                    f"Session {session_state.session_id} sequence overflow, invalidating session"
                )
                self.invalidate_session(session_state.session_id)
                return (
                    "",
                    f"Session sequence overflow - session invalidated, re-handshake required",
                )

            # Use server->client key and increment outgoing sequence (thread-safe)
            with session_state._seq_lock:
                session_state.seq_sc_out += 1
                seq = session_state.seq_sc_out

            encrypted_package = self.crypto_manager.encrypt_with_session(
                data,
                session_state.session_id,
                session_state.k_sc,  # Server to client key
                seq,
                sender_hotkey,
                recipient_hotkey,
                synapse_type,
            )

            session_state.update_last_seen()
            return encrypted_package, None

        except ValueError as e:
            # Handle sequence overflow from crypto manager
            if "exceeds maximum" in str(e):
                bt.logging.warning(
                    f"Session {session_state.session_id} sequence overflow in crypto layer, invalidating session"
                )
                self.invalidate_session(session_state.session_id)
                return (
                    "",
                    f"Session sequence overflow - session invalidated, re-handshake required",
                )
            else:
                # Session not in deque, just add it
                bt.logging.error(f"Response encryption failed: {e}")
                return "", f"Encryption failed: {str(e)}"
        except Exception as e:
            bt.logging.error(f"Response encryption failed: {e}")
            return "", f"Encryption failed: {str(e)}"

    def invalidate_session(self, session_id: str) -> None:
        """Remove session from cache and database"""
        session = self._sessions.get(session_id)
        if session:
            # Decrement session count for peer
            if self._peer_session_counts[session.peer_hotkey] > 0:
                self._peer_session_counts[session.peer_hotkey] -= 1

            # Remove from LRU tracking
            peer_sessions = self._peer_session_lists[session.peer_hotkey]
            try:
                peer_sessions.remove(session_id)
            except ValueError:
                # Session not in deque (e.g., already removed by LRU eviction)
                pass

            # Remove from cache
            del self._sessions[session_id]

            bt.logging.info(f"Session invalidated: {session_id}")

        # Session removed from memory only (no persistence for security)

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        current_time = time.time()

        if (
            current_time - self._last_cleanup
            < CryptoManager.SESSION_CLEANUP_INTERVAL_SECONDS
        ):
            return 0

        expired_sessions = []
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.invalidate_session(session_id)

        self._last_cleanup = current_time

        if expired_sessions:
            bt.logging.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_session_stats(self) -> Dict[str, int]:
        """Get session statistics"""
        self.cleanup_expired_sessions()  # Cleanup first

        # Count peers with active handshake attempts
        current_time = time.time()
        rate_limited_peers = 0
        for peer_hotkey, attempts in self._handshake_attempts.items():
            # Clean up expired attempts
            while (
                attempts
                and attempts[0] < current_time - CryptoManager.RATE_LIMIT_WINDOW_SECONDS
            ):
                attempts.popleft()
            if len(attempts) >= CryptoManager.HANDSHAKE_RATE_LIMIT:
                rate_limited_peers += 1

        return {
            "total_sessions": len(self._sessions),
            "unique_peers": len(self._peer_session_lists),
            "session_count_sum": sum(self._peer_session_counts.values()),
            "rate_limited_peers": rate_limited_peers,
            "peers_with_max_sessions": len(
                [
                    count
                    for count in self._peer_session_counts.values()
                    if count >= CryptoManager.MAX_SESSIONS_PER_PEER
                ]
            ),
        }
