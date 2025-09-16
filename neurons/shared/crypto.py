"""
Session-based Encrypted Communication Module
Secure end-to-end encryption with ephemeral key exchange and session management
"""

import base64
import json
import secrets
import uuid
from typing import Any, Dict, Optional, Tuple

import bittensor as bt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class CryptoManager:
    """Cryptographic manager with forward secrecy"""

    PROTOCOL_VERSION = "v1"
    AAD_PREFIX = "leap-comm"
    SESSION_TTL_SECONDS = 21600
    SESSION_CLEANUP_INTERVAL_SECONDS = 300
    SESSION_CLEANUP_ERROR_RETRY_SECONDS = 60
    REPLAY_WINDOW_SIZE = 1024
    MAX_SESSIONS_PER_PEER = 3
    HANDSHAKE_RATE_LIMIT = 5
    RATE_LIMIT_WINDOW_SECONDS = 60
    HANDSHAKE_TIMEOUT_SECONDS = 30
    HANDSHAKE_RETRY_BACKOFF_SECONDS = 5
    NEAR_EXPIRY_THRESHOLD_SECONDS = 300
    USED_CHALLENGE_CLEANUP_INTERVAL_SECONDS = 3600

    def __init__(self, wallet: bt.wallet):
        """
        Initialize crypto manager with wallet identity

        Args:
            wallet: Bittensor wallet providing identity
        """
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address

        bt.logging.info(f"🔐 Crypto manager initialized | hotkey={self.hotkey}")

    def begin_handshake(self) -> Tuple[str, bytes]:
        """
        Begin session handshake by generating ephemeral key pair

        Returns:
            Tuple of (base64_public_key, private_key_bytes)
        """
        private_key = x25519.X25519PrivateKey.generate()
        public_key = private_key.public_key()

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        return base64.b64encode(public_key_bytes).decode(
            "ascii"
        ), private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def accept_handshake(
        self,
        peer_eph_pub_b64: str,
        client_nonce: bytes,
        server_nonce: bytes,
        peer_hotkey: str = "",
    ) -> Tuple[str, bytes, bytes, str]:
        """
        Accept handshake from peer and derive session keys

        Args:
            peer_eph_pub_b64: Peer's ephemeral public key (base64)
            client_nonce: Client nonce (16 bytes)
            server_nonce: Server nonce (16 bytes)

        Returns:
            Tuple of (session_id, k_cs, k_sc, our_pub_key_b64)
        """
        # Generate our ephemeral key pair
        our_private = x25519.X25519PrivateKey.generate()
        our_public = our_private.public_key()

        # Decode peer's public key
        peer_pub_bytes = base64.b64decode(peer_eph_pub_b64.encode("ascii"))
        peer_public = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)

        # Perform ECDH
        shared_secret = our_private.exchange(peer_public)

        # Encode our public key to be included in HKDF
        our_pub_bytes = our_public.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        our_pub_b64 = base64.b64encode(our_pub_bytes).decode("ascii")

        # Derive directional session keys using HKDF with enhanced binding
        k_cs, k_sc = self._derive_session_keys(
            shared_secret,
            client_nonce,
            server_nonce,
            peer_hotkey,
            self.hotkey,
            peer_eph_pub_b64,
            our_pub_b64,
        )

        # Generate session ID
        session_id = str(uuid.uuid4())

        return session_id, k_cs, k_sc, our_pub_b64

    def complete_handshake(
        self,
        our_private_key_bytes: bytes,
        our_eph_pub_b64: str,
        peer_eph_pub_b64: str,
        client_nonce: bytes,
        server_nonce: bytes,
        peer_hotkey: str = "",
    ) -> Tuple[bytes, bytes]:
        """
        Complete handshake using our private key and peer's public key

        Args:
            our_private_key_bytes: Our ephemeral private key
            our_eph_pub_b64: Our ephemeral public key (base64)
            peer_eph_pub_b64: Peer's ephemeral public key (base64)
            client_nonce: Client nonce (16 bytes)
            server_nonce: Server nonce (16 bytes)

        Returns:
            Tuple of (k_cs, k_sc) session keys
        """
        # Reconstruct our private key
        our_private = x25519.X25519PrivateKey.from_private_bytes(our_private_key_bytes)

        # Decode peer's public key
        peer_pub_bytes = base64.b64decode(peer_eph_pub_b64.encode("ascii"))
        peer_public = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)

        # Perform ECDH
        shared_secret = our_private.exchange(peer_public)

        # Derive directional session keys with enhanced binding
        return self._derive_session_keys(
            shared_secret,
            client_nonce,
            server_nonce,
            self.hotkey,
            peer_hotkey,
            our_eph_pub_b64,
            peer_eph_pub_b64,
        )

    def _derive_session_keys(
        self,
        shared_secret: bytes,
        client_nonce: bytes,
        server_nonce: bytes,
        client_hotkey: str,
        server_hotkey: str,
        client_eph_pub_b64: str,
        server_eph_pub_b64: str,
    ) -> Tuple[bytes, bytes]:
        """
        Derive directional session keys from shared secret and nonces

        Args:
            shared_secret: ECDH shared secret
            client_nonce: Client nonce (16 bytes)
            server_nonce: Server nonce (16 bytes)
            client_hotkey: Client's hotkey
            server_hotkey: Server's hotkey
            client_eph_pub_b64: Client's ephemeral public key (base64)
            server_eph_pub_b64: Server's ephemeral public key (base64)

        Returns:
            Tuple of (k_cs, k_sc) - client-to-server and server-to-client keys
        """
        # Combine nonces for salt
        combined_nonces = client_nonce + server_nonce

        # Use nonces as fixed handshake context
        algorithm_info = b"HKDF-SHA256_AES-256-GCM_X25519"

        # Create a strong binding context for HKDF's info parameter using binary concatenation
        def create_info(direction: bytes) -> bytes:
            return (
                direction
                + b"|"
                + client_hotkey.encode("utf-8")
                + b"|"
                + server_hotkey.encode("utf-8")
                + b"|"
                + client_eph_pub_b64.encode("ascii")
                + b"|"
                + server_eph_pub_b64.encode("ascii")
                + b"|"
                + combined_nonces
                + b"|"  # Direct binary concatenation instead of hex encoding
                + algorithm_info
            )

        # Derive k_cs (client -> server)
        info_cs = create_info(b"session_key_client_to_server")
        hkdf_cs = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=combined_nonces,
            info=info_cs,
        )
        k_cs = hkdf_cs.derive(shared_secret)

        # Derive k_sc (server -> client)
        info_sc = create_info(b"session_key_server_to_client")
        hkdf_sc = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=combined_nonces,
            info=info_sc,
        )
        k_sc = hkdf_sc.derive(shared_secret)

        return k_cs, k_sc

    def encrypt_with_session(
        self,
        plaintext: Any,
        session_id: str,
        session_key: bytes,
        seq: int,
        sender_hotkey: str,
        recipient_hotkey: str,
        synapse_type: str,
    ) -> str:
        """
        Encrypt data using session key with AAD binding

        Args:
            plaintext: Data to encrypt
            session_id: Session identifier
            session_key: Session encryption key
            seq: Sequence number for replay protection
            sender_hotkey: Sender's hotkey for AAD
            recipient_hotkey: Recipient's hotkey for AAD
            synapse_type: Synapse type for AAD binding

        Returns:
            Base64 encoded encrypted package

        Raises:
            ValueError: If sequence number exceeds safe limit
        """
        # Check sequence number bounds to prevent GCM nonce reuse
        from neurons.shared.session_replay_protection import MAX_SEQ

        if seq > MAX_SEQ:
            raise ValueError(
                f"Sequence number {seq} exceeds maximum {MAX_SEQ}, session must be renewed"
            )

        # Serialize plaintext
        json_data = json.dumps(
            plaintext.model_dump() if hasattr(plaintext, "model_dump") else plaintext,
            ensure_ascii=False,
            sort_keys=True,
        )
        plaintext_bytes = json_data.encode("utf-8")

        # Generate deterministic IV (12 bytes for GCM)
        # Format: H(session_id)[:4] || be64(seq)
        session_hash = hashes.Hash(hashes.SHA256())
        session_hash.update(session_id.encode("utf-8"))
        session_digest = session_hash.finalize()
        iv = session_digest[:4] + seq.to_bytes(8, byteorder="big")

        # Create AAD (Additional Authenticated Data)
        aad_string = f"{self.AAD_PREFIX}|{sender_hotkey}|{recipient_hotkey}|{synapse_type}|session_{session_id}|seq_{seq}"
        aad = aad_string.encode("utf-8")

        # Encrypt using high-level AESGCM API
        aesgcm = AESGCM(session_key)
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext_bytes, aad)

        # Package encrypted data (IV omitted as it can be deterministically reconstructed)
        package = {
            "ver": self.PROTOCOL_VERSION,
            "session_id": session_id,
            "seq": seq,
            "ciphertext": base64.b64encode(ciphertext_with_tag).decode("ascii"),
            "sender": sender_hotkey,
            "recipient": recipient_hotkey,
            "synapse_type": synapse_type,
        }

        return json.dumps(package)

    def decrypt_with_session(
        self,
        encrypted_package_json: str,
        session_key: bytes,
        expected_sender: str,
        expected_recipient: str,
        synapse_type: str,
    ) -> Tuple[Any, str, int]:
        """
        Decrypt data using session key with AAD verification

        Args:
            encrypted_package_json: JSON encrypted package
            session_key: Session decryption key
            expected_sender: Expected sender hotkey
            expected_recipient: Expected recipient hotkey
            synapse_type: Expected synapse type

        Returns:
            Tuple of (decrypted_data, session_id, seq)

        Raises:
            ValueError: If decryption fails or AAD verification fails
        """
        try:
            # Parse JSON package
            package = json.loads(encrypted_package_json)

            # Extract components and validate version
            version = package.get("ver")
            if version != self.PROTOCOL_VERSION:
                raise ValueError(
                    f"Unsupported protocol version: got {version}, expected {self.PROTOCOL_VERSION}"
                )

            session_id = package["session_id"]
            seq = package["seq"]
            ciphertext_with_tag = base64.b64decode(package["ciphertext"])
            sender = package["sender"]
            recipient = package["recipient"]
            package_synapse_type = package["synapse_type"]

            # Verify sender/recipient
            if sender != expected_sender or recipient != expected_recipient:
                raise ValueError(
                    f"Sender/recipient mismatch: got {sender}->{recipient}, expected {expected_sender}->{expected_recipient}"
                )

            # Verify synapse type consistency
            if package_synapse_type != synapse_type:
                raise ValueError(
                    f"Synapse type mismatch: package={package_synapse_type}, expected={synapse_type}"
                )

            # Reconstruct deterministic IV
            session_hash = hashes.Hash(hashes.SHA256())
            session_hash.update(session_id.encode("utf-8"))
            session_digest = session_hash.finalize()
            iv = session_digest[:4] + seq.to_bytes(8, byteorder="big")

            # Reconstruct AAD
            aad_string = f"{self.AAD_PREFIX}|{sender}|{recipient}|{synapse_type}|session_{session_id}|seq_{seq}"
            aad = aad_string.encode("utf-8")

            # Decrypt using high-level AESGCM API
            aesgcm = AESGCM(session_key)
            decrypted_bytes = aesgcm.decrypt(iv, ciphertext_with_tag, aad)

            # Deserialize
            json_data = decrypted_bytes.decode("utf-8")
            decrypted_data = json.loads(json_data)

            return decrypted_data, session_id, seq

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid encrypted package: {e}")
        except Exception as e:
            # Catch InvalidTag, base64 decode errors, etc.
            raise ValueError(f"Decryption failed: {e}")

    @staticmethod
    def generate_nonce() -> bytes:
        """Generate cryptographic nonce (16 bytes)"""
        return secrets.token_bytes(16)

    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session identifier"""
        return str(uuid.uuid4())
