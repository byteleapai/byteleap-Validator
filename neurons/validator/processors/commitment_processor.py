"""
Validator Commitment Processor
Handles Phase 1 of two-phase verification: commitment submission and row selection
"""

import base64
import binascii
import secrets
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type

import bittensor as bt

from neurons.shared.protocols import (ChallengeSynapse, Commitment,
                                      CommitmentData, ProofRequest)
from neurons.shared.utils.error_handler import ErrorHandler
from neurons.validator.challenge_status import ChallengeStatus
from neurons.validator.models.database import ComputeChallenge
from neurons.validator.synapse_processor import SynapseProcessor


class MerkleSignatureVerifier:
    """Verifies merkle root ECDSA signatures using the correct prime256v1 curve"""

    # 65 bytes uncompressed secp256r1/prime256v1 public key
    MERKLE_PUBLIC_KEY_HEX = "045db4dcfa2559220159eba6bb8b7f16e4c4962bb9d862a7df8a1ce4138c01e14533213760877d4eaba3f84a7e1e1b29cb0f3320ff90f01dc206c22cffc0b09afb"

    @staticmethod
    def verify_merkle_signature(
        sig_ver: int, seed: str, gpu_uuid: str, merkle_root: str, signature_hex: str
    ) -> bool:
        """
        Verify merkle root signature using ECDSA prime256v1 (secp256r1) verification.

        Uses standard single-hash ECDSA where the cryptography library internally
        applies SHA-256 to the original composite message (not pre-hashed).

        Args:
            sig_ver: Signature version (0x1)
            seed: Challenge seed string
            gpu_uuid: GPU UUID
            merkle_root: Merkle root hash
            signature_hex: ECDSA signature in hex format

        The signed message format is: "0x{sig_ver:x}|{seed}|{gpu_uuid}|{merkle_root}"
        """
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
        except ImportError:
            bt.logging.error(
                "cryptography library is required. Install with: pip install cryptography"
            )
            return False

        if not merkle_root or not signature_hex or not seed or not gpu_uuid:
            bt.logging.warning(
                "Missing required parameters for signature verification."
            )
            return False

        try:
            # Signature may be provided as hex or base64; prefer hex decode first
            try:
                signature_bytes = bytes.fromhex(signature_hex)
            except ValueError:
                # Decode using base64 if hex parsing fails
                signature_bytes = base64.b64decode(signature_hex, validate=True)
            public_key_bytes = bytes.fromhex(
                MerkleSignatureVerifier.MERKLE_PUBLIC_KEY_HEX
            )
        except (ValueError, binascii.Error) as e:
            bt.logging.warning(
                f"Invalid signature encoding for GPU merkle signature: {e}"
            )
            return False

        if len(signature_bytes) != 64:
            bt.logging.warning(
                f"Invalid signature length: {len(signature_bytes)}, expected 64 bytes (r,s)."
            )
            return False

        if signature_bytes == b"\x00" * 64:
            bt.logging.warning("Empty signature detected (all zeros)")
            return False

        try:
            # GPU worker uses prime256v1/SECP256R1, not SECP256K1
            curve = ec.SECP256R1()

            # Uncompressed format starts with 0x04 prefix
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                curve, public_key_bytes
            )

            # Convert raw 64-byte signature to DER format for verification
            r = int.from_bytes(signature_bytes[:32], "big")
            s = int.from_bytes(signature_bytes[32:], "big")

            # Manual DER encoding to match the reference implementation
            def encode_der_integer(value):
                byte_length = (value.bit_length() + 7) // 8 or 1
                value_bytes = value.to_bytes(byte_length, "big")
                if value_bytes[0] & 0x80:
                    value_bytes = b"\x00" + value_bytes
                return b"\x02" + bytes([len(value_bytes)]) + value_bytes

            r_der = encode_der_integer(r)
            s_der = encode_der_integer(s)
            der_signature = b"\x30" + bytes([len(r_der + s_der)]) + r_der + s_der

            # Create composite message matching GPU implementation format
            composite_message = f"0x{sig_ver:x}|{seed}|{gpu_uuid}|{merkle_root}"

            # Standard single-hash ECDSA with SHA-256
            message_bytes = composite_message.encode()
            public_key.verify(der_signature, message_bytes, ec.ECDSA(hashes.SHA256()))

            return True

        except InvalidSignature:
            bt.logging.warning(
                f"❌ Merkle signature verification failed for root: {merkle_root[:16]}..."
            )
            return False
        except Exception as e:
            bt.logging.error(
                f"An unexpected error occurred during signature verification: {e}"
            )
            return False


class CommitmentProcessor(SynapseProcessor):
    """Process commitment submissions (Phase 1 of two-phase verification)"""

    def __init__(
        self, communicator, database_manager, verification_config, config=None
    ):
        super().__init__(communicator)
        self.database_manager = database_manager
        self.error_handler = ErrorHandler()
        self.verification_config = verification_config
        self.config = config

    def _get_cpu_row_count(self) -> int:
        return int(
            self.config.get("validation.cpu.verification.row_verification_count")
        )

    def _get_cpu_row_variance(self) -> float:
        return float(
            self.config.get(
                "validation.cpu.verification.row_verification_count_variance"
            )
        )

    def _get_gpu_coord_count(self) -> int:
        return int(
            self.config.get("validation.gpu.verification.coordinate_sample_count")
        )

    def _get_gpu_coord_variance(self) -> float:
        return float(
            self.config.get(
                "validation.gpu.verification.coordinate_sample_count_variance"
            )
        )

    def _get_gpu_row_count(self) -> int:
        return int(
            self.config.get("validation.gpu.verification.row_verification_count")
        )

    def _get_gpu_row_variance(self) -> float:
        return float(
            self.config.get(
                "validation.gpu.verification.row_verification_count_variance"
            )
        )

    @property
    def synapse_type(self) -> Type[ChallengeSynapse]:
        """Type of synapse this processor handles"""
        return ChallengeSynapse

    @property
    def request_class(self) -> Type[CommitmentData]:
        """Class to deserialize decrypted request data"""
        return CommitmentData

    async def process_request(
        self, commitment_data: CommitmentData, peer_hotkey: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Process a unified commitment submission (Phase 1).
        """
        try:
            challenge_id = commitment_data.challenge_id
            worker_id = commitment_data.worker_id
            commitments = commitment_data.commitments

            if not all([challenge_id, worker_id, commitments]):
                return {"error": "Missing challenge_id, worker_id, or commitments"}, 1

            bt.logging.info(
                f"🧪 Phase1 process | challenge_id={challenge_id} commitments={len(commitments)}"
            )

            with self.database_manager.get_session() as session:
                challenge = (
                    session.query(ComputeChallenge)
                    .filter_by(challenge_id=challenge_id, hotkey=peer_hotkey)
                    .first()
                )

                if not challenge:
                    return {
                        "error": f"Challenge {challenge_id} not found or not accessible"
                    }, 1
                if challenge.challenge_status not in [
                    ChallengeStatus.CREATED,
                    ChallengeStatus.SENT,
                ]:
                    return {"error": f"Challenge {challenge_id} already processed"}, 1

                matrix_size = challenge.challenge_data["matrix_size"]
                challenge_type = challenge.challenge_data["challenge_type"]

                # Step 1: GPU UUID inventory check and signature verification
                if challenge_type == "gpu_matrix":
                    # Get valid GPU UUIDs from heartbeat data for this worker
                    gpu_inventory = self.database_manager.get_gpu_inventory_by_worker(
                        session, peer_hotkey, worker_id
                    )
                    valid_gpu_uuids = {gpu.gpu_uuid for gpu in gpu_inventory}

                    verified_commitments = (
                        []
                    )  # Only store commitments that pass all checks

                    for commitment in commitments:
                        # Reject CPU sentinel commitments in GPU challenges
                        if commitment.uuid == "-1":
                            bt.logging.warning(
                                f"🚨 SECURITY: CPU sentinel commitment received in GPU challenge {challenge_id}"
                            )
                            challenge.challenge_status = ChallengeStatus.FAILED
                            challenge.verification_notes = "Invalid commitment: CPU sentinel submitted for GPU challenge"
                            session.commit()
                            return {"error": "Unexpected commitment uuid: -1"}, 1

                        # GPU UUID must match heartbeat inventory
                        if commitment.uuid not in valid_gpu_uuids:
                            bt.logging.warning(
                                f"🚨 SECURITY: Invalid GPU UUID {commitment.uuid} from {peer_hotkey}"
                                f"not found in heartbeat inventory (valid: {len(valid_gpu_uuids)})"
                            )
                            challenge.challenge_status = ChallengeStatus.FAILED
                            challenge.verification_notes = f"Security: Invalid GPU UUID {commitment.uuid} not in heartbeat inventory"
                            session.commit()
                            return {
                                "error": f"GPU UUID {commitment.uuid} not found in heartbeat inventory"
                            }, 1

                        # Second check: GPU commitment must have valid signature
                        if not commitment.sig_val:
                            return {
                                "error": f"GPU commitment for {commitment.uuid} is missing signature."
                            }, 1

                        if not MerkleSignatureVerifier.verify_merkle_signature(
                            commitment.sig_ver,
                            challenge.challenge_data["seed"],
                            commitment.uuid,
                            commitment.merkle_root,
                            commitment.sig_val,
                        ):
                            bt.logging.warning(
                                f"❌ Merkle signature verification failed for {commitment.uuid}"
                            )
                            challenge.challenge_status = ChallengeStatus.FAILED
                            challenge.verification_notes = f"Merkle signature verification failed for {commitment.uuid}"
                            session.commit()
                            return {
                                "error": f"Merkle signature verification failed for {commitment.uuid}"
                            }, 1

                        # Both checks passed - add to verified commitments
                        verified_commitments.append(commitment)

                    bt.logging.info(
                        f"✅ GPU validation | valid_gpus={len(verified_commitments)} inventory={len(valid_gpu_uuids)}"
                    )

                    # If no GPU commitments pass validation, fail fast
                    if len(verified_commitments) == 0:
                        challenge.challenge_status = ChallengeStatus.FAILED
                        challenge.verification_notes = "No valid GPU commitments after signature and inventory checks"
                        session.commit()
                        return {
                            "error": "No valid GPU commitments for this challenge",
                        }, 1

                # Step 2: Verify worker_id matches original assignment
                if challenge.worker_id != worker_id:
                    return {
                        "error": f"Worker ID mismatch: expected {challenge.worker_id}, got {worker_id}"
                    }, 1

                # Store commitments - only include verified GPUs and CPU commitments
                if challenge_type == "gpu_matrix":
                    # For GPU challenges: only store verified GPU commitments
                    challenge.merkle_commitments = {
                        c.uuid: c.merkle_root for c in verified_commitments
                    }
                else:
                    # CPU challenges store single commitment
                    challenge.merkle_commitments = {
                        c.uuid: c.merkle_root for c in commitments
                    }
                challenge.challenge_status = ChallengeStatus.COMMITTED
                challenge.computed_at = datetime.utcnow()

                # Calculate computation_time_ms: commitment received time - task sent time
                if challenge.sent_at:
                    computation_time_ms = (
                        challenge.computed_at - challenge.sent_at
                    ).total_seconds() * 1000
                    challenge.computation_time_ms = computation_time_ms

                proof_requests = []
                rows_to_check = []

                # Generate proof requests based on challenge type and verified commitments
                if challenge_type == "cpu_matrix":
                    # For CPU challenges: use original commitments
                    for commitment in commitments:
                        if commitment.uuid == "-1":  # CPU Challenge
                            shared_rows = self._generate_verification_rows(
                                matrix_size, challenge_id, "cpu_matrix"
                            )
                            proof_requests.append(
                                ProofRequest(
                                    uuid="-1", rows=shared_rows, coordinates=[]
                                )
                            )
                            # Record rows as [row,null] format for CPU
                            rows_to_check.extend([[row, None] for row in shared_rows])
                            break  # Should only have one CPU commitment
                else:
                    # For GPU challenges: use only verified GPU commitments
                    if verified_commitments:
                        # Generate shared verification targets once for all verified GPUs
                        shared_coords, shared_verification_rows = (
                            self._generate_gpu_verification_targets(
                                matrix_size, challenge_id
                            )
                        )

                        for commitment in verified_commitments:
                            proof_requests.append(
                                ProofRequest(
                                    uuid=commitment.uuid,
                                    rows=shared_verification_rows,
                                    coordinates=shared_coords,
                                )
                            )

                        # All GPUs share same verification targets
                        if shared_coords:
                            rows_to_check.extend([[r, c] for r, c in shared_coords])
                        if shared_verification_rows:
                            rows_to_check.extend(
                                [[row, None] for row in shared_verification_rows]
                            )

                # Store rows_to_check in database field using JSON format
                import json

                challenge.verification_targets = rows_to_check

                # Handle empty proof_requests case - mark as verified to prevent hanging
                if not proof_requests:
                    bt.logging.warning(
                        f"⚠️ Zero proof requests generated for challenge {challenge_id} "
                        f"(coordinate_count=0, row_count=0). Marking as VERIFIED."
                    )
                    challenge.challenge_status = ChallengeStatus.VERIFIED
                    challenge.verification_notes = "Auto-verified: zero proof requests generated due to configuration"
                    session.commit()

                    return {
                        "proof_requests": [],
                        "auto_verified": True,
                        "message": "Challenge auto-verified due to zero verification requirements",
                    }, 0

                response_data = {
                    "proof_requests": [pr.model_dump() for pr in proof_requests]
                }

                session.commit()

                bt.logging.info(
                    f"🧪 Phase1 complete | commitments={len(commitments)} proof_requests={len(proof_requests)}"
                )
                return response_data, 0

        except Exception as e:
            bt.logging.error(
                f"❌ Commitment processing error | error={e}", exc_info=True
            )
            return {"error": f"Commitment processing failed: {str(e)}"}, 1

    def _generate_verification_rows(
        self, matrix_size: int, challenge_id: str, challenge_type: str
    ) -> List[int]:
        """
        Generate cryptographically random rows for verification

        Security: Uses validator-controlled randomness to prevent gaming
        """
        try:
            base_row_count = self._get_cpu_row_count()
            row_variance = self._get_cpu_row_variance()

            # Apply random variance
            if row_variance > 0:
                variance_range = int(base_row_count * row_variance)
                variance = (
                    secrets.randbelow(2 * variance_range + 1) - variance_range
                )  # ±variance_range
                verify_count = max(1, min(matrix_size, base_row_count + variance))
            else:
                verify_count = min(matrix_size, base_row_count)

            # Use cryptographically secure randomness to select rows
            crypto_random = secrets.SystemRandom()
            verify_rows = sorted(crypto_random.sample(range(matrix_size), verify_count))

            bt.logging.debug(
                f"Generated {verify_count}/{matrix_size} verification rows for {challenge_type}"
            )

            return verify_rows

        except Exception as e:
            bt.logging.error(f"Error generating verification rows: {e}")
            raise

    def _generate_verification_coordinates(
        self, matrix_size: int, challenge_id: str, challenge_type: str
    ) -> List[List[int]]:
        """
        Generate cryptographically random coordinates for GPU verification

        Based on subnet-miner-gpu approach with 50-200 coordinate spot checks
        Coordinates are stored in verification_targets field as coordinate pairs
        """
        try:
            base_coord_count = self._get_gpu_coord_count()
            coord_variance = self._get_gpu_coord_variance()

            # Apply random variance
            if coord_variance > 0:
                variance_range = int(base_coord_count * coord_variance)
                variance = (
                    secrets.randbelow(2 * variance_range + 1) - variance_range
                )  # ±variance_range
                coord_count = max(1, base_coord_count + variance)
            else:
                coord_count = base_coord_count

            # Generate unique random coordinates
            crypto_random = secrets.SystemRandom()
            coordinates = []
            coordinate_set = set()

            attempts = 0
            max_attempts = coord_count * 10  # Prevent infinite loops

            while len(coordinates) < coord_count and attempts < max_attempts:
                row = crypto_random.randrange(matrix_size)
                col = crypto_random.randrange(matrix_size)
                coord_tuple = (row, col)

                if coord_tuple not in coordinate_set:
                    coordinate_set.add(coord_tuple)
                    coordinates.append([row, col])

                attempts += 1

            # Sort for consistency
            coordinates.sort()

            bt.logging.debug(
                f"Generated {len(coordinates)} verification coordinates "
                f"for {matrix_size}x{matrix_size} {challenge_type} matrix"
            )

            return coordinates

        except Exception as e:
            bt.logging.error(f"Error generating verification coordinates: {e}")
            raise

    def _generate_gpu_verification_targets(
        self, matrix_size: int, challenge_id: str
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate GPU verification targets without pre-sampling row coordinates

        Args:
            matrix_size: Size of the matrix to generate targets for
            challenge_id: Challenge identifier for deterministic generation

        Returns:
            Tuple of (all_coordinates, verification_rows)
            - all_coordinates: [row,col] coordinates + [row,None] row markers
            - verification_rows: Rows selected for full data retrieval
        """
        try:
            crypto_random = secrets.SystemRandom()

            base_coord_count = self._get_gpu_coord_count()
            coord_variance = self._get_gpu_coord_variance()

            spot_check_coords = []
            coordinate_set = set()

            if base_coord_count > 0:
                if coord_variance > 0:
                    variance_range = int(int(base_coord_count) * float(coord_variance))
                    variance = crypto_random.randint(-variance_range, variance_range)
                    coord_count = max(0, base_coord_count + variance)
                else:
                    coord_count = base_coord_count

                if coord_count > 0:
                    attempts = 0
                    max_attempts = coord_count * 10

                    while (
                        len(spot_check_coords) < coord_count and attempts < max_attempts
                    ):
                        row = crypto_random.randrange(matrix_size)
                        col = crypto_random.randrange(matrix_size)
                        coord_tuple = (row, col)

                        if coord_tuple not in coordinate_set:
                            coordinate_set.add(coord_tuple)
                            spot_check_coords.append([row, col])

                        attempts += 1

            base_row_count = self._get_gpu_row_count()
            row_variance = self._get_gpu_row_variance()

            verification_rows = []

            if base_row_count > 0:
                if row_variance > 0:
                    variance_range = int(int(base_row_count) * float(row_variance))
                    variance = crypto_random.randint(-variance_range, variance_range)
                    row_count = max(0, base_row_count + variance)
                else:
                    row_count = base_row_count

                if row_count > 0:
                    verification_rows = crypto_random.sample(
                        range(matrix_size), min(row_count, matrix_size)
                    )
                    verification_rows.sort()
            bt.logging.debug(
                f"Generated GPU verification targets: "
                f"{len(spot_check_coords)} coordinate checks + {len(verification_rows)} row requests"
            )

            return spot_check_coords, verification_rows

        except Exception as e:
            bt.logging.error(f"GPU verification target generation failed: {e}")
            return [], []
