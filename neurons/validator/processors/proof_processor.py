"""
Proof Processor
Handles Phase 2 of two-phase verification: proof data storage only
Actual verification is handled by AsyncChallengeVerifier
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple, Type

import bittensor as bt

from neurons.shared.protocols import (ChallengeProofSynapse, ProofData,
                                      ProofResponse)
from neurons.shared.utils.error_handler import ErrorHandler
from neurons.validator.challenge_status import ChallengeStatus
from neurons.validator.models.database import ComputeChallenge
from neurons.validator.services.proof_cache import LRUProofCache
from neurons.validator.synapse_processor import SynapseProcessor


class ProofProcessor(SynapseProcessor):
    """
    Proof processor that stores proof data for background verification

    This processor:
    1. Validates basic proof structure
    2. Stores proof data in the database
    3. Updates challenge status to committed
    4. Returns success response

    Actual verification is handled by AsyncChallengeVerifier
    """

    def __init__(
        self,
        communicator,
        database_manager,
        proof_cache: LRUProofCache,
        verification_config=None,
    ):
        super().__init__(communicator)
        self.database_manager = database_manager
        self.proof_cache = proof_cache
        self.error_handler = ErrorHandler()
        self.verification_config = verification_config or {}

    @property
    def synapse_type(self) -> Type[ChallengeProofSynapse]:
        """Type of synapse this processor handles"""
        return ChallengeProofSynapse

    @property
    def request_class(self) -> Type[ProofData]:
        """Class to deserialize decrypted request data"""
        return ProofData

    async def process_request(
        self, proof_data: ProofData, peer_hotkey: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Process a unified proof submission (Phase 2).
        Store proof data and mark challenge as committed for async verification.
        """
        try:
            challenge_id = proof_data.challenge_id
            proofs = proof_data.proofs

            if not all([challenge_id, proofs]):
                return {"error": "Missing challenge_id or proofs"}, 1

            bt.logging.info(
                f"🧪 Phase2 store | challenge_id={challenge_id} proofs={len(proofs)}"
            )

            with self.database_manager.get_session() as session:
                challenge = (
                    session.query(ComputeChallenge)
                    .filter_by(challenge_id=challenge_id, hotkey=peer_hotkey)
                    .first()
                )

                if not challenge:
                    return {"error": f"Challenge {challenge_id} not found"}, 1
                if challenge.challenge_status != ChallengeStatus.COMMITTED:
                    return {
                        "error": f"Challenge {challenge_id} not in committed state"
                    }, 1
                if not challenge.merkle_commitments:
                    return {
                        "error": f"No commitment found for challenge {challenge_id}"
                    }, 1

                # Basic validation of proof structure
                validation_error = self._validate_proof_structure(challenge, proofs)
                if validation_error:
                    return {"error": validation_error}, 1

                # Store proof data in memory cache for async verification
                # SECURITY: Only serialize proofs for verified commitments
                verified_proofs = self._filter_verified_proofs(challenge, proofs)
                proof_data_dict = self._serialize_proof_data(verified_proofs)

                # Require worker_id to key cache per-worker
                worker_id = challenge.worker_id
                if not worker_id:
                    return {"error": "Challenge missing worker_id for proof storage"}, 1

                # Store in LRU cache and handle evictions
                cache_data = {
                    "proofs": proof_data_dict,
                    "received_at": datetime.utcnow().isoformat(),
                    "challenge_id": challenge_id,
                }
                cache_key = f"{peer_hotkey}:{worker_id}"
                evicted_challenge_ids = self.proof_cache.store_proof(
                    cache_key, cache_data
                )

                # Mark evicted challenges as failed
                if evicted_challenge_ids:
                    evicted_challenges = (
                        session.query(ComputeChallenge)
                        .filter(
                            ComputeChallenge.challenge_id.in_(evicted_challenge_ids)
                        )
                        .all()
                    )

                    for evicted_challenge in evicted_challenges:
                        evicted_challenge.challenge_status = ChallengeStatus.FAILED
                        evicted_challenge.verification_result = False
                        evicted_challenge.verification_notes = (
                            "Cache eviction - queue full"
                        )
                        evicted_challenge.verified_at = datetime.utcnow()

                    bt.logging.warning(
                        f"⚠️ Cache eviction | evicted={len(evicted_challenge_ids)} challenges due to queue full"
                    )

                # Store debug info if available
                if proof_data.debug_info:
                    challenge.debug_info = proof_data.debug_info

                # Mark as verifying for async verification
                challenge.computed_at = datetime.utcnow()
                challenge.challenge_status = ChallengeStatus.VERIFYING

                session.commit()

                bt.logging.info(
                    f"✅ Phase 2: Proof data cached for challenge {challenge_id}, "
                    f"queued for async verification"
                )

                return {
                    "challenge_status": "queued",
                    "message": "Proof data received and queued for verification",
                    "challenge_id": challenge_id,
                }, 0

        except Exception as e:
            bt.logging.error(
                f"❌ Proof processing error | peer={peer_hotkey} error={e}"
            )
            return {"error": f"Processing failed: {str(e)}"}, 1

    def _validate_proof_structure(
        self, challenge: ComputeChallenge, proofs: List
    ) -> str:
        """
        Validate basic proof structure

        Args:
            challenge: Database challenge record
            proofs: List of proof items

        Returns:
            Error message if validation fails, None if valid
        """
        try:
            challenge_type = challenge.challenge_data.get("challenge_type")
            commitment_merkle_root = challenge.merkle_commitments or {}

            # SECURITY: Validate that proofs match exactly with merkle_commitments
            if not commitment_merkle_root:
                return "No verified commitments found in challenge (merkle_commitments is empty)"

            committed_uuids = set(commitment_merkle_root.keys())
            proof_uuids = set(proof.uuid for proof in proofs)

            missing_uuids = committed_uuids - proof_uuids
            if missing_uuids:
                bt.logging.warning(
                    f"🚨 SECURITY: Missing proofs for verified commitments: {missing_uuids}"
                )
                return f"Missing proofs for verified commitments: {missing_uuids}"

            extra_uuids = proof_uuids - committed_uuids
            if extra_uuids:
                bt.logging.warning(
                    f"🚨 SECURITY: Proofs for unverified UUIDs detected: {extra_uuids} "
                    f"(verified commitments: {list(committed_uuids)})"
                )
                return f"Unexpected proofs for unverified UUIDs: {extra_uuids}"

            # Basic structure validation per challenge type
            if challenge_type == "cpu_matrix":
                return self._validate_cpu_proof_structure(proofs)
            elif challenge_type == "gpu_matrix":
                return self._validate_gpu_proof_structure(proofs)
            else:
                return f"Unknown challenge type: {challenge_type}"

        except Exception as e:
            return f"Proof structure validation error: {str(e)}"

        return None

    def _filter_verified_proofs(
        self, challenge: ComputeChallenge, proofs: List
    ) -> List:
        """
        Filter proofs to only include those for verified commitments

        SECURITY: Only process proofs for UUIDs that exist in merkle_commitments.
        This prevents processing of phantom GPU proofs that weren't validated in Phase 1.

        Args:
            challenge: Database challenge record with merkle_commitments
            proofs: List of all proof items received from miner

        Returns:
            List of proofs only for verified commitments
        """
        if not challenge.merkle_commitments:
            bt.logging.warning(
                f"🚨 SECURITY: No merkle_commitments found for challenge {challenge.challenge_id}"
            )
            return []

        verified_uuids = set(challenge.merkle_commitments.keys())
        verified_proofs = []

        for proof in proofs:
            if proof.uuid in verified_uuids:
                verified_proofs.append(proof)
            else:
                bt.logging.warning(
                    f"🚨 SECURITY: Rejecting proof for unverified UUID: {proof.uuid} "
                    f"(not in merkle_commitments: {list(verified_uuids)})"
                )

        bt.logging.info(
            f"🔒 Proof filter | accepted={len(verified_proofs)}/{len(proofs)} commitments={len(verified_uuids)}"
        )

        return verified_proofs

    def _validate_cpu_proof_structure(self, proofs: List) -> str:
        """Validate CPU proof structure"""
        if len(proofs) != 1:
            return f"CPU challenge should have exactly 1 proof, got {len(proofs)}"

        proof = proofs[0]
        if proof.uuid != "-1":
            return f"CPU proof should have UUID '-1', got '{proof.uuid}'"

        # Check required fields
        if not proof.row_hashes:
            return "CPU proof missing row_hashes"

        if not proof.merkle_proofs:
            return "CPU proof missing merkle_proofs"

        return None

    def _validate_gpu_proof_structure(self, proofs: List) -> str:
        """Validate GPU proof structure"""
        if len(proofs) < 1:
            return f"GPU challenge should have at least 1 proof, got {len(proofs)}"

        for proof in proofs:
            # Check UUID format for GPU
            if not proof.uuid or proof.uuid == "-1":
                return f"GPU proof should have valid GPU UUID, got '{proof.uuid}'"

            # Check required fields for GPU proofs
            if not proof.coordinate_values:
                return f"GPU proof {proof.uuid} missing coordinate_values"

            # GPU proofs must have row hashes for Merkle verification
            if not proof.row_hashes:
                return f"GPU proof {proof.uuid} missing row_hashes"

            # GPU proofs must have Merkle proofs for verification
            if not proof.merkle_proofs:
                return f"GPU proof {proof.uuid} missing merkle_proofs"

        return None

    def _serialize_proof_data(self, proofs: List) -> Dict[str, Any]:
        """
        Serialize proof data for database storage

        Args:
            proofs: List of proof items

        Returns:
            Serialized proof data
        """
        serialized_proofs = {}

        for proof in proofs:
            proof_data = {
                "uuid": proof.uuid,
            }

            # Add proof-specific data
            if proof.row_hashes:
                proof_data["row_hashes"] = proof.row_hashes

            if proof.merkle_proofs:
                proof_data["merkle_proofs"] = proof.merkle_proofs

            if proof.coordinate_values:
                proof_data["coordinate_values"] = proof.coordinate_values

            if getattr(proof, "computation_time_ms", None):
                proof_data["computation_time_ms"] = proof.computation_time_ms

            serialized_proofs[proof.uuid] = proof_data

        return serialized_proofs
