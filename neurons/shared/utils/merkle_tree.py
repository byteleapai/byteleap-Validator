"""
Merkle Tree Implementation for Challenge Verification
Provides efficient cryptographic proofs for row-based computation verification
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt


@dataclass
class MerkleNode:
    """A node in the Merkle tree"""

    hash_value: str
    left_child: Optional["MerkleNode"] = None
    right_child: Optional["MerkleNode"] = None
    is_leaf: bool = False
    leaf_index: Optional[int] = None  # Only set for leaf nodes


@dataclass
class MerkleProof:
    """Merkle proof for a specific leaf node"""

    leaf_index: int
    leaf_hash: str
    proof_hashes: List[str]  # Sibling hashes from leaf to root
    proof_directions: List[bool]  # True = right sibling, False = left sibling


class MerkleTree:
    """
    Merkle tree implementation for challenge verification.

    Features:
    - Proof generation for arbitrary row indices
    - Batch proof generation for multiple rows
    - Secure hashing using SHA-256
    - Handles non-power-of-2 leaf counts
    """

    def __init__(self, leaf_hashes: List[str]):
        """
        Initialize Merkle tree from list of leaf hashes

        Args:
            leaf_hashes: List of hex-encoded hash strings (e.g., row hashes)
        """
        if not leaf_hashes:
            raise ValueError("leaf_hashes cannot be empty")

        self.leaf_hashes = leaf_hashes.copy()
        self.leaf_count = len(leaf_hashes)
        self.root: Optional[MerkleNode] = None

        # Build the tree
        self._build_tree()

        bt.logging.debug(f"Built Merkle tree with {self.leaf_count} leaves")

    def _build_tree(self) -> None:
        """Build the Merkle tree from leaf hashes"""

        # Create leaf nodes
        current_level = []
        for i, leaf_hash in enumerate(self.leaf_hashes):
            leaf_node = MerkleNode(hash_value=leaf_hash, is_leaf=True, leaf_index=i)
            current_level.append(leaf_node)

        # Build tree level by level
        while len(current_level) > 1:
            next_level = []

            # Process pairs of nodes
            for i in range(0, len(current_level), 2):
                left_child = current_level[i]

                if i + 1 < len(current_level):
                    # We have a pair
                    right_child = current_level[i + 1]
                else:
                    # Odd number of nodes - duplicate the last one
                    right_child = current_level[i]

                # Create parent node
                combined_hash = self._hash_pair(
                    left_child.hash_value, right_child.hash_value
                )
                parent_node = MerkleNode(
                    hash_value=combined_hash,
                    left_child=left_child,
                    right_child=right_child,
                    is_leaf=False,
                )

                next_level.append(parent_node)

            current_level = next_level

        # Set the root
        self.root = current_level[0]

    def _hash_pair(self, left_hash: str, right_hash: str) -> str:
        """Hash a pair of child hashes to create parent hash"""
        # Concatenate and hash
        combined = left_hash + right_hash
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get_root_hash(self) -> str:
        """Get the Merkle root hash"""
        if not self.root:
            raise RuntimeError("Tree not built")
        return self.root.hash_value

    def generate_proof(self, leaf_index: int) -> MerkleProof:
        """
        Generate Merkle proof for a specific leaf

        Args:
            leaf_index: Index of the leaf to prove (0-based)

        Returns:
            MerkleProof object containing proof data
        """
        if not (0 <= leaf_index < self.leaf_count):
            raise ValueError(
                f"leaf_index {leaf_index} out of range [0, {self.leaf_count})"
            )

        if not self.root:
            raise RuntimeError("Tree not built")

        # Collect proof hashes and directions from leaf to root
        proof_hashes = []
        proof_directions = []

        self._collect_proof_path_bottom_up(
            self.root, leaf_index, proof_hashes, proof_directions
        )

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=self.leaf_hashes[leaf_index],
            proof_hashes=proof_hashes,
            proof_directions=proof_directions,
        )

    def _contains_leaf_index(
        self, node: Optional[MerkleNode], target_index: int
    ) -> bool:
        """Check if a subtree contains a specific leaf index"""
        if not node:
            return False

        if node.is_leaf:
            return node.leaf_index == target_index

        # Check both children
        return self._contains_leaf_index(
            node.left_child, target_index
        ) or self._contains_leaf_index(node.right_child, target_index)

    def _collect_proof_path_bottom_up(
        self,
        node: MerkleNode,
        target_index: int,
        proof_hashes: List[str],
        proof_directions: List[bool],
    ) -> bool:
        """
        Collect proof path from leaf to root (natural verification order)

        Returns True if target was found in this subtree
        """
        if node.is_leaf:
            return node.leaf_index == target_index

        # Check which child contains our target
        left_contains_target = self._contains_leaf_index(node.left_child, target_index)

        if left_contains_target:
            # Target is in left subtree
            found = self._collect_proof_path_bottom_up(
                node.left_child, target_index, proof_hashes, proof_directions
            )
            if found:
                # Right sibling hash for verification path
                proof_hashes.append(node.right_child.hash_value)
                proof_directions.append(True)  # Right sibling
            return found
        else:
            # Target is in right subtree
            found = self._collect_proof_path_bottom_up(
                node.right_child, target_index, proof_hashes, proof_directions
            )
            if found:
                # Left sibling hash for verification path
                proof_hashes.append(node.left_child.hash_value)
                proof_directions.append(False)  # Left sibling
            return found

    def generate_batch_proofs(self, leaf_indices: List[int]) -> List[MerkleProof]:
        """
        Generate proofs for multiple leaves efficiently

        Args:
            leaf_indices: List of leaf indices to generate proofs for

        Returns:
            List of MerkleProof objects in the same order as input indices
        """
        proofs = []
        for leaf_index in leaf_indices:
            proof = self.generate_proof(leaf_index)
            proofs.append(proof)

        return proofs


class MerkleVerifier:
    """Utilities for verifying Merkle proofs"""

    @staticmethod
    def verify_proof(proof: MerkleProof, expected_root: str) -> bool:
        """
        Verify a Merkle proof against an expected root hash

        Args:
            proof: MerkleProof object to verify
            expected_root: Expected Merkle root hash

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            # Start with the leaf hash
            current_hash = proof.leaf_hash

            # Apply each proof hash according to direction
            for sibling_hash, is_right_sibling in zip(
                proof.proof_hashes, proof.proof_directions
            ):
                if is_right_sibling:
                    # Sibling is on the right, so current goes on left
                    combined = current_hash + sibling_hash
                else:
                    # Sibling is on the left, so current goes on right
                    combined = sibling_hash + current_hash

                # Hash the combination
                current_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

            # Check if we reached the expected root
            return current_hash == expected_root

        except Exception as e:
            bt.logging.error(f"Merkle proof verification failed: {e}")
            return False

    @staticmethod
    def verify_batch_proofs(
        proofs: List[MerkleProof], expected_root: str
    ) -> Tuple[bool, List[bool]]:
        """
        Verify multiple Merkle proofs against the same root

        Args:
            proofs: List of MerkleProof objects to verify
            expected_root: Expected Merkle root hash

        Returns:
            Tuple of (all_valid, individual_results)
        """
        individual_results = []

        for proof in proofs:
            is_valid = MerkleVerifier.verify_proof(proof, expected_root)
            individual_results.append(is_valid)

        all_valid = all(individual_results)

        return all_valid, individual_results


def create_merkle_tree_from_row_hashes(row_hashes: List[str]) -> MerkleTree:
    """
    Convenience function to create Merkle tree from challenge row hashes

    Args:
        row_hashes: List of hex-encoded row hash strings

    Returns:
        MerkleTree instance ready for proof generation
    """
    if not row_hashes:
        raise ValueError("row_hashes cannot be empty")

    return MerkleTree(row_hashes)


def generate_proofs_for_rows(
    row_hashes: List[str], row_indices: List[int]
) -> Tuple[str, List[MerkleProof]]:
    """
    Generate Merkle proofs for specific row indices

    Args:
        row_hashes: Complete list of row hashes from challenge computation
        row_indices: Indices of rows to generate proofs for

    Returns:
        Tuple of (merkle_root, list_of_proofs)
    """
    if not row_hashes:
        raise ValueError("row_hashes cannot be empty")

    if not row_indices:
        raise ValueError("row_indices cannot be empty")

    # Validate indices
    for idx in row_indices:
        if not (0 <= idx < len(row_hashes)):
            raise ValueError(f"Row index {idx} out of range [0, {len(row_hashes)})")

    # Create tree and generate proofs
    tree = MerkleTree(row_hashes)
    proofs = tree.generate_batch_proofs(row_indices)

    return tree.get_root_hash(), proofs


def verify_row_proofs(
    row_indices: List[int],
    row_hashes: List[str],
    merkle_proofs: List[Dict[str, Any]],
    expected_merkle_root: str,
) -> Tuple[bool, str]:
    """
    Verify row proofs against expected Merkle root

    Args:
        row_indices: Indices of rows being verified
        row_hashes: Hash values for the rows being verified
        merkle_proofs: List of serialized Merkle proof dictionaries
        expected_merkle_root: Expected Merkle root hash

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if len(row_indices) != len(row_hashes) or len(row_hashes) != len(merkle_proofs):
            return (
                False,
                "Mismatched array lengths: row_indices, row_hashes, and merkle_proofs must have same length",
            )

        if not row_indices:
            return False, "No rows to verify"

        # Convert dictionary proofs back to MerkleProof objects
        proof_objects = []
        for i, proof_dict in enumerate(merkle_proofs):
            try:
                proof = MerkleProof(
                    leaf_index=proof_dict["leaf_index"],
                    leaf_hash=proof_dict["leaf_hash"],
                    proof_hashes=proof_dict["proof_hashes"],
                    proof_directions=proof_dict["proof_directions"],
                )

                # Validate that proof matches expected data
                if proof.leaf_index != row_indices[i]:
                    return (
                        False,
                        f"Proof leaf_index {proof.leaf_index} doesn't match expected {row_indices[i]}",
                    )

                if proof.leaf_hash != row_hashes[i]:
                    return (
                        False,
                        f"Proof leaf_hash doesn't match expected row hash for index {row_indices[i]}",
                    )

                proof_objects.append(proof)

            except KeyError as e:
                return False, f"Invalid proof format: missing key {e}"

        # Verify all proofs
        all_valid, individual_results = MerkleVerifier.verify_batch_proofs(
            proof_objects, expected_merkle_root
        )

        if not all_valid:
            failed_indices = [
                row_indices[i]
                for i, valid in enumerate(individual_results)
                if not valid
            ]
            return (
                False,
                f"Merkle proof verification failed for row indices: {failed_indices}",
            )

        return True, "All Merkle proofs verified successfully"

    except Exception as e:
        bt.logging.error(f"Merkle proof verification error: {e}")
        return False, f"Verification error: {str(e)}"


def create_proof_payload(
    row_hashes: List[str], row_indices: List[int]
) -> Dict[str, Any]:
    """
    Creates a serializable dictionary containing Merkle proofs for specific rows.

    This function wraps the proof generation and formats it for network transmission.

    Args:
        row_hashes: Complete list of row hashes from the challenge computation.
        row_indices: List of row indices to generate proofs for.

    Returns:
        A dictionary containing the proof data for the requested rows.
    """
    if not row_hashes:
        raise ValueError("row_hashes cannot be empty")

    if not row_indices:
        raise ValueError("row_indices cannot be empty")

    # Validate row indices
    for idx in row_indices:
        if not (0 <= idx < len(row_hashes)):
            raise ValueError(f"Row index {idx} out of range [0, {len(row_hashes)})")

    try:
        # Generate Merkle proofs using the utility function in this module
        merkle_root, proof_objects = generate_proofs_for_rows(row_hashes, row_indices)

        # Convert proof objects to a serializable dictionary format
        merkle_proofs = []
        for proof in proof_objects:
            proof_dict = {
                "leaf_index": proof.leaf_index,
                "leaf_hash": proof.leaf_hash,
                "proof_hashes": proof.proof_hashes,
                "proof_directions": proof.proof_directions,
            }
            merkle_proofs.append(proof_dict)

        # Extract the hashes for the requested rows
        requested_row_hashes = [row_hashes[idx] for idx in row_indices]

        proof_payload = {
            "row_indices": row_indices,
            "row_hashes": requested_row_hashes,
            "merkle_proofs": merkle_proofs,
            "merkle_root": merkle_root,
            "proof_count": len(merkle_proofs),
        }

        bt.logging.debug(f"Generated proof payload for {len(row_indices)} rows")
        return proof_payload

    except Exception as e:
        bt.logging.error(f"Proof payload creation failed: {e}")
        raise RuntimeError(f"Failed to create proof payload: {str(e)}")
