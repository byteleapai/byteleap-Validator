"""
CPU Matrix Multiplication Challenge
A scalable CPU-intensive algorithm for performance evaluation
"""

import hashlib
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from neurons.shared.challenges.base_matrix_challenge import BaseMatrixChallenge
from neurons.shared.utils.merkle_tree import create_merkle_tree_from_row_hashes


class CPUMatrixChallenge(BaseMatrixChallenge):
    """
    CPU-intensive matrix multiplication challenge that scales with core count
    """

    MIN_MATRIX_SIZE = 100
    MAX_MATRIX_SIZE = 10000
    MAX_SIZE_VARIANCE = 0.5
    MATRIX_VALUE_MIN = -500
    MATRIX_VALUE_MAX = 501

    @staticmethod
    def generate_challenge(
        matrix_size: int,
        validator_hotkey: str,
        enable_dynamic_size: bool,
        size_variance: float,
        iterations: int,
    ) -> Dict[str, Any]:
        """
        Generate a matrix multiplication challenge
        """
        CPUMatrixChallenge.validate_common_parameters(
            matrix_size,
            validator_hotkey,
            size_variance,
            min_size=CPUMatrixChallenge.MIN_MATRIX_SIZE,
            max_size=CPUMatrixChallenge.MAX_MATRIX_SIZE,
            max_variance=CPUMatrixChallenge.MAX_SIZE_VARIANCE,
        )

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer")

        challenge_seed = CPUMatrixChallenge.generate_secure_seed()

        actual_matrix_size = matrix_size
        if enable_dynamic_size and size_variance > 0:
            actual_matrix_size = CPUMatrixChallenge._calculate_dynamic_matrix_size(
                base_size=matrix_size, variance=size_variance, seed=challenge_seed
            )

        challenge_data = {
            "challenge_type": "cpu_matrix",
            "matrix_size": actual_matrix_size,
            "seed": challenge_seed.hex(),  # Convert bytes to hex string for JSON storage
            "iterations": iterations,
        }

        return challenge_data

    @staticmethod
    def _calculate_dynamic_matrix_size(
        base_size: int, variance: float, seed: bytes
    ) -> int:
        """
        Calculate dynamic matrix size with deterministic randomness
        """
        seed_int = int.from_bytes(seed[:4], "big")
        rng = np.random.RandomState(seed_int)
        min_size = max(
            CPUMatrixChallenge.MIN_MATRIX_SIZE, int(base_size * (1.0 - variance))
        )
        max_size = min(
            CPUMatrixChallenge.MAX_MATRIX_SIZE, int(base_size * (1.0 + variance))
        )
        if min_size >= max_size:
            return base_size
        return rng.randint(min_size, max_size + 1)

    @staticmethod
    def _generate_matrices_from_seed(
        seed: bytes, matrix_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate matrices from seed for deterministic reproduction
        """
        seed_int = int.from_bytes(seed[:4], "big")
        rng = np.random.RandomState(seed_int)
        matrix_a = rng.randint(
            CPUMatrixChallenge.MATRIX_VALUE_MIN,
            CPUMatrixChallenge.MATRIX_VALUE_MAX,
            (matrix_size, matrix_size),
            dtype=np.int64,
        )
        matrix_b = rng.randint(
            CPUMatrixChallenge.MATRIX_VALUE_MIN,
            CPUMatrixChallenge.MATRIX_VALUE_MAX,
            (matrix_size, matrix_size),
            dtype=np.int64,
        )
        return matrix_a, matrix_b

    @staticmethod
    def execute_challenge(
        challenge_data: Dict[str, Any], actual_cores: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the matrix multiplication challenge and return in unified format.

        Returns:
            A tuple containing:
            - The unified challenge result dictionary.
            - A dictionary with data to be cached (row_hashes).
        """
        try:
            seed_hex = challenge_data["seed"]
            seed = bytes.fromhex(seed_hex)  # Convert hex string back to bytes
            matrix_size = challenge_data["matrix_size"]
            matrix_a, matrix_b = CPUMatrixChallenge._generate_matrices_from_seed(
                seed, matrix_size
            )

            start_time = time.time()

            iterations = challenge_data.get("iterations", 1)
            if iterations > 1:
                # For iterations > 1, use the result of the previous multiplication as input.
                # Data dependencies prevent naive optimizations
                result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))
                for _ in range(iterations - 1):
                    result = np.dot(result, matrix_b.astype(np.int64))
            else:
                # Default single-iteration computation
                result = np.dot(matrix_a.astype(np.int64), matrix_b.astype(np.int64))

            execution_time = time.time() - start_time

            row_hashes = [
                hashlib.sha256(result[i].tobytes()).hexdigest()[:16]
                for i in range(matrix_size)
            ]

            merkle_start_time = time.time()
            tree = create_merkle_tree_from_row_hashes(row_hashes)
            merkle_end_time = time.time()
            logger.debug(
                f"Merkle tree build time: {(merkle_end_time - merkle_start_time) * 1000:.2f}ms for {matrix_size} rows"
            )

            merkle_root = tree.get_root_hash()

            # Create the commitment object for the CPU challenge
            cpu_commitment = {
                "uuid": "-1",  # Fixed UUID for CPU challenges
                "merkle_root": merkle_root,
                "sig_ver": 0,
                "sig_val": "",
            }

            # Build the unified result structure
            unified_result = {
                "computation_time_ms": execution_time * 1000,
                "matrix_size": matrix_size,
                "commitments": [cpu_commitment],
            }

            # Data to be cached by the worker
            cacheable_data = {"-1": {"row_hashes": row_hashes}}

            return unified_result, cacheable_data

        except Exception as e:
            logger.error(f"CPU challenge execution failed: {e}")
            # Return a result structure that indicates failure
            error_result = {
                "computation_time_ms": float("inf"),
                "matrix_size": challenge_data.get("matrix_size", 0),
                "commitments": [],
                "error": str(e),
            }
            return error_result, {}
