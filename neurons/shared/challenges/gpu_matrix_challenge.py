"""
GPU Matrix Multiplication Challenge
GPU-intensive algorithm for performance evaluation using CUDA
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from neurons.shared.challenges.base_matrix_challenge import BaseMatrixChallenge


class GPUMatrixChallenge(BaseMatrixChallenge):
    """
    GPU-intensive matrix multiplication challenge using CUDA
    """

    @staticmethod
    def generate_challenge(
        matrix_size: int,
        validator_hotkey: str,
        enable_dynamic_size: bool,
        size_variance: float,
        iterations: int = 1,
        mode: int = 0,
        min_matrix_size: int = 4096,
        max_matrix_size: int = 16384,
        max_size_variance: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Generate a GPU matrix multiplication challenge
        """
        if not isinstance(matrix_size, int) or matrix_size <= 0:
            raise ValueError(
                f"matrix_size must be a positive integer, got {matrix_size}"
            )

        if not validator_hotkey or not isinstance(validator_hotkey, str):
            raise ValueError("validator_hotkey must be a non-empty string")

        if not isinstance(size_variance, (int, float)) or size_variance < 0:
            raise ValueError(
                f"size_variance must be a non-negative number, got {size_variance}"
            )

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {iterations}")

        if not isinstance(mode, int) or mode < 0:
            raise ValueError(f"mode must be a non-negative integer, got {mode}")

        # Generate secure seed like CPU matrix challenge
        challenge_seed = GPUMatrixChallenge.generate_secure_seed()

        actual_matrix_size = matrix_size
        if enable_dynamic_size and size_variance > 0:
            actual_matrix_size = GPUMatrixChallenge._calculate_dynamic_matrix_size(
                base_size=matrix_size, variance=size_variance, seed=challenge_seed
            )

        challenge_data = {
            "challenge_type": "gpu_matrix",
            "matrix_size": actual_matrix_size,
            "seed": challenge_seed.hex(),  # Convert bytes to hex string for JSON storage
            "iterations": iterations,
            "mode": mode,
            "target_gpu_id": -1,  # All available GPUs
        }

        return challenge_data

    @staticmethod
    def _calculate_dynamic_matrix_size(
        base_size: int, variance: float, seed: bytes
    ) -> int:
        """
        Calculate dynamic matrix size with deterministic randomness
        """
        if variance == 0.0:
            return base_size

        import numpy as np

        seed_int = int.from_bytes(seed[:4], "big")
        rng = np.random.RandomState(seed_int)

        min_size = int(base_size * (1.0 - variance))
        max_size = int(base_size * (1.0 + variance))

        # Ensure minimum size is at least 1
        min_size = max(1, min_size)

        if min_size >= max_size:
            return base_size

        return rng.randint(min_size, max_size + 1)

    @staticmethod
    def execute_challenge(
        challenge_data: Dict[str, Any],
        gpu_response: Dict[str, Any],
        execution_time_ms: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process GPU execution results and return in unified format.

        Args:
            challenge_data: Original challenge parameters
            gpu_response: Response from GPU server execution
            execution_time_ms: Actual execution time in milliseconds

        Returns:
            A tuple containing:
            - The unified challenge result dictionary.
            - A dictionary with data to be cached (row_hashes per GPU).
        """
        try:
            if not gpu_response or not gpu_response.get("success"):
                error_msg = (
                    gpu_response.get("error", "Unknown error")
                    if gpu_response
                    else "No response"
                )
                raise RuntimeError(f"GPU server returned an error: {error_msg}")

            gpu_results = gpu_response.get("results", [])
            if not gpu_results:
                raise ValueError("GPU server response missing 'results'")

            # Process results into the unified format
            commitments = []
            row_hashes_to_cache = {}

            for res in gpu_results:
                gpu_uuid = res.get("gpu_uuid")
                if not gpu_uuid:
                    logger.warning("Skipping GPU result due to missing UUID")
                    continue

                # Create the commitment object for this GPU
                commitment = {
                    "uuid": gpu_uuid,
                    "merkle_root": res.get("merkle_root", ""),
                    "sig_ver": res.get("sig_ver", 1),
                    "sig_val": res.get("sig_val", ""),
                }
                commitments.append(commitment)

                # Store row_hashes for caching
                if "row_hashes" in res:
                    row_hashes_to_cache[gpu_uuid] = {"row_hashes": res["row_hashes"]}

            if not commitments:
                raise ValueError("No valid GPU results could be processed.")

            # Build the unified result structure
            unified_result = {
                "computation_time_ms": execution_time_ms,
                "matrix_size": challenge_data["matrix_size"],
                "commitments": commitments,
            }

            return unified_result, row_hashes_to_cache

        except Exception as e:
            logger.error(f"GPU challenge execution failed: {e}", exc_info=True)
            error_result = {
                "computation_time_ms": float("inf"),
                "matrix_size": challenge_data.get("matrix_size", 0),
                "commitments": [],
                "error": str(e),
            }
            return error_result, {}
