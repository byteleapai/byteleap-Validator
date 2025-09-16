"""
Base Matrix Challenge
Common functionality for CPU and GPU matrix multiplication challenges
"""

import secrets
from typing import Any


class BaseMatrixChallenge:
    """Base class for matrix multiplication challenges"""

    @staticmethod
    def generate_secure_seed() -> bytes:
        """
        Generate 128-bit secure seed to prevent pre-computation attacks.

        Returns:
            128-bit cryptographically secure random bytes
        """
        return secrets.token_bytes(16)

    @staticmethod
    def calculate_dynamic_matrix_size(
        base_size: int, size_variance: float, enable_dynamic_size: bool = True
    ) -> int:
        """
        Calculate matrix size with dynamic variance for anti-gaming

        Args:
            base_size: Base matrix size
            size_variance: Variance factor (0.0 to 1.0)
            enable_dynamic_size: Whether to apply dynamic sizing

        Returns:
            Actual matrix size after applying variance
        """
        if not enable_dynamic_size or size_variance <= 0:
            return base_size

        variance_range = int(base_size * size_variance)
        if variance_range <= 0:
            return base_size

        variance = secrets.randbelow(2 * variance_range + 1) - variance_range
        actual_size = max(base_size // 2, base_size + variance)

        return actual_size

    @staticmethod
    def validate_common_parameters(
        matrix_size: int,
        validator_hotkey: str,
        size_variance: float,
        min_size: int = 1,
        max_size: int = 100000,
        max_variance: float = 1.0,
    ) -> None:
        """
        Validate common challenge parameters

        Args:
            matrix_size: Matrix size to validate
            validator_hotkey: Validator hotkey to validate
            size_variance: Size variance to validate
            min_size: Minimum allowed matrix size
            max_size: Maximum allowed matrix size
            max_variance: Maximum allowed size variance

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(matrix_size, int) or not (
            min_size <= matrix_size <= max_size
        ):
            raise ValueError(
                f"matrix_size must be an integer between {min_size} "
                f"and {max_size}, got {matrix_size}"
            )

        if not validator_hotkey or not isinstance(validator_hotkey, str):
            raise ValueError("validator_hotkey must be a non-empty string")

        if not isinstance(size_variance, (int, float)) or not (
            0.0 <= size_variance <= max_variance
        ):
            raise ValueError(
                f"size_variance must be a number between 0.0 and {max_variance}, got {size_variance}"
            )
