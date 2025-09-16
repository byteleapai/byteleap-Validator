"""
Synapse Validation and Processing
Handles synapse validation and address extraction logic
"""

from typing import Any

import bittensor as bt


class SynapseHandler:
    """Handles synapse validation and processing"""

    def validate_synapse(self, synapse: Any) -> bool:
        """
        Validate synapse has encrypted request data

        Args:
            synapse: Synapse to validate

        Returns:
            True if synapse has encrypted request
        """
        if not getattr(synapse, "request", None):
            bt.logging.error("⚠️ Synapse missing request data")
            return False
        return True

    def get_peer_address(self, synapse: Any) -> str:
        """
        Extract peer address from synapse

        Args:
            synapse: Synapse to extract address from

        Returns:
            Peer address string
        """
        ip = (
            getattr(synapse.dendrite, "ip", "unknown")
            if synapse.dendrite
            else "unknown"
        )
        port = (
            getattr(synapse.dendrite, "port", "unknown")
            if synapse.dendrite
            else "unknown"
        )
        return f"{ip}:{port}"

    def is_response_valid(self, synapse_response: Any) -> bool:
        """
        Check if synapse response indicates successful communication

        Args:
            synapse_response: Response to validate

        Returns:
            True if response is valid
        """
        if synapse_response is None:
            return False

        if isinstance(synapse_response, str):
            return False

        if hasattr(synapse_response, "dendrite"):
            dendrite = synapse_response.dendrite
            if dendrite and hasattr(dendrite, "status_code"):
                if dendrite.status_code != 200:
                    return False

            if hasattr(synapse_response, "response"):
                return synapse_response.response is not None

        return True

    def get_response_error(self, synapse_response: Any) -> str:
        """
        Extract error message from failed response

        Args:
            synapse_response: Failed response

        Returns:
            Error message string
        """
        if synapse_response is None:
            return "No response received"

        if isinstance(synapse_response, str):
            return "Connection failed"

        if hasattr(synapse_response, "dendrite"):
            dendrite = synapse_response.dendrite
            if dendrite and hasattr(dendrite, "status_code"):
                if dendrite.status_code != 200:
                    return getattr(
                        dendrite, "status_message", f"HTTP {dendrite.status_code}"
                    )

        return "Unknown error"
