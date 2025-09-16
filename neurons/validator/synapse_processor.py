"""
Validator Synapse Processing
Abstract base for processing specific synapse types in validator
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Type

from neurons.shared.protocols import EncryptedSynapse


class SynapseProcessor(ABC):
    """Abstract base for processing specific synapse types"""

    def __init__(self, communicator):
        self.communicator = communicator

    @property
    @abstractmethod
    def synapse_type(self) -> Type[EncryptedSynapse]:
        """Type of synapse this processor handles"""
        pass

    @property
    @abstractmethod
    def request_class(self) -> Type:
        """Class to deserialize decrypted request data"""
        pass

    @abstractmethod
    async def process_request(
        self, request_data: Any, peer_hotkey: str
    ) -> Tuple[Any, int]:
        """
        Process decrypted request data

        Returns:
            Tuple of (response_data, error_code)
        """
        pass
