"""
Base Communication Service
Common functionality shared between miner and validator communication services
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import bittensor as bt

from neurons.shared.communication_logging import CommunicationLogger
from neurons.shared.protocols import CommunicationResult, ErrorCodes
from neurons.shared.synapse import SynapseHandler


class BaseCommunicationService:
    """Base class for communication services with common functionality"""

    def __init__(self, wallet: bt.wallet, config, component_name: str):
        """Initialize common communication components"""
        # Core components
        self.wallet = wallet
        self.config = config
        self.synapse_handler = SynapseHandler()
        self.logger = CommunicationLogger(component_name)

        bt.logging.info(
            f"🚀 {component_name} comm initialized | wallet={self.wallet.name}"
        )

    def communication_operation(self, operation_name: str):
        """
        Decorator for communication operations with unified error handling and logging

        Args:
            operation_name: Name of the operation for logging
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> CommunicationResult:
                start_time = time.time()
                result = CommunicationResult(success=False)

                try:
                    # Execute the wrapped function
                    response = await func(*args, **kwargs)

                    # Handle different return types
                    if isinstance(response, CommunicationResult):
                        result = response
                    elif isinstance(response, tuple) and len(response) == 2:
                        data, error_code = response
                        result.success = error_code == 0
                        result.data = data
                        result.error_code = error_code
                    else:
                        result.success = True
                        result.data = response

                except Exception as e:
                    result.error_code = ErrorCodes.INVALID_RESPONSE
                    result.error_message = f"{operation_name} failed: {str(e)}"
                    bt.logging.error(f"❌ {operation_name} error | error={e}")

                finally:
                    result.processing_time_ms = (time.time() - start_time) * 1000

                    # Log completion
                    if result.success:
                        bt.logging.debug(
                            f"✅ {operation_name} done | time={result.processing_time_ms:.1f}ms"
                        )
                    else:
                        bt.logging.error(
                            f"❌ {operation_name} fail | err={result.error_message}"
                        )

                return result

            return wrapper

        return decorator

    def get_peer_info(self, synapse_or_hotkey) -> Dict[str, str]:
        """Extract peer information for logging"""
        if hasattr(synapse_or_hotkey, "dendrite"):
            # It's a synapse
            hotkey = (
                synapse_or_hotkey.dendrite.hotkey
                if synapse_or_hotkey.dendrite
                else "unknown"
            )
            address = self.synapse_handler.get_peer_address(synapse_or_hotkey)
            return {"hotkey": hotkey, "address": address}
        else:
            # It's a hotkey string
            return {"hotkey": str(synapse_or_hotkey), "address": "unknown"}
