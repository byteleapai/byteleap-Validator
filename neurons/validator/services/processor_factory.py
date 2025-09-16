"""
Processor Factory
Creates concrete SynapseProcessor instances for specific synapse types
"""

from typing import Any

from neurons.validator.synapse_processor import SynapseProcessor


class ValidatorProcessorFactory:
    """Factory for creating validator-specific synapse processors"""

    def __init__(self, communicator: Any):
        self.communicator = communicator

    def create_heartbeat_processor(self, handler_func) -> SynapseProcessor:
        from neurons.shared.protocols import HeartbeatData, HeartbeatSynapse

        class HeartbeatProcessor(SynapseProcessor):
            def __init__(self, communicator, handler):
                super().__init__(communicator)
                self.handler = handler

            @property
            def synapse_type(self):
                return HeartbeatSynapse

            @property
            def request_class(self):
                return HeartbeatData

            async def process_request(self, request_data, peer_hotkey):
                return await self.handler(request_data, peer_hotkey)

        return HeartbeatProcessor(self.communicator, handler_func)

    def create_task_processor(self, handler_func) -> SynapseProcessor:
        from neurons.shared.protocols import TaskRequest, TaskSynapse

        class TaskProcessor(SynapseProcessor):
            def __init__(self, communicator, handler):
                super().__init__(communicator)
                self.handler = handler

            @property
            def synapse_type(self):
                return TaskSynapse

            @property
            def request_class(self):
                return TaskRequest

            async def process_request(self, request_data, peer_hotkey):
                return await self.handler(request_data, peer_hotkey)

        return TaskProcessor(self.communicator, handler_func)
