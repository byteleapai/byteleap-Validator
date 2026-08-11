"""
Subtensor access guard
Serializes access to a shared bt.Subtensor instance to avoid thread contention.
"""

import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


class SubtensorAccessGuard:
    """Coordinates serialized access to a shared subtensor client."""

    def __init__(self, subtensor):
        if subtensor is None:
            raise ValueError("subtensor cannot be None")
        self._subtensor = subtensor
        self._lock = asyncio.Lock()

    @property
    def subtensor(self):
        """Expose underlying subtensor for read-only metadata."""
        return self._subtensor

    async def run(self, func: Callable[[], T]) -> T:
        """
        Execute a blocking subtensor call inside the shared lock.

        Args:
            func: Callable that performs the actual subtensor work on a worker thread.

        Returns:
            Result of the callable.
        """
        if not callable(func):
            raise ValueError("func must be callable")
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func)
