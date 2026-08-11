"""
Metagraph Cache Service
Maintains a periodically refreshed, in-memory snapshot of the subnet metagraph.

Design:
- Background task refreshes `hotkeys` snapshot at a fixed interval from config
- Business code reads from this cache: no direct sync nor TTL logic at call sites
- Provides helpers: get_hotkeys(), is_member(), get_uid(), wait_until_ready()
"""

import asyncio
import time
from typing import List, Optional

import bittensor as bt

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.services.subtensor_access import SubtensorAccessGuard


class MetagraphCache:
    def __init__(
        self,
        subtensor_guard: SubtensorAccessGuard,
        netuid: int,
        metagraph: bt.Metagraph,
        config: ConfigManager,
    ) -> None:
        self._subtensor_guard = subtensor_guard
        self.subtensor = subtensor_guard.subtensor
        self.netuid = int(netuid)
        self._metagraph = metagraph  # shared instance; we only sync here
        self.config = config

        self._sync_interval = self.config.get_positive_number(
            "metagraph.sync_interval", int
        )

        # Snapshot state
        self._hotkeys: List[str] = []

        # Runtime
        self._is_running: bool = False
        self._sync_task: Optional[asyncio.Task] = None
        self._ready_event = asyncio.Event()
        self._last_sync_ts: float = 0.0

    async def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        # Perform an immediate sync to accelerate readiness
        try:
            await self._sync_once()
        except Exception as e:
            bt.logging.warning(f"⚠️ Initial metagraph sync failed | error={e}")

        self._sync_task = asyncio.create_task(self._background_loop())
        bt.logging.debug("MetagraphCache started")

    async def stop(self) -> None:
        if not self._is_running:
            return
        self._is_running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        bt.logging.debug("MetagraphCache stopped")

    async def wait_until_ready(self, timeout: Optional[float] = 120.0) -> None:
        if self._hotkeys:
            return
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise RuntimeError("Metagraph unavailable within timeout") from e

    def get_hotkeys(self) -> List[str]:
        return list(self._hotkeys)

    def is_member(self, hotkey: Optional[str]) -> bool:
        if not isinstance(hotkey, str) or not hotkey:
            return False
        return hotkey in self._hotkeys

    def get_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self._hotkeys.index(hotkey)
        except ValueError:
            return None

    async def _background_loop(self) -> None:
        while self._is_running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.warning(f"⚠️ Metagraph cache sync error | error={e}")

    async def _sync_once(self) -> None:
        sync_start = time.monotonic()
        # Reuse same metagraph instance, but sync it here only
        await self._subtensor_guard.run(
            lambda: self._metagraph.sync(subtensor=self.subtensor)
        )
        # Take a snapshot of hotkeys after sync
        hk = getattr(self._metagraph, "hotkeys", None)
        if isinstance(hk, list):
            self._hotkeys = list(hk)
        else:
            self._hotkeys = []

        if self._hotkeys and not self._ready_event.is_set():
            self._ready_event.set()

        self._last_sync_ts = time.time()
        elapsed = time.monotonic() - sync_start
        bt.logging.debug(
            f"Metagraph cache refreshed | miners={len(self._hotkeys)} elapsed={elapsed:.2f}s"
        )
