import asyncio
from datetime import datetime
from datetime import time as dtime
from datetime import timedelta
from typing import Dict, Optional

import bittensor as bt

from neurons.validator.models.database import DatabaseManager


class DataCleanupService:
    """
    Periodic database data retention service.

    - Runs daily at 00:00 UTC
    - Keeps only the last N days of data (default: 7)
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        retention_days: int = 7,
        min_heartbeat_hours: Optional[int] = None,
    ):
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        self.db = database_manager
        self.retention_days = retention_days
        self.min_heartbeat_hours = (
            int(min_heartbeat_hours) if isinstance(min_heartbeat_hours, int) else None
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            bt.logging.warning("⚠️ Data cleanup already running")
            return
        self._running = True
        # Start background scheduler
        self._task = asyncio.create_task(self._loop())
        bt.logging.info("🧹 Data cleanup scheduler started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        bt.logging.info("🧹 Data cleanup scheduler stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                # Sleep until next UTC midnight
                sleep_seconds = self._seconds_until_next_utc_midnight()
                await asyncio.sleep(sleep_seconds)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"❌ Data cleanup loop error | error={e}")
                # Backoff retry in case of transient DB errors
                await asyncio.sleep(60)

    async def run_once(self) -> None:
        """Perform one cleanup pass with retention policy."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        try:
            with self.db.get_session() as session:
                deleted = self.db.cleanup_old_data(
                    session,
                    self.retention_days,
                    min_heartbeat_hours=self.min_heartbeat_hours,
                )
                total = sum(deleted.values())
                # Log concise aggregation and per-table metrics at DEBUG
                if self.min_heartbeat_hours and self.min_heartbeat_hours > 0:
                    bt.logging.info(
                        f"🧹 DB cleanup | total={total} days={self.retention_days} hb_hours={self.min_heartbeat_hours} cutoff={cutoff.isoformat()}"
                    )
                else:
                    bt.logging.info(
                        f"🧹 DB cleanup | total={total} days={self.retention_days} cutoff={cutoff.isoformat()}"
                    )
                for table, count in (deleted or {}).items():
                    bt.logging.debug(f"cleanup_detail | table={table} deleted={count}")
        except Exception as e:
            bt.logging.error(f"❌ DB cleanup error | error={e}")

    @staticmethod
    def _seconds_until_next_utc_midnight() -> float:
        now_utc = datetime.utcnow()
        tomorrow_utc = now_utc.date() + timedelta(days=1)
        next_midnight_utc = datetime.combine(tomorrow_utc, dtime.min)
        delta = (next_midnight_utc - now_utc).total_seconds()
        return max(1.0, delta)
