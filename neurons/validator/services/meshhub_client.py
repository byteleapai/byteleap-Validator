"""MeshHub WebSocket Client"""

import asyncio
import base64
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import bittensor as bt
from sqlalchemy import tuple_

from neurons.shared.config.config_manager import ConfigManager
from neurons.shared.crypto import CryptoManager
from neurons.validator.models.database import DatabaseManager

# Network/performance batching limits
SCORE_REPORT_BATCH_SIZE = 50  # workerScores per SCORE_REPORT
RESOURCE_REPORT_MAX_WORKERS = 50  # workers per RESOURCE_REPORT


@dataclass
class _SessionState:
    session_id: str
    k_cs: bytes
    k_sc: bytes
    server_hotkey: str
    seq_out: int = 0


@dataclass
class _EnrollTokenCacheEntry:
    token: Optional[str]
    enrollment_url: Optional[str]
    expires_at_iso: Optional[str]
    cache_valid_until: datetime
    pending: bool = False


class MeshHubClient:
    """WebSocket client for MeshHub with session encryption."""

    ENROLL_TOKEN_PENDING_TTL_SECONDS = 60
    ENROLL_TOKEN_RETRY_SECONDS = 5
    ENROLL_TOKEN_CACHE_OFFSET = timedelta(minutes=5)

    @staticmethod
    def validate_config(config: ConfigManager) -> None:
        """Validate MeshHub-related configuration (fail-fast).

        Raises KeyError/ValueError on invalid config. Intended to be the single
        source of truth for MeshHub config validation and reusable by startup.
        """

        ws_url = config.get_non_empty_string("meshhub.ws_url")
        access_token = config.get_non_empty_string("meshhub.access_token")

        caps = config.get_list("meshhub.capabilities", min_length=1)
        if not all(isinstance(c, str) and c.strip() for c in caps):
            bt.logging.error(
                "❌ Config error | meshhub.capabilities must be non-empty strings"
            )
            raise ValueError("meshhub.capabilities must be non-empty strings")

        reconnect_delay = config.get_positive_number(
            "meshhub.reconnect_delay_seconds", int
        )
        if reconnect_delay < 1:
            bt.logging.error(
                "❌ Config error | meshhub.reconnect_delay_seconds must be >= 1"
            )
            raise ValueError("meshhub.reconnect_delay_seconds must be >= 1")

        hb_interval = config.get_positive_number(
            "meshhub.heartbeat_interval_seconds", int
        )
        if hb_interval < 1:
            bt.logging.error(
                "❌ Config error | meshhub.heartbeat_interval_seconds must be >= 1"
            )
            raise ValueError("meshhub.heartbeat_interval_seconds must be >= 1")

        res_interval = config.get_positive_number(
            "meshhub.resource_report_interval_seconds", int
        )
        if res_interval < 5:
            bt.logging.error(
                "❌ Config error | meshhub.resource_report_interval_seconds must be >= 5"
            )
            raise ValueError("meshhub.resource_report_interval_seconds must be >= 5")

    def __init__(
        self,
        wallet: bt.Wallet,
        config: ConfigManager,
        db_manager: DatabaseManager,
        on_fatal: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.wallet = wallet
        self.config = config
        self.db = db_manager

        # Validate configuration before loading values
        self.validate_config(config)

        # Load validated configuration values
        self.ws_url = self.config.get_non_empty_string("meshhub.ws_url")
        self.access_token = self.config.get_non_empty_string("meshhub.access_token")
        self.capabilities = self.config.get_list("meshhub.capabilities", min_length=1)
        self.reconnect_delay = self.config.get_positive_number(
            "meshhub.reconnect_delay_seconds", int
        )
        self.heartbeat_interval = self.config.get_positive_number(
            "meshhub.heartbeat_interval_seconds", int
        )
        self.resource_report_interval = self.config.get_positive_number(
            "meshhub.resource_report_interval_seconds", int
        )

        self.crypto = CryptoManager(self.wallet)
        self.session: Optional[_SessionState] = None
        self._enroll_token_cache: Dict[str, _EnrollTokenCacheEntry] = {}
        self._enroll_token_lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._hb_task: Optional[asyncio.Task] = None
        self._rs_task: Optional[asyncio.Task] = None

        self._ws = None
        self._ws_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._start_time_ms = int(asyncio.get_event_loop().time() * 1000)
        # Wall-clock startup time (UTC, naive) for DB window filtering
        try:
            self._start_time_utc = datetime.utcnow()
        except Exception:
            self._start_time_utc = None
        # Client version from version.txt at repository root
        self._client_version = self._load_client_version()
        self._on_fatal = on_fatal
        # No outbound error correlation state (handled by MeshHub)

        # Incremental resource reporting state
        # Per-worker last hashes: state and hardware
        self._worker_state_hash: Dict[str, str] = {}
        self._worker_hw_hash: Dict[str, str] = {}
        # Last successful resource report time (UTC)
        self._last_resource_report_utc: Optional[datetime] = None
        # Whether we have sent an initial full snapshot in this process
        self._initial_resource_full_sent: bool = False

    def _now_ms(self) -> int:
        """Wall-clock epoch milliseconds in UTC."""
        return int(time.time() * 1000)

    def _utc_now_iso(self) -> str:
        """UTC timestamp in ISO-8601 with 'Z' suffix (UTC)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _utc_now(self) -> datetime:
        """Current UTC time as timezone-aware datetime."""
        return datetime.now(timezone.utc)

    def _to_iso_utc(self, dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to UTC ISO-8601 string with 'Z' offset (Instant-compatible)."""
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    async def start_blocking_initial(self) -> None:
        """Perform a blocking connect+handshake; exit on missing token or auth failure."""
        token = (self.access_token or "").strip()
        if not token:
            bt.logging.error("❌ MeshHub token missing; abort startup")
            raise SystemExit(1)
        await self._connect_once_blocking()

        await self.start()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._hb_task = asyncio.create_task(self._heartbeat_loop())
        self._rs_task = asyncio.create_task(self._resource_report_loop())
        bt.logging.info(f"🚀 MeshHub client started | url={self.ws_url}")

    async def stop(self) -> None:
        """Stop client quickly and unblock any pending websocket recv."""
        self._stop.set()

        for t in (self._hb_task, self._rs_task):
            if t:
                t.cancel()
        for t in (self._hb_task, self._rs_task):
            if t:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        try:
            async with self._ws_lock:
                ws = self._ws
                self._ws = None
            if ws is not None:
                try:
                    await asyncio.wait_for(ws.close(), timeout=2.0)
                except Exception:
                    pass
        except Exception:
            pass

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                try:
                    self._task.cancel()
                    await self._task
                except Exception:
                    pass

        bt.logging.info("MeshHub client stopped")

    async def _run_loop(self) -> None:
        import websockets

        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url, max_size=10 * 1024 * 1024
                ) as ws:
                    async with self._ws_lock:
                        self._ws = ws
                    await self._handshake(ws)
                    await self._recv_loop(ws)
            except Exception as e:
                bt.logging.error(f"❌ MeshHub connection error | error={e}")
            finally:
                async with self._ws_lock:
                    self._ws = None

            if self._stop.is_set():
                break
            await asyncio.sleep(self.reconnect_delay)

    async def _connect_once_blocking(self) -> None:
        import websockets

        try:
            async with websockets.connect(self.ws_url, max_size=10 * 1024 * 1024) as ws:
                await self._handshake(ws)
        except SystemExit:
            raise
        except Exception as e:
            bt.logging.error(f"❌ MeshHub initial connect failed | error={e}")
            raise SystemExit(1)

    async def _handshake(self, ws) -> None:

        client_pub_b64, client_priv_bytes = self.crypto.begin_handshake()
        client_nonce = self.crypto.generate_nonce()
        client_nonce_b64 = base64.b64encode(client_nonce).decode("ascii")

        payload = {
            "validatorHotkey": self.wallet.hotkey.ss58_address,
            "accessToken": self.access_token,
            # Project version sourced from version.txt at repo root
            "clientVersion": self._client_version,
            "capabilities": list(self.capabilities or []),
            "clientPublicKey": client_pub_b64,
            "clientNonce": client_nonce_b64,
        }
        message = {
            "type": "MESH_SESSION_INIT_V1",
            "timestamp": self._now_ms(),
            "data": payload,
        }

        await ws.send(json.dumps(message))
        bt.logging.debug("MeshHub handshake sent")

        raw = await ws.recv()
        ack = json.loads(raw)
        if ack.get("type") == "MESH_ERROR_V1":
            code = (ack.get("data") or {}).get("code")
            # if code == 4001:
            #     bt.logging.error("❌ MeshHub auth failed | code=4001 invalid_token")
            #     if self._on_fatal:
            #         self._on_fatal("meshhub_auth_invalid")
            #     raise RuntimeError("meshhub_auth_invalid")
            raise RuntimeError(f"MeshHub error during handshake: code={code}")
        if ack.get("type") != "MESH_SESSION_INIT_RESPONSE_V1":
            raise RuntimeError("MeshHub handshake failed: unexpected response type")

        data = ack.get("data") or {}
        session_id = data.get("sessionId")
        server_pub_b64 = data.get("validatorEphemeralPublicKey")
        server_nonce_b64 = data.get("serverNonce")
        server_hotkey = data.get("serverHotkey")

        if not (session_id and server_pub_b64 and server_nonce_b64 and server_hotkey):
            raise RuntimeError("MeshHub handshake failed: missing fields")

        server_nonce = base64.b64decode(server_nonce_b64.encode("ascii"))

        k_cs, k_sc = self.crypto.complete_handshake(
            our_private_key_bytes=client_priv_bytes,
            our_eph_pub_b64=client_pub_b64,
            peer_eph_pub_b64=server_pub_b64,
            client_nonce=client_nonce,
            server_nonce=server_nonce,
            peer_hotkey=server_hotkey,
        )

        self.session = _SessionState(
            session_id=session_id,
            k_cs=k_cs,
            k_sc=k_sc,
            server_hotkey=server_hotkey,
            seq_out=0,
        )
        bt.logging.info(f"🔐 MeshHub session established | id={session_id}")

    async def _recv_loop(self, ws) -> None:
        while not self._stop.is_set():
            try:
                raw = await ws.recv()
            except asyncio.CancelledError:
                return
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "MESH_ERROR_V1":
                data = msg.get("data") or {}
                code = data.get("code")
                if code in (4002, 4011, 4020):
                    bt.logging.warning("⚠️ Session expired; re-handshake")
                    self.session = None
                    raise RuntimeError("session_expired")
                if code == 4001:
                    # Authentication invalid; do not exit, trigger reconnect
                    bt.logging.error("❌ MeshHub auth invalid during session; retrying")
                    self.session = None
                    raise RuntimeError("meshhub_auth_invalid")
                bt.logging.warning(f"⚠️ MeshHub error | code={code}")
                continue
            if msg_type in ("MESH_SESSION_INIT_V1", "MESH_SESSION_INIT_RESPONSE_V1"):
                bt.logging.debug(f"MeshHub control message | type={msg_type}")
                continue

            decrypted = self._decrypt_inbound(msg)
            if not decrypted:
                continue
            dtype = msg_type
            data = decrypted

            if dtype == "MESH_LEASE_PUBLISH_V1":
                await self._handle_lease_publish(data)
            elif dtype == "MESH_CONFIG_UPDATE_V1":
                await self._handle_config_update(ws, msg, data)
            elif dtype == "MESH_TASK_PUBLISH_V1":
                await self._handle_task_publish(ws, msg, data)
            elif dtype == "MESH_VMGW_ENROLL_TOKEN_RESPONSE_V1":
                await self._handle_enroll_token_response(data)
            else:
                bt.logging.debug(f"MeshHub unknown type | type={dtype}")

    def _decrypt_inbound(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.session:
            return None
        payload = msg.get("encrypted")
        if not payload:
            bt.logging.error("❌ Encrypted payload missing for MeshHub message")
            return None

        package = {
            "ver": payload.get("version"),
            "session_id": payload.get("sessionId"),
            "seq": payload.get("sequence"),
            "ciphertext": payload.get("ciphertext"),
            "sender": payload.get("sender"),
            "recipient": payload.get("recipient"),
            "synapse_type": payload.get("messageType"),
        }
        try:
            plain, session_id, seq = self.crypto.decrypt_with_session(
                json.dumps(package),
                session_key=self.session.k_sc,
                expected_sender=self.session.server_hotkey,
                expected_recipient=self.wallet.hotkey.ss58_address,
                synapse_type=payload.get("messageType"),
            )
            return plain if isinstance(plain, dict) else {}
        except Exception as e:
            bt.logging.error(f"❌ MeshHub decrypt error | error={e}")
            return None

    async def _handle_lease_publish(self, data: Dict[str, Any]) -> None:
        scores = data.get("workerScores") or []
        try:
            with self.db.get_session() as session:
                updated, changes = self.db.apply_meshhub_lease_scores(session, scores)

                if changes:

                    def _fmt(v: Any) -> str:
                        try:
                            return f"{float(v):.4f}"
                        except Exception:
                            return str(v)

                    change_str = ",".join(
                        f"{c.get('workerKey')} {_fmt(c.get('from'))}->{_fmt(c.get('to'))}"
                        for c in changes
                    )
                    bt.logging.info(
                        f"✅ Lease sync | updated={updated} changes=[{change_str}]"
                    )
                else:
                    bt.logging.info(f"✅ Lease sync | updated={updated}")
        except Exception as e:
            bt.logging.error(f"❌ Lease sync failed | error={e}")

    async def _handle_config_update(
        self, ws, msg: Dict[str, Any], data: Dict[str, Any]
    ) -> None:

        raw_payload = data.get("payload") or {}
        allowed_roots = ["validation", "weight_management"]

        if isinstance(raw_payload, dict) and any(
            r in raw_payload for r in allowed_roots
        ):
            overrides = raw_payload
        else:
            overrides = {}

        if not overrides:
            provided_keys = (
                list(raw_payload.keys()) if isinstance(raw_payload, dict) else []
            )
            bt.logging.error(
                f"❌ Config merge failed | reason=no_allowed_root allowed=validation,weight_management provided={provided_keys}"
            )
            ack_id = msg.get("messageId")
            if ack_id:
                await self._send_ack(
                    ws,
                    message_id=ack_id,
                    message_type="MESH_CONFIG_UPDATE_V1",
                    status="failed",
                    metadata={
                        "error": "no_allowed_root",
                        "allowed": ["validation", "weight_management"],
                        "provided_keys": provided_keys,
                    },
                )
            return
        try:

            roots = allowed_roots
            old_snapshots: Dict[str, Any] = {}
            for r in roots:
                try:
                    val = self.config.get(r)
                    old_snapshots[r] = (
                        json.loads(json.dumps(val)) if isinstance(val, dict) else val
                    )
                except Exception:
                    old_snapshots[r] = {}

            self.config.merge_overrides(overrides, roots)

            def diff_subset(old: Any, new: Any, subset: Any) -> Any:
                if not isinstance(subset, dict):

                    return new if old != new else None
                result: Dict[str, Any] = {}
                for k, sub in subset.items():
                    old_v = old.get(k) if isinstance(old, dict) else None
                    new_v = new.get(k) if isinstance(new, dict) else None
                    if isinstance(sub, dict):
                        child = diff_subset(
                            old_v if isinstance(old_v, dict) else {},
                            new_v if isinstance(new_v, dict) else {},
                            sub,
                        )
                        if child not in (None, {}, []):
                            result[k] = child
                    else:
                        if old_v != new_v:
                            result[k] = new_v
                return result

            diffs: Dict[str, Any] = {}
            if isinstance(overrides, dict):
                for r in roots:
                    if r in overrides:
                        try:
                            new_val = self.config.get(r)
                        except Exception:
                            new_val = {}
                        subset_src = overrides.get(r) or {}
                        diff_r = diff_subset(
                            old_snapshots.get(r, {}),
                            new_val if isinstance(new_val, dict) else {},
                            subset_src,
                        )
                        if diff_r not in (None, {}, []):
                            diffs[r] = diff_r

            try:
                updates_json = json.dumps(
                    diffs, ensure_ascii=False, separators=(",", ":")
                )
            except Exception:
                updates_json = str(diffs)
            bt.logging.info(f"✅ Config merged | updates={updates_json}")
            ack_id = msg.get("messageId")
            if ack_id:
                await self._send_ack(
                    ws,
                    message_id=ack_id,
                    message_type="MESH_CONFIG_UPDATE_V1",
                    status="accepted",
                )
        except Exception as e:
            bt.logging.error(f"❌ Config merge failed | error={e}")
            ack_id = msg.get("messageId")
            if ack_id:
                await self._send_ack(
                    ws,
                    message_id=ack_id,
                    message_type="MESH_CONFIG_UPDATE_V1",
                    status="failed",
                    metadata={"error": str(e)},
                )

    async def _handle_task_publish(
        self, ws, msg: Dict[str, Any], data: Dict[str, Any]
    ) -> None:

        task_key = data.get("taskKey")
        worker_key = data.get("workerKey")
        task_type = (data.get("taskType") or "vm_creation").strip()
        task_payload = data.get("payload") or {}
        priority = int(data.get("priority") or 0)
        ttl_ms = data.get("ttl")

        hotkey = None
        worker_id = None
        if isinstance(worker_key, str) and ":" in worker_key:
            hotkey, worker_id = worker_key.split(":", 1)

        expires_at = None
        if ttl_ms and isinstance(ttl_ms, int) and ttl_ms > 0:
            from datetime import datetime, timedelta

            expires_at = datetime.utcnow() + timedelta(milliseconds=ttl_ms)

        if not task_key:
            bt.logging.error("❌ Mesh task missing taskKey; ignored")
            return

        try:
            with self.db.get_session() as session:
                self.db.record_meshhub_task(
                    session=session,
                    task_id=task_key,
                    task_type=task_type,
                    task_config=task_payload,
                    priority=priority,
                    worker_id=worker_id,
                    hotkey=hotkey,
                    expires_at=expires_at,
                    status="pending",
                )
            bt.logging.info(f"✅ Mesh task stored | id={task_key} type={task_type}")
        except Exception as e:
            bt.logging.error(f"❌ Store mesh task failed | id={task_key} error={e}")

        ack_id = msg.get("messageId")
        if ack_id:
            await self._send_ack(
                ws,
                message_id=ack_id,
                message_type="MESH_TASK_PUBLISH_V1",
                status="accepted",
            )

    async def _send_ack(
        self,
        ws,
        message_id: str,
        message_type: str,
        status: str = "accepted",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ack = {
            "messageId": message_id,
            "messageType": message_type,
            "status": status,
            "metadata": metadata or {},
        }
        await self._send_encrypted(ws, "MESH_ACK_V1", ack)

    async def _send_encrypted(self, ws, msg_type: str, data: Dict[str, Any]) -> None:
        if not self.session:
            return

        try:
            async with self._send_lock:
                self.session.seq_out += 1
                package_json = self.crypto.encrypt_with_session(
                    plaintext=data,
                    session_id=self.session.session_id,
                    session_key=self.session.k_cs,
                    seq=self.session.seq_out,
                    sender_hotkey=self.wallet.hotkey.ss58_address,
                    recipient_hotkey=self.session.server_hotkey,
                    synapse_type=msg_type,
                )
                package = json.loads(package_json)
                encrypted_payload = {
                    "version": package.get("ver"),
                    "sessionId": package.get("session_id"),
                    "sequence": package.get("seq"),
                    "ciphertext": package.get("ciphertext"),
                    "sender": package.get("sender"),
                    "recipient": package.get("recipient"),
                    "messageType": package.get("synapse_type"),
                }

            message = {
                "type": msg_type,
                "timestamp": self._now_ms(),
                "encrypted": encrypted_payload,
            }
            await ws.send(json.dumps(message))
        except Exception as e:
            bt.logging.error(
                f"❌ MeshHub encrypt/send failed | type={msg_type} error={e}"
            )

    async def _send_encrypted_ws(self, msg_type: str, data: Dict[str, Any]) -> None:
        async with self._ws_lock:
            ws = self._ws
        if ws is None:
            return
        await self._send_encrypted(ws, msg_type, data)

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.session:
                    payload = self._build_heartbeat_payload()
                    await self._send_encrypted_ws("MESH_HEARTBEAT_V1", payload)
            except Exception:
                pass
            await asyncio.sleep(max(1, self.heartbeat_interval))

    async def _resource_report_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.session:
                    # Full on first run, then delta
                    full_mode = not self._initial_resource_full_sent
                    payload, changed, pending_updates = self._build_resource_report(
                        full=full_mode
                    )
                    if payload:
                        miners = payload.get("miners") or []
                        mode = payload.get("mode") or ("full" if full_mode else "delta")
                        since_iso = payload.get("since")

                        if miners:
                            for miners_batch in self._chunk_miners_by_workers(
                                miners, RESOURCE_REPORT_MAX_WORKERS
                            ):
                                batched_payload = {"miners": miners_batch, "mode": mode}
                                if since_iso:
                                    batched_payload["since"] = since_iso
                                await self._send_encrypted_ws(
                                    "MESH_RESOURCE_REPORT_V1", batched_payload
                                )
                        # Apply pending worker hash updates after send attempt
                        try:
                            for wk_key, h in (
                                pending_updates.get("workers") or {}
                            ).items():
                                if "state" in h:
                                    self._worker_state_hash[wk_key] = h["state"]
                                if "hardware" in h and h["hardware"] is not None:
                                    self._worker_hw_hash[wk_key] = h["hardware"]
                        except Exception:
                            pass

                        # Mark full sent and update watermark after send attempt
                        if full_mode:
                            self._initial_resource_full_sent = True
                        try:
                            self._last_resource_report_utc = datetime.utcnow()
                        except Exception:
                            self._last_resource_report_utc = None
            except Exception:
                pass
            await asyncio.sleep(max(5, self.resource_report_interval))

    def _chunk_miners_by_workers(
        self, miners: List[Dict[str, Any]], max_workers: int
    ) -> List[List[Dict[str, Any]]]:
        """Split miners array into batches with at most max_workers workers total per batch.

        Each batch contains miner entries with a subset of their workers.
        Miners without workers are omitted to reduce bandwidth.
        """
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_count = 0

        for miner in miners:
            workers = list(miner.get("workers") or [])
            if not workers:
                continue

            i = 0
            while i < len(workers):
                remaining = max_workers - current_count
                take = min(remaining, len(workers) - i)
                slice_workers = workers[i : i + take]

                miner_entry = {
                    "hotkey": miner.get("hotkey"),
                    "status": miner.get("status"),
                    "version": miner.get("version"),
                    "workers": slice_workers,
                }
                current_batch.append(miner_entry)
                current_count += take
                i += take

                if current_count >= max_workers:
                    batches.append(current_batch)
                    current_batch = []
                    current_count = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    def _build_heartbeat_payload(self) -> Dict[str, Any]:
        now_ms = int(asyncio.get_event_loop().time() * 1000)
        uptime_ms = max(0, now_ms - self._start_time_ms)

        active_miners = 0
        active_workers = 0
        total_tasks = 0
        success_tasks = 0
        pending_verifications = 0

        try:
            from neurons.validator.models.database import (MeshHubTask,
                                                           MinerInfo,
                                                           WorkerInfo)

            with self.db.get_session() as session:
                # Online miners and workers (soft-delete aware)
                active_miners = (
                    session.query(MinerInfo)
                    .filter(
                        MinerInfo.is_online.is_(True), MinerInfo.deleted_at.is_(None)
                    )
                    .count()
                )
                active_workers = (
                    session.query(WorkerInfo)
                    .filter(
                        WorkerInfo.is_online.is_(True), WorkerInfo.deleted_at.is_(None)
                    )
                    .count()
                )

                # Task window since process startup (fallback to all-time on missing startup time)
                q_total = session.query(MeshHubTask).filter(
                    MeshHubTask.deleted_at.is_(None)
                )
                q_success = session.query(MeshHubTask).filter(
                    MeshHubTask.deleted_at.is_(None)
                )
                if self._start_time_utc is not None:
                    q_total = q_total.filter(
                        MeshHubTask.created_at >= self._start_time_utc
                    )
                    q_success = q_success.filter(
                        MeshHubTask.created_at >= self._start_time_utc
                    )
                # Success criteria: status in {completed, success}
                q_success = q_success.filter(
                    MeshHubTask.status.in_(["completed", "success"])
                )

                total_tasks = q_total.count()
                success_tasks = q_success.count()

        except Exception:
            # Keep heartbeat resilient to DB issues
            pass

        success_rate = (
            1.0 if total_tasks == 0 else float(success_tasks) / float(total_tasks)
        )

        stats = {
            "uptime": uptime_ms,
            "activeWorkers": int(active_workers),
            "activeMiners": int(active_miners),
            "totalTasks": int(total_tasks),
            "successRate": float(success_rate),
            "pendingVerifications": int(pending_verifications),
        }

        return {
            # Use project version (same as handshake)
            "version": self._client_version,
            "statistics": stats,
        }

    def _stable_hash(self, obj: Any) -> str:
        try:
            data = json.dumps(
                obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except Exception:
            data = str(obj)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _build_worker_payload(
        self,
        w,
        hardware,
        gpu_devices,
        heartbeat,
        include_hardware: bool,
    ) -> Dict[str, Any]:
        gpu_list = [g.gpu_info for g in gpu_devices if g.gpu_info is not None]
        hardware_full = {
            "cpu": (
                hardware.cpu_info if hardware and hardware.cpu_info is not None else {}
            ),
            "memory": (
                hardware.memory_info
                if hardware and hardware.memory_info is not None
                else {}
            ),
            "storage": (
                hardware.storage_info
                if hardware and hardware.storage_info is not None
                else []
            ),
            "gpus": gpu_list,
            "mb_info": (
                hardware.motherboard_info
                if hardware and hardware.motherboard_info is not None
                else {}
            ),
        }

        # Latest utilization from most recent heartbeat
        avg_cpu = None
        avg_mem = None
        public_ip = None
        if heartbeat is not None:
            try:
                avg_cpu = (
                    float(heartbeat.cpu_usage)
                    if heartbeat.cpu_usage is not None
                    else None
                )
            except Exception:
                avg_cpu = None
            try:
                avg_mem = (
                    float(heartbeat.memory_usage)
                    if heartbeat.memory_usage is not None
                    else None
                )
            except Exception:
                avg_mem = None
            public_ip = heartbeat.public_ip

        net_obj: Dict[str, Any] = {}
        if public_ip:
            net_obj["public_ip"] = public_ip

        payload: Dict[str, Any] = {
            "workerKey": f"{w.hotkey}:{w.worker_id}",
            "workerId": w.worker_id,
            "workerName": (w.worker_name or None),
            "status": "ACTIVE" if w.is_online else "OFFLINE",
            "version": w.worker_version or None,
            "capabilities": w.capabilities or [],
            "leaseScore": w.lease_score or 0.0,
            "lastSeenAt": self._to_iso_utc(w.last_heartbeat),
            # stats and uptime are lightweight, always include
            "stats": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_mem,
                "avg_storage_usage": None,
            },
            # Uptime comes from worker hardware record
            "uptimeSeconds": (
                int(hardware.uptime_seconds)
                if hardware and hardware.uptime_seconds
                else None
            ),
            # Network object handled same as os_info: include object as-is
            "network": net_obj,
            "os_info": (
                hardware.system_info
                if hardware and hardware.system_info is not None
                else {}
            ),
        }
        if include_hardware:
            payload["hardware"] = hardware_full
        return payload

    def _build_worker_hashes(self, worker_payload: Dict[str, Any]) -> Dict[str, str]:
        # Build separate hashes for state and hardware
        state_basis = {
            "workerName": worker_payload.get("workerName"),
            "status": worker_payload.get("status"),
            "version": worker_payload.get("version"),
            "capabilities": sorted(worker_payload.get("capabilities") or []),
            "stats": worker_payload.get("stats") or {},
            "uptimeSeconds": worker_payload.get("uptimeSeconds"),
            "network": worker_payload.get("network") or {},
        }
        hw_obj = worker_payload.get("hardware")
        # When hardware is not present in payload, we compute hash from empty
        hw_basis = hw_obj if isinstance(hw_obj, dict) else {}
        return {
            "state": self._stable_hash(state_basis),
            "hardware": self._stable_hash(hw_basis),
        }

    def _build_resource_report(
        self, full: bool
    ) -> (Dict[str, Any], bool, Dict[str, Any]):
        """Build resource report payload.

        Returns (payload, changed). For delta mode, changed=False means no diffs found.
        """
        try:
            from neurons.validator.models.database import (GPUInventory,
                                                           HardwareInfo,
                                                           HeartbeatRecord,
                                                           MinerInfo,
                                                           WorkerInfo)

            miners_out: List[Dict[str, Any]] = []
            changed_any = False
            pending_updates: Dict[str, Any] = {"workers": {}}
            since_iso = (
                self._to_iso_utc(self._last_resource_report_utc) if not full else None
            )

            with self.db.get_session() as session:
                miner_rows = (
                    session.query(MinerInfo)
                    .filter(MinerInfo.deleted_at.is_(None))
                    .all()
                )

                if not miner_rows:
                    return (
                        {"miners": [], "mode": "full" if full else "delta"},
                        False,
                        {"workers": {}},
                    )

                hotkeys = [m.hotkey for m in miner_rows]
                workers = (
                    session.query(WorkerInfo)
                    .filter(
                        WorkerInfo.hotkey.in_(hotkeys),
                        WorkerInfo.deleted_at.is_(None),
                    )
                    .all()
                )

                workers_by_hotkey: Dict[str, List[WorkerInfo]] = defaultdict(list)
                for worker in workers:
                    workers_by_hotkey[worker.hotkey].append(worker)

                worker_pairs = list(
                    {(w.hotkey, w.worker_id) for w in workers if w.worker_id}
                )

                hardware_rows = (
                    session.query(HardwareInfo)
                    .filter(HardwareInfo.deleted_at.is_(None))
                    .filter(
                        tuple_(HardwareInfo.hotkey, HardwareInfo.worker_id).in_(
                            worker_pairs
                        )
                        if worker_pairs
                        else False
                    )
                    .all()
                )
                hardware_map = {
                    (row.hotkey, row.worker_id): row for row in hardware_rows
                }

                gpu_rows = (
                    session.query(GPUInventory)
                    .filter(GPUInventory.deleted_at.is_(None))
                    .filter(
                        tuple_(GPUInventory.hotkey, GPUInventory.worker_id).in_(
                            worker_pairs
                        )
                        if worker_pairs
                        else False
                    )
                    .all()
                )
                gpu_map: Dict[Tuple[str, str], List[GPUInventory]] = defaultdict(list)
                for gpu in gpu_rows:
                    gpu_map[(gpu.hotkey, gpu.worker_id)].append(gpu)

                heartbeat_rows = (
                    session.query(HeartbeatRecord)
                    .distinct(HeartbeatRecord.hotkey, HeartbeatRecord.worker_id)
                    .filter(HeartbeatRecord.deleted_at.is_(None))
                    .filter(
                        tuple_(HeartbeatRecord.hotkey, HeartbeatRecord.worker_id).in_(
                            worker_pairs
                        )
                        if worker_pairs
                        else False
                    )
                    .order_by(
                        HeartbeatRecord.hotkey,
                        HeartbeatRecord.worker_id,
                        HeartbeatRecord.created_at.desc(),
                    )
                    .all()
                )
                heartbeat_map = {(hb.hotkey, hb.worker_id): hb for hb in heartbeat_rows}

                for miner in miner_rows:
                    worker_list: List[Dict[str, Any]] = []
                    for worker in workers_by_hotkey.get(miner.hotkey, []):
                        wk_key = f"{worker.hotkey}:{worker.worker_id}"
                        cache_key = (worker.hotkey, worker.worker_id)

                        hardware = hardware_map.get(cache_key)
                        gpu_devices = gpu_map.get(cache_key, [])
                        heartbeat = heartbeat_map.get(cache_key)

                        full_payload = self._build_worker_payload(
                            worker,
                            hardware,
                            gpu_devices,
                            heartbeat,
                            include_hardware=True,
                        )
                        hashes = self._build_worker_hashes(full_payload)
                        state_changed = hashes["state"] != self._worker_state_hash.get(
                            wk_key
                        )
                        hw_changed = hashes["hardware"] != self._worker_hw_hash.get(
                            wk_key
                        )

                        if full or state_changed or hw_changed:
                            include_hw = full or hw_changed
                            if include_hw:
                                worker_payload = full_payload
                            else:
                                worker_payload = dict(full_payload)
                                worker_payload.pop("hardware", None)
                            worker_list.append(worker_payload)
                            pending_updates["workers"][wk_key] = {
                                "state": hashes["state"],
                                "hardware": hashes["hardware"],
                            }

                    miners_out.append(
                        {
                            "hotkey": miner.hotkey,
                            "status": "ACTIVE" if miner.is_online else "OFFLINE",
                            "version": miner.miner_version or None,
                            "workers": worker_list,
                        }
                    )
                    changed_any = changed_any or bool(worker_list)

            payload: Dict[str, Any] = {
                "miners": miners_out,
                "mode": "full" if full else "delta",
            }
            if since_iso:
                payload["since"] = since_iso
            return payload, changed_any, pending_updates
        except Exception as e:
            bt.logging.error(f"❌ Build resource report failed | error={e}")
            return (
                {"miners": [], "mode": "delta" if not full else "full"},
                False,
                {"workers": {}},
            )

    def _load_client_version(self) -> str:
        """Load project version from version.txt at repository root."""
        try:
            root = Path(__file__).resolve().parents[3]
            version_path = root / "version.txt"
            text = version_path.read_text(encoding="utf-8").strip()
            return text if text else "unknown"
        except Exception:
            return "unknown"

    def _cleanup_enroll_cache(self, now: Optional[datetime] = None) -> None:
        """Remove expired enroll token cache entries (caller holds lock)."""
        now = now or self._utc_now()
        expired = [
            key
            for key, entry in self._enroll_token_cache.items()
            if entry.cache_valid_until <= now
        ]
        for key in expired:
            self._enroll_token_cache.pop(key, None)

    @staticmethod
    def _parse_iso8601(value: str) -> datetime:
        """Parse ISO-8601 string with optional trailing 'Z'."""
        if not value:
            raise ValueError("timestamp is empty")
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    async def acquire_enroll_token(
        self, hotkey: str
    ) -> Tuple[str, Optional[Dict[str, str]]]:
        """
        Retrieve or request a VM gateway enroll token for a miner hotkey.

        Returns:
            state: one of {"ready", "pending", "unavailable"}
            payload: token data when state == "ready"
        """
        if not hotkey:
            return "unavailable", None

        async with self._enroll_token_lock:
            now = self._utc_now()
            self._cleanup_enroll_cache(now)
            entry = self._enroll_token_cache.get(hotkey)
            if entry and entry.cache_valid_until > now:
                if entry.pending:
                    return "pending", None
                return (
                    "ready",
                    {
                        "token": entry.token,
                        "expires_at": entry.expires_at_iso,
                        "enrollment_url": entry.enrollment_url,
                    },
                )

            if not self.session:
                return "unavailable", None

            pending_entry = _EnrollTokenCacheEntry(
                token=None,
                enrollment_url=None,
                expires_at_iso=None,
                cache_valid_until=now
                + timedelta(seconds=self.ENROLL_TOKEN_PENDING_TTL_SECONDS),
                pending=True,
            )
            self._enroll_token_cache[hotkey] = pending_entry

        try:
            bt.logging.debug(f"Requesting VMGW enroll token | hotkey={hotkey}")
            await self._send_enroll_token_request(hotkey)
        except Exception as e:
            bt.logging.error(
                f"❌ VMGW enroll token request send failed | hotkey={hotkey} | error={e}"
            )
            async with self._enroll_token_lock:
                current = self._enroll_token_cache.get(hotkey)
                if current and current.pending:
                    self._enroll_token_cache.pop(hotkey, None)
            return "unavailable", None

        return "pending", None

    async def _send_enroll_token_request(self, hotkey: str) -> None:
        """Send enroll token request to MeshHub over encrypted channel."""
        payload = {"hotkey": hotkey}
        await self._send_encrypted_ws("MESH_VMGW_ENROLL_TOKEN_REQUEST_V1", payload)

    async def _handle_enroll_token_response(self, data: Dict[str, Any]) -> None:
        """Process enroll token response pushed from MeshHub."""
        hotkey = (data.get("hotkey") or "").strip()
        token = data.get("token")
        expires_at = data.get("expiresAt")
        enrollment_url = data.get("enrollmentUrl")

        if not hotkey:
            bt.logging.warning("⚠️ VMGW enroll token response missing hotkey")
            return

        if not token or not expires_at or not enrollment_url:
            bt.logging.warning(
                f"⚠️ VMGW enroll token response incomplete | hotkey={hotkey}"
            )
            async with self._enroll_token_lock:
                current = self._enroll_token_cache.get(hotkey)
                if current and current.pending:
                    self._enroll_token_cache.pop(hotkey, None)
            return

        try:
            expires_at_dt = self._parse_iso8601(expires_at)
        except Exception as e:
            bt.logging.warning(
                f"⚠️ VMGW enroll token expiry parse failed | hotkey={hotkey} | error={e}"
            )
            async with self._enroll_token_lock:
                current = self._enroll_token_cache.get(hotkey)
                if current and current.pending:
                    self._enroll_token_cache.pop(hotkey, None)
            return

        now = self._utc_now()
        cache_valid_until = expires_at_dt - self.ENROLL_TOKEN_CACHE_OFFSET
        if cache_valid_until <= now:
            cache_valid_until = expires_at_dt

        entry = _EnrollTokenCacheEntry(
            token=token,
            enrollment_url=enrollment_url,
            expires_at_iso=expires_at,
            cache_valid_until=cache_valid_until,
            pending=False,
        )

        async with self._enroll_token_lock:
            self._enroll_token_cache[hotkey] = entry

        bt.logging.info(
            f"✅ VMGW enroll token cached | hotkey={hotkey} expires_at={expires_at}"
        )

    async def publish_score_report(
        self,
        effective_at: Optional[str] = None,
        worker_scores: Optional[List[Dict[str, Any]]] = None,
        miner_scores: Optional[List[Dict[str, Any]]] = None,
        global_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish a score report on demand (event-driven)."""
        if not self.session:
            return
        if not worker_scores and not miner_scores:
            return

        worker_scores = worker_scores or []
        miner_scores = miner_scores or []

        # First packet: only minerScores (+ optional globalStats/effectiveAt)
        if miner_scores:
            first_payload: Dict[str, Any] = {
                "minerScores": miner_scores,
                "globalStats": global_stats,
                "effectiveAt": effective_at,
                "timestamp": self._utc_now_iso(),
                "version": "1.0",
            }
            await self._send_encrypted_ws("MESH_SCORE_REPORT_V1", first_payload)

        # Subsequent packets: chunked workerScores only
        total = len(worker_scores)
        sent = 0
        while sent < total:
            chunk = worker_scores[sent : sent + SCORE_REPORT_BATCH_SIZE]
            payload: Dict[str, Any] = {
                "workerScores": chunk,
                "effectiveAt": effective_at,
                "timestamp": self._utc_now_iso(),
                "version": "1.0",
            }
            await self._send_encrypted_ws("MESH_SCORE_REPORT_V1", payload)
            sent += len(chunk)
