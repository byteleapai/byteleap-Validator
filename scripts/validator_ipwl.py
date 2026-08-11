#!/usr/bin/env python3
"""
Maintain local iptables allowlist from Bittensor metagraph validators.

- Select validators by: has validator_permit AND stake > --stake-limit
- Extract IPv4 Axon addresses only
- Maintain a dedicated iptables chain per port: BTMG_ALLOW_<PORT>
  - Attach to INPUT: -p tcp --dport <port> -j BTMG_ALLOW_<PORT>
  - Flush + repopulate on each sync

Usage example:
  python scripts/iptables_allowlist_from_metagraph.py \
    --uid 397 --network finney --stake-limit 1000 --port 9091 -i 60
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Iterable, List, Optional, Set, Tuple

import bittensor as bt

DRY_RUN = False


def _log_cmd(cmd: List[str], op: str) -> None:
    # INFO level per logging standards (with concise k=v fields).
    # Rely on bt.logging timestamp; do not add our own.
    if op == "modify":
        bt.logging.info(
            f"🧰 iptables | op=modify dry_run={DRY_RUN} cmd={' '.join(cmd)}"
        )
    else:
        bt.logging.info(f"iptables | op=check dry_run={DRY_RUN} cmd={' '.join(cmd)}")


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a shell command, return (code, stdout, stderr)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()


def _run_readonly(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a read-only command even in dry-run (to detect state when possible).

    Falls back to simulated success if execution is not permitted.
    """
    # Always log the command
    _log_cmd(cmd, op="check")

    # In dry-run: do not execute, simulate sensible return codes
    if DRY_RUN:
        # Heuristic: treat state checks as "not present" to exercise flow
        rc = 0
        if len(cmd) >= 2 and cmd[0] == "iptables":
            flag = cmd[1]
            if flag in ("-S", "-C"):
                rc = 1  # trigger creation/insertion paths
            elif flag == "-V":
                rc = 0  # pretend available
        return rc, "", ""

    # Non-dry-run: execute
    return _run(cmd)


def _emit_modify(cmd: List[str]) -> Tuple[int, str, str]:
    """Emit a modifying iptables command.

    - In dry-run: print only and return success
    - Otherwise: execute
    """
    # Always log the command
    _log_cmd(cmd, op="modify")
    if DRY_RUN:
        return 0, "", ""
    return _run(cmd)


def _check_root() -> bool:
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except Exception:
        # Non-POSIX; assume not root
        return False


def _check_iptables_available() -> bool:
    code, out, err = _run_readonly(["iptables", "-V"])
    if code != 0 and DRY_RUN:
        bt.logging.warning("⚠️ iptables not found; continuing in dry-run")
        return True
    return code == 0


def _ensure_chain(chain: str) -> None:
    code, _, _ = _run_readonly(["iptables", "-S", chain])
    if code != 0:
        _emit_modify(["iptables", "-N", chain])


def _ensure_input_jump(port: int, chain: str) -> None:
    check_cmd = [
        "iptables",
        "-C",
        "INPUT",
        "-p",
        "tcp",
        "--dport",
        str(port),
        "-j",
        chain,
    ]
    code, _, _ = _run_readonly(check_cmd)
    if code != 0:
        # Insert near top
        _emit_modify(
            ["iptables", "-I", "INPUT", "-p", "tcp", "--dport", str(port), "-j", chain]
        )


def _flush_chain(chain: str) -> None:
    _emit_modify(["iptables", "-F", chain])


def _populate_allowlist(chain: str, ips: Iterable[str]) -> None:
    for ip in sorted(set(ips)):
        _emit_modify(["iptables", "-A", chain, "-s", ip, "-j", "ACCEPT"])


def _is_valid_ipv4(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        return isinstance(ipaddress.ip_address(s), ipaddress.IPv4Address)
    except Exception:
        return False


def _ip_from_axon(axon: object) -> Optional[str]:
    # Try multiple representations for robustness across bittensor versions
    candidates = []
    try:
        v = getattr(axon, "ip_str", None)
        if callable(v):
            candidates.append(v())
        elif v is not None:
            candidates.append(str(v))
    except Exception:
        pass
    try:
        v = getattr(axon, "ip_string", None)
        if callable(v):
            candidates.append(v())
        elif v is not None:
            candidates.append(str(v))
    except Exception:
        pass
    try:
        v = getattr(axon, "ip", None)
        if isinstance(v, int) and 0 < v < (1 << 32):
            try:
                candidates.append(socket.inet_ntoa(struct.pack("!I", v)))
            except Exception:
                pass
        elif isinstance(v, str):
            candidates.append(v)
    except Exception:
        pass

    for s in candidates:
        if _is_valid_ipv4(s):
            return s  # first valid IPv4
    return None


def _load_metagraph(network: str, netuid: int):
    bt.logging.info(f"Connect | network={network} netuid={netuid}")
    subtensor = bt.Subtensor(network=network)
    mg = bt.Metagraph(netuid=netuid, subtensor=subtensor)
    try:
        mg.sync(subtensor=subtensor)
    except Exception:
        pass
    return subtensor, mg


def _collect_allowed_ipv4(mg, stake_limit: float) -> Set[str]:
    n = int(getattr(mg, "n", 0) or 0)
    ips: Set[str] = set()

    # Vectors
    try:
        vpermit = list(getattr(mg, "validator_permit"))
    except Exception:
        vpermit = [0] * n
    try:
        stake_vec = getattr(mg, "stake", None)
        if hasattr(stake_vec, "tolist"):
            stake = list(stake_vec.tolist())
        elif isinstance(stake_vec, (list, tuple)):
            stake = list(stake_vec)
        else:
            # Broadcast if scalar
            sval = float(stake_vec) if stake_vec is not None else 0.0
            stake = [sval] * n
    except Exception:
        stake = [0.0] * n

    axons = list(getattr(mg, "axons", []))

    for uid in range(n):
        try:
            if uid >= len(axons):
                continue
            has_permit = bool(vpermit[uid]) if uid < len(vpermit) else False
            st = float(stake[uid]) if uid < len(stake) else 0.0
            if not has_permit or not (st > float(stake_limit)):
                continue
            ip = _ip_from_axon(axons[uid])
            if _is_valid_ipv4(ip):
                ips.add(str(ip))
        except Exception:
            continue
    return ips


def maintain_iptables_loop(
    network: str, netuid: int, stake_limit: float, port: int, interval: int
) -> None:
    chain = f"BTMG_ALLOW_{port}"

    if not _check_iptables_available():
        bt.logging.error("❌ iptables not available")
        sys.exit(2)
    if not _check_root():
        bt.logging.warning("⚠️ Not running as root; iptables may fail")

    # Prepare chain & jump once
    _ensure_chain(chain)
    _ensure_input_jump(port, chain)

    subtensor, mg = _load_metagraph(network, netuid)

    stop = False
    stop_event = threading.Event()

    def _on_sigterm(signum, frame):
        nonlocal stop
        stop = True
        try:
            stop_event.set()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _on_sigterm)
    signal.signal(signal.SIGTERM, _on_sigterm)

    last_ips: Set[str] = set()

    while not stop:
        try:
            # Sync metagraph
            try:
                mg.sync(subtensor=subtensor)
            except Exception:
                pass

            ips = _collect_allowed_ipv4(mg, stake_limit)

            # Only update iptables if changed
            if ips != last_ips:
                bt.logging.info(f"🔐 Update iptables | port={port} allowed={len(ips)}")
                _flush_chain(chain)
                _populate_allowlist(chain, ips)
                last_ips = ips
            else:
                bt.logging.debug(f"No change | validators={len(ips)} port={port}")
        except Exception as e:
            bt.logging.error(f"❌ Update error | err={getattr(e, 'args', [e])[0]}")

        # Wait for next cycle or signal-triggered stop
        if stop_event.wait(timeout=max(1, int(interval))):
            break

    bt.logging.info("Validator IP whitelist script exiting")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maintain iptables allowlist from metagraph validators"
    )
    parser.add_argument("--uid", type=int, required=True, help="Subnet netuid")
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="Network identifier (finney/test/local) or custom RPC endpoint",
    )
    parser.add_argument(
        "--stake-limit",
        type=float,
        required=True,
        help="Minimum TAO stake to qualify (strictly greater than)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="TCP port to allow for qualified validators",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=60,
        help="Metagraph fetch interval in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print iptables commands without applying changes",
    )

    args = parser.parse_args()

    # Initialize Bittensor logging and set default level to DEBUG for verbose output
    # Ensure console handler is attached, then raise level to DEBUG
    try:
        bt.logging.enable_default()
        bt.logging.enable_debug()
    except Exception:
        # Fallback for older bittensor versions
        try:
            bt.logging.set_debug(True)
        except Exception:
            pass

    bt.logging.info(
        f"🚦 Start | uid={args.uid} network={args.network} stake_limit={args.stake_limit} port={args.port} interval={args.interval}s"
    )

    try:
        global DRY_RUN
        DRY_RUN = bool(args.dry_run)
        maintain_iptables_loop(
            network=args.network,
            netuid=int(args.uid),
            stake_limit=float(args.stake_limit),
            port=int(args.port),
            interval=int(args.interval),
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
