#!/usr/bin/env python3
"""
Interactive inspector for Bittensor subnets.

Features:
1) List all neurons on a subnet (stake, validator_permit, validator_trust, emission)
2) Select network (finney/test/local) and input netuid
3) Inspect a validator's latest and historical weight assignments, with target miner info

Usage:
  python scripts/inspect_network.py

Notes:
- Chain history access depends on the connected node. Historical block queries may not be available on non-archive nodes.
- Stake values are displayed in TAO (as reported by Metagraph).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bittensor as bt

# ---------- Helpers ----------


def _read_choice(prompt: str, choices: Sequence[str], default_index: int = 0) -> str:
    labels = "/".join(f"{i+1}:{c}" for i, c in enumerate(choices))
    while True:
        raw = input(f"{prompt} [{labels}] (default {default_index+1}): ").strip()
        if not raw:
            return choices[default_index]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        # allow direct text match
        for c in choices:
            if raw.lower() == c.lower():
                return c
        print("Invalid choice. Try again.")


def _read_int(
    prompt: str, default: Optional[int] = None, min_value: Optional[int] = None
) -> int:
    while True:
        raw = input(
            f"{prompt}{f' (default {default})' if default is not None else ''}: "
        ).strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if min_value is not None and val < min_value:
                print(f"Value must be >= {min_value}")
                continue
            return val
        except Exception:
            print("Please input a valid integer.")


def _fmt_float(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _get_attr_vec(mg: Any, names: Sequence[str]) -> Optional[List[float]]:
    for n in names:
        if hasattr(mg, n):
            try:
                val = getattr(mg, n)
                # Convert torch tensor or numpy to list if needed
                if hasattr(val, "tolist"):
                    return list(val.tolist())
                if isinstance(val, (list, tuple)):
                    return list(val)
                # Single value broadcast
                if isinstance(val, (int, float)) and hasattr(mg, "n"):
                    return [float(val)] * int(mg.n)
            except Exception:
                pass
    return None


def _index_by_hotkey(hotkeys: Sequence[str]) -> Dict[str, int]:
    return {hk: i for i, hk in enumerate(hotkeys)}


def _resolve_validator_uid(mg: Any, query: str) -> Optional[int]:
    # Accept UID integer
    if query.isdigit():
        uid = int(query)
        return uid if 0 <= uid < mg.n else None
    # Try hotkey exact match or case-insensitive
    hotkeys = list(mg.hotkeys)
    m = _index_by_hotkey(hotkeys)
    if query in m:
        return m[query]
    low_map = {hk.lower(): i for hk, i in m.items()}
    return low_map.get(query.lower())


# ---------- Chain Access ----------


def load_metagraph(network: str, netuid: int) -> Tuple[Any, Any]:
    bt.logging.info(f"🌐 Connect | network={network} netuid={netuid}")
    subtensor = bt.subtensor(network=network)
    mg = bt.metagraph(netuid=netuid, subtensor=subtensor)
    # Ensure synced to latest
    try:
        mg.sync()
    except Exception:
        pass
    return subtensor, mg


def get_current_block_safe(subtensor: Any, mg: Any) -> Optional[int]:
    """Best-effort retrieval of current block height with multiple fallbacks.

    Order:
    1) subtensor.get_current_block()
    2) substrate.get_block_header(None)["number"] (supports hex/str/int)
    3) metagraph.block (snapshot height)
    """
    # 1) Preferred bittensor API
    try:
        blk = getattr(subtensor, "get_current_block", None)
        if callable(blk):
            v = blk()
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass

    # 2) Substrate header number (may be hex string)
    try:
        head = subtensor.substrate.get_block_header(None)
        num = head.get("number") if isinstance(head, dict) else None
        if isinstance(num, int):
            return num
        if isinstance(num, str):
            s = num.strip()
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s)
        # ScaleType or other: try coercion
        try:
            return int(num)  # type: ignore[arg-type]
        except Exception:
            pass
    except Exception:
        pass

    # 3) Metagraph snapshot height
    try:
        mg_block = getattr(mg, "block", None)
        if isinstance(mg_block, int) and mg_block > 0:
            return mg_block
    except Exception:
        pass
    return None


def list_neurons(mg: Any, limit: Optional[int] = None) -> None:
    n = int(mg.n)
    hotkeys = list(mg.hotkeys)
    stake = _get_attr_vec(mg, ["stake", "S", "total_stake", "tao_stake"]) or [None] * n
    vpermit = _get_attr_vec(mg, ["validator_permit"]) or [0] * n
    vtrust = _get_attr_vec(mg, ["validator_trust", "V"]) or [None] * n
    emission = _get_attr_vec(mg, ["emission", "E"]) or [None] * n

    rows = []
    for uid in range(n):
        rows.append(
            (
                uid,
                hotkeys[uid],
                stake[uid] if uid < len(stake) else None,
                bool(vpermit[uid]) if uid < len(vpermit) else False,
                vtrust[uid] if uid < len(vtrust) else None,
                emission[uid] if uid < len(emission) else None,
            )
        )

    # Sort: validators first then by stake desc
    rows.sort(key=lambda r: (not r[3], -(r[2] or 0.0)))

    print("\nAll neurons (validators first):")
    print(
        "UID  HOTKEY                                                   STAKE(TAO)   V-PERMIT  VTRUST     EMISSION"
    )
    print(
        "---- -------------------------------------------------------- ----------- --------- --------- ----------"
    )
    shown = 0
    for uid, hk, st, vp, vt, em in rows:
        print(
            f"{uid:>4} {hk[:48]:<48} {_fmt_float(st, 6):>11} {str(vp):>9} {_fmt_float(vt, 4):>9} {_fmt_float(em, 6):>10}"
        )
        shown += 1
        if limit and shown >= limit:
            break
    print(f"Total: {n} neurons\n")


def fetch_validator_weights(
    subtensor: Any, netuid: int, block: Optional[int] = None
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    return subtensor.weights(netuid=netuid, block=block)


def show_validator_assignment(
    subtensor: Any,
    mg: Any,
    validator_uid: int,
    block: Optional[int] = None,
    top_k: Optional[int] = None,
) -> None:
    wmap = fetch_validator_weights(subtensor, mg.netuid, block=block)
    entry = next((t for t in wmap if int(t[0]) == int(validator_uid)), None)
    if entry is None:
        print("No weights found for this validator (maybe no permit / never set).")
        return

    to_list = list(entry[1]) if entry[1] else []
    if not to_list:
        print("Validator has no weight assignments.")
        return

    total = sum(w for _, w in to_list) or 1
    # Map miner info
    hotkeys = list(mg.hotkeys)
    stake = _get_attr_vec(mg, ["stake", "S", "total_stake", "tao_stake"]) or []
    trust = _get_attr_vec(mg, ["trust", "T"]) or []
    emission = _get_attr_vec(mg, ["emission", "E"]) or []

    rows: List[Tuple[int, str, float, float, float, float]] = []
    for to_uid, w in to_list:
        ratio = float(w) / float(total)
        hk = hotkeys[to_uid] if 0 <= to_uid < len(hotkeys) else "?"
        st = stake[to_uid] if 0 <= to_uid < len(stake) else None
        tr = trust[to_uid] if 0 <= to_uid < len(trust) else None
        em = emission[to_uid] if 0 <= to_uid < len(emission) else None
        rows.append((to_uid, hk, ratio, st or 0.0, tr or 0.0, em or 0.0))

    rows.sort(key=lambda r: -r[2])

    block_str = f"block {block}" if block is not None else "latest"
    print(f"\nValidator {validator_uid} → miner weights ({block_str}):")
    print(
        "TO_UID HOTKEY                                           WEIGHT%    STAKE(TAO)   TRUST       EMISSION"
    )
    print(
        "------ ------------------------------------------------ --------- ------------ ----------- ----------"
    )
    shown = 0
    for to_uid, hk, ratio, st, tr, em in rows:
        print(
            f"{to_uid:>6} {hk[:48]:<48} {ratio*100:>8.2f}% {_fmt_float(st, 6):>12} {_fmt_float(tr, 6):>11} {_fmt_float(em, 6):>10}"
        )
        shown += 1
        if top_k and shown >= top_k:
            break
    print(f"Total targets: {len(rows)}\n")


# ---------- Interactive Flow ----------


def interactive_loop() -> None:
    print("Bittensor Subnet Inspector")

    # Initial selection
    network = _read_choice(
        "Select network", ["finney", "test", "local"], default_index=0
    )
    netuid = _read_int("Enter netuid", min_value=0)

    subtensor, mg = load_metagraph(network, netuid)
    print(f"Loaded subnet: network={network} netuid={netuid} neurons={mg.n}")

    while True:
        print("\nMenu:")
        print("  1) List all neurons")
        print("  2) Inspect a validator's scores (latest / history)")
        print("  3) Switch network/netuid")
        print("  0) Exit")

        choice = input("Select option: ").strip()
        if choice == "1":
            list_neurons(mg)
        elif choice == "2":
            q = input("Enter validator identifier (UID or hotkey): ").strip()
            uid = _resolve_validator_uid(mg, q)
            if uid is None:
                print("Invalid UID/hotkey")
                continue
            show_validator_assignment(subtensor, mg, uid, block=None, top_k=None)

            # History option
            hist = input("View history? (y/N): ").strip().lower()
            if hist == "y":
                # Try sampling a few recent blocks with step
                step = _read_int(
                    "Block step (≈ weight_update_tempo, default 105)",
                    default=105,
                    min_value=1,
                )
                count = _read_int("Show last N history points", default=5, min_value=1)
                current_block = get_current_block_safe(subtensor, mg)
                if current_block is None:
                    print("⚠️ Unable to get current block; history may be limited.")
                for i in range(1, count + 1):
                    blk = (
                        None
                        if current_block is None
                        else max(0, current_block - i * step)
                    )
                    try:
                        show_validator_assignment(
                            subtensor, mg, uid, block=blk, top_k=30
                        )
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Failed to read history @ block={blk}: {e}")
                        break

        elif choice == "3":
            network = _read_choice(
                "Select network", ["finney", "test", "local"], default_index=0
            )
            netuid = _read_int("Enter netuid", min_value=0)
            subtensor, mg = load_metagraph(network, netuid)
            print(f"Loaded subnet: network={network} netuid={netuid} neurons={mg.n}")
        elif choice == "0":
            print("Exit.")
            return
        else:
            print("Invalid choice.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bittensor subnet inspector")
    p.add_argument(
        "--network", choices=["finney", "test", "local"], help="Network name"
    )
    p.add_argument("--netuid", type=int, help="Subnet netuid")
    p.add_argument("--validator", help="Validator uid or hotkey for direct query")
    p.add_argument("--block", type=int, help="Block number for weight query")
    p.add_argument("--top", type=int, default=None, help="Show only top K targets")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.network or args.netuid is None:
        return interactive_loop()

    subtensor, mg = load_metagraph(args.network, args.netuid)
    list_neurons(mg)

    if args.validator:
        uid = _resolve_validator_uid(mg, args.validator)
        if uid is None:
            print("Invalid validator identifier.")
            return
        show_validator_assignment(subtensor, mg, uid, block=args.block, top_k=args.top)


if __name__ == "__main__":
    main()
