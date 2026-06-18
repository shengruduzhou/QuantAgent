#!/usr/bin/env python3
"""Forward Level-2 depth-snapshot collector for held names.

TickFlow's ``Depth.get(symbol)`` returns a REAL-TIME order-book snapshot only
(bid/ask prices+volumes at the current instant) -- there is NO historical depth
or transaction endpoint.  So an order-flow do-T model cannot be backtested; the
features must be COLLECTED FORWARD.  This script snapshots depth for the held
book and persists derived order-flow features so the data starts accumulating
toward a future L2 do-T training set.

Run it on an intraday schedule (e.g. every 30-60s via systemd/cron during
09:30-15:00 CST), or with ``--loop-seconds N`` to self-poll.  Output:

    runtime/data/v7/silver/depth_snapshots/{YYYY-MM-DD}.parquet  (appended)

Derived features per snapshot (the LEVEL2_FEATURE_COLUMNS the EV engine already
knows how to consume): order_book_imbalance, bid_depth, ask_depth,
bid_ask_spread, queue_pressure_near_limit, microprice_dev.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("runtime/data/v7/silver/depth_snapshots")


def _client():
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    import tickflow

    return tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                             base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)


def _depth_features(d) -> dict | None:
    bp = list(getattr(d, "bid_prices", None) or (d.get("bid_prices") if isinstance(d, dict) else []) or [])
    bv = list(getattr(d, "bid_volumes", None) or (d.get("bid_volumes") if isinstance(d, dict) else []) or [])
    ap = list(getattr(d, "ask_prices", None) or (d.get("ask_prices") if isinstance(d, dict) else []) or [])
    av = list(getattr(d, "ask_volumes", None) or (d.get("ask_volumes") if isinstance(d, dict) else []) or [])
    ts = getattr(d, "timestamp", None) or (d.get("timestamp") if isinstance(d, dict) else None)
    sym = getattr(d, "symbol", None) or (d.get("symbol") if isinstance(d, dict) else None)
    bp = [float(x) for x in bp if x is not None]
    bv = [float(x) for x in bv if x is not None]
    ap = [float(x) for x in ap if x is not None]
    av = [float(x) for x in av if x is not None]
    if not bp or not ap:
        return None
    bid_depth = float(np.sum(bv)) if bv else 0.0
    ask_depth = float(np.sum(av)) if av else 0.0
    best_bid, best_ask = bp[0], ap[0]
    mid = (best_bid + best_ask) / 2.0
    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0
    bv0, av0 = (bv[0] if bv else 0.0), (av[0] if av else 0.0)
    micro = (best_bid * av0 + best_ask * bv0) / (bv0 + av0) if (bv0 + av0) > 0 else mid
    return {
        "symbol": str(sym), "timestamp": ts, "snapshot_time": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
        "bid_ask_spread": (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else np.nan,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "order_book_imbalance": imbalance,
        "queue_pressure_near_limit": bv0 / (av0 + 1.0),
        "microprice_dev": (micro - mid) / mid * 10_000.0 if mid > 0 else 0.0,
        "n_bid_levels": len(bp), "n_ask_levels": len(ap),
    }


def _load_symbols(args) -> list[str]:
    if args.book_csv and Path(args.book_csv).exists():
        df = pd.read_csv(args.book_csv)
        if "symbol" in df.columns:
            return sorted(df["symbol"].astype(str).unique())
    if args.symbols_file and Path(args.symbols_file).exists():
        txt = Path(args.symbols_file).read_text(encoding="utf-8")
        return [t.strip() for t in txt.replace(",", "\n").splitlines() if t.strip()]
    return []


def snapshot_once(tf, symbols: list[str], sleep_s: float = 0.05) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            d = tf.depth.get(sym)
        except Exception:  # noqa: BLE001
            continue
        feat = _depth_features(d)
        if feat is not None:
            rows.append(feat)
        if sleep_s:
            time.sleep(sleep_s)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book-csv", default="runtime/paper/forward/A_default/targets_latest.csv")
    ap.add_argument("--symbols-file", default="")
    ap.add_argument("--loop-seconds", type=int, default=0, help="0 = single snapshot; >0 = poll every N s")
    ap.add_argument("--max-iterations", type=int, default=0, help="0 = until 15:00 CST")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    symbols = _load_symbols(args)
    if not symbols:
        raise SystemExit("no symbols (provide --book-csv or --symbols-file)")
    tf = _client()

    # Preflight: market-depth permission is a paid TickFlow entitlement. As of
    # 2026-06-17 the current API key returns "无市场深度权限（市场: CN）". Fail fast
    # with an actionable message rather than logging errors all session.
    try:
        tf.depth.get(symbols[0])
    except Exception as exc:  # noqa: BLE001
        if "权限" in str(exc) or "permission" in str(exc).lower():
            raise SystemExit(
                "TickFlow depth permission missing for CN (无市场深度权限). Level-2 order-flow "
                "collection requires upgrading the TickFlow subscription to include CN market depth. "
                f"Detail: {exc}"
            )
        # other transient errors: continue (will be retried per snapshot)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _persist(df: pd.DataFrame) -> None:
        if df.empty:
            return
        day = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
        path = OUT_DIR / f"{day}.parquet"
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
        print(f"{pd.Timestamp.now()}: +{len(df)} rows -> {path}", flush=True)

    if args.loop_seconds <= 0:
        _persist(snapshot_once(tf, symbols, args.sleep))
        return 0
    it = 0
    while True:
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        if now.strftime("%H:%M") >= "15:00":
            print("session closed; stopping", flush=True)
            break
        _persist(snapshot_once(tf, symbols, args.sleep))
        it += 1
        if args.max_iterations and it >= args.max_iterations:
            break
        time.sleep(args.loop_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
