#!/usr/bin/env python3
"""Build a PIT financial-feature panel from TickFlow for the v8 training universe.

Pulls ``tf.financials.metrics(symbol)`` (quarterly history, with ``announce_date`` =
the real disclosure date → PIT-safe) for every symbol in the training dataset, then
as-of joins each symbol's quarterly metrics onto its daily trade_dates (forward-fill
the latest report whose announce_date <= trade_date). Output columns are named to
match the model's LONG feature patterns (roe / gross_margin / net_margin /
revenue_yoy / net_income_yoy) so ``train-v8-deep`` picks them up automatically.

Output: runtime/data/v7/gold/training_dataset/tickflow_fin_features.parquet
        (symbol, trade_date, roe, gross_margin, net_margin, revenue_yoy, net_income_yoy)

Resumable: per-symbol quarterly pulls are disk-cached. Re-run to fill gaps.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATASET = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
OUT = "runtime/data/v7/gold/training_dataset/tickflow_fin_features.parquet"
QCACHE = Path("runtime/data/v7/silver/tickflow_fin_quarterly")
METRICS = ["roe", "gross_margin", "net_margin", "revenue_yoy", "net_income_yoy"]


def _tf():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow
    return tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                             base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)


def _pull_quarterly(tf, symbol: str) -> pd.DataFrame | None:
    cf = QCACHE / f"{symbol}.parquet"
    if cf.exists():
        try:
            return pd.read_parquet(cf)
        except Exception:
            pass
    try:
        m = tf.financials.metrics(symbol, as_dataframe=True)
    except Exception:
        return None
    if m is None or getattr(m, "empty", True):
        return None
    keep = ["symbol", "period_end", "announce_date"] + [c for c in METRICS if c in m.columns]
    m = m[keep].copy()
    try:
        QCACHE.mkdir(parents=True, exist_ok=True)
        m.to_parquet(cf, index=False)
    except Exception:
        pass
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (0=all; for smoke)")
    ap.add_argument("--workers", type=int, default=6, help="parallel tickflow pulls")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = pd.read_parquet(args.dataset, columns=["symbol", "trade_date"])
    ds["trade_date"] = pd.to_datetime(ds["trade_date"])
    symbols = sorted(ds["symbol"].astype(str).unique())
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"universe: {len(symbols)} symbols; pulling tickflow quarterly metrics (workers={args.workers})", flush=True)

    tf = _tf()
    quarterly: dict[str, pd.DataFrame] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_pull_quarterly, tf, s): s for s in symbols}
        for fut in futs:
            s = futs[fut]
            q = fut.result()
            done += 1
            if q is not None and not q.empty:
                quarterly[s] = q
            if done % 200 == 0:
                print(f"  pulled {done}/{len(symbols)} ({len(quarterly)} with data)", flush=True)
    print(f"pulled {len(quarterly)}/{len(symbols)} symbols with quarterly metrics", flush=True)

    # per-symbol as-of join: latest report with announce_date <= trade_date
    frames = []
    for s, q in quarterly.items():
        q = q.copy()
        q["announce_date"] = pd.to_datetime(q["announce_date"], errors="coerce")
        q = q.dropna(subset=["announce_date"]).sort_values("announce_date")
        if q.empty:
            continue
        days = ds[ds["symbol"] == s][["trade_date"]].drop_duplicates().sort_values("trade_date")
        if days.empty:
            continue
        merged = pd.merge_asof(days, q, left_on="trade_date", right_on="announce_date", direction="backward")
        merged["symbol"] = s
        cols = ["symbol", "trade_date"] + [c for c in METRICS if c in merged.columns]
        frames.append(merged[cols])
    if not frames:
        raise SystemExit("no financial features built")
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    cov = out[METRICS].notna().mean().round(3).to_dict()
    print(f"wrote {args.out}: rows={len(out)} symbols={out['symbol'].nunique()} coverage={cov}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
