#!/usr/bin/env python3
"""Multi-window regime-stability evaluator: skill vs luck.

A single-window CAGR can be one lucky bet. This slices the OOS span into
NON-OVERLAPPING windows and, for a given policy (score column + top-k), reports
in EACH window the strategy's after-cost CAGR, the SAME-UNIVERSE tradable
equal-weight basket CAGR, and the EXCESS — then aggregates a consistency
profile (fraction of windows with positive excess, mean/std/IR of excess, worst
window). Consistent positive excess across regimes = skill; one big window +
the rest flat/negative = luck.

Strategy returns come from the trusted strict backtest (baseline_protocol
variant-C: T+1, eligible top-k, cost/slippage/limit). The basket uses the same
tradability filters (ST/suspended excluded) so EXCESS largely cancels any
survivorship bias in the universe.

Usage:
    AI_quant_venv/bin/python3 scripts/stage6_regime_stability.py \
        --predictions <oos.parquet> --score-column alpha_5d --top-k 50 \
        --output-dir runtime/stage6_regime_stability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp  # noqa: E402

from quantagent.portfolio.policy_search import prepare_working_frame, universe_benchmark  # noqa: E402

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"


def _label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.date()}..{end.date()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--score-column", default="alpha_5d")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--window-days", type=int, default=120, help="Non-overlapping window size in trading days (~half year).")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "amount", "is_st", "is_suspended", "is_limit_up"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    sector = pd.read_parquet(SECTOR)
    # work frame (has realised ret + tradability) for the same-universe basket.
    sc = args.score_column
    p2 = preds.rename(columns={sc: "alpha_5d"})
    for c in ("alpha_1d", "alpha_20d"):
        if c not in p2.columns:
            p2[c] = p2["alpha_5d"]
    work = prepare_working_frame(p2, panel, sector)

    dates = sorted(work["trade_date"].dropna().unique())
    windows = [(dates[i], dates[min(i + args.window_days, len(dates)) - 1])
               for i in range(0, len(dates), args.window_days)]
    # drop a trailing stub window (<40 trading days) — too short to be meaningful.
    windows = [(s, e) for (s, e) in windows
               if pd.Index(dates).get_indexer([e])[0] - pd.Index(dates).get_indexer([s])[0] >= 40]

    rows = []
    for (ws, we) in windows:
        res = bp.evaluate(args.predictions, top_k=args.top_k, start=str(ws.date()), end=str(we.date()),
                          slippage_bps=8.0, variants=["C_flags_eligible_delay1"], score_column=sc)
        strat = res["variants"]["C_flags_eligible_delay1"]
        wk = work[(work["trade_date"] >= ws) & (work["trade_date"] <= we)]
        bm = universe_benchmark(wk)
        rows.append({
            "window": _label(ws, we),
            "strat_cagr": strat["ann"], "basket_cagr": bm["cagr"],
            "excess": strat["ann"] - bm["cagr"],
            "strat_maxDD": strat["maxDD"], "strat_sharpe": strat["sharpe"],
            "n_days": bm["n_days"],
        })
        r = rows[-1]
        print(f"  [{r['window']}] strat {r['strat_cagr']:+.2%} | basket {r['basket_cagr']:+.2%} | "
              f"excess {r['excess']:+.2%} | maxDD {r['strat_maxDD']:.2%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "regime_windows.csv", index=False)
    ex = df["excess"].to_numpy(dtype=float)
    n = len(ex)
    mean_ex, std_ex = float(np.mean(ex)), float(np.std(ex, ddof=1)) if n > 1 else float("nan")
    ir = mean_ex / std_ex if std_ex and std_ex > 1e-9 else float("nan")
    stability = {
        "score_column": sc, "top_k": args.top_k, "window_days": args.window_days,
        "n_windows": n,
        "n_positive_excess": int((ex > 0).sum()),
        "pct_positive_excess": round(float((ex > 0).mean()) * 100, 1),
        "mean_excess": mean_ex, "median_excess": float(np.median(ex)),
        "std_excess": std_ex, "excess_IR": ir,
        "worst_window_excess": float(np.min(ex)), "best_window_excess": float(np.max(ex)),
        "mean_strat_cagr": float(df["strat_cagr"].mean()),
        "mean_basket_cagr": float(df["basket_cagr"].mean()),
    }
    # skill requires CONSISTENT positive excess, not one big window.
    consistent = stability["pct_positive_excess"] >= 60 and mean_ex > 0 and (np.isfinite(ir) and ir >= 0.5)
    stability["verdict"] = ("ROBUST skill: consistent positive excess across regimes" if consistent
                            else "NOT robust: excess is inconsistent across regimes (likely window/luck)")
    (outdir / "stability.json").write_text(json.dumps(stability, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== REGIME STABILITY ===")
    print(json.dumps(stability, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
