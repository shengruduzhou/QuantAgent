#!/usr/bin/env python3
"""EXP-004 / H-004: long-sleeve information diagnostic (CPU-only, no selection).

Questions: does long_30d_120d carry regime-conditional cross-sectional
information that weight=0 discards? Measured on the SEARCH window with
PIT-clean labels: for each horizon h, IC dates are restricted to rows whose
``label_end_{h}d`` closes BEFORE the quarantined holdout (2025-09-01) so no
holdout-period outcome leaks into the diagnostic.

Outputs LONG_SLEEVE_DIAGNOSTIC data (json) consumed by the md report.
Diagnostic only — no production weights are tuned here (registry H-004).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import baseline_protocol as bp  # noqa: E402

GOLD = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
COMPOSITE = REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
RETURNS_MATRIX = REPO / "runtime/reports/v89_closed_loop/pbo_dsr_retro/candidate_val_daily_returns.csv"
OUT = REPO / "runtime/reports/v89_closed_loop/exp004_long_sleeve"
VAL_START = pd.Timestamp("2024-08-28")
VAL_END = pd.Timestamp("2025-08-31")
QUARANTINE_START = pd.Timestamp("2025-09-01")
HORIZONS = (20, 60, 120)
SLEEVES = ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score")
ANN = 244


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def per_date_ic(df: pd.DataFrame, score: str, label: str) -> pd.Series:
    """Spearman rank-IC per trade_date."""
    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 30:
            return np.nan
        return g[score].rank().corr(g[label].rank())
    return df.groupby("trade_date").apply(_ic, include_groups=False).dropna()


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    label_cols = [f"forward_return_{h}d" for h in HORIZONS] + [f"label_end_{h}d" for h in HORIZONS]
    gold = pd.read_parquet(GOLD, columns=["symbol", "trade_date", *label_cols],
                           filters=[("trade_date", ">=", VAL_START), ("trade_date", "<=", VAL_END)])
    gold["trade_date"] = pd.to_datetime(gold["trade_date"])
    scores = pd.read_parquet(COMPOSITE, columns=["symbol", "trade_date", *SLEEVES])
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    scores = scores[(scores["trade_date"] >= VAL_START) & (scores["trade_date"] <= VAL_END)]
    df = scores.merge(gold, on=["symbol", "trade_date"], how="inner")
    print(f"merged rows {len(df):,}  RSS {rss_gib():.2f} GiB", flush=True)

    # regime labels from the frictionless eqw bench (same rule as baseline_protocol)
    panel = pd.read_parquet(REPO / bp.PANEL, columns=["symbol", "trade_date", "close"],
                            filters=[("trade_date", ">=", VAL_START - pd.Timedelta(days=100)),
                                     ("trade_date", "<=", VAL_END)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    bench = bp._bench_daily(panel, sorted(panel["trade_date"].unique()))
    regime = bp._regime_label(bench)
    # bench 20d drawdown state for "drawdown contribution"
    cum = (1 + bench).cumprod()
    dd = 1.0 - cum / cum.cummax()
    dd_state = (dd > 0.05).rename("in_drawdown")

    results: dict[str, object] = {}
    for h in HORIZONS:
        lab, lend = f"forward_return_{h}d", f"label_end_{h}d"
        sub = df[pd.to_datetime(df[lend]) < QUARANTINE_START].dropna(subset=[lab])
        n_dates = sub["trade_date"].nunique()
        block: dict[str, object] = {"n_eval_dates": int(n_dates),
                                    "last_eval_date": str(sub["trade_date"].max().date()) if n_dates else None}
        for score in SLEEVES:
            ics = per_date_ic(sub, score, lab)
            reg = regime.reindex(ics.index)
            dds = dd_state.reindex(ics.index).fillna(False)
            ent = {
                "mean_ic": round(float(ics.mean()), 4),
                "icir": round(float(ics.mean() / ics.std()), 3) if ics.std() > 0 else None,
                "ic_by_regime": {r: round(float(ics[reg == r].mean()), 4)
                                 for r in ("bull", "sideways", "bear") if (reg == r).sum() >= 10},
                "ic_in_bench_drawdown": round(float(ics[dds].mean()), 4) if dds.sum() >= 10 else None,
                "ic_by_quarter": {str(q): round(float(g.mean()), 4)
                                  for q, g in ics.groupby(pd.PeriodIndex(ics.index, freq="Q"))},
            }
            block[score] = ent
        results[f"h{h}"] = block

    # cross-sleeve cross-sectional rank correlation (per date, averaged)
    def xsec_corr(a: str, b: str) -> float:
        cors = df.groupby("trade_date").apply(
            lambda g: g[a].rank().corr(g[b].rank()) if len(g) > 30 else np.nan,
            include_groups=False).dropna()
        return round(float(cors.mean()), 3)
    results["sleeve_rank_corr"] = {
        "long_vs_short": xsec_corr("long_30d_120d_score", "short_5d_score"),
        "long_vs_mid": xsec_corr("long_30d_120d_score", "mid_5d_30d_score"),
        "short_vs_mid": xsec_corr("short_5d_score", "mid_5d_30d_score"),
    }

    # shrinkage stability from EXISTING replay matrix (zero new backtests)
    m = pd.read_csv(RETURNS_MATRIX, index_col=0, parse_dates=True)
    T = len(m)
    bounds = np.linspace(0, T, 5, dtype=int)
    def qcagr(col: str) -> list[float]:
        out = []
        for i in range(4):
            r = m[col].iloc[bounds[i]:bounds[i + 1]].to_numpy()
            nav = float(np.prod(1 + r))
            out.append(round(nav ** (ANN / len(r)) - 1, 4))
        return out
    results["shrinkage_stability_from_existing_matrix"] = {
        c: {"quarter_cagr": qcagr(c), "worst": min(qcagr(c))}
        for c in ("w1_1_0_k10", "w1_1_0.5_k10", "w1_1_1_k10", "w1_1_0_k30", "w1_1_0.5_k30")
        if c in m.columns
    }
    results["peak_rss_gib"] = round(rss_gib(), 2)
    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "diagnostic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2)[:3000])
    print(f"peak RSS {results['peak_rss_gib']} GiB, {results['runtime_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
