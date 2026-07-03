#!/usr/bin/env python3
"""EXP-001 / H-001: family-robust blend vs the single searched winner.

Pre-registered in HYPOTHESIS_REGISTRY.md (N=4, a-priori candidates, NO search):
  C1_apriori_avg   composite_score column of ensemble_composite.parquet
                   (HorizonEnsembleWeights 0.30/0.45/0.25 weighted average —
                   the historical code default, zero selection)
  C2_prod_rank110  per-date pct-rank sum, weights (1,1,0)  [production candidate]
  C3_rank_median   per-date median of the three sleeve pct-ranks
  C4_rank_sum111   per-date pct-rank sum, weights (1,1,1)

Evaluation: strict variant C (eligible + delay-1, top_k=10, slippage 8bps) on
the SEARCH window 2024-08-28..2025-08-31 ONLY (quarantine-pure; asserted).
Stability: quarterly sub-slices of the full-window daily returns (position
carry-over across quarter boundaries is retained — realistic, and identical
treatment for all candidates).

Outputs: runtime/reports/v89_closed_loop/exp001_family_blend/{results.json,
daily_returns.csv} — small artifacts only.
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

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

COMPOSITE = REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
OUT_DIR = REPO / "runtime/reports/v89_closed_loop/exp001_family_blend"
VAL_START = pd.Timestamp("2024-08-28")
VAL_END = pd.Timestamp("2025-08-31")
QUARANTINE_START = pd.Timestamp("2025-09-01")
TOP_K = 10
ANN = 244
QUARTERS = [("2024-08-30", "2024-11-30"), ("2024-12-01", "2025-02-28"),
            ("2025-03-01", "2025-05-31"), ("2025-06-01", "2025-08-31")]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def cagr(r: np.ndarray) -> float:
    nav = float(np.prod(1.0 + r))
    return nav ** (ANN / len(r)) - 1.0 if len(r) and nav > 0 else -1.0


def max_dd(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max()) if len(r) else 0.0


def build_candidates() -> dict[str, pd.DataFrame]:
    base = pd.read_parquet(COMPOSITE)
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base = base[base["trade_date"] <= VAL_END]
    assert base["trade_date"].max() < QUARANTINE_START
    ranks = {}
    for s in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score"):
        ranks[s] = base.groupby("trade_date")[s].rank(pct=True)
    keys = base[["trade_date", "symbol"]].copy()

    def frame(score: pd.Series | np.ndarray) -> pd.DataFrame:
        f = keys.copy()
        f["alpha_score"] = np.asarray(score)
        return f

    return {
        "C1_apriori_avg": frame(base["composite_score"].to_numpy()),
        "C2_prod_rank110": frame(ranks["short_5d_score"] + ranks["mid_5d_30d_score"]),
        "C3_rank_median": frame(np.median(np.column_stack([ranks[s].to_numpy() for s in ranks]), axis=1)),
        "C4_rank_sum111": frame(ranks["short_5d_score"] + ranks["mid_5d_30d_score"] + ranks["long_30d_120d_score"]),
    }


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = build_candidates()

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                            filters=[("trade_date", ">=", VAL_START - pd.Timedelta(days=10)),
                                     ("trade_date", "<=", VAL_END)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUARANTINE_START, "quarantine breach"
    sector = pd.read_parquet(REPO / bp.SECTOR)
    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    trade_dates = sorted(panel["trade_date"].unique())
    cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)

    rets: dict[str, pd.Series] = {}
    rows: dict[str, dict] = {}
    for name, preds in cands.items():
        t1 = time.time()
        p = preds[preds["trade_date"] >= VAL_START].merge(flags, on=["symbol", "trade_date"], how="left")
        tw = bp._target_weights(p, "alpha_score", TOP_K, eligible_only=True,
                                delay_days=1, trade_dates=trade_dates)
        res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
        nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
        r = nav.pct_change().dropna()
        rets[name] = r
        m = res.metrics
        q_cagr = []
        for qs, qe in QUARTERS:
            seg = r[(r.index >= qs) & (r.index <= qe)].to_numpy()
            q_cagr.append(round(cagr(seg), 4))
        rows[name] = {
            "cagr": round(m.annualized_return, 4), "sharpe": round(m.sharpe, 3),
            "maxdd": round(m.max_drawdown, 4), "turnover": round(float(m.turnover), 4),
            "quarter_cagr": q_cagr, "worst_quarter": min(q_cagr),
            "n_positive_quarters": int(sum(q > 0 for q in q_cagr)),
        }
        print(f"{name:16s} CAGR {m.annualized_return:+.1%} maxDD {m.max_drawdown:.1%} "
              f"turn {m.turnover:.3f} quarters {q_cagr} ({time.time()-t1:.0f}s)", flush=True)

    # per-quarter candidate ranking stability (1 = best)
    qranks = {}
    for i in range(len(QUARTERS)):
        vals = {n: rows[n]["quarter_cagr"][i] for n in rows}
        order = sorted(vals, key=lambda n: -vals[n])
        for n in rows:
            qranks.setdefault(n, []).append(order.index(n) + 1)
    for n in rows:
        rows[n]["quarter_rank"] = qranks[n]

    # H-001 acceptance check
    c2_worst = rows["C2_prod_rank110"]["worst_quarter"]
    verdicts = {}
    for n in ("C1_apriori_avg", "C3_rank_median", "C4_rank_sum111"):
        verdicts[n] = {
            "worst_quarter_ge_C2": rows[n]["worst_quarter"] >= c2_worst,
            "turnover_le_0.25": rows[n]["turnover"] <= 0.25,
            "all_quarters_positive": rows[n]["n_positive_quarters"] == 4,
        }

    out = {
        "hypothesis": "H-001", "window": f"{VAL_START.date()}..{VAL_END.date()}",
        "top_k": TOP_K, "n_registered_trials": 4,
        "candidates": rows, "acceptance_checks_vs_C2": verdicts,
        "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rets).to_csv(OUT_DIR / "daily_returns.csv")
    print(json.dumps(out["acceptance_checks_vs_C2"], indent=2))
    print(f"peak RSS {out['peak_rss_gib']} GiB, {out['runtime_sec']}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
