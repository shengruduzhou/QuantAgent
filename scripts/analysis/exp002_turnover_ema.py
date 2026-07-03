#!/usr/bin/env python3
"""EXP-002 / H-002: turnover-aware EMA smoothing on the C3 rank-median blend.

Pre-registered N=3 (HYPOTHESIS_REGISTRY.md): per-symbol recursive EMA
(s_t = a*x_t + (1-a)*s_{t-1}, pandas ewm(adjust=False)) with a in {0.3,0.5,0.7}
applied to the C3 rank-median score from EXP-001 (the only aggregate that
failed solely on turnover). Reference for degradation = unsmoothed C3.

Window: SEARCH 2024-08-28..2025-08-31 only (quarantine-pure, asserted).
Acceptance (registered): exists a with turnover <= 0.10/day AND worst-quarter
CAGR >= reference - 3pp. Note: EMA over a symbol's available score history
ignores calendar gaps (delistings/suspensions treated as consecutive obs).
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
OUT_DIR = REPO / "runtime/reports/v89_closed_loop/exp002_turnover_ema"
VAL_START = pd.Timestamp("2024-08-28")
VAL_END = pd.Timestamp("2025-08-31")
QUARANTINE_START = pd.Timestamp("2025-09-01")
TOP_K = 10
ANN = 244
ALPHAS = (0.3, 0.5, 0.7)
QUARTERS = [("2024-08-30", "2024-11-30"), ("2024-12-01", "2025-02-28"),
            ("2025-03-01", "2025-05-31"), ("2025-06-01", "2025-08-31")]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def cagr(r: np.ndarray) -> float:
    nav = float(np.prod(1.0 + r))
    return nav ** (ANN / len(r)) - 1.0 if len(r) and nav > 0 else -1.0


def c3_frame() -> pd.DataFrame:
    base = pd.read_parquet(COMPOSITE)
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base = base[base["trade_date"] <= VAL_END]
    assert base["trade_date"].max() < QUARANTINE_START
    ranks = np.column_stack([
        base.groupby("trade_date")[c].rank(pct=True).to_numpy()
        for c in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score")
    ])
    f = base[["trade_date", "symbol"]].copy()
    f["alpha_score"] = np.median(ranks, axis=1)
    return f.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c3 = c3_frame()

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

    def run(name: str, preds: pd.DataFrame) -> dict:
        t1 = time.time()
        p = preds[preds["trade_date"] >= VAL_START].merge(flags, on=["symbol", "trade_date"], how="left")
        tw = bp._target_weights(p, "alpha_score", TOP_K, eligible_only=True,
                                delay_days=1, trade_dates=trade_dates)
        res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
        nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
        r = nav.pct_change().dropna()
        m = res.metrics
        q = [round(cagr(r[(r.index >= qs) & (r.index <= qe)].to_numpy()), 4) for qs, qe in QUARTERS]
        row = {"cagr": round(m.annualized_return, 4), "sharpe": round(m.sharpe, 3),
               "maxdd": round(m.max_drawdown, 4), "turnover": round(float(m.turnover), 4),
               "quarter_cagr": q, "worst_quarter": min(q)}
        print(f"{name:12s} CAGR {m.annualized_return:+.1%} maxDD {m.max_drawdown:.1%} "
              f"turn {m.turnover:.3f} quarters {q} ({time.time()-t1:.0f}s)", flush=True)
        return row

    rows = {"C3_raw": run("C3_raw", c3)}
    for a in ALPHAS:
        sm = c3.copy()
        sm["alpha_score"] = (sm.groupby("symbol")["alpha_score"]
                             .transform(lambda s: s.ewm(alpha=a, adjust=False).mean()))
        rows[f"C3_ema{a}"] = run(f"C3_ema{a}", sm)

    ref = rows["C3_raw"]
    checks = {}
    for a in ALPHAS:
        k = f"C3_ema{a}"
        checks[k] = {
            "turnover_le_0.10": rows[k]["turnover"] <= 0.10,
            "worst_quarter_within_3pp_of_ref": rows[k]["worst_quarter"] >= ref["worst_quarter"] - 0.03,
            "cagr_within_3pp_of_ref": rows[k]["cagr"] >= ref["cagr"] - 0.03,
        }

    out = {"hypothesis": "H-002", "window": f"{VAL_START.date()}..{VAL_END.date()}",
           "top_k": TOP_K, "n_registered_trials": 3, "candidates": rows,
           "acceptance_checks": checks,
           "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1)}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"peak RSS {out['peak_rss_gib']} GiB, {out['runtime_sec']}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
