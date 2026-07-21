#!/usr/bin/env python3
"""H-028 Track C: final NON-SELECTIVE historical strict risk audit (U0 cohort).

Frozen books only (no tuning, no selection, no winner declaration):
  S2 = L1_c3ema07_minhold10, S3 = L1+D1_regime w=0.5 — simulated here via the
  EXP-024 book reconstruction (numeric reproduction check vs the freeze
  manifest reference is part of this run, fail-closed on mismatch).
  S4 = RW1_4state — fold-level record stands in EXP-023 results.json
  (8/25bps); carrier pool/turnover identical to S2 => cost/capacity cells
  transfer (EXP-024 argument). Not re-simulated (learner stays out of a
  diagnostic script).
  S1 = production blend — SEARCH-window record on file (EXP-000/001,
  PBO_DSR_ANALYSIS); its predictions do not span the H-008 fold set.
Grid (preregistered d444872): {8,15,25}bps @1M; 8bps @{10M,30M};
sqrt-impact overlay eta=1.0/2.0 on the 1M and 30M 8bps runs.
Impact overlay = per-name |dw|*AUM traded value, vol20 * sqrt(traded/ADV20),
applied as a daily return drag — analysis-layer, conservative (ignores the
simulator's own fill caps), never touches trusted-evaluator defaults.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import FOLDS, cagr, max_dd  # noqa: E402
from exp024_capacity_study import fold_books, rss_gib  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402
from quantagent.backtest.impact_model import SqrtImpactParams, ETA_BASE, ETA_STRESSED  # noqa: E402

OUT = REPO / "runtime/reports/h028/track_c"
REPRO_REF = {"L1": 0.364, "L1_d1_regime": 0.253}   # freeze-manifest medians @8bps/1M
REPRO_TOL = 0.005


def impact_drag(tw: pd.DataFrame, panel: pd.DataFrame, aum: float, eta: float) -> pd.Series:
    """Daily sqrt-impact return drag from target-weight changes (conservative)."""
    p = panel.sort_values(["symbol", "trade_date"]).copy()
    g = p.groupby("symbol", sort=False)
    p["adv20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    p["vol20"] = g["close"].transform(lambda s: s.pct_change().rolling(20, min_periods=5).std())
    ref = p.set_index(["trade_date", "symbol"])[["adv20", "vol20"]]

    w = tw.copy()  # already wide: index=trade_date, columns=symbol, values=weight
    w.index = pd.to_datetime(w.index)
    dw = w.diff().abs()
    dw.iloc[0] = w.iloc[0].abs()
    drag = {}
    for d, row in dw.iterrows():
        traded = row[row > 1e-9] * aum
        if traded.empty:
            drag[d] = 0.0
            continue
        tot = 0.0
        for sym, v in traded.items():
            try:
                adv, vol = ref.loc[(d, sym), "adv20"], ref.loc[(d, sym), "vol20"]
            except KeyError:
                continue
            if not (np.isfinite(adv) and np.isfinite(vol)) or adv < 1.0:
                continue
            tot += eta * vol * np.sqrt(min(v, 0.10 * adv) / adv) * (v / aum)
        drag[d] = tot
    return pd.Series(drag)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    rows, navs, repro = [], {}, {}
    for fold in FOLDS:
        books, bt_panel, _ = fold_books(fold)
        for name, tw in books.items():
            cells = [(1e6, 8.0), (1e6, 15.0), (1e6, 25.0), (1e7, 8.0), (3e7, 8.0)]
            for aum, bps in cells:
                cfg = AShareExecutionSimulationConfig(initial_cash=float(aum), slippage_bps=bps)
                r = run_strict_backtest_v8(tw, bt_panel, sector_map=sector, config=cfg)
                nav = r.nav.copy(); nav.index = pd.to_datetime(nav.index)
                rr = nav.pct_change().dropna()
                key = f"{name}@{int(aum/1e6)}M@{int(bps)}bps"
                navs[f"{fold}|{key}"] = nav
                row = {"fold": fold, "candidate": name, "aum_m": aum / 1e6, "cost": f"{int(bps)}bps",
                       "cagr": round(cagr(rr.to_numpy()), 4), "maxdd": round(max_dd(rr.to_numpy()), 4),
                       "turnover": round(float(r.metrics.turnover), 4),
                       "failed_orders": int(len(r.failed_orders) if r.failed_orders is not None else 0)}
                rows.append(row)
                # sqrt overlays on the 8bps cells at 1M and 30M
                if bps == 8.0 and aum in (1e6, 3e7):
                    for eta, tag in ((ETA_BASE, "sqrt_base"), (ETA_STRESSED, "sqrt_stressed")):
                        drag = impact_drag(tw, bt_panel, aum, eta).reindex(rr.index).fillna(0.0)
                        rr2 = rr - drag
                        rows.append({**row, "cost": tag,
                                     "cagr": round(cagr(rr2.to_numpy()), 4),
                                     "maxdd": round(max_dd(rr2.to_numpy()), 4)})
                print(f"{fold} {key:28s} CAGR {row['cagr']:+.1%} DD {row['maxdd']:.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "historical_strict_metrics.csv", index=False)
    pd.DataFrame({k: v for k, v in navs.items()}).to_csv(OUT / "nav_by_candidate.csv")
    # aggregates + reproduction check
    agg = (df.groupby(["candidate", "aum_m", "cost"])
             .agg(median_cagr=("cagr", "median"), worst_fold=("cagr", "min"),
                  worst_dd=("maxdd", "max"), med_turnover=("turnover", "median"),
                  failed=("failed_orders", "sum")).reset_index())
    agg.to_csv(OUT / "capacity_report.csv", index=False)
    imp = agg[agg["cost"].str.startswith("sqrt") | (agg["cost"] == "8bps")]
    imp.to_csv(OUT / "impact_sensitivity.csv", index=False)
    for name, ref in REPRO_REF.items():
        med = float(agg[(agg["candidate"] == name) & (agg["aum_m"] == 1.0)
                        & (agg["cost"] == "8bps")]["median_cagr"].iloc[0])
        repro[name] = {"reference": ref, "reproduced": med,
                       "abs_diff": round(abs(med - ref), 4),
                       "pass": bool(abs(med - ref) <= REPRO_TOL)}
    summary = {"repro_check": repro,
               "s4_coverage": "EXP-023 results.json (8/25bps folds) + EXP-024 transfer argument; not re-simulated",
               "s1_coverage": "SEARCH-window record (EXP-000/001, PBO 0.886/DSR 0.919); predictions do not span H-008 folds",
               "no_winner_declared": True,
               "trust": "fixed_cohort_searched_validation / candidate_research_only_not_fresh_holdout_validated",
               "peak_rss_gib": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(v["pass"] for v in repro.values()):
        print("REPRODUCTION FAILED — fail closed"); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
