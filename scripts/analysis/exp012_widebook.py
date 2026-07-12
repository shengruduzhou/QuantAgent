#!/usr/bin/env python3
"""EXP-012 / H-012: k=30 wide-book structural robustness.

Pre-registered (HYPOTHESIS_REGISTRY.md H-012, frozen at commit 66560e2, N=3):
  W1_c3ema_k30   C3_ema0.7 scores, plain top-30 equal weight
  W2_c3ema_k30_partial  W1 + B3 partial adjustment (0.7/0.3, prune 0.05/30)
  W3_c2_k30      C2_prod_rank110 scores, plain top-30 equal weight

Evaluation: strict variant-C on the H-008 folds at bps in {8, 9, 10} per
candidate (noise band = fold-CAGR range across bps points, REPORT-ONLY;
gates judged at the 8bps point, consistent with EXP-008..011). The k=10
carrier C3_ema0.7 runs the same 3 bps points as the reference noise band.
Gates: identical to H-011 (turn<=0.10, worstDD<=0.2503, F2>=-0.249,
median>=0.2802, sector<=0.33, no leverage, quarantine). PBO/DSR at N=63.
"""
from __future__ import annotations

import itertools
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import (  # noqa: E402
    FOLDS, build_candidates, norm_cdf, norm_ppf, sleeve_frame,
)
from exp009_exposure_overlay import CARRIER  # noqa: E402
from exp011_book_churn import eligible_rank_lists, build_book  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp012_widebook"
QUARANTINE_START = pd.Timestamp("2025-09-01")
EULER = 0.5772156649015329
K_WIDE = 30
BPS_POINTS = (8.0, 9.0, 10.0)
N_TRIALS = 63
CANDS = ("W1_c3ema_k30", "W2_c3ema_k30_partial", "W3_c2_k30", "REF_c3ema_k10")

BASE = {"worst_dd": 0.2503, "f2_floor": -0.249, "turn_cap": 0.10,
        "median_floor": 0.2802, "sector_cap": 0.33}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    smap = dict(zip(sector["symbol"].astype(str), sector.iloc[:, 1].astype(str))) if len(sector.columns) > 1 else {}
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]

    results: dict[str, dict] = {c: {} for c in CANDS}          # 8bps gate metrics
    band: dict[str, dict[str, dict]] = {c: {} for c in CANDS}  # per-bps CAGR
    daily: dict[str, dict[str, pd.Series]] = {c: {} for c in CANDS}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START
        frame = sleeve_frame(fold)
        cands_scores = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())

        books: dict[str, pd.DataFrame] = {}
        p_c3 = cands_scores[CARRIER].merge(flags, on=["symbol", "trade_date"], how="left")
        p_c2 = cands_scores["C2_prod_rank110"].merge(flags, on=["symbol", "trade_date"], how="left")
        books["W1_c3ema_k30"] = bp._target_weights(p_c3, "alpha_score", K_WIDE, eligible_only=True,
                                                   delay_days=1, trade_dates=trade_dates)
        books["W2_c3ema_k30_partial"] = bp._apply_delay(
            build_book(eligible_rank_lists(p_c3), "B3_partial30", k=K_WIDE), trade_dates, 1)
        books["W3_c2_k30"] = bp._target_weights(p_c2, "alpha_score", K_WIDE, eligible_only=True,
                                                delay_days=1, trade_dates=trade_dates)
        books["REF_c3ema_k10"] = bp._target_weights(p_c3, "alpha_score", 10, eligible_only=True,
                                                    delay_days=1, trade_dates=trade_dates)

        for name, tw in books.items():
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all(), "leverage breach"
            band[name][fold] = {}
            for bps in BPS_POINTS:
                cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=bps)
                res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
                m = res.metrics
                band[name][fold][f"{bps:g}bps"] = round(m.annualized_return, 4)
                if bps == 8.0:
                    nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
                    daily[name][fold] = nav.pct_change().dropna()
                    tw_long = tw.stack()
                    tw_long = tw_long[tw_long > 0].rename("w").reset_index()
                    tw_long.columns = ["trade_date", "symbol", "w"]
                    tw_long["sec"] = tw_long["symbol"].astype(str).map(smap).fillna("?")
                    sec_max = float(tw_long.groupby(["trade_date", "sec"])["w"].sum()
                                    .groupby("trade_date").max().mean()) if len(tw_long) else 0.0
                    results[name][fold] = {
                        "cagr": round(m.annualized_return, 4), "maxdd": round(m.max_drawdown, 4),
                        "sharpe": round(m.sharpe, 3), "turnover": round(float(m.turnover), 4),
                        "mean_max_sector_weight": round(sec_max, 3),
                    }
            cs = band[name][fold]
            spread = round(max(cs.values()) - min(cs.values()), 4)
            band[name][fold]["spread"] = spread
            print(f"{fold} {name:22s} 8bps {results[name][fold]['cagr']:+.1%} "
                  f"DD {results[name][fold]['maxdd']:.1%} turn {results[name][fold]['turnover']:.3f} "
                  f"| band {cs['8bps']:+.3f}/{cs['9bps']:+.3f}/{cs['10bps']:+.3f} spread {spread:.3f}",
                  flush=True)

    folds = list(FOLDS)
    growth = np.array([[float(np.log1p(daily[n][f]).sum()) for n in CANDS] for f in folds])
    lam = []
    for combo in itertools.combinations(range(len(folds)), 2):
        mask = np.zeros(len(folds), dtype=bool); mask[list(combo)] = True
        tr, te = growth[mask].sum(0), growth[~mask].sum(0)
        w = int(np.argmax(tr))
        omega = float((te <= te[w]).sum()) / (len(CANDS) + 1)
        lam.append(math.log(omega / (1 - omega)))
    pbo = round(float((np.array(lam) <= 0).mean()), 3)

    def dsr(name: str) -> float:
        r = pd.concat([daily[name][f] for f in folds]).to_numpy()
        sr = float(r.mean() / r.std(ddof=1))
        z = (r - r.mean()) / r.std(ddof=1)
        g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
        srs = [float(pd.concat([daily[n][f] for f in folds]).to_numpy().mean()
                     / pd.concat([daily[n][f] for f in folds]).to_numpy().std(ddof=1))
               for n in CANDS]
        v = float(np.var(srs, ddof=1))
        sr0 = math.sqrt(v) * ((1 - EULER) * norm_ppf(1 - 1 / N_TRIALS)
                              + EULER * norm_ppf(1 - 1 / (N_TRIALS * math.e)))
        denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        return round(norm_cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)

    summary: dict[str, object] = {
        "protocol": "HYPOTHESIS_REGISTRY.md H-012 (commit 66560e2)",
        "baseline_frozen": BASE, "cumulative_trials_N": N_TRIALS,
        "candidates": {}, "noise_bands": band,
        "fold_block_pbo": pbo, "dsr_stitched": {n: dsr(n) for n in CANDS},
    }
    for name in CANDS:
        cs = [results[name][f]["cagr"] for f in folds]
        dd = [results[name][f]["maxdd"] for f in folds]
        to = [results[name][f]["turnover"] for f in folds]
        sc = [results[name][f]["mean_max_sector_weight"] for f in folds]
        med = float(np.median(cs))
        gates = {
            "G1_turnover": max(to) <= BASE["turn_cap"],
            "G2_worst_dd": max(dd) <= BASE["worst_dd"],
            "G3_f2_material": results[name]["F2"]["cagr"] >= BASE["f2_floor"],
            "G4_median": med >= BASE["median_floor"],
            "G5_sector": max(sc) <= BASE["sector_cap"],
            "G6_no_leverage": True, "G7_quarantine": True,
        }
        summary["candidates"][name] = {
            "fold_cagrs": cs, "median_cagr": round(med, 4), "min_cagr": round(min(cs), 4),
            "worst_dd": round(max(dd), 4), "max_turnover": round(max(to), 4),
            "max_sector": round(max(sc), 3), "per_fold": results[name],
            "max_bps_spread": round(max(band[name][f]["spread"] for f in folds), 4),
            "gates": gates, "all_gates": all(gates.values()),
        }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for name in CANDS:
        for f in folds:
            rows.append({"candidate": name, "fold": f, **results[name][f],
                         **{k: v for k, v in band[name][f].items()}})
    pd.DataFrame(rows).to_csv(OUT / "widebook_fold_metrics.csv", index=False)
    pd.DataFrame({n: pd.concat([daily[n][f] for f in folds]) for n in CANDS}) \
        .to_csv(OUT / "stitched_daily_returns.csv")
    print(json.dumps({n: {"gates": summary["candidates"][n]["gates"],
                          "all": summary["candidates"][n]["all_gates"],
                          "spread": summary["candidates"][n]["max_bps_spread"]}
                      for n in CANDS}, indent=2))
    print(json.dumps({"pbo": pbo, "dsr": summary["dsr_stitched"]}, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
