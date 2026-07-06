#!/usr/bin/env python3
"""EXP-013 / H-013: low-churn book x fast regime de-risk synthesis.

FINAL H-008-fold batch of this cycle (hard stop-clause: after this run the
four folds are frozen until the FRESH-window first read ~2026-11).

Pre-registered (HYPOTHESIS_REGISTRY.md H-013, frozen at commit 4fd2b20, N=2):
  S1_instant  W2 book (C3_ema0.7 @k30 partial-adjust 0.7/0.3) x R2a confirm-5
              MA60 gross {1.0, 0.5}, instant switching (EXP-010 semantics)
  S2_ramp25   same book, gross moves toward state by at most 0.25/day

Evaluation identical to EXP-012 (bps in {8,9,10}; gates at 8bps; H-011 gate
set G1-G7). PBO/DSR at N=65 in a 4-book pool {S1, S2, W2 no-overlay, k10
carrier}. W2/carrier daily returns reused from the frozen EXP-012 artifacts.
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
from exp008_walkforward_eval import FOLDS, build_candidates, norm_cdf, norm_ppf, sleeve_frame  # noqa: E402
from exp009_exposure_overlay import CARRIER, bench_series  # noqa: E402
from exp010_hysteresis_overlay import gross_series  # noqa: E402
from exp011_book_churn import eligible_rank_lists, build_book  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp013_synthesis"
EXP012 = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp012_widebook"
QUARANTINE_START = pd.Timestamp("2025-09-01")
EULER = 0.5772156649015329
K_WIDE = 30
BPS_POINTS = (8.0, 9.0, 10.0)
N_TRIALS = 65
RULES = ("S1_instant", "S2_ramp25")
POOL = ("S1_instant", "S2_ramp25", "W2_c3ema_k30_partial", "REF_c3ema_k10")

BASE = {"worst_dd": 0.2503, "f2_floor": -0.249, "turn_cap": 0.10,
        "median_floor": 0.2802, "sector_cap": 0.33}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def apply_gross(state: pd.Series, mode: str) -> pd.Series:
    """State series is already t-1 shifted (exp010 gross_series)."""
    if mode == "S1_instant":
        out = state
    elif mode == "S2_ramp25":
        g = np.empty(len(state))
        cur = 1.0
        tgt = state.to_numpy()
        for i in range(len(state)):
            cur += float(np.clip(tgt[i] - cur, -0.25, 0.25))
            g[i] = cur
        out = pd.Series(g, index=state.index)
    else:
        raise ValueError(mode)
    assert out.between(0.5 - 1e-9, 1.0 + 1e-9).all()
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    smap = dict(zip(sector["symbol"].astype(str), sector.iloc[:, 1].astype(str))) if len(sector.columns) > 1 else {}
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]

    results: dict[str, dict] = {r: {} for r in RULES}
    band: dict[str, dict[str, dict]] = {r: {} for r in RULES}
    daily: dict[str, dict[str, pd.Series]] = {r: {} for r in RULES}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)[CARRIER]
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        p = carrier.merge(flags, on=["symbol", "trade_date"], how="left")
        w2_tw = bp._apply_delay(build_book(eligible_rank_lists(p), "B3_partial30", k=K_WIDE),
                                trade_dates, 1)
        bench = bench_series(oos_s, oos_e)
        state = gross_series(bench, "R2a_confirm5").reindex(w2_tw.index).fillna(1.0)

        for rule in RULES:
            tw = w2_tw.mul(apply_gross(state, rule), axis=0)
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all(), "leverage breach"
            band[rule][fold] = {}
            for bps in BPS_POINTS:
                cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=bps)
                res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
                m = res.metrics
                band[rule][fold][f"{bps:g}bps"] = round(m.annualized_return, 4)
                if bps == 8.0:
                    nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
                    daily[rule][fold] = nav.pct_change().dropna()
                    tw_long = tw.stack()
                    tw_long = tw_long[tw_long > 0].rename("w").reset_index()
                    tw_long.columns = ["trade_date", "symbol", "w"]
                    tw_long["sec"] = tw_long["symbol"].astype(str).map(smap).fillna("?")
                    sec_max = float(tw_long.groupby(["trade_date", "sec"])["w"].sum()
                                    .groupby("trade_date").max().mean()) if len(tw_long) else 0.0
                    results[rule][fold] = {
                        "cagr": round(m.annualized_return, 4), "maxdd": round(m.max_drawdown, 4),
                        "sharpe": round(m.sharpe, 3), "turnover": round(float(m.turnover), 4),
                        "mean_max_sector_weight": round(sec_max, 3),
                        "mean_gross": round(float(apply_gross(state, rule)
                                                  .loc[lambda s: s.index >= oos_s].mean()), 3),
                    }
            cs = band[rule][fold]
            spread = round(max(v for k, v in cs.items() if k.endswith("bps"))
                           - min(v for k, v in cs.items() if k.endswith("bps")), 4)
            band[rule][fold]["spread"] = spread
            print(f"{fold} {rule:12s} 8bps {results[rule][fold]['cagr']:+.1%} "
                  f"DD {results[rule][fold]['maxdd']:.1%} turn {results[rule][fold]['turnover']:.3f} "
                  f"gross {results[rule][fold]['mean_gross']} spread {spread:.3f}", flush=True)

    # PBO/DSR pool: reuse frozen EXP-012 stitched daily returns for W2 + k10 carrier
    st12 = pd.read_csv(EXP012 / "stitched_daily_returns.csv", index_col=0, parse_dates=True)
    daily_all: dict[str, dict[str, pd.Series]] = {**daily}
    for name in ("W2_c3ema_k30_partial", "REF_c3ema_k10"):
        daily_all[name] = {
            f: st12[name].loc[pd.Timestamp(s["oos"][0]):pd.Timestamp(s["oos"][1])].dropna()
            for f, s in FOLDS.items()
        }
    folds = list(FOLDS)
    growth = np.array([[float(np.log1p(daily_all[n][f]).sum()) for n in POOL] for f in folds])
    lam = []
    for combo in itertools.combinations(range(len(folds)), 2):
        mask = np.zeros(len(folds), dtype=bool); mask[list(combo)] = True
        tr, te = growth[mask].sum(0), growth[~mask].sum(0)
        w = int(np.argmax(tr))
        omega = float((te <= te[w]).sum()) / (len(POOL) + 1)
        lam.append(math.log(omega / (1 - omega)))
    pbo = round(float((np.array(lam) <= 0).mean()), 3)

    def dsr(name: str) -> float:
        r = pd.concat([daily_all[name][f] for f in folds]).to_numpy()
        sr = float(r.mean() / r.std(ddof=1))
        z = (r - r.mean()) / r.std(ddof=1)
        g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
        srs = [float(pd.concat([daily_all[n][f] for f in folds]).to_numpy().mean()
                     / pd.concat([daily_all[n][f] for f in folds]).to_numpy().std(ddof=1))
               for n in POOL]
        v = float(np.var(srs, ddof=1))
        sr0 = math.sqrt(v) * ((1 - EULER) * norm_ppf(1 - 1 / N_TRIALS)
                              + EULER * norm_ppf(1 - 1 / (N_TRIALS * math.e)))
        denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        return round(norm_cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)

    summary: dict[str, object] = {
        "protocol": "HYPOTHESIS_REGISTRY.md H-013 (commit 4fd2b20) — FINAL fold batch",
        "baseline_frozen": BASE, "cumulative_trials_N": N_TRIALS,
        "rules": {}, "noise_bands": band,
        "fold_block_pbo_pool4": pbo, "dsr_stitched": {n: dsr(n) for n in POOL},
    }
    for rule in RULES:
        cs = [results[rule][f]["cagr"] for f in folds]
        dd = [results[rule][f]["maxdd"] for f in folds]
        to = [results[rule][f]["turnover"] for f in folds]
        sc = [results[rule][f]["mean_max_sector_weight"] for f in folds]
        med = float(np.median(cs))
        gates = {
            "G1_turnover": max(to) <= BASE["turn_cap"],
            "G2_worst_dd": max(dd) <= BASE["worst_dd"],
            "G3_f2_material": results[rule]["F2"]["cagr"] >= BASE["f2_floor"],
            "G4_median": med >= BASE["median_floor"],
            "G5_sector": max(sc) <= BASE["sector_cap"],
            "G6_no_leverage": True, "G7_quarantine": True,
        }
        summary["rules"][rule] = {
            "fold_cagrs": cs, "median_cagr": round(med, 4), "min_cagr": round(min(cs), 4),
            "worst_dd": round(max(dd), 4), "max_turnover": round(max(to), 4),
            "max_sector": round(max(sc), 3),
            "max_bps_spread": round(max(band[rule][f]["spread"] for f in folds), 4),
            "per_fold": results[rule], "gates": gates, "all_gates": all(gates.values()),
        }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for rule in RULES:
        for f in folds:
            rows.append({"rule": rule, "fold": f, **results[rule][f], **band[rule][f]})
    pd.DataFrame(rows).to_csv(OUT / "synthesis_fold_metrics.csv", index=False)
    pd.DataFrame({n: pd.concat([daily_all[n][f] for f in folds]) for n in POOL}) \
        .to_csv(OUT / "stitched_daily_returns.csv")
    print(json.dumps({r: {"gates": summary["rules"][r]["gates"],
                          "all": summary["rules"][r]["all_gates"]} for r in RULES}, indent=2))
    print(json.dumps({"pbo": pbo, "dsr": summary["dsr_stitched"]}, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
