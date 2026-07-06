#!/usr/bin/env python3
"""D1_low_vol_20 integration test (materialization plan, DUAL_TRACK_FACTOR_BATCH_PLAN.md).

Tilt the corrected C3_ema0.7 carrier by D1's per-date rank at an a-priori weight,
build the L1 min-hold-10 book, and check whether it improves the F2 crash /
worst-DD without wrecking median or turnover. New candidate = D1-tilt (w=0.3);
w=0 reproduces L1 (already counted). N 73->74. Corrected sim, strict variant-C,
H-008 folds, 8/15/25 bps. Zero retrain, zero fresh-holdout.
"""
from __future__ import annotations
import json
import sys
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import FOLDS, TOP_K, build_candidates, sleeve_frame, cagr, max_dd  # noqa: E402
from exp011_book_churn import eligible_rank_lists  # noqa: E402
from dual_track_eval import build_book, avg_holding_days  # noqa: E402
from quantagent.factors import expr as E  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp016_d1_integration"
QUAR = pd.Timestamp("2025-09-01")
BPS = (8.0, 15.0, 25.0)
WEIGHTS = {"L1_baseline_w0": 0.0, "L1_d1tilt_w30": 0.30}
D1 = E.Mul(E.Constant(-1.0), E.TsStd(E.Returns(E.Close, 1), 20))


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
             "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfgs = {b: AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=b) for b in BPS}
    res: dict[str, dict] = {k: {} for k in WEIGHTS}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUAR
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)["C3_ema0.7"].copy()
        # D1 on a padded panel (200d warmup), then slice
        panel = pd.read_parquet(REPO / bp.PANEL, columns=pcols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=210)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        pan = panel.sort_values(["symbol", "trade_date"]).copy()
        pan["d1"] = D1.evaluate(pan).to_numpy()
        d1 = pan[["symbol", "trade_date", "d1"]]
        bt_panel = panel[panel["trade_date"] >= oos_s - pd.Timedelta(days=10)].copy()
        flags = bt_panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(bt_panel["trade_date"].unique())

        c = carrier.merge(d1, on=["symbol", "trade_date"], how="left")
        c["rc"] = c.groupby("trade_date")["alpha_score"].rank(pct=True)
        c["rd"] = c.groupby("trade_date")["d1"].rank(pct=True)
        for name, w in WEIGHTS.items():
            cc = c.copy()
            # names with no D1 (insufficient history) fall back to carrier rank
            cc["blend"] = (1 - w) * cc["rc"] + w * cc["rd"].fillna(cc["rc"])
            score = cc[["trade_date", "symbol"]].copy()
            score["alpha_score"] = cc["blend"].to_numpy()
            p = score.merge(flags, on=["symbol", "trade_date"], how="left")
            days = eligible_rank_lists(p)
            book = build_book(days, "minhold", {"n": 10})
            tw = bp._apply_delay(book, trade_dates, 1)
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all()
            row = {"hold_days": avg_holding_days(book)}
            for b in BPS:
                r = run_strict_backtest_v8(tw, bt_panel, sector_map=sector, config=cfgs[b])
                nav = r.nav.copy(); nav.index = pd.to_datetime(nav.index)
                rr = nav.pct_change().dropna().to_numpy()
                row[f"cagr{int(b)}"] = round(cagr(rr), 4)
                if b == 8.0:
                    row["maxdd"] = round(max_dd(rr), 4)
                    row["turnover"] = round(float(r.metrics.turnover), 4)
            res[name][fold] = row
            print(f"{fold} {name:18s} CAGR8 {row['cagr8']:+.1%} 25 {row['cagr25']:+.1%} "
                  f"DD {row['maxdd']:.1%} turn {row['turnover']:.3f} hold {row['hold_days']:.1f}d", flush=True)

    folds = list(FOLDS)
    summary = {"weights": WEIGHTS, "cumulative_trials_N": 74, "per_fold": res, "agg": {}}
    for name in WEIGHTS:
        cs = [res[name][f]["cagr8"] for f in folds]
        summary["agg"][name] = {
            "median_cagr8": round(float(np.median(cs)), 4), "worst_fold8": round(min(cs), 4),
            "f2_cagr8": round(res[name]["F2"]["cagr8"], 4),
            "median_cagr25": round(float(np.median([res[name][f]["cagr25"] for f in folds])), 4),
            "worst_dd8": round(max(res[name][f]["maxdd"] for f in folds), 4),
            "max_turnover": round(max(res[name][f]["turnover"] for f in folds), 4),
        }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    b, t = summary["agg"]["L1_baseline_w0"], summary["agg"]["L1_d1tilt_w30"]
    print("\n=== D1 tilt (w=0.3) vs L1 baseline (w=0) ===")
    for k in ("median_cagr8", "worst_fold8", "f2_cagr8", "median_cagr25", "worst_dd8", "max_turnover"):
        print(f"  {k:16s} {b[k]:+.4f} -> {t[k]:+.4f}")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
