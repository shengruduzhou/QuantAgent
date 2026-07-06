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

QUAR = pd.Timestamp("2025-09-01")
BPS = (8.0, 15.0, 25.0)
D1 = E.Mul(E.Constant(-1.0), E.TsStd(E.Returns(E.Close, 1), 20))
FIN = REPO / "runtime/data/v7/gold/training_dataset/tickflow_fin_features.parquet"
QUALITY_COLS = ["roe", "net_margin", "gross_margin"]
# factor -> (output subdir, tilt-label). Default d1 reproduces EXP-016.
TILTS = {
    "d1": ("exp016_d1_integration", "d1tilt"),
    "quality": ("exp017_quality_integration", "qualtilt"),
}


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def tilt_series(pan: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Return [symbol, trade_date, tilt] for the chosen tilt factor. `pan` is a
    padded (200d warmup) price panel sorted by symbol,trade_date."""
    if factor == "d1":
        out = pan[["symbol", "trade_date"]].copy()
        out["tilt"] = D1.evaluate(pan).to_numpy()
        return out
    # quality: PIT-safe fundamentals, +1-day per-symbol lag, rank-mean composite
    fin = pd.read_parquet(FIN, columns=["symbol", "trade_date"] + QUALITY_COLS)
    fin["trade_date"] = pd.to_datetime(fin["trade_date"])
    fin = fin.sort_values(["symbol", "trade_date"])
    fin[QUALITY_COLS] = fin.groupby("symbol", sort=False)[QUALITY_COLS].shift(1)
    m = pan[["symbol", "trade_date"]].merge(fin, on=["symbol", "trade_date"], how="left")
    for c in QUALITY_COLS:
        m[f"_r_{c}"] = m.groupby("trade_date")[c].rank(pct=True)
    m["tilt"] = m[[f"_r_{c}" for c in QUALITY_COLS]].mean(axis=1)
    return m[["symbol", "trade_date", "tilt"]]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor", choices=list(TILTS), default="d1")
    ap.add_argument("--weight", type=float, default=0.30)
    args = ap.parse_args()
    subdir, lbl = TILTS[args.factor]
    OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008" / subdir
    weights = {"L1_baseline_w0": 0.0, f"L1_{lbl}_w{int(args.weight*100)}": args.weight}
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
             "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfgs = {b: AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=b) for b in BPS}
    res: dict[str, dict] = {k: {} for k in weights}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUAR
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)["C3_ema0.7"].copy()
        # tilt factor on a padded panel (200d warmup), then slice
        panel = pd.read_parquet(REPO / bp.PANEL, columns=pcols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=210)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        pan = panel.sort_values(["symbol", "trade_date"]).copy()
        tilt = tilt_series(pan, args.factor)
        bt_panel = panel[panel["trade_date"] >= oos_s - pd.Timedelta(days=10)].copy()
        flags = bt_panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(bt_panel["trade_date"].unique())

        c = carrier.merge(tilt, on=["symbol", "trade_date"], how="left")
        c["rc"] = c.groupby("trade_date")["alpha_score"].rank(pct=True)
        c["rd"] = c.groupby("trade_date")["tilt"].rank(pct=True)
        for name, w in weights.items():
            cc = c.copy()
            # names with no tilt value (insufficient history / no coverage) fall
            # back to carrier rank so the book universe is unchanged
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
    tilt_name = [k for k in weights if k != "L1_baseline_w0"][0]
    summary = {"factor": args.factor, "weight": args.weight, "weights": weights,
               "per_fold": res, "agg": {}}
    for name in weights:
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
    b, t = summary["agg"]["L1_baseline_w0"], summary["agg"][tilt_name]
    print(f"\n=== {args.factor} tilt (w={args.weight}) vs L1 baseline (w=0) ===")
    for k in ("median_cagr8", "worst_fold8", "f2_cagr8", "median_cagr25", "worst_dd8", "max_turnover"):
        print(f"  {k:16s} {b[k]:+.4f} -> {t[k]:+.4f}")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
