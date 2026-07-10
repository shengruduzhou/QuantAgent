#!/usr/bin/env python3
"""EXP-024: capacity study of the FROZEN champions (diagnostic, no selection).

Measures how the frozen books degrade with AUM under the trusted corrected
simulator's 10% volume-participation cap. No candidate is tuned or selected;
configs are the frozen champions from FRESH_HOLDOUT_FREEZE_MANIFEST.md.

Evidence-based AUM grid (holdings' median daily amount 23-39M CNY, median
liquidity rank ~0.10): 1M (historic baseline) / 10M / 30M / 100M / 300M CNY.
8 bps everywhere + 25 bps at 30M/100M as a conservative impact proxy (the
simulator has no nonlinear impact model -- stated honestly in the report).

Books: L1_c3ema07_minhold10 (raw-CAGR champion) and L1+D1_regime w=0.5
(hand-designed risk champion). RW1_4state shares the same carrier pool /
turnover / holding profile (EXP-023 results.json) -- capacity conclusions
transfer; not re-derived here to keep the learner out of a diagnostic script.
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
from exp008_walkforward_eval import FOLDS, build_candidates, sleeve_frame, cagr, max_dd  # noqa: E402
from exp011_book_churn import eligible_rank_lists  # noqa: E402
from exp009_exposure_overlay import bench_series  # noqa: E402
from exp010_hysteresis_overlay import gross_series  # noqa: E402
from dual_track_eval import build_book  # noqa: E402
from dual_track_d1_integration import tilt_series  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

QUAR = pd.Timestamp("2025-09-01")
OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp024_capacity_study"
AUM_GRID = (1e6, 1e7, 3e7, 1e8, 3e8)
BPS25_AT = (3e7, 1e8)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def fold_books(fold: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list]:
    """Frozen champion target-weight matrices for one fold + panel."""
    spec = FOLDS[fold]
    oos_s, oos_e = map(pd.Timestamp, spec["oos"])
    assert oos_e < QUAR
    frame = sleeve_frame(fold)
    carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)["C3_ema0.7"].copy()
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
             "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(REPO / bp.PANEL, columns=pcols,
                            filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=210)),
                                     ("trade_date", "<=", oos_e)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    pan = panel.sort_values(["symbol", "trade_date"]).copy()
    bt_panel = panel[panel["trade_date"] >= oos_s - pd.Timedelta(days=10)].copy()
    flags = bt_panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    trade_dates = sorted(bt_panel["trade_date"].unique())

    c = carrier.merge(tilt_series(pan, "d1"), on=["symbol", "trade_date"], how="left")
    c["rc"] = c.groupby("trade_date")["alpha_score"].rank(pct=True)
    c["rd"] = c.groupby("trade_date")["tilt"].rank(pct=True)
    regime = gross_series(bench_series(oos_s, oos_e), "R2a_confirm5")
    wser = (regime < 1.0).astype(float) * 0.5  # frozen EXP-019 weights

    books = {}
    for name in ("L1", "L1_d1_regime"):
        cc = c.copy()
        w = cc["trade_date"].map(wser).fillna(0.0).to_numpy() if name == "L1_d1_regime" else 0.0
        cc["blend"] = (1 - w) * cc["rc"] + w * cc["rd"].fillna(cc["rc"])
        score = cc[["trade_date", "symbol"]].copy()
        score["alpha_score"] = cc["blend"].to_numpy()
        p = score.merge(flags, on=["symbol", "trade_date"], how="left")
        book = build_book(eligible_rank_lists(p), "minhold", {"n": 10})
        books[name] = bp._apply_delay(book, trade_dates, 1)
    return books, bt_panel, trade_dates


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    res: dict[str, dict] = {}
    for fold in FOLDS:
        books, bt_panel, _ = fold_books(fold)
        for name, tw in books.items():
            for aum in AUM_GRID:
                runs = [8.0] + ([25.0] if aum in BPS25_AT else [])
                for bps in runs:
                    cfg = AShareExecutionSimulationConfig(initial_cash=float(aum), slippage_bps=bps)
                    r = run_strict_backtest_v8(tw, bt_panel, sector_map=sector, config=cfg)
                    nav = r.nav.copy(); nav.index = pd.to_datetime(nav.index)
                    rr = nav.pct_change().dropna().to_numpy()
                    fo = r.failed_orders if r.failed_orders is not None else pd.DataFrame()
                    n_failed = int(len(fo))
                    val_col = next((c for c in ("order_value", "value", "notional", "amount")
                                    if c in fo.columns), None)
                    failed_val = float(fo[val_col].abs().sum()) if (n_failed and val_col) else 0.0
                    key = f"{name}@{int(aum/1e6)}M@{int(bps)}bps"
                    res.setdefault(key, {})[fold] = {
                        "cagr": round(cagr(rr), 4), "maxdd": round(max_dd(rr), 4),
                        "turnover": round(float(r.metrics.turnover), 4),
                        "failed_orders": n_failed,
                        "failed_value_over_aum": round(failed_val / aum, 3),
                    }
                    print(f"{fold} {key:26s} CAGR {res[key][fold]['cagr']:+.1%} "
                          f"DD {res[key][fold]['maxdd']:.1%} failed {n_failed} "
                          f"({res[key][fold]['failed_value_over_aum']:.1f}x AUM)", flush=True)

    folds = list(FOLDS)
    agg = {}
    for key, per in res.items():
        cs = [per[f]["cagr"] for f in folds]
        agg[key] = {"median_cagr": round(float(np.median(cs)), 4),
                    "worst_fold": round(min(cs), 4),
                    "worst_dd": round(max(per[f]["maxdd"] for f in folds), 4),
                    "total_failed": int(sum(per[f]["failed_orders"] for f in folds))}
    summary = {"grid_basis": "book median ADV 23-39M CNY, median liq rank ~0.10 (bottom decile)",
               "per_fold": res, "agg": agg,
               "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== EXP-024 capacity aggregates ===")
    for key in sorted(agg, key=lambda k: (k.split("@")[0], float(k.split("@")[1][:-1]))):
        a = agg[key]
        print(f"  {key:26s} med {a['median_cagr']:+.1%} worst {a['worst_fold']:+.1%} "
              f"DD {a['worst_dd']:.1%} failed {a['total_failed']}")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
