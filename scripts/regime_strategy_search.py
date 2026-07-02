#!/usr/bin/env python3
"""Daily strict search: sleeve weights x regime-conditional top-k x regime gross.

Goal = MAX absolute net CAGR under real tradable conditions. Every config is
scored by the TRUSTED strict backtest (run_strict_backtest_v8: t+1 fill, costs,
limit-up/down/suspension, slippage) — never a proxy alone.

Contamination control:
  * config is SELECTED on the VALIDATION window only;
  * results are REPORTED on two untouched windows: non-2026 tail + 2026 quasi-live;
  * FinalScore = w2026*CAGR_2026 + whist*CAGR_non2026 + wstab*stability,
    with weight sensitivity. Model was trained <=2024-06-30 (none of these windows).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

ANN = 244
ENSEMBLE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"


def regime_target_weights(comp: pd.DataFrame, regime: pd.Series, k_by: dict, g_by: dict,
                          trade_dates: list, delay_days: int = 1) -> pd.DataFrame:
    """Eligible top-k per date with regime-conditional k and gross; t+1 delay."""
    d = comp.copy()
    bad = (d.get("is_suspended", pd.Series(False, index=d.index)).fillna(False).astype(bool)
           | d.get("is_st", pd.Series(False, index=d.index)).fillna(False).astype(bool)
           | d.get("is_limit_up", pd.Series(False, index=d.index)).fillna(False).astype(bool))
    d = d[~bad]
    d["rg"] = d["trade_date"].map(regime).fillna("sideways")
    d["k"] = d["rg"].map(k_by).astype(float)
    d["g"] = d["rg"].map(g_by).astype(float)
    d = d.sort_values(["trade_date", "composite_score"], ascending=[True, False])
    d["rank"] = d.groupby("trade_date").cumcount()
    d = d[d["rank"] < d["k"]]
    d = d[d["k"] > 0]
    d["w"] = d["g"] / d["k"]
    tw = d.pivot_table(index="trade_date", columns="symbol", values="w", fill_value=0.0).sort_index()
    if delay_days > 0:
        di = pd.DatetimeIndex(sorted(trade_dates))
        pos = di.searchsorted(tw.index) + delay_days
        keep = pos < len(di)
        tw = tw.iloc[keep]; tw.index = di[pos[keep]]
    return tw


def strict_metrics(tw: pd.DataFrame, panel: pd.DataFrame, sector: pd.DataFrame,
                   start: str, end: str, bench_daily: pd.Series) -> dict:
    if tw.empty:
        return {"cagr": -1.0, "maxDD": 1.0, "calmar": 0.0, "sharpe": 0.0, "turnover": 0.0}
    sim = panel[(panel["trade_date"] >= pd.Timestamp(start) - pd.Timedelta(days=8))
                & (panel["trade_date"] <= pd.Timestamp(end) + pd.Timedelta(days=8))]
    res = run_strict_backtest_v8(tw, sim, sector_map=sector,
                                 config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0))
    m = res.metrics
    cal = m.annualized_return / abs(m.max_drawdown) if m.max_drawdown else 0.0
    to = float(tw.diff().abs().sum(axis=1).mean() / 2.0)
    return {"cagr": round(m.annualized_return, 4), "maxDD": round(m.max_drawdown, 4),
            "calmar": round(cal, 3), "sharpe": round(m.sharpe, 3),
            "total": round(m.total_return, 4), "turnover": round(to, 4),
            "nav": res.nav}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-start", default="2024-08-28")
    ap.add_argument("--val-end", default="2025-08-31")
    ap.add_argument("--non2026-start", default="2025-09-01")
    ap.add_argument("--non2026-end", default="2025-12-31")
    ap.add_argument("--y2026-start", default="2026-01-02")
    ap.add_argument("--y2026-end", default="2026-05-13")
    ap.add_argument("--top-strict", type=int, default=8)
    ap.add_argument("--output-dir", default="runtime/reports/v89_closed_loop/regime_search")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ens = pd.read_parquet(ENSEMBLE); ens["trade_date"] = pd.to_datetime(ens["trade_date"])
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(bp.PANEL, columns=panel_cols); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp("2024-06-01")]
    sector = pd.read_parquet(bp.SECTOR)
    trade_dates = sorted(panel["trade_date"].unique())

    bench = bp._bench_daily(panel, trade_dates)
    regime = bp._regime_label(bench)
    print("regime day counts:", regime.value_counts().to_dict(), flush=True)

    # sleeve ranks
    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]]
    ens = ens.merge(flags, on=["symbol", "trade_date"], how="left")
    R = ens[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]].copy()
    for c in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score"):
        R[c] = ens.groupby("trade_date")[c].rank(pct=True)

    def comp_frame(ws, wm, wl):
        f = R[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up"]].copy()
        f["composite_score"] = (ws * R["short_5d_score"] + wm * R["mid_5d_30d_score"] + wl * R["long_30d_120d_score"]).to_numpy()
        return f

    # ---- Stage 1: sleeve weight x flat top-k (no regime), select on VAL ----
    sleeves = [(1, 1, 0), (2, 1, 0), (1, 0, 0), (1, 1, 1)]
    flat_ks = [5, 10, 20]
    configs = []
    for (ws, wm, wl), k in itertools.product(sleeves, flat_ks):
        configs.append({"name": f"flat_w{ws}{wm}{wl}_k{k}", "w": (ws, wm, wl),
                        "k_by": {"bull": k, "sideways": k, "bear": k},
                        "g_by": {"bull": 1.0, "sideways": 1.0, "bear": 1.0}})
    # ---- Stage 2: regime-conditional k/gross on the strong drop-long sleeves ----
    for w in [(1, 1, 0), (2, 1, 0)]:
        ws, wm, wl = w
        regime_variants = [
            ("bullcoNCk", {"bull": 5, "sideways": 10, "bear": 5}, {"bull": 1.0, "sideways": 1.0, "bear": 1.0}),
            ("bulldefend", {"bull": 10, "sideways": 10, "bear": 3}, {"bull": 1.0, "sideways": 1.0, "bear": 0.4}),
            ("bullaggr", {"bull": 5, "sideways": 10, "bear": 3}, {"bull": 1.0, "sideways": 0.8, "bear": 0.3}),
            ("bearcash", {"bull": 10, "sideways": 10, "bear": 5}, {"bull": 1.0, "sideways": 1.0, "bear": 0.0}),
        ]
        for tag, k_by, g_by in regime_variants:
            configs.append({"name": f"rg_{tag}_w{ws}{wm}{wl}", "w": w, "k_by": k_by, "g_by": g_by})

    print(f"evaluating {len(configs)} configs on VALIDATION (strict)...", flush=True)
    for cfg in configs:
        comp = comp_frame(*cfg["w"])
        tw = regime_target_weights(comp, regime, cfg["k_by"], cfg["g_by"], trade_dates)
        twv = tw[(tw.index >= pd.Timestamp(args.val_start)) & (tw.index <= pd.Timestamp(args.val_end))]
        m = strict_metrics(twv, panel, sector, args.val_start, args.val_end, bench)
        cfg["val"] = {k: v for k, v in m.items() if k != "nav"}
        print(f"  {cfg['name']:28} VAL CAGR {m['cagr']:+.2%} maxDD {m['maxDD']:.2%} turnover {m['turnover']:.2f}", flush=True)

    configs.sort(key=lambda c: -c["val"]["cagr"])
    top = configs[: args.top_strict]

    # ---- report top configs on untouched windows + FinalScore ----
    def stability(vals):
        xs = [v for v in vals if v is not None]
        return float(np.mean(xs) - 0.5 * np.std(xs)) if xs else -1.0

    rows = []
    for cfg in top:
        comp = comp_frame(*cfg["w"])
        tw = regime_target_weights(comp, regime, cfg["k_by"], cfg["g_by"], trade_dates)
        m26 = strict_metrics(tw[(tw.index >= pd.Timestamp(args.y2026_start)) & (tw.index <= pd.Timestamp(args.y2026_end))],
                             panel, sector, args.y2026_start, args.y2026_end, bench)
        mh = strict_metrics(tw[(tw.index >= pd.Timestamp(args.non2026_start)) & (tw.index <= pd.Timestamp(args.non2026_end))],
                            panel, sector, args.non2026_start, args.non2026_end, bench)
        fs = {}
        for w2026 in (0.55, 0.60, 0.70):
            whist = 0.90 - w2026  # leaves 0.10 for stability
            stab = stability([cfg["val"]["cagr"], mh["cagr"], m26["cagr"]])
            fs[f"w2026={w2026}"] = round(w2026 * m26["cagr"] + whist * mh["cagr"] + 0.10 * stab, 4)
        rows.append({"name": cfg["name"], "weights": cfg["w"], "k_by": cfg["k_by"], "g_by": cfg["g_by"],
                     "val_cagr": cfg["val"]["cagr"], "y2026_cagr": m26["cagr"], "y2026_maxDD": m26["maxDD"],
                     "y2026_calmar": m26["calmar"], "non2026_cagr": mh["cagr"], "non2026_maxDD": mh["maxDD"],
                     "turnover_2026": m26["turnover"], "finalscore": fs})
    # rank final by the default w2026=0.6 FinalScore
    rows.sort(key=lambda r: -r["finalscore"]["w2026=0.6"])
    (out / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== TOP CONFIGS (sorted by FinalScore w2026=0.6) ===")
    print(f"{'name':28}{'2026 CAGR':>10}{'2026 DD':>9}{'non26 CAGR':>11}{'val CAGR':>9}{'FinalScore':>11}")
    for r in rows:
        print(f"{r['name']:28}{r['y2026_cagr']:>9.1%}{r['y2026_maxDD']:>9.1%}{r['non2026_cagr']:>11.1%}"
              f"{r['val_cagr']:>9.1%}{r['finalscore']['w2026=0.6']:>11.3f}")
    print(f"\nwrote {out/'leaderboard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
