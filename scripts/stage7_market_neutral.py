#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 7: EXECUTABLE market-neutral test — long top-k basket minus index future.

A-share single-stock shorting is restricted, so the realizable version of the
robust long-short spread is: long the factor-selected top-k basket (long-only,
executable, after-cost, T+1), short an index FUTURE to hedge market beta. This
captures only the LONG leg's excess over the index — not the full spread.

Critical: the hedge index choice isolates (or confounds) factor alpha vs SIZE
beta. The universe is small-cap-tilted, so CSI1000 (sh000852) is the
size-matched hedge that isolates factor alpha; CSI300 (large-cap) would let the
small-vs-large size premium masquerade as alpha. We report all three and judge
robustness on the SIZE-MATCHED hedge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.portfolio.policy_search import PolicyConfig, backtest_policy, prepare_working_frame

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
FUND = "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
LGBM = "runtime/stage6_classical_2018/wf/walkforward_predictions.parquet"
INDEXES = {"CSI1000_sizematch": "csi1000", "CSI500": "csi500", "CSI300_largecap": "csi300"}


def _index_returns() -> dict[str, pd.Series]:
    idx = pd.read_parquet(INDEX)
    idx["observation_date"] = pd.to_datetime(idx["observation_date"], errors="coerce")
    out = {}
    for name, lab in INDEXES.items():
        s = idx[idx["label"] == lab].sort_values("observation_date").set_index("observation_date")["close"]
        out[name] = pd.to_numeric(s, errors="coerce").pct_change()
    return out


def _window_cagr(daily: pd.Series, ws, we) -> float:
    d = daily[(daily.index >= ws) & (daily.index <= we)].dropna()
    if len(d) < 40:
        return float("nan")
    nav = (1.0 + d).prod()
    return float(nav ** (252.0 / len(d)) - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rebalance-days", type=int, default=20)
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--output-dir", default="runtime/stage7_market_neutral")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "amount", "is_st", "is_suspended", "is_limit_up"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[panel["trade_date"] >= args.start].reset_index(drop=True)
    sector = pd.read_parquet(SECTOR)
    idx_rets = _index_returns()

    fund = pd.read_parquet(FUND)
    f = fund[["symbol", "available_at", "eps_basic", "bps"]].copy()
    f["available_at"] = pd.to_datetime(f["available_at"], errors="coerce")
    f = f.dropna(subset=["available_at"]).sort_values("available_at")
    m = pd.merge_asof(panel.sort_values("trade_date"), f, left_on="trade_date", right_on="available_at", by="symbol", direction="backward")
    close = pd.to_numeric(m["close"], errors="coerce")
    bp = pd.to_numeric(m["bps"], errors="coerce") / close
    ey = pd.to_numeric(m["eps_basic"], errors="coerce") / close
    base = m[["symbol", "trade_date"]].copy()
    factors = {"value_book_to_price": bp,
               "value_composite": bp.groupby(m["trade_date"]).rank(pct=True) + ey.groupby(m["trade_date"]).rank(pct=True)}
    try:
        lg = pd.read_parquet(LGBM, columns=["symbol", "trade_date", "alpha_5d"]); lg["trade_date"] = pd.to_datetime(lg["trade_date"])
        factors["momentum_lgbm"] = base.merge(lg, on=["symbol", "trade_date"], how="left")["alpha_5d"]
    except Exception:
        pass

    dates = sorted(panel["trade_date"].dropna().unique())
    windows = [(dates[i], dates[min(i + args.window_days, len(dates)) - 1]) for i in range(0, len(dates), args.window_days)]
    windows = [(s, e) for (s, e) in windows if pd.Index(dates).get_indexer([e])[0] - pd.Index(dates).get_indexer([s])[0] >= 40]

    report = {}
    for name, fac in factors.items():
        preds = base.copy(); preds["alpha_5d"] = np.asarray(fac, dtype=float)
        preds["alpha_1d"] = preds["alpha_5d"]; preds["alpha_20d"] = preds["alpha_5d"]
        work = prepare_working_frame(preds, panel, sector)
        cfg = PolicyConfig(horizon=5, top_k=args.top_k, rebalance_days=args.rebalance_days,
                           side="long_only", transform="csrank", neutralize="none", liquidity_filter="ex_bottom_30pct")
        res = backtest_policy(work, cfg)
        book_ret = res.nav.pct_change()
        book_ret.index = pd.to_datetime(book_ret.index)
        report[name] = {}
        for idx_name, idx_ret in idx_rets.items():
            net = (book_ret - idx_ret.reindex(book_ret.index)).dropna()
            per = [(_window_cagr(net, ws, we)) for (ws, we) in windows]
            per = [x for x in per if np.isfinite(x)]
            arr = np.array(per, dtype=float); n = len(arr)
            if n < 4:
                continue
            mean_n, std_n = float(np.mean(arr)), float(np.std(arr, ddof=1))
            report[name][idx_name] = {
                "n_windows": n, "pct_pos": round(float((arr > 0).mean()) * 100, 1),
                "mean_cagr": round(mean_n, 4), "median_cagr": round(float(np.median(arr)), 4),
                "IR": round(mean_n / std_n, 3) if std_n > 1e-9 else None, "worst": round(float(np.min(arr)), 4),
            }
        # print size-matched verdict
        sm = report[name].get("CSI1000_sizematch", {})
        print(f"  {name:20} | CSI1000(size-matched): %pos {sm.get('pct_pos')} median {sm.get('median_cagr')} IR {sm.get('IR')} worst {sm.get('worst')}", flush=True)

    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== EXECUTABLE MARKET-NEUTRAL (long top-k − index future), by hedge ===")
    for name, d in report.items():
        print(f"\n{name}:")
        for idx_name, s in d.items():
            print(f"  vs {idx_name:18} %pos {s['pct_pos']:>5}  median {s['median_cagr']:+.2%}  IR {s['IR']}  worst {s['worst']:+.2%}")
    print("\n(judge robustness on CSI1000_sizematch — isolates factor alpha from the small-vs-large size premium)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
