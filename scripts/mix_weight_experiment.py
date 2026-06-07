#!/usr/bin/env python3
"""混合权重实验 — 在多个 as-of 日期上扫 (因子名额, 产业链名额, 加权方式)，用 PIT 前视隔离的
前瞻收益评估，挑出【跨日期稳健】的并集配置。回答用户的"权重/混合方法要反复实验"。

依赖每个日期的:
  - runtime/tmp/real_preds_<YYYYMMDD>.parquet            (因子, symbol+prediction)
  - runtime/reports/monthly/chain_candidates_<date>.parquet (产业链全量候选, 含 chain_conviction)
前瞻收益来自 market_panel 的 close (forward N 交易日)；基准=当日等权全A平均。

输出 runtime/reports/v8/mix_weight_experiment.json + 控制台稳健性排名。
不跑 LLM (复用已生成的产业链候选池)。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRED_SRC = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
CHAIN_DIR = Path("runtime/reports/monthly")


def _code6(s: str) -> str:
    return str(s).split(".")[0].zfill(6)


def _fwd_returns(panel: pd.DataFrame, tds, as_of: pd.Timestamp, fwd_td: int):
    future = [t for t in tds if t > as_of][:fwd_td]
    if not future:
        return None, None
    c0 = panel[panel.trade_date == as_of].set_index("symbol")["close"]
    c1 = panel[panel.trade_date == future[-1]].set_index("symbol")["close"]
    fwd = (c1 / c0 - 1.0).dropna()
    return fwd, future[-1]


def _port_excess(syms, weights, fwd, bench):
    """加权组合前瞻超额(%). syms 与 weights 对齐; 缺收益的剔除后重新归一。"""
    s = pd.Series(weights, index=syms)
    s = s[s.index.isin(fwd.index)]
    if s.empty:
        return None
    s = s / s.sum()
    r = float((s * fwd.reindex(s.index)).sum())
    return round((r - bench) * 100, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", default=["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    ap.add_argument("--fwd-td", type=int, default=10, help="前瞻交易日数 (周频~5, 双周~10)")
    ap.add_argument("--n-factor", nargs="+", type=int, default=[12, 16, 20, 25])
    ap.add_argument("--n-chain", nargs="+", type=int, default=[0, 6, 8, 10, 12])
    ap.add_argument("--weighting", nargs="+", default=["equal", "factor_tilt", "mix"])
    ap.add_argument("--suffix", default="", help="产业链候选池文件后缀 (如 _nonews)")
    args = ap.parse_args()

    allp = pd.read_parquet(PRED_SRC); allp["trade_date"] = pd.to_datetime(allp["trade_date"])
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    tds = sorted(panel["trade_date"].unique())

    # 预载每个日期的因子+产业链候选+前瞻收益
    per_date = {}
    for d in args.dates:
        ts = pd.Timestamp(d)
        preds = allp[allp.trade_date == ts][["symbol", "alpha_score"]].rename(columns={"alpha_score": "prediction"})
        if preds.empty:
            print(f"[skip] no preds {d}"); continue
        preds = preds.copy(); preds["code"] = preds["symbol"].map(_code6)
        preds["factor_rank_pct"] = preds["prediction"].rank(pct=True)
        cpath = CHAIN_DIR / f"chain_candidates_{d}{args.suffix}.parquet"
        chain = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame()
        fwd, to = _fwd_returns(panel, tds, ts, args.fwd_td)
        if fwd is None:
            print(f"[skip] no forward {d}"); continue
        per_date[d] = {"preds": preds, "chain": chain, "fwd": fwd, "bench": float(fwd.mean()), "to": to}
        print(f"loaded {d}->{to.date()} | chain_candidates={len(chain)} | bench {per_date[d]['bench']*100:+.2f}%")

    if not per_date:
        print("no usable dates"); return 1

    def build_union(pd_obj, nf, nc, weighting):
        preds, chain = pd_obj["preds"], pd_obj["chain"]
        fac = preds.nlargest(nf, "factor_rank_pct")[["symbol", "code", "factor_rank_pct"]].copy()
        fac["chain_conviction"] = 0.0
        if nc > 0 and not chain.empty and "chain_conviction" in chain:
            ch = chain.sort_values("chain_conviction", ascending=False).head(nc)[
                ["symbol", "code", "chain_conviction"]].copy()
            ch["factor_rank_pct"] = ch["code"].map(preds.set_index("code")["factor_rank_pct"]).fillna(0.0)
            u = pd.concat([fac, ch], ignore_index=True).drop_duplicates("symbol")
        else:
            u = fac
        if weighting == "equal":
            w = np.ones(len(u))
        elif weighting == "factor_tilt":
            w = pd.to_numeric(u["factor_rank_pct"], errors="coerce").fillna(0).clip(lower=1e-6).values
        else:  # mix 0.6 factor / 0.4 chain
            w = (0.6 * pd.to_numeric(u["factor_rank_pct"], errors="coerce").fillna(0)
                 + 0.4 * pd.to_numeric(u["chain_conviction"], errors="coerce").fillna(0)).clip(lower=1e-6).values
        return u["symbol"].tolist(), w

    # factor-only baseline per date (n_factor=20 equal) for win-rate reference
    base = {}
    for d, o in per_date.items():
        syms, w = build_union(o, 20, 0, "equal")
        base[d] = _port_excess(syms, w, o["fwd"], o["bench"])

    results = []
    for nf, nc, wt in itertools.product(args.n_factor, args.n_chain, args.weighting):
        if nc == 0 and wt != "equal":
            continue  # 无产业链时加权方式等价, 只留 equal 去重
        ex = {}
        for d, o in per_date.items():
            syms, w = build_union(o, nf, nc, wt)
            ex[d] = _port_excess(syms, w, o["fwd"], o["bench"])
        vals = [v for v in ex.values() if v is not None]
        if not vals:
            continue
        wins = sum(1 for d in ex if ex[d] is not None and base[d] is not None and ex[d] > base[d])
        results.append({"n_factor": nf, "n_chain": nc, "weighting": wt,
                        "mean_excess": round(float(np.mean(vals)), 3),
                        "worst_excess": round(float(np.min(vals)), 3),
                        "median_excess": round(float(np.median(vals)), 3),
                        "win_vs_factor": f"{wins}/{len(vals)}",
                        "per_date": ex})

    # 稳健性排名: 先看 worst-case(min) 再看 mean (用户要"跨regime/跨日期最佳")
    results.sort(key=lambda r: (r["worst_excess"], r["mean_excess"]), reverse=True)
    out = Path("runtime/reports/v8/mix_weight_experiment.json")
    out.write_text(json.dumps({"fwd_td": args.fwd_td, "dates": list(per_date),
                               "factor_only_base": base, "configs": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\n=== 因子-only(20,equal) 各日期超额%: " +
          " ".join(f"{d}:{base[d]}" for d in per_date) + " ===")
    print(f"\n{'rank':<5}{'nf':<4}{'nc':<4}{'weight':<12}{'mean':<8}{'worst':<8}{'median':<8}{'win_vs_factor'}")
    for i, r in enumerate(results[:18], 1):
        print(f"{i:<5}{r['n_factor']:<4}{r['n_chain']:<4}{r['weighting']:<12}"
              f"{r['mean_excess']:<8}{r['worst_excess']:<8}{r['median_excess']:<8}{r['win_vs_factor']}")
    best = results[0]
    print(f"\n最稳健配置: n_factor={best['n_factor']} n_chain={best['n_chain']} weighting={best['weighting']} "
          f"| mean超额 {best['mean_excess']}% worst {best['worst_excess']}% win {best['win_vs_factor']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
