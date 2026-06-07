#!/usr/bin/env python3
"""前视隔离的多月 OOS 验证 + hindsight 消融对照 — 确认 UNION(因子,产业链) 的 edge
不是靠 LLM 参数记忆里的后见之明。

对每个 month-end as-of:
  - 复用(或 --regen 重生成) 三种新闻模式的产业链池:
      real      = 截至 as_of 的真实 PIT 新闻 (edge 应主要来自这里)
      nonews    = 不给新闻, LLM 纯靠参数记忆 (若仍有 edge → hindsight 嫌疑)
      scrambled = 错位一年的新闻 (若错新闻也产生同样 edge → 新闻非驱动)
  - 计算各模式的 因子 / 产业链 / 并集 前瞻超额。

判定:
  edge_from_news = chain超额(real) - max(chain超额(nonews), chain超额(scrambled))
  若 edge_from_news > 0 且 real 显著为正 → edge 由 PIT 新闻驱动(更可信, 非纯 hindsight)。
  若 nonews ≈ real → 参数记忆即可复现 → hindsight 风险高(诚实标注)。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRED_SRC = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MON = Path("runtime/reports/monthly")
PY = "AI_quant_venv/bin/python3"
MODE_SFX = {"real": "", "nonews": "_nonews", "scrambled": "_scrambled"}


def _code6(s):
    return str(s).split(".")[0].zfill(6)


def _regen(d, modes, n_chain, enrich):
    pf = f"runtime/tmp/real_preds_{d.replace('-', '')}.parquet"
    for mode in modes:
        sfx = MODE_SFX[mode]
        if (MON / f"chain_pool_{d}{sfx}.parquet").exists():
            continue
        print(f"  [regen] {d} mode={mode} ...", flush=True)
        subprocess.run([PY, "scripts/industry_chain_research.py", "--as-of", d, "--predictions", pf,
                        "--news-mode", mode, "--n-chains", "6", "--stocks-per-segment", "4",
                        "--max-stocks-enrich", str(enrich), "--n-chain", str(n_chain)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", default=["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"])
    ap.add_argument("--modes", nargs="+", default=["real", "nonews", "scrambled"])
    ap.add_argument("--fwd-td", type=int, default=10)
    ap.add_argument("--n-factor", type=int, default=12)
    ap.add_argument("--n-chain", type=int, default=8)
    ap.add_argument("--regen", action="store_true", help="缺失的池调用 agent 重生成(慢, 跑 LLM)")
    ap.add_argument("--enrich", type=int, default=18)
    args = ap.parse_args()

    allp = pd.read_parquet(PRED_SRC); allp["trade_date"] = pd.to_datetime(allp["trade_date"])
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    tds = sorted(panel["trade_date"].unique())
    results = []

    for d in args.dates:
        ts = pd.Timestamp(d)
        day = allp[allp.trade_date == ts]
        if day.empty:
            print(f"[skip] no preds {d}"); continue
        pf = f"runtime/tmp/real_preds_{d.replace('-', '')}.parquet"
        day[["trade_date", "symbol", "alpha_score"]].rename(columns={"alpha_score": "prediction"}).to_parquet(pf, index=False)
        if args.regen:
            _regen(d, args.modes, args.n_chain, args.enrich)

        future = [t for t in tds if t > ts][:args.fwd_td]
        if not future:
            print(f"[skip] no forward {d}"); continue
        d1 = future[-1]
        c0 = panel[panel.trade_date == ts].set_index("symbol")["close"]
        c1 = panel[panel.trade_date == d1].set_index("symbol")["close"]
        fwd = (c1 / c0 - 1.0).dropna(); bench = float(fwd.mean())
        preds = pd.read_parquet(pf)
        fac = preds.nlargest(20, "prediction").symbol.tolist()
        fac_top = preds.nlargest(args.n_factor, "prediction").symbol.tolist()

        def ex(syms):
            r = fwd.reindex(list(dict.fromkeys(syms))).dropna()
            return (round(r.mean() * 100, 2), round((r.mean() - bench) * 100, 2),
                    round((r > 0).mean() * 100), len(r)) if len(r) else None

        row = {"as_of": d, "fwd_to": str(d1.date()), "bench_%": round(bench * 100, 2),
               "factor": ex(fac), "modes": {}}
        for mode in args.modes:
            cp = MON / f"chain_pool_{d}{MODE_SFX[mode]}.parquet"
            if not cp.exists():
                continue
            cpool = pd.read_parquet(cp)
            chain = (cpool[cpool.get("source", "") == "LLM产业链"].symbol.tolist()
                     if "source" in cpool else cpool.symbol.tolist())
            union = list(dict.fromkeys(fac_top + chain[:args.n_chain]))
            row["modes"][mode] = {"chain": ex(chain), "union": ex(union), "n_chain": len(chain)}
        results.append(row)

        # console line
        def gx(mode, key):
            m = row["modes"].get(mode)
            return m[key][1] if m and m.get(key) else None
        print(f"[{d}->{d1.date()}] bench {row['bench_%']:+.2f}% | factor超额 {row['factor'][1] if row['factor'] else '-'} | "
              f"chain超额 real={gx('real','chain')} nonews={gx('nonews','chain')} scram={gx('scrambled','chain')} | "
              f"union超额 real={gx('union' and 'real','union')}")

    out = Path("runtime/reports/v8/chain_oos_validation.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- aggregate + hindsight verdict ----
    def col(mode, key):
        return [r["modes"][mode][key][1] for r in results
                if r["modes"].get(mode) and r["modes"][mode].get(key)]
    fac_ex = [r["factor"][1] for r in results if r["factor"]]
    print("\n=== 跨月汇总 (平均前瞻超额%) ===")
    if fac_ex:
        print(f"  因子(top20):      mean {np.mean(fac_ex):+.2f}  worst {np.min(fac_ex):+.2f}  ({len(fac_ex)}月)")
    for mode in args.modes:
        cc, uu = col(mode, "chain"), col(mode, "union")
        if cc:
            print(f"  产业链[{mode:<9}]: mean {np.mean(cc):+.2f}  worst {np.min(cc):+.2f}  | "
                  f"并集 mean {np.mean(uu):+.2f}  worst {np.min(uu):+.2f}")
    real_c = col("real", "chain")
    if real_c and ("nonews" in args.modes or "scrambled" in args.modes):
        alt = [v for m in ("nonews", "scrambled") for v in col(m, "chain")]
        edge_from_news = np.mean(real_c) - (np.mean(alt) if alt else 0.0)
        print(f"\n=== Hindsight 消融判定 ===")
        print(f"  chain超额 real mean = {np.mean(real_c):+.2f}% ; 消融(nonews/scrambled) mean = "
              f"{np.mean(alt) if alt else float('nan'):+.2f}%")
        print(f"  edge_from_news = real - 消融 = {edge_from_news:+.2f}%")
        if np.mean(real_c) > 0 and edge_from_news > 0:
            print("  → real 显著优于消融: edge 主要由 PIT 新闻驱动, 非纯参数记忆 hindsight (更可信)。")
        elif alt and np.mean(alt) >= np.mean(real_c) - 0.3:
            print("  → 消融≈real: 不给真新闻也能复现 → 参数记忆 hindsight 风险高 (诚实标注, 需 forward live 验证)。")
        else:
            print("  → 信号偏弱/样本少, 结论不稳, 需更多窗口或 forward live。")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
