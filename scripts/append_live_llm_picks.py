#!/usr/bin/env python3
"""Append the live LLM hybrid picks to the regime-conditional research report.

Reads the artifacts written by ``build-llm-hybrid-stock-pool-v8`` and appends a
'实盘 LLM 选股' section (LLM raw pool + final hybrid pool with score ranking,
sector, old-dealer/do-T flags and position hints) to the research report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _col(df, name, default=""):
    return df[name] if name in df.columns else pd.Series([default] * len(df))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", type=Path, default=Path("runtime/reports/v8/llm_hybrid_live"))
    ap.add_argument("--report", type=Path, default=Path("runtime/reports/v8/regime_conditional/research_report.md"))
    ap.add_argument("--as-of", default="2026-06-04")
    args = ap.parse_args()

    analysis = json.loads((args.live_dir / "capital_flow_stock_pool_analysis.json").read_text(encoding="utf-8"))
    hyb = pd.read_parquet(args.live_dir / "hybrid_stock_pool.parquet")
    llm = pd.read_parquet(args.live_dir / "llm_stock_pool.parquet")
    sector_pool = pd.read_parquet(args.live_dir / "sector_pool.parquet")

    md = ["", "## 实盘 LLM 选股（as_of " + args.as_of + "，四路证据：政策/债市/投行/国家队）", "",
          f"- LLM provider/model: `{analysis.get('provider')}` / `{analysis.get('model')}`  | used_fallback=**{analysis.get('used_fallback')}**（False=真LLM出池）",
          f"- 板块池（capital-flow thesis）方向数: {len(sector_pool)}", ""]

    # LLM raw pool ranking
    md += ["### 1) LLM 原始选股池（按 LLM 评分）", ""]
    lc = [c for c in ["symbol", "llm_rank", "llm_stock_score", "llm_confidence", "llm_horizon_bucket"] if c in llm.columns]
    if "llm_stock_score" in llm.columns:
        llm = llm.sort_values("llm_stock_score", ascending=False)
    md += ["| " + " | ".join(lc) + " |", "|" + "---|" * len(lc) + ""]
    for _, r in llm.head(15).iterrows():
        md.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in lc) + " |")

    # Final hybrid pool (factor+LLM merged)
    md += ["", "### 2) 混合后最终股池（因子+LLM，含评分 ranking 与风控）", ""]
    hc = [c for c in ["hybrid_rank", "symbol", "sector_level_1", "hybrid_score", "llm_stock_score",
                      "old_dealer_risk_score", "do_t_suitability_score", "action_bucket",
                      "research_weight_hint", "research_amount_hint"] if c in hyb.columns]
    if "hybrid_rank" in hyb.columns:
        hyb = hyb.sort_values("hybrid_rank")
    md += ["| " + " | ".join(hc) + " |", "|" + "---|" * len(hc) + ""]
    for _, r in hyb.head(15).iterrows():
        cells = []
        for c in hc:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "> 历史回测里的 'LLM侧' 用确定性证据叠加做无前视替身；此处是**真实 LLM** 在最新证据上的前瞻选股，"
           "回测无法回放，故作为前瞻 overlay 与风控层使用。", ""]

    report = args.report.read_text(encoding="utf-8") if args.report.exists() else "# 研报\n"
    args.report.write_text(report + "\n".join(md), encoding="utf-8")
    print(f"appended live LLM section ({len(llm)} LLM rows, {len(hyb)} hybrid rows) -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
