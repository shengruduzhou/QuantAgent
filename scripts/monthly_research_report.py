#!/usr/bin/env python3
"""月度研报生成 — private-fund-style monthly research report (选股池参考).

Synthesizes the month's evidence into a research report:
  * 宏观/政策方向: LLM 十五五 sector priorities + bond-market regime (yields/spreads)
  * 板块池: capital-flow sector pool (policy-driven)
  * 个股池: hybrid stock pool (factor + LLM + evidence) with scores
  * LLM 写研报: outlook / sector thesis / top-pick rationale / risks / next-month catalysts

Inputs are produced by the fetch/build/hybrid jobs (cron runs those first).
Output: runtime/reports/monthly/research_report_<YYYYMM>.md  (+ pool parquet)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def _load(p, cols=None):
    p = Path(p)
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" else pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--priorities", default="runtime/data/v7/raw/policy/llm_policy_priorities.parquet")
    ap.add_argument("--hybrid-pool", default="runtime/reports/v8/llm_hybrid_combined/hybrid_stock_pool.parquet")
    ap.add_argument("--sector-pool", default="runtime/reports/v8/llm_hybrid_combined/sector_pool.parquet")
    ap.add_argument("--bond", default="runtime/data/v7/silver/bond_flows/bond_flows.parquet")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/monthly"))
    args = ap.parse_args()

    ym = pd.Timestamp(args.as_of).strftime("%Y%m")
    pri = _load(args.priorities)
    hyb = _load(args.hybrid_pool)
    sp = _load(args.sector_pool)
    bond = _load(args.bond)

    # macro snapshot from latest bond row
    macro = {}
    if bond is not None and not bond.empty:
        b = bond.sort_values("trade_date").iloc[-1]
        macro = {k: (round(float(b[k]), 3) if k in b and pd.notna(b[k]) else None)
                 for k in ["yield_10y", "yield_1y", "spread_10y_1y", "credit_spread_aa"]}
    top_sectors = []
    if pri is not None and not pri.empty:
        for _, r in pri.iterrows():
            ent = r["entities"]; ent = ent.tolist() if hasattr(ent, "tolist") else list(ent)
            sec = next((e.split(":")[1] for e in ent if str(e).startswith("sector:")), None)
            top_sectors.append({"sector": sec, "priority": round(float(r["policy_direction_score"]), 2)})
    top_picks = []
    if hyb is not None and not hyb.empty:
        cols = [c for c in ["symbol", "sector_level_1", "hybrid_score", "llm_stock_score", "action_bucket"] if c in hyb.columns]
        top_picks = hyb.sort_values("hybrid_rank" if "hybrid_rank" in hyb.columns else cols[0]).head(15)[cols].to_dict("records")

    narrative = {}
    if not args.no_llm:
        from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
        c = LLMSkillClient(LLMSkillConfig.from_env())
        payload = {"as_of": args.as_of, "policy_priority_sectors": top_sectors[:10],
                   "bond_macro": macro, "candidate_top_picks": top_picks[:12]}
        res = c.invoke("monthly_research",
            system_prompt="你是私募基金研究总监。写一份月度A股策略研报。只输出JSON。",
            user_text=("依据以下证据写月度研报，输出 JSON: {\"market_outlook\":\"宏观与流动性判断3-4句\","
                       "\"sector_thesis\":[{\"sector\":申万,\"thesis\":\"一句逻辑\"}],"
                       "\"top_pick_rationale\":[{\"symbol\":代码,\"reason\":\"一句\"}],"
                       "\"key_risks\":[\"...\"],\"next_month_catalysts\":[\"...\"]}。证据: "
                       + json.dumps(payload, ensure_ascii=False)),
            fallback={})
        if not res.used_fallback:
            narrative = res.output

    md = [f"# 月度A股策略研报 — {ym}", "", f"*as_of {args.as_of} · 选股池参考（非交易指令）*", ""]
    if isinstance(narrative, dict) and narrative.get("market_outlook"):
        md += ["## 一、宏观与流动性判断", "", narrative["market_outlook"], ""]
    md += ["## 二、债市/资金面快照", "",
           f"10Y国债 {macro.get('yield_10y','-')}% · 期限利差(10Y-1Y) {macro.get('spread_10y_1y','-')} · 信用利差(AAA-国债) {macro.get('credit_spread_aa','-')}", ""]
    md += ["## 三、政策方向 / 板块优先级（LLM 十五五研判）", "", "| 申万板块 | 政策优先级 |", "|---|---|"]
    for s in top_sectors[:10]:
        md.append(f"| {s['sector']} | {s['priority']:.2f} |")
    if isinstance(narrative, dict) and narrative.get("sector_thesis"):
        md += ["", "**板块逻辑**:"] + [f"- **{t.get('sector')}**: {t.get('thesis')}" for t in narrative["sector_thesis"][:8]]
    md += ["", "## 四、月度个股池（因子+LLM+证据混合）", "",
           "| symbol | 板块 | hybrid_score | llm_score | action |", "|---|---|---|---|---|"]
    for p in top_picks:
        md.append(f"| {p.get('symbol')} | {p.get('sector_level_1','')} | {p.get('hybrid_score',0):.3f} | "
                  f"{p.get('llm_stock_score','-')} | {p.get('action_bucket','-')} |")
    if isinstance(narrative, dict):
        if narrative.get("top_pick_rationale"):
            md += ["", "**重点个股逻辑**:"] + [f"- {r.get('symbol')}: {r.get('reason')}" for r in narrative["top_pick_rationale"][:10]]
        if narrative.get("key_risks"):
            md += ["", "## 五、风险提示", ""] + [f"- {r}" for r in narrative["key_risks"][:6]]
        if narrative.get("next_month_catalysts"):
            md += ["", "## 六、下月关注催化", ""] + [f"- {r}" for r in narrative["next_month_catalysts"][:6]]
    md += ["", "> 用途：月度研报作为**选股池参考**，与因子排名混合后再做风控回测；不构成交易指令。"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"research_report_{ym}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    if top_picks:
        pd.DataFrame(top_picks).to_parquet(args.out_dir / f"pool_{ym}.parquet", index=False)
    print(f"wrote {out} ({len(top_picks)} picks, {len(top_sectors)} priority sectors, llm={'yes' if narrative else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
