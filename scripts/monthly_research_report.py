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

from quantagent.research.forward_report import (
    build_forward_research_contract,
    render_forward_research_header,
    validate_forward_research_payload,
)


def _load(p, cols=None):
    p = Path(p)
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" else pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--cadence", choices=["weekly", "monthly"], default="monthly")
    ap.add_argument("--priorities", default="runtime/data/v7/raw/policy/llm_policy_priorities.parquet")
    ap.add_argument("--hybrid-pool", default="runtime/reports/v8/llm_hybrid_combined/hybrid_stock_pool.parquet")
    ap.add_argument("--sector-pool", default="runtime/reports/v8/llm_hybrid_combined/sector_pool.parquet")
    ap.add_argument("--bond", default="runtime/data/v7/silver/bond_flows/bond_flows.parquet")
    ap.add_argument("--top-picks", type=int, default=40)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/monthly"))
    args = ap.parse_args()

    as_of_ts = pd.Timestamp(args.as_of)
    contract = build_forward_research_contract(args.as_of, cadence=args.cadence)
    ym = as_of_ts.strftime("%Y%m")
    report_stem = f"research_report_{ym}" if args.cadence == "monthly" else f"research_report_weekly_{as_of_ts.strftime('%G-W%V')}"
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
        sort_col = "hybrid_rank" if "hybrid_rank" in hyb.columns else cols[0]
        top_picks = hyb.sort_values(sort_col).head(args.top_picks)[cols].to_dict("records")

    target_label = contract.window.label
    narrative = {}
    if not args.no_llm:
        from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
        c = LLMSkillClient(LLMSkillConfig.from_env())
        payload = {"as_of": args.as_of, "policy_priority_sectors": top_sectors[:10],
                   "bond_macro": macro, "candidate_top_picks": top_picks[:25],
                   "prediction_contract": contract.as_dict()}
        res = c.invoke("monthly_research",
            system_prompt=("你是顶级私募基金研究总监，擅长'事件催化+景气验证'的多主题投资策略。"
                           "写一份详细、可落地的A股前瞻策略研报。必须预测未来窗口，不要复盘总结。"
                           "只输出一个JSON对象，不要多余文字。"),
            user_text=(f"针对 {target_label} 写一份'事件催化 + 景气验证'的多主题A股前瞻研报。"
                       f"as_of={args.as_of}; PIT cutoff={contract.pit_cutoff_at}; 绝不能使用 cutoff 之后的信息或后见之明。"
                       "结合下方证据，以及你对该时段已知的周期性事件(PMI/社融/CPI/LPR/政治局会议/财报季/重要产业会议等)的认知。"
                       "输出 JSON: {"
                       "\"market_outlook\":\"大盘方向与资金面研判, 4-6句\","
                       "\"event_calendar\":[{\"date\":\"约几号(如月初/15日前后)\",\"event\":\"将发生的事/数据/会议\",\"benefit\":\"利好的方向或板块\"}],"
                       "\"themes\":[{\"theme\":\"投资主题\",\"catalyst\":\"催化事件\",\"prosperity\":\"景气验证依据(订单/价格/产量/政策落地)\",\"sectors\":[受益申万板块],\"logic\":\"现象→板块传导逻辑\"}],"
                       "\"market_style\":\"风格研判: 高低切与板块轮动方向, 谁高谁低, 资金从哪流向哪\","
                       "\"key_risks\":[\"风险点\"],\"next_month_catalysts\":[\"关键催化\"]}。"
                       "event_calendar 和 themes 必须覆盖多个事件与多个股池方向，不能只写一个事件。"
                       "不要编造具体未公布的数字。证据: "
                       + json.dumps(payload, ensure_ascii=False)),
            fallback={})
        if not res.used_fallback:
            narrative = res.output

    n = narrative if isinstance(narrative, dict) else {}
    validation = validate_forward_research_payload(n, contract, stock_count=len(top_picks))
    md = [f"# {target_label} A股多主题投资策略研报", "",
          "*基于事件催化与景气验证的配置框架 · 选股池参考（非交易指令）*",
          f"*as_of {args.as_of}*", "",
          render_forward_research_header(contract), ""]
    if validation.warnings:
        md += ["## 覆盖度提示", ""] + [f"- {w}" for w in validation.warnings] + [""]
    md += ["## 一、大盘研判与资金面", ""]
    if n.get("market_outlook"):
        md += [n["market_outlook"], ""]
    md += [f"- **债市/资金面**：10Y国债 {macro.get('yield_10y','-')}% · 期限利差(10Y-1Y) {macro.get('spread_10y_1y','-')} · 信用利差(AAA-国债) {macro.get('credit_spread_aa','-')}",
           f"- **风格研判（高低切/板块轮动）**：{n.get('market_style','-')}", ""]
    # event calendar
    md += [f"## 二、{target_label} 事件日历与催化", "", "| 时点 | 事件/数据/会议 | 利好方向 |", "|---|---|---|"]
    for e in n.get("event_calendar", [])[:10]:
        md.append(f"| {e.get('date','-')} | {e.get('event','-')} | {e.get('benefit','-')} |")
    # multi-theme analysis
    md += ["", "## 三、多主题分析（催化 → 景气验证 → 受益板块）", ""]
    for i, t in enumerate(n.get("themes", [])[:6], 1):
        md += [f"### 主题{i}：{t.get('theme','-')}",
               f"- **催化事件**：{t.get('catalyst','-')}",
               f"- **景气验证**：{t.get('prosperity','-')}",
               f"- **受益板块**：{'、'.join(t.get('sectors',[]) or [])}",
               f"- **传导逻辑**：{t.get('logic','-')}", ""]
    # policy direction
    md += ["## 四、政策方向 / 板块优先级（LLM 十五五研判）", "", "| 申万板块 | 政策优先级 |", "|---|---|"]
    for s in top_sectors[:10]:
        md.append(f"| {s['sector']} | {s['priority']:.2f} |")
    # stock pool (factor-dominant mix)
    md += ["", "## 五、前瞻个股池（因子主干 + LLM/产业逻辑扩展 + 风控门）", "",
           "| symbol | 板块 | hybrid_score | action |", "|---|---|---|---|"]
    for p in top_picks:
        md.append(f"| {p.get('symbol')} | {p.get('sector_level_1','')} | {p.get('hybrid_score',0):.3f} | {p.get('action_bucket','-')} |")
    md += ["", "> 选股口径：因子模型(已验证 +α)主导排序；政策/证据仅做小幅板块倾斜；老庄/流动性为硬性风控门；个股短线用做T管理。"]
    if n.get("key_risks"):
        md += ["", "## 六、风险提示", ""] + [f"- {r}" for r in n["key_risks"][:6]]
    if n.get("next_month_catalysts"):
        md += ["", "## 七、关键催化", ""] + [f"- {r}" for r in n["next_month_catalysts"][:8]]
    md += [
        "",
        "## 八、OOS 验证计划",
        "",
        "- 本报告股池只作为 forward candidate pool；下个窗口结束后，用同一 as_of contract 对应的真实 forward return 复盘。",
        "- 因子-only、LLM产业链-only、UNION、加权混合四组都必须跑相同 T+1/涨跌停/停牌/成本/滑点约束。",
        "- 对历史窗口必须跑 real/nonews/scrambled 新闻消融，若 nonews 接近 real，则标记 hindsight 风险。",
        "",
        "---",
        "> 用途：前瞻研报作为**选股池参考**，与因子排名混合后再做 OOS、风控回测和 paper trading；不构成交易指令。",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{report_stem}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    if top_picks:
        pd.DataFrame(top_picks).to_parquet(args.out_dir / f"{report_stem}_pool.parquet", index=False)
    contract.write(args.out_dir / f"{report_stem}_contract.json")
    (args.out_dir / f"{report_stem}_validation.json").write_text(
        json.dumps(validation.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out} ({args.cadence}, {len(top_picks)} picks, {len(top_sectors)} priority sectors, llm={'yes' if narrative else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
