#!/usr/bin/env python3
"""LLM-driven policy interpretation -> sector-priority canonical evidence.

The deterministic keyword tagger is sampling-biased (it counted finance-heavy
noise and under-weighted tech). The correct mechanism is to let the LLM READ
the policy regime (here: the 十五五 / 15th Five-Year Plan 2026-2030 priorities)
and emit per-申万一级 priority weights. We convert those into canonical policy
evidence (entities=[sector:X], policy_direction_score=priority) so the
capital-flow sector pool reflects the real policy direction (tech-led), not
keyword frequency.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.data.thesis.builder import SHENWAN_L1_SECTORS

# LLM may use colloquial names; map to 申万一级.
ALIAS = {
    "半导体": "电子", "芯片": "电子", "集成电路": "电子", "消费电子": "电子",
    "人工智能": "计算机", "数字经济": "计算机", "软件": "计算机", "信创": "计算机",
    "新能源": "电力设备", "光伏": "电力设备", "储能": "电力设备", "锂电": "电力设备",
    "航空航天军工": "国防军工", "航空航天": "国防军工", "军工": "国防军工",
    "新能源汽车": "汽车", "智能汽车": "汽车",
    "生物医药": "医药生物", "创新药": "医药生物", "医药": "医药生物",
    "6G": "通信", "5G": "通信", "算力": "通信",
    "新材料": "基础化工", "高端装备": "机械设备", "机器人": "机械设备",
    "电力": "公用事业", "核电": "公用事业",
}


def _to_shenwan(name: str) -> str | None:
    n = str(name or "").strip()
    if n in SHENWAN_L1_SECTORS:
        return n
    return ALIAS.get(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--plan", default='中国"十五五"规划(2026-2030)')
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("runtime/data/v7/raw/policy/llm_policy_priorities.parquet"))
    args = ap.parse_args()

    client = LLMSkillClient(LLMSkillConfig.from_env())
    res = client.invoke(
        "policy_priority_ranking",
        system_prompt="你是中国宏观政策与A股策略分析师。只输出一个JSON对象，不要多余文字。",
        user_text=(f'依据{args.plan}的政策重心，对申万一级行业按"国家资源与政策倾斜力度"从高到低排序，'
                   f'输出 JSON: {{"top_sectors":[{{"sector":申万一级名,"priority":0到1的数,"reason":一句话}}],'
                   f'"dominant_theme":"..."}}。只基于规划真实重心判断，不要平均分配，给出前{args.top_n}个。'),
        fallback={"top_sectors": []},
    )
    if res.used_fallback:
        print(f"LLM unavailable ({res.fallback_reason}); no priorities written.")
        return 1
    out = res.output
    theme = str(out.get("dominant_theme", ""))
    rows = []
    for item in out.get("top_sectors", [])[: args.top_n]:
        sec = _to_shenwan(item.get("sector"))
        if not sec:
            continue
        prio = float(item.get("priority", 0.5) or 0.5)
        rows.append({
            "evidence_id": "llmpol_" + hashlib.sha256(f"{sec}{args.as_of}".encode()).hexdigest()[:14],
            "source_name": "llm_policy_analyst", "source_type": "policy",
            "url_or_file_id": None, "publish_time": pd.Timestamp(args.as_of),
            "crawl_time": pd.Timestamp(args.as_of), "available_at": pd.Timestamp(args.as_of),
            "entity_type": "policy_priority", "entities": [f"sector:{sec}"], "raw_text_hash": None,
            "extracted_claims": [f"{args.plan} 优先级: {item.get('reason','')}"],
            "sentiment_score": 0.0, "policy_direction_score": max(0.0, min(1.0, prio)),
            "capital_flow_direction_score": 0.0, "confidence": max(0.3, min(1.0, prio)),
            "contradiction_score": 0.0, "lag_window_candidates": [20, 60, 120, 250],
            "audit_trace": {"adapter": "llm_policy_priorities", "dominant_theme": theme},
        })
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"dominant_theme: {theme}")
    print(f"wrote {len(df)} LLM policy-priority sector records -> {args.output}")
    print("sectors:", [(r['entities'][0], round(r['policy_direction_score'], 2)) for _, r in df.iterrows()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
