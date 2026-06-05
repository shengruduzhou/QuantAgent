#!/usr/bin/env python3
"""Build ONE canonical evidence set combining 红头文件 + 债市 + 投行 + 国家队 + 舆情.

Cleans MOF/外事/党务 noise from the policy raw, rebuilds policy_events, converts
all silver feeds to canonical, then concatenates the news-sentiment canonical.
The result is passed to build-llm-hybrid-stock-pool-v8 via --canonical-evidence-path
so policy(direction) and news(sentiment/timing) are fused before the LLM decides.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantagent.data.evidence.canonical import CANONICAL_EVIDENCE_COLUMNS, to_canonical_evidence_frame
from quantagent.data.policy import PolicyEventConfig, PolicyEventBuilder
from quantagent.data.thesis.builder import SHENWAN_L1_SECTORS

# external-affairs / party / procedural noise that is NOT sectoral policy
NOISE = ["会见", "出席", "党委", "理事会", "领导小组", "磋商", "答记者问",
         "座谈", "工作会议", "高级搜索", "信息公开指南"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-raw", type=Path, default=Path("runtime/data/v7/raw/policy/policy_raw.csv"))
    ap.add_argument("--bond", default="runtime/data/v7/silver/bond_flows/bond_flows.parquet")
    ap.add_argument("--broker", default="runtime/data/v7/silver/broker_reports/broker_reports.parquet")
    ap.add_argument("--state-team", default="runtime/data/v7/silver/state_team_inference/state_team_inference.parquet")
    ap.add_argument("--news", default="runtime/data/v7/raw/news/news_canonical.parquet")
    ap.add_argument("--llm-priorities", default="runtime/data/v7/raw/policy/llm_policy_priorities.parquet")
    ap.add_argument("--output", type=Path, default=Path("runtime/data/v7/silver/combined_canonical.parquet"))
    args = ap.parse_args()

    raw = pd.read_csv(args.policy_raw)
    before = len(raw)
    mask = ~raw["title"].astype(str).str.contains("|".join(NOISE), na=False)
    raw = raw[mask].reset_index(drop=True)
    print(f"policy noise filter: {before} -> {len(raw)} (dropped {before - len(raw)} 外事/党务/导航)")

    pe = PolicyEventBuilder(PolicyEventConfig(min_events=1, min_theme_coverage=0.0, min_strength_median=0.0))
    policy_events = pe.build(raw).frame

    def _rd(p):
        return pd.read_parquet(p) if p and Path(p).exists() else None

    canonical = to_canonical_evidence_frame(
        policy_events=policy_events,
        bond_flows=_rd(args.bond),
        broker_reports=_rd(args.broker),
        state_team_events=_rd(args.state_team),
    )
    # Only POLICY drives sector DIRECTION (通过政策文件判断方向). Broker / bond /
    # state-team contribute per-stock/macro signals, so strip their sector:
    # entities — otherwise numerous broker sector ratings dilute and crowd out
    # the authoritative 十五五 policy direction at the thesis-aggregation level.
    def _strip_sector(ents):
        # drop both "sector:X" and bare 申万一级 names -> keep themes/symbols only
        ents = ents.tolist() if hasattr(ents, "tolist") else list(ents or [])
        return [e for e in ents if not str(e).startswith("sector:") and str(e) not in SHENWAN_L1_SECTORS]
    # Only the LLM policy analyst (source_name=llm_policy_analyst) is authoritative
    # for sector DIRECTION. Strip keyword-tagged sectors from everything else
    # (broker/bond/state-team AND the raw-policy docs whose keyword tagger
    # mis-tags 农业/城市更新 into 交通运输/银行/...), so the 十五五 科技 priority is
    # not drowned out by tagger noise + high-n broker theses.
    keep_sector = canonical["source_name"] == "llm_policy_analyst"
    canonical.loc[~keep_sector, "entities"] = canonical.loc[~keep_sector, "entities"].map(_strip_sector)
    parts = [canonical]
    news = _rd(args.news)
    if news is not None and not news.empty:
        # News sentiment is a PER-STOCK timing signal, not a sector-direction
        # claim — strip sector: entities so it cannot dilute the LLM/policy
        # sector direction (policy decides 板块, sentiment decides 个股择时).
        def _symbol_only(ents):
            ents = ents.tolist() if hasattr(ents, "tolist") else list(ents or [])
            return [e for e in ents if not str(e).startswith("sector:")]
        news = news.copy()
        news["entities"] = news["entities"].map(_symbol_only)
        for c in CANONICAL_EVIDENCE_COLUMNS:
            if c not in news.columns:
                news[c] = None
        parts.append(news[list(CANONICAL_EVIDENCE_COLUMNS)])
    pri = _rd(args.llm_priorities)
    if pri is not None and not pri.empty:
        for c in CANONICAL_EVIDENCE_COLUMNS:
            if c not in pri.columns:
                pri[c] = None
        parts.append(pri[list(CANONICAL_EVIDENCE_COLUMNS)])
    combined = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)

    by_src = combined["source_type"].value_counts().to_dict()
    print(f"wrote {len(combined)} canonical records -> {args.output}")
    print(f"by source_type: {by_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
