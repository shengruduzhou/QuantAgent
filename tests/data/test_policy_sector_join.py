"""Regression tests: policy 申万 sectors_hint must reach the sector pool.

These pin three bugs that silently zeroed the policy → stock-selection signal:

1. ``_ensure_list`` collapsed a numpy array (parquet round-trip of a list
   column) into a single stringified-array entity.
2. Bare 申万 Chinese sector names were classified as ``theme`` instead of
   ``sector`` (the policy adapter keeps entities bare, no ``sector:`` prefix).
3. Together these meant ``sector_level_1`` was never set, so the sector pool
   never joined onto candidate stocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.data.evidence.canonical import _ensure_list, policy_events_to_evidence
from quantagent.data.thesis.builder import (
    CapitalFlowThesisConfig,
    _classify_entity,
    build_capital_flow_theses,
)


def test_ensure_list_explodes_numpy_array():
    arr = np.array(["银行", "非银金融", "计算机"])
    assert _ensure_list(arr) == ["银行", "非银金融", "计算机"]


def test_classify_bare_shenwan_name_is_sector():
    assert _classify_entity("银行") == ("sector", "银行")
    assert _classify_entity("非银金融") == ("sector", "非银金融")
    # non-sector tokens stay themes
    assert _classify_entity("tech_innovation") == ("theme", "tech_innovation")
    assert _classify_entity("600519.SH") == ("symbol", "600519.SH")


def test_policy_shenwan_hint_produces_sector_thesis():
    # sectors_hint stored as a numpy array, exactly like a parquet round-trip.
    policy = pd.DataFrame(
        [
            {
                "event_id": "p1",
                "source": "gov.cn",
                "url": "https://www.gov.cn/p1",
                "announced_at": pd.Timestamp("2026-06-01"),
                "available_at": pd.Timestamp("2026-06-01 10:00"),
                "fetched_at": pd.Timestamp("2026-06-01 10:01"),
                "title": "国务院关于支持银行业的通知",
                "body_summary": "",
                "themes": np.array(["fiscal"]),
                "sectors_hint": np.array(["银行", "非银金融"]),
                "policy_strength": 0.7,
            }
        ]
    )
    canonical = policy_events_to_evidence(policy)
    # entities stay bare (other tests rely on this)
    assert "银行" in canonical.iloc[0]["entities"]

    theses = build_capital_flow_theses(
        canonical, config=CapitalFlowThesisConfig(min_supporting=1, min_aggregate_confidence=0.15)
    )
    sectors = {(t.direction_kind, t.direction_value) for t in theses}
    assert ("sector", "银行") in sectors
    assert ("sector", "非银金融") in sectors
