"""Unit tests for the RD-Agent-style factor-loop knowledge helpers.

These cover the lightweight knowledge-management port: bucketing factors into
economic structures × horizons, the coverage map fed to the LLM proposer, and
the orthogonal-whitespace ("uncovered directions") signal that steers each
round away from crowded dead ends.
"""

from __future__ import annotations

from quantagent.factors.factor_loop_memory import (
    classify_horizon,
    classify_structure,
    coverage_map,
    uncovered_directions,
)


def test_classify_structure_buckets_known_mechanisms():
    assert classify_structure("short-term reversal of returns") == "short-term reversal"
    assert classify_structure("price-volume divergence signal") == "volume-price divergence"
    assert classify_structure("decayed momentum trend following") == "decayed momentum"
    assert classify_structure("turnover crowding proxy") == "turnover crowding"
    assert classify_structure("something totally unrelated") == "other"


def test_classify_horizon_normalises_text_and_numbers():
    assert classify_horizon("short_5d") == "short_5d"
    assert classify_horizon("long horizon ~120 days") == "long_30d_120d"
    assert classify_horizon("mid 5d 30d") == "mid_5d_30d"
    assert classify_horizon(3) == "short_5d"
    assert classify_horizon(20) == "mid_5d_30d"
    assert classify_horizon(60) == "long_30d_120d"
    assert classify_horizon(None) == "unspecified"


def test_coverage_map_aggregates_and_flags_crowded_cells():
    entries = [
        {"structure": "short-term reversal", "horizon": "short_5d",
         "status": "selected", "validation_rank_ic": 0.08},
        {"structure": "short-term reversal", "horizon": "short_5d", "status": "rejected"},
        {"structure": "low volatility", "horizon": "mid_5d_30d", "status": "rejected"},
        {"structure": "low volatility", "horizon": "mid_5d_30d", "status": "rejected"},
        {"structure": "low volatility", "horizon": "mid_5d_30d", "status": "rejected"},
    ]
    cov = coverage_map(entries)
    cells = {(c["structure"], c["horizon"]): c for c in cov["cells"]}

    rev = cells[("short-term reversal", "short_5d")]
    assert rev["attempted"] == 2
    assert rev["accepted"] == 1
    assert abs(rev["best_validation_rank_ic"] - 0.08) < 1e-12

    # 3 attempts, 0 acceptances -> a crowded-but-failing cell the proposer must avoid.
    crowded = {c["structure"] for c in cov["crowded_but_failing"]}
    assert "low volatility" in crowded
    assert "short-term reversal" not in crowded


def test_uncovered_directions_lists_structures_without_acceptance():
    entries = [
        {"structure": "short-term reversal", "status": "selected", "validation_rank_ic": 0.05},
        {"structure": "low volatility", "status": "rejected"},
    ]
    uncovered = uncovered_directions(entries)
    assert "short-term reversal" not in uncovered  # already represented
    assert "low volatility" in uncovered  # attempted but never survived
    assert "turnover crowding" in uncovered  # never tried at all
