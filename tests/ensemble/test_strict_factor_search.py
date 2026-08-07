from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.ensemble.strict_factor_search import (
    StrictFactorSearchConfig,
    build_factor_composite,
)
from quantagent.models.interactions import INTERACTION_SEPARATOR


def test_strict_factor_search_config_enables_subset_beam_by_default():
    cfg = StrictFactorSearchConfig()

    assert cfg.subset_beam_search is True
    assert cfg.beam_width >= 1
    assert cfg.max_subset_size == 0
    assert cfg.excess_weight >= cfg.turnover_penalty


def test_deprecated_interaction_aliases_still_read():
    """Old artifact readers and scripts must keep working across the rename."""
    cfg = StrictFactorSearchConfig(subset_beam_search=False, max_subset_size=7)

    assert cfg.interaction_search is False
    assert cfg.max_interaction_size == 7


def test_composite_is_additive_not_interactive():
    """The composite must be exactly the mean of per-factor ranks.

    This is the property the old ``interaction_search`` name denied. If a real
    ``x_i x_j`` term were ever added to :func:`build_factor_composite`, the
    score would stop being reproducible from the per-factor ranks alone and
    this test would fail — which is the point.
    """
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    symbols = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, len(symbols)),
            "symbol": symbols * len(dates),
            "alpha_a": rng.normal(size=len(symbols) * len(dates)),
            "alpha_b": rng.normal(size=len(symbols) * len(dates)),
        }
    )

    both = build_factor_composite(frame, ("alpha_a", "alpha_b"), factor_signs={"alpha_a": 1.0, "alpha_b": 1.0})
    only_a = build_factor_composite(frame, ("alpha_a",), factor_signs={"alpha_a": 1.0})
    only_b = build_factor_composite(frame, ("alpha_b",), factor_signs={"alpha_b": 1.0})

    reconstructed = (only_a["composite_score"] + only_b["composite_score"]) / 2.0
    np.testing.assert_allclose(
        both["composite_score"].to_numpy(), reconstructed.to_numpy(), atol=1e-12
    )
    # And it emits no interaction column.
    assert not any(INTERACTION_SEPARATOR in column for column in both.columns)


def test_search_result_declares_its_model_class():
    from quantagent.ensemble.strict_factor_search import StrictFactorSearchResult

    result = StrictFactorSearchResult(
        best_factors=("alpha_a",),
        best_top_k=10,
        best_score=1.0,
        best_metrics={},
        trials=[],
        candidate_factors=("alpha_a",),
        factor_signs={"alpha_a": 1.0},
        regime_filter="all",
        config=StrictFactorSearchConfig(),
    )
    payload = result.as_dict()

    assert payload["model_class"] == "rank_weighted_additive"
    # Old keys stay present so pre-rename artifacts remain comparable.
    assert payload["config"]["interaction_search"] == payload["config"]["subset_beam_search"]
    assert payload["config"]["max_interaction_size"] == payload["config"]["max_subset_size"]
