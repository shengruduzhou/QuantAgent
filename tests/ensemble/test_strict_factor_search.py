from __future__ import annotations

from quantagent.ensemble.strict_factor_search import StrictFactorSearchConfig


def test_strict_factor_search_config_enables_interaction_beam_by_default():
    cfg = StrictFactorSearchConfig()

    assert cfg.interaction_search is True
    assert cfg.beam_width >= 1
    assert cfg.max_interaction_size == 0
    assert cfg.excess_weight >= cfg.turnover_penalty
