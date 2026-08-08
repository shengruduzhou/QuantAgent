from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.dynamic_top_k import DynamicTopKConfig, resolve_dynamic_top_k


def test_diagnostics_call_score_dispersion_what_it_is() -> None:
    decision = resolve_dynamic_top_k(
        eligible_count=100,
        predictions_for_date=pd.Series(np.linspace(-1.0, 1.0, 100)),
    )
    assert "score_dispersion" in decision.diagnostics
    assert "alpha_ic_proxy" not in decision.diagnostics
    assert "score_dispersion" in decision.contributions
    assert "alpha_ic_proxy" not in decision.contributions
    assert decision.diagnostics["score_dispersion_semantics"] == (
        "prediction_cross_section_only_not_information_coefficient"
    )


def test_score_dispersion_is_invariant_to_positive_scale() -> None:
    scores = pd.Series([-0.5, -0.2, 0.0, 0.3, 0.7])
    a = resolve_dynamic_top_k(eligible_count=100, predictions_for_date=scores)
    b = resolve_dynamic_top_k(eligible_count=100, predictions_for_date=scores * 100.0)
    assert a.diagnostics["score_dispersion"] == b.diagnostics["score_dispersion"]
    assert a.top_k == b.top_k


def test_sign_inversion_proves_dispersion_is_not_predictive_ic() -> None:
    scores = pd.Series([-0.5, -0.2, 0.0, 0.3, 0.7])
    normal = resolve_dynamic_top_k(eligible_count=100, predictions_for_date=scores)
    inverted = resolve_dynamic_top_k(eligible_count=100, predictions_for_date=-scores)
    # A predictive IC would reverse sign when every signal score is inverted.
    # Dispersion cannot: this is why the resolver must never label it IC.
    assert normal.diagnostics["score_dispersion"] == inverted.diagnostics["score_dispersion"]
    assert normal.top_k == inverted.top_k


def test_new_threshold_names_control_only_dispersion_rule() -> None:
    cfg = DynamicTopKConfig(
        base_top_k=30,
        top_k_min=5,
        top_k_max=80,
        score_dispersion_strong_threshold=0.0,
        score_dispersion_strong_bonus=7,
        policy_strength_bonus=0.0,
    )
    decision = resolve_dynamic_top_k(
        eligible_count=100,
        predictions_for_date=pd.Series([0.0, 1.0, 2.0, 3.0]),
        config=cfg,
    )
    assert decision.contributions["score_dispersion"] == 7
    assert decision.top_k == 37
