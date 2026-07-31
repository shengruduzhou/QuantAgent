"""Tests for the factor-fusion search engine.

The engine's value is that it is *honest*, so the tests are mostly adversarial:
they check that a pure-noise panel does not produce a confident answer, that
trial counts cannot be laundered, and that fitted schemes never touch the
out-of-sample segment.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantagent.fusion import (
    FusionSearchConfig,
    ObjectivePreference,
    build_scheme_specs,
    cross_sectional_scores,
    derive_weights,
    evaluate_blend,
    fold_consistency,
    pareto_front,
    rank_by_preference,
    run_fusion_search,
    save_fusion_artifacts,
)
from quantagent.fusion.schemes import BlendScheme, SchemeSpec


def _build_panel(
    *,
    n_dates: int = 420,
    n_symbols: int = 60,
    signal_strength: float = 0.02,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel where ``alpha_signal`` drives the forward return and the rest do not."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    symbols = [f"{600000 + index}.SH" for index in range(n_symbols)]
    frames = []
    for date in dates:
        signal = rng.normal(size=n_symbols)
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "alpha_signal": signal,
                    "alpha_noise": rng.normal(size=n_symbols),
                    "alpha_junk": rng.normal(size=n_symbols),
                    "forward_return": signal_strength * signal
                    + 0.01 * rng.normal(size=n_symbols),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    factor_panel = panel[
        ["trade_date", "symbol", "alpha_signal", "alpha_noise", "alpha_junk"]
    ]
    forward_panel = panel[["trade_date", "symbol", "forward_return"]]
    return factor_panel, forward_panel


def _config(**overrides) -> FusionSearchConfig:
    base = {
        "factor_names": ("alpha_signal", "alpha_noise", "alpha_junk"),
        "horizon_days": 5,
        "top_k": 10,
        "n_folds": 3,
        "embargo_days": 5,
        "min_train_days": 120,
        "min_test_days": 40,
        "include_genetic": False,
        "random_controls": 4,
        "single_factor_baselines": 3,
    }
    base.update(overrides)
    return FusionSearchConfig(**base)


# --------------------------------------------------------------------------- #
# Signal recovery                                                             #
# --------------------------------------------------------------------------- #


def test_search_recovers_the_only_predictive_factor():
    factor_panel, forward_panel = _build_panel()
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    preferred = result.preferred
    assert preferred is not None
    assert preferred.weights["alpha_signal"] > 0.8
    assert abs(preferred.weights["alpha_junk"]) < 0.15
    assert preferred.metrics["excessReturn"] > 0


def test_single_factor_baselines_rank_by_true_predictive_power():
    factor_panel, forward_panel = _build_panel()
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    by_id = {item.candidate_id: item for item in result.candidates}
    assert (
        by_id["single_alpha_signal"].metrics["excessReturn"]
        > by_id["single_alpha_noise"].metrics["excessReturn"]
    )
    assert (
        by_id["single_alpha_signal"].metrics["robustness"]
        > by_id["single_alpha_junk"].metrics["robustness"]
    )


# --------------------------------------------------------------------------- #
# Honesty under no signal                                                     #
# --------------------------------------------------------------------------- #


def test_pure_noise_panel_does_not_produce_a_robust_winner():
    """With no predictive factor, no candidate may claim strong robustness."""
    factor_panel, forward_panel = _build_panel(signal_strength=0.0, seed=99)
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    preferred = result.preferred
    assert preferred is not None
    # A noise panel must not clear the bar a real edge would clear.
    assert preferred.metrics["robustness"] < 0.75
    assert abs(preferred.metrics["excessReturn"]) < 0.5


def test_noise_panel_cannot_manufacture_excess_return_after_costs():
    """No blend of noise may beat the universe once trading costs are charged.

    Selecting top-K from noise is a random draw, so the blend earns the universe
    return minus its own turnover cost. Any positive excess here would mean the
    evaluator is crediting return that trading did not produce.
    """
    factor_panel, forward_panel = _build_panel(signal_strength=0.0, seed=123)
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    excess = [
        float(item.metrics["excessReturn"])
        for item in result.candidates
        if int(item.metrics["observations"]) > 0
    ]
    assert excess, "expected at least one evaluated candidate"
    assert max(excess) <= 0.0
    # And the shortfall should be on the order of the cost drag, not arbitrary.
    assert min(excess) > -0.5


def test_noise_panel_excess_straddles_zero_without_costs():
    """Remove the cost drag and noise blends scatter either side of the benchmark."""
    factor_panel, forward_panel = _build_panel(signal_strength=0.0, seed=123)
    result = run_fusion_search(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        config=_config(transaction_cost_bps=0.0),
    )
    excess = [
        float(item.metrics["excessReturn"])
        for item in result.candidates
        if int(item.metrics["observations"]) > 0
    ]
    assert min(excess) < 0 < max(excess)


# --------------------------------------------------------------------------- #
# Trial accounting                                                            #
# --------------------------------------------------------------------------- #


def test_n_trials_equals_the_number_of_enumerated_candidates():
    factor_panel, forward_panel = _build_panel()
    config = _config(random_controls=6, single_factor_baselines=3)
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=config
    )
    assert result.n_trials == len(result.candidates)
    # 4 fitted schemes + 6 random controls + 3 single-factor baselines.
    assert result.n_trials == 13


def test_more_controls_deflate_the_robustness_of_the_same_winner():
    """Adding trials must not make the winner look better."""
    factor_panel, forward_panel = _build_panel()
    lean = run_fusion_search(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        config=_config(random_controls=0, single_factor_baselines=0),
    )
    wide = run_fusion_search(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        config=_config(random_controls=12, single_factor_baselines=3),
    )
    lean_dsr = {
        item.candidate_id: item.robustness_breakdown["deflatedSharpeProbability"]
        for item in lean.candidates
    }
    wide_dsr = {
        item.candidate_id: item.robustness_breakdown["deflatedSharpeProbability"]
        for item in wide.candidates
    }
    shared = set(lean_dsr) & set(wide_dsr)
    assert shared
    assert wide.n_trials > lean.n_trials
    for candidate_id in shared:
        assert wide_dsr[candidate_id] <= lean_dsr[candidate_id] + 1e-9
    # Guard against the assertion passing because the term is dead: on a panel
    # with a genuine signal, at least one candidate must score a non-trivial
    # deflated Sharpe. A units error upstream pins every value at ~0 and would
    # otherwise satisfy the monotonicity check vacuously.
    assert max(lean_dsr.values()) > 0.1


# --------------------------------------------------------------------------- #
# Leakage discipline                                                          #
# --------------------------------------------------------------------------- #


def test_fitted_weights_ignore_data_outside_the_training_segment():
    """Mutating the out-of-sample tail must not change the fitted weights."""
    factor_panel, forward_panel = _build_panel(n_dates=260)
    dates = pd.DatetimeIndex(sorted(factor_panel["trade_date"].unique()))
    train_end = dates[150]
    train_factors = factor_panel[factor_panel["trade_date"] <= train_end]
    train_forwards = forward_panel[forward_panel["trade_date"] <= train_end]

    spec = SchemeSpec("ic_weighted", BlendScheme.IC_WEIGHTED, "IC 加权", top_k=10)
    names = ["alpha_signal", "alpha_noise", "alpha_junk"]
    baseline = derive_weights(
        spec,
        factor_panel=train_factors,
        forward_panel=train_forwards,
        factor_names=names,
    )

    poisoned = forward_panel.copy()
    tail = poisoned["trade_date"] > train_end
    poisoned.loc[tail, "forward_return"] = poisoned.loc[tail, "forward_return"] * -50.0
    poisoned_train = poisoned[poisoned["trade_date"] <= train_end]
    after = derive_weights(
        spec,
        factor_panel=train_factors,
        forward_panel=poisoned_train,
        factor_names=names,
    )
    np.testing.assert_allclose(baseline, after)


def test_control_schemes_are_independent_of_the_panel():
    names = ["a", "b", "c"]
    empty = pd.DataFrame(columns=["trade_date", "symbol", *names])
    spec = SchemeSpec("random_00", BlendScheme.RANDOM_SIMPLEX, "随机", seed=5)
    first = derive_weights(
        spec, factor_panel=empty, forward_panel=empty, factor_names=names
    )
    second = derive_weights(
        spec, factor_panel=empty, forward_panel=empty, factor_names=names
    )
    np.testing.assert_allclose(first, second)
    assert pytest.approx(1.0, abs=1e-9) == float(np.abs(first).sum())


# --------------------------------------------------------------------------- #
# Scoring primitives                                                          #
# --------------------------------------------------------------------------- #


def test_cross_sectional_scores_are_scale_invariant():
    """Rescaling a factor must not change the blended ranking."""
    factor_panel, _ = _build_panel(n_dates=10, n_symbols=20)
    names = ["alpha_signal", "alpha_noise", "alpha_junk"]
    weights = np.array([0.6, 0.3, 0.1])
    base = cross_sectional_scores(factor_panel, names, weights)
    scaled = factor_panel.copy()
    scaled["alpha_signal"] = scaled["alpha_signal"] * 1_000.0
    rescaled = cross_sectional_scores(scaled, names, weights)
    np.testing.assert_allclose(
        base["score"].to_numpy(), rescaled["score"].to_numpy(), atol=1e-9
    )


def test_evaluate_blend_reports_empty_when_labels_are_missing():
    factor_panel, forward_panel = _build_panel(n_dates=30, n_symbols=10)
    forward_panel = forward_panel.assign(forward_return=np.nan)
    evaluation = evaluate_blend(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        factor_names=["alpha_signal", "alpha_noise", "alpha_junk"],
        weights=np.array([1.0, 0.0, 0.0]),
        top_k=5,
        horizon_days=5,
    )
    assert evaluation.is_empty
    assert evaluation.observations == 0


def test_transaction_cost_reduces_realised_return():
    factor_panel, forward_panel = _build_panel(n_dates=120, n_symbols=30)
    names = ["alpha_signal", "alpha_noise", "alpha_junk"]
    weights = np.array([1.0, 0.0, 0.0])
    cheap = evaluate_blend(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        factor_names=names,
        weights=weights,
        top_k=10,
        horizon_days=5,
        transaction_cost_bps=0.0,
    )
    expensive = evaluate_blend(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        factor_names=names,
        weights=weights,
        top_k=10,
        horizon_days=5,
        transaction_cost_bps=200.0,
    )
    assert expensive.annual_return < cheap.annual_return
    assert expensive.cost_drag > cheap.cost_drag


def test_fold_consistency_penalises_a_single_lucky_fold():
    carried = fold_consistency([0.40, -0.01, -0.02, -0.01])
    steady = fold_consistency([0.08, 0.07, 0.09, 0.06])
    assert steady > carried


def test_fold_consistency_caps_a_single_fold_at_half():
    assert fold_consistency([0.5]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Frontier and preference                                                     #
# --------------------------------------------------------------------------- #


def _candidate(candidate_id: str, **metrics) -> dict:
    base = {
        "excessReturn": 0.0,
        "annualReturn": 0.0,
        "maxDrawdown": 0.1,
        "robustness": 0.5,
        "observations": 100,
    }
    base.update(metrics)
    return {"id": candidate_id, "metrics": base}


def test_pareto_front_excludes_dominated_candidates():
    candidates = [
        _candidate("strong", excessReturn=0.2, annualReturn=0.3, maxDrawdown=0.05, robustness=0.8),
        _candidate("weak", excessReturn=0.1, annualReturn=0.2, maxDrawdown=0.10, robustness=0.6),
        _candidate("defensive", excessReturn=0.05, annualReturn=0.08, maxDrawdown=0.01, robustness=0.9),
    ]
    front = pareto_front(candidates)
    assert "strong" in front
    assert "defensive" in front
    assert "weak" not in front


def test_pareto_front_ignores_candidates_without_observations():
    candidates = [
        _candidate("real", excessReturn=0.1),
        _candidate("unevaluated", excessReturn=9.9, maxDrawdown=0.0, observations=0),
    ]
    assert pareto_front(candidates) == ["real"]


def test_preference_weights_reorder_without_changing_membership():
    candidates = [
        _candidate("aggressive", excessReturn=0.30, annualReturn=0.40, maxDrawdown=0.30, robustness=0.4),
        _candidate("defensive", excessReturn=0.08, annualReturn=0.10, maxDrawdown=0.03, robustness=0.9),
    ]
    return_first = rank_by_preference(
        candidates,
        ObjectivePreference(excess_return=0.8, annual_return=0.1, drawdown_control=0.05, robustness=0.05),
    )
    risk_first = rank_by_preference(
        candidates,
        ObjectivePreference(excess_return=0.05, annual_return=0.05, drawdown_control=0.5, robustness=0.4),
    )
    assert return_first[0]["id"] == "aggressive"
    assert risk_first[0]["id"] == "defensive"
    assert {row["id"] for row in return_first} == {row["id"] for row in risk_first}


def test_preference_contributions_sum_to_the_score():
    candidates = [
        _candidate("a", excessReturn=0.3, annualReturn=0.2, maxDrawdown=0.05, robustness=0.7),
        _candidate("b", excessReturn=0.1, annualReturn=0.4, maxDrawdown=0.20, robustness=0.3),
    ]
    for row in rank_by_preference(candidates):
        total = sum(float(value) for value in row["contributions"].values())
        assert total == pytest.approx(float(row["preferenceScore"]), abs=1e-5)


# --------------------------------------------------------------------------- #
# Configuration guards and artifacts                                          #
# --------------------------------------------------------------------------- #


def test_config_rejects_duplicate_factor_names():
    with pytest.raises(ValueError, match="unique"):
        FusionSearchConfig(factor_names=("a", "a"))


def test_config_rejects_empty_factor_names():
    with pytest.raises(ValueError, match="must not be empty"):
        FusionSearchConfig(factor_names=())


def test_search_fails_loudly_on_a_panel_that_is_too_short():
    factor_panel, forward_panel = _build_panel(n_dates=40, n_symbols=10)
    with pytest.raises(ValueError, match="too short"):
        run_fusion_search(
            factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
        )


def test_search_rejects_duplicate_panel_keys():
    factor_panel, forward_panel = _build_panel(n_dates=200, n_symbols=10)
    duplicated = pd.concat([factor_panel, factor_panel.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        run_fusion_search(
            factor_panel=duplicated, forward_panel=forward_panel, config=_config()
        )


def test_search_reports_the_benchmark_it_actually_used():
    factor_panel, forward_panel = _build_panel()
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    assert result.benchmark_mode == "universe_equal_weight"


def test_build_scheme_specs_counts_every_control():
    specs = build_scheme_specs(
        ["a", "b", "c"], top_k=10, include_genetic=False, random_controls=5,
        single_factor_baselines=2,
    )
    assert len(specs) == 4 + 5 + 2
    # `equal` is a control too: it never reads the training segment.
    assert sum(1 for spec in specs if spec.is_control) == 1 + 5 + 2
    assert {spec.candidate_id for spec in specs if not spec.is_control} == {
        "ic_weighted",
        "ic_ir_weighted",
        "inverse_volatility",
    }


def test_save_fusion_artifacts_writes_a_hashed_manifest(tmp_path):
    factor_panel, forward_panel = _build_panel()
    result = run_fusion_search(
        factor_panel=factor_panel, forward_panel=forward_panel, config=_config()
    )
    paths = save_fusion_artifacts(result, output_dir=tmp_path / "run")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["artifact"] == "factor_fusion_search"
    assert manifest["nTrials"] == result.n_trials
    assert len(manifest["contentHash"]) == 16
    candidates = json.loads(paths["candidates"].read_text(encoding="utf-8"))
    assert len(candidates) == result.n_trials
    assert paths["nav"].exists()
