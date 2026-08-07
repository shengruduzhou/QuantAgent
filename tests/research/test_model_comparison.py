"""Tests for the linear-vs-nonlinear comparison protocol.

The harness is only worth anything if it *refuses*. Three synthetic
data-generating processes pin that down: one with a planted interaction (must
accept), one purely additive (must keep the linear baseline), and one pure noise
(must keep the linear baseline). A harness that accepted all three would be a
rubber stamp, and a harness that rejected all three would be useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.research.model_comparison import (
    ComparisonConfig,
    LINEAR_BASELINE,
    run_model_comparison,
    save_comparison_report,
)


def _panel(kind: str, seed: int = 7, n_days: int = 600, n_symbols: int = 120):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    regime = pd.Series(
        np.where(np.arange(n_days) // 60 % 2 == 0, "bull", "bear"), index=dates
    )
    frames = []
    for position, date in enumerate(dates):
        x1 = rng.normal(size=n_symbols)
        x2 = rng.normal(size=n_symbols)
        x3 = rng.normal(size=n_symbols)
        noise = rng.normal(0, 0.02, n_symbols)
        if kind == "interaction":
            sign = 1.0 if regime.iloc[position] == "bull" else -1.0
            label = 0.004 * sign * x1 + 0.006 * np.sign(x1) * np.abs(x1) * np.tanh(x2) + noise
        elif kind == "additive":
            label = 0.004 * x1 + 0.003 * x2 + noise
        else:
            label = noise
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "f1": x1,
                    "f2": x2,
                    "f3": x3,
                    "forward_return_5d": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), regime


def _config(**overrides) -> ComparisonConfig:
    base = dict(
        n_folds=5,
        holdout_folds=1,
        valid_size_days=40,
        min_train_days=200,
        min_symbols_per_date=30,
        top_k=20,
        # Test-suite economics, not research settings: the planted signals are
        # strong enough that 60 trees separate the arms as cleanly as 300, and
        # capping threads keeps a full suite run from oversubscribing the box.
        gbm_estimators=60,
        gbm_n_jobs=2,
    )
    base.update(overrides)
    return ComparisonConfig(**base)


class TestVerdicts:
    def test_planted_interaction_is_accepted(self):
        panel, regime = _panel("interaction", seed=7)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        assert report.verdict == "production_accepted"
        assert report.champion != LINEAR_BASELINE
        winner = next(t for t in report.incremental if t.arm == report.champion)
        assert winner.ic_delta_t_stat > 2.0
        assert winner.ic_delta > 0.005

    def test_additive_process_keeps_the_linear_baseline(self):
        panel, regime = _panel("additive", seed=11)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        assert report.verdict == "hypothesis_rejected"
        assert report.champion == LINEAR_BASELINE
        assert all(not test.passes for test in report.incremental)

    def test_pure_noise_keeps_the_linear_baseline(self):
        panel, regime = _panel("noise", seed=23)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        assert report.verdict == "hypothesis_rejected"
        assert report.champion == LINEAR_BASELINE


class TestGates:
    def test_significant_but_immaterial_gain_is_refused(self):
        """A stable +0.001 IC clears any t-test and is still not worth carrying.

        This is the failure the materiality floor exists for: on the additive
        process the OOF stack is reliably a hair better than the baseline and
        would otherwise be promoted on statistical significance alone.
        """
        panel, regime = _panel("additive", seed=11)

        permissive = run_model_comparison(
            panel,
            ["f1", "f2", "f3"],
            config=_config(min_ic_delta=0.0, min_net_return_delta=0.0),
            regime_by_date=regime,
        )
        strict = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        # Same data, same folds; only the materiality floor differs.
        assert permissive.champion != strict.champion
        assert strict.champion == LINEAR_BASELINE

    def test_reasons_name_every_failed_gate(self):
        panel, regime = _panel("noise", seed=23)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        for test in report.incremental:
            assert test.reasons, f"{test.arm} failed without saying why"


class TestHoldoutDiscipline:
    def test_holdout_folds_are_scored_but_excluded_from_selection(self):
        panel, regime = _panel("interaction", seed=7)
        config = _config(holdout_folds=1)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=config, regime_by_date=regime
        )

        roles = [window["role"] for window in report.fold_windows]
        assert roles.count("holdout") == config.holdout_folds
        assert roles[-config.holdout_folds :] == ["holdout"] * config.holdout_folds
        measured = [arm for arm in report.arms if arm.status == "measured"]
        assert measured
        for arm in measured:
            # Holdout is measured and reported...
            assert arm.holdout_metrics, f"{arm.name} has no holdout metrics"
            # ...but every selection statistic is built from selection folds only.
            assert set(arm.fold_ic).isdisjoint(
                {
                    int(window["foldId"])
                    for window in report.fold_windows
                    if window["role"] == "holdout"
                }
            )

    def test_selection_ic_ignores_holdout_dates(self):
        panel, regime = _panel("interaction", seed=7)
        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )
        holdout_start = pd.Timestamp(
            next(w["validStart"] for w in report.fold_windows if w["role"] == "holdout")
        )
        arm = next(a for a in report.arms if a.status == "measured")

        assert arm.daily_ic.index.max() < holdout_start


class TestDataValidity:
    def test_missing_label_is_data_invalid_not_rejection(self):
        panel, _ = _panel("additive")
        panel = panel.drop(columns=["forward_return_5d"])

        report = run_model_comparison(panel, ["f1", "f2"], config=_config())

        assert report.verdict == "data_invalid"
        assert "forward_return_5d" in report.verdict_reasons[0]

    def test_too_few_folds_is_data_invalid(self):
        panel, _ = _panel("additive", n_days=120)

        report = run_model_comparison(panel, ["f1", "f2"], config=_config())

        assert report.verdict == "data_invalid"
        assert "fold" in report.verdict_reasons[0]

    def test_single_factor_is_data_invalid(self):
        panel, _ = _panel("additive")

        report = run_model_comparison(panel, ["f1"], config=_config())

        assert report.verdict == "data_invalid"


class TestFailureClassification:
    """An unusable prediction and a crash are different research events."""

    def test_constant_prediction_is_model_invalid_not_pipeline_failed(self, monkeypatch):
        import quantagent.research.model_comparison as mc

        panel, regime = _panel("additive", seed=11)

        def constant(train_x, train_y, test_x, alpha):
            return np.zeros(len(test_x))

        def constant_gbm(train_x, train_y, test_x, config):
            return np.zeros(len(test_x))

        monkeypatch.setattr(mc, "_ridge_fit_predict", constant)
        monkeypatch.setattr(mc, "_gbm_fit_predict", constant_gbm)

        report = mc.run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        # The baseline itself returns a constant, so there is nothing to compare
        # against — and the reason is the model, not the machinery.
        assert report.verdict == "model_invalid"
        assert any("constant prediction" in arm.error for arm in report.arms)

    def test_estimator_exception_is_pipeline_failed(self, monkeypatch):
        import quantagent.research.model_comparison as mc

        panel, regime = _panel("additive", seed=11)

        def boom(*args, **kwargs):
            raise RuntimeError("BLAS exploded")

        monkeypatch.setattr(mc, "_ridge_fit_predict", boom)
        monkeypatch.setattr(mc, "_gbm_fit_predict", boom)

        report = mc.run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        assert report.verdict == "pipeline_failed"
        assert any("BLAS exploded" in reason for reason in report.verdict_reasons)


class TestReporting:
    def test_report_declares_model_class_per_arm(self):
        panel, regime = _panel("interaction", seed=7)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        by_name = {arm.name: arm for arm in report.arms}
        assert by_name[LINEAR_BASELINE].model_class == "rank_weighted_additive"
        assert by_name[LINEAR_BASELINE].feature_summary["representsInteraction"] is False
        assert by_name["linear_pair_interaction"].model_class == "factor_interaction"
        assert by_name["linear_pair_interaction"].feature_summary["representsInteraction"] is True

    def test_report_carries_all_four_evaluation_dimensions(self):
        panel, regime = _panel("interaction", seed=7)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )
        metrics = next(a for a in report.arms if a.status == "measured").selection_metrics

        for key in ("rank_ic_mean", "rank_ic_ir", "r2_oos_vs_zero"):  # prediction
            assert key in metrics
        for key in ("net_annual_return", "net_sharpe_overlapping", "calmar"):  # economic value
            assert key in metrics
        for key in ("avg_turnover", "cost_drag_annual"):  # trading reality
            assert key in metrics
        assert np.isfinite(report.n_trials)  # robustness accounting

    def test_trial_count_is_the_number_of_arms_run(self):
        panel, regime = _panel("interaction", seed=7)

        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        assert report.n_trials == len(report.arms)

    def test_save_writes_a_readable_artifact(self, tmp_path):
        panel, regime = _panel("additive", seed=11)
        report = run_model_comparison(
            panel, ["f1", "f2", "f3"], config=_config(), regime_by_date=regime
        )

        path = save_comparison_report(report, tmp_path)

        assert path.exists()
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["verdict"] == report.verdict
        assert payload["champion"] == report.champion
        assert len(payload["arms"]) == len(report.arms)


def test_config_rejects_a_holdout_that_consumes_every_fold():
    with pytest.raises(ValueError, match="n_folds must exceed holdout_folds"):
        ComparisonConfig(n_folds=2, holdout_folds=2)
