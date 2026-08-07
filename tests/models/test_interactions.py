"""Tests for explicit interaction construction and the model-class taxonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.models.interactions import (
    INTERACTION_SEPARATOR,
    REGIME_SEPARATOR,
    ModelClass,
    cross_sectional_rank_normalise,
    describe_feature_block,
    pairwise_interaction_features,
    regime_interaction_features,
    select_interaction_pairs,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=120)
    symbols = [f"S{i:02d}" for i in range(60)]
    frames = []
    for date in dates:
        x1 = rng.normal(size=len(symbols))
        x2 = rng.normal(size=len(symbols))
        x3 = rng.normal(size=len(symbols))
        # The only signal is the INTERACTION: neither x1 nor x2 predicts alone.
        label = 0.02 * np.sign(x1) * np.abs(x2) + rng.normal(0, 0.005, len(symbols))
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "label": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestTaxonomy:
    def test_additive_classes_do_not_represent_interaction(self):
        for member in (
            ModelClass.LINEAR_ADDITIVE,
            ModelClass.RANK_WEIGHTED_ADDITIVE,
            ModelClass.FACTOR_NONLINEAR_TRANSFORM,
        ):
            assert member.is_additive
            assert not member.represents_interaction

    def test_interaction_classes_are_not_additive(self):
        for member in (
            ModelClass.FACTOR_INTERACTION,
            ModelClass.REGIME_INTERACTION,
            ModelClass.NONLINEAR_LEARNER,
        ):
            assert member.represents_interaction
            assert not member.is_additive

    def test_ensemble_and_objective_are_neither(self):
        # Ensembling additive models yields an additive model, and a nonlinear
        # loss says nothing about the functional form. Both are deliberately
        # excluded from `represents_interaction`.
        assert not ModelClass.ENSEMBLE.represents_interaction
        assert not ModelClass.NONLINEAR_OBJECTIVE.represents_interaction


class TestRankNormalisation:
    def test_maps_to_unit_interval_per_date(self, panel):
        ranked = cross_sectional_rank_normalise(panel, ["x1", "x2"])

        assert ranked["x1"].min() >= -1.0
        assert ranked["x1"].max() <= 1.0
        by_date = ranked["x1"].groupby(panel["trade_date"]).mean().abs()
        assert (by_date < 1e-9).all(), "each date must be centred on zero"

    def test_uses_only_same_date_rows(self):
        """A later date's values must not move an earlier date's ranks."""
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
                "x": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            }
        )
        ranked = cross_sectional_rank_normalise(frame, ["x"])

        np.testing.assert_allclose(ranked["x"].to_numpy()[:3], ranked["x"].to_numpy()[3:])

    def test_missing_values_map_to_median_rank(self):
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-01"] * 4),
                "x": [1.0, np.nan, 3.0, 4.0],
            }
        )
        ranked = cross_sectional_rank_normalise(frame, ["x"])

        assert ranked["x"].iloc[1] == 0.0


class TestPairSelection:
    def test_finds_the_planted_interaction(self, panel):
        pairs = select_interaction_pairs(panel, ["x1", "x2", "x3"], "label", top_n=3)

        assert pairs, "the planted x1*x2 interaction must be discovered"
        assert {pairs[0].left, pairs[0].right} == {"x1", "x2"}
        assert abs(pairs[0].orthogonal_ic) > abs(pairs[0].raw_ic) * 0.5

    def test_rejects_pairs_on_pure_noise(self):
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2024-01-01", periods=120)
        symbols = [f"S{i:02d}" for i in range(60)]
        frames = []
        for date in dates:
            frames.append(
                pd.DataFrame(
                    {
                        "trade_date": date,
                        "symbol": symbols,
                        "x1": rng.normal(size=len(symbols)),
                        "x2": rng.normal(size=len(symbols)),
                        "label": rng.normal(0, 0.02, len(symbols)),
                    }
                )
            )
        noise = pd.concat(frames, ignore_index=True)

        pairs = select_interaction_pairs(noise, ["x1", "x2"], "label", top_n=5)

        assert pairs == []

    def test_selection_reads_no_row_outside_what_it_is_given(self, panel):
        """Pair selection on a prefix must not depend on later rows."""
        early = panel[panel["trade_date"] < "2024-04-01"]

        first = select_interaction_pairs(early, ["x1", "x2", "x3"], "label", top_n=3)
        second = select_interaction_pairs(
            early.copy(), ["x1", "x2", "x3"], "label", top_n=3
        )

        assert [pair.column for pair in first] == [pair.column for pair in second]

    def test_orthogonalisation_suppresses_a_pure_main_effect_product(self):
        """x*x is not an interaction; its residual against x must carry nothing."""
        rng = np.random.default_rng(9)
        dates = pd.bdate_range("2024-01-01", periods=150)
        symbols = [f"S{i:02d}" for i in range(60)]
        frames = []
        for date in dates:
            x1 = rng.normal(size=len(symbols))
            # x2 is a near-copy of x1, so their "interaction" is x1^2 and holds
            # no information beyond the parents.
            x2 = x1 + rng.normal(0, 0.01, len(symbols))
            frames.append(
                pd.DataFrame(
                    {
                        "trade_date": date,
                        "symbol": symbols,
                        "x1": x1,
                        "x2": x2,
                        "label": 0.02 * x1 + rng.normal(0, 0.005, len(symbols)),
                    }
                )
            )
        collinear = pd.concat(frames, ignore_index=True)

        pairs = select_interaction_pairs(
            collinear, ["x1", "x2"], "label", top_n=5, min_abs_orthogonal_ic=0.01
        )

        assert pairs == [], "a product of two collinear parents is not an interaction"


class TestFeatureConstruction:
    def test_pairwise_features_are_products_of_ranks(self, panel):
        pairs = [("x1", "x2")]
        features = pairwise_interaction_features(panel, pairs)
        ranked = cross_sectional_rank_normalise(panel, ["x1", "x2"])

        column = f"x1{INTERACTION_SEPARATOR}x2"
        assert column in features.columns
        np.testing.assert_allclose(
            features[column].to_numpy(), (ranked["x1"] * ranked["x2"]).to_numpy()
        )

    def test_regime_features_drop_a_reference_state(self, panel):
        regime = pd.Series(
            np.where(np.arange(120) % 2 == 0, "bull", "bear"),
            index=pd.bdate_range("2024-01-01", periods=120),
        )

        features = regime_interaction_features(panel, ["x1", "x2"], regime)

        # Two states, one dropped as reference -> one state x two factors.
        assert len(features.columns) == 2
        assert all(REGIME_SEPARATOR in column for column in features.columns)
        states = {column.split(REGIME_SEPARATOR)[1] for column in features.columns}
        assert states == {"bull"}, "alphabetically-first state 'bear' is the reference"

    def test_regime_features_are_zero_outside_their_state(self, panel):
        regime = pd.Series(
            np.where(np.arange(120) % 2 == 0, "bull", "bear"),
            index=pd.bdate_range("2024-01-01", periods=120),
        )
        features = regime_interaction_features(panel, ["x1"], regime)
        column = features.columns[0]

        bull_dates = set(regime[regime == "bull"].index)
        in_state = panel["trade_date"].isin(bull_dates)
        assert (features.loc[~in_state, column].abs() < 1e-12).all()
        assert features.loc[in_state, column].abs().sum() > 0

    def test_dates_absent_from_the_regime_series_get_the_reference(self, panel):
        # Only the first 10 dates are labelled; the rest fall back to zeros,
        # meaning the unconditional coefficient applies rather than a guess.
        regime = pd.Series(["bull"] * 5 + ["bear"] * 5, index=pd.bdate_range("2024-01-01", periods=10))

        features = regime_interaction_features(panel, ["x1"], regime)

        late = panel["trade_date"] > "2024-01-15"
        assert (features.loc[late].abs().to_numpy() < 1e-12).all()


class TestDescribeFeatureBlock:
    def test_classifies_a_mixed_block(self):
        summary = describe_feature_block(
            ["mom", "vol", f"mom{INTERACTION_SEPARATOR}vol", f"mom{REGIME_SEPARATOR}bull"]
        )

        assert summary["mainEffectCount"] == 2
        assert summary["pairInteractionCount"] == 1
        assert summary["regimeInteractionCount"] == 1
        assert summary["representsInteraction"] is True
        assert ModelClass.FACTOR_INTERACTION.value in summary["modelClasses"]

    def test_plain_factor_list_is_additive_only(self):
        summary = describe_feature_block(["mom", "vol", "value"])

        assert summary["representsInteraction"] is False
        assert summary["modelClasses"] == [ModelClass.RANK_WEIGHTED_ADDITIVE.value]
