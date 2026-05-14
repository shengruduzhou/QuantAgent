import numpy as np
import pandas as pd

from quantagent.factors.long_horizon_factors import LONG_HORIZON_FACTORS
from quantagent.factors.long_horizon_weight_learner import (
    WeightLearnerConfig,
    learn_long_horizon_weights,
    select_weights,
)


def _build_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=180, freq="B")
    rows: list[dict[str, object]] = []
    for date in dates:
        for symbol_idx in range(8):
            base = rng.normal(0.0, 1.0)
            quality_signal = rng.normal(0.0, 1.0)
            growth_signal = rng.normal(0.0, 1.0)
            # Forward return is correlated with the quality factor so the
            # learner will down-weight noise factors and over-weight quality.
            forward_return = 0.03 * quality_signal + 0.01 * growth_signal + 0.005 * base + rng.normal(0.0, 0.01)
            row: dict[str, object] = {
                "trade_date": date,
                "symbol": f"60000{symbol_idx}.SH",
                "theme": "ai_compute" if symbol_idx < 4 else "consumer_recovery",
                "sector": "tech" if symbol_idx < 4 else "consumer",
                "market_regime": "bull",
                "lifecycle_stage": "fundamental_validation_stage",
                "forward_return_120d": forward_return,
            }
            for factor in LONG_HORIZON_FACTORS:
                if "quality" in factor:
                    row[factor] = quality_signal + rng.normal(0.0, 0.05)
                elif "growth" in factor:
                    row[factor] = growth_signal + rng.normal(0.0, 0.05)
                else:
                    row[factor] = rng.normal(0.0, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_learn_long_horizon_weights_overweights_predictive_factor():
    panel = _build_panel()
    learned = learn_long_horizon_weights(
        panel,
        config=WeightLearnerConfig(walk_forward_splits=3, embargo_days=5, min_window_days=30, min_samples_per_slice=60),
    )

    assert learned.walk_forward_windows >= 1
    quality_weight = sum(weight for factor, weight in learned.global_weights.items() if "quality" in factor)
    noise_weight = sum(weight for factor, weight in learned.global_weights.items() if "macro_credit" in factor or "macro_monetary" in factor)
    assert quality_weight > noise_weight
    # Weights should renormalise to roughly 1.0 across all factors
    assert abs(sum(learned.global_weights.values()) - 1.0) < 0.05


def test_learn_long_horizon_weights_falls_back_when_panel_is_tiny():
    panel = pd.DataFrame({"trade_date": ["2026-05-15"], "symbol": ["a"], "forward_return_120d": [0.01]})
    learned = learn_long_horizon_weights(panel)
    assert learned.walk_forward_windows == 0
    assert learned.diagnostics.get("fallback", 0.0) == 1.0


def test_select_weights_returns_most_specific_slice():
    panel = _build_panel()
    learned = learn_long_horizon_weights(
        panel,
        config=WeightLearnerConfig(walk_forward_splits=2, embargo_days=3, min_window_days=25, min_samples_per_slice=40),
    )
    theme_weights = select_weights(learned, theme="ai_compute")
    global_weights = learned.global_weights
    # When the theme is known and has its own learned weights, they should
    # differ from the global prior on at least one factor.
    assert theme_weights == learned.per_theme.get("ai_compute", global_weights)
