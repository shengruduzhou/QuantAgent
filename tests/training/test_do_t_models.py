from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.training.do_t_models import TrainedDoTModels, predict_model_signals


class _FixedClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class _FixedRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def test_predict_model_signals_does_not_read_future_labels() -> None:
    models = TrainedDoTModels(
        feature_columns=["entry_feature"],
        backend="test",
        classifiers={
            "model_sell_high_success": _FixedClassifier(0.61),
            "model_buyback_now": _FixedClassifier(0.62),
            "model_buy_low_success": _FixedClassifier(0.63),
            "model_sell_after_buy_success": _FixedClassifier(0.64),
            "model_failure_risk": _FixedClassifier(0.11),
            "model_breakdown_risk": _FixedClassifier(0.12),
            "model_eod_restore": _FixedClassifier(0.13),
        },
        regressors={
            "model_sell_high_edge": _FixedRegressor(21.0),
            "model_buy_low_edge": _FixedRegressor(22.0),
            "model_buyback_edge": _FixedRegressor(23.0),
            "model_wait_extra_edge": _FixedRegressor(24.0),
            "model_miss_rebound_risk": _FixedRegressor(25.0),
            "model_adverse_after_sell": _FixedRegressor(26.0),
            "model_adverse_after_buy": _FixedRegressor(-27.0),
        },
    )
    rows = pd.DataFrame(
        {
            "entry_feature": [1.0],
            "label_buyback_now_edge_bps": [9_999.0],
            "label_wait_extra_edge_bps": [8_888.0],
            "label_miss_rebound_risk": [7_777.0],
            "label_adverse_excursion_after_sell": [6_666.0],
            "label_adverse_excursion_after_buy": [-5_555.0],
            "label_sell_after_buy_success": [0.0],
            "label_sell_high_eod_restore": [1.0],
        }
    )

    signal = predict_model_signals(models, rows)[0]

    assert signal.p_sell_high_success == 0.61
    assert signal.p_buyback_now == 0.62
    assert signal.p_buy_low_success == 0.63
    assert signal.p_sell_after_buy_success == 0.64
    assert signal.p_fail_new_high == 0.11
    assert signal.p_fail_breakdown == 0.12
    assert signal.p_eod_restore == 0.13
    assert signal.expected_sell_high_gain_bps == 21.0
    assert signal.expected_buy_low_gain_bps == 22.0
    assert signal.expected_buyback_edge_bps == 23.0
    assert signal.wait_extra_edge_bps == 24.0
    assert signal.miss_rebound_risk_bps == 25.0
    assert signal.expected_chase_loss_bps == 26.0
    assert signal.expected_breakdown_loss_bps == 27.0
    assert signal.risk_score == 0.12
