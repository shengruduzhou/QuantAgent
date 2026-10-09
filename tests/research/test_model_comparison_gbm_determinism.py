"""The GBM arm must be reproducible run to run, not just seeded (round-29 ENG_FUSION).

Two identical CLI runs of ``audit-nonlinear-factors`` on the same panel and
commit produced different gbm / ensemble_stack metrics and a different PBO:
LightGBM was left to choose col-wise or row-wise histogram building by timing
both, which depends on machine load.
"""

from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from quantagent.research import model_comparison as mc  # noqa: E402


def _data(seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(4000, 12))
    y = 0.02 * x[:, 0] * x[:, 1] + 0.01 * x[:, 2] + 0.05 * rng.standard_t(3, size=4000)
    return x, y, rng.uniform(-1.0, 1.0, size=(500, 12))


def test_gbm_pins_histogram_mode_and_requests_deterministic_training(monkeypatch):
    captured: dict[str, object] = {}

    class Spy(lgb.LGBMRegressor):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(lgb, "LGBMRegressor", Spy)
    x, y, test_x = _data()
    mc._gbm_fit_predict(x, y, test_x, mc.ComparisonConfig(gbm_estimators=5, gbm_n_jobs=1))
    assert captured.get("deterministic") is True
    # Auto mode times both histogram builders and keeps the faster: load-dependent.
    assert captured.get("force_col_wise") is True or captured.get("force_row_wise") is True


def test_gbm_predictions_do_not_depend_on_thread_count():
    x, y, test_x = _data()
    one = mc._gbm_fit_predict(x, y, test_x, mc.ComparisonConfig(gbm_estimators=60, gbm_n_jobs=1))
    four = mc._gbm_fit_predict(x, y, test_x, mc.ComparisonConfig(gbm_estimators=60, gbm_n_jobs=4))
    again = mc._gbm_fit_predict(x, y, test_x, mc.ComparisonConfig(gbm_estimators=60, gbm_n_jobs=4))
    assert np.array_equal(one, four)
    assert np.array_equal(four, again)
