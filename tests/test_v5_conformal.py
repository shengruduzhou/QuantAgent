import numpy as np
import pandas as pd

from quantagent.training.conformal_calibrator import ConformalCalibrator


def test_split_conformal_attaches_intervals_with_target_coverage():
    rng = np.random.default_rng(42)
    truth = rng.normal(0.0, 0.02, size=500)
    pred = truth + rng.normal(0.0, 0.01, size=500)
    calib_pred, test_pred = pred[:250], pred[250:]
    calib_truth, test_truth = truth[:250], truth[250:]

    calibrator = ConformalCalibrator(alpha=0.1, mode="split").fit(calib_pred, calib_truth)
    predictions = pd.DataFrame({"alpha": test_pred})
    out = calibrator.attach_interval(predictions, vol_column=None)
    coverage = calibrator.coverage(out, test_truth)
    assert 0.80 <= coverage <= 1.0
    assert "alpha_lower" in out.columns and "alpha_upper" in out.columns
    assert (out["alpha_upper"] >= out["alpha_lower"]).all()


def test_cqr_conformal_with_quantile_heads():
    rng = np.random.default_rng(7)
    truth = rng.normal(0.0, 0.03, size=400)
    pred = truth + rng.normal(0.0, 0.01, size=400)
    q_low = pred - 0.02
    q_high = pred + 0.02
    calib_idx = slice(0, 200)
    test_idx = slice(200, 400)

    calibrator = ConformalCalibrator(alpha=0.1, mode="cqr").fit(
        pred[calib_idx],
        truth[calib_idx],
        calibration_lower=q_low[calib_idx],
        calibration_upper=q_high[calib_idx],
    )
    predictions = pd.DataFrame(
        {"alpha": pred[test_idx], "q_low": q_low[test_idx], "q_high": q_high[test_idx]}
    )
    out = calibrator.attach_interval(predictions, vol_column=None)
    assert (out["alpha_upper"] >= out["alpha_lower"]).all()
    alert = calibrator.drift_alert(out, truth[test_idx])
    assert "realized_coverage" in alert
