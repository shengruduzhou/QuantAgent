from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalInterval:
    lower: float
    upper: float
    coverage: float


def split_conformal_residuals(
    calibration_pred: np.ndarray,
    calibration_truth: np.ndarray,
) -> np.ndarray:
    """Absolute residuals on the calibration set."""
    return np.abs(calibration_truth - calibration_pred)


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """(1 - alpha) finite-sample-corrected quantile of residuals."""
    n = len(residuals)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(residuals)[rank - 1])


def split_conformal_intervals(
    test_pred: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    q = conformal_quantile(residuals, alpha)
    return test_pred - q, test_pred + q


def cqr_intervals(
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
    calibration_lower: np.ndarray,
    calibration_upper: np.ndarray,
    calibration_truth: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Romano et al. 2019 Conformalized Quantile Regression."""
    err = np.maximum(calibration_lower - calibration_truth, calibration_truth - calibration_upper)
    q = conformal_quantile(err, alpha)
    return lower_pred - q, upper_pred + q


def confidence_from_interval(width: float, vol_scale: float) -> float:
    """Map interval width to a 0-1 confidence: narrower vs vol baseline -> higher."""
    if vol_scale <= 0:
        return 0.0
    return float(np.exp(-(width / (2.0 * vol_scale))))


def attach_conformal_to_predictions(
    predictions: pd.DataFrame,
    calibration_pred: np.ndarray,
    calibration_truth: np.ndarray,
    pred_column: str = "alpha",
    alpha: float = 0.1,
    vol_column: str | None = "volatility_forecast",
) -> pd.DataFrame:
    """Attach lower/upper/conformal_confidence columns to a predictions frame."""
    residuals = split_conformal_residuals(calibration_pred, calibration_truth)
    lower, upper = split_conformal_intervals(predictions[pred_column].to_numpy(), residuals, alpha)
    out = predictions.copy()
    out["alpha_lower"] = lower
    out["alpha_upper"] = upper
    width = upper - lower
    if vol_column and vol_column in out.columns:
        baseline = out[vol_column].clip(lower=1e-6).to_numpy()
    else:
        baseline = np.full_like(width, fill_value=max(np.std(calibration_truth), 1e-6))
    out["conformal_confidence"] = np.exp(-width / (2.0 * baseline))
    return out
