"""V5 conformal calibration wrapper.

Bridges ``quant_math.conformal`` (split conformal + CQR) into the
training/inference pipeline. V4 emitted q_low / q_high heads from the
multi-tower model but never calibrated them; V5 closes that loop:

- ``fit(val_predictions, val_truth)`` stores residuals
- ``attach_interval(predictions)`` augments a DataFrame with calibrated
  lower / upper / conformal_confidence columns
- ``coverage(test_predictions, test_truth)`` reports realized empirical
  coverage so downstream services can detect drift

The calibrator is intentionally stateless beyond the fitted residuals; in
production, persist via ``state()`` and ``load_state()``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.quant_math.conformal import (
    conformal_quantile,
    cqr_intervals,
    split_conformal_residuals,
)


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    mode: str = "split"  # "split" or "cqr"
    _residuals: np.ndarray | None = None
    _cqr_err: np.ndarray | None = None

    def fit(
        self,
        calibration_pred: np.ndarray,
        calibration_truth: np.ndarray,
        calibration_lower: np.ndarray | None = None,
        calibration_upper: np.ndarray | None = None,
    ) -> "ConformalCalibrator":
        if self.mode == "cqr":
            if calibration_lower is None or calibration_upper is None:
                raise ValueError("CQR mode requires calibration_lower and calibration_upper")
            self._cqr_err = np.maximum(
                calibration_lower - calibration_truth,
                calibration_truth - calibration_upper,
            )
        else:
            self._residuals = split_conformal_residuals(calibration_pred, calibration_truth)
        return self

    def attach_interval(
        self,
        predictions: pd.DataFrame,
        pred_column: str = "alpha",
        q_low_column: str | None = "q_low",
        q_high_column: str | None = "q_high",
        vol_column: str | None = "volatility_forecast",
    ) -> pd.DataFrame:
        out = predictions.copy()
        if self.mode == "cqr":
            if self._cqr_err is None:
                raise RuntimeError("Call fit(...) before attach_interval in CQR mode")
            if q_low_column is None or q_high_column is None:
                raise ValueError("CQR mode requires q_low_column and q_high_column")
            q = conformal_quantile(self._cqr_err, self.alpha)
            out["alpha_lower"] = out[q_low_column].to_numpy() - q
            out["alpha_upper"] = out[q_high_column].to_numpy() + q
        else:
            if self._residuals is None:
                raise RuntimeError("Call fit(...) before attach_interval in split mode")
            q = conformal_quantile(self._residuals, self.alpha)
            base = out[pred_column].to_numpy()
            out["alpha_lower"] = base - q
            out["alpha_upper"] = base + q
        width = out["alpha_upper"].to_numpy() - out["alpha_lower"].to_numpy()
        if vol_column and vol_column in out.columns:
            baseline = out[vol_column].clip(lower=1e-6).to_numpy()
        else:
            ref = np.std(self._residuals) if self._residuals is not None else 1.0
            baseline = np.full_like(width, fill_value=max(float(ref), 1e-6))
        out["conformal_confidence"] = np.exp(-width / (2.0 * baseline))
        return out

    def coverage(self, predictions: pd.DataFrame, truth: np.ndarray) -> float:
        if "alpha_lower" not in predictions.columns or "alpha_upper" not in predictions.columns:
            raise RuntimeError("Run attach_interval before coverage()")
        lower = predictions["alpha_lower"].to_numpy()
        upper = predictions["alpha_upper"].to_numpy()
        inside = (truth >= lower) & (truth <= upper)
        return float(inside.mean())

    def drift_alert(self, predictions: pd.DataFrame, truth: np.ndarray, tolerance: float = 0.05) -> dict[str, float | bool]:
        target_coverage = 1.0 - self.alpha
        realized = self.coverage(predictions, truth)
        return {
            "realized_coverage": realized,
            "target_coverage": target_coverage,
            "drift": realized < (target_coverage - tolerance),
            "severe_drift": realized < (target_coverage - 2 * tolerance),
        }

    def state(self) -> dict[str, np.ndarray | None]:
        return {"residuals": self._residuals, "cqr_err": self._cqr_err}

    def load_state(self, state: dict[str, np.ndarray | None]) -> "ConformalCalibrator":
        self._residuals = state.get("residuals")
        self._cqr_err = state.get("cqr_err")
        return self
