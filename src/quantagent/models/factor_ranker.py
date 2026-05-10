from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRankerPrediction:
    prediction: pd.Series
    uncertainty: pd.Series


class FactorRanker:
    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.model: object | None = None
        self.feature_columns: list[str] = []
        self.backend = "linear_fallback"
        self._coef: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> "FactorRanker":
        data = frame[feature_columns + [target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        self.feature_columns = feature_columns
        if data.empty:
            self._coef = np.zeros(len(feature_columns) + 1)
            return self
        x = data[feature_columns].to_numpy(dtype=float)
        y = data[target_column].to_numpy(dtype=float)
        model = self._build_model()
        if model is not None:
            model.fit(x, y)
            self.model = model
        else:
            x_design = np.column_stack([np.ones(len(x)), x])
            self._coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        return self

    def predict(self, frame: pd.DataFrame) -> FactorRankerPrediction:
        if not self.feature_columns:
            raise ValueError("Model is not fitted")
        x = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        if self.model is not None:
            pred = np.asarray(self.model.predict(x), dtype=float)
        else:
            coef = self._coef if self._coef is not None else np.zeros(len(self.feature_columns) + 1)
            pred = np.column_stack([np.ones(len(x)), x]) @ coef
        prediction = pd.Series(pred, index=frame.index, name="factor_ranker_prediction")
        uncertainty = pd.Series(np.nan, index=frame.index, name="prediction_uncertainty")
        return FactorRankerPrediction(prediction=prediction, uncertainty=uncertainty)

    def feature_importance(self) -> pd.Series:
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            values = getattr(self.model, "feature_importances_")
            return pd.Series(values, index=self.feature_columns, dtype=float)
        if self._coef is not None and len(self._coef) == len(self.feature_columns) + 1:
            return pd.Series(np.abs(self._coef[1:]), index=self.feature_columns, dtype=float)
        return pd.Series(0.0, index=self.feature_columns, dtype=float)

    def _build_model(self) -> object | None:
        try:
            from lightgbm import LGBMRegressor

            self.backend = "lightgbm"
            return LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
                objective="regression",
                verbosity=-1,
            )
        except Exception:
            pass
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.backend = "sklearn_hist_gradient_boosting"
            return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=self.random_state)
        except Exception:
            self.backend = "linear_fallback"
            return None


def purged_walk_forward_split(
    dates: pd.Series,
    n_splits: int = 5,
    embargo: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    if n_splits <= 1 or len(unique_dates) < n_splits:
        raise ValueError("n_splits is too large for available dates")
    fold_sizes = np.array_split(unique_dates.to_numpy(), n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = pd.to_datetime(dates)
    for test_dates in fold_sizes:
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])
        train_mask = (date_values < test_start - pd.Timedelta(days=embargo)) | (date_values > test_end + pd.Timedelta(days=embargo))
        test_mask = (date_values >= test_start) & (date_values <= test_end)
        splits.append((np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())))
    return splits

