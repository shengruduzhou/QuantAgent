"""Dependency-light meta-labeling for signal filtering and sizing.

The primary model decides direction. This second-stage logistic model predicts
whether a signal succeeds using only information available at entry time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray
    intercept: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = _sigmoid(
            np.asarray(values, dtype=float) @ self.coefficients + self.intercept
        )
        return np.column_stack([1.0 - probability, probability])


@dataclass
class MetaLabeler:
    model: LogisticModel
    features: list[str]
    mean: pd.Series
    std: pd.Series

    def predict_success(self, X: pd.DataFrame) -> np.ndarray:
        missing = [feature for feature in self.features if feature not in X.columns]
        if missing:
            raise ValueError(f"meta-label input missing features: {missing}")
        z = (
            (X[self.features].astype(float) - self.mean) / self.std
        ).fillna(0.0).to_numpy(dtype=float)
        return self.model.predict_proba(z)[:, 1]


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
    learning_rate: float,
    class_weight_balanced: bool,
) -> LogisticModel:
    n_rows, n_features = X.shape
    coefficients = np.zeros(n_features, dtype=float)
    intercept = 0.0

    if class_weight_balanced:
        positives = max(1, int((y == 1).sum()))
        negatives = max(1, int((y == 0).sum()))
        sample_weight = np.where(
            y == 1,
            n_rows / (2.0 * positives),
            n_rows / (2.0 * negatives),
        )
    else:
        sample_weight = np.ones(n_rows, dtype=float)

    previous_loss: float | None = None
    for _ in range(max_iter):
        logits = X @ coefficients + intercept
        probability = np.clip(_sigmoid(logits), 1e-8, 1.0 - 1e-8)
        error = (probability - y) * sample_weight
        grad_w = X.T @ error / n_rows + regularization * coefficients
        grad_b = float(error.mean())

        coefficients -= learning_rate * grad_w
        intercept -= learning_rate * grad_b

        updated_probability = np.clip(
            _sigmoid(X @ coefficients + intercept), 1e-8, 1.0 - 1e-8
        )
        loss = float(
            -np.mean(
                sample_weight
                * (
                    y * np.log(updated_probability)
                    + (1.0 - y) * np.log(1.0 - updated_probability)
                )
            )
            + 0.5 * regularization * np.dot(coefficients, coefficients)
        )
        if previous_loss is not None and abs(previous_loss - loss) <= 1e-10 * max(
            1.0, abs(previous_loss)
        ):
            break
        previous_loss = loss

    return LogisticModel(coefficients=coefficients, intercept=intercept)


def fit_meta_labeler(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "success",
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    learning_rate: float = 0.05,
) -> MetaLabeler:
    """Fit a balanced L2 logistic model on completed primary signals.

    ``C`` follows scikit-learn's inverse-regularisation convention. Because the
    data loss is averaged, the equivalent penalty scale is ``1 / (C * n)``;
    using ``1 / C`` over-regularises large datasets and collapses probabilities
    toward 0.5.
    """
    if C <= 0:
        raise ValueError("C must be positive")
    required = set(features) | {label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"meta-label frame missing columns: {sorted(missing)}")

    data = df.dropna(subset=[label_col]).copy()
    if data.empty:
        raise ValueError("meta-label frame has no labelled rows")
    y = pd.to_numeric(data[label_col], errors="coerce").dropna().astype(int)
    data = data.loc[y.index]
    if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
        raise ValueError("meta-label target must contain both binary classes")

    X = data[features].apply(pd.to_numeric, errors="coerce")
    mean = X.mean()
    std = X.std(ddof=0).replace(0, 1.0).fillna(1.0)
    z = ((X - mean) / std).fillna(0.0).to_numpy(dtype=float)
    model = _fit_logistic(
        z,
        y.to_numpy(dtype=float),
        regularization=1.0 / (C * max(len(data), 1)),
        max_iter=max_iter,
        learning_rate=learning_rate,
        class_weight_balanced=True,
    )
    return MetaLabeler(model=model, features=list(features), mean=mean, std=std)


def build_dot_meta_dataset(fsm_results: pd.DataFrame) -> pd.DataFrame:
    """Build one completed round-trip row per entered intraday signal."""
    data = fsm_results.copy()
    if "exit_reason" not in data.columns:
        raise ValueError("fsm_results missing exit_reason")
    data["success"] = data["exit_reason"].astype(str).eq("止盈").astype(int)
    return data


def meta_filter(p_success: np.ndarray, *, floor: float = 0.5) -> np.ndarray:
    """Return a zero-to-one size multiplier, not a direct order instruction."""
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    probability = np.asarray(p_success, dtype=float)
    take = probability >= floor
    return np.where(
        take,
        np.clip(
            (probability - floor) / max(1.0 - floor, 1e-12), 0.0, 1.0
        ),
        0.0,
    )
