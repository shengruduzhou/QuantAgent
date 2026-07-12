"""Score calibration and split-conformal uncertainty.

This implementation intentionally has no mandatory scikit-learn dependency.
A small weighted pool-adjacent-violators (PAV) calibrator is sufficient for the
one-dimensional cross-sectional rank mapping and keeps module import usable in
base installations and CI collection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _xrank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(pct=True)


@dataclass(frozen=True)
class IsotonicMap:
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if self.x_thresholds.size == 0:
            return np.full(x.shape, 0.5, dtype=float)
        if self.x_thresholds.size == 1:
            return np.full(x.shape, float(self.y_thresholds[0]), dtype=float)
        return np.interp(
            x,
            self.x_thresholds,
            self.y_thresholds,
            left=float(self.y_thresholds[0]),
            right=float(self.y_thresholds[-1]),
        )


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicMap:
    """Fit non-decreasing weighted PAV with tied-x aggregation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return IsotonicMap(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    weights = np.bincount(inverse).astype(float)
    means = sums / np.maximum(weights, 1.0)

    block_x: list[float] = []
    block_y: list[float] = []
    block_w: list[float] = []
    for xi, yi, wi in zip(unique_x, means, weights):
        block_x.append(float(xi))
        block_y.append(float(yi))
        block_w.append(float(wi))
        while len(block_y) >= 2 and block_y[-2] > block_y[-1]:
            new_w = block_w[-2] + block_w[-1]
            new_y = (block_y[-2] * block_w[-2] + block_y[-1] * block_w[-1]) / new_w
            new_x = (block_x[-2] * block_w[-2] + block_x[-1] * block_w[-1]) / new_w
            block_x[-2:] = [new_x]
            block_y[-2:] = [new_y]
            block_w[-2:] = [new_w]

    return IsotonicMap(
        x_thresholds=np.asarray(block_x, dtype=float),
        y_thresholds=np.clip(np.asarray(block_y, dtype=float), 0.0, 1.0),
    )


@dataclass
class Calibrator:
    iso: IsotonicMap
    bucket_edges: np.ndarray
    bucket_width: np.ndarray
    bucket_center: np.ndarray
    width_max: float
    alpha: float
    score_col: str
    fwd_col: str

    def apply(self, preds: pd.DataFrame) -> pd.DataFrame:
        required = {"trade_date", self.score_col}
        missing = required - set(preds.columns)
        if missing:
            raise ValueError(f"calibration input missing columns: {sorted(missing)}")
        out = preds.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        rank = _xrank(out, self.score_col).fillna(0.5)
        rank_values = rank.to_numpy(dtype=float)
        out["calib_rank"] = rank_values
        out["p_beat"] = np.clip(self.iso.predict(rank_values), 0.0, 1.0)
        bucket = np.clip(
            np.digitize(rank_values, self.bucket_edges[1:-1]),
            0,
            len(self.bucket_width) - 1,
        )
        out["expected_forward_return"] = self.bucket_center[bucket]
        out["conformal_width"] = self.bucket_width[bucket]
        out["conformal_lower"] = out["expected_forward_return"] - out["conformal_width"]
        out["conformal_upper"] = out["expected_forward_return"] + out["conformal_width"]
        out["uncertainty"] = np.clip(
            out["conformal_width"] / max(self.width_max, 1e-12), 0.0, 1.0
        )
        return out


def fit_calibrator(
    df: pd.DataFrame,
    *,
    score_col: str = "alpha_score",
    fwd_col: str = "forward_return_5d",
    alpha: float = 0.10,
    n_buckets: int = 10,
    min_rows: int = 100,
) -> Calibrator:
    """Fit only on a calibration window strictly preceding application data."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    required = {"trade_date", score_col, fwd_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")

    d = df.dropna(subset=[score_col, fwd_col]).copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d = d.dropna(subset=["trade_date"])
    if len(d) < min_rows:
        raise ValueError(f"insufficient calibration rows: {len(d)} < {min_rows}")
    d["rank"] = _xrank(d, score_col)
    fwd = pd.to_numeric(d[fwd_col], errors="coerce")
    day_mean = d.assign(_fwd=fwd).groupby("trade_date")["_fwd"].transform("mean")
    beat = (fwd > day_mean).astype(float)
    iso = _fit_isotonic(d["rank"].to_numpy(dtype=float), beat.to_numpy(dtype=float))

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket = np.clip(
        np.digitize(d["rank"].to_numpy(dtype=float), edges[1:-1]),
        0,
        n_buckets - 1,
    )
    d["_bucket"] = bucket
    d["_fwd"] = fwd
    global_center = float(fwd.mean())
    centers = d.groupby("_bucket")["_fwd"].mean()
    bucket_center = np.asarray(
        [float(centers.get(i, global_center)) for i in range(n_buckets)], dtype=float
    )
    residual = np.abs(fwd.to_numpy(dtype=float) - bucket_center[bucket])
    global_width = float(np.nanquantile(residual, 1.0 - alpha))
    widths: list[float] = []
    for i in range(n_buckets):
        values = residual[bucket == i]
        widths.append(
            float(np.nanquantile(values, 1.0 - alpha))
            if values.size >= 5
            else global_width
        )
    bucket_width = np.asarray(widths, dtype=float)
    bucket_width = np.where(np.isfinite(bucket_width), bucket_width, global_width)
    return Calibrator(
        iso=iso,
        bucket_edges=edges,
        bucket_width=bucket_width,
        bucket_center=bucket_center,
        width_max=float(np.nanmax(bucket_width)) if bucket_width.size else 1.0,
        alpha=alpha,
        score_col=score_col,
        fwd_col=fwd_col,
    )
