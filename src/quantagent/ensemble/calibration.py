"""Score calibration + conformal uncertainty for model predictions (governance ②).

Raw ``alpha_score`` is an uncalibrated cross-sectional rank signal — a threshold like
``confidence_floor=0.55`` on it has no probabilistic meaning. This maps the score to:
  * ``calib_rank``     — per-day cross-sectional rank in [0,1] (stationary across days)
  * ``p_beat``         — isotonic-calibrated P(stock beats the cross-sectional mean over 5d)
  * ``conformal_width``— locally-adaptive split-conformal interval width (return units);
                          higher = more uncertain. Feeds ``RiskGate(conformal_width=...)``
                          which rejects names above ``conformal_uncertainty_threshold``.
  * ``uncertainty``    — width normalized to [0,1].

Split-conformal: fit a per-rank-bucket return center on a calibration window, take the
(1−α) quantile of |residual| PER BUCKET so ambiguous mid-rank picks get a wider (more
uncertain) interval than the high-conviction tails. Forward-looking-safe: the calibrator
is fit on a window strictly before where it is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _xrank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(pct=True)


@dataclass
class Calibrator:
    iso: IsotonicRegression
    bucket_edges: np.ndarray          # rank-bucket edges
    bucket_width: np.ndarray          # conformal half-width per bucket (return units)
    width_max: float                  # for [0,1] normalization
    alpha: float
    score_col: str

    def apply(self, preds: pd.DataFrame) -> pd.DataFrame:
        out = preds.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        rank = _xrank(out, self.score_col).fillna(0.5)
        out["calib_rank"] = rank.to_numpy()
        out["p_beat"] = np.clip(self.iso.predict(rank.to_numpy()), 0.0, 1.0)
        bi = np.clip(np.digitize(rank.to_numpy(), self.bucket_edges[1:-1]), 0, len(self.bucket_width) - 1)
        out["conformal_width"] = self.bucket_width[bi]
        out["uncertainty"] = np.clip(out["conformal_width"] / (self.width_max or 1.0), 0.0, 1.0)
        return out


def fit_calibrator(df: pd.DataFrame, *, score_col: str = "alpha_score",
                   fwd_col: str = "forward_return_5d", alpha: float = 0.10,
                   n_buckets: int = 10) -> Calibrator:
    """Fit on a calibration window (df with trade_date, score_col, fwd_col)."""
    d = df.dropna(subset=[score_col, fwd_col]).copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d["rank"] = _xrank(d, score_col)
    # label: beats the day's cross-sectional mean return (P(beat universe))
    d["beat"] = (pd.to_numeric(d[fwd_col], errors="coerce")
                 > d.groupby("trade_date")[fwd_col].transform("mean")).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(d["rank"].to_numpy(), d["beat"].to_numpy())
    # locally-adaptive conformal width per rank bucket
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bi = np.clip(np.digitize(d["rank"].to_numpy(), edges[1:-1]), 0, n_buckets - 1)
    d["_bi"] = bi
    fwd = pd.to_numeric(d[fwd_col], errors="coerce")
    center = d.groupby("_bi")[fwd_col].transform("mean")
    resid = (fwd - center).abs()
    bw = resid.groupby(d["_bi"]).quantile(1.0 - alpha)
    bucket_width = np.array([float(bw.get(i, resid.quantile(1.0 - alpha))) for i in range(n_buckets)])
    return Calibrator(iso=iso, bucket_edges=edges, bucket_width=bucket_width,
                      width_max=float(np.nanmax(bucket_width)) if bucket_width.size else 1.0,
                      alpha=alpha, score_col=score_col)
