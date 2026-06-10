"""Meta-labeling (governance ②) — a 2nd-stage model deciding whether a PRIMARY signal
should actually be executed / how much to size it (López de Prado style).

Primary model decides direction (factor rank / 做T 低吸 entry). The meta-labeler predicts
P(this specific signal succeeds) from context features, so we can:
  * filter: skip low-P(success) signals (高抛后还更高 = 高抛失败; 低吸后继续跌 = 低吸失败)
  * size:   scale position by P(success)

Use cases:
  * factor pick  : y = pick beat the universe over the holding window
  * 做T entry     : y = the FSM round-trip hit 止盈 (not 止损/尾盘亏损)

Generic wrapper over a sklearn classifier; works on any (features, success) table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class MetaLabeler:
    model: LogisticRegression
    features: list[str]
    mean: pd.Series
    std: pd.Series

    def predict_success(self, X: pd.DataFrame) -> np.ndarray:
        z = ((X[self.features] - self.mean) / self.std).fillna(0.0).to_numpy()
        return self.model.predict_proba(z)[:, 1]


def fit_meta_labeler(df: pd.DataFrame, features: list[str], label_col: str = "success",
                     *, C: float = 1.0) -> MetaLabeler:
    """df: rows = primary signals, columns = features + binary ``label_col`` (1=succeeded)."""
    d = df.dropna(subset=[label_col]).copy()
    X = d[features].astype(float)
    mean, std = X.mean(), X.std(ddof=0).replace(0, 1.0)
    z = ((X - mean) / std).fillna(0.0)
    y = d[label_col].astype(int).to_numpy()
    model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    model.fit(z.to_numpy(), y)
    return MetaLabeler(model=model, features=list(features), mean=mean, std=std)


def build_dot_meta_dataset(fsm_results: pd.DataFrame) -> pd.DataFrame:
    """From 做T FSM outcomes (one row per entered signal) build the meta-label set.

    ``fsm_results`` needs at least: open_auction_gap, intraday_range_pos (at entry),
    net_buy_pressure, vwap_deviation, and ``ret`` / ``exit_reason``. success = 止盈."""
    d = fsm_results.copy()
    d["success"] = (d.get("exit_reason", "") == "止盈").astype(int)
    return d


def meta_filter(p_success: np.ndarray, *, floor: float = 0.5) -> np.ndarray:
    """Execution mask + size multiplier from P(success): skip below floor, size ∝ P above it."""
    take = p_success >= floor
    size = np.where(take, np.clip((p_success - floor) / (1.0 - floor), 0.0, 1.0), 0.0)
    return size
