"""Stage 3 — three setup-conditional sub-models with regime ensemble.

This module provides:

* :func:`build_regime_sub_labels` — augments a panel that already has
  ``forward_return_{H}d`` columns with three new label columns per
  horizon: ``lowbuy_label_{H}d``, ``breakout_label_{H}d``,
  ``limitup_risk_label_{H}d``.  Each label is the forward return on
  setup days and ``NaN`` everywhere else, so the existing FT-Transformer
  trainer learns *setup-conditional* alpha without changes (dropna on
  labels filters non-setup rows out of the training loss).

* :func:`ensemble_sub_model_predictions` — at inference, combine the
  three sub-models' predictions into a single per-(date, symbol) score.
  Routes by regime: bear ↑LowBuy, normal/bull ↑Breakout; the
  LimitUpRisk score always enters with a *negative* weight (predicts
  downside, used to exclude overheated names).

The setup heuristics are deliberately simple and stateless — every
trigger is a function of OHLCV history available *strictly before* the
label horizon, so the labels remain PIT-safe.

Setup triggers (any one fires → that day enters the sub-model's training
data):

* **LowBuy**: 20-day cumulative return ≤ −10% AND 5-day cumulative
  return ≥ −2% (the bleed has stopped but the drawdown hasn't healed).
* **Breakout**: close ≥ 60-day rolling high AND mean 5-day volume ≥
  1.5 × mean 60-day volume (price and volume both confirm).
* **LimitUpRisk**: at least one limit-up in the last 3 sessions AND
  20-day cumulative return ≥ +25% (parabolic + recent ceiling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubModelLabelConfig:
    """Tunable thresholds for the three setup detectors."""

    # LowBuy
    lowbuy_drawdown_window: int = 20
    lowbuy_drawdown_threshold: float = -0.10
    lowbuy_bleed_stopped_window: int = 5
    lowbuy_bleed_stopped_threshold: float = -0.02

    # Breakout
    breakout_high_window: int = 60
    breakout_volume_window_short: int = 5
    breakout_volume_window_long: int = 60
    breakout_volume_multiplier: float = 1.5

    # LimitUpRisk
    limit_up_threshold: float = 0.095
    limit_up_lookback: int = 3
    limitup_high_run_window: int = 20
    limitup_high_run_threshold: float = 0.25

    # Output
    horizons: tuple[int, ...] = (1, 5, 20)


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

def _per_symbol_apply(
    panel: pd.DataFrame,
    fn,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Sort by date within symbol, apply ``fn`` to each group, concat back.

    Returns a frame indexed identically to the input.
    """
    out = panel.sort_values([symbol_col, date_col]).copy()
    # Note: pandas warns about apply on groupby including grouping cols. We
    # explicitly need them in `fn`, so suppress the deprecation locally.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=FutureWarning)
        out = out.groupby(symbol_col, group_keys=False).apply(fn)
    return out.reset_index(drop=True)


def _lowbuy_setup(group: pd.DataFrame, config: SubModelLabelConfig) -> pd.DataFrame:
    g = group.copy()
    close = g["close"].astype(float)
    g["_ret_20"] = close.pct_change(config.lowbuy_drawdown_window)
    g["_ret_5"] = close.pct_change(config.lowbuy_bleed_stopped_window)
    g["lowbuy_setup"] = (
        (g["_ret_20"] <= config.lowbuy_drawdown_threshold)
        & (g["_ret_5"] >= config.lowbuy_bleed_stopped_threshold)
    ).fillna(False)
    return g.drop(columns=["_ret_20", "_ret_5"])


def _breakout_setup(group: pd.DataFrame, config: SubModelLabelConfig) -> pd.DataFrame:
    g = group.copy()
    close = g["close"].astype(float)
    vol = g["volume"].astype(float) if "volume" in g.columns else pd.Series(
        np.nan, index=g.index, dtype=float
    )
    rolling_high = close.shift(1).rolling(
        config.breakout_high_window, min_periods=max(10, config.breakout_high_window // 4)
    ).max()
    vol_short = vol.shift(1).rolling(
        config.breakout_volume_window_short,
        min_periods=max(2, config.breakout_volume_window_short // 2),
    ).mean()
    vol_long = vol.shift(1).rolling(
        config.breakout_volume_window_long,
        min_periods=max(5, config.breakout_volume_window_long // 4),
    ).mean()
    has_volume = vol.notna().any()
    price_ok = (close >= rolling_high).fillna(False)
    if has_volume:
        vol_ok = (vol_short >= vol_long * config.breakout_volume_multiplier).fillna(False)
    else:
        # Without volume data, gate purely on price breakout — still valid.
        vol_ok = pd.Series(True, index=g.index)
    g["breakout_setup"] = (price_ok & vol_ok).fillna(False)
    return g


def _limitup_risk_setup(group: pd.DataFrame, config: SubModelLabelConfig) -> pd.DataFrame:
    g = group.copy()
    close = g["close"].astype(float)
    prev = close.shift(1)
    daily_ret = (close / prev) - 1.0
    is_limit_up = daily_ret >= config.limit_up_threshold
    # ≥1 limit-up within the last `limit_up_lookback` sessions
    recent_lu = (
        is_limit_up.shift(1)
        .rolling(config.limit_up_lookback, min_periods=1)
        .sum()
        >= 1
    )
    high_run = close.pct_change(config.limitup_high_run_window)
    high_run_ok = (high_run >= config.limitup_high_run_threshold).fillna(False)
    g["limitup_risk_setup"] = (recent_lu.fillna(False) & high_run_ok).fillna(False)
    return g


# ---------------------------------------------------------------------------
# Label augmentation
# ---------------------------------------------------------------------------

def build_regime_sub_labels(
    panel: pd.DataFrame,
    config: SubModelLabelConfig | None = None,
) -> pd.DataFrame:
    """Augment ``panel`` with three setup-conditional label columns per horizon.

    Parameters
    ----------
    panel : DataFrame with columns ``symbol``, ``trade_date``, ``close``
        (required), ``volume`` (optional — breakout falls back to
        price-only when missing), and one ``forward_return_{H}d`` column
        per horizon listed in ``config.horizons``.
    config : Tunable thresholds (see :class:`SubModelLabelConfig`).

    Returns
    -------
    DataFrame
        The input panel with three new columns per horizon:
        ``lowbuy_label_{H}d``, ``breakout_label_{H}d``,
        ``limitup_risk_label_{H}d``.  Setup-day rows carry the
        (possibly sign-flipped) forward return; non-setup rows are
        ``NaN`` so the trainer's label dropna naturally restricts the
        loss to setup days.
    """
    cfg = config or SubModelLabelConfig()
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()

    missing_cols = {"symbol", "trade_date", "close"} - set(panel.columns)
    if missing_cols:
        raise ValueError(f"panel is missing required columns: {sorted(missing_cols)}")

    # Detect setups per-symbol (each setup depends only on that symbol's history)
    out = _per_symbol_apply(panel, lambda g: _lowbuy_setup(g, cfg))
    out = _per_symbol_apply(out, lambda g: _breakout_setup(g, cfg))
    out = _per_symbol_apply(out, lambda g: _limitup_risk_setup(g, cfg))

    for horizon in cfg.horizons:
        ret_col = f"forward_return_{horizon}d"
        if ret_col not in out.columns:
            # Skip silently — the panel may not have this horizon; the trainer
            # will warn elsewhere if a required horizon is missing.
            continue
        ret = out[ret_col].astype(float)
        out[f"lowbuy_label_{horizon}d"] = np.where(out["lowbuy_setup"], ret, np.nan)
        out[f"breakout_label_{horizon}d"] = np.where(out["breakout_setup"], ret, np.nan)
        out[f"limitup_risk_label_{horizon}d"] = np.where(
            out["limitup_risk_setup"], -ret, np.nan
        )

    return out


# ---------------------------------------------------------------------------
# Inference-time ensemble
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleWeights:
    """Per-regime sub-model weights (sum to 1 for the long side).

    The LimitUpRisk weight is always *negative* — it's predicting
    downside risk and used to subtract from the composite score, not
    add to it. The long-side weights for LowBuy + Breakout must sum
    to 1 within each regime.
    """

    # regime → (lowbuy, breakout, limitup_risk)
    weights: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "normal": (0.35, 0.55, -0.10),
            "caution": (0.50, 0.40, -0.10),
            "bear": (0.65, 0.20, -0.15),
            "crisis": (0.80, 0.10, -0.10),
        }
    )
    default: tuple[float, float, float] = (0.40, 0.50, -0.10)

    def for_regime(self, regime: str | None) -> tuple[float, float, float]:
        if regime is None:
            return self.default
        return self.weights.get(str(regime), self.default)


def ensemble_sub_model_predictions(
    lowbuy: pd.DataFrame,
    breakout: pd.DataFrame,
    limitup_risk: pd.DataFrame,
    *,
    regime_states: pd.Series | pd.DataFrame | None = None,
    weights: EnsembleWeights | None = None,
    prediction_column: str = "alpha_score",
) -> pd.DataFrame:
    """Fuse three sub-model prediction frames into a single composite.

    Each input frame must have columns ``trade_date``, ``symbol`` and a
    score column (default ``alpha_score``).  Missing rows in any
    sub-model are treated as 0 contribution from that sub-model on
    that (date, symbol) — i.e. the sub-model abstains.

    Parameters
    ----------
    regime_states : optional per-date regime label
        Either a pd.Series indexed by trade_date with values like
        "normal"/"caution"/"bear"/"crisis", or a DataFrame with columns
        ``trade_date``, ``regime_state``.  When omitted, the default
        weights are used for every row.

    Returns
    -------
    DataFrame with columns ``trade_date``, ``symbol``,
    ``composite_score``, plus per-sub-model contributions for audit:
    ``lowbuy_score``, ``breakout_score``, ``limitup_risk_score``,
    ``regime_state``, ``lowbuy_weight``, ``breakout_weight``,
    ``limitup_risk_weight``.
    """
    w = weights or EnsembleWeights()

    def _prep(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date", "symbol", label])
        f = frame[["trade_date", "symbol", prediction_column]].copy()
        f["trade_date"] = pd.to_datetime(f["trade_date"], errors="coerce")
        f["symbol"] = f["symbol"].astype(str)
        f = f.rename(columns={prediction_column: label})
        return f

    lb = _prep(lowbuy, "lowbuy_score")
    bo = _prep(breakout, "breakout_score")
    lr = _prep(limitup_risk, "limitup_risk_score")

    # Outer-join the three frames on (date, symbol). Missing scores → 0.
    joined = lb.merge(bo, on=["trade_date", "symbol"], how="outer").merge(
        lr, on=["trade_date", "symbol"], how="outer"
    )
    for col in ("lowbuy_score", "breakout_score", "limitup_risk_score"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)

    # Attach regime labels
    if regime_states is None:
        joined["regime_state"] = None
    else:
        if isinstance(regime_states, pd.DataFrame):
            rs = regime_states[["trade_date", "regime_state"]].copy()
        else:
            rs = regime_states.rename_axis("trade_date").reset_index(name="regime_state")
        rs["trade_date"] = pd.to_datetime(rs["trade_date"], errors="coerce")
        rs["regime_state"] = rs["regime_state"].astype(str)
        joined = joined.merge(rs, on="trade_date", how="left")

    weight_rows = joined["regime_state"].map(lambda r: w.for_regime(r))
    weight_arr = np.array(weight_rows.tolist(), dtype=float)
    joined["lowbuy_weight"] = weight_arr[:, 0]
    joined["breakout_weight"] = weight_arr[:, 1]
    joined["limitup_risk_weight"] = weight_arr[:, 2]

    joined["composite_score"] = (
        joined["lowbuy_score"] * joined["lowbuy_weight"]
        + joined["breakout_score"] * joined["breakout_weight"]
        + joined["limitup_risk_score"] * joined["limitup_risk_weight"]
    )

    return joined[
        [
            "trade_date",
            "symbol",
            "composite_score",
            "lowbuy_score",
            "breakout_score",
            "limitup_risk_score",
            "regime_state",
            "lowbuy_weight",
            "breakout_weight",
            "limitup_risk_weight",
        ]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Setup statistics (for audit reporting)
# ---------------------------------------------------------------------------

def sub_model_setup_stats(panel_with_setups: pd.DataFrame) -> dict[str, float]:
    """Coverage statistics of the three setup detectors.

    Used by the daily health report so ops can spot a sub-model whose
    setup is firing too rarely (no training signal) or too often
    (no selectivity).  Returns coverage as fractions of total rows.
    """
    if panel_with_setups is None or panel_with_setups.empty:
        return {
            "n_rows": 0,
            "lowbuy_setup_rate": 0.0,
            "breakout_setup_rate": 0.0,
            "limitup_risk_setup_rate": 0.0,
        }
    n = int(len(panel_with_setups))
    stats: dict[str, float] = {"n_rows": n}
    for col in ("lowbuy_setup", "breakout_setup", "limitup_risk_setup"):
        if col in panel_with_setups.columns:
            stats[f"{col}_rate"] = float(panel_with_setups[col].sum() / max(n, 1))
        else:
            stats[f"{col}_rate"] = 0.0
    return stats
