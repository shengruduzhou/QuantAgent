"""Minimal ATR-based technical timing agent.

The full ``TechnicalTimingPlan`` producer in
``services/v7_pipeline_service.py`` returns ``entry_zone=None`` for every
name — a stub for an unimplemented dependency. The portfolio layer needs
something concrete so that ``portfolio.timing_gate`` can do its job, and
nothing in the data we already have prevents us from running a
rule-based timing producer entirely on the market panel.

This module computes:

* ``atr_14`` — Wilder's average true range over 14 trading days.
* ``entry_zone_low / entry_zone_high`` — the close ± a fraction of ATR;
  prices inside the zone are eligible for opening new positions.
* ``invalidation_level`` — close − ``invalidation_atr`` × ATR. If the
  next day's low pierces this level we treat the thesis as invalidated
  and the gate forces a close.

The output is a frame keyed on ``(trade_date, symbol)`` that can be
merged into the optimiser's eligibility table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TechnicalTimingConfig:
    atr_window: int = 14
    entry_zone_atr_low: float = 0.5  # close - 0.5 ATR ≤ price ≤ close + ATR
    entry_zone_atr_high: float = 1.0
    invalidation_atr: float = 2.0


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / max(1, window), adjust=False, min_periods=window).mean()


def compute_technical_timing(
    market_panel: pd.DataFrame,
    config: TechnicalTimingConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TechnicalTimingConfig()
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "symbol",
                "atr",
                "entry_zone_low",
                "entry_zone_high",
                "invalidation_level",
            ]
        )

    needed = {"trade_date", "symbol", "close", "high", "low"}
    missing = needed - set(market_panel.columns)
    if missing:
        raise ValueError(f"market_panel is missing columns for timing: {sorted(missing)}")

    frame = market_panel[list(needed)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])

    pieces: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        atr = _wilder_atr(group["high"], group["low"], group["close"], cfg.atr_window)
        zone_low = group["close"] - cfg.entry_zone_atr_low * atr
        zone_high = group["close"] + cfg.entry_zone_atr_high * atr
        invalidation = group["close"] - cfg.invalidation_atr * atr
        pieces.append(
            pd.DataFrame(
                {
                    "trade_date": group["trade_date"].to_numpy(),
                    "symbol": group["symbol"].to_numpy(),
                    "atr": atr.to_numpy(),
                    "entry_zone_low": zone_low.to_numpy(),
                    "entry_zone_high": zone_high.to_numpy(),
                    "invalidation_level": invalidation.to_numpy(),
                }
            )
        )
    if not pieces:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "symbol",
                "atr",
                "entry_zone_low",
                "entry_zone_high",
                "invalidation_level",
            ]
        )
    out = pd.concat(pieces, ignore_index=True)
    return out.dropna(subset=["atr"]).reset_index(drop=True)


__all__ = ["TechnicalTimingConfig", "compute_technical_timing"]
