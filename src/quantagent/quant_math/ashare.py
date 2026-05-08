from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ASharePriceLimit:
    main_board: float = 0.10
    chinext: float = 0.20
    star_board: float = 0.20
    bse: float = 0.30
    st: float = 0.05


def board_for_symbol(symbol: str, is_st: bool = False) -> str:
    """Resolve A-share board label from ticker prefix."""
    if is_st:
        return "st"
    head = symbol[:3]
    if head in {"688"}:
        return "star_board"
    if head in {"300", "301"}:
        return "chinext"
    if head.startswith("8") or head.startswith("4"):
        return "bse"
    return "main_board"


def daily_price_limit(symbol: str, is_st: bool = False, limits: ASharePriceLimit | None = None) -> float:
    limits = limits or ASharePriceLimit()
    board = board_for_symbol(symbol, is_st)
    return getattr(limits, board)


def limit_up_mask(
    frame: pd.DataFrame,
    symbol_column: str = "symbol",
    is_st_column: str | None = "is_st",
    tolerance: float = 1e-3,
    limits: ASharePriceLimit | None = None,
) -> pd.Series:
    """True when close >= prev_close * (1 + board_limit) within tolerance bps."""
    limits = limits or ASharePriceLimit()
    prev_close = frame.groupby(symbol_column)["close"].shift(1)
    is_st = frame[is_st_column] if is_st_column and is_st_column in frame.columns else False
    pct_limit = frame.apply(
        lambda r: daily_price_limit(r[symbol_column], bool(is_st.loc[r.name]) if isinstance(is_st, pd.Series) else False, limits),
        axis=1,
    )
    target = prev_close * (1.0 + pct_limit)
    return (frame["close"] >= target * (1.0 - tolerance)) & (prev_close > 0)


def limit_down_mask(
    frame: pd.DataFrame,
    symbol_column: str = "symbol",
    is_st_column: str | None = "is_st",
    tolerance: float = 1e-3,
    limits: ASharePriceLimit | None = None,
) -> pd.Series:
    limits = limits or ASharePriceLimit()
    prev_close = frame.groupby(symbol_column)["close"].shift(1)
    is_st = frame[is_st_column] if is_st_column and is_st_column in frame.columns else False
    pct_limit = frame.apply(
        lambda r: daily_price_limit(r[symbol_column], bool(is_st.loc[r.name]) if isinstance(is_st, pd.Series) else False, limits),
        axis=1,
    )
    target = prev_close * (1.0 - pct_limit)
    return (frame["close"] <= target * (1.0 + tolerance)) & (prev_close > 0)


def suspension_mask(frame: pd.DataFrame, symbol_column: str = "symbol") -> pd.Series:
    """Volume == 0 AND close == prev_close marks a trading halt."""
    prev_close = frame.groupby(symbol_column)["close"].shift(1)
    return (frame["volume"].fillna(0) == 0) & (frame["close"] == prev_close)


def tradable_universe(
    frame: pd.DataFrame,
    nav: float | None = None,
    min_amount_20d_rmb: float = 5e7,
    exclude_st: bool = True,
    exclude_suspended: bool = True,
    exclude_listed_days_lt: int = 250,
    symbol_column: str = "symbol",
) -> pd.Series:
    """Per-row boolean mask. True when the row is eligible to be bought today."""
    mask = pd.Series(True, index=frame.index)
    if "amount_mean_20d" in frame.columns:
        mask &= frame["amount_mean_20d"].fillna(0.0) >= min_amount_20d_rmb
    if exclude_st and "is_st" in frame.columns:
        mask &= ~frame["is_st"].fillna(False)
    if exclude_suspended:
        mask &= ~suspension_mask(frame, symbol_column)
    if exclude_listed_days_lt and "listed_days" in frame.columns:
        mask &= frame["listed_days"].fillna(0) >= exclude_listed_days_lt
    mask &= ~limit_up_mask(frame, symbol_column)
    return mask


@dataclass
class TPlusOnePosition:
    """Track shares bought today (frozen) vs available for sale."""

    available_shares: int = 0
    frozen_today_shares: int = 0

    def settle_overnight(self) -> None:
        self.available_shares += self.frozen_today_shares
        self.frozen_today_shares = 0

    def buy(self, shares: int) -> None:
        if shares <= 0:
            return
        self.frozen_today_shares += shares

    def sell(self, shares: int) -> int:
        sellable = min(self.available_shares, max(shares, 0))
        self.available_shares -= sellable
        return sellable

    def total_shares(self) -> int:
        return self.available_shares + self.frozen_today_shares


def enforce_tradability(
    target_weights: pd.Series,
    current_weights: pd.Series,
    can_buy: pd.Series,
    can_sell: pd.Series,
) -> pd.Series:
    """Cap target_weights so any position increase respects can_buy and decrease respects can_sell."""
    aligned_current = current_weights.reindex(target_weights.index).fillna(0.0)
    can_buy = can_buy.reindex(target_weights.index).fillna(False)
    can_sell = can_sell.reindex(target_weights.index).fillna(False)
    delta = target_weights - aligned_current
    capped = target_weights.copy()
    capped[(delta > 0) & ~can_buy] = aligned_current[(delta > 0) & ~can_buy]
    capped[(delta < 0) & ~can_sell] = aligned_current[(delta < 0) & ~can_sell]
    return capped
