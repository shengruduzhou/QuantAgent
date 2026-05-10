from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ASharePriceLimit:
    main_board: float = 0.10
    chinext: float = 0.20
    star_board: float = 0.20
    bse: float = 0.30
    st: float = 0.05


@dataclass(frozen=True)
class BoardRule:
    board: str
    price_limit_ratio: float
    lot_size: int = 100
    minimum_buy_quantity: int = 100
    sell_odd_lot_policy: str = "allow_odd_lot_full_sell"
    t_plus_one: bool = True
    auction_price_band_lower: float | None = None
    auction_price_band_upper: float | None = None
    effective_date: date | None = None
    rule_source: str = "default_v4_config"
    version: str = "v4.0"
    instrument_type: str = "equity"


@dataclass(frozen=True)
class AshareRuleEngineConfig:
    st_price_limit_ratio: float = 0.05
    star_minimum_buy_quantity: int = 200
    no_buy_limit_up: bool = True
    no_sell_limit_down: bool = True
    board_rules: dict[str, BoardRule] = field(default_factory=dict)


class AshareRuleEngine:
    """Board-aware A-share rule adapter used by portfolio, backtest, and execution."""

    def __init__(self, config: AshareRuleEngineConfig | None = None) -> None:
        self.config = config or AshareRuleEngineConfig()
        self._rules = _default_board_rules(self.config) | dict(self.config.board_rules)

    def infer_board(self, symbol: str) -> str:
        text = str(symbol).upper()
        code = text.split(".")[0]
        if text.startswith(("IF", "IH", "IC", "IM")):
            return "futures_hedge"
        if code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "588", "159")):
            return "etf"
        if code.startswith(("110", "113", "118", "123", "127", "128")):
            return "convertible_bond"
        if text.endswith(".BJ") or code.startswith(("8", "4")):
            return "bse"
        if code.startswith(("688", "689")):
            return "star"
        if code.startswith(("300", "301")):
            return "chinext"
        return "main_board"

    def get_rule(self, symbol: str, trade_date: Any | None = None) -> BoardRule:
        del trade_date
        return self._rules[self.infer_board(symbol)]

    def price_limit_rule(
        self,
        symbol: str,
        prev_close: float,
        trade_date: Any | None = None,
        is_st: bool = False,
    ) -> dict[str, float | str]:
        rule = self.get_rule(symbol, trade_date)
        ratio = self.config.st_price_limit_ratio if is_st and rule.instrument_type == "equity" else rule.price_limit_ratio
        prev = float(prev_close)
        if prev <= 0 or not np.isfinite(prev):
            return {"board": rule.board, "ratio": ratio, "limit_up": np.nan, "limit_down": np.nan}
        return {
            "board": rule.board,
            "ratio": float(ratio),
            "limit_up": float(round(prev * (1.0 + ratio), 2)),
            "limit_down": float(round(prev * (1.0 - ratio), 2)),
        }

    def board_lot_rule(self, symbol: str, side: str, trade_date: Any | None = None) -> dict[str, int | str | bool]:
        rule = self.get_rule(symbol, trade_date)
        return {
            "board": rule.board,
            "side": _side_value(side),
            "lot_size": rule.lot_size,
            "minimum_buy_quantity": rule.minimum_buy_quantity,
            "sell_odd_lot_policy": rule.sell_odd_lot_policy,
            "t_plus_one": rule.t_plus_one,
        }

    def round_order_quantity(
        self,
        symbol: str,
        side: str,
        quantity: float,
        trade_date: Any | None = None,
    ) -> int:
        rule = self.get_rule(symbol, trade_date)
        qty = int(max(0, np.floor(float(quantity))))
        if qty <= 0:
            return 0
        side_value = _side_value(side)
        if side_value == "buy":
            minimum = max(rule.minimum_buy_quantity, rule.lot_size)
            if qty < minimum:
                return 0
            return int(np.floor(qty / rule.lot_size) * rule.lot_size)
        if rule.sell_odd_lot_policy == "allow_odd_lot_full_sell" and qty < rule.lot_size:
            return qty
        return int(np.floor(qty / rule.lot_size) * rule.lot_size)

    def is_tradable(self, row_or_state: Any) -> bool:
        state = _as_mapping(row_or_state)
        if bool(state.get("is_suspended", state.get("suspended", False))):
            return False
        if bool(state.get("is_delisted", state.get("delisted", False))):
            return False
        if "volume" in state and pd.notna(state["volume"]) and float(state["volume"]) <= 0:
            return False
        if "tradable" in state:
            return bool(state["tradable"])
        return True

    def filter_tradable(self, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return panel.copy()
        mask = panel.apply(self.is_tradable, axis=1)
        return panel.loc[mask].sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def validate_order_intent(self, intent: Any, state: Any) -> tuple[bool, str]:
        order = _as_mapping(intent)
        market = _as_mapping(state)
        symbol = str(order.get("symbol", market.get("symbol", "")))
        side = _side_value(order.get("side", ""))
        quantity = float(order.get("quantity", 0))
        if not symbol:
            return False, "missing_symbol"
        if side not in {"buy", "sell"}:
            return False, "invalid_side"
        if not self.is_tradable(market):
            return False, "not_tradable"
        rounded = self.round_order_quantity(symbol, side, quantity, market.get("trade_date"))
        if rounded <= 0:
            return False, "invalid_lot_quantity"
        if side == "buy" and self.config.no_buy_limit_up and bool(market.get("is_limit_up", market.get("limit_up", False))):
            return False, "limit_up_no_buy"
        if side == "sell" and self.config.no_sell_limit_down and bool(market.get("is_limit_down", market.get("limit_down", False))):
            return False, "limit_down_no_sell"
        rule = self.get_rule(symbol, market.get("trade_date"))
        if side == "sell" and rule.t_plus_one:
            available = int(market.get("available_shares", market.get("sellable_shares", rounded)))
            if rounded > available:
                return False, "t_plus_one_insufficient_available_shares"
        return True, "ok"


def board_for_symbol(symbol: str, is_st: bool = False) -> str:
    """Resolve A-share board label from ticker prefix."""
    if is_st:
        return "st"
    board = AshareRuleEngine().infer_board(symbol)
    if board == "star":
        return "star_board"
    if board == "chinext":
        return "chinext"
    if board == "bse":
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


def _default_board_rules(config: AshareRuleEngineConfig) -> dict[str, BoardRule]:
    return {
        "main_board": BoardRule("main_board", 0.10),
        "chinext": BoardRule("chinext", 0.20),
        "star": BoardRule(
            "star",
            0.20,
            minimum_buy_quantity=config.star_minimum_buy_quantity,
        ),
        "bse": BoardRule("bse", 0.30),
        "etf": BoardRule("etf", 0.10, t_plus_one=False, instrument_type="etf"),
        "convertible_bond": BoardRule(
            "convertible_bond",
            0.20,
            lot_size=10,
            minimum_buy_quantity=10,
            t_plus_one=False,
            instrument_type="convertible_bond",
        ),
        "futures_hedge": BoardRule(
            "futures_hedge",
            0.10,
            lot_size=1,
            minimum_buy_quantity=1,
            t_plus_one=False,
            instrument_type="futures",
        ),
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _side_value(side: Any) -> str:
    value = getattr(side, "value", side)
    return str(value).lower()
