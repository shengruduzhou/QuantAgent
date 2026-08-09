"""Mainland A-share trading rules, encoded once with dated authorities.

This module is intentionally dependency-light: data, quant-math, backtest,
paper and execution may all import it without creating an architectural cycle.
Every constant here is a market/venue rule rather than a strategy parameter, so
changing rules key off ``trade_date`` rather than the wall clock.

Primary exchange sources for the 2026 risk-warning change:

* SSE trading-rule revision, published 2026-04-24 and effective 2026-07-06:
  https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml
* SSE explanation: main-board risk-warning stocks change from 5% to 10%,
  effective 2026-07-06:
  https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml
* SZSE technical notice 2025-06-27: main-board ST/*ST ``LimitUpRate`` and
  ``LimitDownRate`` move from 0.050 to 0.100:
  https://www.szse.cn/marketServices/technicalservice/notice/t20250627_614645.html
* SZSE 2026 technical preparation notice carries the adjustment into the 2026
  revised trading rules, effective 2026-07-06:
  https://www.szse.cn/marketServices/technicalservice/notice/t20260424_620199.html

Quantity authorities used by the execution layer:

* SSE 2026 Trading Rules §6.7: STAR limit orders 200..100,000 shares, market
  orders 200..50,000 shares; a residual below 200 is sold in one order.
* SSE investor education clarifies that quantities above 200 may increase by
  one share for both buys and sells.
* BSE 2026 Trading Rules §§3.3.8-3.3.9: competitive stock orders are at least
  100 shares, residuals below 100 are sold once, and one order is at most
  1,000,000 shares. The absence of an integer-lot requirement above the minimum
  means whole-share increments are valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
SH_MAIN = "SH_Main"
SZ_MAIN = "SZ_Main"
CHINEXT = "ChiNext"
STAR = "STAR"
BSE = "BSE"

BOARDS: tuple[str, ...] = (SH_MAIN, SZ_MAIN, CHINEXT, STAR, BSE)

# ---------------------------------------------------------------------------
# Price limits
# ---------------------------------------------------------------------------
ORDINARY_LIMIT: dict[str, float] = {
    SH_MAIN: 0.10,
    SZ_MAIN: 0.10,
    CHINEXT: 0.20,
    STAR: 0.20,
    BSE: 0.30,
}

MAIN_BOARD_ST_LIMIT_REFORM = date(2026, 7, 6)
ST_LIMIT_LEGACY: dict[str, float] = {
    SH_MAIN: 0.05,
    SZ_MAIN: 0.05,
    CHINEXT: 0.20,
    STAR: 0.20,
    BSE: 0.30,
}
ST_LIMIT_CURRENT: dict[str, float] = {
    SH_MAIN: 0.10,
    SZ_MAIN: 0.10,
    CHINEXT: 0.20,
    STAR: 0.20,
    BSE: 0.30,
}
#: Compatibility alias for callers needing the *current* table only.
ST_LIMIT: dict[str, float] = ST_LIMIT_CURRENT

IPO_UNLIMITED_DAYS: dict[str, int] = {
    SH_MAIN: 5,
    SZ_MAIN: 5,
    CHINEXT: 5,
    STAR: 5,
    BSE: 1,
}
REGISTRATION_SYSTEM_MAIN_BOARD = date(2023, 4, 10)
LEGACY_IPO_UP = 0.44
LEGACY_IPO_DOWN = -0.36
IPO_HALT_THRESHOLDS: tuple[float, ...] = (0.30, 0.60)
IPO_HALT_MINUTES = 10

# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
STAMP_DUTY_CURRENT = 0.0005
STAMP_DUTY_LEGACY = 0.0010
STAMP_DUTY_HALVED_FROM = date(2023, 8, 28)
TRANSFER_FEE = 0.00001
DEFAULT_COMMISSION_RATE = 0.00025
COMMISSION_MINIMUM_CNY = 5.0

# ---------------------------------------------------------------------------
# Order quantities
# ---------------------------------------------------------------------------
#: (minimum normal order, whole-share increment above the minimum).
LOT_RULES: dict[str, tuple[int, int]] = {
    SH_MAIN: (100, 100),
    SZ_MAIN: (100, 100),
    CHINEXT: (100, 100),
    STAR: (200, 1),
    BSE: (100, 1),
}

#: Exchange-defined maximum quantity for boards with a rule required by the
#: current execution contract.  Missing entries are deliberately ``None`` at
#: the API boundary rather than guessed from another venue/board.
MAX_ORDER_QUANTITY: dict[str, dict[str, int]] = {
    STAR: {"LIMIT": 100_000, "MARKET": 50_000},
    BSE: {"LIMIT": 1_000_000, "MARKET": 1_000_000},
}


@dataclass(frozen=True)
class PriceLimits:
    limit_up: float | None
    limit_down: float | None
    ratio: float | None
    regime: str
    reference_close: float

    @property
    def unlimited(self) -> bool:
        return self.limit_up is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"unlimited": self.unlimited}


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _round_tick(value: float) -> float:
    return round(value + 1e-12, 2)


def exchange_board_for_symbol(symbol: str) -> str:
    """Resolve the underlying cash-equity board without importing data layers.

    The function is intentionally syntax-only. Instrument-type discrimination
    (ETF, convertible bond, futures) remains with higher-level rule adapters;
    callers should use this only after establishing that ``symbol`` is equity.
    """
    text = str(symbol).strip().upper()
    code = text.split(".", 1)[0]
    if text.endswith(".BJ") or code.startswith(("8", "4", "920")):
        return BSE
    if code.startswith(("688", "689")):
        return STAR
    if code.startswith(("300", "301", "302")):
        return CHINEXT
    if text.endswith(".SZ") or code.startswith(("000", "001", "002", "003")):
        return SZ_MAIN
    return SH_MAIN


def max_order_quantity(board: str, order_type: str = "LIMIT") -> int | None:
    """Return the exchange maximum for a known board/order type.

    ``None`` means this neutral module has no authoritative bound encoded for
    that combination; callers must not invent one.
    """
    return MAX_ORDER_QUANTITY.get(board, {}).get(str(order_type).upper())


def _st_limit_ratio(board: str, when: date | None) -> float | None:
    if board in (SH_MAIN, SZ_MAIN):
        if when is None:
            raise ValueError(
                "valid trade_date is required for main-board risk-warning price "
                "limits because the 5%->10% rule changed on 2026-07-06"
            )
        table = ST_LIMIT_CURRENT if when >= MAIN_BOARD_ST_LIMIT_REFORM else ST_LIMIT_LEGACY
        return table.get(board)
    return ST_LIMIT_CURRENT.get(board)


def price_limits(
    *,
    board: str,
    previous_close: float,
    trade_date: Any,
    listing_date: Any = None,
    sessions_since_listing: int | None = None,
    is_st: bool = False,
) -> PriceLimits:
    """Resolve the legal price band for one security-session.

    ``sessions_since_listing`` is deliberately not guessed from calendar days.
    Risk-warning limits are date-versioned; SH/SZ main-board ST/*ST use 5%
    before 2026-07-06 and 10% on/after that date.
    """
    when = _as_date(trade_date)
    listed = _as_date(listing_date)
    if sessions_since_listing is None and listed is not None and when is not None:
        sessions_since_listing = None

    unlimited_days = IPO_UNLIMITED_DAYS.get(board, 0)
    if sessions_since_listing is not None and sessions_since_listing < unlimited_days:
        main_board = board in (SH_MAIN, SZ_MAIN)
        pre_reform = when is not None and when < REGISTRATION_SYSTEM_MAIN_BOARD
        if main_board and pre_reform:
            return PriceLimits(
                limit_up=_round_tick(previous_close * (1 + LEGACY_IPO_UP)),
                limit_down=_round_tick(previous_close * (1 + LEGACY_IPO_DOWN)),
                ratio=LEGACY_IPO_UP,
                regime="IPO_LEGACY_APPROVAL_SYSTEM",
                reference_close=previous_close,
            )
        return PriceLimits(
            limit_up=None,
            limit_down=None,
            ratio=None,
            regime="IPO_NO_LIMIT_WINDOW",
            reference_close=previous_close,
        )

    ratio = _st_limit_ratio(board, when) if is_st else ORDINARY_LIMIT.get(board)
    if ratio is None:
        return PriceLimits(None, None, None, "UNKNOWN_BOARD", previous_close)
    return PriceLimits(
        limit_up=_round_tick(previous_close * (1 + ratio)),
        limit_down=_round_tick(previous_close * (1 - ratio)),
        ratio=ratio,
        regime="ST_BAND" if is_st else "ORDINARY",
        reference_close=previous_close,
    )


def stamp_duty_rate(trade_date: Any) -> float:
    when = _as_date(trade_date)
    if when is None or when >= STAMP_DUTY_HALVED_FROM:
        return STAMP_DUTY_CURRENT
    return STAMP_DUTY_LEGACY


@dataclass(frozen=True)
class TradingCosts:
    commission: float
    stamp_duty: float
    transfer_fee: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"total": self.total}


def trading_costs(
    *,
    notional_cny: float,
    side: str,
    trade_date: Any,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    commission_minimum: float = COMMISSION_MINIMUM_CNY,
) -> TradingCosts:
    notional = abs(float(notional_cny))
    commission = max(notional * commission_rate, commission_minimum) if notional else 0.0
    duty = notional * stamp_duty_rate(trade_date) if side.upper() == "SELL" else 0.0
    return TradingCosts(
        commission=commission,
        stamp_duty=duty,
        transfer_fee=notional * TRANSFER_FEE,
    )


def round_to_lot(
    shares: float,
    *,
    board: str,
    side: str,
    is_full_liquidation: bool = False,
) -> int:
    """Round to a legal board quantity without inventing an odd-lot exception.

    A normal sell must satisfy the same board minimum as a normal buy.  The only
    sub-minimum exception is a *complete* residual sell.  For main-board and
    ChiNext, quantities at/above the minimum still round to the 100-share
    increment; therefore 250 shares becomes a normal 200-share order rather than
    an invalid 250-share full-liquidation order.  STAR/BSE preserve their
    one-share increments above their minimums.
    """
    minimum, increment = LOT_RULES.get(board, (100, 100))
    shares = float(shares)
    if shares <= 0:
        return 0
    if shares < minimum:
        if side.upper() == "SELL" and is_full_liquidation:
            return int(shares)
        return 0
    return int(minimum + ((shares - minimum) // increment) * increment)


def execution_price(
    *,
    last_price: float,
    side: str,
    quantity: float,
    session_volume: float,
    slippage_bps: float,
    impact_coefficient: float,
    limits: PriceLimits | None = None,
    limit_price: float | None = None,
) -> float:
    base = float(last_price)
    slippage = base * (float(slippage_bps) / 10_000.0)
    participation = float(quantity) / float(session_volume) if session_volume > 0 else 0.0
    impact = base * float(impact_coefficient) * (participation ** 0.5)
    adjustment = slippage + impact
    buying = side.upper() == "BUY"
    price = base + adjustment if buying else base - adjustment

    if limits is not None and not limits.unlimited:
        if limits.limit_up is not None:
            price = min(price, limits.limit_up)
        if limits.limit_down is not None:
            price = max(price, limits.limit_down)
    if limit_price is not None:
        price = min(price, limit_price) if buying else max(price, limit_price)
    return round(price, 2)


@dataclass(frozen=True)
class TradabilityVerdict:
    can_buy: bool
    can_sell: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tradability(
    *,
    is_suspended: bool = False,
    at_limit_up: bool = False,
    at_limit_down: bool = False,
    is_delisting_period: bool = False,
    holding_acquired_today: bool = False,
) -> TradabilityVerdict:
    reasons: list[str] = []
    can_buy = True
    can_sell = True
    if is_suspended:
        can_buy = can_sell = False
        reasons.append("security is suspended")
    if at_limit_up:
        can_buy = False
        reasons.append("locked at limit up: no offers to lift")
    if at_limit_down:
        can_sell = False
        reasons.append("locked at limit down: no bids to hit")
    if is_delisting_period:
        can_buy = False
        reasons.append("delisting arrangement period: entry prohibited")
    if holding_acquired_today:
        can_sell = False
        reasons.append("T+1: shares bought today settle tomorrow")
    return TradabilityVerdict(can_buy, can_sell, tuple(reasons))


__all__ = [
    "SH_MAIN", "SZ_MAIN", "CHINEXT", "STAR", "BSE", "BOARDS",
    "ORDINARY_LIMIT", "MAIN_BOARD_ST_LIMIT_REFORM", "ST_LIMIT_LEGACY",
    "ST_LIMIT_CURRENT", "ST_LIMIT", "IPO_UNLIMITED_DAYS",
    "REGISTRATION_SYSTEM_MAIN_BOARD", "LEGACY_IPO_UP", "LEGACY_IPO_DOWN",
    "IPO_HALT_THRESHOLDS", "IPO_HALT_MINUTES", "STAMP_DUTY_CURRENT",
    "STAMP_DUTY_LEGACY", "STAMP_DUTY_HALVED_FROM", "TRANSFER_FEE",
    "DEFAULT_COMMISSION_RATE", "COMMISSION_MINIMUM_CNY", "LOT_RULES",
    "MAX_ORDER_QUANTITY", "PriceLimits", "TradingCosts", "TradabilityVerdict",
    "exchange_board_for_symbol", "max_order_quantity", "price_limits",
    "stamp_duty_rate", "trading_costs", "round_to_lot", "execution_price",
    "tradability",
]
