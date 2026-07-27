"""Mainland A-share trading rules, encoded once with their sources.

Every constant here is a *market rule*, not a modelling choice, so each carries
the authority and effective date it comes from. Rules change: the stamp duty
halved on 2023-08-28, main-board IPOs stopped having a first-day limit on
2023-04-10 under the full registration system. A simulator that hard-codes the
old value silently backtests a market that no longer exists, so the date-aware
lookups below take the trade date rather than assuming "today's rules".

Nothing in this module is an FX or futures convention. A-share cash equities
settle T+1 with no intraday round trip, price limits are per-board, and the
sell side alone pays stamp duty.

Sources (accessed 2026-07-27):

* 全面注册制主板新股前 5 日不设涨跌幅 — 中国新闻网 2023-04-10
  https://www.chinanews.com.cn/cj/2023/04-10/9986979.shtml
* 主板交易规则变化（含盘中临时停牌 30%/60%）— 武汉市地方金融管理局 2023-04-11
  https://jrj.wuhan.gov.cn/ztzl_57/xyrd/dcczbsc/202304/t20230411_2184063.shtml
* 新股上市前 5 日规则汇总（各板对比）— 新浪新闻
  https://www.sina.cn/news/detail/5322735332887589.html
* 印花税自 2023-08-28 减半至 0.05%、单边卖出 — 21 经济网 2023-08-27
  https://www.21jingji.com/article/20230827/herald/ac10f7d4d2834c760d7f8b798bcc6d0c.html
* 过户费 0.001% 双边、佣金约万 2.5-万 3 — A 股交易费用一览
  https://zhuanlan.zhihu.com/p/653909009
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from typing import Any, Mapping

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
#: Ordinary daily price-limit ratio by board.
#: ChiNext moved to ±20% on 2020-08-24; STAR has been ±20% since inception
#: (2019-07-22); BSE has been ±30% since it opened (2021-11-15).
ORDINARY_LIMIT: dict[str, float] = {
    SH_MAIN: 0.10, SZ_MAIN: 0.10, CHINEXT: 0.20, STAR: 0.20, BSE: 0.30,
}

#: ST / *ST names trade inside a narrower band on the main boards. ChiNext and
#: STAR ST names keep the board's ±20%; BSE has no separate ST band.
ST_LIMIT: dict[str, float] = {
    SH_MAIN: 0.05, SZ_MAIN: 0.05, CHINEXT: 0.20, STAR: 0.20, BSE: 0.30,
}

#: Trading days after listing during which no price limit applies.
#: Main boards gained this on 2023-04-10 with the full registration system;
#: before that a main-board IPO was capped at +44%/-36% on day one.
IPO_UNLIMITED_DAYS: dict[str, int] = {
    SH_MAIN: 5, SZ_MAIN: 5, CHINEXT: 5, STAR: 5, BSE: 1,
}

#: Date the main boards adopted the no-limit IPO window.
REGISTRATION_SYSTEM_MAIN_BOARD = date(2023, 4, 10)
#: Legacy approval-system main-board IPO day-one band.
LEGACY_IPO_UP = 0.44
LEGACY_IPO_DOWN = -0.36

#: Intraday circuit breakers for securities trading without a price limit:
#: a first move of ±30% from the open halts for 10 minutes, ±60% halts again.
IPO_HALT_THRESHOLDS: tuple[float, ...] = (0.30, 0.60)
IPO_HALT_MINUTES = 10

# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
#: Stamp duty, sell side only. Halved from 0.10% effective 2023-08-28.
STAMP_DUTY_CURRENT = 0.0005
STAMP_DUTY_LEGACY = 0.0010
STAMP_DUTY_HALVED_FROM = date(2023, 8, 28)

#: 过户费, charged on both sides, unified across SSE and SZSE in 2022.
TRANSFER_FEE = 0.00001

#: Retail commission. A modelling input, not a market rule -- brokers differ --
#: so it is a default the caller may override, with the ¥5 floor most retail
#: schedules apply.
DEFAULT_COMMISSION_RATE = 0.00025
COMMISSION_MINIMUM_CNY = 5.0

# ---------------------------------------------------------------------------
# Lot sizes
# ---------------------------------------------------------------------------
#: Minimum buy quantity and the increment above it, in shares.
#: STAR requires 200 shares minimum and then any whole-share increment; BSE
#: requires 100 and then single shares. The main boards and ChiNext trade in
#: 100-share lots throughout.
LOT_RULES: dict[str, tuple[int, int]] = {
    SH_MAIN: (100, 100), SZ_MAIN: (100, 100), CHINEXT: (100, 100),
    STAR: (200, 1), BSE: (100, 1),
}


@dataclass(frozen=True)
class PriceLimits:
    """The tradable band for one security on one day."""

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
    """A-share prices quote to 0.01 CNY; limits round to the nearest fen."""
    return round(value + 1e-12, 2)


def price_limits(
    *,
    board: str,
    previous_close: float,
    trade_date: Any,
    listing_date: Any = None,
    sessions_since_listing: int | None = None,
    is_st: bool = False,
) -> PriceLimits:
    """Resolve the price band for one security-day.

    ``sessions_since_listing`` is 0 on the listing day. When it is not supplied
    it is not guessed from calendar days -- the caller either knows the session
    count or the IPO window is treated as inapplicable, because a calendar-day
    approximation would silently mis-band names around holidays.
    """
    when = _as_date(trade_date)
    listed = _as_date(listing_date)

    if sessions_since_listing is None and listed is not None and when is not None:
        sessions_since_listing = None  # deliberately not approximated

    unlimited_days = IPO_UNLIMITED_DAYS.get(board, 0)
    if sessions_since_listing is not None and sessions_since_listing < unlimited_days:
        main_board = board in (SH_MAIN, SZ_MAIN)
        pre_reform = when is not None and when < REGISTRATION_SYSTEM_MAIN_BOARD
        if main_board and pre_reform:
            return PriceLimits(
                limit_up=_round_tick(previous_close * (1 + LEGACY_IPO_UP)),
                limit_down=_round_tick(previous_close * (1 + LEGACY_IPO_DOWN)),
                ratio=LEGACY_IPO_UP, regime="IPO_LEGACY_APPROVAL_SYSTEM",
                reference_close=previous_close,
            )
        return PriceLimits(
            limit_up=None, limit_down=None, ratio=None,
            regime="IPO_NO_LIMIT_WINDOW", reference_close=previous_close,
        )

    table = ST_LIMIT if is_st else ORDINARY_LIMIT
    ratio = table.get(board)
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
    """Sell-side stamp duty in force on ``trade_date``."""
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
    """Round-trip-aware A-share transaction costs for one fill.

    Stamp duty is sell-side only; commission and transfer fee are both sides.
    """
    notional = abs(float(notional_cny))
    commission = max(notional * commission_rate, commission_minimum) if notional else 0.0
    duty = notional * stamp_duty_rate(trade_date) if side.upper() == "SELL" else 0.0
    return TradingCosts(
        commission=commission, stamp_duty=duty, transfer_fee=notional * TRANSFER_FEE
    )


def round_to_lot(shares: float, *, board: str, side: str,
                 is_full_liquidation: bool = False) -> int:
    """Round an order size to a tradable quantity.

    Buys must respect the board's minimum and increment. Sells may go odd-lot
    **only** when liquidating a whole position -- that is the actual exchange
    rule, and modelling it as "sells are always free-form" overstates how
    cleanly a book can be exited.
    """
    minimum, increment = LOT_RULES.get(board, (100, 100))
    shares = float(shares)
    if shares <= 0:
        return 0
    if side.upper() == "SELL":
        if is_full_liquidation:
            return int(shares)
        rounded = int(shares // increment) * increment
        return rounded if rounded >= 0 else 0
    if shares < minimum:
        return 0
    return int(minimum + ((shares - minimum) // increment) * increment)


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
    """What a participant may actually do with this security right now.

    The asymmetries matter and are the ones retail backtests get wrong: you
    cannot buy a locked limit-up, you cannot sell a locked limit-down, and T+1
    means shares bought today cannot be sold today at any price.
    """
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
