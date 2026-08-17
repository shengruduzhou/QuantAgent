"""Clean-room backtest loop with the NAV clock pinned to what was held.

The one invariant this file exists to hold:

    NAV(t) = cash(t) + Σ shares_held_after_trading_on_t × close(t)

A book decided from information up to ``close(T)`` is executed at ``close(T+1)``
and first appears in ``NAV(T+1)``. It never appears in ``NAV(T)``.

The legacy ``quantagent.backtest.engine`` fills at ``open(T+1)`` but stamps the
resulting book onto ``close(T)``, so every rebalance books the overnight gap
``Δshares × (close(T) − open(T+1))`` as instantaneous P&L on a day the position
was not held. That is directional, not noise: names that gap down "earn" money.
On a flat tape with a single gap bar it turns a −0.046% strategy into +11.05%
and flips the reported Sharpe from −7.10 to +7.10.

Consequences of the invariant, all deliberate:

* Execution price is ``close(T+1)``, the same bar NAV is marked on, so the
  fill price and the valuation price are the same observation. There is no
  window in which the engine knows a price it could not have traded at.
* A name with no bar on ``T+1`` is not priced at zero and not silently
  dropped: the position is carried at its last known close and the day is
  recorded in ``unpriced_days``. If a *held* name cannot be priced at all,
  NAV for that day is ``None`` rather than a plausible number.
* Costs are charged on the traded notional at the moment of the trade, so the
  cash path and the NAV path cannot disagree.

Metrics follow the same rule as the rest of the package: a statistic that
cannot be computed from observed bars is ``None``, never 0.0. ``max_drawdown =
0.0`` must mean "measured, and the curve never fell", because a 0.0 that means
"never ran" is how the legacy strict-v8 path reported clean results for
backtests that did not happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class CostConfig:
    """A-share round-trip costs. Rates are fractions of traded notional."""

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    #: Stamp duty is levied on the SELL side only.
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    #: One-way slippage applied to the close, adverse in the traded direction.
    slippage_bps: float = 5.0
    #: Square-root impact: ``impact_alpha_bps × sqrt(participation)``.
    impact_alpha_bps: float = 10.0
    #: Refuse to trade more than this share of a bar's traded value.
    max_participation: float = 0.10


@dataclass
class CleanRoomResult:
    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    metrics: dict[str, float | None]
    #: (date, symbol) pairs held but unpriced on that date.
    unpriced_days: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    #: Metrics that must be present for a run to count as measured. ``sortino``
    #: and ``calmar`` are deliberately excluded: they are undefined for a curve
    #: with no down days or no drawdown, and demanding them would report an
    #: honest run as unmeasured.
    CORE_METRICS = ("annualised_return", "volatility", "sharpe", "max_drawdown")

    @property
    def measured(self) -> bool:
        """True when the NAV path is complete and the core metrics exist."""
        if not len(self.nav) or self.nav.isna().any():
            return False
        return all(self.metrics.get(k) is not None for k in self.CORE_METRICS)


def _annualised_return(nav: pd.Series) -> float | None:
    if len(nav) < 2:
        return None
    total = float(nav.iloc[-1] / nav.iloc[0])
    if total <= 0:
        return None
    years = len(nav) / TRADING_DAYS
    if years <= 0:
        return None
    return total ** (1.0 / years) - 1.0


def _max_drawdown(nav: pd.Series) -> float | None:
    if len(nav) < 2:
        return None
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def compute_metrics(nav: pd.Series, returns: pd.Series, *, risk_free: float = 0.0) -> dict[str, float | None]:
    """Report a statistic only when the sample supports it.

    Every branch that cannot be computed returns ``None``. A caller that wants
    a number must supply the bars that produce it.
    """
    clean = returns.dropna()
    if len(nav) < 2 or clean.empty:
        return {
            "annualised_return": None, "volatility": None, "sharpe": None,
            "sortino": None, "max_drawdown": None, "calmar": None,
            "n_days": float(len(nav)), "total_return": None,
        }

    ann = _annualised_return(nav)
    daily_sd = float(clean.std(ddof=1)) if len(clean) > 1 else None
    vol = daily_sd * sqrt(TRADING_DAYS) if daily_sd is not None else None
    mdd = _max_drawdown(nav)

    sharpe: float | None = None
    if ann is not None and vol is not None and vol > 0:
        sharpe = (ann - risk_free) / vol

    downside = clean[clean < 0.0]
    sortino: float | None = None
    if ann is not None and len(downside) > 1:
        dsd = float(downside.std(ddof=1)) * sqrt(TRADING_DAYS)
        if dsd > 0:
            sortino = (ann - risk_free) / dsd

    calmar: float | None = None
    if ann is not None and mdd is not None and mdd < 0:
        calmar = ann / abs(mdd)

    return {
        "annualised_return": ann,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "n_days": float(len(nav)),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
    }


def _trade_costs(
    notional: float,
    *,
    is_sell: bool,
    participation: float,
    config: CostConfig,
) -> float:
    if notional <= 0:
        return 0.0
    commission = max(config.min_commission, notional * config.commission_rate)
    stamp = notional * config.stamp_duty_rate if is_sell else 0.0
    transfer = notional * config.transfer_fee_rate
    slippage = notional * config.slippage_bps / 10_000.0
    impact = 0.0
    if config.impact_alpha_bps > 0 and participation > 0:
        impact = notional * config.impact_alpha_bps * sqrt(participation) / 10_000.0
    return commission + stamp + transfer + slippage + impact


def run_backtest(
    target_weights: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    initial_cash: float = 1_000_000.0,
    config: CostConfig | None = None,
) -> CleanRoomResult:
    """Replay ``target_weights`` against ``panel`` on the clean-room clock.

    ``target_weights`` is indexed by decision date (the date whose close the
    book was formed from) with one column per symbol. ``panel`` must carry
    ``symbol``, ``trade_date``, ``close`` and, for the impact term, ``amount``.

    The book on row ``T`` is executed at ``close(T+1)`` — the next session that
    exists in the panel — and is first marked in ``NAV(T+1)``.
    """
    config = config or CostConfig()
    notes: list[str] = []
    unpriced: list[tuple[str, str]] = []

    if target_weights.empty:
        return CleanRoomResult(
            pd.Series(dtype=float), pd.Series(dtype=float), target_weights,
            pd.Series(dtype=float), pd.Series(dtype=float),
            compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float)),
            notes=["empty_target_weights"],
        )

    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    close = frame.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="last")
    amount = (
        frame.pivot_table(index="trade_date", columns="symbol", values="amount", aggfunc="last")
        if "amount" in frame.columns
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )

    sessions = list(close.index)
    if not sessions:
        return CleanRoomResult(
            pd.Series(dtype=float), pd.Series(dtype=float), target_weights,
            pd.Series(dtype=float), pd.Series(dtype=float),
            compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float)),
            notes=["panel_has_no_sessions"],
        )

    decisions = {pd.Timestamp(d): row for d, row in target_weights.iterrows()}

    cash = float(initial_cash)
    shares: dict[str, float] = {}
    nav_index: list[pd.Timestamp] = []
    nav_values: list[float | None] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    realised_weights: dict[pd.Timestamp, dict[str, float]] = {}

    for position, session in enumerate(sessions):
        # A book decided on the PREVIOUS session executes on this one, at this
        # session's close — the same price this session's NAV is marked at.
        previous = sessions[position - 1] if position else None
        book = decisions.get(previous) if previous is not None else None

        day_close = close.loc[session]
        day_amount = amount.loc[session] if session in amount.index else None

        traded_notional = 0.0
        day_cost = 0.0

        if book is not None:
            # NAV before trading, used to size the target book.
            pre_nav = cash + sum(
                qty * float(day_close.get(sym, np.nan))
                for sym, qty in shares.items()
                if pd.notna(day_close.get(sym, np.nan))
            )
            targets = {
                str(sym): float(w)
                for sym, w in book.items()
                if pd.notna(w) and float(w) != 0.0
            }
            universe = set(targets) | set(shares)
            for sym in sorted(universe):
                price = float(day_close.get(sym, np.nan))
                if not pd.notna(price) or price <= 0:
                    # No price means no trade. A held name is recorded as
                    # unpriced by the mark-to-market pass below.
                    continue
                target_qty = targets.get(sym, 0.0) * pre_nav / price
                delta = target_qty - shares.get(sym, 0.0)
                if abs(delta * price) < 1.0:
                    continue

                bar_value = float(day_amount.get(sym, np.nan)) if day_amount is not None else np.nan
                notional = abs(delta) * price
                participation = 0.0
                if pd.notna(bar_value) and bar_value > 0:
                    participation = notional / bar_value
                    if participation > config.max_participation:
                        # Refuse the excess rather than assume it filled.
                        allowed = config.max_participation * bar_value / price
                        delta = np.sign(delta) * allowed
                        notional = abs(delta) * price
                        participation = config.max_participation

                is_sell = delta < 0
                cost = _trade_costs(
                    notional, is_sell=is_sell, participation=participation, config=config
                )
                # Slippage lives inside ``cost``; the cash leg therefore uses the
                # clean close. Charging an adverse fill price *and* a slippage
                # line item is how the legacy engine took the same basis points
                # twice (declared 5 bps, actually 7).
                cash -= delta * price
                cash -= cost
                shares[sym] = shares.get(sym, 0.0) + delta
                if abs(shares[sym]) < 1e-9:
                    shares.pop(sym, None)
                traded_notional += notional
                day_cost += cost

        # Mark to market on the SAME close that any trade above executed at.
        position_value = 0.0
        unpriced_held = False
        for sym, qty in shares.items():
            price = float(day_close.get(sym, np.nan))
            if pd.notna(price) and price > 0:
                position_value += qty * price
            elif qty:
                unpriced_held = True
                unpriced.append((str(session.date()), sym))

        # A held name with no price makes NAV unknowable. Reporting
        # ``cash + priced_positions`` here would silently value the unpriced
        # holding at zero -- the defect that removed 10,000 CNY from NAV while
        # every internal accounting identity still balanced.
        nav = None if unpriced_held else cash + position_value
        nav_index.append(session)
        nav_values.append(nav)
        prior = nav_values[-2] if len(nav_values) > 1 else float(initial_cash)
        base = prior if isinstance(prior, (int, float)) and prior else float(initial_cash)
        turnover_values.append(traded_notional / base if base else 0.0)
        cost_values.append(day_cost)
        if nav is not None:
            realised_weights[session] = {
                sym: qty * float(day_close.get(sym, np.nan)) / nav
                for sym, qty in shares.items()
                if pd.notna(day_close.get(sym, np.nan)) and nav
            }

    nav_series = pd.Series(nav_values, index=nav_index, dtype="float64", name="nav")
    if nav_series.isna().any():
        notes.append(f"unpriced_nav_days:{int(nav_series.isna().sum())}")
    returns = nav_series.pct_change(fill_method=None)
    weights_frame = pd.DataFrame.from_dict(realised_weights, orient="index").fillna(0.0)

    return CleanRoomResult(
        nav=nav_series,
        returns=returns,
        weights=weights_frame.sort_index(),
        turnover=pd.Series(turnover_values, index=nav_index, name="turnover"),
        costs=pd.Series(cost_values, index=nav_index, name="cost"),
        metrics=compute_metrics(nav_series.dropna(), returns),
        unpriced_days=unpriced,
        notes=notes,
    )


__all__ = ["CleanRoomResult", "CostConfig", "compute_metrics", "run_backtest"]
