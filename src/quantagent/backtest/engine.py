from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import (
    ASharePriceLimit,
    AshareRuleEngine,
    TPlusOnePosition,
    enforce_tradability,
    limit_down_mask,
    limit_up_mask,
    suspension_mask,
)
from quantagent.backtest.fill_model import AShareFillModel, FillModelConfig
from quantagent.domain.ledger import CanonicalLedger, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    Fill,
    OrderBook,
    OrderEventType,
    OrderIntent,
    RiskDecision,
    Side,
    Signal,
)
from quantagent.quant_math.transaction_cost import CostModelConfig


@dataclass(frozen=True)
class BacktestConfig:
    initial_nav: float = 1_000_000.0
    lot_size: int = 100
    fill_price_column: str = "open"
    next_day_fill: bool = True
    cost: CostModelConfig = field(default_factory=CostModelConfig)
    limits: ASharePriceLimit = field(default_factory=ASharePriceLimit)
    block_buy_limit_up: bool = True
    block_sell_limit_down: bool = True
    fill_model: FillModelConfig = field(default_factory=FillModelConfig)
    #: Written for every economic event. When None an in-memory ledger is used,
    #: so the canonical path is exercised even for a throwaway run — the engine
    #: has no code path that skips it.
    ledger_path: str | None = None
    #: Chain of custody stamped onto every entity this run produces.
    lineage: Lineage = field(default_factory=Lineage)


@dataclass
class BacktestResult:
    nav_curve: pd.Series
    daily_returns: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: dict[str, float]
    rejects: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict[str, float] = field(default_factory=dict)
    #: The canonical record. `trades` is a projection of `ledger`'s fills, not a
    #: second copy of the truth — reconciliation and replay read this.
    ledger: CanonicalLedger | None = None
    order_book: OrderBook | None = None


class EventDrivenBacktester:
    """Vectorized-by-day, T+1 aware A-share simulator. Inputs: weights + OHLCV."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.rule_engine = AshareRuleEngine()
        self.fill_model = AShareFillModel(self.config.fill_model)

    def run(
        self,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> BacktestResult:
        """target_weights: index=trade_date, columns=symbol. prices: long-form OHLCV."""
        prices = prices.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        prices = prices.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        flag_up = limit_up_mask(prices)
        flag_down = limit_down_mask(prices)
        flag_susp = suspension_mask(prices)
        prices["flag_up"] = flag_up
        prices["flag_down"] = flag_down
        prices["flag_susp"] = flag_susp

        target_weights.index = pd.to_datetime(target_weights.index)
        target_weights = target_weights.sort_index().fillna(0.0)
        dates = target_weights.index.unique()
        symbols = target_weights.columns.tolist()

        ledger = CanonicalLedger(self.config.ledger_path)
        # Start from whatever the chain already records. This engine's order ids
        # are content-addressed over (run_id, symbol, session), so re-running with
        # the same lineage against a populated ledger re-derives ids the file
        # already holds. Beginning with an empty book appended a second CREATED
        # and RISK_APPROVED for an order the file recorded as filled, leaving a
        # chain that no longer replays (DEF-014). Hydrated, the collision is
        # refused on the write and the file stays readable.
        book = ledger.replay_book() if len(ledger) else OrderBook()
        self._ledger = ledger
        self._book = book

        nav = self.config.initial_nav
        cash = nav
        positions: dict[str, TPlusOnePosition] = {s: TPlusOnePosition() for s in symbols}
        nav_curve: dict[pd.Timestamp, float] = {}
        weight_curve: list[dict] = []
        trade_log: list[dict] = []
        reject_log: list[dict] = []
        release_dates: dict[str, pd.Timestamp] = {s: dates[0] for s in symbols}

        for i, date in enumerate(dates):
            for sym, pos in positions.items():
                if pos.frozen_today_shares > 0 and date >= release_dates.get(sym, date):
                    pos.settle_overnight()
            daily_prices = prices[prices["trade_date"] == date].set_index("symbol")
            if daily_prices.empty:
                nav_curve[date] = nav
                continue
            fill_date = dates[i + 1] if (self.config.next_day_fill and i + 1 < len(dates)) else date
            fill_prices = (
                prices[prices["trade_date"] == fill_date]
                .set_index("symbol")[self.config.fill_price_column]
            )
            fill_flags_up = prices[prices["trade_date"] == fill_date].set_index("symbol")["flag_up"] if fill_date in prices["trade_date"].values else pd.Series(False, index=symbols)
            fill_flags_down = prices[prices["trade_date"] == fill_date].set_index("symbol")["flag_down"] if fill_date in prices["trade_date"].values else pd.Series(False, index=symbols)
            fill_flags_susp = prices[prices["trade_date"] == fill_date].set_index("symbol")["flag_susp"] if fill_date in prices["trade_date"].values else pd.Series(False, index=symbols)

            target = target_weights.loc[date]
            current_weights = self._current_weights(positions, daily_prices["close"], nav)
            can_buy = (~fill_flags_up.fillna(True)) & (~fill_flags_susp.fillna(True))
            can_sell = (~fill_flags_down.fillna(True)) & (~fill_flags_susp.fillna(True))
            tradable_target = enforce_tradability(target, current_weights, can_buy, can_sell)

            for sym in symbols:
                if sym not in fill_prices.index or pd.isna(fill_prices.loc[sym]):
                    self._reject(reject_log, fill_date, sym, "missing_price")
                    continue
                price = float(fill_prices.loc[sym])
                pos = positions[sym]
                current_total = pos.total_shares()
                # What the strategy actually asked for, before tradability was
                # enforced. `enforce_tradability` pins a blocked exit to the
                # current *weight*, measured at the signal-day price; re-solving
                # that weight at a limit-down fill price demands MORE shares, so
                # a sell the venue refused turned into a buy into the fall.
                intended_shares = (
                    float(target.reindex(symbols).fillna(0.0).get(sym, 0.0)) * nav / max(price, 1e-12)
                )
                if intended_shares < current_total and not bool(can_sell.get(sym, True)):
                    blocked = "suspended" if bool(fill_flags_susp.get(sym, False)) else "limit_down_no_sell"
                    self._reject(reject_log, fill_date, sym, blocked)
                    self._record_rejected_intent(
                        book, ledger, sym, Side.SELL,
                        abs(int(round(current_total - intended_shares))),
                        fill_date, price, blocked,
                    )
                    continue
                raw_target = float(tradable_target.reindex(symbols).fillna(0.0).get(sym, 0.0)) * nav / max(price, 1e-12)
                side_guess = "buy" if raw_target >= current_total else "sell"
                if side_guess == "buy":
                    desired_total = self.rule_engine.round_order_quantity(sym, "buy", raw_target, fill_date)
                else:
                    sell_qty = self.rule_engine.round_order_quantity(sym, "sell", current_total - raw_target, fill_date)
                    desired_total = current_total - sell_qty
                delta = desired_total - current_total
                if delta == 0:
                    # `enforce_tradability` already neutralised blocked targets
                    # above, so a limit-up buy arrives here as a no-op and the
                    # explicit reject paths below are unreachable. Without this
                    # the run shows neither a fill nor a reason — the order
                    # simply vanishes from the audit trail.
                    intended = float(target.reindex(symbols).fillna(0.0).get(sym, 0.0)) * nav / max(price, 1e-12)
                    intended_delta = intended - current_total
                    if intended_delta > 0 and not bool(can_buy.get(sym, True)):
                        blocked = "suspended" if bool(fill_flags_susp.get(sym, False)) else "limit_up_no_buy"
                        self._reject(reject_log, fill_date, sym, blocked)
                        self._record_rejected_intent(
                            book, ledger, sym, Side.BUY, abs(int(round(intended_delta))),
                            fill_date, price, blocked,
                        )
                    elif intended_delta < 0 and not bool(can_sell.get(sym, True)):
                        blocked = "suspended" if bool(fill_flags_susp.get(sym, False)) else "limit_down_no_sell"
                        self._reject(reject_log, fill_date, sym, blocked)
                        self._record_rejected_intent(
                            book, ledger, sym, Side.SELL, abs(int(round(intended_delta))),
                            fill_date, price, blocked,
                        )
                    continue
                if delta > 0 and bool(fill_flags_up.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "limit_up_no_buy")
                    self._record_rejected_intent(book, ledger, sym, Side.BUY, abs(int(delta)), fill_date, price, "limit_up_no_buy")
                    continue
                if delta < 0 and bool(fill_flags_down.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "limit_down_no_sell")
                    self._record_rejected_intent(book, ledger, sym, Side.SELL, abs(int(delta)), fill_date, price, "limit_down_no_sell")
                    continue
                if bool(fill_flags_susp.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "suspended")
                    self._record_rejected_intent(book, ledger, sym, Side.BUY if delta > 0 else Side.SELL, abs(int(delta)), fill_date, price, "suspended")
                    continue
                state = {
                    "symbol": sym,
                    "trade_date": fill_date,
                    "is_limit_up": bool(fill_flags_up.get(sym, False)),
                    "is_limit_down": bool(fill_flags_down.get(sym, False)),
                    "is_suspended": bool(fill_flags_susp.get(sym, False)),
                    "volume": float(prices.loc[(prices["trade_date"] == fill_date) & (prices["symbol"] == sym), "volume"].iloc[0])
                    if ((prices["trade_date"] == fill_date) & (prices["symbol"] == sym)).any()
                    else 0.0,
                    "available_shares": pos.available_shares,
                }
                valid, reason = self.rule_engine.validate_order_intent(
                    {"symbol": sym, "side": "buy" if delta > 0 else "sell", "quantity": abs(delta)},
                    state,
                )
                if not valid:
                    self._reject(reject_log, fill_date, sym, reason)
                    self._record_rejected_intent(book, ledger, sym, Side.BUY if delta > 0 else Side.SELL, abs(int(delta)), fill_date, price, reason)
                    continue
                fill = self.fill_model.fill("buy" if delta > 0 else "sell", abs(delta), price, state["volume"])
                if fill.filled_quantity <= 0:
                    self._reject(reject_log, fill_date, sym, fill.reject_reason or "zero_fill")
                    self._record_rejected_intent(book, ledger, sym, Side.BUY if delta > 0 else Side.SELL, abs(int(delta)), fill_date, price, fill.reject_reason or "zero_fill")
                    continue
                # The canonical order exists before any cash moves, so an order
                # that then fails on cash is still a queryable order with a
                # reason rather than an absence.
                canonical = self._open_canonical_order(
                    book, ledger, sym,
                    Side.BUY if delta > 0 else Side.SELL,
                    abs(int(delta)), fill_date, price,
                )
                if delta > 0:
                    cash, traded_shares, executed = self._execute_buy(cash, pos, sym, fill.filled_quantity, fill.fill_price, trade_log, fill_date, reference_price=price, order=canonical)
                    if traded_shares <= 0:
                        self._reject(reject_log, fill_date, sym, "insufficient_cash")
                        self._close_canonical_order(
                            book, ledger, canonical, OrderEventType.REJECTED,
                            reason="insufficient_cash", trade_date=fill_date,
                        )
                    else:
                        # Settlement must fall strictly after the fill session.
                        # Clamping this index to the last date used to make the
                        # final session settle its own purchases, so a buy
                        # filled that day was immediately sellable — a T+1
                        # violation that inflated the closing book. With no
                        # later session in the sample the shares simply never
                        # become sellable, which is the honest answer.
                        release_idx = i + (2 if self.config.next_day_fill else 1)
                        release_dates[sym] = (
                            dates[release_idx] if release_idx < len(dates) else pd.Timestamp.max
                        )
                        self._record_canonical_fill(book, ledger, canonical, executed, fill_date)
                else:
                    cash, traded_shares, executed = self._execute_sell(cash, pos, sym, fill.filled_quantity, fill.fill_price, trade_log, fill_date, reference_price=price, order=canonical)
                    if traded_shares <= 0:
                        self._close_canonical_order(
                            book, ledger, canonical, OrderEventType.REJECTED,
                            reason="no_sellable_inventory", trade_date=fill_date,
                        )
                    else:
                        self._record_canonical_fill(book, ledger, canonical, executed, fill_date)
            equity = cash + sum(
                pos.total_shares() * float(daily_prices["close"].get(sym, 0.0))
                for sym, pos in positions.items()
            )
            nav = equity
            nav_curve[date] = nav
            weight_curve.append(
                {
                    "trade_date": date,
                    **{sym: positions[sym].total_shares() * float(daily_prices["close"].get(sym, 0.0)) / max(nav, 1e-6) for sym in symbols},
                }
            )

        nav_series = pd.Series(nav_curve).sort_index()
        returns = nav_series.pct_change().dropna()
        holdings = pd.DataFrame(weight_curve).set_index("trade_date") if weight_curve else pd.DataFrame()
        trades = pd.DataFrame(trade_log)
        rejects = pd.DataFrame(reject_log)
        report = _performance_report(nav_series, returns, trades, rejects)
        sleeve_diagnostics = target_weights.attrs.get("sleeve_diagnostics", {}) if hasattr(target_weights, "attrs") else {}
        stop_loss_diagnostics = target_weights.attrs.get("stop_loss_diagnostics", {}) if hasattr(target_weights, "attrs") else {}
        return BacktestResult(
            nav_curve=nav_series,
            daily_returns=returns,
            holdings=holdings,
            trades=trades,
            diagnostics={
                "final_nav": float(nav),
                "total_return": float(nav / self.config.initial_nav - 1.0),
                "trade_count": float(len(trades)),
                "reject_count": float(len(rejects)),
                "turnover": float(report.get("turnover", 0.0)),
                "fill_ratio": float(report.get("fill_ratio", 1.0)),
                "sleeve_count": float(sleeve_diagnostics.get("sleeve_count", 0.0)),
                "stop_event_count": float(stop_loss_diagnostics.get("stop_event_count", 0.0)),
                "blocked_exit_count": float(stop_loss_diagnostics.get("blocked_exit_count", 0.0)),
            },
            rejects=rejects,
            report=report,
            ledger=ledger,
            order_book=book,
        )

    # -- canonical entity emission -----------------------------------------
    def _open_canonical_order(
        self,
        book: OrderBook,
        ledger: CanonicalLedger,
        symbol: str,
        side: Side,
        quantity: int,
        trade_date: pd.Timestamp,
        reference_price: float,
    ):
        """Signal -> OrderIntent -> Order, all recorded before cash moves."""
        session = str(pd.Timestamp(trade_date).date())
        signal = Signal.create(
            symbol=symbol, trade_date=session, score=0.0, lineage=self.config.lineage,
        )
        intent = OrderIntent.create(
            symbol=symbol, side=side, quantity=quantity, trade_date=session,
            lineage=signal.lineage, reference_price=reference_price,
        )
        order = mirror_open(book, ledger, intent, trade_date=session)
        # The fast engine applies its constraints upstream (tradability, lots,
        # price bands), so reaching here *is* the approval. Recording it keeps
        # the state machine honest rather than letting orders skip a stage.
        decision = RiskDecision.create(
            approved=True, rule="fast_engine_pretrade", threshold="tradability+lot+band",
            measured="passed", reason="passed upstream constraints", lineage=order.lineage,
        )
        for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
            book.apply(
                order.order_id, event,
                risk_decision=decision if event is OrderEventType.RISK_APPROVED else None,
            )
            ledger.append(book.history_of(order.order_id)[-1], trade_date=session)
        return order

    def _record_canonical_fill(
        self,
        book: OrderBook,
        ledger: CanonicalLedger,
        order,
        executed: Fill | None,
        trade_date: pd.Timestamp,
    ) -> None:
        if executed is None:
            return
        session = str(pd.Timestamp(trade_date).date())
        event_type = (
            OrderEventType.FILL
            if executed.quantity >= order.quantity
            else OrderEventType.PARTIAL_FILL
        )
        book.apply(order.order_id, event_type, fill=executed)
        ledger.append(book.history_of(order.order_id)[-1], trade_date=session)

    def _close_canonical_order(
        self,
        book: OrderBook,
        ledger: CanonicalLedger,
        order,
        event_type: OrderEventType,
        *,
        reason: str,
        trade_date: pd.Timestamp,
    ) -> None:
        session = str(pd.Timestamp(trade_date).date())
        book.apply(order.order_id, event_type, reason=reason)
        ledger.append(book.history_of(order.order_id)[-1], trade_date=session)

    def _canonical_fill(
        self,
        *,
        order,
        symbol: str,
        side: Side,
        shares: int,
        price: float,
        reference_price: float | None,
        commission: float,
        transfer: float,
        stamp: float,
        date: pd.Timestamp,
    ) -> Fill:
        execution_id = f"{order.order_id}-ex{len(order.fills) + 1}"
        return Fill(
            execution_id=execution_id, order_id=order.order_id, symbol=symbol, side=side,
            quantity=int(shares), price=float(price), reference_price=reference_price,
            commission=float(commission), transfer_fee=float(transfer), stamp_duty=float(stamp),
            filled_at=str(pd.Timestamp(date).date()),
            lineage=order.lineage.derive(execution_id=execution_id),
        )

    def _execute_buy(
        self,
        cash: float,
        pos: TPlusOnePosition,
        symbol: str,
        shares: int,
        price: float,
        trade_log: list[dict],
        date: pd.Timestamp,
        reference_price: float | None = None,
        order=None,
    ) -> tuple[float, int, Fill | None]:
        if shares <= 0 or price <= 0:
            return cash, 0, None
        # `price` is already the fill model's executed price: it has been moved
        # away from the reference by spread and impact. Charging
        # `cost.slippage_bps` on top of it billed the same friction twice — the
        # engine's effective slippage was ~7 bps while every config declared 5.
        # Slippage is the price move, and it is recorded, not re-charged.
        gross = shares * price
        slippage = abs(price - reference_price) * shares if reference_price else 0.0
        commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
        transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
        cost_total = gross + commission + transfer
        if cost_total > cash:
            affordable_lots = int(cash // (price * self.config.lot_size)) * self.config.lot_size
            if affordable_lots <= 0:
                return cash, 0, None
            shares = affordable_lots
            gross = shares * price
            slippage = abs(price - reference_price) * shares if reference_price else 0.0
            commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
            transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
            cost_total = gross + commission + transfer
        cash -= cost_total
        pos.buy(shares)
        executed = self._canonical_fill(
            order=order, symbol=symbol, side=Side.BUY, shares=shares, price=price,
            reference_price=reference_price, commission=commission, transfer=transfer,
            stamp=0.0, date=date,
        ) if order is not None else None
        # The dict row is a projection of the canonical fill, not a parallel
        # record: derived here so the two can never disagree.
        trade_log.append(
            self._trade_row(executed, date)
            if executed is not None
            else {
                "trade_date": date, "symbol": symbol, "side": "buy", "shares": shares,
                "price": price, "commission": commission, "transfer_fee": transfer,
                "slippage": slippage,
            }
        )
        return cash, shares, executed

    def _execute_sell(
        self,
        cash: float,
        pos: TPlusOnePosition,
        symbol: str,
        shares: int,
        price: float,
        trade_log: list[dict],
        date: pd.Timestamp,
        reference_price: float | None = None,
        order=None,
    ) -> tuple[float, int, Fill | None]:
        sellable = pos.sell(shares)
        if sellable <= 0 or price <= 0:
            return cash, 0, None
        # See `_execute_buy`: the fill price already carries the slippage.
        gross = sellable * price
        slippage = abs(price - reference_price) * sellable if reference_price else 0.0
        commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
        transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
        stamp = gross * self.config.cost.sell_stamp_duty_bps / 10000.0
        proceeds = gross - commission - transfer - stamp
        cash += proceeds
        executed = self._canonical_fill(
            order=order, symbol=symbol, side=Side.SELL, shares=sellable, price=price,
            reference_price=reference_price, commission=commission, transfer=transfer,
            stamp=stamp, date=date,
        ) if order is not None else None
        trade_log.append(
            self._trade_row(executed, date)
            if executed is not None
            else {
                "trade_date": date, "symbol": symbol, "side": "sell", "shares": sellable,
                "price": price, "commission": commission, "transfer_fee": transfer,
                "slippage": slippage, "stamp_duty": stamp,
            }
        )
        return cash, sellable, executed

    @staticmethod
    def _trade_row(fill: Fill, date: pd.Timestamp) -> dict:
        """Project a canonical fill into the legacy trade-log shape.

        One-way and derived: every field comes from the fill, so the DataFrame
        cannot drift from the ledger it is rendered from.
        """
        row = {
            "trade_date": date,
            "symbol": fill.symbol,
            "side": fill.side.value.lower(),
            "shares": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "transfer_fee": fill.transfer_fee,
            "slippage": fill.slippage,
        }
        if fill.side is Side.SELL:
            row["stamp_duty"] = fill.stamp_duty
        return row

    @staticmethod
    def _current_weights(
        positions: dict[str, TPlusOnePosition],
        last_close: pd.Series,
        nav: float,
    ) -> pd.Series:
        if nav <= 0:
            return pd.Series(0.0, index=last_close.index)
        rows = {}
        for sym, pos in positions.items():
            close = float(last_close.get(sym, 0.0))
            rows[sym] = pos.total_shares() * close / nav
        return pd.Series(rows)

    @staticmethod
    def _reject(log: list[dict], date: pd.Timestamp, symbol: str, reason: str) -> None:
        log.append({"trade_date": date, "symbol": symbol, "reason": reason})

    def _record_rejected_intent(
        self,
        book: OrderBook,
        ledger: CanonicalLedger,
        symbol: str,
        side: Side,
        quantity: int,
        trade_date: pd.Timestamp,
        reference_price: float,
        reason: str,
    ) -> None:
        """Put a refused order on the canonical ledger.

        A refusal that exists only in `reject_log` is invisible to replay and to
        anything reading the ledger, so "every rejected order is queryable" was
        true of the DataFrame and false of the record of account. The strategy
        intended this order; the refusal is part of its history.
        """
        if quantity <= 0:
            quantity = 1  # the intent existed even when it rounded to nothing
        session = str(pd.Timestamp(trade_date).date())
        signal = Signal.create(
            symbol=symbol, trade_date=f"{session}-rej-{reason}", score=0.0,
            lineage=self.config.lineage,
        )
        intent = OrderIntent.create(
            symbol=symbol, side=side, quantity=int(quantity), trade_date=session,
            lineage=signal.lineage, reference_price=reference_price,
        )
        order = mirror_open(book, ledger, intent, trade_date=session)
        decision = RiskDecision.create(
            approved=False, rule=reason, threshold="a-share tradability",
            measured=reason, reason=reason, lineage=order.lineage,
        )
        book.apply(order.order_id, OrderEventType.RISK_REJECTED, reason=reason, risk_decision=decision)
        ledger.append(book.history_of(order.order_id)[-1], trade_date=session)


def _performance_report(
    nav: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    rejects: pd.DataFrame,
) -> dict[str, float]:
    if nav.empty:
        return {}
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(len(nav), 1)) - 1.0 if nav.iloc[0] > 0 else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() * 252) / vol) if vol > 1e-12 else 0.0
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(252) if len(returns[returns < 0]) > 1 else 0.0
    sortino = float((returns.mean() * 252) / downside) if downside and downside > 1e-12 else 0.0
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(ann_ret / abs(max_dd)) if max_dd < -1e-12 else 0.0
    traded_notional = 0.0
    cost = 0.0
    if not trades.empty:
        traded_notional = float((trades["shares"] * trades["price"]).abs().sum())
        cost_columns = [c for c in ["commission", "transfer_fee", "slippage", "stamp_duty"] if c in trades.columns]
        cost = float(trades[cost_columns].fillna(0.0).sum().sum()) if cost_columns else 0.0
    attempts = len(trades) + len(rejects)
    return {
        "annualized_return": float(ann_ret),
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "turnover": traded_notional / float(nav.iloc[0]) if nav.iloc[0] else 0.0,
        "cost_attribution": cost,
        "fill_ratio": len(trades) / attempts if attempts else 1.0,
        "reject_summary_count": float(len(rejects)),
        "capacity_proxy": traded_notional / max(len(nav), 1),
    }
