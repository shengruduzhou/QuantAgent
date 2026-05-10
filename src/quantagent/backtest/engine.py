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


@dataclass
class BacktestResult:
    nav_curve: pd.Series
    daily_returns: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: dict[str, float]
    rejects: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict[str, float] = field(default_factory=dict)


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
                raw_target = float(tradable_target.reindex(symbols).fillna(0.0).get(sym, 0.0)) * nav / max(price, 1e-12)
                side_guess = "buy" if raw_target >= current_total else "sell"
                if side_guess == "buy":
                    desired_total = self.rule_engine.round_order_quantity(sym, "buy", raw_target, fill_date)
                else:
                    sell_qty = self.rule_engine.round_order_quantity(sym, "sell", current_total - raw_target, fill_date)
                    desired_total = current_total - sell_qty
                delta = desired_total - current_total
                if delta == 0:
                    continue
                if delta > 0 and bool(fill_flags_up.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "limit_up_no_buy")
                    continue
                if delta < 0 and bool(fill_flags_down.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "limit_down_no_sell")
                    continue
                if bool(fill_flags_susp.get(sym, False)):
                    self._reject(reject_log, fill_date, sym, "suspended")
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
                    continue
                fill = self.fill_model.fill("buy" if delta > 0 else "sell", abs(delta), price, state["volume"])
                if fill.filled_quantity <= 0:
                    self._reject(reject_log, fill_date, sym, fill.reject_reason or "zero_fill")
                    continue
                if delta > 0:
                    cash, traded_shares = self._execute_buy(cash, pos, sym, fill.filled_quantity, fill.fill_price, trade_log, fill_date)
                    if traded_shares <= 0:
                        self._reject(reject_log, fill_date, sym, "insufficient_cash")
                    else:
                        release_idx = min(i + (2 if self.config.next_day_fill else 1), len(dates) - 1)
                        release_dates[sym] = dates[release_idx]
                else:
                    cash, traded_shares = self._execute_sell(cash, pos, sym, fill.filled_quantity, fill.fill_price, trade_log, fill_date)
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
    ) -> tuple[float, int]:
        if shares <= 0 or price <= 0:
            return cash, 0
        gross = shares * price
        commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
        transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
        slippage = gross * self.config.cost.slippage_bps / 10000.0
        cost_total = gross + commission + transfer + slippage
        if cost_total > cash:
            affordable_lots = int(cash // (price * self.config.lot_size)) * self.config.lot_size
            if affordable_lots <= 0:
                return cash, 0
            shares = affordable_lots
            gross = shares * price
            commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
            transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
            slippage = gross * self.config.cost.slippage_bps / 10000.0
            cost_total = gross + commission + transfer + slippage
        cash -= cost_total
        pos.buy(shares)
        trade_log.append(
            {
                "trade_date": date,
                "symbol": symbol,
                "side": "buy",
                "shares": shares,
                "price": price,
                "commission": commission,
                "transfer_fee": transfer,
                "slippage": slippage,
            }
        )
        return cash, shares

    def _execute_sell(
        self,
        cash: float,
        pos: TPlusOnePosition,
        symbol: str,
        shares: int,
        price: float,
        trade_log: list[dict],
        date: pd.Timestamp,
    ) -> tuple[float, int]:
        sellable = pos.sell(shares)
        if sellable <= 0 or price <= 0:
            return cash, 0
        gross = sellable * price
        commission = max(gross * self.config.cost.commission_bps / 10000.0, self.config.cost.commission_min_rmb)
        transfer = gross * self.config.cost.transfer_fee_bps / 10000.0
        slippage = gross * self.config.cost.slippage_bps / 10000.0
        stamp = gross * self.config.cost.sell_stamp_duty_bps / 10000.0
        proceeds = gross - commission - transfer - slippage - stamp
        cash += proceeds
        trade_log.append(
            {
                "trade_date": date,
                "symbol": symbol,
                "side": "sell",
                "shares": sellable,
                "price": price,
                "commission": commission,
                "transfer_fee": transfer,
                "slippage": slippage,
                "stamp_duty": stamp,
            }
        )
        return cash, sellable

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
