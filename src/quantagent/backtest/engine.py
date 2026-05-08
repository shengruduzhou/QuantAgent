from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import (
    ASharePriceLimit,
    TPlusOnePosition,
    enforce_tradability,
    limit_down_mask,
    limit_up_mask,
    suspension_mask,
)
from quantagent.quant_math.constraints import weights_to_lot_shares
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


@dataclass
class BacktestResult:
    nav_curve: pd.Series
    daily_returns: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: dict[str, float]


class EventDrivenBacktester:
    """Vectorized-by-day, T+1 aware A-share simulator. Inputs: weights + OHLCV."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

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

        for i, date in enumerate(dates):
            for pos in positions.values():
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

            target_shares = weights_to_lot_shares(
                tradable_target.reindex(symbols).fillna(0.0),
                nav,
                fill_prices.reindex(symbols),
                lot_size=self.config.lot_size,
            )
            for sym in symbols:
                if sym not in fill_prices.index or pd.isna(fill_prices.loc[sym]):
                    continue
                desired = int(target_shares.get(sym, 0))
                pos = positions[sym]
                current_total = pos.total_shares()
                delta = desired - current_total
                if delta == 0:
                    continue
                price = float(fill_prices.loc[sym])
                if delta > 0 and bool(fill_flags_up.get(sym, False)):
                    continue
                if delta < 0 and bool(fill_flags_down.get(sym, False)):
                    continue
                if bool(fill_flags_susp.get(sym, False)):
                    continue
                if delta > 0:
                    cash, traded_shares = self._execute_buy(cash, pos, sym, delta, price, trade_log, fill_date)
                else:
                    cash, traded_shares = self._execute_sell(cash, pos, sym, -delta, price, trade_log, fill_date)
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
        return BacktestResult(
            nav_curve=nav_series,
            daily_returns=returns,
            holdings=holdings,
            trades=trades,
            diagnostics={
                "final_nav": float(nav),
                "total_return": float(nav / self.config.initial_nav - 1.0),
                "trade_count": float(len(trades)),
            },
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
