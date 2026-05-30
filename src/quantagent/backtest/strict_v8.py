"""StrictBacktestV8 — spec section 9 full-output backtest wrapper.

Wraps the existing :func:`simulate_ashare_target_weights` (T+1 +
limit-up/down + suspension + ST + cost model + risk events) with the
extra metric computation + CSV emission demanded by the spec:

* total_return / annualized_return / max_drawdown / sharpe / calmar /
  volatility / turnover / win_rate / avg_profit_per_trade
* profit_by_stock / profit_by_sector
* selected_stocks.csv / trades.csv / pnl.csv / failed_orders.csv /
  risk_events.json / factor_weights.json / metrics.json

The output bundle is a single :class:`StrictBacktestArtifactSet` so
callers can ``set.write(output_dir)`` and get every file in one place.

This module does NOT model anything new — it is a thin reporting
layer on top of the existing simulator + cost_model. That keeps the
PIT / T+1 / cost / slippage / kill-switch guarantees of the
upstream layer intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    AShareExecutionSimulationResult,
    simulate_ashare_target_weights,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrictBacktestMetrics:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    calmar: float
    volatility: float
    turnover: float
    win_rate: float
    avg_profit_per_trade: float
    n_trades: int
    start_date: str
    end_date: str

    def to_dict(self) -> dict[str, object]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "turnover": self.turnover,
            "win_rate": self.win_rate,
            "avg_profit_per_trade": self.avg_profit_per_trade,
            "n_trades": int(self.n_trades),
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def _compute_metrics(
    nav: pd.Series,
    order_audit: pd.DataFrame,
    *,
    periods: int = 252,
) -> StrictBacktestMetrics:
    if nav is None or nav.empty:
        return StrictBacktestMetrics(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
            start_date="", end_date="",
        )
    nav_clean = nav.sort_index().dropna()
    daily_ret = nav_clean.pct_change().dropna()
    n_days = max(1, len(daily_ret))
    total_return = float(nav_clean.iloc[-1] / nav_clean.iloc[0] - 1.0) if len(nav_clean) >= 2 else 0.0
    ann_return = float((1.0 + total_return) ** (periods / n_days) - 1.0) if n_days >= 1 else 0.0
    if len(daily_ret) >= 2:
        std = float(daily_ret.std(ddof=1))
        sharpe = float(daily_ret.mean() / std * (periods ** 0.5)) if std > 1e-12 else 0.0
        vol = float(std * (periods ** 0.5))
    else:
        sharpe = 0.0
        vol = 0.0
    nav_curve = nav_clean.values
    peaks = np.maximum.accumulate(nav_curve)
    drawdowns = nav_curve / peaks - 1.0
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) else 0.0
    calmar = float(ann_return / max_dd) if max_dd > 1e-9 else float(ann_return * 10.0)
    # Turnover: |Δposition_value| / nav, summed per trade
    if order_audit is not None and not order_audit.empty and "filled_quantity" in order_audit.columns:
        filled = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0]
        if not filled.empty and "avg_price" in filled.columns:
            turnover_value = float(
                (filled["filled_quantity"].astype(float).abs() * filled["avg_price"].astype(float)).sum()
            )
            nav_mean = float(nav_clean.mean()) if not nav_clean.empty else 0.0
            turnover = turnover_value / max(1.0, nav_mean) / max(1, n_days)
            # Per-trade pnl: use signed (quantity * price). Buys spend,
            # sells receive — net per symbol gives realised pnl signed.
            sign = filled["side"].map({"buy": -1.0, "sell": 1.0, "BUY": -1.0, "SELL": 1.0}).fillna(0.0)
            pnl_per_trade = (filled["filled_quantity"].astype(float).abs() *
                              filled["avg_price"].astype(float) * sign)
            n_trades = int(len(filled))
            avg_profit = float(pnl_per_trade.mean()) if n_trades > 0 else 0.0
            wins = int((pnl_per_trade > 0).sum())
            win_rate = float(wins / n_trades) if n_trades > 0 else 0.0
        else:
            turnover = 0.0
            avg_profit = 0.0
            win_rate = 0.0
            n_trades = 0
    else:
        turnover = 0.0
        avg_profit = 0.0
        win_rate = 0.0
        n_trades = 0
    return StrictBacktestMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        calmar=calmar,
        volatility=vol,
        turnover=turnover,
        win_rate=win_rate,
        avg_profit_per_trade=avg_profit,
        n_trades=n_trades,
        start_date=str(nav_clean.index[0]) if len(nav_clean) else "",
        end_date=str(nav_clean.index[-1]) if len(nav_clean) else "",
    )


def _profit_by_stock(order_audit: pd.DataFrame) -> pd.DataFrame:
    if order_audit is None or order_audit.empty:
        return pd.DataFrame(columns=["symbol", "n_fills", "gross_value", "pnl_proxy"])
    work = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0].copy()
    if work.empty:
        return pd.DataFrame(columns=["symbol", "n_fills", "gross_value", "pnl_proxy"])
    sign = work["side"].astype(str).str.lower().map({"buy": -1.0, "sell": 1.0}).fillna(0.0)
    work["signed_value"] = (
        work["filled_quantity"].astype(float).abs()
        * work["avg_price"].astype(float)
        * sign
    )
    work["gross_value"] = (
        work["filled_quantity"].astype(float).abs() * work["avg_price"].astype(float)
    )
    grp = work.groupby("symbol")
    return (
        pd.DataFrame({
            "symbol": grp["symbol"].first(),
            "n_fills": grp.size(),
            "gross_value": grp["gross_value"].sum(),
            "pnl_proxy": grp["signed_value"].sum(),
        })
        .reset_index(drop=True)
        .sort_values("pnl_proxy", ascending=False)
    )


def _profit_by_sector(
    by_stock: pd.DataFrame,
    sector_map: pd.DataFrame | None,
) -> pd.DataFrame:
    if by_stock is None or by_stock.empty:
        return pd.DataFrame(columns=["sector_level_1", "gross_value", "pnl_proxy"])
    if sector_map is None or sector_map.empty or "sector_level_1" not in sector_map.columns:
        return pd.DataFrame(columns=["sector_level_1", "gross_value", "pnl_proxy"])
    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    by_stock = by_stock.copy()
    by_stock["symbol"] = by_stock["symbol"].astype(str)
    joined = by_stock.merge(
        sm[["symbol", "sector_level_1"]].drop_duplicates("symbol"),
        on="symbol", how="left",
    )
    joined["sector_level_1"] = joined["sector_level_1"].fillna("UNKNOWN")
    return (
        joined.groupby("sector_level_1")
        .agg(gross_value=("gross_value", "sum"), pnl_proxy=("pnl_proxy", "sum"))
        .reset_index()
        .sort_values("pnl_proxy", ascending=False)
    )


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------

@dataclass
class StrictBacktestArtifactSet:
    metrics: StrictBacktestMetrics
    nav: pd.Series
    daily_pnl: pd.DataFrame
    selected_stocks: pd.DataFrame
    trades: pd.DataFrame
    failed_orders: pd.DataFrame
    risk_events: list[dict]
    profit_by_stock: pd.DataFrame
    profit_by_sector: pd.DataFrame
    factor_weights: dict[str, float] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["metrics"] = out / "metrics.json"
        paths["metrics"].write_text(
            json.dumps(self.metrics.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        paths["nav"] = out / "nav.csv"
        self.nav.to_frame("nav").to_csv(paths["nav"], index_label="trade_date")
        paths["pnl"] = out / "pnl.csv"
        self.daily_pnl.to_csv(paths["pnl"], index=False)
        paths["selected_stocks"] = out / "selected_stocks.csv"
        self.selected_stocks.to_csv(paths["selected_stocks"], index=False)
        paths["trades"] = out / "trades.csv"
        self.trades.to_csv(paths["trades"], index=False)
        paths["failed_orders"] = out / "failed_orders.csv"
        self.failed_orders.to_csv(paths["failed_orders"], index=False)
        paths["risk_events"] = out / "risk_events.json"
        paths["risk_events"].write_text(
            json.dumps(self.risk_events, indent=2, default=str),
            encoding="utf-8",
        )
        paths["profit_by_stock"] = out / "profit_by_stock.csv"
        self.profit_by_stock.to_csv(paths["profit_by_stock"], index=False)
        paths["profit_by_sector"] = out / "profit_by_sector.csv"
        self.profit_by_sector.to_csv(paths["profit_by_sector"], index=False)
        paths["factor_weights"] = out / "factor_weights.json"
        paths["factor_weights"].write_text(
            json.dumps(self.factor_weights, indent=2, default=str),
            encoding="utf-8",
        )
        return paths


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_strict_backtest_v8(
    target_weights: pd.DataFrame,
    market_panel: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
    factor_weights: Mapping[str, float] | None = None,
    config: AShareExecutionSimulationConfig | None = None,
) -> StrictBacktestArtifactSet:
    """Run the existing strict simulator + emit the full v8 report bundle."""
    cfg = config or AShareExecutionSimulationConfig()
    sim: AShareExecutionSimulationResult = simulate_ashare_target_weights(
        target_weights, market_panel, cfg,
    )
    metrics = _compute_metrics(sim.nav, sim.order_audit)
    by_stock = _profit_by_stock(sim.order_audit)
    by_sector = _profit_by_sector(by_stock, sector_map)

    nav_series = sim.nav.copy() if sim.nav is not None else pd.Series(dtype=float)
    if not nav_series.empty:
        daily_pnl = nav_series.pct_change().fillna(0.0).rename("daily_return").to_frame()
        daily_pnl["nav"] = nav_series.values
        daily_pnl = daily_pnl.reset_index().rename(columns={"index": "trade_date"})
    else:
        daily_pnl = pd.DataFrame(columns=["trade_date", "daily_return", "nav"])

    # selected_stocks = unique symbols with at least one filled order
    if sim.order_audit is not None and not sim.order_audit.empty:
        filled = sim.order_audit[sim.order_audit["filled_quantity"].astype(float).abs() > 0]
        if not filled.empty:
            selected = (
                filled.groupby("symbol")
                .agg(first_filled=("trade_date", "min"),
                     last_filled=("trade_date", "max"),
                     n_fills=("symbol", "size"))
                .reset_index()
            )
        else:
            selected = pd.DataFrame(columns=["symbol", "first_filled", "last_filled", "n_fills"])
    else:
        selected = pd.DataFrame(columns=["symbol", "first_filled", "last_filled", "n_fills"])

    trades = sim.order_audit.copy() if sim.order_audit is not None else pd.DataFrame()
    failed = sim.failed_order_audit.copy() if sim.failed_order_audit is not None else pd.DataFrame()

    return StrictBacktestArtifactSet(
        metrics=metrics,
        nav=nav_series,
        daily_pnl=daily_pnl,
        selected_stocks=selected,
        trades=trades,
        failed_orders=failed,
        risk_events=list(sim.risk_events) if sim.risk_events else [],
        profit_by_stock=by_stock,
        profit_by_sector=by_sector,
        factor_weights=dict(factor_weights or {}),
        config=dict(sim.config or {}),
    )


__all__ = [
    "StrictBacktestArtifactSet",
    "StrictBacktestMetrics",
    "run_strict_backtest_v8",
]
