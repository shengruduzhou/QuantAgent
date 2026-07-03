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
from quantagent.backtest.quarantine import (
    FORENSICS_TRUST_CLASS,
    check_window,
)


def quarantine_trust_stamp(dates) -> dict[str, object] | None:
    """Return a forensics trust stamp when any of ``dates`` falls in a
    quarantined window, else None. Metadata only — never blocks execution
    (27 research scripts call run_strict_backtest_v8 directly; the trusted
    fail-closed path is baseline_protocol.evaluate)."""
    if dates is None or len(dates) == 0:
        return None
    start, end = min(dates), max(dates)
    hit = check_window(start, end)
    if hit is None:
        return None
    return {
        "trust_class": FORENSICS_TRUST_CLASS,
        "quarantine_window": f"{hit.start.date()}..{hit.end.date()}",
        "quarantine_reason": hit.reason,
        "quarantine_evidence": hit.evidence,
    }


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
    # ── realised round-trip trade statistics (FIFO-matched, net of cost) ──
    win_rate: float                 # fraction of CLOSED trades with net_pnl > 0
    avg_profit_per_trade: float     # mean net realised PnL per closed trade (yuan)
    median_profit_per_trade: float
    profit_factor: float            # gross_profit / gross_loss
    gross_profit: float
    gross_loss: float
    total_cost: float               # commission+stamp+transfer on matched trades
    n_trades: int                   # number of CLOSED round-trip trades
    n_fills: int                    # number of individual fills (was the old n_trades)
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
            "median_profit_per_trade": self.median_profit_per_trade,
            "profit_factor": self.profit_factor,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "total_cost": self.total_cost,
            "n_trades": int(self.n_trades),
            "n_fills": int(self.n_fills),
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def _realized_round_trip_pnl(
    order_audit: pd.DataFrame,
    cost_model: "AShareCostModel | None" = None,
) -> pd.DataFrame:
    """FIFO-match buy/sell fills per symbol into closed round-trip trades.

    Slippage is already baked into ``avg_price`` by the fill simulator; the
    explicit commission / stamp / transfer fees are re-derived here with the
    engine's own :class:`AShareCostModel` (single source of truth) and
    charged to the matched quantity, so ``net_pnl`` is a faithful realised
    PnL per closed trade. Lots still open at the end are left unrealised and
    excluded — the NAV already reflects their mark-to-market.
    """
    from collections import deque

    from quantagent.execution.broker_base import OrderSide
    from quantagent.execution.cost_model import AShareCostModel

    cols = ["symbol", "buy_date", "sell_date", "quantity", "buy_price",
            "sell_price", "gross_pnl", "cost", "net_pnl"]
    if order_audit is None or order_audit.empty or "side" not in order_audit.columns:
        return pd.DataFrame(columns=cols)
    cm = cost_model or AShareCostModel()
    f = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0].copy()
    if f.empty:
        return pd.DataFrame(columns=cols)
    f["side"] = f["side"].astype(str).str.lower()
    f["filled_quantity"] = f["filled_quantity"].astype(float).abs()
    f["avg_price"] = pd.to_numeric(f["avg_price"], errors="coerce")
    f = f.dropna(subset=["avg_price"]).sort_values("trade_date")

    def _fee_per_share(side: str, qty: float, price: float) -> float:
        if qty <= 0:
            return 0.0
        try:
            total = cm.calculate(OrderSide(side), int(qty), float(price))["total"]
        except Exception:  # noqa: BLE001 — unknown side ⇒ no fee rather than crash
            return 0.0
        return total / qty

    trades: list[dict] = []
    for sym, g in f.groupby("symbol"):
        lots: deque[list[float]] = deque()  # [qty, price, date, buy_fee_per_share]
        for _, r in g.iterrows():
            qty, price, date, side = r["filled_quantity"], r["avg_price"], r["trade_date"], r["side"]
            if side == "buy":
                lots.append([qty, price, date, _fee_per_share("buy", qty, price)])
            elif side == "sell":
                sell_fee_ps = _fee_per_share("sell", qty, price)
                remaining = qty
                while remaining > 1e-9 and lots:
                    lot = lots[0]
                    matched = min(remaining, lot[0])
                    gross = matched * (price - lot[1])
                    cost = matched * lot[3] + matched * sell_fee_ps
                    trades.append({
                        "symbol": sym, "buy_date": lot[2], "sell_date": date,
                        "quantity": matched, "buy_price": lot[1], "sell_price": price,
                        "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost,
                    })
                    lot[0] -= matched
                    remaining -= matched
                    if lot[0] <= 1e-9:
                        lots.popleft()
    return pd.DataFrame(trades, columns=cols)


def _compute_metrics(
    nav: pd.Series,
    order_audit: pd.DataFrame,
    *,
    periods: int = 252,
) -> StrictBacktestMetrics:
    if nav is None or nav.empty:
        return StrictBacktestMetrics(
            total_return=0.0, annualized_return=0.0, max_drawdown=0.0,
            sharpe=0.0, calmar=0.0, volatility=0.0, turnover=0.0,
            win_rate=0.0, avg_profit_per_trade=0.0, median_profit_per_trade=0.0,
            profit_factor=0.0, gross_profit=0.0, gross_loss=0.0, total_cost=0.0,
            n_trades=0, n_fills=0, start_date="", end_date="",
        )
    nav_clean = nav.sort_index().dropna()
    daily_ret = nav_clean.pct_change().dropna()
    n_obs = max(1, len(daily_ret))
    elapsed_calendar_days = max(1, int((nav_clean.index[-1] - nav_clean.index[0]).days))
    elapsed_years = max(elapsed_calendar_days / 365.25, 1.0 / periods)
    elapsed_trading_days = max(1, len(pd.bdate_range(nav_clean.index[0], nav_clean.index[-1])) - 1)
    obs_per_year = n_obs / elapsed_years
    total_return = float(nav_clean.iloc[-1] / nav_clean.iloc[0] - 1.0) if len(nav_clean) >= 2 else 0.0
    ann_return = float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 else 0.0
    if len(daily_ret) >= 2:
        std = float(daily_ret.std(ddof=1))
        sharpe = float(daily_ret.mean() / std * (obs_per_year ** 0.5)) if std > 1e-12 else 0.0
        vol = float(std * (obs_per_year ** 0.5))
    else:
        sharpe = 0.0
        vol = 0.0
    nav_curve = nav_clean.values
    peaks = np.maximum.accumulate(nav_curve)
    drawdowns = nav_curve / peaks - 1.0
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) else 0.0
    calmar = float(ann_return / max_dd) if max_dd > 1e-9 else float(ann_return * 10.0)
    # Turnover proxy: average daily traded value / NAV.
    n_fills = 0
    turnover = 0.0
    if order_audit is not None and not order_audit.empty and "filled_quantity" in order_audit.columns:
        filled = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0]
        if not filled.empty and "avg_price" in filled.columns:
            turnover_value = float(
                (filled["filled_quantity"].astype(float).abs() * filled["avg_price"].astype(float)).sum()
            )
            nav_mean = float(nav_clean.mean()) if not nav_clean.empty else 0.0
            turnover = turnover_value / max(1.0, nav_mean) / max(1, elapsed_trading_days)
            n_fills = int(len(filled))

    # Realised round-trip trade statistics (FIFO-matched, net of cost).
    rt = _realized_round_trip_pnl(order_audit)
    if not rt.empty:
        net = rt["net_pnl"].astype(float)
        n_trades = int(len(net))
        win_rate = float((net > 0).mean())
        avg_profit = float(net.mean())
        median_profit = float(net.median())
        gross_profit = float(net[net > 0].sum())
        gross_loss = float(-net[net < 0].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-9 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        total_cost = float(rt["cost"].astype(float).sum())
    else:
        n_trades = 0
        win_rate = avg_profit = median_profit = 0.0
        gross_profit = gross_loss = profit_factor = total_cost = 0.0

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
        median_profit_per_trade=median_profit,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_cost=total_cost,
        n_trades=n_trades,
        n_fills=n_fills,
        start_date=str(nav_clean.index[0]) if len(nav_clean) else "",
        end_date=str(nav_clean.index[-1]) if len(nav_clean) else "",
    )


def _profit_by_stock(realized_trades: pd.DataFrame, order_audit: pd.DataFrame) -> pd.DataFrame:
    """Per-stock realised PnL attribution using the same FIFO trade ledger.

    Earlier v8 reports used signed traded value as ``pnl_proxy``. That was
    useful for checking cash-flow direction, but it is not a PnL measure and
    can mis-rank stocks when a book still has open lots. The report now uses
    closed round-trip trades net of engine costs; the legacy ``pnl_proxy``
    column is retained as an alias for compatibility with existing readers.
    """
    cols = [
        "symbol", "n_trades", "n_fills", "quantity", "gross_pnl", "cost",
        "net_pnl", "win_rate", "avg_profit_per_trade", "pnl_proxy",
    ]
    if realized_trades is None or realized_trades.empty:
        if order_audit is None or order_audit.empty:
            return pd.DataFrame(columns=cols)
        filled = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0]
        if filled.empty:
            return pd.DataFrame(columns=cols)
        grp = filled.groupby("symbol")
        out = pd.DataFrame({
            "symbol": grp["symbol"].first(),
            "n_trades": 0,
            "n_fills": grp.size(),
            "quantity": 0.0,
            "gross_pnl": 0.0,
            "cost": 0.0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "pnl_proxy": 0.0,
        }).reset_index(drop=True)
        return out.sort_values("net_pnl", ascending=False)

    rt = realized_trades.copy()
    rt["net_pnl"] = pd.to_numeric(rt["net_pnl"], errors="coerce").fillna(0.0)
    rt["gross_pnl"] = pd.to_numeric(rt["gross_pnl"], errors="coerce").fillna(0.0)
    rt["cost"] = pd.to_numeric(rt["cost"], errors="coerce").fillna(0.0)
    rt["quantity"] = pd.to_numeric(rt["quantity"], errors="coerce").fillna(0.0)
    grp = rt.groupby("symbol")
    out = pd.DataFrame({
        "symbol": grp["symbol"].first(),
        "n_trades": grp.size(),
        "quantity": grp["quantity"].sum(),
        "gross_pnl": grp["gross_pnl"].sum(),
        "cost": grp["cost"].sum(),
        "net_pnl": grp["net_pnl"].sum(),
        "win_rate": grp["net_pnl"].apply(lambda s: float((s > 0).mean()) if len(s) else 0.0),
        "avg_profit_per_trade": grp["net_pnl"].mean(),
    }).reset_index(drop=True)

    if order_audit is not None and not order_audit.empty:
        filled = order_audit[order_audit["filled_quantity"].astype(float).abs() > 0]
        fills = (
            filled.groupby("symbol").size().rename("n_fills").reset_index()
            if not filled.empty else pd.DataFrame(columns=["symbol", "n_fills"])
        )
        out = out.merge(fills, on="symbol", how="left")
    else:
        out["n_fills"] = 0
    out["n_fills"] = out["n_fills"].fillna(0).astype(int)
    out["pnl_proxy"] = out["net_pnl"]
    return out[cols].sort_values("net_pnl", ascending=False).reset_index(drop=True)


def _profit_by_sector(
    by_stock: pd.DataFrame,
    sector_map: pd.DataFrame | None,
) -> pd.DataFrame:
    if by_stock is None or by_stock.empty:
        return pd.DataFrame(columns=["sector_level_1", "n_trades", "n_fills", "gross_pnl", "cost", "net_pnl", "pnl_proxy"])
    if sector_map is None or sector_map.empty or "sector_level_1" not in sector_map.columns:
        return pd.DataFrame(columns=["sector_level_1", "n_trades", "n_fills", "gross_pnl", "cost", "net_pnl", "pnl_proxy"])
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
        .agg(
            n_trades=("n_trades", "sum"),
            n_fills=("n_fills", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            cost=("cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            pnl_proxy=("pnl_proxy", "sum"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
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
    realized_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_weights: dict[str, float] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)
    # Set when the simulated window overlaps a quarantined holdout: merged into
    # metrics.json so direct callers cannot emit trusted-looking numbers there.
    trust_stamp: dict[str, object] | None = None

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["metrics"] = out / "metrics.json"
        metrics_payload = self.metrics.to_dict()
        if self.trust_stamp:
            metrics_payload.update(self.trust_stamp)
        paths["metrics"].write_text(
            json.dumps(metrics_payload, indent=2, default=str),
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
        paths["realized_trades"] = out / "realized_trades.csv"
        self.realized_trades.to_csv(paths["realized_trades"], index=False)
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
    # Non-blocking quarantine check (P-G): warn + stamp, never alter execution.
    trust_stamp = quarantine_trust_stamp(
        pd.to_datetime(target_weights.index) if target_weights is not None and len(target_weights) else None
    )
    if trust_stamp:
        import sys as _sys
        print(
            f"[strict_v8] QUARANTINE WARNING: simulated window overlaps quarantined "
            f"holdout {trust_stamp['quarantine_window']} — results are "
            f"{trust_stamp['trust_class']}, NOT trusted evaluation evidence "
            f"(see {trust_stamp['quarantine_evidence']}).",
            file=_sys.stderr, flush=True,
        )
    sim: AShareExecutionSimulationResult = simulate_ashare_target_weights(
        target_weights, market_panel, cfg,
    )
    metrics = _compute_metrics(sim.nav, sim.order_audit)
    realized_trades = _realized_round_trip_pnl(sim.order_audit)
    by_stock = _profit_by_stock(realized_trades, sim.order_audit)
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
        realized_trades=realized_trades,
        factor_weights=dict(factor_weights or {}),
        config=dict(sim.config or {}),
        trust_stamp=trust_stamp,
    )


__all__ = [
    "StrictBacktestArtifactSet",
    "StrictBacktestMetrics",
    "run_strict_backtest_v8",
]
