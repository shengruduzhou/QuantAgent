from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.quant_api.services.market_playbooks import _finite, _perf, _records
from services.quant_api.services.market_playbooks_v2 import (
    MarketPlaybookService as _V2MarketPlaybookService,
    PLAYBOOKS,
)


class MarketPlaybookService(_V2MarketPlaybookService):
    """Fuyao playbooks with the official 13/14/15 execution recipes hardened.

    The V2 service already covers all 16 workbenches and the Financial-API data
    plumbing.  This layer only overrides the three strategy examples whose old
    implementation used simplified research parameters.
    """

    def price_volume_breakout(
        self,
        symbol: str,
        benchmark: str,
        cost_bps: float,
    ) -> dict[str, Any]:
        """Playbook 13: 55d breakout + volume + MA60, 20d-low exit, T+1 open."""
        bars = self._stock_bars(symbol, 900)
        close = bars["close"].astype(float)
        volume = bars["volume"].astype(float)
        prior_high_55 = close.rolling(55, min_periods=55).max().shift(1)
        prior_low_20 = close.rolling(20, min_periods=20).min().shift(1)
        avg_volume_20 = volume.rolling(20, min_periods=20).mean().shift(1)
        ma60 = close.rolling(60, min_periods=60).mean()
        volume_ratio = volume / avg_volume_20.replace(0.0, np.nan)

        entry = (
            (close > prior_high_55)
            & (volume_ratio >= 1.5)
            & (close > ma60)
        ).fillna(False)
        exit_signal = (close < prior_low_20).fillna(False)
        desired = pd.Series(0.0, index=bars.index)
        active = False
        events: list[dict[str, Any]] = []
        for i in range(len(bars)):
            date = str(pd.Timestamp(bars["date"].iloc[i]).date())
            if not active and bool(entry.iloc[i]):
                active = True
                events.append(
                    {
                        "signalDate": date,
                        "action": "enter_next_open",
                        "reason": "close > prior 55d high; volume >= 1.5x prior 20d mean; close > MA60",
                    }
                )
            elif active and bool(exit_signal.iloc[i]):
                active = False
                events.append(
                    {
                        "signalDate": date,
                        "action": "exit_next_open",
                        "reason": "close < prior 20d low",
                    }
                )
            desired.iloc[i] = 1.0 if active else 0.0

        out = self._backtest(
            "13",
            bars,
            desired,
            benchmark,
            cost_bps,
            {
                "entry": "close > prior 55d high (today excluded) + volume ratio >= 1.5 + close > MA60",
                "exit": "close < prior 20d low (today excluded)",
                "adjustment": "stock history uses forward-adjusted prices from provider",
            },
        )
        indicators = pd.DataFrame(
            {
                "date": pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d"),
                "priorHigh55": prior_high_55,
                "priorLow20": prior_low_20,
                "ma60": ma60,
                "volumeRatio20": volume_ratio,
                "entrySignal": entry,
                "exitSignal": exit_signal,
            }
        )
        by_date = {str(row["date"]): row for row in _records(indicators)}
        for row in out.get("rows", []):
            extra = by_date.get(str(row.get("date")), {})
            row.update(
                {
                    key: extra.get(key)
                    for key in (
                        "priorHigh55",
                        "priorLow20",
                        "ma60",
                        "volumeRatio20",
                        "entrySignal",
                        "exitSignal",
                    )
                }
            )
        out["events"] = events
        out["formation"] = {
            "breakoutWindow": 55,
            "exitLowWindow": 20,
            "volumeWindow": 20,
            "volumeMultiple": 1.5,
            "trendWindow": 60,
        }
        return out

    @staticmethod
    def _portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
        asset_turn = weights.diff().abs().sum(axis=1)
        cash = 1.0 - weights.sum(axis=1)
        cash_turn = cash.diff().abs()
        turnover = 0.5 * (asset_turn + cash_turn)
        if len(turnover):
            turnover.iloc[0] = 0.5 * (
                float(weights.iloc[0].abs().sum()) + abs(float(cash.iloc[0] - 1.0))
            )
        return turnover.fillna(0.0)

    def _momentum_case(
        self,
        close: pd.DataFrame,
        open_price: pd.DataFrame,
        *,
        window: int,
        cost_bps: float,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        daily_close_return = close.pct_change(fill_method=None)
        momentum = close.pct_change(window, fill_method=None)
        ma = close.rolling(window, min_periods=window).mean()
        vol = daily_close_return.rolling(60, min_periods=40).std(ddof=1) * np.sqrt(252.0)
        active = (momentum > 0) & (close > ma) & vol.gt(0)
        inv_vol = active.astype(float).div(vol.replace(0.0, np.nan))
        target = inv_vol.div(inv_vol.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        executed = target.shift(1).fillna(0.0)
        open_to_open = open_price.shift(-1).div(open_price).sub(1.0)
        gross = (executed * open_to_open).sum(axis=1, min_count=1).fillna(0.0)
        turnover = self._portfolio_turnover(executed)
        net = gross - turnover * float(cost_bps) / 10_000.0
        return net, target, active, vol

    def time_series_momentum(
        self,
        symbol: str,
        benchmark: str,
        cost_bps: float,
    ) -> dict[str, Any]:
        """Playbook 14: multi-asset 120d TSMOM with inverse-vol weights and cash."""
        assets = [item.strip().upper() for item in str(symbol).replace(";", ",").split(",") if item.strip()]
        if not assets:
            raise ValueError("time-series momentum requires at least one index symbol")
        if len(assets) > 12:
            raise ValueError("time-series momentum asset pool is capped at 12 symbols per run")

        close_parts: list[pd.Series] = []
        open_parts: list[pd.Series] = []
        for asset in assets:
            bars = self._index_bars(asset, 1200).set_index("date").sort_index()
            if len(bars) < 200:
                continue
            close_parts.append(bars["close"].astype(float).rename(asset))
            open_parts.append(bars["open"].astype(float).rename(asset))
        if not close_parts:
            raise ValueError("no asset has enough index history for time-series momentum")

        close = pd.concat(close_parts, axis=1).sort_index().ffill()
        open_price = pd.concat(open_parts, axis=1).reindex(close.index).ffill()
        net, target, active, vol = self._momentum_case(
            close,
            open_price,
            window=120,
            cost_bps=cost_bps,
        )
        nav = (1.0 + net).cumprod()
        cash_weight = 1.0 - target.sum(axis=1)
        rows = pd.DataFrame(
            {
                "date": close.index,
                "netReturn": net,
                "nav": nav,
                "cashWeight": cash_weight,
                "activeCount": active.sum(axis=1),
            }
        )

        latest_target = target.iloc[-1] if len(target) else pd.Series(dtype=float)
        latest_vol = vol.iloc[-1] if len(vol) else pd.Series(dtype=float)
        raw_contribution = (latest_target * latest_vol).dropna()
        total_risk = float(raw_contribution.abs().sum())
        risk_contribution = {
            asset: _finite(value / total_risk if total_risk > 0 else 0.0)
            for asset, value in raw_contribution.items()
        }
        sensitivity: dict[str, Any] = {}
        for window in (60, 120, 180):
            case_net, _, _, _ = self._momentum_case(
                close,
                open_price,
                window=window,
                cost_bps=cost_bps,
            )
            sensitivity[str(window)] = _perf(case_net)

        return {
            "playbookId": "14",
            "metrics": _perf(net),
            "rows": _records(rows.tail(520)),
            "assetPool": list(close.columns),
            "latestWeights": {
                str(key): _finite(value) for key, value in latest_target.items()
            },
            "riskContribution": risk_contribution,
            "windowSensitivity": sensitivity,
            "assumptions": {
                "signal": "120d own momentum > 0 AND close > MA120",
                "volatility": "60d annualised close volatility",
                "weighting": "inverse volatility across Active assets",
                "cash": "100% minus active asset weights; no Active asset => 100% cash",
                "execution": "T close target -> T+1 open; open-to-open marking",
                "costBps": cost_bps,
            },
            "benchmark": benchmark.upper(),
        }

    @staticmethod
    def _holding_targets(
        selection: pd.DataFrame,
        *,
        holding_days: int,
        cooldown_days: int,
    ) -> pd.DataFrame:
        target = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)
        remaining = {column: 0 for column in selection.columns}
        cooldown = {column: 0 for column in selection.columns}
        for row_i, (_, row) in enumerate(selection.iterrows()):
            for symbol in selection.columns:
                if remaining[symbol] > 0:
                    target.iat[row_i, target.columns.get_loc(symbol)] = 1.0
                    remaining[symbol] -= 1
                    if remaining[symbol] == 0:
                        cooldown[symbol] = cooldown_days
                    continue
                if cooldown[symbol] > 0:
                    cooldown[symbol] -= 1
                    continue
                if bool(row[symbol]):
                    remaining[symbol] = max(0, holding_days - 1)
                    target.iat[row_i, target.columns.get_loc(symbol)] = 1.0
                    if remaining[symbol] == 0:
                        cooldown[symbol] = cooldown_days
        return target

    def _reversal_case(
        self,
        close: pd.DataFrame,
        open_price: pd.DataFrame,
        volume: pd.DataFrame,
        benchmark_close: pd.Series,
        *,
        formation_days: int,
        holding_days: int,
        cooldown_days: int,
        cost_bps: float,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
        stock_ret_1d = close.pct_change(fill_method=None)
        relative = close.pct_change(formation_days, fill_method=None).sub(
            benchmark_close.pct_change(formation_days, fill_method=None), axis=0
        )
        ma120 = close.rolling(120, min_periods=120).mean()
        liquidity = (close * volume).rolling(20, min_periods=15).mean()
        liquid_cut = liquidity.quantile(0.30, axis=1)
        liquid = liquidity.ge(liquid_cut, axis=0)
        non_extreme = stock_ret_1d.abs() < 0.095
        market_trend = benchmark_close > benchmark_close.rolling(120, min_periods=120).mean()
        eligible = (close > ma120) & liquid & non_extreme
        percentile = relative.rank(axis=1, pct=True, ascending=True)
        selected = (percentile <= 0.10) & eligible & market_trend.to_numpy()[:, None]
        raw_target = self._holding_targets(
            selected.fillna(False),
            holding_days=holding_days,
            cooldown_days=cooldown_days,
        )
        target = raw_target.div(raw_target.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        executed = target.shift(1).fillna(0.0)
        open_to_open = open_price.shift(-1).div(open_price).sub(1.0)
        gross = (executed * open_to_open).sum(axis=1, min_count=1).fillna(0.0)
        turnover = self._portfolio_turnover(executed)
        net = gross - turnover * float(cost_bps) / 10_000.0

        # A close-T score is compared with the first tradable open-to-open
        # interval after T+1 entry, not with the same day's close return.
        future_trade_return = open_to_open.shift(-1)
        score = -relative
        rank_ic = score.corrwith(future_trade_return, axis=1, method="spearman")
        return net, target, score, rank_ic

    def short_term_reversal(
        self,
        index_symbol: str,
        benchmark: str,
        cost_bps: float,
    ) -> dict[str, Any]:
        """Playbook 15: bottom-decile 5d relative reversal with 5d hold/cooldown."""
        provider = self.market._provider()
        constituents = provider.index_constituents(index_symbol.upper())
        source = constituents.get(
            "thscode", constituents.get("symbol", pd.Series(dtype=str))
        )
        symbols = [str(item).upper() for item in source.dropna().tolist()[:80]]
        if len(symbols) < 10:
            raise ValueError("insufficient current constituents for reversal lab")

        close_parts: list[pd.Series] = []
        open_parts: list[pd.Series] = []
        volume_parts: list[pd.Series] = []
        for symbol in symbols:
            try:
                bars = self._stock_bars(symbol, 720).set_index("date").sort_index()
            except Exception:
                continue
            if len(bars) < 160:
                continue
            close_parts.append(bars["close"].astype(float).rename(symbol))
            open_parts.append(bars["open"].astype(float).rename(symbol))
            volume_parts.append(bars["volume"].astype(float).rename(symbol))
        if len(close_parts) < 10:
            raise ValueError("fewer than 10 constituents have sufficient history")

        close = pd.concat(close_parts, axis=1).sort_index()
        open_price = pd.concat(open_parts, axis=1).reindex(close.index)
        volume = pd.concat(volume_parts, axis=1).reindex(close.index)
        benchmark_bars = self._index_bars(benchmark, 720).set_index("date").sort_index()
        benchmark_close = benchmark_bars["close"].astype(float).reindex(close.index).ffill()

        net, target, score, rank_ic = self._reversal_case(
            close,
            open_price,
            volume,
            benchmark_close,
            formation_days=5,
            holding_days=5,
            cooldown_days=5,
            cost_bps=cost_bps,
        )
        nav = (1.0 + net).cumprod()

        future_trade_return = open_price.shift(-1).div(open_price).sub(1.0).shift(-1)
        percentile = score.rank(axis=1, pct=True)
        decile_means: dict[str, float | None] = {}
        for decile in range(1, 11):
            lo = (decile - 1) / 10.0
            hi = decile / 10.0
            mask = (percentile > lo) & (percentile <= hi)
            value = future_trade_return.where(mask).mean(axis=1).mean()
            decile_means[f"D{decile}"] = _finite(value)

        sensitivity: dict[str, Any] = {}
        for formation in (3, 5, 10):
            for holding in (3, 5, 10):
                case_net, _, _, _ = self._reversal_case(
                    close,
                    open_price,
                    volume,
                    benchmark_close,
                    formation_days=formation,
                    holding_days=holding,
                    cooldown_days=holding,
                    cost_bps=cost_bps,
                )
                sensitivity[f"f{formation}_h{holding}"] = _perf(case_net)

        rows = pd.DataFrame(
            {
                "date": close.index,
                "netReturn": net,
                "nav": nav,
                "rankIc": rank_ic.reindex(close.index),
                "selectedCount": (target > 0).sum(axis=1),
                "marketActive": (
                    benchmark_close > benchmark_close.rolling(120, min_periods=120).mean()
                ),
            }
        )
        return {
            "playbookId": "15",
            "metrics": _perf(net),
            "rankIcMean": _finite(rank_ic.mean()),
            "rankIcIr": _finite(
                rank_ic.mean() / rank_ic.std(ddof=1)
                if rank_ic.std(ddof=1) not in {0, np.nan}
                else np.nan
            ),
            "decileMeanReturns": decile_means,
            "sensitivity": sensitivity,
            "rows": _records(rows.tail(520)),
            "assumptions": {
                "formation": "past 5d stock return minus benchmark 5d return",
                "selection": "bottom 10% relative return among eligible names",
                "filters": "close > MA120; 20d traded-value above cross-sectional 30th percentile; abs(1d return) < 9.5%",
                "holding": "fixed 5 trading days followed by 5 trading-day cooldown per name",
                "execution": "T close selection -> T+1 open; open-to-open marking",
                "costBps": cost_bps,
                "constituentCaveat": "current constituents only; historical membership is not backfilled",
            },
            "benchmark": benchmark.upper(),
        }


__all__ = ["MarketPlaybookService", "PLAYBOOKS"]
