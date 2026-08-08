"""Fuyao / Financial-API best-practice strategy recipes 13-15.

These functions stop at *T-day target weights and research diagnostics*.  They
never simulate fills themselves.  QuantAgent's existing A-share execution
simulator is the single source of truth for T+1 open fills, price limits,
suspensions, lot size, slippage/impact, costs and the canonical ledger.

Input prices must already be a PIT-safe, consistently adjusted daily panel.  A
raw market dump may only be used after corporate-action adjustment has been
applied; the recipes deliberately fail closed when ``adjustment='none'`` is
visible in the input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


Weighting = Literal["equal", "inverse_volatility"]
Rebalance = Literal["day", "week", "month"]


@dataclass(frozen=True)
class BreakoutConfig:
    breakout_window: int = 55
    exit_window: int = 20
    volume_window: int = 20
    volume_ratio_min: float = 1.5
    ma_window: int = 60
    max_positions: int = 20


@dataclass(frozen=True)
class MomentumConfig:
    momentum_window: int = 120
    ma_window: int = 120
    volatility_window: int = 60
    weighting: Weighting = "inverse_volatility"
    rebalance: Rebalance = "week"
    max_weight_per_asset: float = 0.35


@dataclass(frozen=True)
class ReversalConfig:
    formation_days: int = 5
    holding_days: int = 5
    cooldown_days: int = 5
    bottom_quantile: float = 0.10
    ma_window: int = 120
    liquidity_window: int = 20
    min_amount: float = 0.0
    abnormal_drop_threshold: float = -0.095
    max_positions: int = 50


@dataclass(frozen=True)
class RecipeResult:
    strategy: str
    target_weights: pd.DataFrame
    diagnostics: dict[str, object]
    signal_frame: pd.DataFrame
    config: dict[str, object]


def price_volume_breakout_weights(
    panel: pd.DataFrame,
    config: BreakoutConfig | None = None,
) -> RecipeResult:
    """Official example 13: 55d breakout + volume confirmation + MA60.

    Signal inputs are all known at T close.  ``prior_high`` and ``exit_low`` are
    shifted one session so the current bar never helps define its own breakout
    or exit threshold.  The returned T-day weights must be fed to the existing
    next-day-fill execution simulator.
    """

    config = config or BreakoutConfig()
    data = _prepare_panel(panel)
    _require_adjusted_if_declared(data)
    pieces: list[pd.DataFrame] = []
    for symbol, group in data.groupby("symbol", sort=False):
        g = group.sort_values("trade_date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce")
        g["prior_high"] = close.rolling(config.breakout_window, min_periods=config.breakout_window).max().shift(1)
        g["exit_low"] = close.rolling(config.exit_window, min_periods=config.exit_window).min().shift(1)
        g["ma"] = close.rolling(config.ma_window, min_periods=config.ma_window).mean()
        prior_volume = volume.rolling(config.volume_window, min_periods=config.volume_window).mean().shift(1)
        g["volume_ratio"] = volume / prior_volume.replace(0, np.nan)
        g["entry_signal"] = (
            (close > g["prior_high"])
            & (g["volume_ratio"] >= config.volume_ratio_min)
            & (close > g["ma"])
        )
        g["exit_signal"] = close < g["exit_low"]
        g["breakout_strength"] = close / g["prior_high"] - 1.0
        pieces.append(g)
    signals = pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "symbol"])

    active: set[str] = set()
    daily: dict[pd.Timestamp, dict[str, float]] = {}
    false_breakouts = 0
    prior_entries: dict[str, pd.Timestamp] = {}
    for date, group in signals.groupby("trade_date", sort=True):
        exiting = set(group.loc[group["exit_signal"].fillna(False), "symbol"].astype(str))
        for symbol in exiting:
            if symbol in active:
                entry = prior_entries.get(symbol)
                if entry is not None and (pd.Timestamp(date) - entry).days <= 10:
                    false_breakouts += 1
                active.discard(symbol)
                prior_entries.pop(symbol, None)

        candidates = group.loc[group["entry_signal"].fillna(False)].copy()
        candidates = candidates.sort_values(["volume_ratio", "breakout_strength"], ascending=False)
        capacity = max(0, config.max_positions - len(active))
        for symbol in candidates["symbol"].astype(str):
            if capacity <= 0:
                break
            if symbol not in active:
                active.add(symbol)
                prior_entries[symbol] = pd.Timestamp(date)
                capacity -= 1
        weight = 1.0 / len(active) if active else 0.0
        daily[pd.Timestamp(date)] = {symbol: weight for symbol in active}

    weights = _dict_weights(daily, signals["symbol"].unique())
    diagnostics = {
        "entryCount": int(signals["entry_signal"].fillna(False).sum()),
        "exitCount": int(signals["exit_signal"].fillna(False).sum()),
        "falseBreakoutApproxCount": int(false_breakouts),
        "signalTiming": "T close",
        "executionTiming": "T+1 open via QuantAgent A-share simulator",
        "corporateActionBoundary": "input must be consistently adjusted; raw dump alone is invalid",
        "parameterSensitivityRequired": True,
    }
    return RecipeResult("price_volume_breakout", weights, diagnostics, signals, asdict(config))


def time_series_momentum_weights(
    panel: pd.DataFrame,
    config: MomentumConfig | None = None,
) -> RecipeResult:
    """Official example 14: 120d own momentum + MA120 + 60d volatility."""

    config = config or MomentumConfig()
    data = _prepare_panel(panel)
    _require_adjusted_if_declared(data)
    pieces: list[pd.DataFrame] = []
    for symbol, group in data.groupby("symbol", sort=False):
        g = group.sort_values("trade_date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        ret = close.pct_change()
        g["momentum"] = close / close.shift(config.momentum_window) - 1.0
        g["ma"] = close.rolling(config.ma_window, min_periods=config.ma_window).mean()
        g["volatility"] = ret.rolling(config.volatility_window, min_periods=config.volatility_window).std(ddof=1)
        g["active"] = (g["momentum"] > 0) & (close > g["ma"]) & g["volatility"].notna()
        pieces.append(g)
    signals = pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "symbol"])

    dates = pd.DatetimeIndex(sorted(signals["trade_date"].dropna().unique()))
    rebalance_dates = _rebalance_dates(dates, config.rebalance)
    last: dict[str, float] = {}
    daily: dict[pd.Timestamp, dict[str, float]] = {}
    cash_days = 0
    for date in dates:
        if pd.Timestamp(date) in rebalance_dates:
            group = signals[signals["trade_date"] == pd.Timestamp(date)]
            active = group[group["active"].fillna(False)].copy()
            if active.empty:
                last = {}
            elif config.weighting == "equal":
                raw = pd.Series(1.0, index=active["symbol"].astype(str))
                last = _cap_and_normalise(raw, config.max_weight_per_asset)
            else:
                vol = pd.Series(
                    pd.to_numeric(active["volatility"], errors="coerce").to_numpy(),
                    index=active["symbol"].astype(str),
                )
                inv = 1.0 / vol.replace(0, np.nan)
                last = _cap_and_normalise(inv.dropna(), config.max_weight_per_asset)
        if not last:
            cash_days += 1
        daily[pd.Timestamp(date)] = dict(last)

    weights = _dict_weights(daily, signals["symbol"].unique())
    diagnostics = {
        "cashStateDays": int(cash_days),
        "cashStateShare": float(cash_days / max(1, len(dates))),
        "signalTiming": "T close",
        "executionTiming": "T+1 open via QuantAgent A-share simulator",
        "rebalance": config.rebalance,
        "weighting": config.weighting,
        "cashBoundary": "no active assets means 100% cash, not missing market data",
        "windowSensitivityRequired": True,
        "riskContributionRequired": True,
    }
    return RecipeResult("time_series_momentum", weights, diagnostics, signals, asdict(config))


def short_term_reversal_weights(
    panel: pd.DataFrame,
    benchmark: pd.Series,
    config: ReversalConfig | None = None,
) -> RecipeResult:
    """Official example 15: bottom-decile 5d relative reversal with 5d hold/cooldown."""

    config = config or ReversalConfig()
    data = _prepare_panel(panel)
    _require_adjusted_if_declared(data)
    benchmark = pd.Series(benchmark, dtype=float).copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark = benchmark.sort_index()
    benchmark_ret = benchmark.pct_change(config.formation_days)

    pieces: list[pd.DataFrame] = []
    for symbol, group in data.groupby("symbol", sort=False):
        g = group.sort_values("trade_date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        amount = pd.to_numeric(g.get("amount", pd.Series(index=g.index, dtype=float)), errors="coerce")
        daily_ret = close.pct_change()
        g["stock_return_formation"] = close.pct_change(config.formation_days)
        g["benchmark_return_formation"] = g["trade_date"].map(benchmark_ret)
        g["relative_return"] = g["stock_return_formation"] - g["benchmark_return_formation"]
        g["ma"] = close.rolling(config.ma_window, min_periods=config.ma_window).mean()
        g["amount_mean"] = amount.rolling(config.liquidity_window, min_periods=config.liquidity_window).mean()
        g["daily_return"] = daily_ret
        g["eligible"] = (
            (close > g["ma"])
            & (g["amount_mean"] >= config.min_amount)
            & (daily_ret > config.abnormal_drop_threshold)
            & g["relative_return"].notna()
        )
        g["future_return"] = close.shift(-config.holding_days) / close - 1.0
        g["future_benchmark_return"] = g["trade_date"].map(benchmark.pct_change(config.holding_days).shift(-config.holding_days))
        g["future_relative_return"] = g["future_return"] - g["future_benchmark_return"]
        pieces.append(g)
    signals = pd.concat(pieces, ignore_index=True).sort_values(["trade_date", "symbol"])

    selected_by_date: dict[pd.Timestamp, list[str]] = {}
    decile_rows: list[dict[str, object]] = []
    daily_ic: dict[pd.Timestamp, float] = {}
    for date, group in signals.groupby("trade_date", sort=True):
        eligible = group[group["eligible"].fillna(False)].copy()
        if eligible.empty:
            selected_by_date[pd.Timestamp(date)] = []
            continue
        eligible["bucket"] = pd.qcut(
            eligible["relative_return"].rank(method="first"),
            q=min(10, len(eligible)),
            labels=False,
            duplicates="drop",
        )
        cutoff = eligible["relative_return"].quantile(config.bottom_quantile)
        selected = eligible[eligible["relative_return"] <= cutoff].nsmallest(config.max_positions, "relative_return")
        selected_by_date[pd.Timestamp(date)] = selected["symbol"].astype(str).tolist()
        if eligible["future_relative_return"].notna().sum() >= 5:
            ic = eligible["relative_return"].rank().corr(eligible["future_relative_return"].rank())
            if pd.notna(ic):
                daily_ic[pd.Timestamp(date)] = float(ic)
        for bucket, bucket_frame in eligible.groupby("bucket", dropna=True):
            decile_rows.append({
                "trade_date": pd.Timestamp(date),
                "bucket": int(bucket),
                "mean_future_relative_return": float(bucket_frame["future_relative_return"].mean()) if bucket_frame["future_relative_return"].notna().any() else np.nan,
                "count": int(len(bucket_frame)),
            })

    dates = pd.DatetimeIndex(sorted(signals["trade_date"].dropna().unique()))
    active_until: dict[str, int] = {}
    cooldown_until: dict[str, int] = {}
    daily: dict[pd.Timestamp, dict[str, float]] = {}
    for index, date in enumerate(dates):
        expired = [symbol for symbol, until in active_until.items() if index >= until]
        for symbol in expired:
            active_until.pop(symbol, None)
            cooldown_until[symbol] = index + config.cooldown_days
        for symbol in selected_by_date.get(pd.Timestamp(date), []):
            if symbol in active_until:
                continue
            if index < cooldown_until.get(symbol, -1):
                continue
            active_until[symbol] = index + config.holding_days
        active = sorted(active_until)
        weight = 1.0 / len(active) if active else 0.0
        daily[pd.Timestamp(date)] = {symbol: weight for symbol in active}

    weights = _dict_weights(daily, signals["symbol"].unique())
    ic_series = pd.Series(daily_ic, dtype=float).sort_index()
    mean_ic = float(ic_series.mean()) if not ic_series.empty else float("nan")
    reversal_evidence = bool(np.isfinite(mean_ic) and mean_ic < 0)
    diagnostics = {
        "rankIcMean": None if not np.isfinite(mean_ic) else mean_ic,
        "rankIcExpectedSign": "negative",
        "reversalEvidence": reversal_evidence,
        "evidenceMessage": (
            "negative Rank IC is consistent with reversal"
            if reversal_evidence
            else "sample is momentum-like or evidence is insufficient; do not claim reversal"
        ),
        "deciles": _json_records(pd.DataFrame(decile_rows)),
        "signalTiming": "T close",
        "executionTiming": "T+1 open via QuantAgent A-share simulator",
        "holdingDays": config.holding_days,
        "cooldownDays": config.cooldown_days,
        "formationHoldingSensitivityRequired": True,
    }
    return RecipeResult("short_term_reversal", weights, diagnostics, signals, asdict(config))


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"daily panel missing required columns: {missing}")
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str)
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    if data.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("daily panel contains duplicate (symbol, trade_date) rows")
    return data


def _require_adjusted_if_declared(data: pd.DataFrame) -> None:
    if "adjustment" not in data.columns:
        return
    values = set(data["adjustment"].dropna().astype(str).str.lower())
    if values and values <= {"none", "raw", "unadjusted"}:
        raise ValueError("strategy recipe requires a corporate-action-adjusted price series")


def _dict_weights(daily: dict[pd.Timestamp, dict[str, float]], symbols: object) -> pd.DataFrame:
    columns = sorted({str(symbol) for symbol in symbols})
    frame = pd.DataFrame.from_dict(daily, orient="index").reindex(columns=columns).fillna(0.0)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "trade_date"
    return frame.sort_index()


def _rebalance_dates(dates: pd.DatetimeIndex, mode: Rebalance) -> set[pd.Timestamp]:
    if mode == "day":
        return {pd.Timestamp(date) for date in dates}
    frame = pd.DataFrame({"date": dates})
    if mode == "week":
        key = frame["date"].dt.to_period("W-FRI")
    elif mode == "month":
        key = frame["date"].dt.to_period("M")
    else:
        raise ValueError("rebalance must be day/week/month")
    # Last observed session of each period. Its close determines the next open.
    return {pd.Timestamp(value) for value in frame.groupby(key)["date"].max().tolist()}


def _cap_and_normalise(raw: pd.Series, cap: float) -> dict[str, float]:
    weights = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights[weights > 0]
    if weights.empty:
        return {}
    weights = weights / weights.sum()
    if cap <= 0 or cap >= 1:
        return {str(k): float(v) for k, v in weights.items()}
    # Iterative water-fill so the final result respects the per-asset ceiling.
    capped = pd.Series(0.0, index=weights.index)
    remaining = weights.copy()
    budget = 1.0
    while not remaining.empty and budget > 1e-12:
        proposal = remaining / remaining.sum() * budget
        over = proposal > cap
        if not over.any():
            capped.loc[proposal.index] = proposal
            break
        capped.loc[proposal[over].index] = cap
        budget = 1.0 - float(capped.sum())
        remaining = remaining.drop(index=proposal[over].index)
    total = float(capped.sum())
    if total > 1.0 + 1e-9:
        capped /= total
    return {str(k): float(v) for k, v in capped[capped > 0].items()}


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype(str)
    return out.replace({np.nan: None}).to_dict(orient="records")


__all__ = [
    "BreakoutConfig",
    "MomentumConfig",
    "RecipeResult",
    "ReversalConfig",
    "price_volume_breakout_weights",
    "short_term_reversal_weights",
    "time_series_momentum_weights",
]
