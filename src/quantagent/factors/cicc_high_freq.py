from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.factors.registry import FactorMeta, default_registry

DAILY_SCHEMA = ("trade_date", "symbol", "open", "high", "low", "close", "volume", "amount")
INTRADAY_SCHEMA = ("trade_date", "datetime", "symbol", "open", "high", "low", "close", "volume", "amount")

DAILY_COMPATIBLE_FACTORS = (
    "last_30min_return",
    "daily_amihud",
    "close_volume_corr",
    "lead_lag_price_volume_corr",
    "turnover_concentration",
    "amount_zscore",
    "money_flow_strength",
    "opening_flow_ratio",
    "closing_flow_ratio",
)

INTRADAY_ONLY_FACTORS = (
    "top_volume_bar_return",
    "intraday_skew",
    "intraday_kurtosis",
    "amihud_1min",
    "crowding_fft_ratio",
)


@dataclass(frozen=True)
class CICCFactorResult:
    factors: pd.DataFrame
    unavailable: tuple[str, ...]


def compute_cicc_high_freq_factors(frame: pd.DataFrame, window: int = 20) -> CICCFactorResult:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
        factors = _compute_intraday(data, window)
        unavailable: tuple[str, ...] = ()
    else:
        factors = _compute_daily(data, window)
        unavailable = INTRADAY_ONLY_FACTORS
    return CICCFactorResult(factors=factors.sort_values(["factor_name", "trade_date", "symbol"]).reset_index(drop=True), unavailable=unavailable)


def last_30min_return(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "last_30min_return")


def daily_amihud(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "daily_amihud")


def close_volume_corr(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "close_volume_corr")


def lead_lag_price_volume_corr(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "lead_lag_price_volume_corr")


def turnover_concentration(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "turnover_concentration")


def amount_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "amount_zscore")


def money_flow_strength(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "money_flow_strength")


def opening_flow_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "opening_flow_ratio")


def closing_flow_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    result = compute_cicc_high_freq_factors(frame)
    return _select(result.factors, "closing_flow_ratio")


def _compute_daily(data: pd.DataFrame, window: int) -> pd.DataFrame:
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    prev_close = data.groupby("symbol", sort=False)["close"].shift(1)
    returns = data["close"] / prev_close - 1.0
    vwap = (data["amount"] / data["volume"].replace(0.0, np.nan)).fillna(data["close"])
    volume_change = data.groupby("symbol", sort=False)["volume"].pct_change()
    amount_mean = data.groupby("symbol", sort=False)["amount"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    volume_sum = data.groupby("symbol", sort=False)["volume"].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    rows = [
        _make_factor(data, "last_30min_return", data["close"] / vwap.replace(0.0, np.nan) - 1.0),
        _make_factor(data, "daily_amihud", returns.abs() / (data["amount"].replace(0.0, np.nan) / 1e8)),
        _make_factor(data, "close_volume_corr", _rolling_corr(data, returns, volume_change, window)),
        _make_factor(data, "lead_lag_price_volume_corr", _rolling_corr(data, returns, volume_change.groupby(data["symbol"], sort=False).shift(1), window)),
        _make_factor(data, "turnover_concentration", data["volume"] / volume_sum.replace(0.0, np.nan)),
        _make_factor(data, "amount_zscore", _date_zscore(data, data["amount"])),
        _make_factor(data, "money_flow_strength", ((data["close"] - data["open"]) / (data["high"] - data["low"]).replace(0.0, np.nan)) * (data["amount"] / amount_mean.replace(0.0, np.nan))),
        _make_factor(data, "opening_flow_ratio", ((data["open"] / prev_close) - 1.0) * (data["amount"] / amount_mean.replace(0.0, np.nan))),
        _make_factor(data, "closing_flow_ratio", ((data["close"] / vwap.replace(0.0, np.nan)) - 1.0) * (data["amount"] / amount_mean.replace(0.0, np.nan))),
    ]
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _compute_intraday(data: pd.DataFrame, window: int) -> pd.DataFrame:
    daily = _daily_from_intraday(data)
    daily_factors = _compute_daily(daily, window)
    daily_factors = daily_factors[~daily_factors["factor_name"].isin({"opening_flow_ratio", "closing_flow_ratio"})]
    factors = [daily_factors]
    data = data.sort_values(["symbol", "trade_date", "datetime"]).reset_index(drop=True)
    data["bar_return"] = data.groupby(["symbol", "trade_date"], sort=False)["close"].pct_change().fillna(data["close"] / data["open"] - 1.0)
    grouped = data.groupby(["trade_date", "symbol"], sort=False)
    top_bar = grouped.apply(lambda g: g.loc[g["volume"].idxmax(), "close"] / g.loc[g["volume"].idxmax(), "open"] - 1.0, include_groups=False)
    skew = grouped["bar_return"].skew()
    kurtosis = grouped["bar_return"].apply(lambda s: s.kurt())
    amihud = grouped.apply(lambda g: (g["bar_return"].abs() / (g["amount"].replace(0.0, np.nan) / 1e6)).mean(), include_groups=False)
    fft_ratio = grouped["volume"].apply(_fft_crowding_ratio)
    for name, series in {
        "top_volume_bar_return": top_bar,
        "intraday_skew": skew,
        "intraday_kurtosis": kurtosis,
        "amihud_1min": amihud,
        "crowding_fft_ratio": fft_ratio,
        "opening_flow_ratio": grouped.apply(lambda g: _edge_amount_ratio(g, head=True), include_groups=False),
        "closing_flow_ratio": grouped.apply(lambda g: _edge_amount_ratio(g, head=False), include_groups=False),
    }.items():
        frame = series.rename("factor_value").reset_index()
        frame["factor_name"] = name
        factors.append(frame[["trade_date", "symbol", "factor_name", "factor_value"]])
    return pd.concat(factors, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _daily_from_intraday(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.sort_values(["symbol", "trade_date", "datetime"]).groupby(["trade_date", "symbol"], sort=False)
    return grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()


def _make_factor(data: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": data["trade_date"].to_numpy(),
            "symbol": data["symbol"].to_numpy(),
            "factor_name": name,
            "factor_value": values.to_numpy(dtype=float),
        }
    )


def _rolling_corr(data: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    left = pd.Series(left.to_numpy(dtype=float), index=data.index)
    right = pd.Series(right.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = left.loc[group.index].rolling(window, min_periods=window).corr(right.loc[group.index])
    return values


def _date_zscore(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    values = pd.Series(values.to_numpy(dtype=float), index=data.index)

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - s.mean()) / std

    return values.groupby(data["trade_date"], sort=False).transform(_z)


def _fft_crowding_ratio(volume: pd.Series) -> float:
    clean = volume.fillna(0.0).to_numpy(dtype=float)
    if clean.size < 8 or clean.sum() <= 0:
        return np.nan
    spectrum = np.abs(np.fft.rfft(clean - clean.mean())) ** 2
    if spectrum[1:].sum() <= 0:
        return 0.0
    return float(spectrum[1:3].sum() / spectrum[1:].sum())


def _edge_amount_ratio(group: pd.DataFrame, head: bool, bars: int = 30) -> float:
    ordered = group.sort_values("datetime")
    edge = ordered.head(bars) if head else ordered.tail(bars)
    total = ordered["amount"].sum()
    if total <= 0:
        return np.nan
    return float(edge["amount"].sum() / total)


def _select(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame.loc[frame["factor_name"] == name].reset_index(drop=True)


for _name in DAILY_COMPATIBLE_FACTORS:
    default_registry.add(
        FactorMeta(
            name=f"cicc_{_name}",
            category="cicc_high_freq",
            horizon_days=5,
            required_columns=tuple(column for column in DAILY_SCHEMA if column not in {"trade_date", "symbol"}),
            direction=1,
            description=f"CICC-style A-share microstructure factor: {_name}.",
            source="CICC-inspired daily/minute-compatible A-share factor approximation",
        ),
        globals()[_name],
    )
