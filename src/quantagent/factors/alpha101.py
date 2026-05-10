from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantagent.factors.registry import FactorMeta, default_registry

BASE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def alpha001(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 1)


def alpha002(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 2)


def alpha003(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 3)


def alpha004(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 4)


def alpha005(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 5)


def alpha006(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 6)


def alpha007(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 7)


def alpha008(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 8)


def alpha009(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 9)


def alpha010(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 10)


def alpha011(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 11)


def alpha012(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 12)


def alpha013(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 13)


def alpha014(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 14)


def alpha015(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 15)


def alpha016(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 16)


def alpha017(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 17)


def alpha018(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 18)


def alpha019(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 19)


def alpha020(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 20)


def alpha021(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 21)


def alpha022(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 22)


def alpha023(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 23)


def alpha024(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 24)


def alpha025(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 25)


def alpha026(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 26)


def alpha027(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 27)


def alpha028(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 28)


def alpha029(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 29)


def alpha030(frame: pd.DataFrame) -> pd.DataFrame:
    return _compute_alpha(frame, 30)


def compute_alpha101(frame: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    names = names or [f"alpha{i:03d}" for i in range(1, 31)]
    return default_registry.batch_compute(frame, names=names)


def _compute_alpha(frame: pd.DataFrame, number: int) -> pd.DataFrame:
    data = _base(frame)
    name = f"alpha{number:03d}"
    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    vwap = data["vwap"]
    returns = data["returns"]
    adv20 = _mean(data, "volume", 20)
    delta_close_1 = _delta(data, "close", 1)

    if number == 1:
        candidate = close.where(returns >= 0.0, _std(data, "returns", 20))
        values = -_rank(data, _argmax(data, candidate.pow(2.0), 5))
    elif number == 2:
        values = -_corr(data, _rank(data, _delta(data, "log_volume", 2)), _rank(data, (close - open_) / open_), 6)
    elif number == 3:
        values = -_corr(data, _rank(data, open_), _rank(data, volume), 10)
    elif number == 4:
        values = -_ts_rank(data, _rank(data, low), 9)
    elif number == 5:
        values = _rank(data, open_ - _mean_series(data, vwap, 10)) * -_rank(data, (close - vwap).abs())
    elif number == 6:
        values = -_corr(data, open_, volume, 10)
    elif number == 7:
        move = _delta(data, "close", 7)
        values = pd.Series(-1.0, index=data.index)
        active = adv20 < volume
        values.loc[active] = -_ts_rank(data, move.abs(), 60).loc[active] * np.sign(move.loc[active])
    elif number == 8:
        product = _sum(data, "open", 5) * _sum_series(data, returns, 5)
        values = -_rank(data, product - _delay_series(data, product, 10))
    elif number == 9:
        values = delta_close_1.copy()
        values.loc[_min_series(data, delta_close_1, 5) <= 0.0] = -delta_close_1
        values.loc[_max_series(data, delta_close_1, 5) < 0.0] = delta_close_1
    elif number == 10:
        raw = delta_close_1.copy()
        raw.loc[_min_series(data, delta_close_1, 4) <= 0.0] = -delta_close_1
        raw.loc[_max_series(data, delta_close_1, 4) < 0.0] = delta_close_1
        values = _rank(data, raw)
    elif number == 11:
        values = (_rank(data, _max_series(data, vwap - close, 3)) + _rank(data, _min_series(data, vwap - close, 3))) * _rank(data, _delta(data, "volume", 3))
    elif number == 12:
        values = -np.sign(_delta(data, "volume", 1)) * delta_close_1
    elif number == 13:
        values = -_rank(data, _cov(data, _rank(data, close), _rank(data, volume), 5))
    elif number == 14:
        values = -_rank(data, _delta_series(data, returns, 3)) * _corr(data, open_, volume, 10)
    elif number == 15:
        values = -_sum_series(data, _rank(data, _corr(data, _rank(data, high), _rank(data, volume), 3)), 3)
    elif number == 16:
        values = -_rank(data, _cov(data, _rank(data, high), _rank(data, volume), 5))
    elif number == 17:
        values = -_rank(data, _ts_rank(data, close, 10)) * _rank(data, _delta_series(data, delta_close_1, 1)) * _rank(data, _ts_rank(data, volume / adv20.replace(0.0, np.nan), 5))
    elif number == 18:
        values = -_rank(data, _std_series(data, (close - open_).abs(), 5) + (close - open_) + _corr(data, close, open_, 10))
    elif number == 19:
        trend = close - _delay(data, "close", 7) + _delta(data, "close", 7)
        values = -np.sign(trend) * (1.0 + _rank(data, _sum_series(data, returns, 60)))
    elif number == 20:
        values = -_rank(data, open_ - _delay(data, "high", 1)) * _rank(data, open_ - _delay(data, "close", 1)) * _rank(data, open_ - _delay(data, "low", 1))
    elif number == 21:
        mean8 = _mean(data, "close", 8)
        std8 = _std(data, "close", 8)
        mean2 = _mean(data, "close", 2)
        values = pd.Series(-1.0, index=data.index)
        values.loc[mean8 + std8 < mean2] = -1.0
        values.loc[mean2 < mean8 - std8] = 1.0
        values.loc[(mean2 >= mean8 - std8) & (mean8 + std8 >= mean2) & (volume / adv20.replace(0.0, np.nan) >= 1.0)] = 1.0
    elif number == 22:
        values = -_delta_series(data, _corr(data, high, volume, 5), 5) * _rank(data, _std(data, "close", 20))
    elif number == 23:
        values = pd.Series(0.0, index=data.index)
        active = _mean(data, "high", 20) < high
        values.loc[active] = -_delta(data, "high", 2).loc[active]
    elif number == 24:
        mean20 = _mean(data, "close", 20)
        trend = _delta_series(data, mean20, 20) / _delay_series(data, close, 20).replace(0.0, np.nan)
        values = -(close - _min(data, "close", 10))
        values.loc[trend <= 0.05] = -_delta(data, "close", 3).loc[trend <= 0.05]
    elif number == 25:
        values = _rank(data, (-returns * adv20 * vwap) * (high - close))
    elif number == 26:
        values = -_max_series(data, _corr(data, _ts_rank(data, volume, 5), _ts_rank(data, high, 5), 5), 3)
    elif number == 27:
        corr = _corr(data, _rank(data, volume), _rank(data, vwap), 6)
        values = pd.Series(1.0, index=data.index)
        values.loc[_rank(data, _mean_series(data, corr, 2)) > 0.5] = -1.0
    elif number == 28:
        raw = _corr(data, adv20, low, 5) + (high + low) / 2.0 - close
        values = _scale(data, raw)
    elif number == 29:
        values = _rank(data, -_delta(data, "close", 5)) * _rank(data, volume / adv20.replace(0.0, np.nan))
    elif number == 30:
        sign_sum = np.sign(delta_close_1) + np.sign(_delay_series(data, delta_close_1, 1)) + np.sign(_delay_series(data, delta_close_1, 2))
        values = ((1.0 - _rank(data, sign_sum)) * _sum(data, "volume", 5)) / _sum(data, "volume", 20).replace(0.0, np.nan)
    else:
        raise ValueError(f"Unsupported alpha number: {number}")
    return _format(data, name, values.replace([np.inf, -np.inf], np.nan))


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    data["returns"] = data.groupby("symbol", sort=False)["close"].pct_change()
    data["vwap"] = data["amount"] / data["volume"].replace(0.0, np.nan)
    data["vwap"] = data["vwap"].fillna(data["close"])
    data["log_volume"] = np.log(data["volume"].clip(lower=1.0))
    return data


def _format(data: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": data["trade_date"].to_numpy(),
            "symbol": data["symbol"].to_numpy(),
            "factor_name": name,
            "factor_value": values.to_numpy(dtype=float),
        }
    )


def _register(name: str, func: Callable[[pd.DataFrame], pd.DataFrame], description: str, direction: int = 1) -> None:
    default_registry.add(
        FactorMeta(
            name=name,
            category="alpha101",
            horizon_days=5,
            required_columns=BASE_COLUMNS,
            direction=direction,
            description=description,
            source="WorldQuant Alpha101 daily OHLCV approximation",
        ),
        func,
    )


def _delay(data: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return data.groupby("symbol", sort=False)[column].shift(periods)


def _delay_series(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).shift(periods)


def _delta(data: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return data[column].astype(float) - _delay(data, column, periods)


def _delta_series(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    return series.astype(float) - _delay_series(data, series, periods)


def _mean(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)


def _mean_series(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)


def _std(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).std().reset_index(level=0, drop=True)


def _std_series(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window).std().reset_index(level=0, drop=True)


def _sum(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)


def _sum_series(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)


def _min(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).min().reset_index(level=0, drop=True)


def _min_series(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window).min().reset_index(level=0, drop=True)


def _max_series(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window).max().reset_index(level=0, drop=True)


def _corr(data: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    left = pd.Series(left.to_numpy(dtype=float), index=data.index)
    right = pd.Series(right.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = left.loc[group.index].rolling(window, min_periods=window).corr(right.loc[group.index])
    return values


def _cov(data: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    left = pd.Series(left.to_numpy(dtype=float), index=data.index)
    right = pd.Series(right.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = left.loc[group.index].rolling(window, min_periods=window).cov(right.loc[group.index])
    return values


def _rank(data: pd.DataFrame, series: pd.Series) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["trade_date"], sort=False).rank(method="average", pct=True)


def _ts_rank(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = tmp.loc[group.index].rolling(window, min_periods=window).apply(
            lambda x: pd.Series(x).rank(method="average").iloc[-1] / len(x),
            raw=True,
        )
    return values


def _argmax(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = tmp.loc[group.index].rolling(window, min_periods=window).apply(
            lambda x: float(np.argmax(x) + 1),
            raw=True,
        )
    return values


def _scale(data: pd.DataFrame, series: pd.Series) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    denom = tmp.abs().groupby(data["trade_date"], sort=False).transform("sum").replace(0.0, np.nan)
    return tmp / denom


for _idx, _func, _desc in [
    (1, alpha001, "Ranked reversal using downside volatility and recent price maxima."),
    (2, alpha002, "Negative correlation between volume acceleration and intraday return ranks."),
    (3, alpha003, "Negative open-volume rank correlation."),
    (4, alpha004, "Negative time-series rank of low-price cross-sectional rank."),
    (5, alpha005, "Open versus VWAP location with close-VWAP reversal."),
    (6, alpha006, "Negative open-volume rolling correlation."),
    (7, alpha007, "Volume-confirmed short-term reversal."),
    (8, alpha008, "Lagged open-return interaction reversal."),
    (9, alpha009, "Directional close delta reversal with trend filters."),
    (10, alpha010, "Ranked variant of alpha009."),
    (11, alpha011, "VWAP-close extrema combined with volume change."),
    (12, alpha012, "Volume direction times negative close delta."),
    (13, alpha013, "Negative covariance of price and volume ranks."),
    (14, alpha014, "Return delta rank times open-volume correlation."),
    (15, alpha015, "Rolling sum of ranked high-volume rank correlation."),
    (16, alpha016, "Negative covariance of high and volume ranks."),
    (17, alpha017, "Composite close rank, second derivative, and volume intensity."),
    (18, alpha018, "Reversal using open-close dispersion and close-open correlation."),
    (19, alpha019, "Trend sign reversal scaled by medium-term return rank."),
    (20, alpha020, "Open gap reversal against prior high, close, and low."),
    (21, alpha021, "Mean-reversion state classifier with volume confirmation."),
    (22, alpha022, "Falling high-volume correlation penalized by volatility rank."),
    (23, alpha023, "High-price breakout reversal."),
    (24, alpha024, "Slow trend filter with short-term reversal."),
    (25, alpha025, "Return, liquidity, VWAP, and high-close pressure rank."),
    (26, alpha026, "Negative maximum correlation of volume and high ranks."),
    (27, alpha027, "VWAP-volume correlation state signal."),
    (28, alpha028, "Scaled liquidity-low correlation and price location."),
    (29, alpha029, "Five-day reversal interacted with volume intensity."),
    (30, alpha030, "Signed return persistence with volume concentration."),
]:
    _register(f"alpha{_idx:03d}", _func, _desc)
