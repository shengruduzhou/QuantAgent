"""CICC-inspired A-share price/volume factor templates.

The public CICC high-frequency factor note groups effective A-share
microstructure factors into eight families: momentum/reversal, volatility,
higher moments, liquidity, price-volume correlation, chip distribution,
crowding, and money flow.  The exact Level-2 formulas are not available in
this repository, so this module implements deterministic daily-compatible
proxies with the same economic intent.  When minute bars are supplied the
existing :mod:`quantagent.factors.cicc_high_freq` module should still be used
for true intraday factors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.factors.registry import FactorMeta, default_registry


REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class CICCAshareFactorSpec:
    name: str
    category: str
    description: str
    direction: int = 1


def compute_cicc_ashare80_factors(
    frame: pd.DataFrame,
    names: list[str] | None = None,
    *,
    wide: bool = False,
) -> pd.DataFrame:
    """Compute the 80 daily-compatible CICC-style factor templates.

    Output formats
    --------------
    * ``wide=False`` (default, backward-compatible): long-form
      ``trade_date, symbol, factor_name, factor_value``.
    * ``wide=True``: wide-form ``trade_date, symbol, cicc_*`` columns.
      Skips the long-form intermediate; preferred for large panels.

    All rolling operations are per-symbol trailing windows; cross-sectional
    transforms use same-day data only.
    """
    data = _base(frame)
    values = _factor_values(data)
    selected = names or list(values)

    if wide:
        cols: dict[str, np.ndarray] = {}
        for name in selected:
            if name not in values:
                raise KeyError(f"unknown CICC A-share factor: {name}")
            series = pd.to_numeric(values[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            cols[name] = series.to_numpy(dtype=float)
        if not cols:
            return pd.DataFrame(columns=["trade_date", "symbol"])
        return pd.DataFrame(
            {"trade_date": data["trade_date"].to_numpy(),
             "symbol": data["symbol"].to_numpy(),
             **cols},
        )

    rows: list[pd.DataFrame] = []
    for name in selected:
        if name not in values:
            raise KeyError(f"unknown CICC A-share factor: {name}")
        rows.append(_format(data, name, values[name]))
    if not rows:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def cicc_ashare80_names() -> tuple[str, ...]:
    return tuple(_SPECS)


def cicc_ashare80_specs() -> dict[str, CICCAshareFactorSpec]:
    return dict(_SPECS)


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("symbol", "trade_date", *REQUIRED_COLUMNS) if column not in frame.columns]
    if missing:
        raise KeyError(f"CICC A-share factors require columns {missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["prev_close"] = _delay(data, data["close"], 1)
    data["return_1d"] = data["close"] / data["prev_close"].replace(0.0, np.nan) - 1.0
    data["intraday_return"] = data["close"] / data["open"].replace(0.0, np.nan) - 1.0
    data["vwap"] = (data["amount"] / data["volume"].replace(0.0, np.nan)).fillna(data["close"])
    data["volume_change"] = data.groupby("symbol", sort=False)["volume"].pct_change()
    data["amount_change"] = data.groupby("symbol", sort=False)["amount"].pct_change()
    data["range_pct"] = (data["high"] - data["low"]) / data["prev_close"].replace(0.0, np.nan)
    data["clv"] = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (data["high"] - data["low"]).replace(0.0, np.nan)
    data["signed_amount"] = data["clv"] * data["amount"]
    return data


def _factor_values(data: pd.DataFrame) -> dict[str, pd.Series]:
    ret = data["return_1d"]
    vol_chg = data["volume_change"]
    amt_chg = data["amount_change"]
    amount = data["amount"]
    volume = data["volume"]
    close = data["close"]
    high = data["high"]
    low = data["low"]
    vwap = data["vwap"]
    range_pct = data["range_pct"]
    clv = data["clv"]
    signed_amount = data["signed_amount"]
    values: dict[str, pd.Series] = {}

    for window in (1, 3, 5, 10, 20):
        values[f"cicc_mmt_ret_{window}d"] = _returns(data, close, window)
    for window in (1, 3, 5):
        values[f"cicc_mmt_reversal_{window}d"] = -_returns(data, close, window)
    values["cicc_mmt_range_position_20d"] = _range_position(data, close, low, high, 20)
    values["cicc_mmt_vwap_gap"] = close / vwap.replace(0.0, np.nan) - 1.0

    for window in (5, 10, 20, 60):
        values[f"cicc_vol_return_std_{window}d"] = _rolling(data, ret, window, "std")
    values["cicc_vol_upside_std_20d"] = _rolling(data, ret.where(ret > 0.0), 20, "std")
    values["cicc_vol_downside_std_20d"] = _rolling(data, ret.where(ret < 0.0), 20, "std")
    values["cicc_vol_range_mean_20d"] = _rolling(data, range_pct, 20, "mean")
    values["cicc_vol_range_std_20d"] = _rolling(data, range_pct, 20, "std")
    values["cicc_vol_intraday_std_20d"] = _rolling(data, data["intraday_return"], 20, "std")
    values["cicc_vol_parkinson_20d"] = np.sqrt(_rolling(data, np.log(high / low.replace(0.0, np.nan)).pow(2.0), 20, "mean") / (4.0 * np.log(2.0)))

    for window in (10, 20, 60):
        values[f"cicc_shape_skew_{window}d"] = _rolling(data, ret, window, "skew")
        values[f"cicc_shape_kurt_{window}d"] = _rolling(data, ret, window, "kurt")
    values["cicc_shape_return_z_20d"] = (ret - _rolling(data, ret, 20, "mean")) / _rolling(data, ret, 20, "std").replace(0.0, np.nan)
    values["cicc_shape_tail_ratio_20d"] = _rolling(data, ret.abs(), 20, "max") / _rolling(data, ret.abs(), 20, "mean").replace(0.0, np.nan)

    for window in (5, 20, 60):
        values[f"cicc_liq_amihud_{window}d"] = _rolling(data, ret.abs() / (amount.replace(0.0, np.nan) / 1e8), window, "mean")
    values["cicc_liq_amount_mean_20d"] = _rolling(data, amount, 20, "mean")
    values["cicc_liq_amount_z_20d"] = (amount - _rolling(data, amount, 20, "mean")) / _rolling(data, amount, 20, "std").replace(0.0, np.nan)
    values["cicc_liq_volume_z_20d"] = (volume - _rolling(data, volume, 20, "mean")) / _rolling(data, volume, 20, "std").replace(0.0, np.nan)
    values["cicc_liq_volume_concentration_5d"] = volume / _rolling(data, volume, 5, "sum").replace(0.0, np.nan)
    values["cicc_liq_volume_concentration_20d"] = volume / _rolling(data, volume, 20, "sum").replace(0.0, np.nan)
    values["cicc_liq_amount_turnover_5d"] = amount / _rolling(data, amount, 5, "sum").replace(0.0, np.nan)
    values["cicc_liq_amount_turnover_20d"] = amount / _rolling(data, amount, 20, "sum").replace(0.0, np.nan)

    for window in (5, 10, 20):
        values[f"cicc_corr_ret_volchg_{window}d"] = _corr(data, ret, vol_chg, window)
        values[f"cicc_corr_close_volume_{window}d"] = _corr(data, close, volume, window)
    values["cicc_corr_ret_amountchg_10d"] = _corr(data, ret, amt_chg, 10)
    values["cicc_corr_ret_amountchg_20d"] = _corr(data, ret, amt_chg, 20)
    values["cicc_corr_ret_lag_volchg_10d"] = _corr(data, ret, _delay(data, vol_chg, 1), 10)
    values["cicc_corr_ret_lag_volchg_20d"] = _corr(data, ret, _delay(data, vol_chg, 1), 20)

    for window in (20, 60, 120):
        values[f"cicc_doc_close_pct_{window}d"] = _range_position(data, close, low, high, window)
    values["cicc_doc_vwap_pct_20d"] = _range_position(data, vwap, low, high, 20)
    values["cicc_doc_drawdown_60d"] = close / _rolling(data, close, 60, "max").replace(0.0, np.nan) - 1.0
    values["cicc_doc_gain_to_min_60d"] = close / _rolling(data, close, 60, "min").replace(0.0, np.nan) - 1.0
    values["cicc_doc_close_to_high_20d"] = close / _rolling(data, high, 20, "max").replace(0.0, np.nan) - 1.0
    values["cicc_doc_close_to_low_20d"] = close / _rolling(data, low, 20, "min").replace(0.0, np.nan) - 1.0
    values["cicc_doc_vwap_to_high_20d"] = vwap / _rolling(data, high, 20, "max").replace(0.0, np.nan) - 1.0
    values["cicc_doc_vwap_to_low_20d"] = vwap / _rolling(data, low, 20, "min").replace(0.0, np.nan) - 1.0

    for window in (5, 10, 20):
        values[f"cicc_crowd_volume_conc_{window}d"] = _rolling(data, volume / _rolling(data, volume, window, "mean").replace(0.0, np.nan), window, "max")
    values["cicc_crowd_amount_conc_20d"] = _rolling(data, amount / _rolling(data, amount, 20, "mean").replace(0.0, np.nan), 20, "max")
    values["cicc_crowd_volume_autocorr_20d"] = _corr(data, volume, _delay(data, volume, 1), 20)
    values["cicc_crowd_amount_autocorr_20d"] = _corr(data, amount, _delay(data, amount, 1), 20)
    values["cicc_crowd_turnover_spike_20d"] = volume / _rolling(data, volume, 20, "mean").replace(0.0, np.nan)
    values["cicc_crowd_range_amount_corr_20d"] = _corr(data, range_pct, amount, 20)
    values["cicc_crowd_up_volume_ratio_20d"] = _rolling(data, volume.where(ret > 0.0), 20, "sum") / _rolling(data, volume, 20, "sum").replace(0.0, np.nan)
    values["cicc_crowd_down_volume_ratio_20d"] = _rolling(data, volume.where(ret < 0.0), 20, "sum") / _rolling(data, volume, 20, "sum").replace(0.0, np.nan)

    for window in (5, 10, 20):
        values[f"cicc_trade_money_flow_{window}d"] = _rolling(data, signed_amount, window, "sum") / _rolling(data, amount.abs(), window, "sum").replace(0.0, np.nan)
    values["cicc_trade_clv_5d"] = _rolling(data, clv, 5, "mean")
    values["cicc_trade_clv_20d"] = _rolling(data, clv, 20, "mean")
    values["cicc_trade_open_gap_amount_20d"] = _rolling(data, (data["open"] / data["prev_close"].replace(0.0, np.nan) - 1.0) * amount, 20, "sum") / _rolling(data, amount, 20, "sum").replace(0.0, np.nan)
    values["cicc_trade_close_strength_20d"] = _rolling(data, (close / vwap.replace(0.0, np.nan) - 1.0) * amount, 20, "sum") / _rolling(data, amount, 20, "sum").replace(0.0, np.nan)
    values["cicc_trade_high_return_volume_20d"] = _rolling(data, volume.where(ret > _rolling(data, ret, 20, "mean")), 20, "sum") / _rolling(data, volume, 20, "sum").replace(0.0, np.nan)
    values["cicc_trade_low_return_volume_20d"] = _rolling(data, volume.where(ret < _rolling(data, ret, 20, "mean")), 20, "sum") / _rolling(data, volume, 20, "sum").replace(0.0, np.nan)
    values["cicc_trade_amount_momentum_20d"] = _rolling(data, amount, 5, "mean") / _rolling(data, amount, 20, "mean").replace(0.0, np.nan) - 1.0
    values["cicc_trade_tail_amount_ratio_20d"] = amount / _rolling(data, amount, 20, "max").replace(0.0, np.nan)
    values["cicc_trade_smart_money_proxy_20d"] = _rolling(data, ret * amount, 20, "sum") / _rolling(data, amount, 20, "sum").replace(0.0, np.nan)

    if len(values) != 80:
        raise RuntimeError(f"internal CICC A-share factor count mismatch: {len(values)}")
    return values


def _returns(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    return series / _delay(data, series, periods).replace(0.0, np.nan) - 1.0


def _delay(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    return tmp.groupby(data["symbol"], sort=False).shift(periods)


def _rolling(data: pd.DataFrame, series: pd.Series, window: int, op: str) -> pd.Series:
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    grouped = tmp.groupby(data["symbol"], sort=False).rolling(window, min_periods=window)
    if op == "mean":
        out = grouped.mean()
    elif op == "std":
        out = grouped.std()
    elif op == "sum":
        out = grouped.sum()
    elif op == "min":
        out = grouped.min()
    elif op == "max":
        out = grouped.max()
    elif op == "skew":
        out = grouped.skew()
    elif op == "kurt":
        out = grouped.kurt()
    else:
        raise ValueError(f"unsupported rolling op: {op}")
    return out.reset_index(level=0, drop=True)


def _corr(data: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    left = pd.Series(left.to_numpy(dtype=float), index=data.index)
    right = pd.Series(right.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = left.loc[group.index].rolling(window, min_periods=window).corr(right.loc[group.index])
    return values


def _range_position(
    data: pd.DataFrame,
    value: pd.Series,
    low: pd.Series,
    high: pd.Series,
    window: int,
) -> pd.Series:
    low_min = _rolling(data, low, window, "min")
    high_max = _rolling(data, high, window, "max")
    return (value - low_min) / (high_max - low_min).replace(0.0, np.nan)


def _format(data: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": data["trade_date"].to_numpy(),
            "symbol": data["symbol"].to_numpy(),
            "factor_name": name,
            "factor_value": pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        }
    )


def _select(name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _wrapped(frame: pd.DataFrame) -> pd.DataFrame:
        return compute_cicc_ashare80_factors(frame, names=[name])

    _wrapped.__name__ = name
    return _wrapped


def _build_specs() -> dict[str, CICCAshareFactorSpec]:
    specs: dict[str, CICCAshareFactorSpec] = {}
    categories = {
        "mmt": "momentum_reversal",
        "vol": "volatility",
        "shape": "higher_moments",
        "liq": "liquidity",
        "corr": "price_volume_correlation",
        "doc": "chip_distribution",
        "crowd": "crowding",
        "trade": "money_flow",
    }
    for name in _factor_values(_sample_frame()).keys():
        group = name.split("_", 2)[1]
        specs[name] = CICCAshareFactorSpec(
            name=name,
            category=categories.get(group, "other"),
            description=f"CICC-style A-share daily proxy: {name}.",
            direction=-1 if group in {"vol", "liq", "crowd"} else 1,
        )
    return specs


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=130, freq="B")
    rows: list[dict[str, object]] = []
    for j, symbol in enumerate(("A", "B")):
        close = 10.0 + j
        for i, date in enumerate(dates):
            close *= 1.0 + 0.001 * np.sin(i / 5.0 + j)
            volume = 1_000_000.0 + 1_000.0 * i + 10_000.0 * j
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                }
            )
    return _base(pd.DataFrame(rows))


_SPECS = _build_specs()


for _factor_name, _spec in _SPECS.items():
    default_registry.add(
        FactorMeta(
            name=_factor_name,
            category="cicc_ashare80",
            horizon_days=5,
            required_columns=REQUIRED_COLUMNS,
            direction=_spec.direction,
            description=_spec.description,
            source="CICC high-frequency factor handbook inspired daily A-share proxy",
        ),
        _select(_factor_name),
    )


__all__ = [
    "CICCAshareFactorSpec",
    "compute_cicc_ashare80_factors",
    "cicc_ashare80_names",
    "cicc_ashare80_specs",
]
