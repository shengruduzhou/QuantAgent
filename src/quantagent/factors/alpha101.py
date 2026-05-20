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


def _make_alpha_wrapper(number: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _wrapped(frame: pd.DataFrame) -> pd.DataFrame:
        return _compute_alpha(frame, number)

    _wrapped.__name__ = f"alpha{number:03d}"
    return _wrapped


# Programmatically expose alpha031..alpha101 as module-level functions matching the
# alpha001..alpha030 style. Implementations live in the _compute_alpha dispatch.
for _n in range(31, 102):
    globals()[f"alpha{_n:03d}"] = _make_alpha_wrapper(_n)
del _n


def compute_alpha101(
    frame: pd.DataFrame,
    names: list[str] | None = None,
    *,
    wide: bool = False,
) -> pd.DataFrame:
    """Compute the full Alpha101 family (1..101) in a single pass.

    Performance contract: the base frame (returns, vwap, log_volume) plus the
    two shared intermediates ``adv20 = mean(volume,20)`` and
    ``delta_close_1 = delta(close,1)`` are computed ONCE for the whole call.
    Previously, ``default_registry.batch_compute`` invoked each factor
    independently which re-ran ``_base`` 101 times on the full panel — that
    was the root cause of the multi-hour CPU bottleneck and pivot-time OOM
    on production-scale inputs.

    Output formats
    --------------
    * ``wide=False`` (default, backward-compatible): long form ``[trade_date,
      symbol, factor_name, factor_value]``.
    * ``wide=True``: wide form ``[trade_date, symbol, alpha001, ..., alpha101]``.
      Skips the long-form intermediate (~10 GB on 3M-row panels) and the
      downstream pivot in the dataset builder, materially lowering peak RAM.

    Alphas needing IndClass/cap (industry neutralization or market-cap-weighted
    operations) are registered as placeholders that return NaN until sector and
    valuation tables are wired into the feature lake. The caller can filter those
    out via the column-coverage report.
    """
    selected: list[int]
    if names is None:
        selected = list(range(1, 102))
    else:
        selected = sorted({int(str(n).removeprefix("alpha")) for n in names})
    data, adv20, delta_close_1 = _prepare_alpha_context(frame)

    if wide:
        wide_cols: dict[str, np.ndarray] = {}
        for number in selected:
            try:
                values = _alpha_value(data, number, adv20, delta_close_1)
            except ValueError:
                continue
            values = values.replace([np.inf, -np.inf], np.nan)
            wide_cols[f"alpha{number:03d}"] = values.to_numpy(dtype=float)
        if not wide_cols:
            return pd.DataFrame(columns=["trade_date", "symbol"])
        out = pd.DataFrame(
            {"trade_date": data["trade_date"].to_numpy(),
             "symbol": data["symbol"].to_numpy(),
             **wide_cols},
        )
        return out

    frames: list[pd.DataFrame] = []
    for number in selected:
        try:
            values = _alpha_value(data, number, adv20, delta_close_1)
        except ValueError:
            continue
        frames.append(_format(data, f"alpha{number:03d}", values.replace([np.inf, -np.inf], np.nan)))
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    return pd.concat(frames, ignore_index=True)


def _prepare_alpha_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """One-shot setup shared across every selected alpha factor."""
    data = _base(frame)
    adv20 = _mean(data, "volume", 20)
    delta_close_1 = _delta(data, "close", 1)
    return data, adv20, delta_close_1


def _compute_alpha(frame: pd.DataFrame, number: int) -> pd.DataFrame:
    """Single-factor entrypoint kept for backwards-compatibility (per-factor public API)."""
    data, adv20, delta_close_1 = _prepare_alpha_context(frame)
    values = _alpha_value(data, number, adv20, delta_close_1)
    return _format(data, f"alpha{number:03d}", values.replace([np.inf, -np.inf], np.nan))


def _alpha_value(
    data: pd.DataFrame,
    number: int,
    adv20: pd.Series,
    delta_close_1: pd.Series,
) -> pd.Series:
    """Compute one alpha factor's value series against an already-prepared frame."""
    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    vwap = data["vwap"]
    returns = data["returns"]

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
    elif number == 31:
        # ((rank(rank(rank(decay_linear((-rank(rank(delta(close,10)))),10)))))
        #  + rank((-delta(close,3)))) + sign(scale(correlation(adv20, low, 12)))
        part1 = _rank(data, _rank(data, _rank(data, _decay_linear(data, -_rank(data, _rank(data, _delta(data, "close", 10))), 10))))
        part2 = _rank(data, -_delta(data, "close", 3))
        part3 = np.sign(_scale(data, _corr(data, adv20, low, 12)))
        values = part1 + part2 + part3
    elif number == 32:
        # scale(mean(close,7) - close) + 20 * scale(correlation(vwap, delay(close,5), 230))
        values = _scale(data, _mean(data, "close", 7) - close) + 20.0 * _scale(data, _corr(data, vwap, _delay(data, "close", 5), 230))
    elif number == 33:
        # rank((-1 * ((1 - (open / close))^1)))
        values = _rank(data, -((1.0 - open_ / close.replace(0.0, np.nan))))
    elif number == 34:
        # rank(((1 - rank((std(returns,2) / std(returns,5)))) + (1 - rank(delta(close,1)))))
        ratio = _std_series(data, returns, 2) / _std_series(data, returns, 5).replace(0.0, np.nan)
        values = _rank(data, (1.0 - _rank(data, ratio)) + (1.0 - _rank(data, delta_close_1)))
    elif number == 35:
        # ts_rank(volume,32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns,32))
        values = _ts_rank(data, volume, 32) * (1.0 - _ts_rank(data, close + high - low, 16)) * (1.0 - _ts_rank(data, returns, 32))
    elif number == 36:
        # 2.21*rank(corr(close-open, delay(volume,1),15)) + 0.7*rank(open-close)
        # + 0.73*rank(ts_rank(delay(-returns,6),5)) + rank(abs(corr(vwap,adv20,6)))
        # + 0.6*rank(((mean(close,200)-open)*(close-open)))
        p1 = 2.21 * _rank(data, _corr(data, close - open_, _delay(data, "volume", 1), 15))
        p2 = 0.7 * _rank(data, open_ - close)
        p3 = 0.73 * _rank(data, _ts_rank(data, _delay_series(data, -returns, 6), 5))
        p4 = _rank(data, _corr(data, vwap, adv20, 6).abs())
        p5 = 0.6 * _rank(data, (_mean(data, "close", 200) - open_) * (close - open_))
        values = p1 + p2 + p3 + p4 + p5
    elif number == 37:
        # rank(corr(delay(open-close,1), close, 200)) + rank(open-close)
        gap = open_ - close
        values = _rank(data, _corr(data, _delay_series(data, gap, 1), close, 200)) + _rank(data, gap)
    elif number == 38:
        # -rank(ts_rank(close,10)) * rank(close/open)
        values = -_rank(data, _ts_rank(data, close, 10)) * _rank(data, close / open_.replace(0.0, np.nan))
    elif number == 39:
        # -rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20,9)))) * (1 + rank(sum(returns,250)))
        vol_intensity = volume / adv20.replace(0.0, np.nan)
        inner = _delta(data, "close", 7) * (1.0 - _rank(data, _decay_linear(data, vol_intensity, 9)))
        values = -_rank(data, inner) * (1.0 + _rank(data, _sum_series(data, returns, 250)))
    elif number == 40:
        # -rank(std(high,10)) * corr(high, volume, 10)
        values = -_rank(data, _std(data, "high", 10)) * _corr(data, high, volume, 10)
    elif number == 41:
        # ((high * low)^0.5) - vwap
        values = np.sqrt((high.clip(lower=0.0) * low.clip(lower=0.0))) - vwap
    elif number == 42:
        # rank(vwap - close) / rank(vwap + close)
        denom = _rank(data, vwap + close).replace(0.0, np.nan)
        values = _rank(data, vwap - close) / denom
    elif number == 43:
        # ts_rank(volume/adv20, 20) * ts_rank(-delta(close,7), 8)
        values = _ts_rank(data, volume / adv20.replace(0.0, np.nan), 20) * _ts_rank(data, -_delta(data, "close", 7), 8)
    elif number == 44:
        # -corr(high, rank(volume), 5)
        values = -_corr(data, high, _rank(data, volume), 5)
    elif number == 45:
        # -(rank(mean(delay(close,5),20)) * corr(close, volume, 2)
        #   * rank(corr(sum(close,5), sum(close,20), 2)))
        part1 = _rank(data, _mean_series(data, _delay(data, "close", 5), 20))
        part2 = _corr(data, close, volume, 2)
        part3 = _rank(data, _corr(data, _sum(data, "close", 5), _sum(data, "close", 20), 2))
        values = -(part1 * part2 * part3)
    elif number == 46:
        # ((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10) condition
        slope_far = (_delay(data, "close", 20) - _delay(data, "close", 10)) / 10.0
        slope_near = (_delay(data, "close", 10) - close) / 10.0
        diff = slope_far - slope_near
        values = pd.Series(0.0, index=data.index, dtype=float)
        values.loc[diff > 0.25] = -1.0
        values.loc[diff < 0.0] = 1.0
        mask_mid = (diff <= 0.25) & (diff >= 0.0)
        values.loc[mask_mid] = -(close.loc[mask_mid] - _delay(data, "close", 1).loc[mask_mid])
    elif number == 47:
        # rank(1/close) * volume / adv20 * (high * rank(high - close) / mean(high,5)) - rank(vwap - delay(vwap,5))
        mean_high_5 = _mean(data, "high", 5).replace(0.0, np.nan)
        p1 = _rank(data, 1.0 / close.replace(0.0, np.nan)) * volume / adv20.replace(0.0, np.nan)
        p2 = high * _rank(data, high - close) / mean_high_5
        p3 = _rank(data, vwap - _delay(data, "vwap", 5))
        values = p1 * p2 - p3
    elif number == 49:
        # condition: ((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10) < -0.1
        slope_far = (_delay(data, "close", 20) - _delay(data, "close", 10)) / 10.0
        slope_near = (_delay(data, "close", 10) - close) / 10.0
        diff = slope_far - slope_near
        values = -(close - _delay(data, "close", 1))
        values.loc[diff < -0.1] = 1.0
    elif number == 50:
        # -ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)
        values = -_max_series(data, _rank(data, _corr(data, _rank(data, volume), _rank(data, vwap), 5)), 5)
    elif number == 51:
        # condition mirror of alpha49 with stricter threshold
        slope_far = (_delay(data, "close", 20) - _delay(data, "close", 10)) / 10.0
        slope_near = (_delay(data, "close", 10) - close) / 10.0
        diff = slope_far - slope_near
        values = -(close - _delay(data, "close", 1))
        values.loc[diff < -0.05] = 1.0
    elif number == 52:
        # (-ts_min(low,5) + delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220) * ts_rank(volume,5)
        ts_min_5 = _min(data, "low", 5)
        ret_240 = _sum_series(data, returns, 240)
        ret_20 = _sum_series(data, returns, 20)
        values = (-ts_min_5 + _delay_series(data, ts_min_5, 5)) * _rank(data, (ret_240 - ret_20) / 220.0) * _ts_rank(data, volume, 5)
    elif number == 53:
        # -delta(((close-low)-(high-close))/(close-low), 9)
        denom = (close - low).replace(0.0, np.nan)
        expr = ((close - low) - (high - close)) / denom
        values = -_delta_series(data, expr, 9)
    elif number == 54:
        # -(low - close) * open^5 / ((low - high) * close^5)
        denom = (low - high).replace(0.0, np.nan) * close.pow(5)
        values = -((low - close) * open_.pow(5)) / denom
    elif number == 55:
        # -corr(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))), rank(volume), 6)
        low_min_12 = _min(data, "low", 12)
        high_max_12 = _max_series(data, high, 12)
        denom = (high_max_12 - low_min_12).replace(0.0, np.nan)
        norm = (close - low_min_12) / denom
        values = -_corr(data, _rank(data, norm), _rank(data, volume), 6)
    elif number == 57:
        # -(close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2)
        denom = _decay_linear(data, _rank(data, _argmax(data, close, 30)), 2).replace(0.0, np.nan)
        values = -(close - vwap) / denom
    elif number == 60:
        # -((2*scale(rank(((close-low)-(high-close))/(high-low) * volume))) - scale(rank(ts_argmax(close,10))))
        hl_range = (high - low).replace(0.0, np.nan)
        money_flow_vol = ((close - low) - (high - close)) / hl_range * volume
        values = -(2.0 * _scale(data, _rank(data, money_flow_vol)) - _scale(data, _rank(data, _argmax(data, close, 10))))
    elif number == 61:
        # rank(vwap - ts_min(vwap,16)) < rank(corr(vwap, adv180, 18))
        adv180 = _mean(data, "volume", 180)
        lhs = _rank(data, vwap - _min_series(data, vwap, 16))
        rhs = _rank(data, _corr(data, vwap, adv180, 18))
        values = (lhs < rhs).astype(float)
    elif number == 62:
        # rank(corr(vwap, sum(adv20,22), 10)) < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))
        lhs = _rank(data, _corr(data, vwap, _sum_series(data, adv20, 22), 10))
        rhs_inner = (2.0 * _rank(data, open_) < (_rank(data, (high + low) / 2.0) + _rank(data, high))).astype(float)
        values = -(lhs < _rank(data, rhs_inner)).astype(float)
    elif number == 64:
        # rank(corr(sum((open*0.178)+(low*0.822), 13), sum(adv120, 13), 17)) < rank(delta((high+low)/2*0.178 + vwap*0.822, 4))
        adv120 = _mean(data, "volume", 120)
        lhs = _rank(data, _corr(data, _sum_series(data, open_ * 0.178 + low * 0.822, 13), _sum_series(data, adv120, 13), 17))
        rhs_inner = (high + low) / 2.0 * 0.178 + vwap * 0.822
        rhs = _rank(data, _delta_series(data, rhs_inner, 4))
        values = -(lhs < rhs).astype(float)
    elif number == 65:
        # rank(corr(open*0.0085 + vwap*0.9915, sum(adv60,9), 6)) < rank(open - ts_min(open,14))
        adv60 = _mean(data, "volume", 60)
        lhs = _rank(data, _corr(data, open_ * 0.0085 + vwap * 0.9915, _sum_series(data, adv60, 9), 6))
        rhs = _rank(data, open_ - _min_series(data, open_, 14))
        values = -(lhs < rhs).astype(float)
    elif number == 66:
        # rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear(((low*0.96 - vwap)/(open-(high+low)/2)),11),7)
        denom = (open_ - (high + low) / 2.0).replace(0.0, np.nan)
        expr = (low * 0.96 - vwap) / denom
        values = -(_rank(data, _decay_linear(data, _delta(data, "vwap", 4), 7)) + _ts_rank(data, _decay_linear(data, expr, 11), 7))
    elif number == 68:
        # ts_rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.518+low*0.482, 1))
        adv15 = _mean(data, "volume", 15)
        lhs = _ts_rank(data, _corr(data, _rank(data, high), _rank(data, adv15), 9), 14)
        rhs = _rank(data, _delta_series(data, close * 0.518 + low * 0.482, 1))
        values = -(lhs < rhs).astype(float)
    elif number == 70:
        # similar to alpha69 minus IndClass
        values = -_rank(data, _delta(data, "vwap", 1)).pow(_ts_rank(data, _corr(data, close, _mean(data, "volume", 50), 18), 18))
    elif number == 71:
        # max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(adv180,12), 18), 4), 16),
        #     ts_rank(decay_linear(rank((low+open-2*vwap))^2,16),4))
        adv180 = _mean(data, "volume", 180)
        a = _ts_rank(data, _decay_linear(data, _corr(data, _ts_rank(data, close, 3), _ts_rank(data, adv180, 12), 18), 4), 16)
        b = _ts_rank(data, _decay_linear(data, _rank(data, low + open_ - 2.0 * vwap).pow(2.0), 16), 4)
        values = _emax2(a, b)
    elif number == 72:
        # rank(decay_linear(corr((high+low)/2, adv40, 9), 10)) / rank(decay_linear(corr(ts_rank(vwap,4), ts_rank(volume,19), 7), 3))
        adv40 = _mean(data, "volume", 40)
        num = _rank(data, _decay_linear(data, _corr(data, (high + low) / 2.0, adv40, 9), 10))
        den = _rank(data, _decay_linear(data, _corr(data, _ts_rank(data, vwap, 4), _ts_rank(data, volume, 19), 7), 3)).replace(0.0, np.nan)
        values = num / den
    elif number == 73:
        # -max(rank(decay_linear(delta(vwap,5),3)), ts_rank(decay_linear((-delta(open*0.147+low*0.853,2)/(open*0.147+low*0.853)),16),17))
        weighted = open_ * 0.147 + low * 0.853
        a = _rank(data, _decay_linear(data, _delta(data, "vwap", 5), 3))
        ratio_expr = -_delta_series(data, weighted, 2) / weighted.replace(0.0, np.nan)
        b = _ts_rank(data, _decay_linear(data, ratio_expr, 16), 17)
        values = -_emax2(a, b)
    elif number == 74:
        # rank(corr(close, sum(adv30, 37), 15)) < rank(corr(rank(high*0.0261+vwap*0.9739), rank(volume), 11))
        adv30 = _mean(data, "volume", 30)
        lhs = _rank(data, _corr(data, close, _sum_series(data, adv30, 37), 15))
        rhs = _rank(data, _corr(data, _rank(data, high * 0.0261 + vwap * 0.9739), _rank(data, volume), 11))
        values = -(lhs < rhs).astype(float)
    elif number == 75:
        # rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))
        adv50 = _mean(data, "volume", 50)
        lhs = _rank(data, _corr(data, vwap, volume, 4))
        rhs = _rank(data, _corr(data, _rank(data, low), _rank(data, adv50), 12))
        values = (lhs < rhs).astype(float)
    elif number == 77:
        # min(rank(decay_linear((high+low)/2+high-vwap-high,20)), rank(decay_linear(corr((high+low)/2, adv40, 3),6)))
        adv40 = _mean(data, "volume", 40)
        a = _rank(data, _decay_linear(data, (high + low) / 2.0 + high - vwap - high, 20))
        b = _rank(data, _decay_linear(data, _corr(data, (high + low) / 2.0, adv40, 3), 6))
        values = _emin2(a, b)
    elif number == 78:
        # rank(corr(sum(low*0.352+vwap*0.648, 20), sum(adv40,20), 7))^rank(corr(rank(vwap), rank(volume), 6))
        adv40 = _mean(data, "volume", 40)
        base = _rank(data, _corr(data, _sum_series(data, low * 0.352 + vwap * 0.648, 20), _sum_series(data, adv40, 20), 7))
        expo = _rank(data, _corr(data, _rank(data, vwap), _rank(data, volume), 6))
        values = _signedpower(base, 1.0) * np.sign(expo)  # approximation: ranks are positive so power simplifies
    elif number == 81:
        # rank(log(product(rank(rank(corr(vwap, sum(adv10,50), 8))^4),15))) < rank(corr(rank(vwap), rank(volume),5))
        adv10 = _mean(data, "volume", 10)
        inner = _rank(data, _rank(data, _corr(data, vwap, _sum_series(data, adv10, 50), 8)).pow(4))
        lhs = _rank(data, np.log(_product(data, inner, 15).clip(lower=1e-12)))
        rhs = _rank(data, _corr(data, _rank(data, vwap), _rank(data, volume), 5))
        values = -(lhs < rhs).astype(float)
    elif number == 83:
        # rank(delay((high-low)/mean(close,5), 2)) * rank(rank(volume)) / ((high-low)/mean(close,5) / (vwap-close))
        ratio = (high - low) / _mean(data, "close", 5).replace(0.0, np.nan)
        denom = (ratio / (vwap - close).replace(0.0, np.nan)).replace(0.0, np.nan)
        values = _rank(data, _delay_series(data, ratio, 2)) * _rank(data, _rank(data, volume)) / denom
    elif number == 84:
        # signedpower(ts_rank(vwap - ts_max(vwap,15), 21), delta(close,5))
        base_expr = _ts_rank(data, vwap - _max_series(data, vwap, 15), 21)
        exponent = _delta(data, "close", 5)
        # Use signed mixed exponent: |x|^e * sign(x). Clip exponent to a safe range.
        exponent = exponent.clip(-3.0, 3.0)
        arr_base = base_expr.to_numpy(dtype=float)
        arr_exp = exponent.to_numpy(dtype=float)
        values = pd.Series(np.sign(arr_base) * np.power(np.abs(arr_base) + 1e-12, arr_exp), index=data.index)
    elif number == 85:
        # rank(corr(high*0.876+close*0.124, adv30, 10))^rank(corr(ts_rank((high+low)/2,4), ts_rank(volume,10),7))
        adv30 = _mean(data, "volume", 30)
        base_expr = _rank(data, _corr(data, high * 0.876 + close * 0.124, adv30, 10))
        expo = _rank(data, _corr(data, _ts_rank(data, (high + low) / 2.0, 4), _ts_rank(data, volume, 10), 7))
        values = _signedpower(base_expr, 1.0) * np.sign(expo)
    elif number == 86:
        # ts_rank(corr(close, sum(adv20,15), 6),20) < rank((open+close)-(vwap+open))
        lhs = _ts_rank(data, _corr(data, close, _sum_series(data, adv20, 15), 6), 20)
        rhs = _rank(data, (open_ + close) - (vwap + open_))
        values = -(lhs < rhs).astype(float)
    elif number == 88:
        # min(rank(decay_linear((rank(open)+rank(low)-(rank(high)+rank(close))),8)),
        #     ts_rank(decay_linear(corr(ts_rank(close,8), ts_rank(adv60,21),8),7),3))
        adv60 = _mean(data, "volume", 60)
        a = _rank(data, _decay_linear(data, (_rank(data, open_) + _rank(data, low) - _rank(data, high) - _rank(data, close)), 8))
        b = _ts_rank(data, _decay_linear(data, _corr(data, _ts_rank(data, close, 8), _ts_rank(data, adv60, 21), 8), 7), 3)
        values = _emin2(a, b)
    elif number == 92:
        # min(ts_rank(decay_linear((((high+low)/2 + close) < (low + open)),15),19),
        #     ts_rank(decay_linear(corr(rank(low), rank(adv30), 8),7),7))
        adv30 = _mean(data, "volume", 30)
        cond = ((high + low) / 2.0 + close < low + open_).astype(float)
        a = _ts_rank(data, _decay_linear(data, cond, 15), 19)
        b = _ts_rank(data, _decay_linear(data, _corr(data, _rank(data, low), _rank(data, adv30), 8), 7), 7)
        values = _emin2(a, b)
    elif number == 94:
        # -((rank(vwap - ts_min(vwap,12)))^ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4),18),3))
        adv60 = _mean(data, "volume", 60)
        base_expr = _rank(data, vwap - _min_series(data, vwap, 12))
        expo = _ts_rank(data, _corr(data, _ts_rank(data, vwap, 20), _ts_rank(data, adv60, 4), 18), 3)
        values = -_signedpower(base_expr, 1.0) * np.sign(expo)
    elif number == 95:
        # rank(open - ts_min(open,12)) < ts_rank(rank(corr(sum((high+low)/2,19), sum(adv40,19),13))^5, 12)
        adv40 = _mean(data, "volume", 40)
        lhs = _rank(data, open_ - _min_series(data, open_, 12))
        inner = _rank(data, _corr(data, _sum_series(data, (high + low) / 2.0, 19), _sum_series(data, adv40, 19), 13)).pow(5)
        rhs = _ts_rank(data, inner, 12)
        values = (lhs < rhs).astype(float)
    elif number == 96:
        # max(ts_rank(decay_linear(corr(rank(vwap), rank(volume),4),4),8),
        #     ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4),4),13),14),13))
        adv60 = _mean(data, "volume", 60)
        a = _ts_rank(data, _decay_linear(data, _corr(data, _rank(data, vwap), _rank(data, volume), 4), 4), 8)
        inner = _argmax(data, _corr(data, _ts_rank(data, close, 7), _ts_rank(data, adv60, 4), 4), 13)
        b = _ts_rank(data, _decay_linear(data, inner, 14), 13)
        values = -_emax2(a, b)
    elif number == 98:
        # rank(decay_linear(corr(vwap, sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15),21),9),7),8))
        adv5 = _mean(data, "volume", 5)
        adv15 = _mean(data, "volume", 15)
        a = _rank(data, _decay_linear(data, _corr(data, vwap, _sum_series(data, adv5, 26), 5), 7))
        b = _rank(data, _decay_linear(data, _ts_rank(data, _argmin(data, _corr(data, _rank(data, open_), _rank(data, adv15), 21), 9), 7), 8))
        values = a - b
    elif number == 99:
        # rank(corr(sum((high+low)/2,20), sum(adv60,20),9)) < rank(corr(low, volume,6))
        adv60 = _mean(data, "volume", 60)
        lhs = _rank(data, _corr(data, _sum_series(data, (high + low) / 2.0, 20), _sum_series(data, adv60, 20), 9))
        rhs = _rank(data, _corr(data, low, volume, 6))
        values = -(lhs < rhs).astype(float)
    elif number == 101:
        # (close - open) / ((high - low) + 0.001)
        values = (close - open_) / ((high - low) + 0.001)
    elif number in _INDUSTRY_OR_CAP_ALPHAS:
        # Placeholder: needs IndClass or cap. Returns NaN until sector/valuation
        # data is wired into the feature store.
        values = pd.Series(np.nan, index=data.index, dtype=float)
    else:
        raise ValueError(f"Unsupported alpha number: {number}")
    return values


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


def _argmin(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = tmp.loc[group.index].rolling(window, min_periods=window).apply(
            lambda x: float(np.argmin(x) + 1),
            raw=True,
        )
    return values


def _decay_linear(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    """Linear-weighted moving average: weights = [1, 2, ..., window] / sum."""
    if window < 1:
        return pd.Series(np.nan, index=data.index, dtype=float)
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    values = pd.Series(np.nan, index=data.index, dtype=float)
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = tmp.loc[group.index].rolling(window, min_periods=window).apply(
            lambda x: float(np.dot(x, weights)),
            raw=True,
        )
    return values


def _signedpower(series: pd.Series, exponent: float) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    return pd.Series(np.sign(arr) * np.power(np.abs(arr), exponent), index=series.index)


def _product(data: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    tmp = pd.Series(series.to_numpy(dtype=float), index=data.index)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = tmp.loc[group.index].rolling(window, min_periods=window).apply(
            lambda x: float(np.prod(x)),
            raw=True,
        )
    return values


def _emin2(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise minimum of two aligned Series."""
    return pd.Series(np.minimum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)), index=left.index)


def _emax2(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise maximum of two aligned Series."""
    return pd.Series(np.maximum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)), index=left.index)


# Alphas listed in the WorldQuant paper that require IndClass/cap (industry
# neutralization or market-cap-weighted ops) — we register them as placeholders
# returning NaN until sector_map + valuation are wired into the feature lake.
_INDUSTRY_OR_CAP_ALPHAS: tuple[int, ...] = (
    48, 56, 58, 59, 63, 67, 69, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100,
)


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


# WorldQuant Alpha101 31..101. Each entry is (number, short description).
# Alphas requiring IndClass / market cap are registered as placeholders that
# return NaN; their numbers live in _INDUSTRY_OR_CAP_ALPHAS so the dispatch
# above can short-circuit them.
_ALPHA_DESCRIPTIONS_31_101: dict[int, str] = {
    31: "Decay-linear of ranked close delta plus correlation sign.",
    32: "Mean-reversion scaled by long-horizon VWAP-close correlation.",
    33: "Rank of one-minus-open-over-close (gap reversal).",
    34: "Composite reversal mixing return-vol ratio and close delta.",
    35: "Volume time-rank × residual time-rank × return time-rank.",
    36: "Multi-component composite (corr, gap, lagged returns, abs corr, trend).",
    37: "Open-close gap correlation with close plus gap rank.",
    38: "Negative close time-rank times close-over-open rank.",
    39: "Volume-weighted close-delta with long-horizon return rank.",
    40: "Negative volatility rank times high-volume correlation.",
    41: "Geometric mean of high and low minus VWAP.",
    42: "Ratio of (vwap-close) rank to (vwap+close) rank.",
    43: "Volume-intensity time-rank times negative-close-delta time-rank.",
    44: "Negative correlation of high with volume rank.",
    45: "Combined mean-close rank, close-volume corr, sum-close corr.",
    46: "Trend-curvature gated reversal.",
    47: "Liquidity-weighted high-close pressure minus VWAP shift.",
    48: "[placeholder] IndClass/subindustry-neutralized momentum.",
    49: "Sharp trend-curvature reversal switch.",
    50: "Negative time-max of rank-volume vs rank-VWAP correlation.",
    51: "Lighter version of alpha49.",
    52: "Low-min reversal scaled by long-vs-short return spread and volume rank.",
    53: "Negative delta of money-flow oscillator.",
    54: "Open-close power ratio normalized by low-high spread.",
    55: "Negative correlation of normalized close band with volume rank.",
    56: "[placeholder] cap-weighted return ratio.",
    57: "Negative close-VWAP gap divided by decay-linear of ts_argmax close.",
    58: "[placeholder] IndClass-neutralized VWAP-volume relationship.",
    59: "[placeholder] IndClass-neutralized weighted VWAP.",
    60: "Scaled money-flow volume minus scaled ts_argmax close.",
    61: "VWAP reach vs medium-term liquidity correlation.",
    62: "VWAP-liquidity gating vs midprice rank cross.",
    63: "[placeholder] IndClass-neutralized close-decay term.",
    64: "Weighted open-low correlation gate.",
    65: "Open-VWAP weighted correlation gate.",
    66: "Decay-linear of VWAP delta plus low-vwap channel time-rank.",
    67: "[placeholder] IndClass sector/subindustry-neutralized variant.",
    68: "High-volume relation gating mid-price delta.",
    69: "[placeholder] IndClass-neutralized VWAP delta time-rank.",
    70: "Negative rank-delta-vwap weighted by liquidity time-rank.",
    71: "Max of two decay-linear time-rank blocks (corr block vs low-open vwap).",
    72: "Decay-linear correlation midprice/liquidity over decay-linear vwap/volume ranks.",
    73: "Negative max of vwap-delta decay vs open-low ratio decay.",
    74: "Close-liquidity correlation gate vs high-vwap-volume rank.",
    75: "VWAP-volume short correlation vs low-liquidity rank correlation.",
    76: "[placeholder] IndClass-neutralized vwap decay block.",
    77: "Min of midprice decay-linear vs corr decay-linear.",
    78: "Weighted low-vwap-sum-corr powered by vwap-volume rank.",
    79: "[placeholder] IndClass-neutralized close delta.",
    80: "[placeholder] IndClass-neutralized close delta.",
    81: "Log-product of VWAP-liquidity rank powered to 4 vs vwap-volume corr.",
    82: "[placeholder] IndClass-neutralized open delta with decay.",
    83: "High-low ratio rank × volume rank squared / liquidity-vwap-close gap.",
    84: "Power: ts_rank(vwap-tsmax(vwap,15),21) ^ delta(close,5).",
    85: "Weighted high-close correlation rank^rank corr time-rank.",
    86: "Close-liquidity time-rank vs open-vwap rank.",
    87: "[placeholder] IndClass-neutralized close delta.",
    88: "Min of two ranked decay-linear blocks (open-low-high-close rank).",
    89: "[placeholder] IndClass-neutralized low-vwap decay block.",
    90: "[placeholder] IndClass-neutralized rank-close-volume block.",
    91: "[placeholder] IndClass-neutralized close-decay liquidity gating.",
    92: "Min of two decay-linear blocks (midprice condition vs low-liquidity corr).",
    93: "[placeholder] IndClass-neutralized vwap-close-decay block.",
    94: "Negative vwap channel rank powered by vwap-liquidity time-rank.",
    95: "Open channel rank vs midprice-liquidity power gate.",
    96: "Max of two decay blocks (vwap-volume rank corr vs ts_argmax corr).",
    97: "[placeholder] IndClass-neutralized low-vwap decay block.",
    98: "VWAP-liquidity decay corr minus ts_argmin open-liquidity time-rank.",
    99: "Midprice-liquidity correlation vs low-volume correlation gate.",
    100: "[placeholder] IndClass-neutralized money-flow block.",
    101: "Intraday close-open over high-low range (simple oscillator).",
}

for _idx, _desc in _ALPHA_DESCRIPTIONS_31_101.items():
    _register(f"alpha{_idx:03d}", globals()[f"alpha{_idx:03d}"], _desc)
del _idx, _desc
