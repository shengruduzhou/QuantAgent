"""Intraday (分时) volume-price factors from 1-minute bars.

These collapse a day's minute bars into one cross-sectional score per
(symbol, trade_date) that captures *how* a stock traded during the session,
not just its close. They target the microstructure the user cares about —
分时量能, VWAP 偏离, 大单/主力流向, 主力异动 (拉升/砸盘) — using only
minute OHLCV (no L2/tick), so they run on any 1-minute feed.

Factor menu (all PIT: computed from a day's *own* minutes, usable at that
day's close for next-day decisions):

* ``first30_return``      — 开盘半小时强弱 (return over the first 30 min).
* ``last30_return``       — 尾盘动量 (return over the last 30 min).
* ``vwap_deviation``      — close / VWAP − 1 (above VWAP = intraday strength).
* ``intraday_range_pos``  — (close − low) / (high − low) close position.
* ``net_buy_pressure``    — (up-minute volume − down-minute volume) / total;
                            proxy for 主力净买入 (buying vs selling pressure).
* ``volume_concentration``— top-10-minute volume share; lumpy prints flag
                            大单/拆单/暗盘 活动.
* ``spike_minutes``       — # minutes with volume > 5× median; 主力异动 count.
* ``am_pm_volume_ratio``  — afternoon / morning volume (尾盘放量 > 1).
* ``minute_ret_skew``     — skew of minute returns (拉升 vs 砸盘 asymmetry).
* ``liq_amihud_1min``     — CICC-style 1-minute Amihud illiquidity.
* ``liq_amihud_1min_m20`` — trailing mean of ``liq_amihud_1min``.
* ``corr_prv``            — CICC-style price-return / volume correlation.
* ``corr_prv_m20``        — trailing mean of ``corr_prv``.
* ``open30_volume_share`` — first-30-minute volume share.
* ``close30_volume_share``— last-30-minute volume share.
* ``close3_volume_share`` — last-3-minute volume share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OPEN_MINUTES = 30
_CLOSE_MINUTES = 30
_SPIKE_MULT = 5.0
_CONCENTRATION_TOP = 10

FACTOR_COLUMNS = (
    "first30_return", "last30_return", "vwap_deviation", "intraday_range_pos",
    "net_buy_pressure", "volume_concentration", "spike_minutes",
    "am_pm_volume_ratio", "minute_ret_skew", "liq_amihud_1min",
    "liq_amihud_1min_m20", "corr_prv", "corr_prv_m20",
    "open30_volume_share", "close30_volume_share", "close3_volume_share",
)


def _day_factors(g: pd.DataFrame) -> dict[str, float]:
    """Compute the intraday factor row for one (symbol, trade_date) group."""
    g = g.sort_values("datetime")
    close = g["close"].to_numpy(dtype="float64")
    openp = g["open"].to_numpy(dtype="float64") if "open" in g else close
    high = g["high"].to_numpy(dtype="float64") if "high" in g else close
    low = g["low"].to_numpy(dtype="float64") if "low" in g else close
    vol = g["volume"].to_numpy(dtype="float64") if "volume" in g else np.zeros_like(close)
    amount = (
        g["amount"].to_numpy(dtype="float64")
        if "amount" in g else close * vol
    )
    n = len(close)
    if n == 0 or close[0] <= 0:
        return {c: np.nan for c in FACTOR_COLUMNS}

    day_open = openp[0]
    day_close = close[-1]
    day_high = float(np.nanmax(high))
    day_low = float(np.nanmin(low))
    total_vol = float(np.nansum(vol))

    first_n = min(_OPEN_MINUTES, n)
    last_n = min(_CLOSE_MINUTES, n)
    first30 = close[first_n - 1] / day_open - 1.0 if day_open > 0 else np.nan
    last30 = day_close / close[-last_n] - 1.0 if close[-last_n] > 0 else np.nan

    vwap = (close * vol).sum() / total_vol if total_vol > 0 else np.nan
    vwap_dev = day_close / vwap - 1.0 if vwap and vwap > 0 else np.nan
    rng = day_high - day_low
    range_pos = (day_close - day_low) / rng if rng > 1e-12 else 0.5

    minute_ret = np.diff(close) / close[:-1]
    log_ret = np.diff(np.log(np.where(close > 0, close, np.nan)))
    up = minute_ret > 0
    down = minute_ret < 0
    up_vol = float(vol[1:][up].sum())
    down_vol = float(vol[1:][down].sum())
    net_buy = (up_vol - down_vol) / total_vol if total_vol > 0 else np.nan

    if total_vol > 0 and n >= _CONCENTRATION_TOP:
        top = np.sort(vol)[-_CONCENTRATION_TOP:].sum()
        concentration = float(top / total_vol)
    else:
        concentration = np.nan
    med = float(np.nanmedian(vol)) if n else 0.0
    spikes = int((vol > _SPIKE_MULT * med).sum()) if med > 0 else 0

    half = n // 2
    am_vol = float(vol[:half].sum())
    pm_vol = float(vol[half:].sum())
    am_pm = pm_vol / am_vol if am_vol > 0 else np.nan

    if len(minute_ret) >= 3 and np.nanstd(minute_ret) > 1e-12:
        skew = float(pd.Series(minute_ret).skew())
    else:
        skew = 0.0

    amount_tail = amount[1:] if len(amount) > 1 else amount
    amount_denom = np.where(amount_tail > 0, amount_tail / 1e6, np.nan)
    amihud = float(np.nanmean(np.abs(log_ret) / amount_denom)) if len(log_ret) else np.nan
    if len(log_ret) >= 3 and np.nanstd(vol[1:]) > 1e-12 and np.nanstd(log_ret) > 1e-12:
        corr_prv = float(pd.Series(log_ret).corr(pd.Series(vol[1:]), method="pearson"))
    else:
        corr_prv = np.nan
    open_share = float(vol[:first_n].sum() / total_vol) if total_vol > 0 else np.nan
    close_share = float(vol[-last_n:].sum() / total_vol) if total_vol > 0 else np.nan
    close3_n = min(3, n)
    close3_share = float(vol[-close3_n:].sum() / total_vol) if total_vol > 0 else np.nan

    return {
        "first30_return": float(first30),
        "last30_return": float(last30),
        "vwap_deviation": float(vwap_dev),
        "intraday_range_pos": float(range_pos),
        "net_buy_pressure": float(net_buy),
        "volume_concentration": float(concentration),
        "spike_minutes": float(spikes),
        "am_pm_volume_ratio": float(am_pm),
        "minute_ret_skew": float(skew),
        "liq_amihud_1min": float(amihud),
        "liq_amihud_1min_m20": np.nan,
        "corr_prv": float(corr_prv),
        "corr_prv_m20": np.nan,
        "open30_volume_share": float(open_share),
        "close30_volume_share": float(close_share),
        "close3_volume_share": float(close3_share),
    }


def compute_intraday_factors(minute_panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse a minute panel into per-(symbol, trade_date) intraday factors.

    ``minute_panel`` is the long frame from
    :func:`quantagent.data.providers.qlib_intraday_reader.build_intraday_panel`
    (columns ``symbol, datetime, trade_date, open, high, low, close, volume``).
    Returns one row per symbol-day with ``FACTOR_COLUMNS``.
    """
    if minute_panel is None or minute_panel.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *FACTOR_COLUMNS])
    rows = []
    for (sym, day), g in minute_panel.groupby(["symbol", "trade_date"], sort=False):
        rec = {"symbol": sym, "trade_date": day}
        rec.update(_day_factors(g))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for source, target in (
        ("liq_amihud_1min", "liq_amihud_1min_m20"),
        ("corr_prv", "corr_prv_m20"),
    ):
        out[target] = (
            out.groupby("symbol", sort=False)[source]
            .rolling(20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return out


__all__ = ["compute_intraday_factors", "FACTOR_COLUMNS"]
