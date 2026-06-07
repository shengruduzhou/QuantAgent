"""Intraday (分时) features for 做T (T+0 band trading) from TickFlow 1-minute bars.

TickFlow's paid tier serves recent 1-minute K-lines (``tf.klines.intraday(symbol,
period="1m")``) with columns ``open/high/low/close/volume/amount/trade_time``.
This module turns one symbol's intraday bars into the feature columns the existing
defensive layers consume — ``net_buy_pressure / vwap_deviation / intraday_range_pos
/ spike_minutes`` (``risk/microstructure_guard.py``) and the 做T band levels used
by ``portfolio/sector_rotation.py``.

Compliance: these are DEFENSIVE / band-trading signals (buy a held core lower,
trim it higher within the day to lower cost / manage risk). They never generate
manipulative orders — consistent with the microstructure_guard contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntradayFeatures:
    symbol: str
    trade_date: str
    last: float
    vwap: float
    day_high: float
    day_low: float
    vwap_deviation: float        # (last - vwap)/vwap ; >0 上方, <0 折价
    intraday_range_pos: float    # 0 = 收在最低, 1 = 收在最高
    net_buy_pressure: float      # [-1,1] 上涨分钟量 − 下跌分钟量 占比 (主动买卖近似)
    spike_minutes: int           # 放量异动分钟数 (>3× 中位量)
    open_auction_gap: float      # (首bar开盘 − 昨收)/昨收 ; 集合竞价情绪
    intraday_return: float       # last/首bar开盘 − 1
    # 做T 价位带 (T+0 围绕核心持仓):
    buy_below: float             # 加T参考: ≤ 此价(vwap 或日内低位)考虑买
    sell_above: float            # 减T参考: ≥ 此价(日内高位)考虑卖
    dot_bias: str                # 偏多做T / 偏空做T / 观望

    def as_dict(self) -> dict:
        return asdict(self)


def _f(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_intraday_features(bars: pd.DataFrame, *, symbol: str | None = None,
                              prev_close: float | None = None) -> IntradayFeatures | None:
    """One symbol's 1-minute bars (one trading day) → intraday 做T features.

    ``bars`` columns: open/high/low/close/volume/amount (+ symbol/trade_date/trade_time).
    ``prev_close`` (前一交易日收盘) enables the opening-auction gap; optional.
    Returns None if the bars are unusable.
    """
    if bars is None or len(bars) == 0:
        return None
    b = bars.copy()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c not in b.columns:
            return None
        b[c] = _f(b[c])
    b = b.dropna(subset=["close"])
    if b.empty:
        return None
    if "trade_time" in b.columns:
        b = b.sort_values("trade_time")
    sym = str(symbol or (b["symbol"].iloc[0] if "symbol" in b.columns else ""))
    tdate = str(b["trade_date"].iloc[0]) if "trade_date" in b.columns else ""

    vol = b["volume"].fillna(0.0)
    tot_vol = float(vol.sum())
    # volume-weighted close (units of volume cancel → robust to 手/股 unit differences;
    # avoids tickflow's amount(元)/volume(手) scale mismatch that 100×-inflated a raw vwap).
    vwap = float((b["close"] * vol).sum() / tot_vol) if tot_vol > 0 else float(b["close"].mean())
    last = float(b["close"].iloc[-1])
    day_high = float(b["high"].max())
    day_low = float(b["low"].min())
    rng = day_high - day_low
    range_pos = float((last - day_low) / rng) if rng > 1e-9 else 0.5
    vwap_dev = float((last - vwap) / vwap) if vwap > 1e-9 else 0.0

    # 主动买卖近似: 每分钟 close>open 记为买量, < 记为卖量
    sign = np.sign(b["close"] - b["open"]).fillna(0.0)
    nbp = float((sign * vol).sum() / tot_vol) if tot_vol > 0 else 0.0

    med = float(vol[vol > 0].median()) if (vol > 0).any() else 0.0
    spike = int((vol > 3.0 * med).sum()) if med > 0 else 0

    open_px = float(b["open"].iloc[0])
    open_gap = float((open_px - prev_close) / prev_close) if (prev_close and prev_close > 0) else 0.0
    intraday_ret = float(last / open_px - 1.0) if open_px > 1e-9 else 0.0

    # 做T 价位带: 买参考取 vwap 与日内低位之间偏低; 卖参考取日内高位略下方.
    buy_below = round(min(vwap, day_low + 0.30 * rng), 4)
    sell_above = round(max(vwap, day_high - 0.15 * rng), 4)
    if range_pos >= 0.7 and nbp <= 0.0:
        bias = "偏空做T"        # 高位且主动卖占优 → 倾向减T
    elif range_pos <= 0.4 and nbp >= 0.0:
        bias = "偏多做T"        # 低位且主动买占优 → 倾向加T
    else:
        bias = "观望"

    return IntradayFeatures(
        symbol=sym, trade_date=tdate, last=round(last, 4), vwap=round(vwap, 4),
        day_high=round(day_high, 4), day_low=round(day_low, 4),
        vwap_deviation=round(vwap_dev, 5), intraday_range_pos=round(range_pos, 4),
        net_buy_pressure=round(nbp, 4), spike_minutes=spike,
        open_auction_gap=round(open_gap, 5), intraday_return=round(intraday_ret, 5),
        buy_below=buy_below, sell_above=sell_above, dot_bias=bias,
    )


def features_frame(bars_by_symbol: dict[str, pd.DataFrame],
                   prev_close: dict[str, float] | None = None) -> pd.DataFrame:
    """Compute features for many symbols → a DataFrame whose columns are exactly the
    inputs ``microstructure_guard`` / ``attach_rotation_and_dot`` expect."""
    prev_close = prev_close or {}
    rows = []
    for sym, bars in bars_by_symbol.items():
        f = compute_intraday_features(bars, symbol=sym, prev_close=prev_close.get(sym))
        if f is not None:
            rows.append(f.as_dict())
    return pd.DataFrame(rows)
