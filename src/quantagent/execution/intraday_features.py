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


CAUSAL_INTRADAY_FEATURE_COLUMNS = [
    "price_vs_vwap_z",
    "price_vs_open",
    "price_vs_preclose",
    "intraday_return",
    "intraday_percentile_since_open",
    "distance_to_high_of_day",
    "distance_to_low_of_day",
    "rolling_return_3m",
    "rolling_return_5m",
    "rolling_return_10m",
    "rolling_return_20m",
    "rolling_volatility_5m",
    "rolling_volatility_10m",
    "rolling_volatility_20m",
    "minute_of_day",
    "session_flag",
    "gap_open",
    "limit_up_distance",
    "limit_down_distance",
    "volume_zscore_5m",
    "volume_zscore_20m",
    "turnover_intraday",
    "relative_volume_vs_20d",
    "amount_zscore",
    "volume_price_divergence",
    "estimated_spread_bps",
    "volume_capacity_ratio",
    "stock_return_minus_index",
    "stock_return_minus_industry",
    "stock_vwap_dev_minus_industry",
    "relative_volume_vs_industry",
    "market_breadth_intraday",
    "up_down_ratio_intraday",
    "limit_up_count",
    "limit_down_count",
    "microcap_style_strength",
    "largecap_style_strength",
    "after_sell_new_high_risk",
    "after_buy_breakdown_risk",
    "trend_strength_score",
    "one_way_trend_probability",
    "mean_reversion_probability",
    "momentum_persistence",
    "failed_breakout_probability",
    "failed_breakdown_probability",
    "near_limit_risk",
]

LEVEL2_FEATURE_COLUMNS = [
    "bid_ask_spread",
    "order_book_imbalance",
    "bid_depth",
    "ask_depth",
    "active_buy_ratio",
    "active_sell_ratio",
    "large_order_buy_ratio",
    "large_order_sell_ratio",
    "cancel_order_ratio",
    "queue_pressure_near_limit",
]

# 东财 push2 per-minute fund-flow order-flow features (forward-collected via
# scripts/collect_eastmoney_fundflow_minute.py). The raw fund-flow columns are
# CUMULATIVE intraday net inflow in 元 (主力/超大单/大单/中单/小单); these derived
# features are causal (use only data up to minute t) and unit-free.
FUNDFLOW_RAW_COLUMNS = ["main_net", "super_net", "large_net", "mid_net", "small_net"]
FUNDFLOW_FEATURE_COLUMNS = [
    "ff_main_net_intensity",      # cum 主力净流入 / cum amount
    "ff_super_large_intensity",   # cum (超大单+大单) / cum amount
    "ff_main_net_delta_z",        # per-minute 主力净流入 (z over 20m)
    "ff_super_net_delta_z",       # per-minute 超大单净流入 (z over 20m)
    "ff_flow_price_divergence",   # flow-intensity change minus price return (accumulation vs markup)
    "ff_main_net_accel",          # change in per-minute 主力 net flow
]


def merge_fundflow_features(minute_panel: pd.DataFrame, fundflow_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach causal 东财 minute fund-flow features to a minute panel.

    ``fundflow_panel`` has columns symbol/trade_time(+trade_date) plus the
    cumulative FUNDFLOW_RAW_COLUMNS.  Missing coverage yields NaN feature
    columns so the pipeline runs whether or not fund-flow has been collected.
    """
    out = minute_panel.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
    if "trade_date" not in out.columns:
        out["trade_date"] = out["trade_time"].dt.normalize()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for c in FUNDFLOW_FEATURE_COLUMNS:
        out[c] = np.nan
    if fundflow_panel is None or fundflow_panel.empty:
        return out
    ff = fundflow_panel.copy()
    ff["symbol"] = ff["symbol"].astype(str)
    ff["trade_time"] = pd.to_datetime(ff["trade_time"], errors="coerce")
    have = [c for c in FUNDFLOW_RAW_COLUMNS if c in ff.columns]
    ff = ff[["symbol", "trade_time", *have]].dropna(subset=["symbol", "trade_time"])
    out = out.merge(ff, on=["symbol", "trade_time"], how="left", suffixes=("", "_ff"))
    if "amount" not in out.columns:
        out["amount"] = out.get("close", 0.0) * out.get("volume", 0.0)

    frames = []
    for _, g in out.sort_values(["symbol", "trade_date", "trade_time"]).groupby(
        ["symbol", "trade_date"], sort=False
    ):
        g = g.copy()
        cum_amount = pd.to_numeric(g["amount"], errors="coerce").fillna(0.0).cumsum().replace(0.0, np.nan)
        main = pd.to_numeric(g.get("main_net"), errors="coerce") if "main_net" in g else pd.Series(np.nan, index=g.index)
        sup = pd.to_numeric(g.get("super_net"), errors="coerce") if "super_net" in g else pd.Series(np.nan, index=g.index)
        lrg = pd.to_numeric(g.get("large_net"), errors="coerce") if "large_net" in g else pd.Series(np.nan, index=g.index)
        g["ff_main_net_intensity"] = (main / cum_amount).replace([np.inf, -np.inf], np.nan)
        g["ff_super_large_intensity"] = ((sup.fillna(0.0) + lrg.fillna(0.0)) / cum_amount).replace([np.inf, -np.inf], np.nan)
        main_delta = main.diff()
        sup_delta = sup.diff()
        g["ff_main_net_delta_z"] = _rolling_zscore(main_delta.fillna(0.0), 20)
        g["ff_super_net_delta_z"] = _rolling_zscore(sup_delta.fillna(0.0), 20)
        ret1 = pd.to_numeric(g["close"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
        g["ff_flow_price_divergence"] = g["ff_main_net_intensity"].diff().fillna(0.0) - ret1
        g["ff_main_net_accel"] = main_delta.diff()
        frames.append(g)
    merged = pd.concat(frames, ignore_index=True) if frames else out
    return merged.drop(columns=[c for c in FUNDFLOW_RAW_COLUMNS if c in merged.columns], errors="ignore")


def build_causal_intraday_feature_frame(
    minute_panel: pd.DataFrame,
    *,
    include_level2: bool = False,
    round_lot: int = 100,
) -> pd.DataFrame:
    """Build causal per-minute features for the EV Do-T engine.

    The function never shifts data backward.  Columns that require external
    market, industry, or 20-day reference data are filled only when the caller
    supplies the needed columns; otherwise they remain ``NaN``.  Level-2 fields
    are copied only when present and ``include_level2=True``.
    """
    if minute_panel is None or minute_panel.empty:
        return pd.DataFrame(columns=list(CAUSAL_INTRADAY_FEATURE_COLUMNS))
    required = {"symbol", "trade_time", "close"}
    missing = required.difference(minute_panel.columns)
    if missing:
        raise ValueError(f"minute_panel missing required columns: {sorted(missing)}")
    panel = minute_panel.copy()
    panel["symbol"] = panel["symbol"].astype(str)
    panel["trade_time"] = pd.to_datetime(panel["trade_time"], errors="coerce")
    if "trade_date" not in panel.columns:
        panel["trade_date"] = panel["trade_time"].dt.normalize()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close", "volume", "amount", "pre_close", "limit_up", "limit_down"):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    if "open" not in panel.columns:
        panel["open"] = panel["close"]
    if "high" not in panel.columns:
        panel["high"] = panel[["open", "close"]].max(axis=1)
    if "low" not in panel.columns:
        panel["low"] = panel[["open", "close"]].min(axis=1)
    if "volume" not in panel.columns:
        panel["volume"] = 0.0
    if "amount" not in panel.columns:
        panel["amount"] = panel["close"] * panel["volume"]
    panel = panel.dropna(subset=["symbol", "trade_time", "trade_date", "close"])
    if panel.empty:
        return pd.DataFrame(columns=list(CAUSAL_INTRADAY_FEATURE_COLUMNS))
    panel = panel.sort_values(["symbol", "trade_date", "trade_time"]).reset_index(drop=True)

    frames = []
    for _, g in panel.groupby(["symbol", "trade_date"], sort=False):
        frames.append(_features_one_day(g, include_level2=include_level2, round_lot=round_lot))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=panel.columns)


def _features_one_day(g: pd.DataFrame, *, include_level2: bool, round_lot: int) -> pd.DataFrame:
    out = g.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].fillna(0.0).astype(float).clip(lower=0.0)
    amount = out["amount"].fillna(close * volume).astype(float)

    day_open = float(open_.iloc[0]) if len(open_) else np.nan
    pre_close = _preclose_series(out, day_open)
    vwap = _causal_vwap(close, volume, amount)
    vwap = pd.Series(vwap, index=out.index).replace([np.inf, -np.inf], np.nan).fillna(close.expanding().mean())
    vwap_dev = close - vwap
    vwap_dev_std = vwap_dev.rolling(20, min_periods=3).std().replace(0.0, np.nan)

    running_high = high.cummax()
    running_low = low.cummin()
    running_range = (running_high - running_low).replace(0.0, np.nan)
    ret1 = close.pct_change(fill_method=None)

    out["price_vs_vwap_z"] = (vwap_dev / vwap_dev_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["price_vs_open"] = close / max(day_open, 1e-12) - 1.0
    out["price_vs_preclose"] = close / pre_close.replace(0.0, np.nan) - 1.0
    out["intraday_return"] = out["price_vs_open"]
    out["intraday_percentile_since_open"] = ((close - running_low) / running_range).clip(0.0, 1.0).fillna(0.5)
    out["distance_to_high_of_day"] = close / running_high.replace(0.0, np.nan) - 1.0
    out["distance_to_low_of_day"] = close / running_low.replace(0.0, np.nan) - 1.0
    for win in (3, 5, 10, 20):
        out[f"rolling_return_{win}m"] = close / close.shift(win).replace(0.0, np.nan) - 1.0
    for win in (5, 10, 20):
        out[f"rolling_volatility_{win}m"] = ret1.rolling(win, min_periods=3).std().fillna(0.0) * np.sqrt(win)
    minute = out["trade_time"].dt.hour * 60 + out["trade_time"].dt.minute
    out["minute_of_day"] = minute
    out["session_flag"] = np.select(
        [minute < 11 * 60 + 30, minute < 14 * 60 + 40],
        [0, 1],
        default=2,
    )
    out["gap_open"] = day_open / pre_close.replace(0.0, np.nan) - 1.0

    limit_up = _limit_series(out, pre_close, up=True)
    limit_down = _limit_series(out, pre_close, up=False)
    out["limit_up_distance"] = limit_up / close.replace(0.0, np.nan) - 1.0
    out["limit_down_distance"] = close / limit_down.replace(0.0, np.nan) - 1.0

    out["volume_zscore_5m"] = _rolling_zscore(volume, 5)
    out["volume_zscore_20m"] = _rolling_zscore(volume, 20)
    if "float_shares" in out.columns:
        float_shares = pd.to_numeric(out["float_shares"], errors="coerce").replace(0.0, np.nan)
        out["turnover_intraday"] = volume.cumsum() / float_shares
    else:
        out["turnover_intraday"] = volume.cumsum()
    if "avg_20d_volume" in out.columns:
        avg20 = pd.to_numeric(out["avg_20d_volume"], errors="coerce").replace(0.0, np.nan)
        out["relative_volume_vs_20d"] = volume.cumsum() / avg20
    else:
        out["relative_volume_vs_20d"] = np.nan
    out["amount_zscore"] = _rolling_zscore(amount, 20)
    out["volume_price_divergence"] = out["volume_zscore_5m"] - out["rolling_return_5m"].fillna(0.0) * 100.0
    out["estimated_spread_bps"] = ((high - low).abs() / close.replace(0.0, np.nan) * 10_000.0).rolling(5, min_periods=1).median().clip(2.0, 100.0)
    if "target_qty" in out.columns:
        target_qty = pd.to_numeric(out["target_qty"], errors="coerce").fillna(0.0)
        out["volume_capacity_ratio"] = target_qty / (volume * 0.05).replace(0.0, np.nan)
    else:
        out["volume_capacity_ratio"] = round_lot / (volume * 0.05).replace(0.0, np.nan)

    _attach_relative_features(out, close, vwap, volume)
    _attach_failure_risk_features(out)

    if include_level2:
        for col in LEVEL2_FEATURE_COLUMNS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    else:
        out = out.drop(columns=[c for c in LEVEL2_FEATURE_COLUMNS if c in out.columns], errors="ignore")

    for col in CAUSAL_INTRADAY_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _preclose_series(g: pd.DataFrame, day_open: float) -> pd.Series:
    if "pre_close" in g.columns:
        s = pd.to_numeric(g["pre_close"], errors="coerce")
        if s.notna().any():
            return s.ffill().bfill()
    return pd.Series([day_open] * len(g), index=g.index, dtype=float)


def _limit_series(g: pd.DataFrame, pre_close: pd.Series, *, up: bool) -> pd.Series:
    col = "limit_up" if up else "limit_down"
    if col in g.columns:
        s = pd.to_numeric(g[col], errors="coerce")
        if s.notna().any():
            return s.ffill().bfill()
    band = _symbol_limit_band(str(g["symbol"].iloc[0]) if len(g) else "")
    return pre_close * (1.0 + band if up else 1.0 - band)


def _causal_vwap(close: pd.Series, volume: pd.Series, amount: pd.Series) -> pd.Series:
    fallback_pv = close * volume
    per_unit_price = amount.where(amount >= 0) / volume.replace(0.0, np.nan)
    price_ratio = (per_unit_price / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    median_ratio = float(price_ratio.dropna().median()) if price_ratio.notna().any() else np.nan
    use_amount = np.isfinite(median_ratio) and 0.25 <= median_ratio <= 4.0
    pv = amount.where(amount >= 0).fillna(fallback_pv) if use_amount else fallback_pv
    cum_vol = volume.cumsum()
    cum_pv = pv.cumsum()
    vwap = np.where(
        cum_vol > 0,
        cum_pv / np.maximum(cum_vol, 1e-12),
        close.expanding().mean().to_numpy(dtype=float),
    )
    return pd.Series(vwap, index=close.index, dtype=float)


def _symbol_limit_band(symbol: str) -> float:
    s = str(symbol)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("8", "4")):
        return 0.30
    return 0.10


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=3).mean()
    std = s.rolling(window, min_periods=3).std().replace(0.0, np.nan)
    return ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _attach_relative_features(out: pd.DataFrame, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> None:
    stock_ret = close.pct_change(fill_method=None).fillna(0.0)
    if "index_return" in out.columns:
        out["stock_return_minus_index"] = stock_ret - pd.to_numeric(out["index_return"], errors="coerce")
    else:
        out["stock_return_minus_index"] = np.nan
    if "industry_return" in out.columns:
        out["stock_return_minus_industry"] = stock_ret - pd.to_numeric(out["industry_return"], errors="coerce")
    else:
        out["stock_return_minus_industry"] = np.nan
    stock_vwap_dev = close / vwap.replace(0.0, np.nan) - 1.0
    if "industry_vwap_dev" in out.columns:
        out["stock_vwap_dev_minus_industry"] = stock_vwap_dev - pd.to_numeric(out["industry_vwap_dev"], errors="coerce")
    else:
        out["stock_vwap_dev_minus_industry"] = np.nan
    if "industry_volume_per_symbol" in out.columns:
        denom = pd.to_numeric(out["industry_volume_per_symbol"], errors="coerce").replace(0.0, np.nan)
        out["relative_volume_vs_industry"] = volume / denom
    else:
        out["relative_volume_vs_industry"] = np.nan
    for col in (
        "market_breadth_intraday",
        "up_down_ratio_intraday",
        "limit_up_count",
        "limit_down_count",
        "microcap_style_strength",
        "largecap_style_strength",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else np.nan


def _attach_failure_risk_features(out: pd.DataFrame) -> None:
    ret5 = out["rolling_return_5m"].fillna(0.0)
    ret20 = out["rolling_return_20m"].fillna(0.0)
    vol20 = out["rolling_volatility_20m"].replace(0.0, np.nan)
    breakout = (out["distance_to_high_of_day"].abs() < 0.001).astype(float)
    breakdown = (out["distance_to_low_of_day"].abs() < 0.001).astype(float)
    trend = (ret20.abs() / vol20).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 5.0)
    one_way = (trend / 5.0).clip(0.0, 1.0)
    mean_rev = (1.0 - one_way).clip(0.0, 1.0)
    near_limit = ((out["limit_up_distance"] < 0.005) | (out["limit_down_distance"] < 0.005)).astype(float)
    out["trend_strength_score"] = trend
    out["one_way_trend_probability"] = one_way
    out["mean_reversion_probability"] = mean_rev
    out["momentum_persistence"] = (ret5 > 0).rolling(5, min_periods=1).mean().fillna(0.0)
    out["failed_breakout_probability"] = (breakout * (out["volume_zscore_20m"] < 0).astype(float)).rolling(5, min_periods=1).max()
    out["failed_breakdown_probability"] = (breakdown * (out["volume_zscore_20m"] < 0).astype(float)).rolling(5, min_periods=1).max()
    out["after_sell_new_high_risk"] = (one_way * (ret20 > 0).astype(float) + near_limit).clip(0.0, 1.0)
    out["after_buy_breakdown_risk"] = (one_way * (ret20 < 0).astype(float) + near_limit).clip(0.0, 1.0)
    out["near_limit_risk"] = near_limit


__all__ = [
    "CAUSAL_INTRADAY_FEATURE_COLUMNS",
    "LEVEL2_FEATURE_COLUMNS",
    "FUNDFLOW_FEATURE_COLUMNS",
    "FUNDFLOW_RAW_COLUMNS",
    "IntradayFeatures",
    "build_causal_intraday_feature_frame",
    "compute_intraday_features",
    "features_frame",
    "merge_fundflow_features",
]
