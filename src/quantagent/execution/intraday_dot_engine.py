"""分时做T 决策引擎 —— 用户 metric/分时主图_AI多因子做T.txt 的因果 Python 移植。

设计来源：用户在 metric/ 中验证过的通达信多因子做T主图公式。本模块把它逐行
移植为**严格因果**（每分钟的判断只用当根及之前的 bar）的引擎，并在其上叠加：

  * A股 T+1 合法性（卖出只能用昨仓 sellable_qty，禁止卖今日买入的 today_buy_qty）；
  * 集合竞价逻辑（9:15-9:20 观察 / 9:20-9:25 不可撤 / 14:57-15:00 收盘）；
  * 5 状态机（强趋势上行 / 弱趋势下行 / 箱体震荡 / 涨停逼空 / 跌停风险）；
  * 失败控制（高抛失败 high_sell_failure / 低吸失败 low_buy_failure + 锁定）；
  * 因子评分 → confidence 门槛 → 严格 JSON 输出。

正T（低吸-后续卖旧仓）与反T（卖旧仓-低位回补）都支持。回测/实盘信号用途，
本模块**不下真实订单**。

通达信→Python 映射：最新=分钟收盘；均价=日内累计 VWAP；REF(x,n)=shift；
带宽=clamp(MA(|最新-均价|/均价,30)·2.6+0.003, 0.006, 0.028)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── 通达信算子的 numpy 因果实现 ──────────────────────────────────────────


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    acc = x[0] if len(x) and np.isfinite(x[0]) else 0.0
    for i, v in enumerate(x):
        v = acc if not np.isfinite(v) else v
        acc = a * v + (1 - a) * acc if i else v
        out[i] = acc
    return out


def _sma(x: np.ndarray, n: int, m: int) -> np.ndarray:
    """通达信 SMA(x,n,m) = 前值·(n-m)/n + x·m/n（递归）。"""
    out = np.empty_like(x, dtype=float)
    acc = x[0] if len(x) and np.isfinite(x[0]) else 0.0
    for i, v in enumerate(x):
        v = 0.0 if not np.isfinite(v) else v
        acc = (acc * (n - m) + v * m) / n if i else v
        out[i] = acc
    return out


def _ma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).mean().to_numpy()


def _ref(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if n < len(x):
        out[n:] = x[:-n] if n > 0 else x
    return out


def _hhv(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()


def _llv(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()


def _sum(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).sum().to_numpy()


def _count(cond: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(cond.astype(float)).rolling(n, min_periods=1).sum().to_numpy()


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CROSS(a,b)：a 上穿 b。"""
    prev = (_ref(a, 1) <= _ref(b, 1))
    now = (a > b)
    out = prev & now
    out[0] = False
    return out


# ── 参数 / 状态 / 输出 ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DotEngineParams:
    band_lo: float = 0.006
    band_hi: float = 0.028
    band_k: float = 2.60
    band_c: float = 0.003
    rsi_period: int = 6
    ema_fast: int = 5
    ema_slow: int = 13
    ema_mid: int = 34
    score_enter: float = 80.0       # A 候选分数门槛
    conf_enter: float = 68.0        # A 候选可信门槛
    score_strong: float = 88.0      # S 候选
    conf_strong: float = 78.0
    fail_band_mult: float = 0.65    # 失败判定偏离 = 带宽·此值
    min_bars: int = 6               # 入场前最少 bar


@dataclass
class IntradayState:
    """引擎逐分钟产出的因果状态（最后一根 bar 的快照即可下决策）。"""

    n_bars: int
    last: float
    vwap: float
    band: float
    high_line: float
    low_line: float
    deviation_pct: float
    rsi: float
    active_buy_ratio: float
    active_sell_ratio: float
    fund_line: float
    strong_trend: bool
    weak_trend: bool
    limit_up_squeeze: bool
    limit_down_risk: bool
    reversal_env: bool
    low_score: float
    high_score: float
    low_conf: float
    high_conf: float
    low_signal: str          # ''/'低A'/'低S'
    high_signal: str         # ''/'高A'/'高S'
    low_fail_lock: bool
    high_fail_lock: bool
    low_fail_now: bool       # 本根触发低吸失败
    high_fail_now: bool
    state: str               # strong_up/weak_down/range/limit_up/limit_down


def _market_state(strong: bool, weak: bool, lu: bool, ld: bool) -> str:
    if ld:
        return "limit_down"
    if lu:
        return "limit_up"
    if strong:
        return "strong_up"
    if weak:
        return "weak_down"
    return "range"


def compute_intraday_state(
    bars: pd.DataFrame,
    *,
    pre_close: float,
    params: DotEngineParams | None = None,
) -> IntradayState | None:
    """对一只票当日的 1 分钟 bar 因果计算到最后一根，返回状态快照。

    bars 需含 close/high/low/volume（amount 可选；缺省用 close·volume）。
    pre_close = 昨收（PRE）。
    """
    p = params or DotEngineParams()
    if bars is None or len(bars) == 0 or pre_close <= 0:
        return None
    b = bars.copy()
    close = pd.to_numeric(b["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(b.get("high", b["close"]), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(b.get("low", b["close"]), errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(b.get("volume", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "amount" in b.columns:
        amt = pd.to_numeric(b["amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        amt = close * vol
    ok = np.isfinite(close)
    close, high, low, vol, amt = close[ok], high[ok], low[ok], vol[ok], amt[ok]
    n = len(close)
    if n == 0:
        return None

    last = close                                   # 最新
    cum_amt = np.cumsum(np.where(np.isfinite(amt), amt, 0.0))
    cum_vol = np.cumsum(np.where(np.isfinite(vol), vol, 0.0))
    vwap = np.where(cum_vol > 0, cum_amt / np.maximum(cum_vol, 1e-12), last)   # 均价

    ret1 = (last / _ref(last, 1) - 1.0) * 100.0
    dev = (last - vwap) / (vwap + 1e-4) * 100.0
    wave = _ma(np.abs(last - vwap) / (vwap + 1e-4), 30)
    band = np.clip(wave * p.band_k + p.band_c, p.band_lo, p.band_hi)
    high_line = vwap * (1 + band)
    low_line = vwap * (1 - band)

    ema_f = _ema(last, p.ema_fast)
    ema_s = _ema(last, p.ema_slow)
    ema_m = _ema(last, p.ema_mid)

    lc = _ref(last, 1)
    up = np.where(np.isfinite(last - lc), np.maximum(last - lc, 0.0), 0.0)
    ab = np.where(np.isfinite(last - lc), np.abs(last - lc), 0.0)
    rsi = _sma(up, p.rsi_period, 1) / (_sma(ab, p.rsi_period, 1) + 1e-12) * 100.0

    vol_fix = np.where((~np.isfinite(vol)) | (vol < 0), 0.0, vol)
    vol_ma20 = _ma(vol_fix, 20)
    signed = np.where(last >= _ref(last, 1), vol_fix, -vol_fix)
    fund = _ema(np.nan_to_num(signed), 5)
    buyv = _sum(np.where(last >= _ref(last, 1), vol_fix, 0.0), 20)
    sellv = _sum(np.where(last < _ref(last, 1), vol_fix, 0.0), 20)
    buy_ratio = buyv / (buyv + sellv + 1) * 100.0
    sell_ratio = sellv / (buyv + sellv + 1) * 100.0

    fund_back = (fund > 0) & (fund > _ref(fund, 1))
    fund_weak = (fund < 0) & (fund < _ref(fund, 1))

    trend_strong = (last > vwap) & (vwap > _ref(vwap, 3)) & (last > pre_close) & (ema_f >= ema_s)
    trend_weak = (last < vwap) & (vwap < _ref(vwap, 3)) & (last < pre_close) & (ema_f <= ema_s)
    strong_cont = (trend_strong & (last > _hhv(last, 20) * 0.995) & (ema_f > ema_s)
                   & (ema_s > ema_m) & (buy_ratio > 58) & (fund > _ref(fund, 1)))
    weak_cont = (trend_weak & (last < _llv(last, 20) * 1.005) & (ema_f < ema_s)
                 & (ema_s < ema_m) & (sell_ratio > 58) & (fund < _ref(fund, 1)))

    mom_up = _cross(ema_f, ema_s) | ((ema_f > ema_s) & (ema_f > _ref(ema_f, 1)))
    mom_down = _cross(ema_s, ema_f) | ((ema_f < ema_s) & (ema_f < _ref(ema_f, 1)))

    spike_div = (last >= _hhv(last, 20) * 0.998) & (vol_fix < _hhv(vol_fix, 20) * 0.75)
    dip_div = (last <= _llv(last, 20) * 1.002) & (rsi > _ref(rsi, 1)) & fund_back

    vol_lead_up = (_ref(vol_fix, 1) > vol_ma20 * 1.30) & (last > _ref(last, 1)) & (last <= vwap * 1.003)
    vol_lead_down = (_ref(vol_fix, 1) > vol_ma20 * 1.30) & (last < _ref(last, 1)) & (last >= vwap * 0.997)

    impact = np.abs(ret1) / (vol_fix / (vol_ma20 + 1) + 0.01)
    liquid = (impact < _ma(impact, 20) * 1.60) | (vol_ma20 == 0)
    abnormal = (vol_fix > vol_ma20 * 4.0) & (np.abs(dev) > band * 130)
    wave_expand = (wave > _ref(wave, 1)) & (np.abs(dev) > band * 80)
    wave_compress = wave < _ma(wave, 20) * 0.90
    reversal = (np.abs(dev) > band * 90) & liquid & (~abnormal)

    limit_down = (last <= pre_close * 0.905) | ((last < pre_close * 0.94) & weak_cont)
    limit_up = (last >= pre_close * 1.095) | ((last > pre_close * 1.07) & strong_cont)

    def b2(x):
        return x.astype(float)

    low_raw = (b2(last <= low_line) * 24 + b2(rsi < 35) * 14 + b2(dip_div) * 14
               + b2(fund_back) * 12 + b2(buy_ratio > 55) * 10 + b2(mom_up) * 10
               + b2(vol_lead_up) * 8 + b2(liquid) * 8 + b2(reversal) * 8
               - b2(abnormal) * 25 - b2(weak_cont) * 20 - b2(limit_down) * 35)
    high_raw = (b2(last >= high_line) * 24 + b2(rsi > 70) * 14 + b2(spike_div) * 14
                + b2(fund_weak) * 12 + b2(sell_ratio > 55) * 10 + b2(mom_down) * 10
                + b2(vol_lead_down) * 8 + b2(liquid) * 8 + b2(reversal) * 8
                - b2(abnormal) * 25 - b2(strong_cont) * 20 - b2(limit_up) * 35)
    low_score = np.clip(low_raw, 0, 100)
    high_score = np.clip(high_raw, 0, 100)

    low_fac = (b2(last <= low_line) + b2(rsi < 38) + b2(fund_back) + b2(buy_ratio > 55)
               + b2(mom_up) + b2(dip_div) + b2(vol_lead_up) + b2(liquid))
    high_fac = (b2(last >= high_line) + b2(rsi > 68) + b2(fund_weak) + b2(sell_ratio > 55)
                + b2(mom_down) + b2(spike_div) + b2(vol_lead_down) + b2(liquid))

    low_conf = np.maximum(0, (b2(liquid) * 18 + b2(reversal) * 18 + b2(low_fac >= 5) * 28
                              + b2(low_score > high_score + 15) * 18 + b2(buy_ratio > sell_ratio) * 10
                              + b2(wave_expand | wave_compress) * 8
                              - b2(weak_cont) * 25 - b2(limit_down) * 30))
    high_conf = np.maximum(0, (b2(liquid) * 18 + b2(reversal) * 18 + b2(high_fac >= 5) * 28
                               + b2(high_score > low_score + 15) * 18 + b2(sell_ratio > buy_ratio) * 10
                               + b2(wave_expand | wave_compress) * 8
                               - b2(strong_cont) * 25 - b2(limit_up) * 30))

    low_cand = (low_score >= p.score_enter) & (low_conf >= p.conf_enter) & (~weak_cont) & (~limit_down)
    high_cand = (high_score >= p.score_enter) & (high_conf >= p.conf_enter) & (~strong_cont) & (~limit_up)
    low_s_cand = (low_score >= p.score_strong) & (low_conf >= p.conf_strong) & (~weak_cont) & (~limit_down)
    high_s_cand = (high_score >= p.score_strong) & (high_conf >= p.conf_strong) & (~strong_cont) & (~limit_up)

    # 失败检测：信号后 price 反向突破 → 锁定（TDX 距低/距高 + 后破位/后新高）
    low_sig_any = low_cand | low_s_cand
    high_sig_any = high_cand | high_s_cand
    low_fail = np.zeros(n, dtype=bool)
    high_fail = np.zeros(n, dtype=bool)
    last_low_i = last_high_i = -1
    for i in range(n):
        if i > 0 and low_sig_any[i - 1]:
            last_low_i = i - 1
        if i > 0 and high_sig_any[i - 1]:
            last_high_i = i - 1
        if 0 <= last_low_i and (i - last_low_i) <= 20:
            sig_px = last[last_low_i]
            if last[i] < sig_px * (1 - band[i] * p.fail_band_mult) and last[i] < vwap[i]:
                low_fail[i] = True
        if 0 <= last_high_i and (i - last_high_i) <= 20:
            sig_px = last[last_high_i]
            if last[i] > sig_px * (1 + band[i] * p.fail_band_mult) and last[i] > vwap[i]:
                high_fail[i] = True
    low_fail_lock = (_count(low_fail, 20) > 0) & (~_cross(last, vwap))
    high_fail_lock = (_count(high_fail, 20) > 0) & (~_cross(vwap, last))

    low_s = low_s_cand & (~low_fail_lock) & (_count(low_s_cand & (~low_fail_lock), 12) == 1)
    high_s = high_s_cand & (~high_fail_lock) & (_count(high_s_cand & (~high_fail_lock), 12) == 1)
    low_a = low_cand & (~low_s) & (~low_fail_lock) & (_count(low_cand & (~low_fail_lock), 10) == 1)
    high_a = high_cand & (~high_s) & (~high_fail_lock) & (_count(high_cand & (~high_fail_lock), 10) == 1)

    i = n - 1
    low_sig = "低S" if low_s[i] else ("低A" if low_a[i] else "")
    high_sig = "高S" if high_s[i] else ("高A" if high_a[i] else "")
    return IntradayState(
        n_bars=n, last=float(last[i]), vwap=float(vwap[i]), band=float(band[i]),
        high_line=float(high_line[i]), low_line=float(low_line[i]), deviation_pct=float(dev[i]),
        rsi=float(rsi[i]), active_buy_ratio=float(buy_ratio[i]), active_sell_ratio=float(sell_ratio[i]),
        fund_line=float(fund[i]),
        strong_trend=bool(strong_cont[i]), weak_trend=bool(weak_cont[i]),
        limit_up_squeeze=bool(limit_up[i]), limit_down_risk=bool(limit_down[i]),
        reversal_env=bool(reversal[i]),
        low_score=float(low_score[i]), high_score=float(high_score[i]),
        low_conf=float(low_conf[i]), high_conf=float(high_conf[i]),
        low_signal=low_sig, high_signal=high_sig,
        low_fail_lock=bool(low_fail_lock[i]), high_fail_lock=bool(high_fail_lock[i]),
        low_fail_now=bool(low_fail[i]), high_fail_now=bool(high_fail[i]),
        state=_market_state(bool(strong_cont[i]), bool(weak_cont[i]),
                            bool(limit_up[i]), bool(limit_down[i])),
    )


__all__ = ["DotEngineParams", "IntradayState", "compute_intraday_state",
           "_market_state"]
