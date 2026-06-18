"""Selective 做T (T+0) FSM — the v2 of ``intraday_dot_strategy`` built from the
negative result of the blanket overlay (2026-06-12: −14.1%/yr uplift, entry
rate 87%, hit 25%).

Three structural fixes:

1. **Selectivity gates** — the blanket dip trigger fired almost every
   name-day and lost the 26bps round-trip cost. A leg is only attempted when
   the (vol / trend / open-gap / regime) gates all pass, so the overlay
   concentrates on name-days whose conditional edge can clear costs.
2. **Vol-adaptive levels** — dip depth, target and stop scale with the
   name's recent ATR%. Fixed 1.5%/1.2% never reaches target on a 1%-range
   name and is noise on a 6%-range name.
3. **Honest intrabar fills** — a limit buy never fills better than the bar
   open; a stop gaps through at the bar open; target fills at
   ``max(open, target)``. Trigger comparisons use the PREVIOUS bar's running
   VWAP (strictly causal at minute granularity).

Two leg modes, both T+1-legal on a held base position:

* ``dip_buy``    (正T): buy the morning dip below running VWAP, sell the same
  notional later (the sold shares are pre-existing inventory).
* ``spike_sell`` (反T): sell pre-existing shares into a morning spike above
  running VWAP, buy the notional back later.

Research/backtest + live signal only — never emits orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DayContext:
    """Per (symbol, day) conditioning features, all PIT-safe.

    ``atr_pct`` / ``mom_5d`` come from data up to t-1; ``gap_open`` uses
    today's open which is known the moment the leg could first trigger.
    """

    atr_pct: float
    mom_5d: float
    gap_open: float
    regime: str = "sideways"


@dataclass(frozen=True)
class SelectiveDotParams:
    mode: str = "auto"                 # dip_buy / spike_sell / auto (trend sign)
    dip_atr_mult: float = 0.30         # 入场偏离: VWAP·(1 ∓ mult·atr_pct)
    target_atr_mult: float = 0.50      # 止盈距离: entry·(1 ± mult·atr_pct)
    stop_atr_mult: float = 0.50        # 止损距离: entry·(1 ∓ mult·atr_pct)
    morning_deadline: str = "10:30:00"
    eod_close: str = "14:50:00"
    min_bars_before_entry: int = 5
    # ---- selectivity gates ----
    min_atr_pct: float = 0.02          # 波动太小连成本都吃不回来
    min_mom_5d: float = 0.0            # dip_buy 仅在5日趋势≥此值的票
    max_mom_5d: float = 0.0            # spike_sell 仅在5日趋势≤此值的票
    max_abs_gap: float = 0.04          # 跳空过大(情绪极端)当日不做T
    regimes: tuple[str, ...] = ("bull", "sideways")


@dataclass(frozen=True)
class SelectiveDotResult:
    symbol: str
    trade_date: str
    mode: str | None          # dip_buy / spike_sell / None (gated out)
    state: str                # gated_out / waiting_no_entry / closed_profit / closed_stop / closed_eod
    gate_reason: str | None
    entry_time: str | None
    entry_px: float | None
    exit_time: str | None
    exit_px: float | None
    exit_reason: str | None
    ret: float | None         # sell_px/buy_px − 1 (round-trip gross, both modes)

    def as_dict(self) -> dict:
        return asdict(self)


def check_gates(ctx: DayContext, params: SelectiveDotParams) -> tuple[str | None, str | None]:
    """Resolve (mode, gate_reason). mode=None means the day is gated out."""
    if not np.isfinite(ctx.atr_pct) or ctx.atr_pct < params.min_atr_pct:
        return None, "low_vol"
    if ctx.regime not in params.regimes:
        return None, f"regime_{ctx.regime}"
    if np.isfinite(ctx.gap_open) and abs(ctx.gap_open) > params.max_abs_gap:
        return None, "extreme_gap"
    if params.mode == "dip_buy":
        if not np.isfinite(ctx.mom_5d) or ctx.mom_5d < params.min_mom_5d:
            return None, "weak_trend"
        return "dip_buy", None
    if params.mode == "spike_sell":
        if not np.isfinite(ctx.mom_5d) or ctx.mom_5d > params.max_mom_5d:
            return None, "strong_trend"
        return "spike_sell", None
    # auto: follow the trend sign
    if not np.isfinite(ctx.mom_5d):
        return None, "no_trend_data"
    return ("dip_buy" if ctx.mom_5d >= params.min_mom_5d else "spike_sell"), None


def _hms(ts) -> str:
    s = str(ts)
    return s.split(" ")[-1] if " " in s else s


def prepare_day_arrays(bars: pd.DataFrame) -> dict | None:
    """Convert one symbol-day of minute bars into reusable numpy arrays.

    Includes the lagged running VWAP so batch backtests can sweep FSM
    parameters without recomputing it per config.
    """
    b = bars
    if b is None or len(b) == 0 or "close" not in b.columns:
        return None
    cols = {}
    for c in ("open", "high", "low", "close", "volume", "amount"):
        cols[c] = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype="float64") \
            if c in b.columns else None
    if cols["close"] is None or cols["high"] is None or cols["low"] is None:
        return None
    times = np.array([_hms(t) for t in b["trade_time"]]) if "trade_time" in b.columns \
        else np.array([""] * len(b))
    order = np.argsort(times, kind="stable")
    ok = np.isfinite(cols["close"][order]) & np.isfinite(cols["high"][order]) & np.isfinite(cols["low"][order])
    idx = order[ok]
    if idx.size == 0:
        return None
    out = {c: (cols[c][idx] if cols[c] is not None else None) for c in cols}
    if out["open"] is None:
        out["open"] = out["close"].copy()
    if out["volume"] is None:
        out["volume"] = np.zeros_like(out["close"])
    out["volume"] = np.nan_to_num(out["volume"], nan=0.0)
    if out["amount"] is None:
        out["amount"] = out["close"] * out["volume"]
    out["amount"] = np.nan_to_num(out["amount"], nan=0.0)
    out["time"] = times[idx]
    cum_v = np.cumsum(out["volume"])
    cum_pv = np.cumsum(out["close"] * out["volume"])
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), out["close"])
    out["vwap_prev"] = np.concatenate([[np.nan], vwap[:-1]])
    return out


def simulate_prepared(
    day: dict,
    atr_pct: float,
    params: SelectiveDotParams,
    mode: str,
) -> tuple[str, float | None, float | None, float | None, int | None, int | None]:
    """Core causal FSM on prepared arrays (gates NOT applied here).

    Returns (state, entry_px, exit_px, ret, entry_idx, exit_idx).
    """
    o, h, l, c, t = day["open"], day["high"], day["low"], day["close"], day["time"]
    vwap_prev = day["vwap_prev"]
    n = len(c)
    p = params
    atr = float(atr_pct)

    deadline_idx = int(np.searchsorted(t, p.morning_deadline, side="right"))
    lo = max(1, int(p.min_bars_before_entry))
    if deadline_idx <= lo:
        return "waiting_no_entry", None, None, None, None, None
    sl = slice(lo, deadline_idx)
    if mode == "dip_buy":
        trig = vwap_prev[sl] * (1.0 - p.dip_atr_mult * atr)
        hit = np.isfinite(trig) & (l[sl] <= trig)
    else:
        trig = vwap_prev[sl] * (1.0 + p.dip_atr_mult * atr)
        hit = np.isfinite(trig) & (h[sl] >= trig)
    if not hit.any():
        return "waiting_no_entry", None, None, None, None, None
    e = lo + int(np.argmax(hit))
    trig_e = float(trig[e - lo])
    entry_px = min(trig_e, float(o[e])) if mode == "dip_buy" else max(trig_e, float(o[e]))
    if not np.isfinite(entry_px) or entry_px <= 0:
        return "waiting_no_entry", None, None, None, None, None

    if mode == "dip_buy":
        target = entry_px * (1.0 + p.target_atr_mult * atr)
        stop = entry_px * (1.0 - p.stop_atr_mult * atr)
    else:
        target = entry_px * (1.0 - p.target_atr_mult * atr)
        stop = entry_px * (1.0 + p.stop_atr_mult * atr)

    j0 = e + 1
    if j0 >= n:
        ret = _rt(mode, entry_px, float(c[-1]))
        return "closed_eod", entry_px, float(c[-1]), ret, e, n - 1
    if mode == "dip_buy":
        stop_hits = l[j0:] <= stop
        tgt_hits = h[j0:] >= target
    else:
        stop_hits = h[j0:] >= stop
        tgt_hits = l[j0:] <= target
    js = int(np.argmax(stop_hits)) if stop_hits.any() else n
    jt = int(np.argmax(tgt_hits)) if tgt_hits.any() else n
    je = int(np.searchsorted(t[j0:], p.eod_close, side="left"))
    je = min(je, n - 1 - j0)

    first = min(js, jt, je)
    bar = j0 + first
    if first == js and js <= jt:
        px = min(stop, float(o[bar])) if mode == "dip_buy" else max(stop, float(o[bar]))
        state = "closed_stop"
    elif first == jt:
        px = max(target, float(o[bar])) if mode == "dip_buy" else min(target, float(o[bar]))
        state = "closed_profit"
    else:
        px = float(c[bar])
        state = "closed_eod"
    return state, entry_px, float(px), _rt(mode, entry_px, px), e, bar


def simulate_selective_dot_day(
    bars: pd.DataFrame,
    ctx: DayContext,
    params: SelectiveDotParams | None = None,
    *,
    symbol: str | None = None,
) -> SelectiveDotResult:
    """Causal selective 做T over one symbol-day of 1-minute bars."""
    p = params or SelectiveDotParams()
    sym = str(symbol or (bars["symbol"].iloc[0] if "symbol" in bars.columns and len(bars) else ""))
    tdate = str(bars["trade_date"].iloc[0]) if "trade_date" in bars.columns and len(bars) else ""

    mode, gate_reason = check_gates(ctx, p)
    if mode is None:
        return SelectiveDotResult(sym, tdate, None, "gated_out", gate_reason,
                                  None, None, None, None, None, None)

    day = prepare_day_arrays(bars)
    if day is None:
        return SelectiveDotResult(sym, tdate, mode, "waiting_no_entry", None,
                                  None, None, None, None, None, None)
    state, entry_px, exit_px, ret, e, bar = simulate_prepared(day, ctx.atr_pct, p, mode)
    if state == "waiting_no_entry":
        return SelectiveDotResult(sym, tdate, mode, state, None,
                                  None, None, None, None, None, None)
    reason = {"closed_stop": "止损", "closed_profit": "止盈", "closed_eod": "尾盘强平"}[state]
    return SelectiveDotResult(sym, tdate, mode, state, None, str(day["time"][e]),
                              round(float(entry_px), 4), str(day["time"][bar]),
                              round(float(exit_px), 4), reason, ret)


def _rt(mode: str, entry_px: float, exit_px: float) -> float:
    """Round-trip gross return = sell/buy − 1 for both leg orders."""
    if mode == "dip_buy":
        return round(float(exit_px) / float(entry_px) - 1.0, 5)
    return round(float(entry_px) / float(exit_px) - 1.0, 5)


def live_selective_dot_action(
    bars_so_far: pd.DataFrame,
    ctx: DayContext,
    params: SelectiveDotParams | None = None,
    *,
    symbol: str | None = None,
    in_position: bool = False,
) -> dict:
    """LIVE: replay the causal FSM on bars-so-far and emit this-minute action.

    ``in_position`` = whether the T leg is already on. Returns
    {action, mode, level, reason}; action ∈ {加T买入, 反T卖出, 减T止盈,
    回补止盈, 止损, 尾盘强平, 持有, 观望, 不做T}.
    """
    p = params or SelectiveDotParams()
    res = simulate_selective_dot_day(bars_so_far, ctx, p, symbol=symbol)
    if res.state == "gated_out":
        return {"action": "不做T", "mode": None, "level": None,
                "reason": f"闸门未过({res.gate_reason})"}
    last_t = _hms(bars_so_far["trade_time"].iloc[-1]) if len(bars_so_far) else ""
    if not in_position:
        if res.entry_time is not None and res.exit_time is None:
            act = "加T买入" if res.mode == "dip_buy" else "反T卖出"
            return {"action": act, "mode": res.mode, "level": res.entry_px,
                    "reason": "触发入场(偏离运行VWAP达ATR阈值)"}
        return {"action": "观望", "mode": res.mode, "level": None,
                "reason": "未触发或已过早盘窗口"}
    if res.state == "closed_profit":
        act = "减T止盈" if res.mode == "dip_buy" else "回补止盈"
        return {"action": act, "mode": res.mode, "level": res.exit_px, "reason": "触及止盈"}
    if res.state == "closed_stop":
        return {"action": "止损", "mode": res.mode, "level": res.exit_px, "reason": "触及止损"}
    if last_t >= p.eod_close:
        px = float(bars_so_far["close"].iloc[-1])
        return {"action": "尾盘强平", "mode": res.mode, "level": px, "reason": "尾盘未达目标"}
    return {"action": "持有", "mode": res.mode, "level": None, "reason": "等止盈/止损"}


def build_day_contexts(panel: pd.DataFrame, *, atr_window: int = 14) -> pd.DataFrame:
    """Compute PIT-safe DayContext columns from the daily market panel.

    Returns a frame keyed (symbol, trade_date) with atr_pct / mom_5d /
    gap_open / prev_close / regime. atr_pct and mom_5d only use data up to
    t-1.
    """
    df = panel[["symbol", "trade_date", "open", "high", "low", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    g = df.groupby("symbol", sort=False)
    prev_close = g["close"].shift(1)
    tr_pct = (df["high"] - df["low"]) / prev_close
    df["atr_pct"] = (
        tr_pct.groupby(df["symbol"]).rolling(atr_window, min_periods=5).mean()
        .reset_index(level=0, drop=True).groupby(df["symbol"]).shift(1)
    )
    df["mom_5d"] = g["close"].shift(1) / g["close"].shift(6) - 1.0
    df["gap_open"] = df["open"] / prev_close - 1.0

    px = df.pivot_table(index="trade_date", columns="symbol", values="close")
    bench = px.pct_change(fill_method=None).mean(axis=1)
    cum = (1 + bench.fillna(0)).cumprod().shift(1)
    trail = cum / cum.shift(60) - 1.0
    regime = pd.Series(np.where(trail > 0.05, "bull",
                                np.where(trail < -0.05, "bear", "sideways")),
                       index=px.index)
    df["regime"] = df["trade_date"].map(regime).fillna("sideways")
    df["prev_close"] = prev_close
    return df[["symbol", "trade_date", "atr_pct", "mom_5d", "gap_open", "prev_close", "regime"]]


__all__ = [
    "DayContext", "SelectiveDotParams", "SelectiveDotResult",
    "check_gates", "prepare_day_arrays", "simulate_prepared",
    "simulate_selective_dot_day", "live_selective_dot_action",
    "build_day_contexts",
]
