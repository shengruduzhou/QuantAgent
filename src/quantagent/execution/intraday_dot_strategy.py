"""Executable 做T (T+0) strategy — a CAUSAL intraday state machine, not hindsight.

The descriptive features in ``intraday_features.py`` look at the WHOLE day (good for a
heat map, but optimistic for execution). This module instead replays 1-minute bars in
time order and only ever uses bars up to the current minute, so the same logic runs in a
backtest and live:

  WAITING  — in the morning window, if price pulls back to/below the RUNNING VWAP
             (低吸 dip), enter a T (加T) at that level.
  HOLDING  — exit when price reaches target = entry·(1+target_pct) (止盈 减T), or
             stop = entry·(1−stop_pct) (跌破 invalidation 止损), whichever first.
  FORCED   — if still holding at the EOD cutoff, force-close at that bar (尾盘回补/放弃).

Same-day round-trip on a HELD core position to lower cost / manage risk — never a
manipulative order. Returns the realized T leg (entry/exit/reason/ret), computed causally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DotParams:
    target_pct: float = 0.015          # 止盈: entry·(1+x)
    stop_pct: float = 0.012            # 止损: entry·(1−x)
    dip_buffer: float = 0.002          # 低吸: 价 ≤ 运行VWAP·(1−buffer) 才触发
    morning_deadline: str = "10:00:00"  # 只在早盘窗口找低吸入场
    eod_close: str = "14:50:00"        # 此后仍持有 → 强制平仓
    min_bars_before_entry: int = 5     # 入场前至少累计几根, 让运行VWAP稳定


@dataclass(frozen=True)
class DotResult:
    symbol: str
    trade_date: str
    state: str                # waiting_no_entry / closed_profit / closed_stop / closed_eod
    entry_time: str | None
    entry_px: float | None
    exit_time: str | None
    exit_px: float | None
    exit_reason: str | None
    ret: float | None         # exit/entry − 1 (the realized T-leg round-trip)

    def as_dict(self) -> dict:
        return asdict(self)


def _hms(ts: str) -> str:
    s = str(ts)
    return s.split(" ")[-1] if " " in s else s


def simulate_dot_day(bars: pd.DataFrame, params: DotParams | None = None, *,
                     symbol: str | None = None) -> DotResult:
    """Causal FSM over one symbol's 1-minute bars (one day). Each decision uses only bars
    up to the current minute (no lookahead)."""
    p = params or DotParams()
    sym = str(symbol or (bars["symbol"].iloc[0] if "symbol" in bars.columns else ""))
    tdate = str(bars["trade_date"].iloc[0]) if "trade_date" in bars.columns and len(bars) else ""
    empty = DotResult(sym, tdate, "waiting_no_entry", None, None, None, None, None, None)
    if bars is None or len(bars) == 0 or "close" not in bars.columns:
        return empty
    b = bars.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna(subset=["close", "high", "low"])
    if "trade_time" in b.columns:
        b = b.sort_values("trade_time")
    b = b.reset_index(drop=True)
    if b.empty:
        return empty

    cum_pv = 0.0  # Σ close·vol  (volume-weighted close VWAP → unit-robust)
    cum_v = 0.0
    entry_px = entry_time = None
    for i, r in b.iterrows():
        vol = float(r.get("volume", 0.0) or 0.0)
        cum_pv += float(r["close"]) * vol
        cum_v += vol
        vwap = (cum_pv / cum_v) if cum_v > 0 else float(r["close"])
        t = _hms(r.get("trade_time", ""))

        if entry_px is None:
            # 低吸入场: 仅早盘窗口, 运行VWAP稳定后, 价回踩到 VWAP·(1−buffer)
            if i + 1 >= p.min_bars_before_entry and t <= p.morning_deadline:
                trig = vwap * (1.0 - p.dip_buffer)
                if float(r["low"]) <= trig:
                    entry_px = min(trig, float(r["high"]))  # 触发价(不优于当根最高)
                    entry_time = t
            continue

        # 持仓: 先到先触发 止盈/止损; 尾盘强制平
        target = entry_px * (1.0 + p.target_pct)
        stop = entry_px * (1.0 - p.stop_pct)
        if float(r["low"]) <= stop:
            return DotResult(sym, tdate, "closed_stop", entry_time, round(entry_px, 4), t,
                             round(stop, 4), "止损", round(stop / entry_px - 1.0, 5))
        if float(r["high"]) >= target:
            return DotResult(sym, tdate, "closed_profit", entry_time, round(entry_px, 4), t,
                             round(target, 4), "止盈", round(target / entry_px - 1.0, 5))
        if t >= p.eod_close:
            px = float(r["close"])
            return DotResult(sym, tdate, "closed_eod", entry_time, round(entry_px, 4), t,
                             round(px, 4), "尾盘强平", round(px / entry_px - 1.0, 5))

    if entry_px is not None:  # held to the last bar without hitting a gate → close at last
        px = float(b["close"].iloc[-1])
        return DotResult(sym, tdate, "closed_eod", entry_time, round(entry_px, 4),
                         _hms(b["trade_time"].iloc[-1]) if "trade_time" in b else None,
                         round(px, 4), "尾盘强平", round(px / entry_px - 1.0, 5))
    return DotResult(sym, tdate, "waiting_no_entry", None, None, None, None, None, None)


def live_dot_action(bars_so_far: pd.DataFrame, params: DotParams | None = None, *,
                    symbol: str | None = None, in_position: bool = False) -> dict:
    """LIVE: given the bars UP TO NOW, what to do this minute. ``in_position`` = whether the
    T leg is already on. Returns {action, level, reason}. action ∈ {加T买入, 减T止盈, 止损, 尾盘强平, 持有, 观望}."""
    p = params or DotParams()
    res = simulate_dot_day(bars_so_far, p, symbol=symbol)
    last = bars_so_far.iloc[-1] if len(bars_so_far) else None
    t = _hms(last.get("trade_time", "")) if last is not None else ""
    if not in_position:
        if res.entry_time is not None and res.exit_time is None:
            return {"action": "加T买入", "level": res.entry_px, "reason": "早盘低吸触发(回踩运行VWAP)"}
        return {"action": "观望", "level": None, "reason": "未触发低吸或非早盘窗口"}
    # already holding the T → manage exits
    if res.state == "closed_profit":
        return {"action": "减T止盈", "level": res.exit_px, "reason": "触及止盈目标"}
    if res.state == "closed_stop":
        return {"action": "止损", "level": res.exit_px, "reason": "跌破invalidation"}
    if t >= p.eod_close:
        return {"action": "尾盘强平", "level": float(last["close"]), "reason": "尾盘未达目标,回补/放弃"}
    return {"action": "持有", "level": None, "reason": "持有等止盈/止损"}
