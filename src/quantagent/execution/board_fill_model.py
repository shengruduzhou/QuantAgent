"""集合竞价 / 打板 fill models — honest A-share execution assumptions.

Why this exists: the phantom-alpha audit (2026-06-11) showed ~40% of the
unconstrained model's daily gross came from names that closed sealed
limit-up — returns nobody can buy. Any 打板/竞价 sleeve must therefore be
evaluated under fill assumptions that encode the REAL adverse selection:

* 集合竞价 (open auction): a buy fills at the single auction price. If the
  stock opens AT its limit-up price (一字板), the queue is dominated by
  earlier/larger orders — a retail-sized order is assumed UNFILLED.
  Participation in the auction volume is capped.
* 打板 (limit-up board chasing): a buy limit order at the board price joins
  the END of the queue when placed. While the seal holds there is no way to
  verify queue position from minute bars, so the honest default is:
  **you are filled only if the board breaks (开板) after your order** — the
  seal collapsing is exactly what clears the queue down to you. A board that
  stays sealed to the close = no fill. This builds the adverse selection
  into the simulation: the weaker the board, the more likely you own it.

Pure functions over minute-bar arrays + daily context; no IO, no orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_TICK = 0.01


def limit_up_pct(symbol: str, *, is_st: bool = False) -> float:
    """Static price-limit band by board (main ±10%, ST ±5%, 创业/科创 ±20%, 北交 ±30%)."""
    s = str(symbol).split(".")[0].zfill(6)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("82", "83", "87", "88", "43", "92")):
        return 0.30
    if is_st:
        return 0.05
    return 0.10


def limit_up_price(prev_close: float, symbol: str, *, is_st: bool = False) -> float:
    return round(float(prev_close) * (1.0 + limit_up_pct(symbol, is_st=is_st)), 2)


@dataclass(frozen=True)
class AuctionFillResult:
    filled_quantity: int
    fill_price: float | None
    fill_ratio: float
    reject_reason: str | None


def auction_fill(
    side: str,
    quantity: int,
    *,
    open_price: float,
    prev_close: float,
    symbol: str,
    is_st: bool = False,
    auction_volume: float = 0.0,
    participation_cap: float = 0.10,
    round_lot: int = 100,
) -> AuctionFillResult:
    """Open-auction fill: single price, one-字板 unbuyable, volume-capped.

    ``auction_volume`` is the 09:30 matched volume (shares); the first
    1-minute bar's volume is an acceptable proxy when the true auction
    print is unavailable.
    """
    if quantity <= 0 or open_price <= 0 or prev_close <= 0:
        return AuctionFillResult(0, None, 0.0, "invalid_order")
    up = limit_up_price(prev_close, symbol, is_st=is_st)
    down = round(prev_close * (1.0 - limit_up_pct(symbol, is_st=is_st)), 2)
    if side == "buy" and open_price >= up - PRICE_TICK / 2:
        return AuctionFillResult(0, None, 0.0, "open_at_limit_up")     # 一字板
    if side == "sell" and open_price <= down + PRICE_TICK / 2:
        return AuctionFillResult(0, None, 0.0, "open_at_limit_down")
    cap = quantity
    if auction_volume > 0:
        cap = int(auction_volume * participation_cap) // round_lot * round_lot
    filled = min(quantity, max(cap, 0))
    if filled < round_lot:
        return AuctionFillResult(0, None, 0.0, "below_lot_after_cap")
    return AuctionFillResult(int(filled), float(open_price), filled / quantity, None)


@dataclass(frozen=True)
class BoardDayState:
    """Limit-up board microstate for one symbol-day, from minute bars."""

    limit_price: float
    touched: bool
    first_touch_time: str | None
    first_seal_time: str | None        # first minute that CLOSES at the limit
    broke_after_seal: bool             # traded below limit after first seal
    first_break_time: str | None
    n_breaks: int
    closed_sealed: bool                # last bar closes at the limit
    volume_at_limit_after: float       # Σ volume of sealed minutes after first seal


def detect_board_day(
    bars: pd.DataFrame,
    *,
    prev_close: float,
    symbol: str,
    is_st: bool = False,
) -> BoardDayState | None:
    """Replay one day's minute bars and extract the limit-up board lifecycle."""
    if bars is None or len(bars) == 0 or prev_close <= 0:
        return None
    b = bars.sort_values("trade_time") if "trade_time" in bars.columns else bars
    h = pd.to_numeric(b["high"], errors="coerce").to_numpy(dtype="float64")
    l = pd.to_numeric(b["low"], errors="coerce").to_numpy(dtype="float64")
    c = pd.to_numeric(b["close"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(b.get("volume", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype="float64") \
        if "volume" in b.columns else np.zeros_like(c)
    t = np.array([str(x).split(" ")[-1] for x in b["trade_time"]]) if "trade_time" in b.columns \
        else np.array([""] * len(c))
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    h, l, c, v, t = h[ok], l[ok], c[ok], v[ok], t[ok]
    if len(c) == 0:
        return None

    lim = limit_up_price(prev_close, symbol, is_st=is_st)
    eps = PRICE_TICK / 2
    touch = h >= lim - eps
    if not touch.any():
        return BoardDayState(lim, False, None, None, False, None, 0, False, 0.0)
    i_touch = int(np.argmax(touch))
    sealed = touch & (c >= lim - eps)
    if not sealed.any():
        # touched but never closed a minute at the limit (immediate fade)
        return BoardDayState(lim, True, str(t[i_touch]), None, False, None, 0,
                             bool(c[-1] >= lim - eps), 0.0)
    i_seal = int(np.argmax(sealed))
    after = slice(i_seal + 1, len(c))
    below = l[after] < lim - eps
    n_breaks = 0
    prev_below = False
    for flag in below:
        if flag and not prev_below:
            n_breaks += 1
        prev_below = bool(flag)
    i_break = (i_seal + 1 + int(np.argmax(below))) if below.any() else None
    vol_after = float(v[after][~below].sum()) if (len(c) - i_seal - 1) > 0 else 0.0
    return BoardDayState(
        limit_price=lim,
        touched=True,
        first_touch_time=str(t[i_touch]),
        first_seal_time=str(t[i_seal]),
        broke_after_seal=bool(below.any()),
        first_break_time=str(t[i_break]) if i_break is not None else None,
        n_breaks=n_breaks,
        closed_sealed=bool(c[-1] >= lim - eps),
        volume_at_limit_after=vol_after,
    )


@dataclass(frozen=True)
class BoardFillResult:
    filled: bool
    fill_price: float | None
    fill_time: str | None
    reason: str                       # filled_on_break / unfilled_sealed / no_seal / not_touched
    closed_sealed: bool               # board state at close (re-seal ⇒ True even if filled)


def board_chase_fill(
    state: BoardDayState | None,
    *,
    order_time: str | None = None,
    fill_on_seal_prob: float = 0.0,
) -> BoardFillResult:
    """Fill a 打板 buy placed at the board price when the first seal forms.

    Honest queue model: the order joins the queue at ``order_time`` (default
    = first seal). It fills ONLY if the board trades below the limit after
    that (the break clears the queue). ``fill_on_seal_prob`` exists solely
    for optimistic sensitivity analysis — keep 0 for honest accounting.
    """
    if state is None or not state.touched:
        return BoardFillResult(False, None, None, "not_touched", False)
    if state.first_seal_time is None:
        return BoardFillResult(False, None, None, "no_seal", state.closed_sealed)
    t0 = order_time or state.first_seal_time
    if state.broke_after_seal and state.first_break_time is not None \
            and state.first_break_time >= t0:
        return BoardFillResult(True, state.limit_price, state.first_break_time,
                               "filled_on_break", state.closed_sealed)
    if fill_on_seal_prob > 0:
        # caller handles the random draw; we just expose the deterministic path
        return BoardFillResult(False, state.limit_price, None,
                               "unfilled_sealed_prob_path", state.closed_sealed)
    return BoardFillResult(False, None, None, "unfilled_sealed", state.closed_sealed)


__all__ = [
    "PRICE_TICK", "limit_up_pct", "limit_up_price",
    "AuctionFillResult", "auction_fill",
    "BoardDayState", "detect_board_day",
    "BoardFillResult", "board_chase_fill",
]
