"""Round-trip labels for cost-sensitive intraday Do-T models.

The labels model a *realistic, non-clairvoyant* round trip from each minute:

* reverse-T (SELL_HIGH): sell now; buy back at the first future close that
  pulls back to ``entry*(1-target)`` (favorable) or runs up to ``entry*(1+stop)``
  (chased) — whichever comes first — else at the horizon-end close.
* positive-T (BUY_LOW): symmetric (buy now, sell into the first rally).

Edge columns ending in ``_gross_edge_bps`` are **before cost** and feed the EV
regressors (``decide_ev`` owns every cost subtraction, so feeding net edge would
double-count cost).  ``_net_edge_bps`` columns are gross minus a single explicit
round-trip cost and drive the binary success labels.  Capacity/fill realism is
handled by the fill simulator, *not* baked into the label as a penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.execution.intraday_fill import CostConfig


ROUND_TRIP_LABEL_COLUMNS = [
    "label_sell_high_success",
    "label_sell_high_gross_edge_bps",
    "label_sell_high_net_edge_bps",
    "label_sell_high_fail_new_high",
    "label_sell_high_eod_restore",
    "label_time_to_buyback",
    "label_adverse_excursion_after_sell",
    "label_buyback_now_success",
    "label_buyback_now_edge_bps",
    "label_wait_extra_edge_bps",
    "label_miss_rebound_risk",
    "label_buy_low_success",
    "label_buy_low_gross_edge_bps",
    "label_buy_low_net_edge_bps",
    "label_buy_low_fail_breakdown",
    "label_sell_after_buy_success",
    "label_adverse_excursion_after_buy",
]


@dataclass(frozen=True)
class RoundTripLabelConfig:
    horizon_minutes: int = 60
    min_required_edge_bps: float = 8.0
    adverse_new_high_bps: float = 15.0
    adverse_breakdown_bps: float = 15.0
    reverse_target_bps: float | None = None  # default: 2x round-trip cost
    reverse_stop_bps: float | None = None    # default: 2x round-trip cost
    open_sell_price_col: str = "open_sell_price"
    cost: CostConfig = CostConfig()

    @property
    def round_trip_cost_bps(self) -> float:
        explicit = (
            2.0 * self.cost.commission_rate
            + self.cost.stamp_tax_sell
            + 2.0 * self.cost.transfer_fee
        ) * 10_000.0
        execution = 2.0 * (self.cost.slippage_bps + self.cost.spread_bps)
        return explicit + execution

    @property
    def target_bps(self) -> float:
        return float(self.reverse_target_bps if self.reverse_target_bps is not None else 2.0 * self.round_trip_cost_bps)

    @property
    def stop_bps(self) -> float:
        return float(self.reverse_stop_bps if self.reverse_stop_bps is not None else 2.0 * self.round_trip_cost_bps)


def build_round_trip_labels(
    minute_panel: pd.DataFrame,
    *,
    config: RoundTripLabelConfig | None = None,
) -> pd.DataFrame:
    """Generate per-minute labels for legal SELL_HIGH/BUY_BACK/BUY_LOW round trips.

    These labels look forward and must be joined only to causal features built
    separately.  They model complete, cost-aware round trips and mark EOD restore
    / adverse excursion instead of treating every local extreme as a success.
    """
    cfg = config or RoundTripLabelConfig()
    if minute_panel is None or minute_panel.empty:
        return pd.DataFrame(columns=list(ROUND_TRIP_LABEL_COLUMNS))
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
    for col in ("open", "high", "low", "close", "volume", "pre_close", "limit_up", "limit_down", cfg.open_sell_price_col):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    if "high" not in panel.columns:
        panel["high"] = panel["close"]
    if "low" not in panel.columns:
        panel["low"] = panel["close"]
    if "volume" not in panel.columns:
        panel["volume"] = 0.0
    panel = panel.dropna(subset=["symbol", "trade_time", "trade_date", "close"])
    if panel.empty:
        return pd.DataFrame(columns=list(ROUND_TRIP_LABEL_COLUMNS))
    panel = panel.sort_values(["symbol", "trade_date", "trade_time"]).reset_index(drop=True)

    frames = []
    for _, g in panel.groupby(["symbol", "trade_date"], sort=False):
        frames.append(_label_one_day(g.reset_index(drop=True), cfg))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(ROUND_TRIP_LABEL_COLUMNS))


def _label_one_day(g: pd.DataFrame, cfg: RoundTripLabelConfig) -> pd.DataFrame:
    out = g.copy()
    n = len(out)
    cols = {c: np.full(n, np.nan, dtype=float) for c in ROUND_TRIP_LABEL_COLUMNS}
    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    times = pd.to_datetime(out["trade_time"], errors="coerce")
    time_min = (times.astype("int64") // 60_000_000_000).to_numpy()  # epoch minutes
    if cfg.open_sell_price_col in out.columns:
        open_sell = pd.to_numeric(out[cfg.open_sell_price_col], errors="coerce").to_numpy(dtype=float)
    else:
        open_sell = np.full(n, np.nan, dtype=float)
    limit_up = _limit_values(out, up=True)
    limit_down = _limit_values(out, up=False)
    cost = cfg.round_trip_cost_bps
    target = cfg.target_bps / 10_000.0
    stop = cfg.stop_bps / 10_000.0

    for i in range(n):
        px = float(close[i])
        if not np.isfinite(px) or px <= 0:
            continue
        j0 = i + 1
        j1 = min(n, i + 1 + int(cfg.horizon_minutes))
        if j0 >= j1:
            _write_no_future(cols, i)
            continue
        _label_sell_high(cols, i, px, close[j0:j1], high[j0:j1], low[j0:j1], time_min[i],
                         time_min[j0:j1], limit_up[j0:j1], target, stop, cost, cfg)
        _label_buyback_now(cols, i, px, close[j0:j1], open_sell[i], cost, cfg)
        _label_buy_low(cols, i, px, close[j0:j1], high[j0:j1], low[j0:j1],
                       limit_down[j0:j1], target, stop, cost, cfg)
    for c, arr in cols.items():
        out[c] = arr
    return out


def _first_touch_exit(
    future_close: np.ndarray,
    *,
    target_px: float,
    stop_px: float,
    favorable_is_low: bool,
) -> tuple[float, int, str]:
    """First future close that hits target (favorable) or stop (adverse).

    ``favorable_is_low=True`` for reverse-T (we want a pullback below target).
    Falls back to the horizon-end close when neither level is reached.
    """
    for k in range(future_close.size):
        c = float(future_close[k])
        if not np.isfinite(c):
            continue
        if favorable_is_low:
            if c <= target_px:
                return c, k, "target"
            if c >= stop_px:
                return c, k, "stop"
        else:
            if c >= target_px:
                return c, k, "target"
            if c <= stop_px:
                return c, k, "stop"
    # horizon-end fallback
    last = future_close.size - 1
    while last >= 0 and not np.isfinite(future_close[last]):
        last -= 1
    if last < 0:
        return float("nan"), -1, "none"
    return float(future_close[last]), last, "horizon_end"


def _label_sell_high(
    cols: dict,
    i: int,
    sell_price: float,
    future_close: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry_min: float,
    future_min: np.ndarray,
    future_limit_up: np.ndarray,
    target: float,
    stop: float,
    cost: float,
    cfg: RoundTripLabelConfig,
) -> None:
    if future_close.size == 0 or not np.isfinite(future_close).any():
        _write_no_future(cols, i)
        return
    target_px = sell_price * (1.0 - target)
    stop_px = sell_price * (1.0 + stop)
    buyback_px, buy_k, _reason = _first_touch_exit(
        future_close, target_px=target_px, stop_px=stop_px, favorable_is_low=True
    )
    if not np.isfinite(buyback_px) or buy_k < 0:
        _write_no_future(cols, i)
        return
    gross = (sell_price - buyback_px) / sell_price * 10_000.0
    net = gross - cost
    near_limit_up = _near_limit(future_high[buy_k], future_limit_up[buy_k], up=True)
    fail_new_high = bool(np.nanmax(future_high) > sell_price * (1.0 + cfg.adverse_new_high_bps / 10_000.0))
    success = bool(net > cfg.min_required_edge_bps and not near_limit_up)
    cols["label_sell_high_gross_edge_bps"][i] = gross
    cols["label_sell_high_net_edge_bps"][i] = net
    cols["label_sell_high_success"][i] = int(success)
    cols["label_sell_high_fail_new_high"][i] = int(fail_new_high)
    cols["label_sell_high_eod_restore"][i] = int(not success)
    cols["label_time_to_buyback"][i] = float(future_min[buy_k] - entry_min)
    cols["label_adverse_excursion_after_sell"][i] = (np.nanmax(future_high) / sell_price - 1.0) * 10_000.0


def _label_buyback_now(
    cols: dict,
    i: int,
    current_price: float,
    future_close: np.ndarray,
    open_sell_price: float,
    cost: float,
    cfg: RoundTripLabelConfig,
) -> None:
    if not np.isfinite(open_sell_price):
        cols["label_buyback_now_success"][i] = 0
        return
    sell_price = float(open_sell_price)
    # gross realized edge of buying back NOW against the prior SELL_HIGH leg
    now_gross = (sell_price - current_price) / sell_price * 10_000.0
    best_future_gross = (sell_price - float(np.nanmin(future_close))) / sell_price * 10_000.0
    wait_extra = best_future_gross - now_gross
    miss_rebound = max(0.0, (float(np.nanmax(future_close)) / current_price - 1.0) * 10_000.0)
    cols["label_buyback_now_edge_bps"][i] = now_gross
    cols["label_wait_extra_edge_bps"][i] = wait_extra
    cols["label_miss_rebound_risk"][i] = miss_rebound
    cols["label_buyback_now_success"][i] = int(now_gross - cost > cfg.min_required_edge_bps and wait_extra <= miss_rebound)


def _label_buy_low(
    cols: dict,
    i: int,
    buy_price: float,
    future_close: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_limit_down: np.ndarray,
    target: float,
    stop: float,
    cost: float,
    cfg: RoundTripLabelConfig,
) -> None:
    if future_close.size == 0 or not np.isfinite(future_close).any():
        return
    target_px = buy_price * (1.0 + target)
    stop_px = buy_price * (1.0 - stop)
    sell_after_px, sell_k, _reason = _first_touch_exit(
        future_close, target_px=target_px, stop_px=stop_px, favorable_is_low=False
    )
    if not np.isfinite(sell_after_px) or sell_k < 0:
        return
    gross = (sell_after_px - buy_price) / buy_price * 10_000.0
    net = gross - cost
    near_limit_down = _near_limit(future_low[sell_k], future_limit_down[sell_k], up=False)
    fail_breakdown = bool(np.nanmin(future_low) < buy_price * (1.0 - cfg.adverse_breakdown_bps / 10_000.0))
    success = bool(net > cfg.min_required_edge_bps and not near_limit_down)
    cols["label_buy_low_gross_edge_bps"][i] = gross
    cols["label_buy_low_net_edge_bps"][i] = net
    cols["label_buy_low_success"][i] = int(success)
    cols["label_buy_low_fail_breakdown"][i] = int(fail_breakdown)
    cols["label_sell_after_buy_success"][i] = int(success)
    cols["label_adverse_excursion_after_buy"][i] = (np.nanmin(future_low) / buy_price - 1.0) * 10_000.0


def _write_no_future(cols: dict, i: int) -> None:
    cols["label_sell_high_success"][i] = 0
    cols["label_sell_high_eod_restore"][i] = 1
    cols["label_buy_low_success"][i] = 0
    cols["label_sell_after_buy_success"][i] = 0


def _limit_values(g: pd.DataFrame, *, up: bool) -> np.ndarray:
    col = "limit_up" if up else "limit_down"
    if col in g.columns and g[col].notna().any():
        return pd.to_numeric(g[col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    pre = pd.to_numeric(g.get("pre_close", pd.Series(np.nan, index=g.index)), errors="coerce")
    if not pre.notna().any():
        pre = pd.Series(g["close"].iloc[0], index=g.index)
    pre = pre.ffill().bfill()
    band = _symbol_limit_band(str(g["symbol"].iloc[0]) if len(g) else "")
    return (pre * (1.0 + band if up else 1.0 - band)).to_numpy(dtype=float)


def _symbol_limit_band(symbol: str) -> float:
    s = str(symbol)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("8", "4")):
        return 0.30
    return 0.10


def _near_limit(price: float, limit_price: float, *, up: bool) -> bool:
    if not np.isfinite(price) or not np.isfinite(limit_price) or limit_price <= 0:
        return False
    if up:
        return price >= limit_price * 0.998
    return price <= limit_price * 1.002


def _minutes_between(start: object, end: object) -> float:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return float((e - s).total_seconds() / 60.0)


__all__ = [
    "ROUND_TRIP_LABEL_COLUMNS",
    "RoundTripLabelConfig",
    "build_round_trip_labels",
]
