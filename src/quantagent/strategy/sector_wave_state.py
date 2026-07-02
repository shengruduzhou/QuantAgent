"""Stage 9 — regime-aware sector wave state machine.

Not a fixed laggard or leader tilt: each rebalance day it reads the *market
regime* and switches which sector wave to ride, plus how much gross to carry:

  market regime              -> wave mode            gross
  -------------------------------------------------------------
  strong uptrend + broad      leader_continuation    1.0   (ride main lines)
  breadth expanding from mid   breakout_ignition      1.0   (new main line starting)
  choppy / range               laggard_reversal       ~0.85 (高低切 rebound)
  weak / risk-off              defensive              ~0.35 (cut exposure)

Within the active mode, sectors are scored so the chosen ones fit the wave:
  leader_continuation : strong RS + broad breadth + breakout + volume accel,
                        excluding sectors that show *exhaustion* (RS high but
                        rolling over with fading volume).
  breakout_ignition   : fresh new-high + breadth expansion + 涨停 diffusion +
                        RS acceleration.
  laggard_reversal    : weak RS but *stabilising* (drawdown recovering, breadth
                        turning up) — not still-falling knives.
  defensive           : low beta / low vol survivors, small gross.

Pure: takes one day's sector frame (+ market scalars) and returns the selected
sectors and gross. All inputs are trailing/known at the rebalance close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEC = "sector_level_1"


@dataclass(frozen=True)
class WaveConfig:
    bull_trend: float = 0.05         # mkt 60d return for strong uptrend
    bull_breadth: float = 0.55       # fraction of sectors with +20d mom
    weak_trend: float = -0.05
    weak_breadth: float = 0.30
    ignition_breadth_exp: float = 0.05
    ignition_breadth: float = 0.45
    gross_bull: float = 1.0
    gross_ignition: float = 1.0
    gross_chop: float = 0.85
    gross_defensive: float = 0.35


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 1e-12 else s * 0.0


def market_scalars(sector_day: pd.DataFrame, mkt_trend_60: float) -> dict:
    """Cross-sector breadth scalars for the regime decision on one date."""
    return {
        "trend": float(mkt_trend_60),
        "breadth": float((sector_day["mom_20"] > 0).mean()),
        "breadth_exp": float(sector_day["breadth_expansion"].mean()),
    }


def classify_regime(m: dict, cfg: WaveConfig) -> tuple[str, float]:
    if m["trend"] < cfg.weak_trend or m["breadth"] < cfg.weak_breadth:
        return "defensive", cfg.gross_defensive
    if m["trend"] > cfg.bull_trend and m["breadth"] > cfg.bull_breadth:
        return "leader_continuation", cfg.gross_bull
    if m["breadth_exp"] > cfg.ignition_breadth_exp and m["breadth"] > cfg.ignition_breadth:
        return "breakout_ignition", cfg.gross_ignition
    return "laggard_reversal", cfg.gross_chop


def _exhaustion_mask(d: pd.DataFrame) -> pd.Series:
    """Sectors topping out: RS in top tercile but decelerating + volume fading."""
    hi_rs = d["rs_60"] > d["rs_60"].quantile(0.66)
    rolling_over = (d["rs_60_chg_20"] < 0) & (d["amt_accel"] < 0)
    return hi_rs & rolling_over


def score_sectors(d: pd.DataFrame, mode: str) -> pd.Series:
    """Per-sector preference score for the active wave mode (higher = pick)."""
    if mode == "leader_continuation":
        s = (_z(d["rs_60"]) + _z(d["breadth_ma60"]) + _z(d["amt_accel"])
             + d["breakout_60"].astype(float) + 0.5 * _z(d["breadth_expansion"]))
        s = s.where(~_exhaustion_mask(d), s - 10.0)  # demote exhausted leaders
    elif mode == "breakout_ignition":
        s = (d["breakout_60"].astype(float) + d["breakout_120"].astype(float)
             + _z(d["breadth_expansion"]) + _z(d["limitup_diffusion_5d"])
             + _z(d["rs_60_chg_20"]))
    elif mode == "laggard_reversal":
        s = _z(-d["rs_60"]) + _z(d["dd_recover_20"]) + 0.5 * _z(d["breadth_up"])
        # avoid still-falling knives: require some stabilization
        falling = (d["rs_60_chg_20"] < d["rs_60_chg_20"].quantile(0.33)) & (d["breadth_expansion"] < 0)
        s = s.where(~falling, s - 10.0)
    elif mode == "defensive":
        s = _z(-d["vol_60"]) + _z(-d["beta_60"].fillna(1.0))
    else:
        raise ValueError(mode)
    return s


def select_wave(sector_day: pd.DataFrame, mkt_trend_60: float, *, n_sectors: int,
                cfg: WaveConfig = WaveConfig()) -> dict:
    """Return {mode, gross, sectors:[...]} for one rebalance date."""
    d = sector_day.dropna(subset=["rs_60", "mom_60"]).copy()
    if d.empty:
        return {"mode": "defensive", "gross": cfg.gross_defensive, "sectors": []}
    m = market_scalars(d, mkt_trend_60)
    mode, gross = classify_regime(m, cfg)
    sc = score_sectors(d, mode).dropna()
    d = d.assign(_score=sc).dropna(subset=["_score"]).sort_values("_score", ascending=False)
    return {"mode": mode, "gross": gross, "sectors": list(d[SEC].head(n_sectors)),
            "market": m}
