"""Stage 3 — Market hard gate.

A hard 0/1 cut-off that fully suppresses gross exposure when one of
several broad-market panic signals fires.  This is intentionally
*orthogonal* to the existing soft regime exposure (which still scales
gross 0..1 by benchmark momentum/MA-200 break) and to the drawdown /
volatility multipliers.  Soft gates degrade exposure; the hard gate
collapses it.

Triggers (any one fires → gate goes hot):
  T1 — extreme short-term crash: benchmark 5-day return ≤ -8%
  T2 — confirmed deep bear: benchmark 20-day return ≤ -15% AND close
       below 200-day MA
  T3 — breadth collapse: cross-section advancers ratio < 20% for ≥3
       consecutive trade days
  T4 — volatility spike: realised 20-day vol ≥ vol_spike_multiplier ×
       trailing 60-day average vol (default 2.0×)

After any trigger, the gate stays hot for at least ``cool_down_days``
sessions (default 5) even if the trigger condition has cleared, to
avoid whipsawing on a single bounce.

The module produces a per-date frame and a manifest of trigger windows
so post-trade audit can replay every gate-active day with attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketHardGateConfig:
    # T1 — extreme crash
    crash_5d_threshold: float = -0.08
    # T2 — deep bear
    bear_20d_threshold: float = -0.15
    ma_window: int = 200
    # T3 — breadth collapse
    breadth_advancer_threshold: float = 0.20
    breadth_consecutive_days: int = 3
    # T4 — volatility spike
    vol_window_short: int = 20
    vol_window_long: int = 60
    vol_spike_multiplier: float = 2.0
    # Post-trigger lockout
    cool_down_days: int = 5
    # Output suppression weight (0 = full block, 0.05 = de minimis exposure)
    blocked_gross_multiplier: float = 0.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MarketHardGateResult:
    frame: pd.DataFrame  # per-date columns: trade_date, hard_gate_active, trigger_reason, cool_down_remaining
    windows: list[dict[str, Any]] = field(default_factory=list)  # contiguous active windows

    def to_manifest(self) -> dict[str, Any]:
        f = self.frame
        n_active = int(f["hard_gate_active"].sum()) if not f.empty else 0
        n_total = int(len(f))
        coverage = (n_active / n_total) if n_total else 0.0
        return {
            "n_dates": n_total,
            "n_hard_gate_active": n_active,
            "active_share": float(coverage),
            "n_windows": len(self.windows),
            "windows": self.windows,
        }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def compute_market_hard_gate(
    benchmark: pd.DataFrame | None,
    *,
    breadth_panel: pd.DataFrame | None = None,
    config: MarketHardGateConfig | None = None,
) -> MarketHardGateResult:
    """Compute the per-date hard-gate state.

    Parameters
    ----------
    benchmark : DataFrame with columns ``trade_date`` and ``close``.
    breadth_panel : Optional wide-form daily returns frame (rows=trade_date,
        columns=symbols) used for T3 (cross-section advancers ratio).
        Without it T3 is silently skipped.
    config : Tunable thresholds. Defaults are tuned for CSI 300 daily data.
    """
    cfg = config or MarketHardGateConfig()
    if not cfg.enabled or benchmark is None or benchmark.empty:
        return MarketHardGateResult(
            frame=pd.DataFrame(columns=["trade_date", "hard_gate_active", "trigger_reason", "cool_down_remaining"]),
            windows=[],
        )

    b = (
        benchmark[["trade_date", "close"]]
        .copy()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    b["trade_date"] = pd.to_datetime(b["trade_date"])
    b["close"] = b["close"].astype(float)

    # Returns
    b["ret_5"] = b["close"].pct_change(5)
    b["ret_20"] = b["close"].pct_change(cfg.vol_window_short)
    b["ma_n"] = b["close"].rolling(cfg.ma_window, min_periods=max(20, cfg.ma_window // 4)).mean()
    b["below_ma"] = b["close"] < b["ma_n"]

    # T1 crash, T2 bear
    t1 = b["ret_5"] <= cfg.crash_5d_threshold
    t2 = (b["ret_20"] <= cfg.bear_20d_threshold) & b["below_ma"]

    # T4 vol spike — realised daily vol short vs long
    daily_ret = b["close"].pct_change()
    vol_short = daily_ret.rolling(cfg.vol_window_short, min_periods=max(5, cfg.vol_window_short // 4)).std()
    vol_long = daily_ret.rolling(cfg.vol_window_long, min_periods=max(10, cfg.vol_window_long // 4)).std()
    # Require a strictly positive long-window baseline to avoid a 0/0 false-fire
    # on perfectly flat synthetic series. Real markets never have vol==0, but
    # synthetic tests and freeze-day stretches can.
    eps = 1e-8
    t4 = (vol_long > eps) & (vol_short > (vol_long * cfg.vol_spike_multiplier))

    # T3 breadth — needs aligned cross-section panel
    t3 = pd.Series(False, index=b.index)
    breadth_pct = pd.Series(np.nan, index=b.index)
    if breadth_panel is not None and not breadth_panel.empty:
        bp = breadth_panel.copy()
        if "trade_date" in bp.columns:
            bp = bp.set_index("trade_date")
        bp.index = pd.to_datetime(bp.index)
        bp = bp.reindex(b["trade_date"])
        adv = (bp > 0).sum(axis=1)
        tot = bp.notna().sum(axis=1).replace(0, np.nan)
        ratio = (adv / tot).astype(float)
        breadth_pct.index = b["trade_date"]
        breadth_pct.iloc[: len(ratio)] = ratio.to_numpy()
        # Rolling-K consecutive days under threshold
        under = ratio < cfg.breadth_advancer_threshold
        cons = under.rolling(cfg.breadth_consecutive_days, min_periods=cfg.breadth_consecutive_days).sum() >= cfg.breadth_consecutive_days
        cons = cons.fillna(False).to_numpy()
        t3 = pd.Series(cons, index=b.index)

    # Trigger reasons — earliest-priority wins for the reason label
    reason: list[str] = []
    raw_active = (t1 | t2 | t3 | t4).fillna(False).to_numpy()
    t1_arr = t1.fillna(False).to_numpy()
    t2_arr = t2.fillna(False).to_numpy()
    t3_arr = t3.fillna(False).to_numpy()
    t4_arr = t4.fillna(False).to_numpy()
    for i in range(len(b)):
        if t1_arr[i]:
            reason.append("crash_5d")
        elif t2_arr[i]:
            reason.append("deep_bear_20d_below_ma")
        elif t3_arr[i]:
            reason.append("breadth_collapse")
        elif t4_arr[i]:
            reason.append("vol_spike")
        else:
            reason.append("")

    # Apply cool-down: once raw_active=True, hard_gate_active stays True for
    # ``cool_down_days`` extra sessions even if the trigger has cleared.
    n = len(b)
    hard_active = np.zeros(n, dtype=bool)
    cooldown_remaining = np.zeros(n, dtype=int)
    cooldown = 0
    for i in range(n):
        if raw_active[i]:
            hard_active[i] = True
            cooldown = cfg.cool_down_days
            cooldown_remaining[i] = cfg.cool_down_days
        elif cooldown > 0:
            hard_active[i] = True
            cooldown -= 1
            cooldown_remaining[i] = cooldown
            reason[i] = "cool_down"
        else:
            cooldown_remaining[i] = 0

    out = pd.DataFrame(
        {
            "trade_date": b["trade_date"].to_numpy(),
            "hard_gate_active": hard_active,
            "trigger_reason": reason,
            "cool_down_remaining": cooldown_remaining,
            "ret_5d": b["ret_5"].to_numpy(),
            "ret_20d": b["ret_20"].to_numpy(),
            "below_ma_n": b["below_ma"].to_numpy(),
            "vol_short": vol_short.to_numpy(),
            "vol_long": vol_long.to_numpy(),
            "breadth_advancer_pct": breadth_pct.to_numpy() if breadth_panel is not None else np.nan,
        }
    )

    windows = _extract_active_windows(out)
    return MarketHardGateResult(frame=out, windows=windows)


def _extract_active_windows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse contiguous hard-gate-active days into (start, end, primary reason)."""
    if frame.empty:
        return []
    windows: list[dict[str, Any]] = []
    start_idx: int | None = None
    reasons: list[str] = []
    for i, row in frame.iterrows():
        if row["hard_gate_active"] and start_idx is None:
            start_idx = i
            reasons = [row["trigger_reason"]] if row["trigger_reason"] else []
        elif row["hard_gate_active"] and start_idx is not None:
            if row["trigger_reason"]:
                reasons.append(row["trigger_reason"])
        elif (not row["hard_gate_active"]) and start_idx is not None:
            primary = _primary_reason(reasons)
            windows.append(
                {
                    "start": str(frame.iloc[start_idx]["trade_date"].date()),
                    "end": str(frame.iloc[i - 1]["trade_date"].date()),
                    "days": int(i - start_idx),
                    "primary_reason": primary,
                    "reasons": reasons,
                }
            )
            start_idx = None
            reasons = []
    if start_idx is not None:
        primary = _primary_reason(reasons)
        windows.append(
            {
                "start": str(frame.iloc[start_idx]["trade_date"].date()),
                "end": str(frame.iloc[-1]["trade_date"].date()),
                "days": int(len(frame) - start_idx),
                "primary_reason": primary,
                "reasons": reasons,
            }
        )
    return windows


def _primary_reason(reasons: list[str]) -> str:
    priority = ["crash_5d", "deep_bear_20d_below_ma", "breadth_collapse", "vol_spike", "cool_down"]
    for p in priority:
        if p in reasons:
            return p
    return "unknown"


# ---------------------------------------------------------------------------
# Multiplier helper for the backtest hot loop
# ---------------------------------------------------------------------------

def hard_gate_multiplier(
    gate_frame: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: MarketHardGateConfig,
) -> float:
    """Return the gross-exposure multiplier for ``trade_date``.

    Returns ``1.0`` when the hard gate is inactive on that date, and
    ``config.blocked_gross_multiplier`` (default 0.0) when active.
    Missing trade_date → 1.0 (fail-open: missing data must not silently
    block a live strategy).
    """
    if gate_frame is None or gate_frame.empty:
        return 1.0
    series = gate_frame.set_index("trade_date")["hard_gate_active"]
    if series.empty:
        return 1.0
    try:
        active = bool(series.reindex([pd.to_datetime(trade_date)], method="ffill").iloc[0])
    except (KeyError, IndexError):
        return 1.0
    return float(config.blocked_gross_multiplier) if active else 1.0


# ---------------------------------------------------------------------------
# Manifest writer (for audit replay)
# ---------------------------------------------------------------------------

def write_hard_gate_manifest(
    result: MarketHardGateResult,
    output_dir: str | Path,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "market_hard_gate.json"
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2),
        encoding="utf-8",
    )
    return manifest_path
