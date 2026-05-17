"""Timing gate that turns a TechnicalTimingPlan-like frame into
``allow_open`` / ``force_close`` flags.

The gate has two operating modes:

* **Permissive** (default when ``entry_zone_low`` is missing or NaN):
  no restriction on opening — relies entirely on the optimiser's
  upstream tradability filter (suspended / ST / limit-up).
* **Strict** (when both ``entry_zone_low`` and ``entry_zone_high`` are
  present and the previous-day close lies inside the band): the gate
  allows the optimiser to open a new position; otherwise the name is
  marked ``allow_open=False`` and is excluded from the long-side
  ``nlargest`` selection. Existing positions remain in place — the gate
  only restricts **new** entries.

If the previous-day low ever pierces ``invalidation_level`` and the
position is held, the gate sets ``force_close=True`` for that symbol on
the next decision day.

Both signals are returned as a thin DataFrame keyed on
``(trade_date, symbol)``. The gate never raises on missing data — when
inputs are degraded it logs a warning into the diagnostics and falls
back to permissive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimingGateConfig:
    enabled: bool = False
    require_in_entry_zone: bool = True
    enforce_invalidation: bool = True


@dataclass(frozen=True)
class TimingGateResult:
    decisions: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


_REQUIRED_TIMING_COLUMNS = (
    "trade_date",
    "symbol",
    "entry_zone_low",
    "entry_zone_high",
    "invalidation_level",
)


def _empty_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "symbol",
            "allow_open",
            "force_close",
            "reason",
        ]
    )


def apply_timing_gate(
    market_panel: pd.DataFrame,
    timing_plan: pd.DataFrame | None,
    config: TimingGateConfig | None = None,
) -> TimingGateResult:
    cfg = config or TimingGateConfig()
    if not cfg.enabled or timing_plan is None or timing_plan.empty:
        return TimingGateResult(_empty_decisions(), {"status": "disabled_or_empty"})

    missing = [c for c in _REQUIRED_TIMING_COLUMNS if c not in timing_plan.columns]
    if missing:
        return TimingGateResult(
            _empty_decisions(),
            {"status": "missing_columns", "missing": missing},
        )

    panel = market_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    if "low" not in panel.columns:
        panel["low"] = panel["close"]
    panel["prev_close"] = panel.groupby("symbol")["close"].shift(1)
    panel["prev_low"] = panel.groupby("symbol")["low"].shift(1)

    plan = timing_plan.copy()
    plan["trade_date"] = pd.to_datetime(plan["trade_date"], errors="coerce")
    plan = plan.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    plan["entry_zone_low_prev"] = plan.groupby("symbol")["entry_zone_low"].shift(1)
    plan["entry_zone_high_prev"] = plan.groupby("symbol")["entry_zone_high"].shift(1)
    plan["invalidation_prev"] = plan.groupby("symbol")["invalidation_level"].shift(1)

    merged = panel.merge(
        plan[["trade_date", "symbol", "entry_zone_low_prev", "entry_zone_high_prev", "invalidation_prev"]],
        on=["trade_date", "symbol"],
        how="left",
    )

    allow_open_default = True if not cfg.require_in_entry_zone else None
    decisions: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        prev_close = getattr(row, "prev_close", np.nan)
        prev_low = getattr(row, "prev_low", np.nan)
        ez_low = getattr(row, "entry_zone_low_prev", np.nan)
        ez_high = getattr(row, "entry_zone_high_prev", np.nan)
        invalidation = getattr(row, "invalidation_prev", np.nan)
        if cfg.require_in_entry_zone and pd.notna(ez_low) and pd.notna(ez_high) and pd.notna(prev_close):
            allow_open = bool(ez_low <= prev_close <= ez_high)
        else:
            allow_open = bool(allow_open_default) if allow_open_default is not None else True
        force_close = False
        if cfg.enforce_invalidation and pd.notna(invalidation) and pd.notna(prev_low):
            force_close = bool(prev_low <= invalidation)
        reason = "ok"
        if not allow_open:
            reason = "outside_entry_zone"
        if force_close:
            reason = "invalidation_breach"
        decisions.append(
            {
                "trade_date": row.trade_date,
                "symbol": row.symbol,
                "allow_open": allow_open,
                "force_close": force_close,
                "reason": reason,
            }
        )

    out = pd.DataFrame(decisions)
    diagnostics = {
        "status": "passed",
        "rows": int(len(out)),
        "force_close_count": int(out["force_close"].sum()) if not out.empty else 0,
        "blocked_open_count": int((~out["allow_open"]).sum()) if not out.empty else 0,
        "config": {
            "enabled": cfg.enabled,
            "require_in_entry_zone": cfg.require_in_entry_zone,
            "enforce_invalidation": cfg.enforce_invalidation,
        },
    }
    return TimingGateResult(out, diagnostics)


__all__ = [
    "TimingGateConfig",
    "TimingGateResult",
    "apply_timing_gate",
]
