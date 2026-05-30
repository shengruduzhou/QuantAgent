"""MarketRegimeDetector — global regime classifier (spec section 5).

Combines two orthogonal signals into a single regime label per
trading day:

1. **Trend × volume** across the five China index benchmarks
   (CSI 300 / CSI 500 / CSI 1000 / SSE Composite / ChiNext).
   Each index contributes ``(trend, volume_state)`` where ``trend ∈
   {up, flat, down}`` and ``volume_state ∈ {expanding, contracting}``.

2. **Market breadth** across the universe: advance / decline ratio,
   limit-up count, limit-down count, max consecutive limit-up
   ("连板"), and 炸板率 (failed limit-up ratio).

The detector outputs one :class:`MarketRegimeSnapshot` per trade
date, which carries:

* ``regime`` ∈ ``{bull_expansion, bull_consolidation, normal, caution,
   bear_capitulation, crisis}``
* ``risk_level`` ∈ ``{low, medium, high, severe}``
* per-axis diagnostic numbers (so the decision chain can surface
  *why* it is in the current regime)

The decision chain and PositionPolicy consume ``risk_level`` to
trim total exposure and gate new entries. The detector is
read-only and produces no orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

REGIME_LABELS: tuple[str, ...] = (
    "bull_expansion",
    "bull_consolidation",
    "normal",
    "caution",
    "bear_capitulation",
    "crisis",
)

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high", "severe")


DEFAULT_BENCHMARK_INDICES: tuple[str, ...] = (
    "csi300",
    "csi500",
    "csi1000",
    "sse_composite",
    "chinext",
)


# ---------------------------------------------------------------------------
# Config + snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketRegimeConfig:
    short_trend_window: int = 5
    long_trend_window: int = 20
    volume_window: int = 20
    bullish_trend_threshold: float = 0.02   # 5-day return ≥ 2% counts as bullish
    bearish_trend_threshold: float = -0.02
    volume_expansion_threshold: float = 1.20   # 5d vol vs 20d ≥ 1.2x
    volume_contraction_threshold: float = 0.80
    # breadth
    advance_decline_strong: float = 1.5
    advance_decline_weak: float = 0.6
    limit_up_strong: int = 40
    limit_down_severe: int = 30
    consecutive_limit_up_strong: int = 5
    zhaban_severe: float = 0.40
    indices: tuple[str, ...] = DEFAULT_BENCHMARK_INDICES


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    trade_date: pd.Timestamp
    regime: str
    risk_level: str
    # diagnostics
    bull_index_count: int = 0
    bear_index_count: int = 0
    volume_expanding_count: int = 0
    volume_contracting_count: int = 0
    advance_decline_ratio: float = 1.0
    limit_up_count: int = 0
    limit_down_count: int = 0
    max_consecutive_limit_up: int = 0
    zhaban_rate: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "regime": self.regime,
            "risk_level": self.risk_level,
            "bull_index_count": self.bull_index_count,
            "bear_index_count": self.bear_index_count,
            "volume_expanding_count": self.volume_expanding_count,
            "volume_contracting_count": self.volume_contracting_count,
            "advance_decline_ratio": float(self.advance_decline_ratio),
            "limit_up_count": int(self.limit_up_count),
            "limit_down_count": int(self.limit_down_count),
            "max_consecutive_limit_up": int(self.max_consecutive_limit_up),
            "zhaban_rate": float(self.zhaban_rate),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Index trend / volume axis
# ---------------------------------------------------------------------------

def _classify_index_trend_volume(
    index_panel: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    config: MarketRegimeConfig,
) -> tuple[int, int, int, int]:
    """Return (bull_idx_count, bear_idx_count, vol_expand_count, vol_contract_count)."""
    if index_panel is None or index_panel.empty:
        return 0, 0, 0, 0
    work = index_panel.copy()
    if not {"trade_date", "index_code", "close"}.issubset(work.columns):
        return 0, 0, 0, 0
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"])
    bull = bear = expand = contract = 0
    cutoff = pd.Timestamp(trade_date)
    for idx_code in config.indices:
        sub = work[(work["index_code"] == idx_code) & (work["trade_date"] <= cutoff)].sort_values("trade_date")
        if sub.empty or len(sub) < config.long_trend_window + 1:
            continue
        close = sub["close"].astype(float).values
        ret_short = close[-1] / close[-1 - config.short_trend_window] - 1.0
        if ret_short >= config.bullish_trend_threshold:
            bull += 1
        elif ret_short <= config.bearish_trend_threshold:
            bear += 1
        if "volume" in sub.columns:
            vols = sub["volume"].astype(float).values
            short_vol = float(np.mean(vols[-config.short_trend_window:]))
            long_vol = float(np.mean(vols[-config.volume_window:]))
            if long_vol > 0:
                ratio = short_vol / long_vol
                if ratio >= config.volume_expansion_threshold:
                    expand += 1
                elif ratio <= config.volume_contraction_threshold:
                    contract += 1
    return bull, bear, expand, contract


# ---------------------------------------------------------------------------
# Breadth axis
# ---------------------------------------------------------------------------

def _classify_breadth(
    breadth_row: pd.Series | None,
    config: MarketRegimeConfig,
) -> dict[str, float]:
    """Take one row of breadth metrics and return a normalised dict.

    Expected columns (any subset OK; missing → neutral default):
    ``advance_count, decline_count, limit_up_count, limit_down_count,
    max_consecutive_limit_up, zhaban_rate``.
    """
    out = {
        "advance_decline_ratio": 1.0,
        "limit_up_count": 0,
        "limit_down_count": 0,
        "max_consecutive_limit_up": 0,
        "zhaban_rate": 0.0,
    }
    if breadth_row is None or breadth_row.empty:
        return out
    adv = float(breadth_row.get("advance_count", 0) or 0)
    dec = float(breadth_row.get("decline_count", 0) or 0)
    if dec > 0:
        out["advance_decline_ratio"] = adv / dec
    elif adv > 0:
        out["advance_decline_ratio"] = 10.0
    out["limit_up_count"] = int(breadth_row.get("limit_up_count", 0) or 0)
    out["limit_down_count"] = int(breadth_row.get("limit_down_count", 0) or 0)
    out["max_consecutive_limit_up"] = int(breadth_row.get("max_consecutive_limit_up", 0) or 0)
    out["zhaban_rate"] = float(breadth_row.get("zhaban_rate", 0.0) or 0.0)
    return out


# ---------------------------------------------------------------------------
# Aggregate to regime + risk level
# ---------------------------------------------------------------------------

def _decide_regime(
    bull_idx: int,
    bear_idx: int,
    vol_expand: int,
    vol_contract: int,
    breadth: dict[str, float],
    config: MarketRegimeConfig,
) -> tuple[str, str, str]:
    n_idx = max(1, len(config.indices))
    bull_ratio = bull_idx / n_idx
    bear_ratio = bear_idx / n_idx
    expand_ratio = vol_expand / n_idx
    adv_ratio = breadth["advance_decline_ratio"]
    lu = breadth["limit_up_count"]
    ld = breadth["limit_down_count"]
    lu_streak = breadth["max_consecutive_limit_up"]
    zhaban = breadth["zhaban_rate"]

    reasons: list[str] = []
    # CRISIS — multiple severe disconfirms
    severe_signals = 0
    if bear_ratio >= 0.80:
        severe_signals += 1
        reasons.append(f"bear_index_ratio={bear_ratio:.2f}")
    if ld >= config.limit_down_severe:
        severe_signals += 1
        reasons.append(f"limit_down={ld}")
    if zhaban >= config.zhaban_severe:
        severe_signals += 1
        reasons.append(f"zhaban={zhaban:.2f}")
    if adv_ratio <= 0.20:
        severe_signals += 1
        reasons.append(f"adv_decline_ratio={adv_ratio:.2f}")
    if severe_signals >= 2:
        return "crisis", "severe", ",".join(reasons)

    # BEAR CAPITULATION — predominantly bear + heavy limit-down
    if bear_ratio >= 0.60 and ld >= config.limit_down_severe // 2:
        return "bear_capitulation", "severe", f"bear_ratio={bear_ratio:.2f},ld={ld}"

    # CAUTION — softening breadth or contracting vol with weak indices
    if bear_ratio >= 0.40 or adv_ratio <= config.advance_decline_weak:
        return "caution", "high", f"bear_ratio={bear_ratio:.2f},adv={adv_ratio:.2f}"

    # BULL EXPANSION — broad bullish trend + volume expansion + strong breadth
    if (
        bull_ratio >= 0.80
        and expand_ratio >= 0.40
        and adv_ratio >= config.advance_decline_strong
        and lu >= config.limit_up_strong
    ):
        return "bull_expansion", "low", (
            f"bull_ratio={bull_ratio:.2f},expand={expand_ratio:.2f},"
            f"adv={adv_ratio:.2f},lu={lu}"
        )

    # BULL CONSOLIDATION — trending but on weaker volume
    if bull_ratio >= 0.60 and adv_ratio >= 1.0:
        return "bull_consolidation", "low", (
            f"bull_ratio={bull_ratio:.2f},adv={adv_ratio:.2f}"
        )

    # NORMAL — none of the above triggers
    return "normal", "medium", (
        f"bull_ratio={bull_ratio:.2f},bear_ratio={bear_ratio:.2f},adv={adv_ratio:.2f}"
    )


def detect_market_regime(
    *,
    trade_date: pd.Timestamp,
    index_panel: pd.DataFrame | None = None,
    breadth: pd.DataFrame | pd.Series | None = None,
    config: MarketRegimeConfig | None = None,
) -> MarketRegimeSnapshot:
    """Compute one regime snapshot for ``trade_date``.

    ``breadth`` may be either a one-row DataFrame keyed on the
    target date or a single ``pd.Series`` view of the breadth metrics.
    """
    cfg = config or MarketRegimeConfig()
    bull_idx, bear_idx, vol_expand, vol_contract = _classify_index_trend_volume(
        index_panel, trade_date=trade_date, config=cfg
    )
    breadth_row: pd.Series | None
    if isinstance(breadth, pd.DataFrame):
        if "trade_date" in breadth.columns:
            br = breadth.copy()
            br["trade_date"] = pd.to_datetime(br["trade_date"], errors="coerce")
            sel = br[br["trade_date"] == pd.Timestamp(trade_date)]
            breadth_row = sel.iloc[-1] if not sel.empty else None
        else:
            breadth_row = breadth.iloc[-1] if len(breadth) else None
    else:
        breadth_row = breadth
    breadth_metrics = _classify_breadth(breadth_row, cfg)
    regime, risk_level, reason = _decide_regime(
        bull_idx, bear_idx, vol_expand, vol_contract, breadth_metrics, cfg
    )
    return MarketRegimeSnapshot(
        trade_date=pd.Timestamp(trade_date),
        regime=regime,
        risk_level=risk_level,
        bull_index_count=bull_idx,
        bear_index_count=bear_idx,
        volume_expanding_count=vol_expand,
        volume_contracting_count=vol_contract,
        advance_decline_ratio=breadth_metrics["advance_decline_ratio"],
        limit_up_count=breadth_metrics["limit_up_count"],
        limit_down_count=breadth_metrics["limit_down_count"],
        max_consecutive_limit_up=breadth_metrics["max_consecutive_limit_up"],
        zhaban_rate=breadth_metrics["zhaban_rate"],
        reason=reason,
    )


def detect_market_regime_series(
    *,
    trade_dates: Iterable[pd.Timestamp],
    index_panel: pd.DataFrame | None = None,
    breadth_panel: pd.DataFrame | None = None,
    config: MarketRegimeConfig | None = None,
) -> pd.DataFrame:
    """Compute snapshots for many dates → tabular history.

    The result is indexed by ``trade_date`` so callers can plug it
    straight into a ``DecisionContext.regime_state`` (column
    ``regime``).
    """
    rows = []
    for d in sorted({pd.Timestamp(td) for td in trade_dates}):
        br_row: pd.DataFrame | pd.Series | None = None
        if breadth_panel is not None and not breadth_panel.empty and "trade_date" in breadth_panel.columns:
            br_row = breadth_panel.copy()
        snap = detect_market_regime(
            trade_date=d, index_panel=index_panel,
            breadth=br_row, config=config,
        )
        rows.append(snap.to_dict())
    return pd.DataFrame(rows)


def regime_risk_to_exposure_cap(risk_level: str) -> float:
    """Map a regime risk_level to a recommended gross exposure cap.

    Spec section 7: high_risk → trim total exposure.
    """
    return {
        "low": 0.80,
        "medium": 0.60,
        "high": 0.40,
        "severe": 0.20,
    }.get(risk_level, 0.40)


__all__ = [
    "DEFAULT_BENCHMARK_INDICES",
    "MarketRegimeConfig",
    "MarketRegimeSnapshot",
    "REGIME_LABELS",
    "RISK_LEVELS",
    "detect_market_regime",
    "detect_market_regime_series",
    "regime_risk_to_exposure_cap",
]
