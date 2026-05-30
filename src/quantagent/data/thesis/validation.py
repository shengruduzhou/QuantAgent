"""Thesis validation loop — 1/5/20/60/120d look-forward.

Re-scores a :class:`CapitalFlowThesis` against realised post-event
returns. Spec section 2 mandates that:

* Theses with policy narrative but no price/volume/fundamental
  follow-through must NOT graduate to ``verified``.
* Each thesis tracks ``validation_status`` in a four-state machine
  ``unverified → partially_verified → verified | rejected``.
* ``decay_score`` falls toward 0 once the expected horizons have all
  elapsed without the thesis getting verified.

The validator is read-only on the thesis; it returns a
:class:`ThesisValidationResult` carrying the new status and the per-
horizon excess returns the verdict was based on, so callers can
audit *why* a thesis was promoted or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quantagent.data.thesis.builder import (
    CAPITAL_FLOW_THESIS_COLUMNS,
    CapitalFlowThesis,
    THESIS_VALIDATION_STATES,
    theses_to_frame,
)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThesisValidationConfig:
    """Validation thresholds.

    A horizon is "confirmed" when the *direction-aligned* return at
    that horizon (sector forward return × thesis_sign) exceeds
    ``min_confirm_excess`` after subtracting the benchmark forward
    return. A thesis with ≥ ``min_horizons_confirmed`` confirmed
    horizons graduates to ``verified``. A thesis that has reached its
    longest expected lag with zero confirmed horizons is
    ``rejected``. Anything in between is ``partially_verified``.
    """

    horizons: tuple[int, ...] = (1, 5, 20, 60, 120)
    min_confirm_excess: float = 0.02       # 2% direction-aligned excess
    reject_disconfirm_excess: float = 0.03  # 3% adverse move
    min_horizons_confirmed: int = 2
    benchmark_column: str = "benchmark_return"
    sector_return_column: str = "sector_return"
    symbol_return_column: str = "forward_return"


@dataclass
class ThesisValidationResult:
    thesis_id: str
    direction_kind: str
    direction_value: str
    thesis_sign: float
    prior_status: str
    new_status: str
    horizons_confirmed: list[int] = field(default_factory=list)
    horizons_disconfirmed: list[int] = field(default_factory=list)
    horizon_excess_returns: dict[int, float] = field(default_factory=dict)
    decay_score: float = 1.0
    tradability_score: float = 0.50
    last_validated_at: pd.Timestamp | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "thesis_id": self.thesis_id,
            "direction_kind": self.direction_kind,
            "direction_value": self.direction_value,
            "thesis_sign": float(self.thesis_sign),
            "prior_status": self.prior_status,
            "new_status": self.new_status,
            "horizons_confirmed": list(self.horizons_confirmed),
            "horizons_disconfirmed": list(self.horizons_disconfirmed),
            "horizon_excess_returns": {
                int(k): float(v) for k, v in self.horizon_excess_returns.items()
            },
            "decay_score": float(self.decay_score),
            "tradability_score": float(self.tradability_score),
            "last_validated_at": self.last_validated_at,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _select_direction_returns(
    panel: pd.DataFrame,
    thesis: CapitalFlowThesis,
    config: ThesisValidationConfig,
) -> pd.DataFrame:
    """Filter the panel down to rows relevant to this thesis direction.

    Panel must contain at minimum: ``trade_date``, the direction key
    column matching ``direction_kind``, and a forward-return column.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"])

    key_map = {
        "sector": "sector_level_1",
        "theme": "theme",
        "province": "province",
        "symbol": "symbol",
    }
    key = key_map.get(thesis.direction_kind)
    if key is None or key not in work.columns:
        return pd.DataFrame()
    return work[work[key].astype(str) == str(thesis.direction_value)]


def _compute_excess_returns(
    direction_rows: pd.DataFrame,
    thesis: CapitalFlowThesis,
    config: ThesisValidationConfig,
) -> dict[int, float]:
    """Return cumulative excess (direction − benchmark) returns over horizons.

    For each horizon ``H``, the cumulative direction return over the
    H business days starting *after* ``thesis.created_at`` is taken
    by compounding ``(1 + daily_return)`` and subtracting the same
    compound on the benchmark. The thesis sign multiplies the result
    so a bearish thesis with adverse realised excess counts as
    confirmed. A horizon is reported only when at least one
    direction-aligned row falls inside the window.
    """
    if direction_rows.empty or thesis.created_at is None:
        return {}
    work = direction_rows.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"])
    start = pd.Timestamp(thesis.created_at)
    out: dict[int, float] = {}
    bench_col = config.benchmark_column
    ret_col = (
        config.symbol_return_column
        if thesis.direction_kind == "symbol"
        else config.sector_return_column
    )
    if ret_col not in work.columns:
        return {}
    sign = np.sign(thesis.thesis_sign) if thesis.thesis_sign else 1.0
    panel_latest = work["trade_date"].max()
    for h in config.horizons:
        end = start + pd.tseries.offsets.BDay(h)
        # Skip horizons the panel hasn't reached yet — a thesis cannot
        # be confirmed at 20d by a panel that only has 5d of data.
        if pd.notna(panel_latest) and panel_latest < end:
            continue
        window = work[(work["trade_date"] > start) & (work["trade_date"] <= end)]
        if window.empty:
            continue
        target_daily = pd.to_numeric(window[ret_col], errors="coerce").dropna()
        if target_daily.empty:
            continue
        target_cum = float(np.prod(1.0 + target_daily.values) - 1.0)
        if bench_col in window.columns:
            bench_daily = pd.to_numeric(window[bench_col], errors="coerce").dropna()
            bench_cum = float(np.prod(1.0 + bench_daily.values) - 1.0) if not bench_daily.empty else 0.0
        else:
            bench_cum = 0.0
        excess = (target_cum - bench_cum) * sign
        out[int(h)] = float(excess)
    return out


def _classify(
    excess_by_h: Mapping[int, float],
    config: ThesisValidationConfig,
) -> tuple[str, list[int], list[int], str]:
    confirmed: list[int] = []
    disconfirmed: list[int] = []
    for h, excess in excess_by_h.items():
        if excess >= config.min_confirm_excess:
            confirmed.append(int(h))
        elif excess <= -config.reject_disconfirm_excess:
            disconfirmed.append(int(h))
    if not excess_by_h:
        return "unverified", confirmed, disconfirmed, "no_horizons_available"
    if len(confirmed) >= config.min_horizons_confirmed:
        return "verified", confirmed, disconfirmed, "min_horizons_confirmed_reached"
    longest_h = max(excess_by_h.keys())
    longest_h_required = max(config.horizons)
    # Reject only when we have seen enough of the horizon to fail it
    if longest_h >= longest_h_required and not confirmed:
        return "rejected", confirmed, disconfirmed, "longest_horizon_no_confirm"
    if len(disconfirmed) >= config.min_horizons_confirmed and len(confirmed) == 0:
        return "rejected", confirmed, disconfirmed, "majority_horizons_disconfirmed"
    if confirmed:
        return "partially_verified", confirmed, disconfirmed, "some_horizons_confirmed"
    return "unverified", confirmed, disconfirmed, "insufficient_evidence"


def _decay(
    excess_by_h: Mapping[int, float],
    new_status: str,
    config: ThesisValidationConfig,
) -> float:
    """1.0 when thesis is fresh; decays toward 0 as horizons elapse."""
    if not excess_by_h:
        return 1.0
    longest_seen = max(excess_by_h.keys())
    longest_required = max(config.horizons)
    base = max(0.0, 1.0 - longest_seen / longest_required)
    # Confirmed theses don't decay
    if new_status == "verified":
        return 1.0
    if new_status == "rejected":
        return 0.0
    return float(base)


def _tradability(
    thesis: CapitalFlowThesis,
    new_status: str,
    confirmed: list[int],
    disconfirmed: list[int],
) -> float:
    """Higher = more tradable. Combines status + confidence + contradiction."""
    base = {
        "verified": 0.85,
        "partially_verified": 0.55,
        "unverified": 0.40,
        "rejected": 0.05,
        "expired": 0.05,
    }.get(new_status, 0.30)
    # Reward confidence, penalise contradiction
    adj = 0.20 * float(thesis.confidence) - 0.30 * float(thesis.contradiction_score)
    if disconfirmed and not confirmed:
        adj -= 0.10
    return float(np.clip(base + adj, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Single + batch validation
# ---------------------------------------------------------------------------

def validate_thesis(
    thesis: CapitalFlowThesis,
    panel: pd.DataFrame,
    *,
    config: ThesisValidationConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> ThesisValidationResult:
    """Re-score a thesis from realised post-creation returns.

    Parameters
    ----------
    thesis:
        The thesis to validate (read-only).
    panel:
        Long-form DataFrame with at minimum ``trade_date``, the
        direction key column (``sector_level_1``, ``theme``,
        ``province`` or ``symbol``), and either
        ``sector_return`` / ``forward_return`` plus optional
        ``benchmark_return``.
    config:
        Validation thresholds.
    as_of:
        Optional override for ``last_validated_at`` (default: now).
    """
    cfg = config or ThesisValidationConfig()
    timestamp = as_of or pd.Timestamp.utcnow().tz_localize(None)
    direction_rows = _select_direction_returns(panel, thesis, cfg)
    excess = _compute_excess_returns(direction_rows, thesis, cfg)
    new_status, confirmed, disconfirmed, reason = _classify(excess, cfg)
    decay = _decay(excess, new_status, cfg)
    tradability = _tradability(thesis, new_status, confirmed, disconfirmed)
    return ThesisValidationResult(
        thesis_id=thesis.thesis_id,
        direction_kind=thesis.direction_kind,
        direction_value=thesis.direction_value,
        thesis_sign=float(thesis.thesis_sign),
        prior_status=thesis.validation_status,
        new_status=new_status,
        horizons_confirmed=confirmed,
        horizons_disconfirmed=disconfirmed,
        horizon_excess_returns=excess,
        decay_score=decay,
        tradability_score=tradability,
        last_validated_at=pd.Timestamp(timestamp),
        reason=reason,
    )


def validate_theses(
    theses: Iterable[CapitalFlowThesis],
    panel: pd.DataFrame,
    *,
    config: ThesisValidationConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[list[CapitalFlowThesis], list[ThesisValidationResult]]:
    """Validate every thesis and return updated copies + per-thesis result.

    The returned :class:`CapitalFlowThesis` list is **new** objects
    (frozen dataclasses) carrying the post-validation status,
    decay and tradability — the originals are not mutated.
    """
    results: list[ThesisValidationResult] = []
    updated: list[CapitalFlowThesis] = []
    for thesis in theses:
        result = validate_thesis(thesis, panel, config=config, as_of=as_of)
        updated.append(
            CapitalFlowThesis(
                thesis_id=thesis.thesis_id,
                direction_kind=thesis.direction_kind,
                direction_value=thesis.direction_value,
                thesis_sign=thesis.thesis_sign,
                supporting_evidence_ids=list(thesis.supporting_evidence_ids),
                contradiction_evidence_ids=list(thesis.contradiction_evidence_ids),
                confidence=thesis.confidence,
                contradiction_score=thesis.contradiction_score,
                expected_lag_days=thesis.expected_lag_days,
                tradability_score=result.tradability_score,
                decay_score=result.decay_score,
                validation_status=result.new_status,
                created_at=thesis.created_at,
                last_validated_at=result.last_validated_at,
            )
        )
        results.append(result)
    return updated, results
