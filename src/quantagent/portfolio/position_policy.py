"""PositionPolicy — short/mid/long position class state machine (spec section 7).

Each held name carries a ``position_class`` ∈ ``{short, mid, long}``
indicating which horizon model owns it. The policy enforces the four
spec rules:

1. Default cap 60% gross exposure; only ≥0.80 conviction + friendly
   regime may extend to 80% (this dovetails with the existing
   ``gross_exposure_budget`` gate in the decision chain).
2. Exclude ST / *ST / suspension-prone / extreme illiquidity / 一字板
   names by default.
3. Consecutive limit-up追入 is restricted: a name with
   ``consecutive_limit_up_count >= max_chase`` may not be opened.
4. Same-day-bought shares may not be sold the same day; only existing
   bottom positions can be used for T+0 做 T.
5. Cross-class transitions (e.g. ``short → mid``) require passing a
   fresh confidence/risk check.

The policy is **read-only**: it does not place orders. It produces
either an ``allow`` verdict or a list of :class:`PositionPolicyViolation`
records explaining why a candidate or position transition was refused.
The decision chain consults the policy before forwarding a candidate
to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

class PositionClass(str, Enum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"


POSITION_TRANSITIONS: dict[PositionClass, tuple[PositionClass, ...]] = {
    PositionClass.SHORT: (PositionClass.MID, PositionClass.LONG),
    PositionClass.MID: (PositionClass.SHORT, PositionClass.LONG),
    PositionClass.LONG: (PositionClass.MID, PositionClass.SHORT),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionPolicyConfig:
    default_max_gross: float = 0.60
    high_conviction_max_gross: float = 0.80
    high_conviction_threshold: float = 0.80
    high_conviction_friendly_regimes: tuple[str, ...] = ("normal", "bull")
    min_cash_buffer: float = 0.20      # at least 20% cash
    max_consecutive_limit_up_chase: int = 2
    block_st: bool = True
    block_suspended: bool = True
    block_one_word_board: bool = True      # 一字板 buy-side cap
    max_position_per_name: float = 0.10
    # Cross-class transition: minimum new-class confidence to allow flip
    transition_min_confidence: float = 0.60


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeldPosition:
    """A symbol already in inventory."""

    symbol: str
    weight: float
    position_class: PositionClass
    cost_basis: float
    open_date: pd.Timestamp
    available_shares: int = 0
    same_day_acquired: int = 0          # shares bought today — cannot sell today


@dataclass(frozen=True)
class PositionCandidate:
    """A would-be new entry."""

    symbol: str
    proposed_weight: float
    proposed_class: PositionClass
    confidence: float
    consecutive_limit_up_count: int = 0
    is_st: bool = False
    is_suspended: bool = False
    is_one_word_board: bool = False
    is_t_zero_sell: bool = False        # the action is a T+0 sell on a same-day buy


@dataclass(frozen=True)
class PositionPolicyViolation:
    symbol: str
    rule: str
    severity: str            # "block" | "warn"
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionPolicyVerdict:
    allowed: bool
    candidate: PositionCandidate
    violations: list[PositionPolicyViolation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.candidate.symbol,
            "allowed": bool(self.allowed),
            "violations": [
                {"rule": v.rule, "severity": v.severity, "reason": v.reason, "detail": dict(v.detail)}
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PositionPolicy:
    """Stateless verdict generator over a snapshot of holdings + candidate."""

    def __init__(self, config: PositionPolicyConfig | None = None) -> None:
        self.config = config or PositionPolicyConfig()

    # ── per-candidate verdict ───────────────────────────────────────────
    def evaluate_candidate(
        self,
        candidate: PositionCandidate,
        *,
        held: Sequence[HeldPosition] = (),
        global_conviction: float = 0.0,
        regime: str | None = None,
    ) -> PositionPolicyVerdict:
        cfg = self.config
        violations: list[PositionPolicyViolation] = []

        # Hard exclusions
        if cfg.block_st and candidate.is_st:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "block_st", "block",
                "candidate is ST/*ST",
            ))
        if cfg.block_suspended and candidate.is_suspended:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "block_suspended", "block",
                "candidate is currently suspended",
            ))
        if cfg.block_one_word_board and candidate.is_one_word_board and candidate.proposed_weight > 0:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "block_one_word_board", "block",
                "buy-side blocked on a 一字板 day",
            ))

        # Per-name cap
        if abs(candidate.proposed_weight) > cfg.max_position_per_name:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "max_position_per_name", "block",
                f"proposed_weight {candidate.proposed_weight:.4f} > {cfg.max_position_per_name:.4f}",
            ))

        # Limit-up chase
        if (
            candidate.consecutive_limit_up_count >= cfg.max_consecutive_limit_up_chase
            and candidate.proposed_weight > 0
        ):
            violations.append(PositionPolicyViolation(
                candidate.symbol, "limit_up_chase", "block",
                f"consecutive_limit_up={candidate.consecutive_limit_up_count} "
                f">= max_chase={cfg.max_consecutive_limit_up_chase}",
            ))

        # T+0 sell on same-day acquired shares
        if candidate.is_t_zero_sell:
            same_day = next(
                (h.same_day_acquired for h in held if h.symbol == candidate.symbol),
                0,
            )
            if same_day > 0:
                violations.append(PositionPolicyViolation(
                    candidate.symbol, "t_plus_one_violation", "block",
                    f"cannot sell {same_day} shares acquired the same day",
                    detail={"same_day_acquired": same_day},
                ))

        # Cross-class transition rule
        existing = next((h for h in held if h.symbol == candidate.symbol), None)
        if existing is not None and existing.position_class != candidate.proposed_class:
            allowed_targets = POSITION_TRANSITIONS.get(existing.position_class, ())
            if candidate.proposed_class not in allowed_targets:
                violations.append(PositionPolicyViolation(
                    candidate.symbol, "illegal_class_transition", "block",
                    f"{existing.position_class.value} → {candidate.proposed_class.value} not allowed",
                ))
            elif candidate.confidence < cfg.transition_min_confidence:
                violations.append(PositionPolicyViolation(
                    candidate.symbol, "class_transition_low_confidence", "block",
                    f"transition needs confidence ≥ {cfg.transition_min_confidence:.2f}, "
                    f"got {candidate.confidence:.2f}",
                ))

        # Gross exposure cap
        current_gross = float(sum(abs(h.weight) for h in held))
        proposed_gross = current_gross + max(0.0, candidate.proposed_weight)
        regime_friendly = regime in cfg.high_conviction_friendly_regimes
        conv_ok = global_conviction >= cfg.high_conviction_threshold
        cap = cfg.default_max_gross if not (regime_friendly and conv_ok) else cfg.high_conviction_max_gross
        if proposed_gross > cap:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "gross_exposure_cap", "block",
                f"proposed_gross={proposed_gross:.3f} > cap={cap:.3f}",
                detail={
                    "current_gross": current_gross,
                    "proposed_gross": proposed_gross,
                    "cap": cap,
                    "regime": regime,
                    "global_conviction": global_conviction,
                },
            ))
        # Cash buffer
        if 1.0 - proposed_gross < cfg.min_cash_buffer:
            violations.append(PositionPolicyViolation(
                candidate.symbol, "min_cash_buffer", "warn",
                f"cash_buffer={(1.0 - proposed_gross):.3f} < {cfg.min_cash_buffer:.3f}",
            ))

        blocking = [v for v in violations if v.severity == "block"]
        return PositionPolicyVerdict(
            allowed=len(blocking) == 0,
            candidate=candidate,
            violations=violations,
        )

    # ── helpers for batch evaluation ────────────────────────────────────
    def evaluate_batch(
        self,
        candidates: Sequence[PositionCandidate],
        *,
        held: Sequence[HeldPosition] = (),
        global_conviction: float = 0.0,
        regime: str | None = None,
    ) -> list[PositionPolicyVerdict]:
        return [
            self.evaluate_candidate(
                c, held=held, global_conviction=global_conviction, regime=regime,
            )
            for c in candidates
        ]


# ---------------------------------------------------------------------------
# Helpers — consecutive limit-up counter
# ---------------------------------------------------------------------------

def compute_consecutive_limit_up_count(
    market_panel: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    limit_up_threshold: float = 0.095,
) -> pd.Series:
    """Return a Series symbol → consecutive limit-up days ending on ``as_of``.

    A day counts as a limit-up when ``daily_return >= limit_up_threshold``.
    The count breaks the streak at the first non-limit-up day going
    backward from ``as_of``.
    """
    if market_panel is None or market_panel.empty:
        return pd.Series(dtype=int)
    work = market_panel.copy()
    if not {"trade_date", "symbol", "daily_return"}.issubset(work.columns):
        return pd.Series(dtype=int)
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work[work["trade_date"] <= pd.Timestamp(as_of_date)]
    if work.empty:
        return pd.Series(dtype=int)
    work = work.sort_values(["symbol", "trade_date"], ascending=[True, False])
    out: dict[str, int] = {}
    for sym, grp in work.groupby("symbol", sort=False):
        count = 0
        for r in grp["daily_return"].astype(float).values:
            if pd.isna(r) or r < limit_up_threshold:
                break
            count += 1
        out[str(sym)] = count
    return pd.Series(out, dtype=int)


__all__ = [
    "HeldPosition",
    "POSITION_TRANSITIONS",
    "PositionCandidate",
    "PositionClass",
    "PositionPolicy",
    "PositionPolicyConfig",
    "PositionPolicyVerdict",
    "PositionPolicyViolation",
    "compute_consecutive_limit_up_count",
]
