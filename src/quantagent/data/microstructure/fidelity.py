"""Decide which simulator fidelity a dataset actually licenses.

The temptation in microstructure work is to run a queue-position simulator on
snapshot data because the code accepts the input. This module makes the
downgrade explicit and mechanical: fidelity is *derived* from the declared data
class and the integrity report, never chosen by the strategy author.

Levels, strongest first::

    LEVEL_A  order-event replay      queue position, order aging, cancels ahead
    LEVEL_B  snapshot replay         visible depth, depletion, approximate fills
    LEVEL_C  tick replay             bid/ask execution, latency, participation
    LEVEL_D  bar simulation          daily/minute strategies only

``LEVEL_D`` is the floor, not a failure. ``NOT_SIMULATABLE`` is the failure:
it means the data's semantics are unknown or its integrity checks failed, and
no honest simulation can be run on it at all.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

from quantagent.data.microstructure import contracts
from quantagent.data.microstructure.integrity import IntegrityReport

LEVEL_A = "LEVEL_A_ORDER_EVENT_REPLAY"
LEVEL_B = "LEVEL_B_SNAPSHOT_REPLAY"
LEVEL_C = "LEVEL_C_TICK_REPLAY"
LEVEL_D = "LEVEL_D_BAR_SIMULATION"
NOT_SIMULATABLE = "NOT_SIMULATABLE"

LEVEL_ORDER: tuple[str, ...] = (LEVEL_A, LEVEL_B, LEVEL_C, LEVEL_D, NOT_SIMULATABLE)

#: What each level is permitted to claim in a report. The backtester reads
#: these and refuses to emit a metric the level does not license.
LEVEL_CLAIMS: dict[str, tuple[str, ...]] = {
    LEVEL_A: (
        "queue_position", "order_aging", "cancellations_ahead",
        "partial_fills", "price_time_priority", "visible_depth",
        "spread_crossing", "latency", "volume_participation",
    ),
    LEVEL_B: (
        "visible_depth", "depth_depletion", "approximate_partial_fills",
        "spread_crossing", "latency", "volume_participation",
    ),
    LEVEL_C: ("spread_crossing", "latency", "volume_participation"),
    LEVEL_D: ("bar_close_execution", "volume_participation"),
    NOT_SIMULATABLE: (),
}

#: Claims that are *never* licensed below Level A, stated explicitly so a
#: reviewer can grep for them.
QUEUE_CLAIMS: frozenset[str] = frozenset(
    {"queue_position", "order_aging", "cancellations_ahead", "price_time_priority"}
)


@dataclass
class FidelityDecision:
    level: str
    reasons: list[str] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    permitted_claims: tuple[str, ...] = ()
    data_classes: list[str] = field(default_factory=list)

    def permits(self, claim: str) -> bool:
        return claim in self.permitted_claims

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"permitted_claims": list(self.permitted_claims)}


def decide_fidelity(
    *,
    data_classes: Sequence[str],
    integrity_reports: Sequence[IntegrityReport] = (),
    has_bars: bool = True,
) -> FidelityDecision:
    """Derive the highest fidelity the supplied evidence supports.

    ``data_classes`` are the declared classes of the datasets on hand.
    ``integrity_reports`` gate the result: a dataset whose checks failed, or
    whose checks could not be run, cannot license the level its class implies.
    """
    classes = [c for c in data_classes if c]
    reasons: list[str] = []
    downgrades: list[str] = []

    unknown = [c for c in classes if c not in contracts.DATA_CLASSES]
    if unknown:
        return FidelityDecision(
            NOT_SIMULATABLE,
            reasons=[f"undeclared data classes: {unknown}"],
            data_classes=classes,
        )
    if contracts.UNKNOWN_SEMANTICS in classes:
        return FidelityDecision(
            NOT_SIMULATABLE,
            reasons=["a dataset's semantics are UNKNOWN_SEMANTICS"],
            data_classes=classes,
        )

    failing = [r for r in integrity_reports if r.failed]
    skipped = [r for r in integrity_reports if r.not_run and not r.failed]
    if failing:
        detail = sorted({check for r in failing for check in r.failed})
        return FidelityDecision(
            NOT_SIMULATABLE,
            reasons=[f"integrity checks failed: {detail}"],
            data_classes=classes,
        )

    candidate = LEVEL_D if has_bars else NOT_SIMULATABLE
    if has_bars:
        reasons.append("daily/minute bars available -> Level D floor")

    if contracts.SNAPSHOT_DERIVED_TRADE_AGGREGATE in classes:
        candidate = LEVEL_C
        reasons.append(
            "3-second snapshot-derived trade aggregates available -> Level C floor"
        )
        downgrades.append(
            "trade records are 3-second aggregates, not individual prints: "
            "intra-bucket sequencing, per-trade size distribution and any "
            "sub-3s latency claim are unobservable"
        )

    if any(c in (contracts.LEVEL1_QUOTE, contracts.TRADE_TICK,
                 contracts.EXCHANGE_TRADE_EVENT) for c in classes):
        candidate = LEVEL_C
        reasons.append("trade prints or Level-1 quotes available -> Level C")

    if contracts.LEVEL2_SNAPSHOT in classes:
        candidate = LEVEL_B
        reasons.append("aggregated depth snapshots available -> Level B")
        downgrades.append(
            "snapshot depth is price-level aggregated; exact queue position is "
            "not observable and must not be claimed"
        )

    if any(c in contracts.ORDER_EVENT_CLASSES for c in classes):
        candidate = LEVEL_A
        reasons.append("per-order exchange events available -> Level A")

    # A synthetic broker feed can never license better than tick replay, and
    # only then if it is the sole evidence for the instrument.
    if contracts.BROKER_SYNTHETIC_QUOTE in classes and candidate in (LEVEL_A, LEVEL_B):
        candidate = LEVEL_C
        downgrades.append(
            "broker-synthetic quotes present; depth from an OTC/CFD venue is not "
            "an exchange order book, so the book-based levels are withdrawn"
        )

    if skipped:
        detail = sorted({check for r in skipped for check in r.not_run})
        if candidate == LEVEL_A:
            candidate = LEVEL_B
            downgrades.append(
                f"integrity checks could not be evaluated ({detail}); Level A "
                "requires proven sequence completeness, so downgraded to Level B"
            )
        else:
            downgrades.append(f"integrity checks not evaluated: {detail}")

    return FidelityDecision(
        level=candidate,
        reasons=reasons,
        downgrades=downgrades,
        permitted_claims=LEVEL_CLAIMS[candidate],
        data_classes=classes,
    )


class FidelityViolation(RuntimeError):
    """Raised when a report claims something its fidelity level forbids."""


def assert_claims_permitted(decision: FidelityDecision, claims: Sequence[str]) -> None:
    """Fail closed when a caller asserts a claim its data does not support."""
    forbidden = [c for c in claims if not decision.permits(c)]
    if forbidden:
        raise FidelityViolation(
            f"simulator fidelity {decision.level} does not license {forbidden}; "
            f"permitted claims are {list(decision.permitted_claims)}"
        )
