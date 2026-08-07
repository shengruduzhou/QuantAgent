"""When one bar touches both the stop and the target, which happened first?

A daily bar says the price reached 9.50 and 10.50. It does not say in which order,
and if a position has a stop at 9.50 and a target at 10.50 the bar is consistent
with a loss and with a gain. That is not a rounding problem — over a long backtest
it is the difference between a strategy that works and one that does not, and the
default failure is silent: whichever branch the code checks first wins, and code
tends to check the profitable one.

So this module refuses to guess quietly. Every resolution says which rule produced
it:

* `TARGET_ONLY` / `STOP_ONLY` / `NEITHER` — the bar is unambiguous. No policy
  involved.
* `AMBIGUOUS_RESOLVED_BY_INTRABAR` — finer data was supplied and settled it. The
  only resolution that is a *measurement* rather than an assumption.
* `AMBIGUOUS_RESOLVED_CONSERVATIVELY` — the adverse level is assumed to have
  triggered first. An assumption, labelled as one.
* `AMBIGUOUS_UNRESOLVED` — the caller asked to be told rather than assumed for,
  and must decide what to do with a path it cannot price.

There is deliberately no policy that resolves an ambiguous bar in the position's
favour. Not "not recommended" — absent, so it cannot be selected by a
configuration file, and `test_no_policy_resolves_ambiguity_favourably` fails if one
is ever added.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from quantagent.domain.orders import Side


class AmbiguityPolicy(str, Enum):
    """What to do with a bar that is consistent with both outcomes.

    Two members, and the missing third is the point: there is no policy that picks
    the favourable outcome.
    """

    #: Assume the adverse level triggered first. Pessimistic, and stated as an
    #: assumption in the result.
    CONSERVATIVE = "CONSERVATIVE"
    #: Report the ambiguity and resolve nothing. For callers that would rather skip
    #: a trade than price a path they cannot observe.
    MARK_AMBIGUOUS = "MARK_AMBIGUOUS"


class PathResolution(str, Enum):
    TARGET_ONLY = "TARGET_ONLY"
    STOP_ONLY = "STOP_ONLY"
    NEITHER = "NEITHER"
    AMBIGUOUS_RESOLVED_BY_INTRABAR = "AMBIGUOUS_RESOLVED_BY_INTRABAR"
    AMBIGUOUS_RESOLVED_CONSERVATIVELY = "AMBIGUOUS_RESOLVED_CONSERVATIVELY"
    AMBIGUOUS_UNRESOLVED = "AMBIGUOUS_UNRESOLVED"


#: Resolutions where the bar itself settled the question. Anything outside this set
#: involved an assumption or finer data, and a report that does not distinguish the
#: two is describing a measurement it did not make.
UNAMBIGUOUS: frozenset[PathResolution] = frozenset(
    {PathResolution.TARGET_ONLY, PathResolution.STOP_ONLY, PathResolution.NEITHER}
)


class InvalidBracket(ValueError):
    """The stop and target are on the wrong sides of each other."""


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLC bar. Validated, because an inconsistent bar resolves anything."""

    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"bar low {self.low} exceeds high {self.high}")
        for name in ("open", "close"):
            price = getattr(self, name)
            if not (self.low - 1e-9 <= price <= self.high + 1e-9):
                raise ValueError(
                    f"bar {name} {price} lies outside [{self.low}, {self.high}]: an "
                    "inconsistent bar can be made to resolve either way"
                )


@dataclass(frozen=True, slots=True)
class BarPathOutcome:
    """What happened, and — crucially — how that was decided."""

    resolution: PathResolution
    #: The level that triggered, or None when neither did or nothing was resolved.
    triggered: str | None
    #: The price the trigger would fill at. A gap through a level fills at the
    #: bar's open, not at the level: assuming otherwise credits a price the market
    #: never offered.
    trigger_price: float | None
    #: True whenever the bar alone could not settle it, regardless of how it was
    #: then resolved. This is the field a report must aggregate.
    ambiguous: bool
    #: The rule that produced the resolution, for the audit trail.
    rule: str

    @property
    def is_assumption(self) -> bool:
        return self.resolution is PathResolution.AMBIGUOUS_RESOLVED_CONSERVATIVELY

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution.value,
            "triggered": self.triggered,
            "triggerPrice": self.trigger_price,
            "ambiguous": self.ambiguous,
            "isAssumption": self.is_assumption,
            "rule": self.rule,
        }


def _touched(bar: Bar, level: float, *, from_above: bool) -> bool:
    """Whether the bar reached `level`. `from_above` means the level is below."""
    return bar.low <= level + 1e-9 if from_above else bar.high >= level - 1e-9


def _fill_price(bar: Bar, level: float, *, adverse_below: bool) -> float:
    """Where a trigger actually fills.

    A bar that opens beyond the level gapped through it: the first price available
    was the open, and it was worse than the level. Filling at the level would credit
    a price the market never offered — the same error class as an unbounded market
    order filling through a limit board.
    """
    if adverse_below:
        return min(bar.open, level) if bar.open <= level else level
    return max(bar.open, level) if bar.open >= level else level


def resolve_same_bar(
    bar: Bar,
    *,
    side: Side,
    stop: float,
    target: float,
    intrabar: Sequence[float] | None = None,
    policy: AmbiguityPolicy = AmbiguityPolicy.CONSERVATIVE,
) -> BarPathOutcome:
    """Decide which of a stop and a target triggered within one bar.

    `intrabar` is a price path at finer resolution. When supplied it *settles* the
    question rather than assuming, and the resolution says so — which is the whole
    reason for distinguishing `AMBIGUOUS_RESOLVED_BY_INTRABAR` from
    `AMBIGUOUS_RESOLVED_CONSERVATIVELY`.
    """
    long_side = side is Side.BUY
    if long_side and stop >= target:
        raise InvalidBracket(
            f"a long bracket needs stop < target, got stop={stop} target={target}"
        )
    if not long_side and stop <= target:
        raise InvalidBracket(
            f"a short bracket needs stop > target, got stop={stop} target={target}"
        )

    stop_hit = _touched(bar, stop, from_above=long_side)
    target_hit = _touched(bar, target, from_above=not long_side)

    if not stop_hit and not target_hit:
        return BarPathOutcome(
            resolution=PathResolution.NEITHER, triggered=None, trigger_price=None,
            ambiguous=False, rule="bar reached neither level",
        )
    if stop_hit and not target_hit:
        return BarPathOutcome(
            resolution=PathResolution.STOP_ONLY, triggered="stop",
            trigger_price=_fill_price(bar, stop, adverse_below=long_side),
            ambiguous=False, rule="bar reached only the stop",
        )
    if target_hit and not stop_hit:
        return BarPathOutcome(
            resolution=PathResolution.TARGET_ONLY, triggered="target",
            trigger_price=_fill_price(bar, target, adverse_below=not long_side),
            ambiguous=False, rule="bar reached only the target",
        )

    # Both touched. The bar cannot say which came first.
    if intrabar:
        first = _first_touch(intrabar, side=side, stop=stop, target=target)
        if first is not None:
            level, price = first
            return BarPathOutcome(
                resolution=PathResolution.AMBIGUOUS_RESOLVED_BY_INTRABAR,
                triggered=level, trigger_price=price, ambiguous=True,
                rule="intrabar path observed; first touch settles the order",
            )

    if policy is AmbiguityPolicy.MARK_AMBIGUOUS:
        return BarPathOutcome(
            resolution=PathResolution.AMBIGUOUS_UNRESOLVED, triggered=None,
            trigger_price=None, ambiguous=True,
            rule="both levels touched; caller asked to be told rather than assumed for",
        )
    return BarPathOutcome(
        resolution=PathResolution.AMBIGUOUS_RESOLVED_CONSERVATIVELY,
        triggered="stop",
        trigger_price=_fill_price(bar, stop, adverse_below=long_side),
        ambiguous=True,
        rule=(
            "both levels touched and no intrabar path was supplied; the adverse level "
            "is assumed to have triggered first"
        ),
    )


def _first_touch(
    path: Iterable[float], *, side: Side, stop: float, target: float
) -> tuple[str, float] | None:
    """Walk a finer path and report which level it reached first."""
    long_side = side is Side.BUY
    for price in path:
        if (price <= stop + 1e-9) if long_side else (price >= stop - 1e-9):
            return "stop", stop
        if (price >= target - 1e-9) if long_side else (price <= target + 1e-9):
            return "target", target
    return None


def ambiguity_report(outcomes: Iterable[BarPathOutcome]) -> dict[str, Any]:
    """Aggregate for a run's evidence.

    Reports assumptions separately from measurements, because "3% of exits were
    ambiguous" and "3% of exits were priced by assumption" are the same number
    describing very different confidence.
    """
    resolved = list(outcomes)
    ambiguous = [o for o in resolved if o.ambiguous]
    return {
        "total": len(resolved),
        "ambiguous": len(ambiguous),
        "resolvedByIntrabar": sum(
            1 for o in ambiguous
            if o.resolution is PathResolution.AMBIGUOUS_RESOLVED_BY_INTRABAR
        ),
        "resolvedByAssumption": sum(1 for o in ambiguous if o.is_assumption),
        "leftUnresolved": sum(
            1 for o in ambiguous if o.resolution is PathResolution.AMBIGUOUS_UNRESOLVED
        ),
        "byResolution": {
            resolution.value: sum(1 for o in resolved if o.resolution is resolution)
            for resolution in PathResolution
        },
    }


__all__ = [
    "AmbiguityPolicy",
    "Bar",
    "BarPathOutcome",
    "InvalidBracket",
    "PathResolution",
    "UNAMBIGUOUS",
    "ambiguity_report",
    "resolve_same_bar",
]
