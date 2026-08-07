"""The difference table. Its only interesting output is the unexplained count.

A reconciliation that reports "close enough" is worthless, so the rules here are
deliberately hostile:

* **Discrete state must match exactly.** Order status, quantities, fill counts,
  lot parcels, event sequences and lineage links get no tolerance at all. A
  one-share or one-event difference is a defect by construction.
* **A float tolerance must be declared per dimension.** There is no global
  epsilon. A dimension without an `ExplanationRule` granting one is compared
  exactly, so a tolerance can only ever be added on purpose.
* **An unmatched difference is `unexplained`, never dropped.** Classification is
  by explicit rule; the default is the failing state. That inversion is what
  stops the table from quietly growing exemptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from quantagent.reconciliation.snapshot import EconomicSnapshot

#: Classifications. `UNEXPLAINED` is the default and the only one that blocks.
BOUNDED_FLOAT = "bounded_float"
DOCUMENTED_ENGINE_DIFFERENCE = "documented_engine_difference"
NOT_APPLICABLE = "not_applicable_path"
UNEXPLAINED = "unexplained"

RESOLVED = "resolved"
BLOCKING = "blocking"


class _Absent:
    """Marker for a dimension an engine never reported. Not the same as zero."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<absent>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Absent)

    def __hash__(self) -> int:
        return hash("<absent>")


ABSENT = _Absent()

#: Dimensions that are discrete facts about what happened. No tolerance is ever
#: legitimate here: these are counts, statuses and identifiers.
_DISCRETE_SUFFIXES = (
    ".status",
    ".cumulative_quantity",
    ".last_quantity",
    ".leaves_quantity",
    ".fill_count",
    ".event_sequence",
    ".lineage_links",
)
_DISCRETE_PREFIXES = (
    "count[",
    "position[",
    "settled_inventory[",
    "sellable_inventory[",
    "reserved_inventory[",
    "lots[",
    "lineage_gaps",
)


def is_discrete(dimension: str) -> bool:
    return dimension.startswith(_DISCRETE_PREFIXES) or dimension.endswith(
        _DISCRETE_SUFFIXES
    )


@dataclass(frozen=True, slots=True)
class ExplanationRule:
    """Permission for one specific difference, naming the rule that causes it.

    `tolerance` is only honoured for continuous dimensions; granting one to a
    discrete dimension is rejected at construction rather than silently ignored,
    because "0.5 orders" is not a thing that can be true.
    """

    dimension: str
    classification: str
    source_rule: str
    explanation: str
    tolerance: float = 0.0
    #: Restrict the rule to one ordered pair of snapshot labels. `None` applies
    #: it to every comparison, which is rarely what a reviewer wants.
    pair: tuple[str, str] | None = None
    #: Prefix match instead of exact dimension match, for per-symbol families.
    prefix: bool = False

    def __post_init__(self) -> None:
        if self.tolerance and is_discrete(self.dimension):
            raise ValueError(
                f"{self.dimension} is a discrete dimension; a tolerance of "
                f"{self.tolerance} cannot be correct"
            )
        if self.classification not in {
            BOUNDED_FLOAT,
            DOCUMENTED_ENGINE_DIFFERENCE,
            NOT_APPLICABLE,
        }:
            raise ValueError(f"{self.classification} is not a permitted classification")
        if not self.explanation.strip() or not self.source_rule.strip():
            raise ValueError("an explanation rule needs both a source rule and a reason")

    def matches(self, dimension: str, pair: tuple[str, str]) -> bool:
        if self.pair is not None and self.pair != pair:
            return False
        return (
            dimension.startswith(self.dimension) if self.prefix else dimension == self.dimension
        )


@dataclass(frozen=True, slots=True)
class Difference:
    dimension: str
    entity_type: str
    entity_id: str
    symbol: str | None
    left_label: str
    right_label: str
    left_value: Any
    right_value: Any
    absolute: float | None
    relative: float | None
    classification: str
    source_rule: str
    explanation: str
    resolution_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "entityType": self.entity_type,
            "entityId": self.entity_id,
            "symbol": self.symbol,
            "leftLabel": self.left_label,
            "rightLabel": self.right_label,
            "leftValue": self.left_value,
            "rightValue": self.right_value,
            "absolute": self.absolute,
            "relative": self.relative,
            "classification": self.classification,
            "sourceRule": self.source_rule,
            "explanation": self.explanation,
            "resolutionStatus": self.resolution_status,
        }


@dataclass
class DifferenceTable:
    left_label: str
    right_label: str
    differences: list[Difference] = field(default_factory=list)
    #: Dimensions one side reported and the other did not, kept separate from
    #: value differences so a missing dimension cannot read as agreement.
    only_left: list[str] = field(default_factory=list)
    only_right: list[str] = field(default_factory=list)

    @property
    def unexplained(self) -> list[Difference]:
        return [d for d in self.differences if d.classification == UNEXPLAINED]

    @property
    def unexplained_economic_differences(self) -> int:
        return len(self.unexplained)

    @property
    def clean(self) -> bool:
        return self.unexplained_economic_differences == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "leftLabel": self.left_label,
            "rightLabel": self.right_label,
            "differences": [d.to_dict() for d in self.differences],
            "onlyLeft": sorted(self.only_left),
            "onlyRight": sorted(self.only_right),
            "unexplainedEconomicDifferences": self.unexplained_economic_differences,
            "clean": self.clean,
        }


def _entity_of(dimension: str) -> tuple[str, str, str | None]:
    """Attribute a dimension to (entity type, entity id, symbol)."""
    if dimension.startswith("order["):
        key = dimension[len("order[") : dimension.index("]")]
        return "order", key, key.split("|")[0]
    for prefix, entity in (
        ("position[", "position"),
        ("settled_inventory[", "position"),
        ("sellable_inventory[", "position"),
        ("reserved_inventory[", "position"),
        ("lots[", "position_lot"),
    ):
        if dimension.startswith(prefix):
            symbol = dimension[len(prefix) : dimension.index("]")]
            return entity, symbol, symbol
    if dimension.startswith("count["):
        return "counter", dimension[len("count[") : dimension.index("]")], None
    return "account", dimension, None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or isinstance(value, _Absent):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def compare_snapshots(
    left: EconomicSnapshot,
    right: EconomicSnapshot,
    *,
    rules: Sequence[ExplanationRule] = (),
) -> DifferenceTable:
    """Diff every dimension. Unmatched differences come back as `unexplained`."""
    return compare_flat(
        left.label, left.flatten(), right.label, right.flatten(), rules=rules
    )


def compare_flat(
    left_label: str,
    left_flat: Mapping[str, Any],
    right_label: str,
    right_flat: Mapping[str, Any],
    *,
    rules: Sequence[ExplanationRule] = (),
    restrict_to_shared: bool = False,
) -> DifferenceTable:
    """Diff two flat dimension maps.

    `restrict_to_shared` is for comparing an engine's own in-memory figures
    against a full replay: the engine maintains a subset of the dimensions, and
    the ones it does not maintain are not evidence of disagreement. It never
    weakens a shared dimension — those are still compared in full.
    """
    pair = (left_label, right_label)
    table = DifferenceTable(left_label=left_label, right_label=right_label)

    table.only_left = [k for k in left_flat if k not in right_flat]
    table.only_right = [k for k in right_flat if k not in left_flat]

    # A dimension one side never reported is compared too, against ABSENT. An
    # engine that simply does not model something would otherwise reconcile
    # perfectly by producing nothing, which is the loophole this closes; the
    # absence has to be excused by a named rule like any other difference.
    dimensions = (
        set(left_flat) & set(right_flat)
        if restrict_to_shared
        else set(left_flat) | set(right_flat)
    )
    for dimension in sorted(dimensions):
        left_value = left_flat.get(dimension, ABSENT)
        right_value = right_flat.get(dimension, ABSENT)
        if left_value == right_value:
            continue

        left_number = _numeric(left_value)
        right_number = _numeric(right_value)
        absolute = (
            abs(left_number - right_number)
            if left_number is not None and right_number is not None
            else None
        )
        relative = (
            absolute / abs(left_number)
            if absolute is not None and left_number not in (None, 0)
            else None
        )

        matched = next(
            (r for r in rules if r.matches(dimension, pair)), None
        )
        if matched is not None and matched.classification == BOUNDED_FLOAT:
            # A tolerance only excuses a difference it actually covers. Beyond it
            # the rule stops applying rather than stretching to fit.
            within = absolute is not None and absolute <= matched.tolerance
            matched = matched if within else None

        classification = matched.classification if matched else UNEXPLAINED
        entity_type, entity_id, symbol = _entity_of(dimension)
        table.differences.append(
            Difference(
                dimension=dimension,
                entity_type=entity_type,
                entity_id=entity_id,
                symbol=symbol,
                left_label=left_label,
                right_label=right_label,
                left_value=None if isinstance(left_value, _Absent) else left_value,
                right_value=None if isinstance(right_value, _Absent) else right_value,
                absolute=absolute,
                relative=relative,
                classification=classification,
                source_rule=matched.source_rule if matched else "",
                explanation=(
                    matched.explanation
                    if matched
                    else "no rule explains this difference"
                ),
                resolution_status=RESOLVED if matched else BLOCKING,
            )
        )
    return table


def missing_dimension_differences(table: DifferenceTable) -> int:
    """A dimension present on one side only is a difference in its own right."""
    return len(table.only_left) + len(table.only_right)


__all__ = [
    "ABSENT",
    "BLOCKING",
    "BOUNDED_FLOAT",
    "DOCUMENTED_ENGINE_DIFFERENCE",
    "Difference",
    "DifferenceTable",
    "ExplanationRule",
    "NOT_APPLICABLE",
    "RESOLVED",
    "UNEXPLAINED",
    "compare_flat",
    "compare_snapshots",
    "is_discrete",
    "missing_dimension_differences",
]
