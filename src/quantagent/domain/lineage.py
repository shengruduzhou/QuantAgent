"""Canonical lineage for every economic object the platform produces.

One chain of custody, shared by research simulation, the fast backtest, the
streaming backtest, paper, shadow and any future broker adapter. Before this,
each engine minted its own identifiers, so an order in a backtest could not be
traced back to the strategy version that produced it, and two engines running
the same strategy produced objects that could not be compared at all.

The rules that make the chain trustworthy:

* **Identifiers are content-derived, not random.** Re-running the same research
  step with the same inputs yields the same identifier, which is what makes
  replay, reconciliation and idempotency possible. A random UUID would make
  every replay look like new economic activity.
* **A child never invents its ancestry.** `derive` copies the parent's chain and
  only adds the new link, so lineage cannot silently diverge mid-pipeline.
* **Identifiers are immutable.** The dataclass is frozen; producing a different
  identity requires deriving a new one, which leaves the old one intact.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from typing import Any, Mapping

#: Ordered from the broadest research context down to a single execution. The
#: order matters: `describe()` and the UI drill-down both walk it.
LINEAGE_FIELDS: tuple[str, ...] = (
    "research_id",
    "experiment_id",
    "strategy_id",
    "strategy_version_id",
    "model_version_id",
    "run_id",
    "signal_id",
    "order_intent_id",
    "order_id",
    "parent_order_id",
    "broker_order_id",
    "execution_id",
    "position_lot_id",
    "risk_decision_id",
)


def content_id(prefix: str, **parts: Any) -> str:
    """A stable identifier derived from the content that defines the object.

    Two runs that genuinely produced the same thing get the same id; anything
    that differs economically gets a different one. Digest length is 16 hex
    chars — collision risk is negligible at the volumes involved, and short ids
    stay readable in a UI table and a log line.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True, slots=True)
class Lineage:
    """The chain of custody carried by every order, fill, ledger row and risk decision."""

    research_id: str | None = None
    experiment_id: str | None = None
    strategy_id: str | None = None
    strategy_version_id: str | None = None
    model_version_id: str | None = None
    run_id: str | None = None
    signal_id: str | None = None
    order_intent_id: str | None = None
    order_id: str | None = None
    parent_order_id: str | None = None
    broker_order_id: str | None = None
    execution_id: str | None = None
    position_lot_id: str | None = None
    risk_decision_id: str | None = None

    def derive(self, **updates: str | None) -> "Lineage":
        """Extend the chain. Unknown names are rejected rather than dropped."""
        unknown = set(updates) - {field.name for field in fields(self)}
        if unknown:
            raise ValueError(f"unknown lineage fields: {sorted(unknown)}")
        return replace(self, **updates)

    def as_dict(self, *, include_empty: bool = False) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in LINEAGE_FIELDS
            if include_empty or getattr(self, name) is not None
        }

    def describe(self) -> str:
        """Human-readable chain, broadest first. Used in logs and UI tooltips."""
        return " -> ".join(f"{name}={value}" for name, value in self.as_dict().items())

    @property
    def depth(self) -> int:
        return len(self.as_dict())

    def is_ancestor_of(self, other: "Lineage") -> bool:
        """Does `other` extend this chain without contradicting it?

        Reconciliation relies on this: a fill claiming an order_id that never
        belonged to the intent must not be accepted as part of that intent's
        history.
        """
        mine = self.as_dict()
        theirs = other.as_dict()
        return all(theirs.get(name) == value for name, value in mine.items())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "Lineage":
        if not payload:
            return cls()
        known = {name: payload.get(name) for name in LINEAGE_FIELDS if payload.get(name)}
        return cls(**known)


__all__ = ["LINEAGE_FIELDS", "Lineage", "content_id"]
