"""Entitlement-aware capability matrix for A-share market-data sources.

The distinction this module exists to enforce: **an API that exists is not an
API that answers.** A vendor SDK exposing ``get_l2_transaction`` tells you what
the vendor sells, not what this account may read. Both facts matter, and they
must never be collapsed into one "supported" boolean.

So every (provider, dataset family) cell carries two axes:

``status``      what a *live call* did on this host, right now.
``entitlement`` what the account is licensed for, independent of whether the
                call could be attempted here.

A cell is only ``SERVING`` when a real call returned real rows. Everything else
names the specific obstacle, so a reader can tell "we are not entitled" from
"we cannot run the client on this OS" from "we never tried".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# --- live-call status -------------------------------------------------------
#: A real call returned real rows on this host.
SERVING = "SERVING"
#: The call succeeded but the provider legitimately has nothing (e.g. a
#: delisted name outside its retention window).
EMPTY = "EMPTY_NO_DATA"
#: The provider answered with an authorisation/entitlement refusal.
UNAUTHORIZED = "UNAUTHORIZED"
#: The client library or terminal cannot run on this host at all.
CLIENT_UNAVAILABLE = "CLIENT_UNAVAILABLE"
#: Network egress, port or DNS blocked the transport.
BLOCKED_BY_ENVIRONMENT = "BLOCKED_BY_ENVIRONMENT"
#: The provider throttled us; capability is unproven, not absent.
THROTTLED = "THROTTLED"
#: The API exists in the SDK but was never called here.
NOT_PROBED = "NOT_PROBED"
#: The provider has no such endpoint.
NOT_OFFERED = "NOT_OFFERED"
#: The call returned something whose meaning the vendor will not document.
UNKNOWN_SEMANTICS = "UNKNOWN_SEMANTICS"

STATUSES: tuple[str, ...] = (
    SERVING, EMPTY, UNAUTHORIZED, CLIENT_UNAVAILABLE, BLOCKED_BY_ENVIRONMENT,
    THROTTLED, NOT_PROBED, NOT_OFFERED, UNKNOWN_SEMANTICS,
)

# --- entitlement class ------------------------------------------------------
#: No account of any kind required.
PUBLIC_FREE = "PUBLIC_FREE"
#: Free registration, then free access.
FREE_ACCOUNT = "FREE_ACCOUNT"
#: Free but delayed (typically 15 minutes).
FREE_DELAYED = "FREE_DELAYED"
#: This repository holds a paid/contracted entitlement and it is active.
ENTITLED_PAID = "ENTITLED_PAID"
#: Vendor sells it; this account does not hold it.
PAID_NOT_HELD = "PAID_NOT_HELD"
#: Requires a funded securities account at a broker offering the terminal.
BROKER_ACCOUNT_REQUIRED = "BROKER_ACCOUNT_REQUIRED"
#: Time-limited trial only.
TRIAL_ONLY = "TRIAL_ONLY"
#: Entitlement genuinely not established yet.
ENTITLEMENT_UNKNOWN = "ENTITLEMENT_UNKNOWN"

ENTITLEMENTS: tuple[str, ...] = (
    PUBLIC_FREE, FREE_ACCOUNT, FREE_DELAYED, ENTITLED_PAID, PAID_NOT_HELD,
    BROKER_ACCOUNT_REQUIRED, TRIAL_ONLY, ENTITLEMENT_UNKNOWN,
)

# --- dataset families -------------------------------------------------------
DATASET_FAMILIES: tuple[str, ...] = (
    "daily_bars_raw",
    "daily_bars_adjusted",
    "minute_bars",
    "trade_ticks",
    "level1_quote",
    "level2_snapshot",
    "level2_order_events",
    "level2_transaction_events",
    "order_queue",
    "cancellations",
    "large_order_stats",
    "auction_data",
    "security_master",
    "st_history",
    "suspension_history",
    "corporate_actions",
    "financials",
    "index_membership",
)


@dataclass
class CapabilityCell:
    """One (provider, dataset family) measurement."""

    provider: str
    dataset_family: str
    status: str
    entitlement: str
    data_class: str | None = None
    probed_at: str | None = None
    endpoint: str | None = None
    rows_returned: int | None = None
    sample_symbol: str | None = None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; known: {list(STATUSES)}")
        if self.entitlement not in ENTITLEMENTS:
            raise ValueError(
                f"unknown entitlement {self.entitlement!r}; known: {list(ENTITLEMENTS)}"
            )
        if self.status == SERVING and not self.rows_returned:
            raise ValueError(
                f"{self.provider}/{self.dataset_family} claims SERVING with no rows; "
                "a cell is only SERVING when a live call returned real data"
            )

    @property
    def available(self) -> bool:
        """Available means *proven by a live call on this host*."""
        return self.status == SERVING

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CapabilityMatrix:
    """Collection of capability cells with machine-readable export."""

    def __init__(self, cells: Iterable[CapabilityCell] | None = None) -> None:
        self._cells: list[CapabilityCell] = list(cells or [])

    def add(self, cell: CapabilityCell) -> None:
        self._cells.append(cell)

    def extend(self, cells: Iterable[CapabilityCell]) -> None:
        self._cells.extend(cells)

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list[CapabilityCell]:
        return list(self._cells)

    def serving(self) -> list[CapabilityCell]:
        return [c for c in self._cells if c.available]

    def providers_for(self, dataset_family: str) -> list[str]:
        """Providers that *proved* they serve a family on this host."""
        return sorted(
            {c.provider for c in self._cells
             if c.dataset_family == dataset_family and c.available}
        )

    def families_without_provider(
        self, families: Sequence[str] = DATASET_FAMILIES
    ) -> list[str]:
        return [f for f in families if not self.providers_for(f)]

    def blockers(self) -> list[CapabilityCell]:
        return [
            c for c in self._cells
            if c.status in (UNAUTHORIZED, CLIENT_UNAVAILABLE, BLOCKED_BY_ENVIRONMENT)
        ]

    def to_frame(self) -> pd.DataFrame:
        if not self._cells:
            return pd.DataFrame(columns=[f.name for f in CapabilityCell.__dataclass_fields__.values()])
        frame = pd.DataFrame([c.to_dict() for c in self._cells])
        frame["evidence"] = frame["evidence"].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        return frame

    def summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for cell in self._cells:
            status_counts[cell.status] = status_counts.get(cell.status, 0) + 1
        return {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cells": len(self._cells),
            "providers": sorted({c.provider for c in self._cells}),
            "status_counts": status_counts,
            "serving_cells": len(self.serving()),
            "families_with_a_serving_provider": {
                family: self.providers_for(family)
                for family in DATASET_FAMILIES
                if self.providers_for(family)
            },
            "families_without_provider": self.families_without_provider(),
            "blockers": [
                {"provider": c.provider, "dataset_family": c.dataset_family,
                 "status": c.status, "entitlement": c.entitlement, "detail": c.detail}
                for c in self.blockers()
            ],
        }

    def write(self, directory: str | Path, *, stem: str = "capability_matrix") -> dict[str, str]:
        """Persist the matrix as JSON + CSV and return the paths written."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / f"{stem}.json"
        csv_path = target / f"{stem}.csv"
        payload = self.summary() | {"cells_detail": [c.to_dict() for c in self._cells]}
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.to_frame().to_csv(csv_path, index=False)
        return {"json": str(json_path), "csv": str(csv_path)}
