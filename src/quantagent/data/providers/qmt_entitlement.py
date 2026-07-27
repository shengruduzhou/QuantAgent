"""QMT / xtquant capability catalogue and entitlement matrix.

The distinction this module exists to enforce, stated once: **a documented API
is not a granted entitlement, and an empty result is not an empty dataset.**

Those two confusions are how a research stack ends up believing it has Level-2
because ``get_l2_order`` appears in the SDK, or believing a stock was never ST
because a permission-denied call returned ``{}``. Every cell here therefore
carries a probe status that names exactly which of the following was
established:

1. the function exists in the SDK;
2. the function imports;
3. MiniQMT is connected;
4. the account holds permission;
5. data downloads;
6. data is non-empty;
7. the semantics are understood;
8. the history is long enough;
9. it may lawfully be stored;
10. it is fit for production.

``SERVING`` requires all of 1-6 plus a declared semantic. Anything less names
its own obstacle.

Permission classes are deliberately not a "free" boolean. QMT access is layered
across what the broker bundles, what a basic account gets, what VIP adds, and
what must be purchased separately -- and the split differs per broker, which is
why ``BROKER_DEPENDENT`` exists as a first-class answer.

Official sources (accessed 2026-07-28):

* https://dict.thinktrader.net/nativeApi/xtdata.html
* https://dict.thinktrader.net/innerApi/data_function.html
* https://dict.thinktrader.net/dictionary/stock.html
* https://dict.thinktrader.net/xuntou_install.md

Note recorded from those sources: the Level-2 section states
「获取lv2数据时需要数据终端有lv2数据权限」 -- Level-2 requires the terminal to hold
Level-2 authorisation. The same documents give **no** concrete VIP slot counts
or history-length limits, so any such figure must come from probing a real
account rather than from a comparison table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# permission classes
# ---------------------------------------------------------------------------
#: Bundled by the broker with an ordinary funded account.
BROKER_INCLUDED = "BROKER_INCLUDED"
#: Included in the basic (non-VIP) QMT data tier.
BASIC_INCLUDED = "BASIC_INCLUDED"
#: Requires the VIP data tier.
VIP_INCLUDED = "VIP_INCLUDED"
#: Sold as a separate data product on top of any tier.
SEPARATE_PURCHASE_REQUIRED = "SEPARATE_PURCHASE_REQUIRED"
#: Varies by brokerage; cannot be answered generically.
BROKER_DEPENDENT = "BROKER_DEPENDENT"
#: Genuinely not established yet -- the honest default.
UNKNOWN_UNTIL_PROBED = "UNKNOWN_UNTIL_PROBED"
#: Confirmed not obtainable at any tier.
UNAVAILABLE = "UNAVAILABLE"
#: Cannot be reached from this operating system at all.
PLATFORM_BLOCKED = "PLATFORM_BLOCKED"

PERMISSION_CLASSES: tuple[str, ...] = (
    BROKER_INCLUDED, BASIC_INCLUDED, VIP_INCLUDED, SEPARATE_PURCHASE_REQUIRED,
    BROKER_DEPENDENT, UNKNOWN_UNTIL_PROBED, UNAVAILABLE, PLATFORM_BLOCKED,
)

# ---------------------------------------------------------------------------
# probe statuses
# ---------------------------------------------------------------------------
#: A real call returned real, non-empty, semantically understood rows.
SERVING = "SERVING"
#: The call succeeded and the provider legitimately has nothing for this key.
EMPTY_VERIFIED = "EMPTY_VERIFIED"
#: Empty came back but permission could not be confirmed -- NOT a valid empty
#: dataset. This is the status that must never be read as "never ST".
EMPTY_UNVERIFIED = "EMPTY_UNVERIFIED"
#: The provider explicitly refused on authorisation grounds.
PERMISSION_DENIED = "PERMISSION_DENIED"
#: Fewer rows/dates than requested came back without an error.
TRUNCATED = "TRUNCATED"
#: The client/terminal cannot run on this host.
PLATFORM_UNAVAILABLE = "PLATFORM_UNAVAILABLE"
#: xtquant imports but no MiniQMT answered.
CLIENT_DISCONNECTED = "CLIENT_DISCONNECTED"
#: The SDK exposes it; it was never called here.
NOT_PROBED = "NOT_PROBED"
#: Called and errored for a non-permission reason.
ERROR = "ERROR"
#: Returned content whose meaning is not documented.
UNKNOWN_SEMANTICS = "UNKNOWN_SEMANTICS"

PROBE_STATUSES: tuple[str, ...] = (
    SERVING, EMPTY_VERIFIED, EMPTY_UNVERIFIED, PERMISSION_DENIED, TRUNCATED,
    PLATFORM_UNAVAILABLE, CLIENT_DISCONNECTED, NOT_PROBED, ERROR, UNKNOWN_SEMANTICS,
)

#: Statuses that may never be treated as usable data.
NOT_USABLE_STATUSES: frozenset[str] = frozenset({
    EMPTY_UNVERIFIED, PERMISSION_DENIED, PLATFORM_UNAVAILABLE,
    CLIENT_DISCONNECTED, NOT_PROBED, ERROR, UNKNOWN_SEMANTICS,
})


@dataclass
class CapabilitySpec:
    """A documented QMT capability, before any probing."""

    capability: str
    api: str
    family: str
    documented: bool = True
    source_url: str = ""
    notes: str = ""
    #: What the vendor documentation *claims*, never what we measured.
    documented_permission_hint: str = UNKNOWN_UNTIL_PROBED


XTDATA_DOC = "https://dict.thinktrader.net/nativeApi/xtdata.html"
INNER_DOC = "https://dict.thinktrader.net/innerApi/data_function.html"
STOCK_DOC = "https://dict.thinktrader.net/dictionary/stock.html"

#: The capability catalogue. Grouped by the mission's four families so a report
#: can state coverage per family rather than as one aggregate number.
CAPABILITY_CATALOGUE: tuple[CapabilitySpec, ...] = (
    # -- security and calendar
    CapabilitySpec("instrument_master", "get_instrument_detail", "security", source_url=XTDATA_DOC),
    CapabilitySpec("instrument_type", "get_instrument_type", "security", source_url=XTDATA_DOC),
    CapabilitySpec("active_stock_list", "get_stock_list_in_sector", "security", source_url=XTDATA_DOC),
    CapabilitySpec("delisted_identities", "get_stock_list_in_sector('沪深退市')", "security",
                   source_url=XTDATA_DOC, notes="delisted cohort membership; sector name is broker/version dependent"),
    CapabilitySpec("board_classification", "get_instrument_detail", "security", source_url=STOCK_DOC),
    CapabilitySpec("listing_date", "get_instrument_detail.OpenDate", "security", source_url=STOCK_DOC),
    CapabilitySpec("delisting_date", "get_instrument_detail.ExpireDate", "security", source_url=STOCK_DOC),
    CapabilitySpec("trading_calendar", "get_trading_dates / get_trading_calendar", "security", source_url=XTDATA_DOC),
    CapabilitySpec("trading_sessions", "get_trading_time", "security", source_url=XTDATA_DOC),
    CapabilitySpec("security_status", "get_instrument_detail.InstrumentStatus", "security", source_url=STOCK_DOC),
    CapabilitySpec("suspension_flag", "get_market_data_ex(suspendFlag)", "security", source_url=STOCK_DOC),
    CapabilitySpec("price_limits", "get_instrument_detail.UpStopPrice/DownStopPrice", "security", source_url=STOCK_DOC),

    # -- daily and intraday bars
    CapabilitySpec("daily_raw", "get_market_data_ex(period='1d', dividend_type='none')", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("daily_front_adjusted", "dividend_type='front'", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("daily_back_adjusted", "dividend_type='back'", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("adjustment_factors", "get_divid_factors", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("minute_1m", "get_market_data_ex(period='1m')", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("minute_5m", "get_market_data_ex(period='5m')", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("tick", "get_market_data_ex(period='tick')", "bars", source_url=XTDATA_DOC),
    CapabilitySpec("amount_field", "get_market_data_ex(field='amount')", "bars", source_url=STOCK_DOC),
    CapabilitySpec("volume_field", "get_market_data_ex(field='volume')", "bars", source_url=STOCK_DOC),
    CapabilitySpec("preclose_field", "get_market_data_ex(field='preClose')", "bars", source_url=STOCK_DOC),

    # -- PIT and corporate
    CapabilitySpec("st_history", "download_his_st_data / get_his_st_data", "pit",
                   source_url=XTDATA_DOC,
                   notes="the single remaining U0 strict-PIT blocker"),
    CapabilitySpec("star_st_history", "get_his_st_data (*ST episodes)", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("pt_history", "get_his_st_data (PT episodes)", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("corporate_actions", "get_divid_factors", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("dividends", "get_divid_factors", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("rights_issues", "get_divid_factors", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("financial_statements", "get_financial_data", "pit", source_url=XTDATA_DOC,
                   notes="tables Balance/Income/CashFlow/Capital/Holdernum/Top10holder/Pershareindex"),
    CapabilitySpec("sector_membership", "get_sector_list / get_stock_list_in_sector", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("index_membership", "get_index_weight", "pit", source_url=XTDATA_DOC),
    CapabilitySpec("announcements", "(not exposed by xtdata)", "pit", documented=False,
                   source_url=XTDATA_DOC, notes="no xtdata endpoint located; CNINFO remains the source"),

    # -- realtime and microstructure
    CapabilitySpec("full_market_snapshot", "subscribe_whole_quote", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("single_symbol_subscription", "subscribe_quote", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("level1_quote", "get_full_tick", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("five_level_quote", "get_market_data_ex(period='tick')", "microstructure", source_url=STOCK_DOC),
    CapabilitySpec("l2quote", "get_l2_quote / period='l2quote'", "microstructure", source_url=XTDATA_DOC,
                   notes="docs: 获取lv2数据时需要数据终端有lv2数据权限"),
    CapabilitySpec("l2quoteaux", "period='l2quoteaux'", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("l2order", "get_l2_order / period='l2order'", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("l2transaction", "get_l2_transaction / period='l2transaction'", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("l2transactioncount", "period='l2transactioncount'", "microstructure", source_url=XTDATA_DOC),
    CapabilitySpec("l2orderqueue", "get_l2thousand_queue / period='l2orderqueue'", "microstructure", source_url=XTDATA_DOC),
)

CAPABILITY_BY_NAME: dict[str, CapabilitySpec] = {c.capability: c for c in CAPABILITY_CATALOGUE}


@dataclass
class EntitlementCell:
    """One measured (or explicitly unmeasured) capability result."""

    capability: str
    api: str
    documented: bool
    platform: str
    permission_class: str
    probe_status: str
    earliest_date: str | None = None
    latest_date: str | None = None
    symbols_requested: int = 0
    symbols_returned: int = 0
    rows_returned: int = 0
    fields: list[str] = field(default_factory=list)
    sample_hash: str | None = None
    error: str | None = None
    source_url: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.permission_class not in PERMISSION_CLASSES:
            raise ValueError(
                f"unknown permission_class {self.permission_class!r}; "
                f"known: {list(PERMISSION_CLASSES)}"
            )
        if self.probe_status not in PROBE_STATUSES:
            raise ValueError(
                f"unknown probe_status {self.probe_status!r}; known: {list(PROBE_STATUSES)}"
            )
        if self.probe_status == SERVING and self.rows_returned <= 0:
            raise ValueError(
                f"{self.capability} claims SERVING with {self.rows_returned} rows; "
                "SERVING requires a live call that returned real data"
            )
        if self.probe_status == EMPTY_VERIFIED and self.permission_class in (
            UNKNOWN_UNTIL_PROBED, PLATFORM_BLOCKED
        ):
            raise ValueError(
                f"{self.capability} claims EMPTY_VERIFIED while its permission is "
                f"{self.permission_class}; an empty result can only be verified as "
                "genuinely empty once entitlement is established, otherwise it is "
                "EMPTY_UNVERIFIED"
            )

    @property
    def usable(self) -> bool:
        return self.probe_status == SERVING

    @property
    def proves_absence(self) -> bool:
        """Whether an empty result here is evidence the data does not exist."""
        return self.probe_status == EMPTY_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntitlementMatrix:
    """Collection of entitlement cells with machine-readable export."""

    def __init__(self, cells: Iterable[EntitlementCell] | None = None) -> None:
        self._cells: list[EntitlementCell] = list(cells or [])

    def add(self, cell: EntitlementCell) -> None:
        self._cells.append(cell)

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list[EntitlementCell]:
        return list(self._cells)

    def by_family(self) -> dict[str, list[EntitlementCell]]:
        grouped: dict[str, list[EntitlementCell]] = {}
        for cell in self._cells:
            family = CAPABILITY_BY_NAME.get(cell.capability)
            grouped.setdefault(family.family if family else "unknown", []).append(cell)
        return grouped

    def serving(self) -> list[EntitlementCell]:
        return [c for c in self._cells if c.usable]

    def blocked(self) -> list[EntitlementCell]:
        return [c for c in self._cells if c.probe_status in NOT_USABLE_STATUSES]

    def summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        permission_counts: dict[str, int] = {}
        for cell in self._cells:
            status_counts[cell.probe_status] = status_counts.get(cell.probe_status, 0) + 1
            permission_counts[cell.permission_class] = (
                permission_counts.get(cell.permission_class, 0) + 1
            )
        return {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capabilities": len(self._cells),
            "probe_status_counts": status_counts,
            "permission_class_counts": permission_counts,
            "serving": [c.capability for c in self.serving()],
            "not_usable": [c.capability for c in self.blocked()],
            "families": {
                family: {
                    "total": len(cells),
                    "serving": len([c for c in cells if c.usable]),
                }
                for family, cells in self.by_family().items()
            },
            "interpretation_rules": {
                "documented_api_is_not_entitlement":
                    "a capability may be documented and still never have been granted",
                "empty_is_not_absence":
                    "EMPTY_UNVERIFIED must never be read as 'this data does not exist'",
                "serving_requires_rows":
                    "SERVING is only assigned when a live call returned real rows",
            },
        }

    def write(self, directory: str | Path, *, stem: str = "entitlement_matrix") -> dict[str, str]:
        import pandas as pd

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / f"{stem}.json"
        csv_path = target / f"{stem}.csv"
        payload = self.summary() | {"cells": [c.to_dict() for c in self._cells]}
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        frame = pd.DataFrame([c.to_dict() for c in self._cells])
        if not frame.empty:
            frame["fields"] = frame["fields"].map(lambda v: "|".join(v or []))
        frame.to_csv(csv_path, index=False)
        return {"json": str(json_path), "csv": str(csv_path)}


def unprobed_matrix(*, platform: str, reason: str) -> EntitlementMatrix:
    """Full catalogue with every cell marked unreachable on this platform.

    This is what an honest run on a host without QMT produces: the complete list
    of what *would* be probed, each explicitly unmeasured, so a reader can see
    the scope of what is unknown rather than an empty file.
    """
    matrix = EntitlementMatrix()
    for spec in CAPABILITY_CATALOGUE:
        matrix.add(EntitlementCell(
            capability=spec.capability,
            api=spec.api,
            documented=spec.documented,
            platform=platform,
            permission_class=PLATFORM_BLOCKED,
            probe_status=PLATFORM_UNAVAILABLE,
            error=reason,
            source_url=spec.source_url,
            notes=spec.notes,
        ))
    return matrix
