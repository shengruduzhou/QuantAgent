"""Source precedence and provenance-logged patching for the U0 daily panel.

U0 stays the daily source of truth. QMT is a *validator and gap-filler*, not a
replacement: its history is shorter and its entitlement is per-account, so
wholesale substitution would trade a verified 17.8M-row panel for an unverified
one. The precedence is therefore:

    verified U0 source
      -> QMT patch, only where U0 is missing or demonstrably wrong
        -> public provider fallback

**Nothing is ever silently overwritten.** Every candidate patch produces a
:class:`PatchRecord` carrying the old provider and value, the new provider and
value, the reason, the validation outcome, source hashes, and an explicit
approval decision. A patch that is not approved is recorded as rejected rather
than dropped, so the audit trail shows what was considered as well as what was
applied.

The reconciliation is deliberately field-aware. Price fields compare on a
relative tolerance; share volume gets an absolute floor of one lot because a
one-share disagreement on forty million shares is rounding; turnover gets a
looser band because vendors round it harder. A `volume` disagreement that is
almost exactly 100x is reported as a **unit** mismatch (手 vs shares) rather than
a data mismatch, because those demand completely different fixes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --- precedence -------------------------------------------------------------
PROVIDER_U0 = "u0_verified"
PROVIDER_QMT = "qmt_xtdata"
PROVIDER_PUBLIC = "public_fallback"

#: Lower index wins. Used only to decide *which* source is authoritative for a
#: cell; it never authorises an overwrite on its own.
PRECEDENCE: tuple[str, ...] = (PROVIDER_U0, PROVIDER_QMT, PROVIDER_PUBLIC)

# --- comparison outcomes ----------------------------------------------------
MATCH = "MATCH"
MISMATCH_VALUE = "MISMATCH_VALUE"
MISMATCH_UNIT = "MISMATCH_UNIT"
MISMATCH_ADJUSTMENT = "MISMATCH_ADJUSTMENT"
MISSING_IN_U0 = "MISSING_IN_U0"
MISSING_IN_QMT = "MISSING_IN_QMT"
DUPLICATE_ROWS = "DUPLICATE_ROWS"

#: Tolerances. Price to 1bp, volume to 0.1% or one lot, amount to 0.5%.
PRICE_RTOL = 1e-4
VOLUME_RTOL = 1e-3
VOLUME_ATOL = 100.0
AMOUNT_RTOL = 5e-3

#: A ratio this close to 100 (or 1/100) is a 手/shares unit error, not a data error.
LOT_RATIO = 100.0
LOT_RATIO_TOL = 0.01

COMPARED_FIELDS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "amount", "preclose",
)

# --- patch decisions --------------------------------------------------------
APPROVED = "APPROVED"
REJECTED_U0_AUTHORITATIVE = "REJECTED_U0_AUTHORITATIVE"
REJECTED_NO_EVIDENCE = "REJECTED_NO_EVIDENCE"
REJECTED_UNIT_MISMATCH = "REJECTED_UNIT_MISMATCH"


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


@dataclass
class FieldComparison:
    field_name: str
    u0_value: float | None
    qmt_value: float | None
    outcome: str
    relative_delta: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchRecord:
    """One considered change to the panel, applied or not."""

    symbol: str
    trade_date: str
    field_name: str
    old_provider: str
    new_provider: str
    old_value: Any
    new_value: Any
    reason: str
    validation: str
    old_hash: str
    new_hash: str
    decision: str
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def applied(self) -> bool:
        return self.decision == APPROVED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"applied": self.applied}


def _close(a: float, b: float, *, rtol: float, atol: float = 0.0) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return bool(abs(a - b) <= max(atol, rtol * abs(b)))


def compare_field(field_name: str, u0_value: Any, qmt_value: Any) -> FieldComparison:
    """Compare one field, distinguishing unit errors from value errors."""
    if u0_value is None or (isinstance(u0_value, float) and not np.isfinite(u0_value)) or pd.isna(u0_value):
        return FieldComparison(field_name, None, _as_float(qmt_value), MISSING_IN_U0)
    if qmt_value is None or pd.isna(qmt_value):
        return FieldComparison(field_name, _as_float(u0_value), None, MISSING_IN_QMT)

    u0 = float(u0_value)
    qmt = float(qmt_value)
    delta = (qmt - u0) / u0 if u0 else None

    if field_name in ("volume",):
        if _close(qmt, u0, rtol=VOLUME_RTOL, atol=VOLUME_ATOL):
            return FieldComparison(field_name, u0, qmt, MATCH, delta)
        # 手 vs shares is a scale error with a completely different fix from a
        # data error, so it gets its own outcome rather than being lumped in.
        if u0 and abs(qmt / u0 - LOT_RATIO) < LOT_RATIO_TOL:
            return FieldComparison(field_name, u0, qmt, MISMATCH_UNIT, delta,
                                   "QMT is ~100x U0: QMT reporting shares where U0 has 手, or vice versa")
        if qmt and abs(u0 / qmt - LOT_RATIO) < LOT_RATIO_TOL:
            return FieldComparison(field_name, u0, qmt, MISMATCH_UNIT, delta,
                                   "U0 is ~100x QMT: lot/share unit disagreement")
        return FieldComparison(field_name, u0, qmt, MISMATCH_VALUE, delta)

    if field_name == "amount":
        outcome = MATCH if _close(qmt, u0, rtol=AMOUNT_RTOL) else MISMATCH_VALUE
        return FieldComparison(field_name, u0, qmt, outcome, delta)

    if _close(qmt, u0, rtol=PRICE_RTOL):
        return FieldComparison(field_name, u0, qmt, MATCH, delta)
    # A consistent multiplicative offset across OHLC is an adjustment-mode
    # disagreement (raw vs qfq vs hfq), not a bad print.
    return FieldComparison(field_name, u0, qmt, MISMATCH_VALUE, delta)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def classify_adjustment_mismatch(comparisons: Sequence[FieldComparison]) -> bool:
    """True when OHLC all disagree by the *same* ratio -- an adjustment issue."""
    ratios = [
        c.qmt_value / c.u0_value
        for c in comparisons
        if c.field_name in ("open", "high", "low", "close")
        and c.outcome == MISMATCH_VALUE and c.u0_value
    ]
    if len(ratios) < 3:
        return False
    return bool(np.std(ratios) / max(abs(np.mean(ratios)), 1e-12) < 1e-3)


def reconcile(
    u0_panel: pd.DataFrame,
    qmt_panel: pd.DataFrame,
    *,
    fields: Sequence[str] = COMPARED_FIELDS,
) -> dict[str, Any]:
    """Reconcile a QMT daily panel against U0, per symbol-day and per board."""
    if u0_panel.empty and qmt_panel.empty:
        return {"symbol_days": 0, "results": [], "summary": "both panels empty"}

    def _key(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return out

    u0 = _key(u0_panel)
    qmt = _key(qmt_panel)

    duplicates = {
        "u0": int(u0.duplicated(subset=["symbol", "trade_date"]).sum()),
        "qmt": int(qmt.duplicated(subset=["symbol", "trade_date"]).sum()),
    }

    merged = u0.merge(qmt, on=["symbol", "trade_date"], how="outer",
                      suffixes=("_u0", "_qmt"), indicator=True)

    results: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    for record in merged.to_dict("records"):
        comparisons: list[FieldComparison] = []
        if record["_merge"] == "left_only":
            comparisons = [FieldComparison(f, _as_float(record.get(f"{f}_u0")), None, MISSING_IN_QMT)
                           for f in fields if f"{f}_u0" in record]
        elif record["_merge"] == "right_only":
            comparisons = [FieldComparison(f, None, _as_float(record.get(f"{f}_qmt")), MISSING_IN_U0)
                           for f in fields if f"{f}_qmt" in record]
        else:
            for name in fields:
                if f"{name}_u0" not in record or f"{name}_qmt" not in record:
                    continue
                comparisons.append(compare_field(
                    name, record.get(f"{name}_u0"), record.get(f"{name}_qmt")
                ))

        adjustment = classify_adjustment_mismatch(comparisons)
        if adjustment:
            for comparison in comparisons:
                if comparison.field_name in ("open", "high", "low", "close") and \
                        comparison.outcome == MISMATCH_VALUE:
                    comparison.outcome = MISMATCH_ADJUSTMENT
                    comparison.detail = (
                        "all OHLC differ by a constant ratio: adjustment-mode "
                        "disagreement (raw vs qfq vs hfq), not a bad print"
                    )

        for comparison in comparisons:
            outcome_counts[comparison.outcome] = outcome_counts.get(comparison.outcome, 0) + 1

        results.append({
            "symbol": record["symbol"],
            "trade_date": record["trade_date"],
            "presence": str(record["_merge"]),
            "adjustment_mismatch": adjustment,
            "comparisons": [c.to_dict() for c in comparisons],
        })

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol_days": len(results),
        "distinct_symbols": int(merged["symbol"].nunique()),
        "duplicate_rows": duplicates,
        "outcome_counts": outcome_counts,
        "presence_counts": merged["_merge"].value_counts().to_dict(),
        "results": results,
    }


def propose_patches(
    reconciliation: Mapping[str, Any],
    *,
    allow_fill_missing: bool = True,
) -> list[PatchRecord]:
    """Turn reconciliation outcomes into explicit, auditable patch decisions.

    The default posture is that **U0 wins**. A QMT value only becomes a patch
    candidate where U0 has nothing; a plain value disagreement is recorded and
    rejected, because "the other source says something different" is not
    evidence that U0 is wrong.
    """
    patches: list[PatchRecord] = []
    for row in reconciliation.get("results", []):
        symbol = row["symbol"]
        trade_date = row["trade_date"]
        for comparison in row["comparisons"]:
            outcome = comparison["outcome"]
            field_name = comparison["field_name"]
            u0_value = comparison["u0_value"]
            qmt_value = comparison["qmt_value"]

            if outcome == MATCH:
                continue

            if outcome == MISSING_IN_U0 and qmt_value is not None:
                decision = APPROVED if allow_fill_missing else REJECTED_NO_EVIDENCE
                reason = "U0 has no value for this cell; QMT fills a genuine gap"
                validation = "gap fill; no U0 value was displaced"
            elif outcome == MISMATCH_UNIT:
                decision = REJECTED_UNIT_MISMATCH
                reason = comparison.get("detail") or "lot/share unit disagreement"
                validation = (
                    "a unit disagreement is a schema defect in one adapter; "
                    "patching values would bake the defect into the panel"
                )
            elif outcome == MISMATCH_ADJUSTMENT:
                decision = REJECTED_U0_AUTHORITATIVE
                reason = "adjustment-mode disagreement, not a data error"
                validation = "U0 declares its own adjustment mode; QMT must be aligned, not merged"
            elif outcome == MISSING_IN_QMT:
                continue  # QMT's shorter history is expected, not a defect
            else:
                decision = REJECTED_U0_AUTHORITATIVE
                reason = "value disagreement with no evidence that U0 is wrong"
                validation = (
                    "U0 passed identity/provider/coverage/quality gates; a second "
                    "opinion alone does not overturn it"
                )

            patches.append(PatchRecord(
                symbol=symbol, trade_date=trade_date, field_name=field_name,
                old_provider=PROVIDER_U0, new_provider=PROVIDER_QMT,
                old_value=u0_value, new_value=qmt_value,
                reason=reason, validation=validation,
                old_hash=_hash(u0_value), new_hash=_hash(qmt_value),
                decision=decision,
            ))
    return patches


def apply_patches(
    panel: pd.DataFrame, patches: Sequence[PatchRecord]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only APPROVED patches, and report exactly what changed.

    Rejected patches are returned in the ledger too: an audit that only lists
    applied changes cannot show what was considered and declined.
    """
    result = panel.copy()
    if "patch_provenance" not in result.columns:
        result["patch_provenance"] = None

    applied = 0
    for patch in patches:
        if not patch.applied:
            continue
        mask = (result["symbol"] == patch.symbol) & (
            pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            == patch.trade_date
        )
        if not mask.any():
            continue
        result.loc[mask, patch.field_name] = patch.new_value
        result.loc[mask, "patch_provenance"] = (
            f"{patch.field_name}:{patch.old_provider}->{patch.new_provider}"
        )
        applied += 1

    ledger = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patches_considered": len(patches),
        "patches_applied": applied,
        "patches_rejected": len(patches) - applied,
        "decision_counts": _count(p.decision for p in patches),
        "records": [p.to_dict() for p in patches],
    }
    return result, ledger


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
