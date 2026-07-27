"""Reconcile canonical tick events against the verified U0 daily panel.

A tick feed that cannot rebuild the day it claims to describe is not a tick
feed for that day. This module aggregates trade events into a synthetic daily
bar and compares it to the U0 raw panel, which has already passed its own
identity/provider/coverage/quality gates.

Deliberate design choices:

* **Relative tolerances on price, absolute floor on volume.** A 1-share
  disagreement on 40 million shares is rounding; a 1-fen disagreement on a
  ¥3.00 stock is a scale error worth catching.
* **One liquid symbol proves nothing.** :func:`reconcile_days` reports per
  symbol-day and the caller must state the cohort. The summary carries
  ``symbol_days`` and ``distinct_symbols`` so a report cannot quietly
  generalise from a single name.
* **Missing panel rows are ``NO_PANEL_ROW``, not a pass.** A tick day with no
  daily counterpart is unverified, and unverified is not verified.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.data.microstructure import contracts

MATCH = "MATCH"
MISMATCH = "MISMATCH"
#: Every field a snapshot-derived aggregate *can* reproduce does reproduce, and
#: the only disagreements are the intra-bucket price extremes that aggregation
#: provably destroys. This is a distinct outcome from MATCH -- it says the feed
#: is consistent with the day *and* names what it cannot see -- and it is not a
#: licence to treat the feed as per-trade data.
MATCH_WITHIN_AGGREGATION_LIMITS = "MATCH_WITHIN_AGGREGATION_LIMITS"
NO_PANEL_ROW = "NO_PANEL_ROW"
NO_TICK_DATA = "NO_TICK_DATA"

#: Fields an aggregated feed cannot recover: the printed price of a 3-second
#: bucket is the snapshot's last price, so a high or low that occurred inside a
#: bucket without ending it is simply not in the data.
AGGREGATION_BLIND_FIELDS: frozenset[str] = frozenset({"high", "low"})

#: Price fields agree to 1 basis point; that is well inside vendor rounding but
#: far tighter than any real scale or adjustment error.
PRICE_RTOL = 1e-4
#: Share volume agrees to 0.1% or 100 shares (one lot), whichever is looser.
VOLUME_RTOL = 1e-3
VOLUME_ATOL = 100.0
#: Turnover agrees to 0.5%: vendors round amount more aggressively than volume.
AMOUNT_RTOL = 5e-3


@dataclass
class SymbolDayReconciliation:
    symbol: str
    trade_date: str
    status: str
    tick_events: int
    fields_compared: int
    fields_matched: int
    mismatched_fields: list[str]
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Phases whose prints the exchange daily bar does not include. Post-close
#: records and STAR after-hours fixed-price trades are real events, but they
#: settle outside the day's OHLCV, so folding them in would break a
#: reconciliation that should pass.
EXCLUDED_FROM_DAILY_BAR: frozenset[str] = frozenset({
    contracts.PHASE_CLOSED,
    contracts.PHASE_POST_CLOSE,
    contracts.PHASE_AFTER_HOURS,
})


def daily_bar_from_trades(trades: pd.DataFrame) -> dict[str, Any]:
    """Aggregate canonical trade events into a synthetic daily bar.

    Only phases the exchange daily bar actually covers count towards OHLCV.
    Everything else is excluded *and counted*, so a feed carrying pre-market or
    post-close prints is visible rather than silently shifting the open or
    inflating turnover.
    """
    if trades.empty:
        return {"events": 0}

    frame = trades.copy()
    if "exchange_time" in frame.columns:
        times = pd.to_datetime(frame["exchange_time"], errors="coerce")
        clock = times.dt.strftime("%H:%M")
        boards = (
            frame["symbol"].astype(str).map(contracts.board_of)
            if "symbol" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
        )
        phases = pd.Series(
            [contracts.session_phase(c, board=b) for c, b in zip(clock, boards)],
            index=frame.index,
        )
        excluded = phases.isin(EXCLUDED_FROM_DAILY_BAR)
        outside = int(excluded.sum())
        excluded_counts = phases[excluded].value_counts().to_dict()
        frame = frame.loc[~excluded]
    else:
        outside = 0
        excluded_counts = {}

    if frame.empty:
        return {"events": 0, "excluded_outside_session": outside,
                "excluded_phase_counts": {str(k): int(v) for k, v in excluded_counts.items()}}

    order = "ingest_sequence" if "ingest_sequence" in frame.columns else "event_time_ns"
    frame = frame.sort_values(order, kind="mergesort")
    price = pd.to_numeric(frame["price"], errors="coerce")
    volume = pd.to_numeric(frame["volume_shares"], errors="coerce")
    if "amount_cny" in frame.columns and frame["amount_cny"].notna().any():
        amount = pd.to_numeric(frame["amount_cny"], errors="coerce").sum()
        amount_source = "published"
    else:
        amount = float((price * volume).sum())
        amount_source = "derived_price_times_volume"

    return {
        "events": int(len(frame)),
        "open": float(price.iloc[0]),
        "high": float(price.max()),
        "low": float(price.min()),
        "close": float(price.iloc[-1]),
        "volume": float(volume.sum()),
        "amount": float(amount),
        "amount_source": amount_source,
        "excluded_outside_session": outside,
        "excluded_phase_counts": {str(k): int(v) for k, v in excluded_counts.items()},
        "first_event_time": str(frame["exchange_time"].iloc[0])
        if "exchange_time" in frame.columns else None,
        "last_event_time": str(frame["exchange_time"].iloc[-1])
        if "exchange_time" in frame.columns else None,
    }


def _close_enough(derived: float, panel: float, *, rtol: float, atol: float = 0.0) -> bool:
    if panel is None or derived is None:
        return False
    if not np.isfinite(derived) or not np.isfinite(panel):
        return False
    return bool(abs(derived - panel) <= max(atol, rtol * abs(panel)))


def _aggregation_only_shortfall(
    mismatched: Sequence[str], detail: Mapping[str, Any]
) -> bool:
    """True when the only disagreements are extremes aggregation cannot see.

    Requires more than "the mismatched fields are high/low": the derived range
    must sit *inside* the panel range. A derived high above the panel high, or a
    derived low below the panel low, is a real error -- aggregation can only
    lose extremes, never invent them.
    """
    if not mismatched or not set(mismatched).issubset(AGGREGATION_BLIND_FIELDS):
        return False
    derived = detail.get("derived", {})
    panel = detail.get("panel", {})
    if "high" in mismatched:
        if not (derived.get("high", float("inf")) < panel.get("high", float("-inf"))):
            return False
    if "low" in mismatched:
        if not (derived.get("low", float("-inf")) > panel.get("low", float("inf"))):
            return False
    return True


def reconcile_symbol_day(
    trades: pd.DataFrame,
    panel_row: Mapping[str, Any] | None,
    *,
    data_class: str | None = None,
) -> SymbolDayReconciliation:
    """Compare one symbol-day of trade events against its daily panel row.

    ``data_class`` lets an aggregated feed be scored on what it can actually
    reproduce. It never loosens a tolerance -- it only distinguishes "this feed
    disagrees with the day" from "this feed agrees with the day everywhere its
    resolution permits".
    """
    symbol = str(trades["symbol"].iloc[0]) if not trades.empty else ""
    trade_date = str(trades["trade_date"].iloc[0]) if not trades.empty else ""

    derived = daily_bar_from_trades(trades)
    if derived.get("events", 0) == 0:
        return SymbolDayReconciliation(
            symbol, trade_date, NO_TICK_DATA, 0, 0, 0, [], derived
        )
    if panel_row is None:
        return SymbolDayReconciliation(
            symbol, trade_date, NO_PANEL_ROW, derived["events"], 0, 0, [],
            {"derived": derived,
             "note": "no U0 daily row for this symbol-day; the tick day is unverified"},
        )

    comparisons = (
        ("open", PRICE_RTOL, 0.0),
        ("high", PRICE_RTOL, 0.0),
        ("low", PRICE_RTOL, 0.0),
        ("close", PRICE_RTOL, 0.0),
        ("volume", VOLUME_RTOL, VOLUME_ATOL),
        ("amount", AMOUNT_RTOL, 0.0),
    )
    mismatched: list[str] = []
    detail: dict[str, Any] = {"derived": derived, "panel": {}, "delta": {}}
    compared = 0
    for field, rtol, atol in comparisons:
        panel_value = panel_row.get(field)
        if panel_value is None or (isinstance(panel_value, float) and not np.isfinite(panel_value)):
            continue
        if pd.isna(panel_value):
            continue
        compared += 1
        panel_value = float(panel_value)
        derived_value = float(derived[field])
        detail["panel"][field] = panel_value
        detail["delta"][field] = derived_value - panel_value
        if not _close_enough(derived_value, panel_value, rtol=rtol, atol=atol):
            mismatched.append(field)

    if not mismatched:
        status = MATCH
    elif (
        data_class == contracts.SNAPSHOT_DERIVED_TRADE_AGGREGATE
        and _aggregation_only_shortfall(mismatched, detail)
    ):
        status = MATCH_WITHIN_AGGREGATION_LIMITS
        detail["aggregation_note"] = (
            "every field the feed can reproduce matches; the disagreement is "
            f"confined to {sorted(mismatched)}, which a snapshot-differenced "
            "aggregate cannot observe because the printed price of a bucket is "
            "the snapshot's last price, not the bucket's extreme"
        )
    else:
        status = MISMATCH
    return SymbolDayReconciliation(
        symbol, trade_date, status, derived["events"], compared,
        compared - len(mismatched), mismatched, detail,
    )


def reconcile_days(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    panel_symbol_column: str = "symbol",
    panel_date_column: str = "trade_date",
) -> dict[str, Any]:
    """Reconcile every symbol-day present in ``trades`` against ``panel``.

    ``panel`` is the U0 raw daily panel (or any slice of it) carrying
    ``open/high/low/close/volume/amount``.
    """
    if trades.empty:
        return {
            "symbol_days": 0, "distinct_symbols": 0, "status_counts": {},
            "results": [], "summary": "no tick events supplied",
        }

    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not panel.empty:
        keyed = panel.copy()
        keyed[panel_date_column] = pd.to_datetime(
            keyed[panel_date_column], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        for record in keyed.to_dict("records"):
            index[(str(record[panel_symbol_column]), str(record[panel_date_column]))] = record

    normalised = trades.copy()
    normalised["trade_date"] = pd.to_datetime(
        normalised["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    results: list[SymbolDayReconciliation] = []
    for (symbol, trade_date), group in normalised.groupby(["symbol", "trade_date"], sort=True):
        classes = (
            set(group["data_class"].dropna().astype(str))
            if "data_class" in group.columns else set()
        )
        results.append(reconcile_symbol_day(
            group,
            index.get((str(symbol), str(trade_date))),
            data_class=classes.pop() if len(classes) == 1 else None,
        ))

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    matched = (
        status_counts.get(MATCH, 0) + status_counts.get(MATCH_WITHIN_AGGREGATION_LIMITS, 0)
    )
    verified = matched + status_counts.get(MISMATCH, 0)
    return {
        "symbol_days": len(results),
        "distinct_symbols": int(normalised["symbol"].nunique()),
        "date_range": [
            str(normalised["trade_date"].min()),
            str(normalised["trade_date"].max()),
        ],
        "status_counts": status_counts,
        "match_rate_over_verifiable": (matched / verified) if verified else None,
        "unverifiable_symbol_days": status_counts.get(NO_PANEL_ROW, 0),
        "results": [r.to_dict() for r in results],
    }
