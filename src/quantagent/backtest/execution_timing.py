"""Trace-proven execution timing for production-grade A-share backtests.

The executable-label dataset defines a signal at session T and an entry at the
next market session close. This module makes that contract machine-verifiable
instead of relying on a narrative ``t_plus_one=True`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import StringIO

import pandas as pd


EXECUTION_TIMING_SEMANTICS = "signal_t_close_next_session_close_v1"
TRACE_SCHEMA_VERSION = "execution_trace_v1"
_REQUIRED_COLUMNS = (
    "record_type",
    "signal_date",
    "execution_date",
    "status",
    "reason",
    "symbol",
    "client_order_id",
    "price_source",
    "reference_price",
    "execution_timing_semantics",
)


@dataclass(frozen=True)
class ExecutionTraceValidation:
    ok: bool
    reasons: tuple[str, ...]
    mapped_signal_days: int
    order_records: int
    skip_records: int
    terminal_censored_signal_days: int = 0


def validate_execution_trace(trace: pd.DataFrame) -> ExecutionTraceValidation:
    """Validate signal->execution mapping without trusting summary flags.

    Exactly one special non-executed schedule is permitted: the chronologically
    final signal may be ``unmapped`` with reason ``no_next_market_session``.
    This is right-censoring, not execution: it cannot contribute an order, PnL
    or return. All other missing mappings remain hard failures.
    """
    reasons: list[str] = []
    if trace is None or trace.empty:
        return ExecutionTraceValidation(False, ("execution_trace_empty",), 0, 0, 0, 0)
    missing = [column for column in _REQUIRED_COLUMNS if column not in trace.columns]
    if missing:
        return ExecutionTraceValidation(
            False,
            tuple(f"execution_trace_missing_column:{column}" for column in missing),
            0,
            0,
            0,
            0,
        )

    frame = trace.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame["execution_date"] = pd.to_datetime(frame["execution_date"], errors="coerce")
    if frame["signal_date"].isna().any():
        reasons.append("execution_trace_invalid_signal_date")
    if not frame["execution_timing_semantics"].eq(EXECUTION_TIMING_SEMANTICS).all():
        reasons.append("execution_trace_semantics_mismatch")

    schedules = frame[frame["record_type"] == "session_mapping"].copy()
    terminal_count = 0
    if schedules.empty:
        reasons.append("execution_trace_missing_session_mapping")
    if schedules["signal_date"].duplicated().any():
        reasons.append("execution_trace_duplicate_signal_mapping")
    if not schedules.empty:
        latest_signal = schedules["signal_date"].max()
        terminal_mask = (
            schedules["status"].eq("unmapped")
            & schedules["reason"].eq("no_next_market_session")
            & schedules["execution_date"].isna()
            & schedules["signal_date"].eq(latest_signal)
        )
        terminal_count = int(terminal_mask.sum())
        if terminal_count > 1:
            reasons.append("execution_trace_multiple_terminal_censored_signals")
        accepted_schedule = schedules["status"].eq("mapped") | terminal_mask
        if not accepted_schedule.all():
            bad = sorted(
                set(
                    str(value)
                    for value in schedules.loc[~accepted_schedule, "reason"]
                )
            )
            reasons.extend(f"execution_trace_unmapped_signal:{value}" for value in bad)
        mapped = schedules[schedules["status"].eq("mapped")].dropna(
            subset=["signal_date", "execution_date"]
        )
        if not mapped.empty and not (mapped["execution_date"] > mapped["signal_date"]).all():
            reasons.append("execution_trace_same_or_prior_session_execution")
        malformed_mapped = schedules[
            schedules["status"].eq("mapped") & schedules["execution_date"].isna()
        ]
        if not malformed_mapped.empty:
            reasons.append("execution_trace_mapped_signal_missing_execution_date")
        if not schedules["price_source"].eq("close").all():
            reasons.append("execution_trace_schedule_price_source_not_close")

    mapping = {
        pd.Timestamp(row.signal_date): pd.Timestamp(row.execution_date)
        for row in schedules.itertuples(index=False)
        if pd.notna(row.signal_date)
        and pd.notna(row.execution_date)
        and str(row.status) == "mapped"
    }
    detail = frame[frame["record_type"].isin(["order", "skip", "execution_error"])].copy()
    if not detail.empty:
        if detail["execution_date"].isna().any():
            reasons.append("execution_trace_detail_missing_execution_date")
        for row in detail.itertuples(index=False):
            signal = pd.Timestamp(row.signal_date) if pd.notna(row.signal_date) else None
            execution = pd.Timestamp(row.execution_date) if pd.notna(row.execution_date) else None
            if signal is None or execution is None or mapping.get(signal) != execution:
                reasons.append("execution_trace_detail_mapping_mismatch")
                break
        priced = detail[detail["record_type"] == "order"]
        if not priced.empty and not priced["price_source"].eq("close").all():
            reasons.append("execution_trace_order_price_source_not_close")
        errors = detail[detail["record_type"] == "execution_error"]
        if not errors.empty:
            for value in sorted(set(str(v) for v in errors["reason"] if str(v))):
                reasons.append(f"execution_trace_error:{value}")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return ExecutionTraceValidation(
        ok=not unique_reasons,
        reasons=unique_reasons,
        mapped_signal_days=int(len(schedules[schedules["status"] == "mapped"])),
        order_records=int((frame["record_type"] == "order").sum()),
        skip_records=int((frame["record_type"] == "skip").sum()),
        terminal_censored_signal_days=terminal_count,
    )


def execution_trace_sha256(trace: pd.DataFrame) -> str:
    """Return a stable SHA-256 over the machine-readable trace contents."""
    if trace is None:
        raise ValueError("execution_trace is required")
    frame = trace.copy()
    ordered = [column for column in _REQUIRED_COLUMNS if column in frame.columns]
    extras = sorted(column for column in frame.columns if column not in ordered)
    frame = frame[ordered + extras]
    for column in ("signal_date", "execution_date"):
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            frame[column] = parsed.dt.strftime("%Y-%m-%d")
            frame.loc[parsed.isna(), column] = ""
    sort_columns = [
        column
        for column in (
            "signal_date",
            "execution_date",
            "record_type",
            "symbol",
            "client_order_id",
            "reason",
        )
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort", na_position="last")
    buffer = StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.12g")
    return sha256(buffer.getvalue().encode("utf-8")).hexdigest()


__all__ = [
    "EXECUTION_TIMING_SEMANTICS",
    "TRACE_SCHEMA_VERSION",
    "ExecutionTraceValidation",
    "execution_trace_sha256",
    "validate_execution_trace",
]
