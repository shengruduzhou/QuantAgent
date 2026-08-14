"""Tradability flags: absent evidence must not become a claim of tradability.

Seven call sites independently reimplemented this loop::

    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in panel.columns:
            panel[col] = False

``False`` on these columns is not neutral. It asserts "definitely not suspended,
definitely not at a price limit" -- i.e. *definitely tradeable at any size and
any price*. So a panel that never carried the flags silently backtests as though
every name traded freely every day, which inflates fills, turnover and return.
The same loop's ``.fillna(False)`` branch does it again for individual NaN cells.

The tell is in the very function that motivated this module: two lines below the
flag loop, ``decision_chain._ensure_state_flags`` writes ``panel["amount"] =
np.nan`` when amount is missing -- correctly recording "unknown" -- while the
tradability flags get a definite ``False``. Same function, same missing-input
situation, opposite treatment.

This module keeps the permissive default so existing pipelines still run, but
makes the substitution *visible*: callers get back the list of columns that were
fabricated, and are expected to surface it next to any number the run produces.
``require_measured=True`` fails closed instead, for paths that must not report a
performance figure derived from assumed tradability.
"""

from __future__ import annotations

import pandas as pd

#: The A-share state flags every backtest path expects on its panel.
TRADABILITY_FLAG_COLUMNS: tuple[str, ...] = (
    "is_suspended",
    "is_st",
    "is_limit_up",
    "is_limit_down",
)


class TradabilityEvidenceMissing(ValueError):
    """Raised when tradability flags are absent and the caller demanded them."""


def ensure_tradability_flags(
    panel: pd.DataFrame,
    *,
    columns: tuple[str, ...] = TRADABILITY_FLAG_COLUMNS,
    require_measured: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalise tradability flags and report which ones were not measured.

    Args:
        panel: market panel; not mutated.
        columns: flag columns to normalise.
        require_measured: raise rather than assume tradability.

    Returns:
        ``(panel, unverified)`` where ``unverified`` names every column that was
        absent entirely or contained NaN cells. An empty tuple means every flag
        came from data.

    Raises:
        TradabilityEvidenceMissing: if ``require_measured`` and anything is
            absent or NaN.
    """
    out = panel.copy()
    unverified: list[str] = []

    for column in columns:
        if column not in out.columns:
            unverified.append(column)
            out[column] = False
            continue
        raw = out[column]
        # A NaN cell is an unknown state, not a measured "no".
        if raw.isna().any():
            unverified.append(column)
        out[column] = raw.fillna(False).astype(bool)

    resolved = tuple(dict.fromkeys(unverified))
    if resolved and require_measured:
        raise TradabilityEvidenceMissing(
            "tradability flags were not measured for "
            f"{list(resolved)}; assuming them tradeable would inflate fills and "
            "returns. Supply the flags, or pass require_measured=False and "
            "record the returned `unverified` list alongside any reported number."
        )
    return out, resolved


def tradability_evidence_note(unverified: tuple[str, ...]) -> dict[str, object]:
    """A result-dict fragment recording whether tradability was measured.

    Attach this to any summary/config a human will read. A backtest whose
    tradability was assumed is not comparable with one whose tradability was
    measured, and nothing downstream can tell them apart unless the run says so.
    """
    return {
        "tradability_measured": not unverified,
        "tradability_unverified_columns": list(unverified),
    }


__all__ = [
    "TRADABILITY_FLAG_COLUMNS",
    "TradabilityEvidenceMissing",
    "ensure_tradability_flags",
    "tradability_evidence_note",
]
