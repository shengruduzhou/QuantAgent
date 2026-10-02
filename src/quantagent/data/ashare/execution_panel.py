"""Certified A-share EXECUTION panel: what a strict simulator may trade and mark.

The training dataset and the execution panel answer different questions, and the
Round 29 audit found the canonical evaluator using one file for both:

* the training dataset is adjusted (one scale for returns), keeps only traded
  sessions, and drops rows whose t+1 entry was infeasible;
* a simulator needs the **raw traded price** (lot rounding, minimum commission,
  limit prices and participation caps are computed on it), dated tradability
  flags, and a row for **every in-life session** -- a held name that did not
  trade still has to be valued, and the strict simulator treats a missing bar
  for a held name as fatal (R1-F01).

So this module publishes, next to the training dataset, one row per in-life
security-session:

``gap_classification``
    ``TRADED`` for a vendor bar; otherwise the U0 ``session_gaps`` class
    (``SUSPENDED`` / ``MISSING_UNEXPLAINED`` / ``PROVIDER_HISTORY_TRUNCATED``).
    A gap row carries the last raw close (re-based across any ex-rights date
    inside the gap) for valuation, ``volume = amount = 0`` and
    ``is_suspended = True``: it can be valued, never traded. Most gaps are
    MISSING_UNEXPLAINED, so "untradeable at a stale price" is an *assumption*
    the classification column keeps visible, not a measurement.

``is_st / is_limit_up / is_limit_down`` (+ tri-state ``*_status``)
    From the gold masks. ``is_st`` is TRUE only where the dated register says
    so (UNKNOWN stays visible in ``st_status``). The limit flags are
    fail-closed: an UNKNOWN limit state blocks the trade in that direction.

``ca_cash_per_share / ca_share_ratio / ca_basis``
    Corporate actions on the session they take effect. A raw-price panel
    without them books every dividend and bonus issue as a loss (a 10-for-10
    bonus halves the raw price). The credits reproduce the backward-adjusted
    (hfq) total return the labels assume: when the U0 dividend record explains
    the factor step it is used as-is (cash + bonus/transfer shares); otherwise
    the step is credited as its cash value (``factor_value_equivalent``).

Provenance: ``serving_provider`` per row and a single provider per symbol --
:func:`verify_execution_panel` refuses a panel that stitches vendors without a
:class:`~quantagent.data.ashare.contracts.SourceBoundary`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from quantagent.data.ashare import contracts
from quantagent.data.ashare.gold_bridge import (
    MASK_FALSE,
    MASK_TRUE,
    MASK_UNKNOWN,
    RegisterCoverage,
    _interval_mask,
)

EXECUTION_PANEL_SCHEMA = "ashare_execution_panel_v1"
GAP_TRADED = "TRADED"
GAP_CLASSES: tuple[str, ...] = (
    "SUSPENDED", "MISSING_UNEXPLAINED", "PROVIDER_HISTORY_TRUNCATED",
)
CA_BASIS_NONE = ""
CA_BASIS_CORPORATE_ACTION = "corporate_action"
CA_BASIS_FACTOR_VALUE = "factor_value_equivalent"
#: Relative tolerance for "the dividend record explains the factor step".
CA_MATCH_TOLERANCE = 0.005

#: Columns the strict simulator reads plus the evidence a reader needs.
EXECUTION_PANEL_COLUMNS: tuple[str, ...] = (
    "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
    "amount_measured", "available_at", "adjustment_method", "adjust_factor",
    "is_suspended", "is_st", "is_limit_up", "is_limit_down",
    "suspension_status", "st_status", "limit_up_status", "limit_down_status",
    "gap_classification", "ca_cash_per_share", "ca_share_ratio", "ca_basis",
    "serving_provider",
)

#: Without these a raw-price panel cannot be marked through an ex-rights date.
CORPORATE_ACTION_COLUMNS: tuple[str, ...] = ("ca_cash_per_share", "ca_share_ratio")


class ExecutionPanelError(RuntimeError):
    """Raised when a panel cannot back a strict A-share simulation."""


def _factor_asof(frame: pd.DataFrame, factors: pd.DataFrame) -> np.ndarray:
    """hfq factor in force on each row's trade date (1.0 before the first step)."""
    if factors is None or factors.empty or frame.empty:
        return np.ones(len(frame))
    left = pd.DataFrame({
        "_row": np.arange(len(frame)),
        "symbol": frame["symbol"].astype(str).to_numpy(),
        "trade_date": pd.to_datetime(frame["trade_date"]).to_numpy(),
    }).sort_values("trade_date", kind="mergesort")
    right = (
        factors[["symbol", "effective_date", "hfq_factor"]]
        .dropna(subset=["effective_date"])
        .assign(symbol=lambda f: f["symbol"].astype(str),
                effective_date=lambda f: pd.to_datetime(f["effective_date"]))
        .sort_values("effective_date", kind="mergesort")
    )
    merged = pd.merge_asof(
        left, right, left_on="trade_date", right_on="effective_date",
        by="symbol", direction="backward",
    )
    out = np.ones(len(frame))
    out[merged["_row"].to_numpy()] = merged["hfq_factor"].fillna(1.0).to_numpy()
    return out


def _status_flag(status: pd.Series, *, fail_closed: bool) -> np.ndarray:
    values = status.astype(str).to_numpy()
    if fail_closed:
        return values != MASK_FALSE
    return values == MASK_TRUE


def build_execution_panel(
    traded: pd.DataFrame,
    *,
    session_gaps: pd.DataFrame | None,
    factors: pd.DataFrame | None,
    corporate_actions: pd.DataFrame | None = None,
    st: pd.DataFrame | None = None,
    st_coverage: RegisterCoverage = False,
    start: Any = None,
    end: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the execution panel from raw traded bars + U0 gap/CA tables.

    ``traded`` carries one row per vendor bar with **raw** ``open/high/low/
    close``, ``volume``, ``amount``, ``available_at``, ``serving_provider`` and
    the gold masks ``mask_is_suspended / mask_is_st / mask_limit_up /
    mask_limit_down`` (see :func:`gold_bridge.build_masks`). It may include a
    look-back before ``start`` so the first in-window gap can carry a close;
    rows outside ``[start, end]`` are trimmed at the end.
    """
    required = {"symbol", "trade_date", "open", "high", "low", "close", "volume",
                "amount", "mask_is_suspended", "mask_is_st", "mask_limit_up",
                "mask_limit_down"}
    missing = sorted(required - set(traded.columns))
    if missing:
        raise ExecutionPanelError(f"traded bars are missing {missing}")
    if "adjustment_method" in traded.columns:
        methods = set(traded["adjustment_method"].dropna().astype(str))
        if methods - {contracts.ADJUST_NONE}:
            raise ExecutionPanelError(
                f"execution panel needs raw traded prices; got adjustment {sorted(methods)}")

    stats: dict[str, Any] = {}
    bars = traded.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["gap_classification"] = GAP_TRADED
    bars["suspension_status"] = bars["mask_is_suspended"].astype(str)
    bars["st_status"] = bars["mask_is_st"].astype(str)
    bars["limit_up_status"] = bars["mask_limit_up"].astype(str)
    bars["limit_down_status"] = bars["mask_limit_down"].astype(str)
    if "available_at" not in bars.columns:
        bars["available_at"] = bars["trade_date"] + pd.Timedelta(hours=15)
    if "serving_provider" not in bars.columns:
        bars["serving_provider"] = bars.get("source", "unknown")

    # ---- explicit rows for in-life sessions without a bar -------------------
    gaps = (session_gaps if session_gaps is not None else pd.DataFrame(
        columns=["symbol", "trade_date", "classification"])).copy()
    gaps["trade_date"] = pd.to_datetime(gaps["trade_date"]).dt.normalize()
    lo = bars["trade_date"].min()
    hi = bars["trade_date"].max() if end is None else pd.Timestamp(end)
    gaps = gaps[gaps["symbol"].isin(set(bars["symbol"].unique()))
                & (gaps["trade_date"] >= lo) & (gaps["trade_date"] <= hi)]
    gaps = gaps.drop_duplicates(["symbol", "trade_date"])
    overlap = gaps.merge(bars[["symbol", "trade_date"]], on=["symbol", "trade_date"])
    if len(overlap):
        raise ExecutionPanelError(
            f"{len(overlap)} session-gap rows collide with traded bars; the U0 gap "
            "table and the panel disagree about which sessions traded")
    unknown_classes = sorted(set(gaps["classification"].astype(str)) - set(GAP_CLASSES))
    if unknown_classes:
        raise ExecutionPanelError(f"unrecognised gap classifications {unknown_classes}")
    gap_rows = pd.DataFrame({
        "symbol": gaps["symbol"].astype(str).to_numpy(),
        "trade_date": gaps["trade_date"].to_numpy(),
        "gap_classification": gaps["classification"].astype(str).to_numpy(),
    })
    if len(gap_rows):
        gap_rows["st_status"] = _interval_mask(
            gap_rows, st if st is not None else pd.DataFrame(), available=st_coverage
        ).to_numpy()
        gap_rows["suspension_status"] = np.where(
            gap_rows["gap_classification"] == "SUSPENDED", MASK_TRUE, MASK_UNKNOWN)
        gap_rows["limit_up_status"] = "NOT_TRADED"
        gap_rows["limit_down_status"] = "NOT_TRADED"
        gap_rows["volume"] = 0.0
        gap_rows["amount"] = 0.0
        gap_rows["available_at"] = gap_rows["trade_date"] + pd.Timedelta(hours=15)
        provider = bars.drop_duplicates("symbol").set_index("symbol")["serving_provider"]
        gap_rows["serving_provider"] = gap_rows["symbol"].map(provider)

    keep_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                 "amount", "available_at", "serving_provider", "gap_classification",
                 "suspension_status", "st_status", "limit_up_status", "limit_down_status"]
    panel = pd.concat([bars[keep_cols], gap_rows.reindex(columns=keep_cols)],
                      ignore_index=True)
    panel = panel.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    del bars, gap_rows

    # ---- carried close and corporate actions on the hfq factor clock --------
    factor = _factor_asof(panel, factors if factors is not None else pd.DataFrame())
    panel["adjust_factor"] = factor
    symbols = panel["symbol"].to_numpy()
    starts = np.r_[True, symbols[1:] != symbols[:-1]]
    group = np.cumsum(starts) - 1
    adjusted_close = panel["close"].to_numpy(dtype=float) * factor
    adjusted_filled = pd.Series(adjusted_close).groupby(group).ffill().to_numpy()
    is_gap = panel["gap_classification"].to_numpy() != GAP_TRADED
    carried = adjusted_filled / factor
    no_prior = is_gap & ~np.isfinite(carried)
    stats["gap_rows_without_prior_close_excluded"] = {
        str(k): int(v) for k, v in
        pd.Series(panel["gap_classification"].to_numpy()[no_prior]).value_counts().items()
    }
    close = panel["close"].to_numpy(dtype=float).copy()
    close[is_gap] = np.round(carried[is_gap], 2)
    panel["close"] = close
    for column in ("open", "high", "low"):
        values = panel[column].to_numpy(dtype=float).copy()
        values[is_gap] = close[is_gap]
        panel[column] = values

    prev_factor = np.r_[np.nan, factor[:-1]]
    prev_close = np.r_[np.nan, close[:-1]]
    prev_date = np.r_[np.datetime64("NaT"), panel["trade_date"].to_numpy()[:-1]]
    prev_factor[starts] = np.nan
    prev_close[starts] = np.nan
    prev_date[starts] = np.datetime64("NaT")
    step = np.isfinite(prev_factor) & (np.abs(factor / prev_factor - 1.0) > 1e-9)
    ca_cash = np.zeros(len(panel))
    ca_share = np.zeros(len(panel))
    ca_basis = np.full(len(panel), CA_BASIS_NONE, dtype=object)

    step_index = np.flatnonzero(step & np.isfinite(prev_close))
    matched = np.zeros(len(step_index), dtype=bool)
    if len(step_index):
        factor_ratio = factor[step_index] / prev_factor[step_index]
        reference = prev_close[step_index] / factor_ratio
        if corporate_actions is not None and not corporate_actions.empty:
            ca = corporate_actions.copy()
            ca["ex_date"] = pd.to_datetime(ca["ex_date"]).dt.normalize()
            ca["share_ratio"] = (
                pd.to_numeric(ca.get("stock_dividend_ratio", 0.0), errors="coerce").fillna(0.0)
                # U0's `rights_ratio` is the 转增 (capital-reserve transfer)
                # column of the Sina 分红送配 table (sources.py), i.e. free
                # shares, not a paid rights issue.
                + pd.to_numeric(ca.get("rights_ratio", 0.0), errors="coerce").fillna(0.0)
            )
            ca["cash"] = pd.to_numeric(
                ca.get("cash_dividend_per_share", 0.0), errors="coerce").fillna(0.0)
            ca = (ca.groupby(["symbol", "ex_date"], as_index=False)[["cash", "share_ratio"]]
                  .sum().sort_values("ex_date", kind="mergesort"))
            left = pd.DataFrame({
                "_i": np.arange(len(step_index)),
                "symbol": symbols[step_index].astype(str),
                "trade_date": panel["trade_date"].to_numpy()[step_index],
            }).sort_values("trade_date", kind="mergesort")
            hit = pd.merge_asof(left, ca, left_on="trade_date", right_on="ex_date",
                                by="symbol", direction="backward")
            hit = hit.sort_values("_i")
            ex_date = hit["ex_date"].to_numpy()
            in_gap = ~pd.isna(ex_date) & (ex_date > prev_date[step_index])
            cash = np.where(in_gap, hit["cash"].to_numpy(dtype=float), 0.0)
            share = np.where(in_gap, hit["share_ratio"].to_numpy(dtype=float), 0.0)
            denominator = prev_close[step_index] - cash
            with np.errstate(divide="ignore", invalid="ignore"):
                implied = (1.0 + share) * prev_close[step_index] / denominator
            matched = in_gap & (denominator > 0) & (
                np.abs(implied / factor_ratio - 1.0) <= CA_MATCH_TOLERANCE)
            ca_cash[step_index[matched]] = cash[matched]
            ca_share[step_index[matched]] = share[matched]
            ca_basis[step_index[matched]] = CA_BASIS_CORPORATE_ACTION
        fallback = step_index[~matched]
        ca_cash[fallback] = prev_close[fallback] - reference[~matched]
        ca_basis[fallback] = CA_BASIS_FACTOR_VALUE
    panel["ca_cash_per_share"] = ca_cash
    panel["ca_share_ratio"] = ca_share
    panel["ca_basis"] = ca_basis
    stats["factor_steps"] = int(len(step_index))
    stats["factor_steps_explained_by_dividend_record"] = int(matched.sum())
    stats["factor_steps_credited_at_value"] = int(len(step_index) - matched.sum())
    stats["factor_steps_without_previous_close"] = int((step & ~np.isfinite(prev_close)).sum())

    # ---- flags ---------------------------------------------------------------
    traded_row = ~is_gap
    panel["is_suspended"] = is_gap | (panel["suspension_status"].astype(str) == MASK_TRUE)
    panel["is_st"] = panel["st_status"].astype(str) == MASK_TRUE
    panel["is_limit_up"] = traded_row & _status_flag(panel["limit_up_status"], fail_closed=True)
    panel["is_limit_down"] = traded_row & _status_flag(panel["limit_down_status"], fail_closed=True)
    amount = pd.to_numeric(panel["amount"], errors="coerce")
    panel["amount_measured"] = amount.notna().to_numpy()
    panel["adjustment_method"] = contracts.ADJUST_NONE

    panel = panel[~no_prior]
    if start is not None:
        panel = panel[panel["trade_date"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["trade_date"] <= pd.Timestamp(end)]
    panel = panel.reset_index(drop=True)[list(EXECUTION_PANEL_COLUMNS)]

    stats.update({
        "schema": EXECUTION_PANEL_SCHEMA,
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "date_range": [str(panel["trade_date"].min())[:10], str(panel["trade_date"].max())[:10]]
        if len(panel) else [None, None],
        "rows_by_gap_classification": {
            str(k): int(v) for k, v in panel["gap_classification"].value_counts().items()},
        "is_suspended_rows": int(panel["is_suspended"].sum()),
        "is_st_rows": int(panel["is_st"].sum()),
        "st_status": {str(k): int(v) for k, v in panel["st_status"].value_counts().items()},
        "limit_up_status": {
            str(k): int(v) for k, v in panel["limit_up_status"].value_counts().items()},
        "limit_down_status": {
            str(k): int(v) for k, v in panel["limit_down_status"].value_counts().items()},
        "amount_unmeasured_rows": int((~panel["amount_measured"]).sum()),
        "amount_unmeasured_symbols": int(
            panel.loc[~panel["amount_measured"], "symbol"].nunique()),
        "ca_basis_rows": {
            str(k): int(v) for k, v in panel["ca_basis"].value_counts().items() if str(k)},
        "flag_policy": {
            "is_limit_up/is_limit_down": "TRUE or UNKNOWN status blocks that side (fail-closed)",
            "is_st": "TRUE only where the dated register says so; UNKNOWN kept in st_status",
            "gap rows": "valued at the carried raw close; volume=amount=0; never tradeable",
        },
    })
    return panel, stats


def verify_execution_panel(
    panel: pd.DataFrame,
    *,
    source_boundaries: pd.DataFrame | Iterable[contracts.SourceBoundary] | None = None,
    require_corporate_actions: bool = True,
) -> dict[str, Any]:
    """Refuse a panel the strict simulator must not be pointed at.

    * every row declares ``adjustment_method == 'none'`` (raw traded prices);
    * one ``serving_provider`` per symbol, unless each provider switch is
      recorded as a SourceBoundary (symbol + boundary date);
    * raw prices come with corporate-action credits, otherwise every ex-rights
      date is booked as a loss.
    """
    if "adjustment_method" not in panel.columns:
        raise ExecutionPanelError(
            "execution panel does not declare adjustment_method; refusing to assume "
            "the prices are raw (the legacy v7 panel was qfq levels with raw volume)")
    methods = sorted(set(panel["adjustment_method"].astype(str).unique()))
    if methods != [contracts.ADJUST_NONE]:
        raise ExecutionPanelError(
            f"execution panel must be raw traded prices (adjustment_method 'none'); "
            f"found {methods}")
    if require_corporate_actions:
        missing = [c for c in CORPORATE_ACTION_COLUMNS if c not in panel.columns]
        if missing:
            raise ExecutionPanelError(
                f"raw execution panel lacks corporate-action credits {missing}; every "
                "dividend/bonus ex-date would be booked as a loss")
    provider_column = next(
        (c for c in ("serving_provider", "source") if c in panel.columns), None)
    if provider_column is None:
        raise ExecutionPanelError(
            "execution panel carries no serving_provider/source column; single-source "
            "per symbol cannot be verified")
    providers = panel.groupby("symbol")[provider_column].nunique(dropna=False)
    stitched = sorted(providers[providers > 1].index.astype(str))
    if stitched:
        declared: set[str] = set()
        if source_boundaries is not None:
            if isinstance(source_boundaries, pd.DataFrame):
                declared = set(source_boundaries["symbol"].astype(str))
            else:
                declared = {str(b.symbol) for b in source_boundaries}
        undeclared = [s for s in stitched if s not in declared]
        if undeclared:
            raise ExecutionPanelError(
                f"{len(undeclared)} symbol(s) mix providers without a SourceBoundary "
                f"(e.g. {undeclared[:5]}); one vendor per symbol history")
    return {
        "adjustment_method": contracts.ADJUST_NONE,
        "single_source_symbols": int((providers <= 1).sum()),
        "boundary_declared_symbols": int(len(stitched)),
        "corporate_action_columns": [c for c in CORPORATE_ACTION_COLUMNS if c in panel.columns],
    }


def execution_panel_summary(stats: Mapping[str, Any]) -> str:
    """One line for logs."""
    return (
        f"rows={stats.get('rows'):,} symbols={stats.get('symbols'):,} "
        f"gaps={ {k: v for k, v in stats.get('rows_by_gap_classification', {}).items() if k != GAP_TRADED} } "
        f"ca={stats.get('ca_basis_rows')}"
    )


__all__ = [
    "CA_BASIS_CORPORATE_ACTION", "CA_BASIS_FACTOR_VALUE", "CORPORATE_ACTION_COLUMNS",
    "EXECUTION_PANEL_COLUMNS", "EXECUTION_PANEL_SCHEMA", "ExecutionPanelError",
    "GAP_CLASSES", "GAP_TRADED", "build_execution_panel", "execution_panel_summary",
    "verify_execution_panel",
]
