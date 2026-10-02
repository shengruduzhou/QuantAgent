"""Read-only unit / adjustment audit of a daily market panel.

Answers, per ``source`` and per board, the questions a panel's own labels cannot:

* **Unit scale** -- does ``amount / volume`` sit inside ``[low, high]`` (the
  panel's own implied VWAP)? At x1, and diagnostically at x100 / x0.01.
* **Against U0 raw** (when a certified raw reference is supplied) -- median and
  modal ratios of volume, amount and close; a lots-vs-shares defect shows up as
  a volume ratio of exactly 0.01 or 100.
* **Adjustment basis** -- the per-(symbol, source) ratio ``close / close_u0``.
  Raw prices give 1 everywhere. Forward adjustment fetched on some later date
  gives a ratio below 1 that is constant between ex-dates and *steps at
  ex-dates* (multiplicative, Sina) or a constant difference that steps at
  ex-dates (subtractive, EastMoney/Tencent). Steps are matched to the U0
  adjust-factor ``effective_date`` list.
* **PIT claim** -- a panel whose prices are qfq-as-of-fetch cannot be
  point-in-time: the history was rewritten on the fetch date. Rows stamped
  ``point_in_time_valid=True`` with a non-raw basis are counted as refuted and
  the panel-level verdict is ``REFUTED`` whenever any exist.

Nothing here writes to the audited panel or modifies a row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.data.ashare.symbols import SymbolError, identify
from quantagent.data.ashare.units import implied_vwap_in_band

BASIS_RAW = "raw"
BASIS_QFQ_MULT = "qfq_multiplicative"
BASIS_QFQ_SUB = "qfq_subtractive"
BASIS_HFQ = "hfq_like"
BASIS_CONSTANT = "constant_scaled"
BASIS_INCONSISTENT = "inconsistent"
BASIS_NO_REFERENCE = "no_reference_overlap"
NON_RAW_BASES = frozenset({BASIS_QFQ_MULT, BASIS_QFQ_SUB, BASIS_HFQ, BASIS_CONSTANT,
                           BASIS_INCONSISTENT})


@dataclass(frozen=True)
class AuditTolerances:
    close_match: float = 0.005      # |close/close_u0 - 1| for a "matching" close
    ratio_unit: float = 0.01        # |ratio - target| / target for 1 / 0.01 / 100 classes
    raw_share: float = 0.99         # share of matching closes to call a group raw
    step_log: float = 1e-3          # |dlog(ratio)| that counts as a price-basis step
    step_abs: float = 0.011         # |d(close_u0 - close)| in CNY that counts as a step
    stable_share: float = 0.95      # share of non-step pairs to call a basis piecewise-constant
    ex_date_alignment: float = 0.8  # share of steps on ex-dates => adjustment, not noise


def _board(symbols: pd.Series) -> pd.Series:
    mapping: dict[str, str] = {}
    for symbol in symbols.dropna().unique():
        try:
            mapping[symbol] = identify(str(symbol)).board
        except SymbolError:
            mapping[symbol] = "UNKNOWN"
    return symbols.map(mapping).fillna("UNKNOWN")


def _share_near(values: pd.Series, target: float, rel: float) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(((values - target).abs() <= rel * target).mean())


def _num(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 6)


def classify_adjustment_basis(
    joined: pd.DataFrame,
    ex_dates: pd.DataFrame | None,
    tol: AuditTolerances = AuditTolerances(),
) -> pd.DataFrame:
    """One row per (symbol, source): inferred price basis vs the raw reference.

    ``joined`` needs symbol, source, trade_date, close, close_ref.
    ``ex_dates`` needs symbol, effective_date (U0 adjust-factor change dates).
    """
    frame = joined[["symbol", "source", "trade_date", "close", "close_ref"]].dropna()
    frame = frame[(frame["close"] > 0) & (frame["close_ref"] > 0)]
    frame = frame.sort_values(["symbol", "source", "trade_date"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "source", "basis"])
    frame["ratio"] = frame["close"] / frame["close_ref"]
    frame["diff"] = frame["close_ref"] - frame["close"]
    frame["match"] = (frame["ratio"] - 1).abs() <= tol.close_match
    key = [frame["symbol"], frame["source"]]
    same_group = (frame["symbol"].eq(frame["symbol"].shift())
                  & frame["source"].eq(frame["source"].shift()))
    dlog = np.log(frame["ratio"]).diff().where(same_group)
    ddiff = frame["diff"].diff().where(same_group)
    # Vendor-adjusted prices are rounded to the 0.01 tick, so on a 4-CNY stock
    # the ratio jitters by ~0.25% from rounding alone. A step must exceed that
    # rounding noise (one tick relative to the smaller adjusted close).
    rounding = 0.01 / pd.concat([frame["close"], frame["close"].shift()], axis=1).min(axis=1)
    frame["mult_step"] = dlog.abs() > (tol.step_log + rounding)
    frame["sub_step"] = ddiff.abs() > tol.step_abs
    frame["pair"] = same_group

    # Did an ex-date fall in (previous session, this session]?
    frame["on_ex_date"] = False
    if ex_dates is not None and not ex_dates.empty:
        ex = ex_dates[["symbol", "effective_date"]].dropna()
        ex = ex.assign(effective_date=pd.to_datetime(ex["effective_date"]).dt.normalize())
        by_symbol = {s: np.sort(g["effective_date"].to_numpy()) for s, g in ex.groupby("symbol")}
        prev_date = frame["trade_date"].shift().where(same_group)
        flags = np.zeros(len(frame), dtype=bool)
        for symbol, idx in frame.groupby("symbol").indices.items():
            dates = by_symbol.get(symbol)
            if dates is None or len(dates) == 0:
                continue
            cur = frame["trade_date"].to_numpy()[idx]
            prev = prev_date.to_numpy()[idx]
            hi = np.searchsorted(dates, cur, side="right")
            valid = ~pd.isna(prev)
            lo = np.zeros(len(idx), dtype=int)
            lo[valid] = np.searchsorted(dates, prev[valid].astype(cur.dtype), side="right")
            flags[idx] = valid & (hi > lo)
        frame["on_ex_date"] = flags

    grouped = frame.groupby(key, sort=False)
    stats = pd.DataFrame({
        "rows": grouped.size(),
        "match_share": grouped["match"].mean(),
        "median_ratio": grouped["ratio"].median(),
        "last_ratio": grouped["ratio"].last(),
        "pairs": grouped["pair"].sum(),
        "mult_steps": grouped["mult_step"].sum(),
        "sub_steps": grouped["sub_step"].sum(),
        "mult_steps_on_ex": grouped.apply(
            lambda g: int((g["mult_step"] & g["on_ex_date"]).sum()), include_groups=False),
        "sub_steps_on_ex": grouped.apply(
            lambda g: int((g["sub_step"] & g["on_ex_date"]).sum()), include_groups=False),
        "ex_dates_in_window": grouped["on_ex_date"].sum(),
    })
    stats.index.names = ["symbol", "source"]
    stats = stats.reset_index()
    pairs = stats["pairs"].clip(lower=1)
    mult_stable = 1 - stats["mult_steps"] / pairs
    sub_stable = 1 - stats["sub_steps"] / pairs

    def _basis(row_index: int) -> str:
        row = stats.iloc[row_index]
        if row["match_share"] >= tol.raw_share:
            return BASIS_RAW
        ms, ss = mult_stable.iloc[row_index], sub_stable.iloc[row_index]
        if ms < tol.stable_share and ss < tol.stable_share:
            return BASIS_INCONSISTENT
        if row["median_ratio"] > 1 + tol.close_match:
            return BASIS_HFQ
        # Multiplicative if the ratio is the more stable statistic, else
        # subtractive; with no step at all the basis is a constant rescale
        # (still not raw, but no ex-date evidence inside the window).
        if ms >= ss:
            return BASIS_CONSTANT if row["mult_steps"] == 0 else BASIS_QFQ_MULT
        return BASIS_CONSTANT if row["sub_steps"] == 0 else BASIS_QFQ_SUB

    stats["basis"] = [_basis(i) for i in range(len(stats))]
    steps = np.where(stats["basis"].eq(BASIS_QFQ_SUB), stats["sub_steps"], stats["mult_steps"])
    on_ex = np.where(stats["basis"].eq(BASIS_QFQ_SUB), stats["sub_steps_on_ex"],
                     stats["mult_steps_on_ex"])
    stats["steps"] = steps
    stats["steps_on_ex_dates"] = on_ex
    stats["steps_on_ex_share"] = np.where(steps > 0, on_ex / np.maximum(steps, 1), np.nan)
    stats["qfq_as_of_fetch"] = (
        stats["basis"].isin([BASIS_QFQ_MULT, BASIS_QFQ_SUB])
        & (stats["steps_on_ex_share"] >= tol.ex_date_alignment)
    )
    return stats


def audit_market_panel(
    panel: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    ex_dates: pd.DataFrame | None = None,
    tol: AuditTolerances = AuditTolerances(),
) -> dict[str, object]:
    """Audit ``panel`` (symbol, trade_date, OHLC, volume, amount, source, ...).

    ``reference`` is a certified RAW panel (shares / CNY), e.g. U0
    ``daily_bars_raw.parquet``. Without it only the self-consistency (VWAP)
    checks run and the adjustment basis is reported as unverifiable.
    """
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel["source"] = panel["source"].astype("string").fillna("<missing>")
    panel["board"] = _board(panel["symbol"])
    for scale, column in ((1.0, "vwap_x1"), (100.0, "vwap_x100"), (0.01, "vwap_x0_01")):
        panel[column] = implied_vwap_in_band(
            panel["volume"] * scale, panel["amount"], panel["low"], panel["high"]
        )
    pit = (
        panel["point_in_time_valid"].fillna(False).astype(bool)
        if "point_in_time_valid" in panel.columns
        else pd.Series(False, index=panel.index)
    )
    panel["pit_stamped"] = pit

    has_reference = reference is not None and not reference.empty
    if has_reference:
        ref = reference[["symbol", "trade_date", "close", "volume", "amount"]].copy()
        ref["trade_date"] = pd.to_datetime(ref["trade_date"]).dt.normalize()
        ref = ref.rename(columns={"close": "close_ref", "volume": "volume_ref",
                                  "amount": "amount_ref"})
        joined = panel.merge(ref, on=["symbol", "trade_date"], how="left")
        joined["has_ref"] = joined["close_ref"].notna()
        joined["close_ratio"] = joined["close"] / joined["close_ref"]
        joined["volume_ratio"] = joined["volume"] / joined["volume_ref"].where(
            joined["volume_ref"] > 0)
        joined["amount_ratio"] = joined["amount"] / joined["amount_ref"].where(
            joined["amount_ref"] > 0)
        basis = classify_adjustment_basis(joined[joined["has_ref"]], ex_dates, tol)
        joined = joined.merge(basis[["symbol", "source", "basis", "qfq_as_of_fetch"]],
                              on=["symbol", "source"], how="left")
        joined["basis"] = joined["basis"].fillna(BASIS_NO_REFERENCE)
        joined["qfq_as_of_fetch"] = joined["qfq_as_of_fetch"].fillna(False).astype(bool)
    else:
        joined = panel.assign(has_ref=False, basis=BASIS_NO_REFERENCE, qfq_as_of_fetch=False,
                              close_ratio=np.nan, volume_ratio=np.nan, amount_ratio=np.nan)
        basis = pd.DataFrame(columns=["symbol", "source", "basis"])

    def _group_report(g: pd.DataFrame) -> dict[str, object]:
        ref_rows = g[g["has_ref"]]
        out: dict[str, object] = {
            "rows": int(len(g)),
            "symbols": int(g["symbol"].nunique()),
            "vwap_testable_rows": int(g["vwap_x1"].notna().sum()),
            "vwap_in_range_rate_x1": _num(g["vwap_x1"].mean()),
            "vwap_in_range_rate_if_x100": _num(g["vwap_x100"].mean()),
            "vwap_in_range_rate_if_x0_01": _num(g["vwap_x0_01"].mean()),
            "rows_with_u0_reference": int(len(ref_rows)),
            "pit_stamped_rows": int(g["pit_stamped"].sum()),
        }
        if len(ref_rows):
            out.update({
                "close_match_rate_vs_u0": _num(
                    ((ref_rows["close_ratio"] - 1).abs() <= tol.close_match).mean()),
                "median_close_ratio_vs_u0": _num(ref_rows["close_ratio"].median()),
                "median_volume_ratio_vs_u0": _num(ref_rows["volume_ratio"].median()),
                "volume_ratio_share_near_1": _num(_share_near(ref_rows["volume_ratio"], 1.0, tol.ratio_unit)),
                "volume_ratio_share_near_0_01": _num(_share_near(ref_rows["volume_ratio"], 0.01, tol.ratio_unit)),
                "volume_ratio_share_near_100": _num(_share_near(ref_rows["volume_ratio"], 100.0, tol.ratio_unit)),
                "median_amount_ratio_vs_u0": _num(ref_rows["amount_ratio"].median()),
                "amount_ratio_share_near_1": _num(_share_near(ref_rows["amount_ratio"], 1.0, tol.ratio_unit)),
                "rows_by_adjustment_basis": {
                    str(k): int(v) for k, v in ref_rows["basis"].value_counts().items()},
                "rows_qfq_as_of_fetch": int(ref_rows["qfq_as_of_fetch"].sum()),
                "pit_stamped_rows_refuted": int(
                    (ref_rows["pit_stamped"] & ref_rows["basis"].isin(NON_RAW_BASES)).sum()),
            })
        return out

    by_source = {str(src): _group_report(g) for src, g in joined.groupby("source", sort=True)}
    by_source_board = {
        f"{src}|{board}": _group_report(g)
        for (src, board), g in joined.groupby(["source", "board"], sort=True)
    }
    basis_counts = (
        basis.groupby(["source", "basis"]).size().unstack(fill_value=0)
        if not basis.empty else pd.DataFrame()
    )
    stamped = int(joined["pit_stamped"].sum())
    refuted = int((joined["pit_stamped"] & joined["basis"].isin(NON_RAW_BASES)).sum())
    qfq_rows = int(joined["qfq_as_of_fetch"].sum())
    if not has_reference:
        verdict = "UNVERIFIABLE_NO_RAW_REFERENCE"
    elif qfq_rows or refuted:
        verdict = "REFUTED"
    elif stamped:
        verdict = "NOT_REFUTED"
    else:
        verdict = "NO_PIT_CLAIM"
    return {
        "rows": int(len(joined)),
        "symbols": int(joined["symbol"].nunique()),
        "date_range": [str(joined["trade_date"].min().date()), str(joined["trade_date"].max().date())]
        if len(joined) else [None, None],
        "rows_with_u0_reference": int(joined["has_ref"].sum()),
        "point_in_time_claim": {
            "rows_stamped_point_in_time_valid": stamped,
            "rows_stamped_but_non_raw_prices": refuted,
            "rows_qfq_as_of_fetch": qfq_rows,
            "verdict": verdict,
            "rule": "qfq-as-of-fetch prices (ratio to U0 raw steps at U0 ex-dates) were "
                    "rewritten on the fetch date and cannot be point-in-time",
        },
        "adjustment_basis_symbol_counts_by_source": {
            str(src): {str(b): int(n) for b, n in row.items() if n}
            for src, row in basis_counts.iterrows()
        },
        "by_source": by_source,
        "by_source_board": by_source_board,
        "tolerances": tol.__dict__,
    }
