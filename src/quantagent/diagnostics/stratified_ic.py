"""Stratified factor IC validation.

The current v7 / v9 / v10 pipeline computes one rank-IC by ``trade_date``
across the entire investable universe. That single number can hide large
dispersion across stock pools — Stage 1 of the v4 strategy spec requires
factor effectiveness to be measured per pool because different pools have
materially different alpha dynamics.

This module computes IC stratified along four axes that we *can* compute
from existing data without any new providers:

* **board** — derived from the A-share symbol prefix (沪主板 / 深主板 /
  创业板 / 科创板 / 北交所 / other). Pure-stringop, no extra data.
* **liquidity tier** — quintile of trailing-20d average dollar amount
  (``amount_mean_20d`` from ``market_features.parquet``). Acts as a proxy
  for size when true market-cap data is not available.
* **volatility tier** — quintile of trailing-20d realised vol
  (``volatility_20d``). Splits low-vol blue chips from high-vol names.
* **regime** — the labelled benchmark regime state (``normal`` /
  ``caution`` / ``bear`` / ``crisis``) computed by
  :func:`quantagent.training.v7_experiment._compute_regime_frame`.

For each (axis, bucket, horizon) we report:

* number of trading days that have non-empty buckets
* number of unique symbols ever in the bucket
* mean rank-IC, IC std, ICIR (mean / std)
* mean ann-return of a long-only top-K (within the bucket) for context

The module is callable from a CLI (``scripts/stratified_ic_report.py``)
but the heavy lifting is in functions you can unit-test. Functions never
re-train models or touch GPU; they only read parquet files that already
exist on disk.

See :mod:`tests.diagnostics.test_stratified_ic` for usage examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_BOARD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("688",), "STAR_科创"),
    (("300", "301"), "ChiNext_创业"),
    (("600", "601", "603", "605"), "SH_Main_沪主板"),
    (("000", "001", "002", "003"), "SZ_Main_深主板"),
    (("4", "8"), "BSE_北交所"),  # single-char prefix; checked last
)


def board_of(symbol: str) -> str:
    """Classify an A-share symbol into a board by code prefix.

    Examples
    --------
    >>> board_of("600519.SH")
    'SH_Main_沪主板'
    >>> board_of("300750.SZ")
    'ChiNext_创业'
    >>> board_of("688981.SH")
    'STAR_科创'
    >>> board_of("832000.BJ")
    'BSE_北交所'
    >>> board_of("XYZ")
    'OTHER'
    """
    if not isinstance(symbol, str) or not symbol:
        return "OTHER"
    code = symbol.split(".")[0]
    if not code:
        return "OTHER"
    for prefixes, label in _BOARD_RULES:
        for prefix in prefixes:
            if code.startswith(prefix):
                return label
    return "OTHER"


def cap_bucket_of(amount_proxy: float, quintile_edges: tuple[float, ...]) -> str:
    """Bucket a per-day amount value into Q1..Q5 size buckets.

    ``quintile_edges`` is a 4-tuple of (q20, q40, q60, q80). Each
    threshold separates the buckets in ascending order: < q20 → Q1
    (smallest cap proxy), … , ≥ q80 → Q5 (largest).
    """
    if not isinstance(quintile_edges, tuple) or len(quintile_edges) != 4:
        raise ValueError("quintile_edges must be a 4-tuple (q20, q40, q60, q80)")
    q20, q40, q60, q80 = quintile_edges
    if amount_proxy is None or (isinstance(amount_proxy, float) and (np.isnan(amount_proxy) or np.isinf(amount_proxy))):
        return "UNKNOWN"
    v = float(amount_proxy)
    if v < q20:
        return "Q1_smallest"
    if v < q40:
        return "Q2"
    if v < q60:
        return "Q3"
    if v < q80:
        return "Q4"
    return "Q5_largest"


@dataclass(frozen=True)
class StratifiedICConfig:
    """Knobs for the stratified IC analysis.

    ``min_symbols_per_date`` — bucket must have ≥ this many symbols on a
    given trade_date for that date's IC to be counted. Below the
    threshold the rank correlation is too noisy.

    ``min_days_per_bucket`` — bucket must contribute ≥ this many dates
    overall to be reported. Filters out buckets with one or two
    accidental matches.
    """

    min_symbols_per_date: int = 5
    min_days_per_bucket: int = 10
    top_k_for_ann_return: int = 30
    cost_bps_per_round_trip: float = 12.0


@dataclass(frozen=True)
class StratifiedICResult:
    by_axis: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


def _rank_ic(group: pd.DataFrame, label_col: str, min_n: int) -> float:
    if len(group) < min_n:
        return float("nan")
    p = pd.to_numeric(group["prediction"], errors="coerce")
    y = pd.to_numeric(group[label_col], errors="coerce")
    mask = p.notna() & y.notna()
    if mask.sum() < min_n:
        return float("nan")
    return float(p[mask].rank().corr(y[mask].rank()))


def _topk_long_only_return(group: pd.DataFrame, label_col: str, top_k: int) -> float:
    if len(group) < top_k:
        return float("nan")
    sub = group.dropna(subset=["prediction", label_col])
    if len(sub) < top_k:
        return float("nan")
    chosen = sub.nlargest(top_k, "prediction")
    return float(chosen[label_col].mean())


def _bucket_ic_table(
    predictions: pd.DataFrame,
    label_col: str,
    bucket_col: str,
    horizon: int,
    config: StratifiedICConfig,
) -> pd.DataFrame:
    """One row per (bucket) summarising IC + top-K return."""
    out_rows: list[dict[str, object]] = []
    for bucket, grp in predictions.groupby(bucket_col):
        if not isinstance(bucket, str) or not bucket:
            continue
        # per-date IC inside the bucket
        per_date = grp.groupby("trade_date", group_keys=False).apply(
            lambda g: _rank_ic(g, label_col, config.min_symbols_per_date)
        )
        per_date = per_date.dropna()
        per_date_topk = grp.groupby("trade_date", group_keys=False).apply(
            lambda g: _topk_long_only_return(g, label_col, config.top_k_for_ann_return)
        ).dropna()
        if len(per_date) < config.min_days_per_bucket:
            continue
        ic_mean = float(per_date.mean())
        ic_std = float(per_date.std(ddof=1)) if len(per_date) > 1 else float("nan")
        icir = ic_mean / (ic_std + 1e-12) if ic_std and ic_std == ic_std else float("nan")
        topk_avg = float(per_date_topk.mean()) if not per_date_topk.empty else float("nan")
        # annualised return assuming the H-day label and 252 trading days
        # (H-day cumulative → divide by H → daily → compound 252)
        if pd.notna(topk_avg):
            daily = topk_avg / max(horizon, 1)
            cost_daily = (config.cost_bps_per_round_trip / 10_000.0) / max(horizon, 1)
            ann_return = (1.0 + daily - cost_daily) ** 252 - 1.0
        else:
            ann_return = float("nan")
        out_rows.append(
            {
                "bucket": str(bucket),
                "horizon": int(horizon),
                "n_dates": int(len(per_date)),
                "n_symbols": int(grp["symbol"].nunique()),
                "ic_mean": round(ic_mean, 5),
                "ic_std": round(ic_std, 5) if ic_std == ic_std else None,
                "ic_ir": round(icir, 5) if icir == icir else None,
                "topk_avg_h_return": round(topk_avg, 5) if topk_avg == topk_avg else None,
                "topk_ann_return": round(ann_return, 5) if ann_return == ann_return else None,
            }
        )
    if not out_rows:
        return pd.DataFrame(columns=["bucket", "horizon", "n_dates", "n_symbols", "ic_mean", "ic_std", "ic_ir", "topk_avg_h_return", "topk_ann_return"])
    return pd.DataFrame(out_rows).sort_values(["horizon", "bucket"]).reset_index(drop=True)


def _attach_board(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["board"] = predictions["symbol"].astype(str).map(board_of)
    return predictions


def _attach_liquidity_bucket(
    predictions: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a per-(date, symbol) liquidity quintile bucket.

    Uses cross-sectional quintiles per trade_date — so the buckets are
    'this stock relative to today's universe', not 'this stock relative
    to all-time'.
    """
    if "amount_mean_20d" not in market_features.columns:
        out = predictions.copy()
        out["liq_bucket"] = "UNKNOWN"
        return out
    mf = market_features[["trade_date", "symbol", "amount_mean_20d"]].copy()
    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors="coerce")
    merged = predictions.copy()
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged.merge(mf, on=["trade_date", "symbol"], how="left")
    # Per-date quintile edges
    def _bucket(grp: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(grp["amount_mean_20d"], errors="coerce")
        valid = values.dropna()
        if len(valid) < 25:  # too few for quintiles
            return pd.Series(["UNKNOWN"] * len(grp), index=grp.index)
        edges = tuple(np.quantile(valid, [0.20, 0.40, 0.60, 0.80]).tolist())
        return values.map(lambda v: cap_bucket_of(v, edges))
    merged["liq_bucket"] = merged.groupby("trade_date", group_keys=False).apply(_bucket)
    return merged


def _attach_volatility_bucket(
    predictions: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    if "volatility_20d" not in market_features.columns:
        out = predictions.copy()
        out["vol_bucket"] = "UNKNOWN"
        return out
    mf = market_features[["trade_date", "symbol", "volatility_20d"]].copy()
    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors="coerce")
    merged = predictions.copy()
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged.merge(mf, on=["trade_date", "symbol"], how="left")

    def _bucket(grp: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(grp["volatility_20d"], errors="coerce")
        valid = values.dropna()
        if len(valid) < 25:
            return pd.Series(["UNKNOWN"] * len(grp), index=grp.index)
        edges = tuple(np.quantile(valid, [0.20, 0.40, 0.60, 0.80]).tolist())
        return values.map(lambda v: cap_bucket_of(v, edges))
    merged["vol_bucket"] = merged.groupby("trade_date", group_keys=False).apply(_bucket)
    # Re-label so "Q1" reads as "lowest vol" not "lowest cap"
    rename = {"Q1_smallest": "Q1_lowest_vol", "Q5_largest": "Q5_highest_vol"}
    merged["vol_bucket"] = merged["vol_bucket"].map(lambda x: rename.get(x, x))
    return merged


def _attach_regime(
    predictions: pd.DataFrame,
    regime_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach regime label per trade_date using the benchmark regime frame."""
    out = predictions.copy()
    if regime_frame is None or regime_frame.empty:
        out["regime"] = "UNKNOWN"
        return out
    rf = regime_frame[["trade_date", "regime_state"]].copy()
    rf["trade_date"] = pd.to_datetime(rf["trade_date"], errors="coerce")
    rf = rf.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.merge(rf.rename(columns={"regime_state": "regime"}), on="trade_date", how="left")
    out["regime"] = out["regime"].fillna("UNKNOWN")
    return out


def _attach_sector(
    predictions: pd.DataFrame,
    sector_map: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach sector buckets with an as-of join.

    The join is deliberately PIT-safe: a sector row is visible to a
    prediction row only when ``sector_map.available_at <= trade_date``.
    Current snapshots fetched in 2026 therefore do not backfill 2020 OOS
    predictions.
    """
    out = predictions.copy()
    out["sector_level_1"] = "UNKNOWN"
    out["sector_level_2"] = "UNKNOWN"
    if sector_map is None or sector_map.empty:
        return out
    required = {"symbol", "available_at", "sector_level_1"}
    if not required.issubset(sector_map.columns):
        return out
    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    sm["available_at"] = pd.to_datetime(sm["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    sm = sm.dropna(subset=["symbol", "available_at"])
    sm = sm[sm.get("coverage_status", "pit_historical").astype(str) != "missing"]
    if "sector_level_2" not in sm.columns:
        sm["sector_level_2"] = sm["sector_level_1"]
    sm = sm[["symbol", "available_at", "sector_level_1", "sector_level_2"]].sort_values(["available_at", "symbol"])
    if sm.empty:
        return out
    left = out.drop(columns=["sector_level_1", "sector_level_2"]).copy()
    left["symbol"] = left["symbol"].astype(str)
    left["trade_date"] = pd.to_datetime(left["trade_date"], errors="coerce")
    left = left.sort_values(["trade_date", "symbol"])
    merged = pd.merge_asof(
        left,
        sm,
        left_on="trade_date",
        right_on="available_at",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["sector_level_1"] = merged["sector_level_1"].fillna("UNKNOWN").astype(str)
    merged["sector_level_2"] = merged["sector_level_2"].fillna("UNKNOWN").astype(str)
    return merged.drop(columns=["available_at"], errors="ignore")


def compute_stratified_ic(
    predictions: pd.DataFrame,
    market_features: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
    sector_map: pd.DataFrame | None = None,
    config: StratifiedICConfig | None = None,
) -> StratifiedICResult:
    """Compute stratified IC tables for an OOS predictions panel.

    Parameters
    ----------
    predictions:
        Long frame with ``trade_date``, ``symbol``, ``horizon``,
        ``prediction``, and at least one ``forward_return_{H}d`` column.
    market_features:
        Optional features panel — needed for ``liq_bucket`` and
        ``vol_bucket`` axes. If omitted, only ``board`` and ``regime``
        axes are produced.
    regime_frame:
        Optional benchmark regime label frame (output of
        ``_compute_regime_frame``). Without it the ``regime`` axis is
        labelled ``UNKNOWN``.
    sector_map:
        Optional ``silver/sector_map/sector_map.parquet`` frame. Joined
        by ``symbol`` with ``available_at <= trade_date``; never used for
        stock selection here.
    """

    config = config or StratifiedICConfig()
    if predictions is None or predictions.empty:
        return StratifiedICResult(by_axis={}, summary={"status": "empty_input"})
    if "horizon" not in predictions.columns or "prediction" not in predictions.columns:
        return StratifiedICResult(by_axis={}, summary={"status": "missing_required_columns"})

    enriched = _attach_board(predictions)
    if market_features is not None and not market_features.empty:
        enriched = _attach_liquidity_bucket(enriched, market_features)
        enriched = _attach_volatility_bucket(enriched, market_features)
    else:
        enriched["liq_bucket"] = "UNKNOWN"
        enriched["vol_bucket"] = "UNKNOWN"
    enriched = _attach_regime(enriched, regime_frame)
    enriched = _attach_sector(enriched, sector_map)

    by_axis: dict[str, pd.DataFrame] = {}
    axis_columns = {
        "board": "board",
        "liquidity_quintile": "liq_bucket",
        "volatility_quintile": "vol_bucket",
        "regime": "regime",
        "sector_level_1": "sector_level_1",
        "sector_level_2": "sector_level_2",
    }

    for axis_name, axis_col in axis_columns.items():
        rows_per_horizon: list[pd.DataFrame] = []
        for horizon, grp in enriched.groupby("horizon"):
            label = f"forward_return_{int(horizon)}d"
            if label not in grp.columns:
                continue
            table = _bucket_ic_table(grp, label, axis_col, int(horizon), config)
            if not table.empty:
                rows_per_horizon.append(table)
        if rows_per_horizon:
            by_axis[axis_name] = pd.concat(rows_per_horizon, ignore_index=True)

    summary = {
        "status": "passed",
        "n_predictions_rows": int(len(predictions)),
        "n_unique_symbols": int(predictions["symbol"].nunique()) if "symbol" in predictions.columns else 0,
        "horizons_present": sorted(int(h) for h in predictions["horizon"].dropna().unique()),
        "axes_computed": list(by_axis.keys()),
        "config": {
            "min_symbols_per_date": int(config.min_symbols_per_date),
            "min_days_per_bucket": int(config.min_days_per_bucket),
            "top_k_for_ann_return": int(config.top_k_for_ann_return),
        },
    }
    return StratifiedICResult(by_axis=by_axis, summary=summary)


def render_markdown(result: StratifiedICResult) -> str:
    """Render the IC tables as a single markdown report string."""

    if not result.by_axis:
        return "# Stratified IC Report\n\n(No data — empty input)\n"

    parts: list[str] = ["# Stratified Factor IC Report\n"]
    summary = result.summary
    parts.append("## Summary\n")
    parts.append(f"- Rows: {summary.get('n_predictions_rows', 0):,}")
    parts.append(f"- Unique symbols: {summary.get('n_unique_symbols', 0):,}")
    parts.append(f"- Horizons: {summary.get('horizons_present')}")
    parts.append(f"- Axes computed: {summary.get('axes_computed')}\n")

    for axis, table in result.by_axis.items():
        parts.append(f"## Axis: `{axis}`\n")
        if table.empty:
            parts.append("(no buckets met the minimum-symbols-per-date / minimum-days threshold)\n")
            continue
        # Pivot to (horizon × bucket) for readability of IC
        ic_pivot = (
            table.pivot(index="horizon", columns="bucket", values="ic_mean")
            .round(4)
            .fillna("—")
        )
        ann_pivot = (
            table.pivot(index="horizon", columns="bucket", values="topk_ann_return")
            .round(4)
            .fillna("—")
        )
        parts.append("**IC (mean):**\n")
        parts.append(ic_pivot.to_markdown())
        parts.append("\n\n**Top-30 ann. return (within bucket):**\n")
        parts.append(ann_pivot.to_markdown())
        parts.append("\n")

    return "\n".join(parts)


def write_report(
    result: StratifiedICResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write JSON tables + markdown report + per-axis CSVs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    json_path = output_dir / "stratified_ic.json"
    payload = {
        "summary": result.summary,
        "tables": {axis: table.to_dict("records") for axis, table in result.by_axis.items()},
    }
    import json
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["json"] = json_path

    md_path = output_dir / "stratified_ic.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    paths["markdown"] = md_path

    for axis, table in result.by_axis.items():
        csv_path = output_dir / f"by_{axis}.csv"
        table.to_csv(csv_path, index=False)
        paths[f"csv_{axis}"] = csv_path

    return paths


__all__ = [
    "StratifiedICConfig",
    "StratifiedICResult",
    "compute_stratified_ic",
    "board_of",
    "cap_bucket_of",
    "render_markdown",
    "write_report",
]
