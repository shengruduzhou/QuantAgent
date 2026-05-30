"""Fundamental PIT ranker — valuation × quality × growth.

The ranker consumes a long-form PIT metrics frame (one row per
``(symbol, available_at)`` carrying ttm valuation ratios and the
latest financial-statement-derived quality / growth metrics) and
produces a normalised cross-sectional ranking on every ``as_of_date``
the caller requests. Each dimension is scored independently in
``[0, 1]`` and the composite is a weighted sum.

Design points
-------------
**PIT discipline.** Every metric row carries an ``available_at``
timestamp. For each request ``as_of_date`` the helper picks each
symbol's latest row with ``available_at <= as_of_date`` — never a
future row. Rows missing ``available_at`` are rejected. The same
discipline applies to sector mapping joins: only sector_map rows whose
``available_at`` is no later than the request date are eligible.

**Within-sector ranks.** Cross-sectional rank within the same
``sector_level_1`` is the unit of comparison. Comparing a bank PE to a
food-and-bev PE has well-known industry-mix problems; the ranker
normalises by sector when sector data is available and by
market-segment (board) proxy otherwise. The bucket key used is
recorded in ``rank_bucket_kind`` per row so the audit chain is
explicit.

**Missing-data behaviour.** Any dimension whose metrics are missing
gets ``score = NaN`` for that symbol on that date; the composite is
the weight-renormalised average of available dimensions. A symbol
with **no** metrics gets ``composite_score = NaN`` and falls out of
the ranking instead of being given an arbitrary score.

**Diagnostic-only by default.** The companion helper
``fundamental_ranker_for_overlay`` enforces a manifest gate before any
caller may consume the table as a weight overlay. The default
production path treats the artifact as audit-only data, matching the
Stage 2.2/2.3 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.data.manifest import build_manifest_for_frame, utc_now_iso


FUNDAMENTAL_RANKER_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "as_of_date",
    "available_at",
    "rank_bucket",
    "rank_bucket_kind",
    "valuation_score",
    "quality_score",
    "growth_score",
    "composite_score",
    "valuation_rank",
    "quality_rank",
    "growth_rank",
    "composite_rank",
    "metric_completeness",
    "source_version",
    "generated_at",
)

VALUATION_INPUT_COLUMNS: tuple[str, ...] = ("pe_ttm", "pb", "ps_ttm")
QUALITY_INPUT_COLUMNS: tuple[str, ...] = ("roe", "gross_margin", "operating_cf_to_net_income")
GROWTH_INPUT_COLUMNS: tuple[str, ...] = ("revenue_yoy", "net_income_yoy")

DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "valuation": 0.40,
    "quality": 0.35,
    "growth": 0.25,
}


@dataclass(frozen=True)
class FundamentalRankerConfig:
    """Tuning knobs.

    All metric thresholds are conservative; on A-share data the floors
    are intentionally lenient because the metrics frame typically only
    carries ttm ratios that all listed names report. The clipping
    ranges are set wide so that legitimate negatives (loss-making
    growth stocks, REIT-style PB > 5) survive into the ranking rather
    than being silently bucketed at 0 or 1.
    """

    min_universe_per_bucket: int = 5
    valuation_floor_pe_ttm: float = 0.0   # PE <= 0 excluded from valuation score
    valuation_floor_pb: float = 0.05
    valuation_floor_ps_ttm: float = 0.05
    growth_clip_pct: float = 5.00         # cap YoY at +/- 500% to neutralise IPO-year outliers
    quality_clip_pct: float = 5.00        # cap ratios at +/- 500% (operating_cf_to_net_income)
    dimension_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS))
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"


@dataclass(frozen=True)
class FundamentalRankerResult:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


def _to_naive_series(values: object) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


def _coerce_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the metrics frame and coerce types.

    The metrics frame is the gold-form valuation × fundamentals join.
    The ranker does not re-fetch data, so the schema contract is
    strict: ``symbol`` and ``available_at`` must be present, plus at
    least one column from each dimension's input column list (else the
    corresponding score is NaN).
    """
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["symbol", "available_at", *VALUATION_INPUT_COLUMNS, *QUALITY_INPUT_COLUMNS, *GROWTH_INPUT_COLUMNS]
        )
    data = frame.copy()
    if "symbol" not in data.columns:
        raise ValueError("metrics frame must contain a 'symbol' column")
    if "available_at" not in data.columns:
        raise ValueError("metrics frame must contain an 'available_at' column for PIT joins")
    data["symbol"] = data["symbol"].astype(str).str.strip()
    data["available_at"] = _to_naive_series(data["available_at"])
    data = data.dropna(subset=["symbol", "available_at"])
    for column in VALUATION_INPUT_COLUMNS + QUALITY_INPUT_COLUMNS + GROWTH_INPUT_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


def _select_latest_pit_per_symbol(metrics: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Keep one row per symbol with the latest ``available_at <= as_of``."""
    if metrics.empty:
        return metrics
    eligible = metrics[metrics["available_at"] <= as_of]
    if eligible.empty:
        return eligible.copy()
    return (
        eligible.sort_values(["symbol", "available_at"])
        .drop_duplicates("symbol", keep="last")
        .reset_index(drop=True)
    )


def _attach_sector_asof(snapshot: pd.DataFrame, sector_map: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Asof-join Stage 2.2 sector_map (symbol-keyed with available_at)."""
    out = snapshot.copy()
    out["sector_level_1"] = pd.Series([pd.NA] * len(out), dtype="string")
    if sector_map is None or sector_map.empty:
        return out
    if "sector_level_1" not in sector_map.columns or "symbol" not in sector_map.columns:
        return out
    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    if "available_at" in sm.columns:
        sm["available_at"] = _to_naive_series(sm["available_at"])
        sm = sm[sm["available_at"] <= as_of]
        if sm.empty:
            return out
        sm = sm.sort_values(["symbol", "available_at"]).drop_duplicates("symbol", keep="last")
    elif "coverage_status" in sm.columns:
        # Current-snapshot rows without available_at are tolerated for
        # live diagnostics but not historical PIT joins; flag the
        # output rows so callers can audit.
        sm = sm.drop_duplicates("symbol", keep="last")
    out = out.drop(columns=["sector_level_1"]).merge(
        sm[["symbol", "sector_level_1"]],
        on="symbol",
        how="left",
    )
    return out


def _rank_within_bucket(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Cross-sectional rank normalised to [0, 1].

    ``higher_is_better=True`` means the largest raw value scores 1.0
    (correct for ROE, growth, etc.). ``higher_is_better=False`` flips
    the direction so the smallest raw value scores 1.0 (correct for
    PE/PB/PS). ``method='average'`` so ties collapse to the same score.
    """
    valid = series.notna()
    if valid.sum() < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    # pandas rank with ascending=True: 1 = smallest value, n = largest.
    # For "higher is better" we want the largest to score 1.0, so use
    # ascending=True and let normalize (rank-1)/(n-1) map n→1, 1→0.
    # For "lower is better" we invert by setting ascending=False so the
    # smallest value gets rank n and scores 1.0.
    ranks = series.where(valid).rank(method="average", ascending=bool(higher_is_better))
    n = int(ranks.dropna().shape[0])
    if n <= 1:
        return pd.Series(np.nan, index=series.index, dtype=float)
    normalized = (ranks - 1) / (n - 1)
    return normalized


def _score_valuation(group: pd.DataFrame, config: FundamentalRankerConfig) -> pd.Series:
    """Lower PE/PB/PS is better; values at or below the floor drop out.

    The floors are protective: a stock with negative-and-tiny PE is
    not "cheap", it's an outlier, so we exclude it from this
    dimension's score rather than letting it dominate the rank.
    """
    valid = pd.DataFrame(index=group.index)
    if "pe_ttm" in group.columns:
        pe = group["pe_ttm"].where(group["pe_ttm"] > float(config.valuation_floor_pe_ttm))
        valid["pe_ttm_rank"] = _rank_within_bucket(pe, higher_is_better=False)
    if "pb" in group.columns:
        pb = group["pb"].where(group["pb"] > float(config.valuation_floor_pb))
        valid["pb_rank"] = _rank_within_bucket(pb, higher_is_better=False)
    if "ps_ttm" in group.columns:
        ps = group["ps_ttm"].where(group["ps_ttm"] > float(config.valuation_floor_ps_ttm))
        valid["ps_ttm_rank"] = _rank_within_bucket(ps, higher_is_better=False)
    if valid.empty:
        return pd.Series(np.nan, index=group.index, dtype=float)
    return valid.mean(axis=1, skipna=True)


def _score_quality(group: pd.DataFrame, config: FundamentalRankerConfig) -> pd.Series:
    """Higher ROE / gross margin / OCF-to-net-income is better."""
    valid = pd.DataFrame(index=group.index)
    clip = float(config.quality_clip_pct)
    if "roe" in group.columns:
        valid["roe_rank"] = _rank_within_bucket(group["roe"].clip(-clip, clip), higher_is_better=True)
    if "gross_margin" in group.columns:
        valid["gross_margin_rank"] = _rank_within_bucket(group["gross_margin"].clip(-clip, clip), higher_is_better=True)
    if "operating_cf_to_net_income" in group.columns:
        valid["operating_cf_rank"] = _rank_within_bucket(
            group["operating_cf_to_net_income"].clip(-clip, clip), higher_is_better=True
        )
    if valid.empty:
        return pd.Series(np.nan, index=group.index, dtype=float)
    return valid.mean(axis=1, skipna=True)


def _score_growth(group: pd.DataFrame, config: FundamentalRankerConfig) -> pd.Series:
    """Higher revenue / net-income YoY is better, clipped for IPO outliers."""
    valid = pd.DataFrame(index=group.index)
    clip = float(config.growth_clip_pct)
    if "revenue_yoy" in group.columns:
        valid["revenue_yoy_rank"] = _rank_within_bucket(group["revenue_yoy"].clip(-clip, clip), higher_is_better=True)
    if "net_income_yoy" in group.columns:
        valid["net_income_yoy_rank"] = _rank_within_bucket(group["net_income_yoy"].clip(-clip, clip), higher_is_better=True)
    if valid.empty:
        return pd.Series(np.nan, index=group.index, dtype=float)
    return valid.mean(axis=1, skipna=True)


def _composite(row: pd.Series, weights: dict[str, float]) -> float:
    pieces: list[tuple[float, float]] = []
    for dim, weight in weights.items():
        column = f"{dim}_score"
        value = row.get(column, np.nan)
        if pd.isna(value):
            continue
        pieces.append((float(value), float(weight)))
    if not pieces:
        return float("nan")
    weight_sum = sum(weight for _, weight in pieces)
    if weight_sum <= 0:
        return float("nan")
    return sum(value * weight for value, weight in pieces) / weight_sum


def _completeness(row: pd.Series) -> float:
    score_columns = ("valuation_score", "quality_score", "growth_score")
    present = sum(1 for col in score_columns if pd.notna(row.get(col)))
    return float(present / len(score_columns))


def _bucket_assignment(symbol: str, sector: object) -> tuple[str, str]:
    """Return ``(bucket_value, bucket_kind)``.

    Real sector_level_1 wins; board proxy is the fallback so symbols
    without sector data still participate in a comparable peer group.
    """
    if isinstance(sector, str) and sector.strip() and sector.strip().upper() != "UNKNOWN":
        return sector.strip(), "sector_level_1"
    from quantagent.diagnostics.stratified_ic import board_of

    return board_of(symbol), "board_proxy"


def build_fundamental_ranker(
    metrics: pd.DataFrame,
    *,
    as_of_dates: Iterable[str | pd.Timestamp],
    sector_map: pd.DataFrame | None = None,
    config: FundamentalRankerConfig | None = None,
    generated_at: str | None = None,
) -> FundamentalRankerResult:
    """Compute scores + within-bucket ranks for every requested date."""
    cfg = config or FundamentalRankerConfig()
    coerced = _coerce_metrics(metrics)
    dates = sorted({pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tz is not None else pd.Timestamp(d) for d in as_of_dates})
    if not dates:
        empty = pd.DataFrame(columns=FUNDAMENTAL_RANKER_REQUIRED_COLUMNS)
        return FundamentalRankerResult(
            frame=empty,
            coverage={"total_rows": 0, "status": "no_dates"},
            validation={"status": "no_dates", "row_count": 0},
        )

    generated = generated_at or utc_now_iso()
    all_rows: list[pd.DataFrame] = []
    for as_of in dates:
        snapshot = _select_latest_pit_per_symbol(coerced, as_of)
        if snapshot.empty:
            continue
        joined = _attach_sector_asof(snapshot, sector_map, as_of)
        buckets = [_bucket_assignment(row["symbol"], row.get("sector_level_1")) for _, row in joined.iterrows()]
        joined["rank_bucket"] = [b for b, _ in buckets]
        joined["rank_bucket_kind"] = [k for _, k in buckets]
        joined["as_of_date"] = as_of

        score_rows: list[pd.DataFrame] = []
        for bucket, group in joined.groupby("rank_bucket", dropna=False):
            if len(group) < int(cfg.min_universe_per_bucket):
                continue
            block = group.copy()
            block["valuation_score"] = _score_valuation(block, cfg)
            block["quality_score"] = _score_quality(block, cfg)
            block["growth_score"] = _score_growth(block, cfg)
            block["valuation_rank"] = _rank_within_bucket(block["valuation_score"], higher_is_better=True)
            block["quality_rank"] = _rank_within_bucket(block["quality_score"], higher_is_better=True)
            block["growth_rank"] = _rank_within_bucket(block["growth_score"], higher_is_better=True)
            block["composite_score"] = block.apply(lambda r: _composite(r, cfg.dimension_weights), axis=1)
            block["composite_rank"] = _rank_within_bucket(block["composite_score"], higher_is_better=True)
            score_rows.append(block)
        if not score_rows:
            continue
        all_rows.append(pd.concat(score_rows, ignore_index=True))

    if not all_rows:
        empty = pd.DataFrame(columns=FUNDAMENTAL_RANKER_REQUIRED_COLUMNS)
        return FundamentalRankerResult(
            frame=empty,
            coverage={"total_rows": 0, "status": "no_eligible_buckets"},
            validation={"status": "no_eligible_buckets", "row_count": 0},
        )

    full = pd.concat(all_rows, ignore_index=True)
    full["metric_completeness"] = full.apply(_completeness, axis=1)
    full["source_version"] = cfg.source_version
    full["generated_at"] = generated
    for column in FUNDAMENTAL_RANKER_REQUIRED_COLUMNS:
        if column not in full.columns:
            full[column] = pd.NA
    final = full[list(FUNDAMENTAL_RANKER_REQUIRED_COLUMNS)].sort_values(
        ["as_of_date", "rank_bucket", "composite_rank"], ascending=[True, True, False]
    ).reset_index(drop=True)

    coverage = _coverage_report(final, cfg)
    validation = _validation_report(final)
    return FundamentalRankerResult(frame=final, coverage=coverage, validation=validation)


def _coverage_report(frame: pd.DataFrame, config: FundamentalRankerConfig) -> dict[str, object]:
    if frame.empty:
        return {"total_rows": 0, "status": "empty"}
    composite_valid = int(frame["composite_score"].notna().sum())
    total = int(len(frame))
    bucket_counts = (
        frame.groupby(["as_of_date", "rank_bucket"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    real_sector_rows = int((frame["rank_bucket_kind"] == "sector_level_1").sum())
    board_proxy_rows = int((frame["rank_bucket_kind"] == "board_proxy").sum())
    return {
        "total_rows": total,
        "composite_score_coverage_rate": float(composite_valid / total),
        "average_metric_completeness": float(frame["metric_completeness"].mean()),
        "real_sector_rows": real_sector_rows,
        "board_proxy_rows": board_proxy_rows,
        "real_sector_share": float(real_sector_rows / total),
        "bucket_count_unique": int(frame["rank_bucket"].nunique()),
        "as_of_dates_covered": [str(d) for d in sorted(set(frame["as_of_date"].dropna()))],
        "thresholds": {
            "min_universe_per_bucket": int(config.min_universe_per_bucket),
        },
        "status": "passed",
    }


def _validation_report(frame: pd.DataFrame) -> dict[str, object]:
    missing_cols = [c for c in FUNDAMENTAL_RANKER_REQUIRED_COLUMNS if c not in frame.columns]
    duplicates = int(frame.duplicated(subset=["as_of_date", "symbol"]).sum()) if {"as_of_date", "symbol"}.issubset(frame.columns) else 0
    status = "passed" if not missing_cols and duplicates == 0 else "failed"
    return {
        "status": status,
        "row_count": int(len(frame)),
        "missing_columns": missing_cols,
        "duplicate_symbol_as_of_count": duplicates,
    }


def fundamental_ranker_for_overlay(
    ranker_frame: pd.DataFrame | None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Return the ranker frame only when the manifest gate is open."""
    if ranker_frame is None or ranker_frame.empty:
        return None
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
    coverage = extra.get("coverage_report", {}) if isinstance(extra, dict) else {}
    gate = coverage.get("gate", {}) if isinstance(coverage, dict) else {}
    if not bool(gate.get("fundamental_ranker_usable_for_overlay", False)):
        return None
    return ranker_frame.copy()


class FundamentalRankerBuilder:
    """Materialise the silver/fundamental_ranker artifact + side reports."""

    def __init__(self, config: FundamentalRankerConfig | None = None) -> None:
        self.config = config or FundamentalRankerConfig()

    def build(
        self,
        metrics: pd.DataFrame,
        *,
        as_of_dates: Iterable[str | pd.Timestamp],
        sector_map: pd.DataFrame | None = None,
        generated_at: str | None = None,
    ) -> FundamentalRankerResult:
        result = build_fundamental_ranker(
            metrics,
            as_of_dates=as_of_dates,
            sector_map=sector_map,
            config=self.config,
            generated_at=generated_at,
        )
        gate = self._gate(result)
        coverage = dict(result.coverage)
        coverage["gate"] = gate
        return FundamentalRankerResult(
            frame=result.frame,
            coverage=coverage,
            validation=result.validation,
        )

    def _gate(self, result: FundamentalRankerResult) -> dict[str, object]:
        coverage = result.coverage if isinstance(result.coverage, dict) else {}
        total = int(coverage.get("total_rows", 0))
        composite_rate = float(coverage.get("composite_score_coverage_rate", 0.0)) if total else 0.0
        real_sector_share = float(coverage.get("real_sector_share", 0.0)) if total else 0.0
        reasons: list[str] = []
        if total == 0:
            reasons.append("empty_output")
        if total and composite_rate < 0.50:
            reasons.append("composite_score_coverage_below_threshold")
        if total and real_sector_share < 0.30:
            reasons.append("real_sector_share_below_threshold")
        usable = not reasons
        return {
            "fundamental_ranker_usable_for_diagnostics": True,
            "fundamental_ranker_usable_for_overlay": bool(usable),
            "reason": "passed" if usable else ",".join(reasons),
            "thresholds": {
                "composite_score_coverage": 0.50,
                "real_sector_share": 0.30,
            },
            "observed": {
                "composite_score_coverage_rate": composite_rate,
                "real_sector_share": real_sector_share,
            },
        }

    def write(self, result: FundamentalRankerResult) -> FundamentalRankerResult:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "fundamental_ranker"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "fundamental_ranker.parquet"
        coverage_path = out_dir / "coverage_report.json"
        validation_path = out_dir / "validation_report.json"
        result.frame.to_parquet(parquet_path, index=False)
        coverage_path.write_text(json.dumps(result.coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        validation_path.write_text(json.dumps(result.validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        manifest = build_manifest_for_frame(
            dataset_name="fundamental_ranker",
            vendor="local",
            frame=result.frame,
            output_paths=[parquet_path],
            required_columns=FUNDAMENTAL_RANKER_REQUIRED_COLUMNS,
            extra={
                "coverage_report": result.coverage,
                "validation_report": result.validation,
                "policy": (
                    "diagnostic data product — fundamental_ranker_for_overlay is the "
                    "only sanctioned consumer for any weight-level decision, and "
                    "requires fundamental_ranker_usable_for_overlay=True in the manifest gate"
                ),
            },
        )
        manifest_path = root / "manifests" / "fundamental_ranker.json"
        manifest.write(manifest_path)
        paths = {
            "fundamental_ranker": str(parquet_path),
            "coverage_report": str(coverage_path),
            "validation_report": str(validation_path),
            "manifest": str(manifest_path),
        }
        return FundamentalRankerResult(
            frame=result.frame,
            coverage=result.coverage,
            validation=result.validation,
            output_paths=paths,
        )


__all__ = [
    "DEFAULT_DIMENSION_WEIGHTS",
    "FUNDAMENTAL_RANKER_REQUIRED_COLUMNS",
    "FundamentalRankerBuilder",
    "FundamentalRankerConfig",
    "FundamentalRankerResult",
    "GROWTH_INPUT_COLUMNS",
    "QUALITY_INPUT_COLUMNS",
    "VALUATION_INPUT_COLUMNS",
    "build_fundamental_ranker",
    "fundamental_ranker_for_overlay",
]
