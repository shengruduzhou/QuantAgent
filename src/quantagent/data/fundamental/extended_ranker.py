"""ExtendedFundamentalRanker — full 19-axis fundamental scorer (spec section 4).

The existing :mod:`quantagent.data.fundamental.ranker` covers 8 metrics
(``pe_ttm / pb / ps_ttm / roe / gross_margin / operating_cf_to_net_income /
revenue_yoy / net_income_yoy``). The v8 spec mandates 19 axes spanning
valuation, profitability, growth, leverage / health, governance and
capital actions, and quality. This module is a **superset**: it can be
fed the same input frame as the v7 ranker plus new columns, applies
**winsorization + z-standardisation** per axis, and emits a fully
expanded scoring table.

Axes covered (spec mapping in comments):

* PE/PER, PB/PBL, PS                        — valuation
* ROE, ROA, gross_margin, net_margin        — profitability
* revenue_yoy, net_profit_yoy               — growth
* operating_cashflow (OCF / total_revenue)  — earnings quality
* debt_to_asset, interest_coverage          — leverage / solvency
* inventory_turnover                        — operating efficiency
* accounts_receivable_growth                — accounting red flag
* goodwill_risk (goodwill / total_assets)   — balance-sheet risk
* accruals_quality (1 − |accruals|/assets)  — earnings quality
* dividend (dividend_yield)                 — capital returns
* repurchase (buyback_yield)                — capital returns
* earnings_surprise                         — momentum / quality

Per-axis direction (higher_better / lower_better) is enforced inside
:func:`_AXIS_TABLE` so callers do not have to think about sign.

Winsorization: each axis is clipped to its 1st / 99th percentile within
the cross-section before z-standardising. The z-scores are then
mapped to a unit [0, 1] cumulative-normal percentile so all axes can be
compared and aggregated without one outlier dominating.

The result is **diagnostic-only** by default. The companion
:func:`extended_fundamental_for_overlay` enforces the manifest gate
before any caller may consume the table as a weight overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import erf, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "as_of_date",
    "available_at",
    "rank_bucket",
    "rank_bucket_kind",
    "valuation_score",
    "profitability_score",
    "growth_score",
    "leverage_score",
    "efficiency_score",
    "quality_score",
    "capital_action_score",
    "composite_score",
    "composite_rank",
    "metric_completeness",
    "source_version",
    "generated_at",
)


# Each entry: (axis name in metrics frame, group, higher_is_better)
_AXIS_TABLE: tuple[tuple[str, str, bool], ...] = (
    # valuation — lower better
    ("pe_ttm",                       "valuation",       False),
    ("pb",                            "valuation",       False),
    ("ps_ttm",                        "valuation",       False),
    # profitability — higher better
    ("roe",                           "profitability",   True),
    ("roa",                           "profitability",   True),
    ("gross_margin",                  "profitability",   True),
    ("net_margin",                    "profitability",   True),
    # growth — higher better
    ("revenue_yoy",                   "growth",          True),
    ("net_income_yoy",                "growth",          True),
    # quality — higher better
    ("operating_cashflow",            "quality",         True),
    ("accruals_quality",              "quality",         True),
    ("earnings_surprise",             "quality",         True),
    # leverage — lower debt is better, higher coverage is better
    ("debt_to_asset",                 "leverage",        False),
    ("interest_coverage",             "leverage",        True),
    # efficiency
    ("inventory_turnover",            "efficiency",      True),
    ("accounts_receivable_growth",    "efficiency",      False),
    ("goodwill_risk",                 "leverage",        False),
    # capital actions — higher (yields, repurchase) better
    ("dividend",                      "capital_action",  True),
    ("repurchase",                    "capital_action",  True),
)


_DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "valuation":      0.20,
    "profitability":  0.25,
    "growth":         0.15,
    "quality":        0.15,
    "leverage":       0.10,
    "efficiency":     0.05,
    "capital_action": 0.10,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtendedFundamentalConfig:
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    min_universe_per_bucket: int = 5
    group_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_GROUP_WEIGHTS))
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"


@dataclass
class ExtendedFundamentalResult:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — winsorize + standardise
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _winsorize_zscore(values: pd.Series, *, lo: float, hi: float, higher_better: bool) -> pd.Series:
    """Winsorize to [lo, hi] quantiles, z-standardise, then map to [0, 1]
    via the standard-normal CDF. Direction is flipped when
    ``higher_better=False`` so "good" always lands near 1.0.
    """
    s = pd.to_numeric(values, errors="coerce")
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    q_lo = float(valid.quantile(lo))
    q_hi = float(valid.quantile(hi))
    clipped = s.clip(q_lo, q_hi)
    mu = float(clipped.mean(skipna=True))
    sigma = float(clipped.std(ddof=0, skipna=True))
    if sigma < 1e-12:
        # All values identical post-clip → neutral 0.5
        return pd.Series(0.5, index=values.index, dtype=float).where(clipped.notna())
    z = (clipped - mu) / sigma
    if not higher_better:
        z = -z
    return z.apply(lambda v: _normal_cdf(float(v)) if pd.notna(v) else np.nan)


def _select_latest_pit_per_symbol(metrics: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
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
    out = snapshot.copy()
    out["sector_level_1"] = pd.Series([pd.NA] * len(out), dtype="string")
    if sector_map is None or sector_map.empty:
        return out
    if "sector_level_1" not in sector_map.columns or "symbol" not in sector_map.columns:
        return out
    sm = sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    if "available_at" in sm.columns:
        sm["available_at"] = pd.to_datetime(sm["available_at"], errors="coerce")
        sm = sm[sm["available_at"] <= as_of]
        if sm.empty:
            return out
        sm = sm.sort_values(["symbol", "available_at"]).drop_duplicates("symbol", keep="last")
    else:
        sm = sm.drop_duplicates("symbol", keep="last")
    out = out.drop(columns=["sector_level_1"]).merge(
        sm[["symbol", "sector_level_1"]], on="symbol", how="left"
    )
    return out


def _bucket(symbol: str, sector: object) -> tuple[str, str]:
    if isinstance(sector, str) and sector.strip() and sector.strip().upper() != "UNKNOWN":
        return sector.strip(), "sector_level_1"
    from quantagent.diagnostics.stratified_ic import board_of

    return board_of(symbol), "board_proxy"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _score_block(block: pd.DataFrame, config: ExtendedFundamentalConfig) -> pd.DataFrame:
    """Compute per-axis [0, 1] scores + per-group means for one bucket."""
    block = block.copy()
    group_to_axis_cols: dict[str, list[str]] = {}
    for axis, group, higher_better in _AXIS_TABLE:
        if axis not in block.columns:
            continue
        score_col = f"{axis}_score"
        block[score_col] = _winsorize_zscore(
            block[axis],
            lo=config.winsor_lower_q,
            hi=config.winsor_upper_q,
            higher_better=higher_better,
        )
        group_to_axis_cols.setdefault(group, []).append(score_col)
    # Group scores = mean of per-axis scores in that group
    for group, cols in group_to_axis_cols.items():
        block[f"{group}_score"] = block[cols].mean(axis=1, skipna=True)
    # Composite = weighted sum of available group scores (renormalised)
    weight_used = 0.0
    composite_acc = pd.Series(0.0, index=block.index, dtype=float)
    weight_acc = pd.Series(0.0, index=block.index, dtype=float)
    for group, weight in config.group_weights.items():
        col = f"{group}_score"
        if col not in block.columns:
            continue
        present = block[col].notna()
        composite_acc = composite_acc + block[col].fillna(0.0) * weight
        weight_acc = weight_acc + present.astype(float) * weight
    with np.errstate(divide="ignore", invalid="ignore"):
        block["composite_score"] = composite_acc / weight_acc.replace(0.0, np.nan)
    block["composite_rank"] = block["composite_score"].rank(method="average", pct=True)
    # Metric completeness: share of *all configured* group scores that
    # are non-null in this row. Denominator is the full
    # ``config.group_weights`` size so missing groups penalise the
    # row's completeness rather than getting silently discounted.
    target_groups = max(1, len(config.group_weights))
    present_cols = [f"{g}_score" for g in config.group_weights if f"{g}_score" in block.columns]
    if present_cols:
        block["metric_completeness"] = (
            block[present_cols].notna().sum(axis=1) / target_groups
        )
    else:
        block["metric_completeness"] = 0.0
    return block


def build_extended_fundamental_ranker(
    metrics: pd.DataFrame,
    *,
    as_of_dates: Iterable[pd.Timestamp | str],
    sector_map: pd.DataFrame | None = None,
    config: ExtendedFundamentalConfig | None = None,
    generated_at: str | None = None,
) -> ExtendedFundamentalResult:
    cfg = config or ExtendedFundamentalConfig()
    if metrics is None or metrics.empty:
        return ExtendedFundamentalResult(
            frame=pd.DataFrame(columns=list(EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS)),
            coverage={"status": "empty_input"},
        )
    if "symbol" not in metrics.columns or "available_at" not in metrics.columns:
        raise ValueError("metrics frame requires symbol + available_at columns")
    work = metrics.copy()
    work["symbol"] = work["symbol"].astype(str)
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    work = work.dropna(subset=["symbol", "available_at"]).reset_index(drop=True)

    dates = sorted({pd.Timestamp(d) for d in as_of_dates})
    all_blocks: list[pd.DataFrame] = []
    generated = generated_at or pd.Timestamp.utcnow().isoformat()
    for as_of in dates:
        snapshot = _select_latest_pit_per_symbol(work, as_of)
        if snapshot.empty:
            continue
        joined = _attach_sector_asof(snapshot, sector_map, as_of)
        buckets = [_bucket(row["symbol"], row.get("sector_level_1")) for _, row in joined.iterrows()]
        joined["rank_bucket"] = [b for b, _ in buckets]
        joined["rank_bucket_kind"] = [k for _, k in buckets]
        joined["as_of_date"] = as_of
        for bucket, group in joined.groupby("rank_bucket", dropna=False):
            if len(group) < int(cfg.min_universe_per_bucket):
                continue
            scored = _score_block(group, cfg)
            all_blocks.append(scored)
    if not all_blocks:
        return ExtendedFundamentalResult(
            frame=pd.DataFrame(columns=list(EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS)),
            coverage={"status": "no_eligible_buckets"},
        )
    out_full = pd.concat(all_blocks, ignore_index=True)
    out_full["source_version"] = cfg.source_version
    out_full["generated_at"] = generated
    for col in EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS:
        if col not in out_full.columns:
            out_full[col] = pd.NA
    final = out_full[list(EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS)].sort_values(
        ["as_of_date", "rank_bucket", "composite_rank"], ascending=[True, True, False]
    ).reset_index(drop=True)
    coverage = {
        "total_rows": int(len(final)),
        "axes_present": {
            axis: int(axis in metrics.columns) for axis, _, _ in _AXIS_TABLE
        },
        "buckets_unique": int(final["rank_bucket"].nunique()),
        "average_metric_completeness": float(final["metric_completeness"].mean()) if not final.empty else 0.0,
        "status": "passed",
    }
    return ExtendedFundamentalResult(frame=final, coverage=coverage)


# ---------------------------------------------------------------------------
# Builder with writer
# ---------------------------------------------------------------------------

class ExtendedFundamentalRankerBuilder:
    def __init__(self, config: ExtendedFundamentalConfig | None = None) -> None:
        self.config = config or ExtendedFundamentalConfig()

    def build(
        self,
        metrics: pd.DataFrame,
        *,
        as_of_dates,
        sector_map: pd.DataFrame | None = None,
        generated_at: str | None = None,
    ) -> ExtendedFundamentalResult:
        return build_extended_fundamental_ranker(
            metrics,
            as_of_dates=as_of_dates,
            sector_map=sector_map,
            config=self.config,
            generated_at=generated_at,
        )

    def write(self, result: ExtendedFundamentalResult) -> ExtendedFundamentalResult:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "fundamental_extended"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "fundamental_extended.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        coverage_path = out_dir / "coverage_report.json"
        coverage_path.write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        manifests = root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests / "fundamental_extended.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "fundamental_extended",
                    "rows": int(len(result.frame)),
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "fundamental_extended": str(parquet_path),
            "coverage_report": str(coverage_path),
            "manifest": str(manifest_path),
        }
        return result


__all__ = [
    "EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS",
    "ExtendedFundamentalConfig",
    "ExtendedFundamentalRankerBuilder",
    "ExtendedFundamentalResult",
    "build_extended_fundamental_ranker",
]
