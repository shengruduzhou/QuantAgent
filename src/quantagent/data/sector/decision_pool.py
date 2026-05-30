"""SectorPoolV8 — decision-axis sector pool (spec section 3).

Distinct from the existing IC-driven ``sector_pool`` in
:mod:`quantagent.data.sector.sector_pool` (which scores **how well
the model predicts** a sector at a given horizon). This module
scores **the sector's investment desirability** based on the union
of policy, capital-flow, sentiment, broker, technical and fundamental
signals. The two pools are complementary and downstream consumers
can intersect them: a sector that is both ``core`` in the IC pool
AND ``BUY`` in this decision pool is the highest-conviction bucket.

Inputs are deliberately *interfaces* — each input frame can be
absent; the corresponding score defaults to ``NaN`` and the
``confidence`` field reflects how many input axes were available.

Output schema (sector_pool_v8.parquet):

    date, sector_code, sector_name,
    policy_score, capital_flow_score, sentiment_score,
    broker_attention_score, market_strength_score, liquidity_score,
    valuation_percentile, risk_score,
    final_sector_rank, confidence

All scores normalised to ``[0, 1]`` (higher = more attractive),
except ``valuation_percentile`` which is a cross-sectional percentile
in ``[0, 1]`` (lower = cheaper). ``risk_score`` is also in ``[0, 1]``
where higher = more risky.

The pool is a **filter**, not a signal: only the optimizer +
OrderManager pipeline may produce target weights. Callers should
treat ``final_sector_rank`` as an input to candidate universe
construction, not as a target weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SECTOR_POOL_V8_COLUMNS: tuple[str, ...] = (
    "date",
    "sector_code",
    "sector_name",
    "policy_score",
    "capital_flow_score",
    "sentiment_score",
    "broker_attention_score",
    "market_strength_score",
    "liquidity_score",
    "valuation_percentile",
    "risk_score",
    "final_sector_rank",
    "confidence",
)


_DEFAULT_WEIGHTS: dict[str, float] = {
    "policy_score": 0.20,
    "capital_flow_score": 0.20,
    "sentiment_score": 0.05,
    "broker_attention_score": 0.10,
    "market_strength_score": 0.20,
    "liquidity_score": 0.10,
    "valuation_score": 0.10,   # 1 - valuation_percentile
    "risk_score": -0.15,       # risk negatively contributes
}


@dataclass(frozen=True)
class SectorPoolV8Config:
    """Axis weights + thresholds for the final aggregate rank."""

    axis_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    min_axes_for_confidence: int = 4
    output_root: str | Path = "runtime/data/v7"
    source_version: str = "unknown"


@dataclass
class SectorPoolV8Result:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-axis normalisers
# ---------------------------------------------------------------------------

def _to_unit_interval(series: pd.Series, *, higher_better: bool = True) -> pd.Series:
    """Min-max normalise into [0, 1]. NaN preserved as NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo = float(s.min(skipna=True))
    hi = float(s.max(skipna=True))
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float).where(s.notna())
    if higher_better:
        return ((s - lo) / (hi - lo)).clip(0.0, 1.0)
    return ((hi - s) / (hi - lo)).clip(0.0, 1.0)


def _cross_sectional_percentile(series: pd.Series, *, higher_better: bool = True) -> pd.Series:
    """Rank-based percentile in [0, 1]; ties get average rank."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranks = valid.rank(method="average", ascending=not higher_better)
    pct = (ranks - 1) / (len(valid) - 1)
    return pct.reindex(series.index)


# ---------------------------------------------------------------------------
# Axis collectors — each takes the optional input and returns a Series
# indexed by sector_code with values in [0, 1] (or NaN where unknown).
# ---------------------------------------------------------------------------

def _axis_policy(
    thesis_frame: pd.DataFrame | None,
    sector_index: pd.Index,
) -> pd.Series:
    """Aggregate capital_flow_thesis directions tied to each sector."""
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if thesis_frame is None or thesis_frame.empty:
        return out
    work = thesis_frame.copy()
    needed = {"direction_kind", "direction_value", "thesis_sign", "confidence",
              "validation_status"}
    if not needed.issubset(work.columns):
        return out
    rows = work[work["direction_kind"].isin(["sector", "theme"])]
    if rows.empty:
        return out
    weights = rows["confidence"].astype(float).clip(0.0, 1.0)
    rows = rows.assign(_w=weights)
    # Verified theses worth more, rejected theses contribute negatively
    status_mult = rows["validation_status"].map(
        {"verified": 1.0, "partially_verified": 0.8, "unverified": 0.5,
         "rejected": -0.5, "expired": 0.0}
    ).fillna(0.5)
    rows = rows.assign(_score=rows["thesis_sign"].astype(float) * status_mult * rows["_w"])
    agg = rows.groupby("direction_value", dropna=True)["_score"].sum()
    # Map signed score to [0, 1] via tanh-ish squash centred at 0.5
    mapped = (np.tanh(agg) + 1.0) / 2.0
    for sector in sector_index:
        if sector in mapped.index:
            out.loc[sector] = float(mapped.loc[sector])
    return out


def _axis_capital_flow(
    capital_flow_panel: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Net inflow / outflow per sector from any flow source.

    Expected long-form: ``trade_date / sector_code / net_flow``.
    """
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if capital_flow_panel is None or capital_flow_panel.empty:
        return out
    work = capital_flow_panel.copy()
    if not {"trade_date", "sector_code", "net_flow"}.issubset(work.columns):
        return out
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[(work["trade_date"] <= cutoff) & (work["trade_date"] > cutoff - pd.Timedelta(days=10))]
    if recent.empty:
        return out
    sums = recent.groupby("sector_code")["net_flow"].sum()
    return _to_unit_interval(sums.reindex(sector_index), higher_better=True)


def _axis_sentiment(
    sentiment_panel: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Mean sentiment by sector over recent window.

    Expected long-form: ``trade_date / sector_code / sentiment``.
    """
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if sentiment_panel is None or sentiment_panel.empty:
        return out
    work = sentiment_panel.copy()
    if not {"trade_date", "sector_code", "sentiment"}.issubset(work.columns):
        return out
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[(work["trade_date"] <= cutoff) & (work["trade_date"] > cutoff - pd.Timedelta(days=5))]
    if recent.empty:
        return out
    avg = recent.groupby("sector_code")["sentiment"].mean()
    # sentiment expected in [-1, 1]
    mapped = ((avg.clip(-1.0, 1.0) + 1.0) / 2.0)
    return mapped.reindex(sector_index)


def _axis_broker_attention(
    broker_panel: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Broker note count + buy-side rating tilt per sector.

    Expected long-form: ``available_at / sector_code / rating_score / broker_credibility``.
    """
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if broker_panel is None or broker_panel.empty:
        return out
    work = broker_panel.copy()
    if not {"available_at", "sector_code"}.issubset(work.columns):
        return out
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[(work["available_at"] <= cutoff) & (work["available_at"] > cutoff - pd.Timedelta(days=30))]
    if recent.empty:
        return out
    if "rating_score" in recent.columns and "broker_credibility" in recent.columns:
        recent = recent.assign(_w=recent["rating_score"].astype(float) * recent["broker_credibility"].astype(float))
        agg = recent.groupby("sector_code")["_w"].sum()
        mapped = (np.tanh(agg) + 1.0) / 2.0
        return mapped.reindex(sector_index)
    counts = recent.groupby("sector_code").size().astype(float)
    return _to_unit_interval(counts.reindex(sector_index), higher_better=True)


def _axis_market_strength(
    sector_returns: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """20-day cumulative return per sector."""
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if sector_returns is None or sector_returns.empty:
        return out
    work = sector_returns.copy()
    if not {"trade_date", "sector_code", "ret"}.issubset(work.columns):
        return out
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[(work["trade_date"] <= cutoff) & (work["trade_date"] > cutoff - pd.Timedelta(days=30))]
    if recent.empty:
        return out
    cum = recent.groupby("sector_code")["ret"].apply(lambda s: float(np.prod(1.0 + s.dropna().values) - 1.0))
    return _to_unit_interval(cum.reindex(sector_index), higher_better=True)


def _axis_liquidity(
    sector_liquidity: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Mean daily amount per sector."""
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if sector_liquidity is None or sector_liquidity.empty:
        return out
    work = sector_liquidity.copy()
    if not {"trade_date", "sector_code", "amount"}.issubset(work.columns):
        return out
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[(work["trade_date"] <= cutoff) & (work["trade_date"] > cutoff - pd.Timedelta(days=20))]
    if recent.empty:
        return out
    mean_amt = recent.groupby("sector_code")["amount"].mean()
    return _to_unit_interval(mean_amt.reindex(sector_index), higher_better=True)


def _axis_valuation_percentile(
    sector_valuation: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Cross-sectional valuation percentile (cheaper = lower percentile)."""
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if sector_valuation is None or sector_valuation.empty:
        return out
    work = sector_valuation.copy()
    if not {"trade_date", "sector_code"}.issubset(work.columns):
        return out
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    recent = work[work["trade_date"] <= cutoff].sort_values("trade_date").drop_duplicates("sector_code", keep="last")
    if recent.empty:
        return out
    val_col = next((c for c in ("pe_ttm", "pb", "ps_ttm") if c in recent.columns), None)
    if val_col is None:
        return out
    pct = _cross_sectional_percentile(
        recent.set_index("sector_code")[val_col], higher_better=False  # cheaper sector → lower percentile
    )
    # higher_better=False makes lower raw value get percentile 0 → cheap = 0
    # Flip semantics to: "lower means cheaper" → return as-is so cheap = 0.
    return pct.reindex(sector_index)


def _axis_risk(
    risk_panel: pd.DataFrame | None,
    sector_index: pd.Index,
    date: pd.Timestamp,
) -> pd.Series:
    """Aggregate risk flags (drawdown, concentration of ST names) per sector."""
    out = pd.Series(np.nan, index=sector_index, dtype=float)
    if risk_panel is None or risk_panel.empty:
        return out
    work = risk_panel.copy()
    if not {"sector_code"}.issubset(work.columns):
        return out
    if "trade_date" in work.columns:
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work[work["trade_date"] <= pd.Timestamp(date)]
    if work.empty:
        return out
    risk_col = next((c for c in ("risk_score", "st_ratio", "drawdown_20d") if c in work.columns), None)
    if risk_col is None:
        return out
    agg = work.groupby("sector_code")[risk_col].mean()
    return _to_unit_interval(agg.reindex(sector_index), higher_better=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _aggregate(
    axes: dict[str, pd.Series],
    config: SectorPoolV8Config,
) -> tuple[pd.Series, pd.Series]:
    """Return (final_score, confidence) per sector.

    The aggregate score uses the configured ``axis_weights``; missing
    axes are dropped from the weighted sum and the remaining weights
    are renormalised. ``confidence`` is the share of axes that
    contributed (clipped to [0, 1]).
    """
    if not axes:
        empty = pd.Series(dtype=float)
        return empty, empty
    # All axes share the same sector_index.
    sector_index = next(iter(axes.values())).index
    # valuation axis: invert so "cheap" (low percentile) contributes positively
    valuation = axes.get("valuation_percentile")
    if valuation is not None:
        axes["valuation_score"] = 1.0 - valuation
    # Discard 'valuation_percentile' from weighting (we use valuation_score key)
    weight_lookup = config.axis_weights
    score = pd.Series(0.0, index=sector_index, dtype=float)
    weight_used = pd.Series(0.0, index=sector_index, dtype=float)
    axes_present = pd.Series(0, index=sector_index, dtype=int)
    for axis_name, weight in weight_lookup.items():
        if axis_name not in axes:
            continue
        series = axes[axis_name].astype(float)
        present = series.notna()
        score = score + (series.fillna(0.0) * abs(weight)) * (np.sign(weight) if weight != 0 else 1.0)
        weight_used = weight_used + present.astype(float) * abs(weight)
        axes_present = axes_present + present.astype(int)
    # Normalise per row so a sector that only had 2 axes available
    # doesn't look weaker than one with 6.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_score = score / weight_used.replace(0.0, np.nan)
    norm_score = norm_score.clip(-1.0, 1.0)
    # map to [0, 1]
    final = (norm_score.fillna(0.5) + 1.0) / 2.0
    # confidence: present axes / target axes
    target = max(1, config.min_axes_for_confidence)
    confidence = (axes_present.astype(float) / target).clip(0.0, 1.0)
    return final.astype(float), confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sector_pool_v8(
    *,
    date: pd.Timestamp,
    sectors: pd.DataFrame,
    capital_flow_theses: pd.DataFrame | None = None,
    capital_flow_panel: pd.DataFrame | None = None,
    sentiment_panel: pd.DataFrame | None = None,
    broker_panel: pd.DataFrame | None = None,
    sector_returns: pd.DataFrame | None = None,
    sector_liquidity: pd.DataFrame | None = None,
    sector_valuation: pd.DataFrame | None = None,
    risk_panel: pd.DataFrame | None = None,
    config: SectorPoolV8Config | None = None,
) -> SectorPoolV8Result:
    """Build a decision-axis sector pool for one date.

    ``sectors`` must have at minimum ``sector_code`` and
    ``sector_name`` columns.
    """
    cfg = config or SectorPoolV8Config()
    if sectors is None or sectors.empty:
        empty = pd.DataFrame(columns=list(SECTOR_POOL_V8_COLUMNS))
        return SectorPoolV8Result(frame=empty, coverage={"status": "no_sectors"})
    if "sector_code" not in sectors.columns:
        raise ValueError("sectors frame requires a sector_code column")
    work = sectors.copy()
    work["sector_code"] = work["sector_code"].astype(str)
    work = work.drop_duplicates("sector_code", keep="last").set_index("sector_code")
    if "sector_name" not in work.columns:
        work["sector_name"] = work.index
    sector_index = work.index

    date = pd.Timestamp(date)

    axes: dict[str, pd.Series] = {
        "policy_score": _axis_policy(capital_flow_theses, sector_index),
        "capital_flow_score": _axis_capital_flow(capital_flow_panel, sector_index, date),
        "sentiment_score": _axis_sentiment(sentiment_panel, sector_index, date),
        "broker_attention_score": _axis_broker_attention(broker_panel, sector_index, date),
        "market_strength_score": _axis_market_strength(sector_returns, sector_index, date),
        "liquidity_score": _axis_liquidity(sector_liquidity, sector_index, date),
        "valuation_percentile": _axis_valuation_percentile(sector_valuation, sector_index, date),
        "risk_score": _axis_risk(risk_panel, sector_index, date),
    }
    final, confidence = _aggregate(axes, cfg)
    rank = final.rank(method="dense", ascending=False).astype("Int64")

    rows = []
    for code in sector_index:
        rows.append({
            "date": date,
            "sector_code": code,
            "sector_name": str(work.loc[code, "sector_name"]),
            "policy_score": _f(axes["policy_score"].get(code)),
            "capital_flow_score": _f(axes["capital_flow_score"].get(code)),
            "sentiment_score": _f(axes["sentiment_score"].get(code)),
            "broker_attention_score": _f(axes["broker_attention_score"].get(code)),
            "market_strength_score": _f(axes["market_strength_score"].get(code)),
            "liquidity_score": _f(axes["liquidity_score"].get(code)),
            "valuation_percentile": _f(axes["valuation_percentile"].get(code)),
            "risk_score": _f(axes["risk_score"].get(code)),
            "final_sector_rank": int(rank.get(code)) if pd.notna(rank.get(code)) else None,
            "confidence": _f(confidence.get(code)),
        })
    frame = pd.DataFrame(rows, columns=list(SECTOR_POOL_V8_COLUMNS))
    coverage = {
        "n_sectors": int(len(frame)),
        "axes_with_data": {k: int(v.notna().sum()) for k, v in axes.items()},
        "axis_weights": dict(cfg.axis_weights),
        "min_axes_for_confidence": int(cfg.min_axes_for_confidence),
    }
    return SectorPoolV8Result(frame=frame, coverage=coverage)


def _f(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


class SectorPoolV8Builder:
    """Stateful wrapper with optional writer."""

    def __init__(self, config: SectorPoolV8Config | None = None) -> None:
        self.config = config or SectorPoolV8Config()

    def build(self, **kwargs) -> SectorPoolV8Result:
        return build_sector_pool_v8(config=self.config, **kwargs)

    def write(self, result: SectorPoolV8Result) -> SectorPoolV8Result:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "sector_pool_v8"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "sector_pool_v8.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        coverage_path = out_dir / "coverage_report.json"
        coverage_path.write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        manifests = root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests / "sector_pool_v8.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "sector_pool_v8",
                    "rows": int(len(result.frame)),
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                    "policy": (
                        "decision-axis sector pool — filter only, "
                        "never directly produces target weights"
                    ),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "sector_pool_v8": str(parquet_path),
            "coverage_report": str(coverage_path),
            "manifest": str(manifest_path),
        }
        return result


__all__ = [
    "SECTOR_POOL_V8_COLUMNS",
    "SectorPoolV8Builder",
    "SectorPoolV8Config",
    "SectorPoolV8Result",
    "build_sector_pool_v8",
]
