"""Horizon-grouped model specs (short / mid / long).

The v8 spec section 5 requires three horizon-specific alpha models:

* **short_5d_model** — 1-5 trading days. Microstructure, intraday
  flow, limit-up / 炸板 / 封单 / 竞价 features, news sentiment
  short tail, top dragon list (龙虎榜).
* **mid_5d_30d_model** — 5-30 trading days. Sector rotation,
  industrial strength, policy follow-through, valuation repair,
  earnings quality, fund persistence.
* **long_30d_120d_model** — 30-120 trading days. Fundamental
  improvement, long-run policy direction, supply-chain position,
  valuation percentile, ROE, cash-flow quality, industry cycle.

This module is the **abstraction layer**: it does not retrain models,
it just maps each :class:`HorizonClass` to its horizons + feature
whitelist + label columns, and provides an ``ensemble_predictions``
helper that blends per-class outputs into one composite score per
(date, symbol).

The existing trainer in :mod:`quantagent.training.v7_experiment`
already accepts ``horizons=(1, 5, 20, 60, 120, 126)``; the new spec
just groups those horizons. Each :class:`HorizonModelSpec` produces
the dataset slice + feature subset for one class, and callers feed
that into the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class HorizonClass(str, Enum):
    SHORT = "short_5d"
    MID = "mid_5d_30d"
    LONG = "long_30d_120d"


# ---------------------------------------------------------------------------
# Feature whitelists per class
# ---------------------------------------------------------------------------

# Substring patterns matched against column names. A column is admitted
# to a class's feature set if any pattern is a substring of its name.
# Conservative: an empty whitelist for a class means *no filtering*
# (caller-supplied features pass through unchanged).
DEFAULT_SHORT_FEATURE_PATTERNS: tuple[str, ...] = (
    "rsi", "macd", "boll", "atr", "vol_ratio", "turnover",
    "amount", "momentum_1d", "momentum_5d", "limit_up", "limit_down",
    "fengdan", "zhaban", "auction", "overnight", "intraday",
    "north_flow", "dragon_list", "block_trade",
    "alpha_1", "alpha_2", "alpha_3", "alpha_4", "alpha_5",
    "alpha_6", "alpha_7", "alpha_8", "alpha_9",
    "ma5", "ma10", "ema5", "ema10",
    "news_short", "sentiment_short",
)

DEFAULT_MID_FEATURE_PATTERNS: tuple[str, ...] = (
    "sector_strength", "industry_strength", "sector_rotation",
    "policy_signal", "fund_persistence", "valuation_repair",
    "earnings_quality", "broker_consensus", "ma20", "ma60",
    "momentum_20d", "momentum_60d", "ic_ir", "regime_state",
    "rsi", "macd", "alpha_10", "alpha_15", "alpha_20",
    "alpha_30", "alpha_50", "north_cumulative",
    # v8.5 augmentation (2026-06-08): core30 evidence with real cross-section 2018-2026.
    # Mid-horizon by nature (policy/质量/板块共振/老庄/趋势); financials go to LONG (roe/...
    # already matched). The ensemble (short price-alpha + mid evidence + long financials)
    # then learns the regime-conditional value the flat overlay only approximated.
    "core_policy_score", "fundamental_quality_score", "sector_resonance_score",
    "old_dealer_risk_score", "trend_strength_score",
)

DEFAULT_LONG_FEATURE_PATTERNS: tuple[str, ...] = (
    "pe_ttm", "pb", "ps_ttm", "ev_to_ebitda",
    "roe", "roa", "gross_margin", "net_margin",
    "revenue_yoy", "net_income_yoy", "operating_cf",
    "debt_to_asset", "interest_coverage", "inventory_turnover",
    "dividend", "earnings_surprise", "accruals",
    "valuation_percentile", "industry_cycle", "supply_chain",
    "long_horizon", "ma120", "ma200",
    "policy_long", "macro_regime",
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonModelSpec:
    """One horizon class — horizons, feature whitelist, label columns."""

    name: HorizonClass
    horizons: tuple[int, ...]
    feature_patterns: tuple[str, ...]
    label_format: str = "forward_return_{h}d"

    @property
    def label_columns(self) -> tuple[str, ...]:
        return tuple(self.label_format.format(h=h) for h in self.horizons)

    def select_features(self, all_columns: Iterable[str]) -> list[str]:
        """Return the subset of ``all_columns`` matching this class's patterns.

        Returns the full list when ``feature_patterns`` is empty (the
        caller has not asked for filtering). Always excludes the
        label columns and the standard (symbol, trade_date,
        available_at) key columns.
        """
        excluded = set(self.label_columns) | {
            "symbol",
            "trade_date",
            "available_at",
            "label",
        }
        if not self.feature_patterns:
            return [c for c in all_columns if c not in excluded]
        keep: list[str] = []
        for col in all_columns:
            if col in excluded:
                continue
            cl = col.lower()
            if any(p.lower() in cl for p in self.feature_patterns):
                keep.append(col)
        return keep


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFAULT_HORIZON_SPECS: dict[HorizonClass, HorizonModelSpec] = {
    HorizonClass.SHORT: HorizonModelSpec(
        name=HorizonClass.SHORT,
        horizons=(1, 5),
        feature_patterns=DEFAULT_SHORT_FEATURE_PATTERNS,
    ),
    HorizonClass.MID: HorizonModelSpec(
        name=HorizonClass.MID,
        horizons=(5, 20),
        feature_patterns=DEFAULT_MID_FEATURE_PATTERNS,
    ),
    HorizonClass.LONG: HorizonModelSpec(
        name=HorizonClass.LONG,
        horizons=(60, 120),
        feature_patterns=DEFAULT_LONG_FEATURE_PATTERNS,
    ),
}


def get_horizon_spec(name: HorizonClass | str) -> HorizonModelSpec:
    cls = HorizonClass(name) if not isinstance(name, HorizonClass) else name
    return DEFAULT_HORIZON_SPECS[cls]


# ---------------------------------------------------------------------------
# Bundle preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonBundle:
    """One trainable dataset slice for a given HorizonClass."""

    name: HorizonClass
    panel: pd.DataFrame
    feature_columns: list[str]
    label_columns: list[str]
    horizons: tuple[int, ...]

    @property
    def primary_label(self) -> str:
        """The label column for the longest horizon in the class."""
        return f"forward_return_{self.horizons[-1]}d"


def build_horizon_bundle(
    panel: pd.DataFrame,
    *,
    spec: HorizonModelSpec,
    drop_rows_missing_primary_label: bool = True,
) -> HorizonBundle:
    """Slice a full training panel to one horizon class's view.

    Drops feature columns that don't match the class whitelist and,
    optionally, rows whose primary label is missing so the slice is
    immediately trainable.
    """
    if panel is None or panel.empty:
        return HorizonBundle(
            name=spec.name,
            panel=pd.DataFrame(),
            feature_columns=[],
            label_columns=list(spec.label_columns),
            horizons=spec.horizons,
        )
    label_cols = [c for c in spec.label_columns if c in panel.columns]
    if not label_cols:
        raise ValueError(
            f"panel lacks any of the required label columns for {spec.name}: "
            f"{spec.label_columns}"
        )
    feature_cols = spec.select_features(panel.columns)
    # always include keys; never include other label columns from sibling classes
    keep = ["symbol", "trade_date"] + (
        ["available_at"] if "available_at" in panel.columns else []
    ) + feature_cols + label_cols
    keep = [c for c in keep if c in panel.columns]
    seen: dict[str, bool] = {}
    unique_keep = [c for c in keep if not (c in seen or seen.update({c: True}))]
    out = panel[unique_keep].copy()
    if drop_rows_missing_primary_label:
        primary = f"forward_return_{spec.horizons[-1]}d"
        if primary in out.columns:
            out = out.dropna(subset=[primary]).reset_index(drop=True)
    return HorizonBundle(
        name=spec.name,
        panel=out,
        feature_columns=feature_cols,
        label_columns=label_cols,
        horizons=spec.horizons,
    )


def build_all_horizon_bundles(
    panel: pd.DataFrame,
    *,
    specs: Mapping[HorizonClass, HorizonModelSpec] | None = None,
    drop_rows_missing_primary_label: bool = True,
) -> dict[HorizonClass, HorizonBundle]:
    """Build bundles for every class with at least one label column present."""
    registry = specs or DEFAULT_HORIZON_SPECS
    out: dict[HorizonClass, HorizonBundle] = {}
    for cls, spec in registry.items():
        any_label_present = any(c in panel.columns for c in spec.label_columns)
        if not any_label_present:
            continue
        out[cls] = build_horizon_bundle(
            panel, spec=spec, drop_rows_missing_primary_label=drop_rows_missing_primary_label
        )
    return out


# ---------------------------------------------------------------------------
# Inference ensemble
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonEnsembleWeights:
    """Blend weights for the three horizon classes.

    Defaults bias mid-horizon slightly higher (the v8 spec
    consistently describes the medium term as the most actionable
    band). Long horizon gets the smallest weight not because it is
    less important but because its forecast is the noisiest in any
    single-period blend.
    """

    short: float = 0.30
    mid: float = 0.45
    long: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            HorizonClass.SHORT.value: float(self.short),
            HorizonClass.MID.value: float(self.mid),
            HorizonClass.LONG.value: float(self.long),
        }


def ensemble_horizon_predictions(
    predictions_by_class: Mapping[HorizonClass | str, pd.DataFrame],
    *,
    weights: HorizonEnsembleWeights | None = None,
    score_column: str = "alpha_score",
    key_columns: tuple[str, ...] = ("trade_date", "symbol"),
) -> pd.DataFrame:
    """Blend per-class prediction frames into a single composite score.

    Each input frame must have the ``key_columns`` plus the
    ``score_column``. Missing rows in any class are imputed as 0
    (the class abstains). The output also surfaces the per-class
    contribution for audit.
    """
    w = weights or HorizonEnsembleWeights()
    norm: dict[HorizonClass, pd.DataFrame] = {}
    for k, frame in predictions_by_class.items():
        cls = HorizonClass(k) if not isinstance(k, HorizonClass) else k
        if frame is None or frame.empty:
            continue
        missing = set(key_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"prediction frame for {cls.value} missing keys {missing}")
        if score_column not in frame.columns:
            raise ValueError(
                f"prediction frame for {cls.value} missing score column '{score_column}'"
            )
        f = frame[[*key_columns, score_column]].copy()
        f = f.rename(columns={score_column: f"{cls.value}_score"})
        norm[cls] = f
    if not norm:
        return pd.DataFrame(columns=[*key_columns, "composite_score"])
    # outer-join everything on the keys
    merged: pd.DataFrame | None = None
    for cls, frame in norm.items():
        merged = frame if merged is None else merged.merge(frame, on=list(key_columns), how="outer")
    assert merged is not None
    score_cols = [f"{cls.value}_score" for cls in norm]
    for col in score_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    wmap = {
        f"{HorizonClass.SHORT.value}_score": w.short,
        f"{HorizonClass.MID.value}_score": w.mid,
        f"{HorizonClass.LONG.value}_score": w.long,
    }
    merged["composite_score"] = sum(
        merged[col] * wmap.get(col, 0.0) for col in score_cols
    )
    keep = [*key_columns, "composite_score", *score_cols]
    return merged[keep].sort_values(list(key_columns)).reset_index(drop=True)


__all__ = [
    "DEFAULT_HORIZON_SPECS",
    "DEFAULT_LONG_FEATURE_PATTERNS",
    "DEFAULT_MID_FEATURE_PATTERNS",
    "DEFAULT_SHORT_FEATURE_PATTERNS",
    "HorizonBundle",
    "HorizonClass",
    "HorizonEnsembleWeights",
    "HorizonModelSpec",
    "build_all_horizon_bundles",
    "build_horizon_bundle",
    "ensemble_horizon_predictions",
    "get_horizon_spec",
]
