"""V7 feature-group registry, feature builders, and PIT join helper.

This module is the single source of truth for which columns belong to
which V7 feature group (short_term, medium_term, long_horizon,
fundamental, valuation, risk, liquidity, regime). It also exposes:

* ``V7FeatureGroup`` — group metadata: required source kind, builder,
  PIT policy, lookback window, missingness policy, expected columns.
* ``select_v7_feature_columns`` — resolves which columns from an
  arbitrary frame fall into which group.
* ``build_v7_feature_groups`` — runs registered builders that can
  synthesise group columns from a market panel (returns/momentum/etc.).
* ``join_pit_features`` — strict per-symbol PIT as-of merge across
  multiple auxiliary frames.

Important: builders never write future labels and never use forward
prices. ``build_v7_feature_groups`` only consumes the inputs declared
in ``V7FeatureGroup.required_columns`` and fails loudly if they are
absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Static column catalogues (kept for backwards compatibility)
# ---------------------------------------------------------------------------

SHORT_TERM_FEATURES: tuple[str, ...] = (
    "return_1d",
    "momentum_5d",
    "intraday_return",
    "gap_open_return",
    "volume_zscore_5d",
    "amount_zscore_5d",
    "turnover_rate_5d",
    "fund_flow_5d",
    "news_sentiment_score",
)
MEDIUM_TERM_FEATURES: tuple[str, ...] = (
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "reversal_5d",
    "sector_rotation_score",
    "earnings_revision_score",
    "amount_mean_20d",
    "volume_mean_20d",
    "liquidity_20d",
)
LONG_HORIZON_FEATURES: tuple[str, ...] = (
    "momentum_120d",
    "momentum_252d",
    "trend_strength_252d",
    "drawdown_252d",
    "policy_strength",
    "policy_chain_centrality_120d",
    "macro_industry_phase_120d",
    "macro_credit_cycle_120d",
    "macro_monetary_tailwind_120d",
    "structural_domestic_substitution_120d",
    "structural_bottleneck_120d",
    "exposure_score",
)
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "revenue_growth",
    "net_income_growth",
    "gross_margin",
    "net_margin",
    "roe",
    "roa",
    "ocf_to_profit",
    "fcff",
    "quality_score",
    "growth_score",
)
VALUATION_FEATURES: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "ps_ttm",
    "ev_ebitda",
    "peg",
    "dividend_yield",
    "valuation_history_zscore_120d",
    "valuation_industry_zscore_120d",
    "valuation_margin_of_safety_120d",
)
RISK_FEATURES: tuple[str, ...] = (
    "fraud_risk_score",
    "management_risk_score",
    "debt_to_asset",
    "current_ratio",
    "quick_ratio",
    "regulatory_penalty_score",
    "audit_opinion_score",
    "is_st",
    "is_suspended",
)
LIQUIDITY_FEATURES: tuple[str, ...] = (
    "turnover_rate",
    "amount_mean_20d",
    "free_float_market_cap",
    "market_cap",
    "capacity_proxy_120d",
)
REGIME_FEATURES: tuple[str, ...] = (
    "market_regime_score",
    "volatility_regime",
    "liquidity_regime",
    "breadth_score",
    "drawdown_risk",
    "market_drawdown_120d",
)


V7_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "short_term": SHORT_TERM_FEATURES,
    "medium_term": MEDIUM_TERM_FEATURES,
    "long_horizon": LONG_HORIZON_FEATURES,
    "fundamental": FUNDAMENTAL_FEATURES,
    "valuation": VALUATION_FEATURES,
    "risk": RISK_FEATURES,
    "liquidity": LIQUIDITY_FEATURES,
    "regime": REGIME_FEATURES,
}


# ---------------------------------------------------------------------------
# Real feature-group registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V7FeatureGroup:
    """Registry entry describing how to materialise a feature group.

    ``builder`` is optional; some groups (e.g. fundamentals, valuation)
    are joined in from external sources rather than computed from the
    market panel.
    """

    name: str
    required_source_kinds: tuple[str, ...]
    required_columns: tuple[str, ...]
    produced_columns: tuple[str, ...]
    pit_policy: str
    lookback_days: int
    missingness_policy: str = "fillna_with_zero"
    builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    description: str = ""


def _require(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"feature group '{name}' requires missing columns {missing}")


def _short_term_features(market: pd.DataFrame) -> pd.DataFrame:
    _require(market, "short_term", ("symbol", "trade_date", "open", "close", "volume", "amount"))
    data = market.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    group = data.groupby("symbol", sort=False)
    data["return_1d"] = group["close"].pct_change()
    data["momentum_5d"] = group["close"].pct_change(5)
    data["intraday_return"] = data["close"] / data["open"].replace(0, np.nan) - 1.0
    data["gap_open_return"] = data["open"] / group["close"].shift(1) - 1.0
    data["volume_zscore_5d"] = group["volume"].transform(_zscore(5))
    data["amount_zscore_5d"] = group["amount"].transform(_zscore(5))
    if "turnover_rate" in data.columns:
        data["turnover_rate_5d"] = group["turnover_rate"].transform(
            lambda s: s.rolling(5, min_periods=2).mean()
        )
    return data


def _medium_term_features(market: pd.DataFrame) -> pd.DataFrame:
    _require(market, "medium_term", ("symbol", "trade_date", "close", "volume", "amount"))
    data = market.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    group = data.groupby("symbol", sort=False)
    ret = group["close"].pct_change()
    data["momentum_20d"] = group["close"].pct_change(20)
    data["momentum_60d"] = group["close"].pct_change(60)
    data["volatility_20d"] = ret.groupby(data["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    data["reversal_5d"] = -group["close"].pct_change(5)
    data["amount_mean_20d"] = group["amount"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    data["volume_mean_20d"] = group["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    data["liquidity_20d"] = (
        data["amount_mean_20d"] / data["volume_mean_20d"].replace(0, np.nan)
    )
    return data


def _long_horizon_features(market: pd.DataFrame) -> pd.DataFrame:
    _require(market, "long_horizon", ("symbol", "trade_date", "close"))
    data = market.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    group = data.groupby("symbol", sort=False)
    data["momentum_120d"] = group["close"].pct_change(120)
    data["momentum_252d"] = group["close"].pct_change(252)
    # Trend strength: rolling close > rolling mean ratio.
    rolling_mean_120 = group["close"].transform(lambda s: s.rolling(120, min_periods=20).mean())
    data["trend_strength_252d"] = data["close"] / rolling_mean_120.replace(0, np.nan) - 1.0
    rolling_max_252 = group["close"].transform(lambda s: s.rolling(252, min_periods=20).max())
    data["drawdown_252d"] = data["close"] / rolling_max_252.replace(0, np.nan) - 1.0
    return data


def _regime_features(market: pd.DataFrame) -> pd.DataFrame:
    _require(market, "regime", ("symbol", "trade_date", "close"))
    data = market.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    # Market-level breadth & drawdown proxy from the cross-section.
    by_date = data.groupby("trade_date")
    breadth = by_date["close"].apply(_breadth)
    data["breadth_score"] = data["trade_date"].map(breadth)
    market_close = by_date["close"].mean()
    rolling_max = market_close.rolling(120, min_periods=20).max()
    drawdown = market_close / rolling_max - 1.0
    data["market_drawdown_120d"] = data["trade_date"].map(drawdown)
    return data


def _zscore(window: int):
    def inner(series: pd.Series) -> pd.Series:
        mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
        std = series.rolling(window, min_periods=max(2, window // 2)).std()
        return (series - mean) / std.replace(0, np.nan)

    return inner


def _breadth(group: pd.Series) -> float:
    if group.empty:
        return 0.0
    advances = (group.pct_change() > 0).sum()
    return float(advances) / max(1, len(group))


V7_FEATURE_GROUP_REGISTRY: dict[str, V7FeatureGroup] = {
    "short_term": V7FeatureGroup(
        name="short_term",
        required_source_kinds=("market",),
        required_columns=("symbol", "trade_date", "open", "close", "volume", "amount"),
        produced_columns=SHORT_TERM_FEATURES,
        pit_policy="close-derived; available_at = next trading day",
        lookback_days=5,
        builder=_short_term_features,
        description="1-5 day flow & price action.",
    ),
    "medium_term": V7FeatureGroup(
        name="medium_term",
        required_source_kinds=("market",),
        required_columns=("symbol", "trade_date", "close", "volume", "amount"),
        produced_columns=MEDIUM_TERM_FEATURES,
        pit_policy="close-derived; available_at = next trading day",
        lookback_days=60,
        builder=_medium_term_features,
        description="20-60 day momentum and volatility.",
    ),
    "long_horizon": V7FeatureGroup(
        name="long_horizon",
        required_source_kinds=("market",),
        required_columns=("symbol", "trade_date", "close"),
        produced_columns=LONG_HORIZON_FEATURES,
        pit_policy="close-derived; available_at = next trading day",
        lookback_days=252,
        builder=_long_horizon_features,
        description="120-252 day trend, drawdown, and structural exposure.",
    ),
    "fundamental": V7FeatureGroup(
        name="fundamental",
        required_source_kinds=("fundamentals",),
        required_columns=("symbol", "available_at"),
        produced_columns=FUNDAMENTAL_FEATURES,
        pit_policy="available_at = ann_date resolved via trading calendar",
        lookback_days=400,
        missingness_policy="leave_nan_with_missingness_flag",
        builder=None,
        description="Quality / growth / margin from PIT statements.",
    ),
    "valuation": V7FeatureGroup(
        name="valuation",
        required_source_kinds=("valuation",),
        required_columns=("symbol", "available_at"),
        produced_columns=VALUATION_FEATURES,
        pit_policy="available_at = snapshot trade_date",
        lookback_days=120,
        missingness_policy="leave_nan_with_missingness_flag",
        builder=None,
        description="PE/PB/PS/EV/EBITDA + history & industry z-scores.",
    ),
    "risk": V7FeatureGroup(
        name="risk",
        required_source_kinds=("fundamentals", "tradability"),
        required_columns=("symbol",),
        produced_columns=RISK_FEATURES,
        pit_policy="available_at = ann_date or trade_date",
        lookback_days=120,
        missingness_policy="leave_nan_with_missingness_flag",
        builder=None,
        description="ST / suspension / leverage / fraud / audit risk.",
    ),
    "liquidity": V7FeatureGroup(
        name="liquidity",
        required_source_kinds=("market", "valuation"),
        required_columns=("symbol", "trade_date"),
        produced_columns=LIQUIDITY_FEATURES,
        pit_policy="close-derived; available_at = next trading day",
        lookback_days=120,
        missingness_policy="fillna_with_zero",
        builder=None,
        description="Amount / turnover / capacity proxies.",
    ),
    "regime": V7FeatureGroup(
        name="regime",
        required_source_kinds=("market",),
        required_columns=("symbol", "trade_date", "close"),
        produced_columns=REGIME_FEATURES,
        pit_policy="cross-sectional; available_at = next trading day",
        lookback_days=120,
        builder=_regime_features,
        description="Market breadth, drawdown, volatility regime proxies.",
    ),
}


@dataclass(frozen=True)
class V7FeatureSelection:
    selected: tuple[str, ...]
    group_to_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)


def select_v7_feature_columns(
    frame: pd.DataFrame,
    groups: Iterable[str] = (),
    extra_columns: Iterable[str] = (),
) -> V7FeatureSelection:
    available = set(frame.columns) if frame is not None else set()
    target_groups = tuple(groups) or tuple(V7_FEATURE_GROUPS.keys())
    selected: list[str] = []
    seen: set[str] = set()
    group_map: dict[str, tuple[str, ...]] = {}
    for group in target_groups:
        if group not in V7_FEATURE_GROUPS:
            raise ValueError(f"unknown V7 feature group: {group}")
        present = tuple(c for c in V7_FEATURE_GROUPS[group] if c in available)
        group_map[group] = present
        for column in present:
            if column not in seen:
                seen.add(column)
                selected.append(column)
    for column in extra_columns:
        if column in available and column not in seen:
            seen.add(column)
            selected.append(column)
    return V7FeatureSelection(selected=tuple(selected), group_to_columns=group_map)


def build_v7_feature_groups(
    market_panel: pd.DataFrame,
    groups: Iterable[str] = (),
) -> pd.DataFrame:
    """Run all registered builders that can run against ``market_panel``.

    Groups without a builder are skipped silently (they need PIT joins).
    Builders fail loudly when their required columns are missing.
    """
    if market_panel is None or market_panel.empty:
        return pd.DataFrame()
    target_groups = tuple(groups) or tuple(V7_FEATURE_GROUP_REGISTRY.keys())
    output = market_panel.copy()
    for group_name in target_groups:
        group = V7_FEATURE_GROUP_REGISTRY.get(group_name)
        if group is None or group.builder is None:
            continue
        produced = group.builder(output)
        new_columns = [c for c in produced.columns if c not in output.columns]
        for column in new_columns:
            output[column] = produced[column].values
    return output


def feature_schema_for_groups(groups: Iterable[str] = ()) -> dict[str, object]:
    """Emit a feature schema (group → columns + PIT policy + lookback)."""
    target = tuple(groups) or tuple(V7_FEATURE_GROUP_REGISTRY.keys())
    schema: dict[str, object] = {"groups": {}, "version": "v7"}
    for name in target:
        group = V7_FEATURE_GROUP_REGISTRY.get(name)
        if group is None:
            continue
        schema["groups"][name] = {  # type: ignore[index]
            "required_source_kinds": list(group.required_source_kinds),
            "required_columns": list(group.required_columns),
            "produced_columns": list(group.produced_columns),
            "pit_policy": group.pit_policy,
            "lookback_days": group.lookback_days,
            "missingness_policy": group.missingness_policy,
            "has_builder": group.builder is not None,
        }
    return schema


def join_pit_features(
    base: pd.DataFrame,
    extras: Iterable[pd.DataFrame],
    *,
    on: tuple[str, str] = ("symbol", "available_at"),
) -> pd.DataFrame:
    """Strict PIT as-of join across multiple auxiliary feature frames."""
    if base is None or base.empty:
        return base
    output = base.copy()
    output[on[1]] = pd.to_datetime(output[on[1]], errors="coerce")
    for extra in extras:
        if extra is None or extra.empty:
            continue
        missing = [column for column in on if column not in extra.columns]
        if missing:
            raise ValueError(f"extra frame missing PIT keys {missing}")
        right = extra.copy()
        right[on[1]] = pd.to_datetime(right[on[1]], errors="coerce")
        right = right.dropna(subset=list(on)).sort_values([on[0], on[1]])
        merged_parts: list[pd.DataFrame] = []
        for symbol, base_part in output.sort_values([on[0], on[1]]).groupby(on[0], sort=False):
            symbol_extra = right[right[on[0]].astype(str) == str(symbol)]
            if symbol_extra.empty:
                merged_parts.append(base_part)
                continue
            merged = pd.merge_asof(
                base_part.sort_values(on[1]),
                symbol_extra.drop(columns=[on[0]]).sort_values(on[1]),
                on=on[1],
                direction="backward",
            )
            merged_parts.append(merged)
        output = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else output
    return output


__all__ = [
    "V7_FEATURE_GROUPS",
    "V7_FEATURE_GROUP_REGISTRY",
    "V7FeatureGroup",
    "V7FeatureSelection",
    "select_v7_feature_columns",
    "build_v7_feature_groups",
    "feature_schema_for_groups",
    "join_pit_features",
    "SHORT_TERM_FEATURES",
    "MEDIUM_TERM_FEATURES",
    "LONG_HORIZON_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "VALUATION_FEATURES",
    "RISK_FEATURES",
    "LIQUIDITY_FEATURES",
    "REGIME_FEATURES",
]
