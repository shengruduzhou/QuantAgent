"""V7 trainable dataset builder from PIT market, fundamentals, evidence and risk data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.data.providers.qlib_provider import validate_qlib_market_schema
from quantagent.data.v7_label_builder import V7_LABEL_HORIZONS, build_forward_return_labels
from quantagent.data.v7_quality_gates import V7DataQualityGateConfig, evaluate_data_quality_gates


@dataclass(frozen=True)
class V7DatasetBuildConfig:
    horizons: tuple[int, ...] = V7_LABEL_HORIZONS
    enforce_quality_gates: bool = True
    min_rows: int = 100
    min_symbols: int = 5
    min_dates: int = 20
    source_name: str = "realdata"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class V7DatasetBuildResult:
    features: pd.DataFrame
    labels: pd.DataFrame
    dataset: pd.DataFrame
    feature_schema: dict[str, object]
    label_schema: dict[str, object]
    quality_report: dict[str, object]


def build_v7_training_dataset(
    market_panel: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    evidence_scores: pd.DataFrame | None = None,
    theme_exposure: pd.DataFrame | None = None,
    risk_features: pd.DataFrame | None = None,
    config: V7DatasetBuildConfig | None = None,
) -> V7DatasetBuildResult:
    config = config or V7DatasetBuildConfig()
    market_report = validate_qlib_market_schema(market_panel)
    if market_report["status"] != "passed":
        raise ValueError(f"market panel schema failed: {market_report}")

    features = build_market_features(market_panel)
    features = merge_pit_features(features, fundamentals, prefix="")
    features = merge_pit_features(features, evidence_scores, prefix="")
    features = merge_pit_features(features, theme_exposure, prefix="")
    features = merge_pit_features(features, risk_features, prefix="")
    label_result = build_forward_return_labels(market_panel, horizons=config.horizons)
    labels = label_result.frame
    label_schema = label_result.label_schema
    label_columns = [column for column in labels.columns if column.startswith("forward_return_")]
    dataset = features.merge(labels[["symbol", "trade_date", *label_columns, *[c for c in labels.columns if c.startswith("label_end_")]]], on=["symbol", "trade_date"], how="inner")
    feature_columns = [
        column
        for column in features.select_dtypes("number").columns
        if column not in {"open", "high", "low", "close", "volume", "amount"}
    ]
    quality = evaluate_data_quality_gates(
        dataset,
        V7DataQualityGateConfig(
            min_rows=config.min_rows,
            min_symbols=config.min_symbols,
            min_dates=config.min_dates,
            require_real_data=config.source_name != "mock",
        ),
    )
    if config.enforce_quality_gates and not quality.passed:
        raise ValueError(f"V7 dataset quality gates failed: {quality.failures}")
    return V7DatasetBuildResult(
        features=features,
        labels=labels,
        dataset=dataset,
        feature_schema={
            "feature_columns": feature_columns,
            "available_at_policy": "close-derived technical features are available from the next trading row",
            "source_name": config.source_name,
        },
        label_schema=label_schema,
        quality_report=quality.to_dict(),
    )


def build_market_features(
    market_panel: pd.DataFrame,
    *,
    st_flags: pd.DataFrame | None = None,
    limit_up_pct: float = 0.099,
    limit_down_pct: float = -0.099,
) -> pd.DataFrame:
    """Compute per-(date, symbol) market features.

    Stage 2.1 rebuild
    -----------------
    Previously ``is_suspended`` / ``is_limit_up`` / ``is_limit_down`` /
    ``is_st`` were hardcoded to ``False`` if missing from the input.
    They are now derived properly:

    * ``is_suspended``, ``is_limit_up``, ``is_limit_down`` come from
      :func:`quantagent.universe.filters.derive_market_flags` so the
      derivation has a single source of truth shared with the
      universe filter (Stage 1).
    * ``is_st`` comes from the optional ``st_flags`` parameter
      (long frame: ``trade_date, symbol, is_st``). Without it the
      column stays ``False`` and downstream callers should treat ST
      checks as "unknown" until Stage 2.2 wires an akshare source.

    All derived values use only data available up to and including
    ``trade_date`` (no look-ahead). ``available_at`` is set to the
    next observation date per symbol, falling back to
    ``trade_date + 1 calendar day`` for the last row.
    """
    data = market_panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    data["symbol"] = data["symbol"].astype(str)
    group = data.groupby("symbol", sort=False)
    data["return_1d"] = group["close"].pct_change()
    data["momentum_5d"] = group["close"].pct_change(5)
    data["momentum_20d"] = group["close"].pct_change(20)
    data["volatility_20d"] = group["return_1d"].transform(lambda s: s.rolling(20, min_periods=5).std())
    data["amount_mean_20d"] = group["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["volume_mean_20d"] = group["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["intraday_return"] = data["close"] / data["open"].replace(0, np.nan) - 1.0
    data["available_at"] = group["trade_date"].shift(-1)
    data["available_at"] = data["available_at"].fillna(data["trade_date"] + pd.Timedelta(days=1))

    # Derive tradability flags from OHLCV (single source of truth lives
    # in quantagent.universe.filters; importing here keeps the schema in
    # sync between feature build and runtime filter).
    from quantagent.universe.filters import derive_market_flags
    flags = derive_market_flags(
        data[["trade_date", "symbol", "volume", "amount", "close"]],
        limit_up_pct=limit_up_pct,
        limit_down_pct=limit_down_pct,
    )
    flags = flags.drop_duplicates(["trade_date", "symbol"], keep="last")
    data = data.drop(columns=["is_suspended", "is_limit_up", "is_limit_down"], errors="ignore").merge(
        flags, on=["trade_date", "symbol"], how="left"
    )
    for column in ("is_suspended", "is_limit_up", "is_limit_down"):
        data[column] = data[column].fillna(False).astype(bool)

    # ST flag: take from caller-provided table; otherwise default False
    # with the understanding that ST checks are "unknown" until Stage 2.2
    # wires the akshare ST source.
    if st_flags is not None and not st_flags.empty and "is_st" in st_flags.columns:
        st = st_flags[["trade_date", "symbol", "is_st"]].copy()
        st["trade_date"] = pd.to_datetime(st["trade_date"], errors="coerce")
        st["symbol"] = st["symbol"].astype(str)
        st["is_st"] = st["is_st"].fillna(False).astype(bool)
        st = st.drop_duplicates(["trade_date", "symbol"], keep="last")
        data = data.drop(columns=["is_st"], errors="ignore").merge(st, on=["trade_date", "symbol"], how="left")
        data["is_st"] = data["is_st"].fillna(False).astype(bool)
    elif "is_st" not in data.columns:
        data["is_st"] = False
    else:
        data["is_st"] = data["is_st"].fillna(False).astype(bool)

    return data.reset_index(drop=True)


def merge_pit_features(feature_frame: pd.DataFrame, extra: pd.DataFrame | None, prefix: str = "") -> pd.DataFrame:
    if extra is None or extra.empty:
        return feature_frame
    if "symbol" not in extra.columns or "available_at" not in extra.columns:
        raise ValueError("extra PIT features must include symbol and available_at")
    left = feature_frame.copy()
    left["available_at"] = pd.to_datetime(left["available_at"], errors="coerce")
    right = extra.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], errors="coerce")
    right = right.dropna(subset=["available_at", "symbol"]).sort_values(["symbol", "available_at"])
    merged_parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in left.sort_values(["symbol", "available_at"]).groupby("symbol", sort=False):
        symbol_extra = right[right["symbol"].astype(str) == str(symbol)]
        if symbol_extra.empty:
            merged_parts.append(symbol_frame)
            continue
        merged = pd.merge_asof(
            symbol_frame.sort_values("available_at"),
            symbol_extra.drop(columns=["symbol"]).sort_values("available_at"),
            on="available_at",
            direction="backward",
            suffixes=("", "_extra"),
        )
        merged_parts.append(merged)
    output = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else left
    if prefix:
        rename = {
            column: f"{prefix}{column}"
            for column in output.columns
            if column.endswith("_extra")
        }
        output = output.rename(columns=rename)
    return output
