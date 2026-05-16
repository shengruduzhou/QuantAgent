"""Build the V7 gold-tier training dataset from PIT silver inputs.

The builder performs strict ``available_at <= trade_date`` as-of joins,
attaches multi-horizon forward-return labels, writes missingness flags
for every joined source, and emits a feature schema plus a DataManifest.
Production callers must point ``--fundamentals-root`` at the V7 PIT cache;
synthetic fallback is forbidden so the resulting frame can be trusted by
the alpha trainer and the live-readiness gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache
from quantagent.data.v7_dataset_builder import build_market_features
from quantagent.data.v7_label_builder import V7_LABEL_HORIZONS
from quantagent.data.v7_quality_gates import V7DataQualityGateConfig, evaluate_data_quality_gates


REQUIRED_ENTITY_COLUMNS: tuple[str, ...] = ("symbol", "trade_date", "available_at")
FORBIDDEN_INFERENCE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


@dataclass(frozen=True)
class V7TrainingDatasetConfig:
    market_panel_path: str
    labels_path: str
    output_path: str
    manifest_path: str | None = None
    fundamentals_root: str | None = None
    valuation_path: str | None = None
    disclosures_path: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple[str, ...] = ()
    horizons: tuple[int, ...] = V7_LABEL_HORIZONS
    min_rows: int = 100
    min_symbols: int = 2
    min_dates: int = 5
    enforce_quality_gates: bool = True
    add_missingness_flags: bool = True
    allow_synthetic_fallback: bool = False
    strict_mode: bool = True
    feature_groups: tuple[str, ...] = ()
    train_end_date: str | None = None
    validation_end_date: str | None = None
    source_name: str = "realdata"
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class V7TrainingDatasetResult:
    dataset: pd.DataFrame
    output_path: Path
    manifest_path: Path
    feature_schema_path: Path
    quality_report: dict[str, object]
    feature_schema: dict[str, object]
    summary: dict[str, object]


def build_v7_training_dataset_artifact(config: V7TrainingDatasetConfig) -> V7TrainingDatasetResult:
    if config.allow_synthetic_fallback:
        raise ValueError(
            "V7 training dataset builder forbids synthetic fallback; set allow_synthetic_fallback=false"
        )
    market = load_table(config.market_panel_path)
    if market is None or market.empty:
        raise ValueError(f"market panel is empty: {config.market_panel_path}")
    labels = load_table(config.labels_path)
    if labels is None or labels.empty:
        raise ValueError(f"labels frame is empty: {config.labels_path}")

    market = _restrict_window(market, config.start_date, config.end_date, config.symbols)
    labels = _restrict_window(labels, config.start_date, config.end_date, config.symbols)
    features = build_market_features(market)

    fundamentals = load_fundamentals_root(config.fundamentals_root) if config.fundamentals_root else pd.DataFrame()
    valuation = load_table(config.valuation_path) if config.valuation_path else pd.DataFrame()
    disclosures = load_table(config.disclosures_path) if config.disclosures_path else pd.DataFrame()

    fundamentals_rows = int(0 if fundamentals is None else len(fundamentals))
    valuation_rows = int(0 if valuation is None else len(valuation))
    disclosure_rows = int(0 if disclosures is None else len(disclosures))

    features = _asof_merge(features, fundamentals, suffix="_fund", add_flag=config.add_missingness_flags, flag_name="missing_fundamentals")
    features = _asof_merge(features, valuation, suffix="_val", add_flag=config.add_missingness_flags, flag_name="missing_valuation")
    features = _asof_merge(features, disclosures, suffix="_disc", add_flag=config.add_missingness_flags, flag_name="missing_disclosures")

    label_columns = [c for c in labels.columns if c.startswith("forward_return_")]
    label_end_columns = [c for c in labels.columns if c.startswith("label_end_")]
    if not label_columns:
        raise ValueError("labels frame is missing forward_return_* columns; rerun build-labels-v7")
    desired_horizons = tuple(h for h in config.horizons if f"forward_return_{h}d" in labels.columns) or tuple(
        int(c.split("_")[2].rstrip("d")) for c in label_columns
    )
    label_keep = ["symbol", "trade_date", *[f"forward_return_{h}d" for h in desired_horizons]]
    label_keep.extend(c for c in label_end_columns if c in labels.columns)
    labels_subset = labels[[c for c in label_keep if c in labels.columns]].copy()

    features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce")
    labels_subset["trade_date"] = pd.to_datetime(labels_subset["trade_date"], errors="coerce")
    dataset = features.merge(labels_subset, on=["symbol", "trade_date"], how="inner")
    dataset = dataset.dropna(subset=["available_at"]).reset_index(drop=True)

    if config.feature_groups:
        from quantagent.data.v7_feature_groups import select_v7_feature_columns

        feature_columns = list(select_v7_feature_columns(dataset, groups=config.feature_groups).selected)
        if not feature_columns:
            feature_columns = _feature_columns(dataset, desired_horizons)
    else:
        feature_columns = _feature_columns(dataset, desired_horizons)
    if not feature_columns:
        raise ValueError("training dataset has no usable feature columns after as-of joins")

    if config.strict_mode:
        _strict_mode_assert(dataset, feature_columns, desired_horizons, config)

    label_column_names = [f"forward_return_{h}d" for h in desired_horizons]
    feature_schema = {
        "feature_columns": feature_columns,
        "label_columns": label_column_names,
        "entity_columns": ["symbol"],
        "timestamp_columns": ["trade_date", "available_at"],
        "forbidden_columns": list(FORBIDDEN_INFERENCE_COLUMNS) + label_column_names + [f"label_end_{h}d" for h in desired_horizons],
        "horizons": list(desired_horizons),
        "available_at_policy": "close-derived features are available from the next trading row; financial joins use available_at <= trade_date",
        "source_name": config.source_name,
    }

    quality = evaluate_data_quality_gates(
        dataset,
        V7DataQualityGateConfig(
            min_rows=config.min_rows,
            min_symbols=config.min_symbols,
            min_dates=config.min_dates,
            require_real_data=config.source_name != "mock",
        ),
    )
    quality_report = quality.to_dict()
    if config.enforce_quality_gates and not quality.passed:
        raise ValueError(f"V7 training dataset quality gates failed: {quality.failures}")

    output_path = _write_table(dataset, Path(config.output_path))
    schema_path = output_path.with_suffix(".feature_schema.json")
    schema_path.write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = Path(config.manifest_path) if config.manifest_path else _default_manifest_path(output_path)
    manifest = build_manifest_for_frame(
        dataset_name="training_dataset",
        vendor="v7_pipeline",
        frame=dataset,
        output_paths=[output_path, schema_path],
        raw_paths=[config.market_panel_path, config.labels_path]
        + ([config.fundamentals_root] if config.fundamentals_root else [])
        + ([config.valuation_path] if config.valuation_path else [])
        + ([config.disclosures_path] if config.disclosures_path else []),
        start_date=config.start_date,
        end_date=config.end_date,
        symbols=config.symbols,
        required_columns=("symbol", "trade_date", "available_at", *label_column_names),
        pit_violation_count=int(quality_report.get("metrics", {}).get("pit_violation_count", 0)),
        warnings=tuple(quality_report.get("failures", ())),
        extra={
            "feature_schema_path": str(schema_path),
            "fundamentals_rows": fundamentals_rows,
            "valuation_rows": valuation_rows,
            "disclosure_rows": disclosure_rows,
            "horizons": list(desired_horizons),
            "feature_column_count": len(feature_columns),
        },
    )
    manifest.write(manifest_path)

    summary = {
        "status": "passed",
        "rows": int(len(dataset)),
        "symbols": int(dataset["symbol"].nunique()) if "symbol" in dataset.columns else 0,
        "dates": int(pd.to_datetime(dataset["trade_date"], errors="coerce").nunique()),
        "feature_count": len(feature_columns),
        "horizons": list(desired_horizons),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "feature_schema_path": str(schema_path),
        "fundamentals_rows": fundamentals_rows,
        "valuation_rows": valuation_rows,
        "disclosure_rows": disclosure_rows,
        "quality_status": manifest.quality_status,
    }
    return V7TrainingDatasetResult(
        dataset=dataset,
        output_path=output_path,
        manifest_path=manifest_path,
        feature_schema_path=schema_path,
        quality_report=quality_report,
        feature_schema=feature_schema,
        summary=summary,
    )


def load_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        fallback = file_path.with_suffix(".csv")
        if fallback.exists():
            return pd.read_csv(fallback)
        return pd.DataFrame()
    if file_path.suffix == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except Exception:
            csv = file_path.with_suffix(".csv")
            if csv.exists():
                return pd.read_csv(csv)
            return pd.DataFrame()
    return pd.read_csv(file_path)


def load_fundamentals_root(root: str | Path) -> pd.DataFrame:
    """Read PIT financial statements out of the V7 fundamentals cache.

    Falls back to direct parquet/csv read if ``root`` is a single file. When
    ``root`` is a directory we use ``FinancialStatementCache`` to read each
    known statement and combine them on ``symbol/available_at``.
    """
    root_path = Path(root)
    if not root_path.exists():
        return pd.DataFrame()
    if root_path.is_file():
        return load_table(root_path)
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(root_path)))
    frames: list[pd.DataFrame] = []
    seen_columns: set[str] = set()
    for statement, file_name in (
        ("income", "income"),
        ("balance_sheet", "balance_sheet"),
        ("cashflow", "cashflow"),
        ("financial_indicator", "financial_indicator"),
    ):
        statement_path = cache._path(statement)  # noqa: SLF001 - intentional: keep one path source
        frame = load_table(statement_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["available_at"] = pd.to_datetime(frame.get("available_at"), errors="coerce")
        prefix = f"{file_name}_"
        rename = {
            column: f"{prefix}{column}"
            for column in frame.columns
            if column not in REQUIRED_ENTITY_COLUMNS
            and not column.startswith(("report_period", "ann_date", "source", "raw_hash", "point_in_time_valid"))
        }
        frame = frame.rename(columns=rename)
        seen_columns.update(frame.columns)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "available_at" not in combined.columns:
        return pd.DataFrame()
    combined = combined.dropna(subset=["symbol", "available_at"]).sort_values(["symbol", "available_at"]).reset_index(drop=True)
    return combined


def _asof_merge(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    suffix: str,
    add_flag: bool,
    flag_name: str,
) -> pd.DataFrame:
    if extra is None or extra.empty or "available_at" not in extra.columns:
        if add_flag:
            base = base.copy()
            base[flag_name] = True
        return base
    left = base.copy()
    left["available_at"] = pd.to_datetime(left["available_at"], errors="coerce")
    right = extra.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], errors="coerce")
    right = right.dropna(subset=["symbol", "available_at"]).sort_values(["symbol", "available_at"])
    left_columns = set(left.columns)
    overlap = [c for c in right.columns if c in left_columns and c not in REQUIRED_ENTITY_COLUMNS]
    right = right.rename(columns={column: f"{column}{suffix}" for column in overlap})
    joined_columns = [c for c in right.columns if c not in ("symbol",)]
    merged_parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in left.sort_values(["symbol", "available_at"]).groupby("symbol", sort=False):
        symbol_extra = right[right["symbol"].astype(str) == str(symbol)]
        if symbol_extra.empty:
            if add_flag:
                symbol_frame = symbol_frame.copy()
                symbol_frame[flag_name] = True
            merged_parts.append(symbol_frame)
            continue
        merged = pd.merge_asof(
            symbol_frame.sort_values("available_at"),
            symbol_extra.drop(columns=["symbol"]).sort_values("available_at"),
            on="available_at",
            direction="backward",
        )
        if add_flag:
            joined_present = [c for c in joined_columns if c in merged.columns and c != "available_at"]
            merged[flag_name] = merged[joined_present].isna().all(axis=1) if joined_present else True
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else left


def _feature_columns(frame: pd.DataFrame, horizons: Iterable[int]) -> list[str]:
    forbidden = set(FORBIDDEN_INFERENCE_COLUMNS)
    forbidden.update(f"forward_return_{h}d" for h in horizons)
    forbidden.update(f"label_end_{h}d" for h in horizons)
    return [
        column
        for column in frame.select_dtypes(include=[np.number, bool]).columns
        if column not in forbidden
    ]


def _restrict_window(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        if start_date:
            data = data[data["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data["trade_date"] <= pd.Timestamp(end_date)]
    if symbols:
        data = data[data["symbol"].astype(str).isin({str(s) for s in symbols})]
    return data.reset_index(drop=True)


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


def _default_manifest_path(output_path: Path) -> Path:
    lake = v7_lake_paths()
    try:
        output_path.resolve().relative_to(lake.root.resolve())
        return lake.manifests / "training_dataset.json"
    except (ValueError, FileNotFoundError, OSError):
        return output_path.with_suffix(".manifest.json")


def _strict_mode_assert(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    horizons: tuple[int, ...],
    config: V7TrainingDatasetConfig,
) -> None:
    """Enforce strict-mode invariants on the gold training dataset.

    Strict mode is the default. It raises ``ValueError`` if any of the
    following holds: labels missing, features empty, PIT columns
    missing, duplicate ``(trade_date, symbol)`` rows, synthetic source
    rows, label columns leaked into the feature set, or train/validation
    date ranges overlap when explicit splits were supplied.
    """
    if dataset is None or dataset.empty:
        raise ValueError("strict mode: training dataset is empty")
    missing_entity = [c for c in REQUIRED_ENTITY_COLUMNS if c not in dataset.columns]
    if missing_entity:
        raise ValueError(f"strict mode: missing PIT/entity columns {missing_entity}")
    if not feature_columns:
        raise ValueError("strict mode: feature set is empty")
    label_cols = [f"forward_return_{h}d" for h in horizons]
    missing_labels = [c for c in label_cols if c not in dataset.columns]
    if missing_labels:
        raise ValueError(f"strict mode: missing forward-return labels {missing_labels}")
    leaked = sorted(set(feature_columns).intersection(set(label_cols)))
    if leaked:
        raise ValueError(f"strict mode: label columns leaked into features {leaked}")
    duplicate_count = int(dataset.duplicated(subset=["trade_date", "symbol"]).sum())
    if duplicate_count:
        raise ValueError(
            f"strict mode: {duplicate_count} duplicate (trade_date, symbol) rows in training dataset"
        )
    for column in ("source", "source_name", "data_source"):
        if column in dataset.columns:
            values = dataset[column].astype(str).str.lower()
            if values.str.contains("mock|synthetic|demo").any():
                raise ValueError("strict mode: synthetic / mock rows present in training dataset")
    if config.train_end_date and config.validation_end_date:
        train_end = pd.Timestamp(config.train_end_date)
        val_end = pd.Timestamp(config.validation_end_date)
        if val_end <= train_end:
            raise ValueError(
                "strict mode: validation_end_date must be strictly after train_end_date"
            )
        in_train = dataset[dataset["trade_date"] <= train_end]
        in_val = dataset[(dataset["trade_date"] > train_end) & (dataset["trade_date"] <= val_end)]
        in_test = dataset[dataset["trade_date"] > val_end]
        for window_name, frame in (("train", in_train), ("validation", in_val), ("test", in_test)):
            if frame.empty:
                raise ValueError(f"strict mode: {window_name} window is empty")


__all__ = [
    "V7TrainingDatasetConfig",
    "V7TrainingDatasetResult",
    "build_v7_training_dataset_artifact",
    "load_table",
    "load_fundamentals_root",
    "FORBIDDEN_INFERENCE_COLUMNS",
    "REQUIRED_ENTITY_COLUMNS",
]
