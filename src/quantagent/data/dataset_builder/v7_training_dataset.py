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
    factor_library: str = "basic"
    synthesized_factors_path: str | None = None
    factor_min_finite_ratio: float = 0.30
    cached_factors_path: str | None = None  # if set, skip compute and load wide parquet
    macro_root: str | None = None
    flow_root: str | None = None
    index_root: str | None = None
    enable_macro: bool = True
    enable_flow: bool = True
    enable_index: bool = True
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
    if config.cached_factors_path:
        features, factor_report = _append_cached_factors(
            features,
            config.cached_factors_path,
            min_finite_ratio=config.factor_min_finite_ratio,
        )
    else:
        features, factor_report = _append_factor_library(
            features,
            market,
            library=config.factor_library,
            synthesized_factors_path=config.synthesized_factors_path,
            min_finite_ratio=config.factor_min_finite_ratio,
        )

    fundamentals = load_fundamentals_root(config.fundamentals_root) if config.fundamentals_root else pd.DataFrame()
    valuation = load_table(config.valuation_path) if config.valuation_path else pd.DataFrame()
    disclosures = load_table(config.disclosures_path) if config.disclosures_path else pd.DataFrame()

    fundamentals_rows = int(0 if fundamentals is None else len(fundamentals))
    valuation_rows = int(0 if valuation is None else len(valuation))
    disclosure_rows = int(0 if disclosures is None else len(disclosures))

    features = _asof_merge(features, fundamentals, suffix="_fund", add_flag=config.add_missingness_flags, flag_name="missing_fundamentals")
    features = _asof_merge(features, valuation, suffix="_val", add_flag=config.add_missingness_flags, flag_name="missing_valuation")
    features = _asof_merge(features, disclosures, suffix="_disc", add_flag=config.add_missingness_flags, flag_name="missing_disclosures")

    macro_report: dict[str, object] = {"status": "skipped"}
    flow_report: dict[str, object] = {"status": "skipped"}
    index_report: dict[str, object] = {"status": "skipped"}
    if config.enable_macro and config.macro_root:
        features, macro_report = _append_macro_features(features, config.macro_root)
    if config.enable_flow and config.flow_root:
        features, flow_report = _append_flow_features(features, config.flow_root)
    if config.enable_index and config.index_root:
        features, index_report = _append_index_features(features, config.index_root)

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
        + ([config.disclosures_path] if config.disclosures_path else [])
        + ([config.synthesized_factors_path] if config.synthesized_factors_path else []),
        start_date=config.start_date,
        end_date=config.end_date,
        symbols=config.symbols or tuple(sorted(dataset["symbol"].astype(str).dropna().unique())),
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
            "factor_library": config.factor_library,
            "factor_report": factor_report,
            "macro_report": macro_report,
            "flow_report": flow_report,
            "index_report": index_report,
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
        "factor_library": config.factor_library,
        "factor_report": factor_report,
        "macro_report": macro_report,
        "flow_report": flow_report,
        "index_report": index_report,
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
        except Exception as exc:
            csv = file_path.with_suffix(".csv")
            if csv.exists():
                return pd.read_csv(csv)
            try:
                import polars as pl
            except ImportError:
                raise RuntimeError(
                    f"failed to read parquet table {file_path}; install pyarrow/fastparquet "
                    "or provide a sibling CSV fallback"
                ) from exc
            try:
                return pl.read_parquet(str(file_path)).to_pandas()
            except Exception as polars_exc:
                raise RuntimeError(
                    f"failed to read parquet table {file_path}; install pyarrow/fastparquet "
                    "or provide a sibling CSV fallback; file may also be corrupt or not a parquet file"
                ) from polars_exc
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

    # Both sides must align on the same key types for pandas merge_asof(by=).
    left["symbol"] = left["symbol"].astype("string")
    right["symbol"] = right["symbol"].astype("string")

    # Vectorised path: single C-level merge_asof with by="symbol" replaces the
    # 3872-iteration Python loop + concat that used to peak at ~30 GB on the
    # full A-share universe. Both frames must be sorted on the `on` key only;
    # the `by` parameter handles per-symbol grouping internally.
    left_sorted = left.sort_values("available_at").reset_index(drop=True)
    right_sorted = right.sort_values("available_at").reset_index(drop=True)
    joined_columns = [c for c in right_sorted.columns if c not in ("symbol",)]
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="available_at",
        by="symbol",
        direction="backward",
    )
    if add_flag:
        joined_present = [c for c in joined_columns
                          if c in merged.columns and c != "available_at"]
        merged[flag_name] = (
            merged[joined_present].isna().all(axis=1) if joined_present else True
        )
    return merged


def _feature_columns(frame: pd.DataFrame, horizons: Iterable[int]) -> list[str]:
    forbidden = set(FORBIDDEN_INFERENCE_COLUMNS)
    forbidden.update(f"forward_return_{h}d" for h in horizons)
    forbidden.update(f"label_end_{h}d" for h in horizons)
    selected: list[str] = []
    for column in frame.select_dtypes(include=[np.number, bool]).columns:
        if column in forbidden:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).notna()
        if not finite.any():
            continue
        if values[finite].nunique(dropna=True) <= 1:
            continue
        selected.append(column)
    return selected


def _append_cached_factors(
    features: pd.DataFrame,
    cached_factors_path: str,
    *,
    min_finite_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a pre-materialised wide factor parquet and merge it onto features.

    Companion to ``qa materialize-alpha181-v7``. Replaces the in-process
    factor compute → long → pivot → merge chain that peaks at 40 GB on
    1500-symbol panels. The streaming-read here keeps peak RAM bounded
    by the factor parquet's row-group size (~1-2 GB typical).
    """
    factor_path = Path(cached_factors_path)
    if not factor_path.exists():
        return features, {"status": "missing", "cached_factors_path": str(factor_path),
                          "columns_added": 0}
    wide = load_table(factor_path)
    if wide.empty:
        return features, {"status": "empty", "cached_factors_path": str(factor_path),
                          "columns_added": 0}
    if "trade_date" not in wide.columns or "symbol" not in wide.columns:
        return features, {"status": "schema_mismatch", "cached_factors_path": str(factor_path),
                          "columns_added": 0,
                          "warnings": ["wide factor parquet missing trade_date or symbol"]}
    wide["trade_date"] = pd.to_datetime(wide["trade_date"], errors="coerce")

    kept: list[str] = []
    dropped: list[str] = []
    for column in [c for c in wide.columns if c not in {"trade_date", "symbol"}]:
        values = pd.to_numeric(wide[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_ratio = float(values.notna().mean())
        if finite_ratio >= min_finite_ratio and values.nunique(dropna=True) > 1:
            wide[column] = values
            kept.append(column)
        else:
            dropped.append(column)
    wide = wide[["trade_date", "symbol", *kept]]
    merged = features.merge(wide, on=["trade_date", "symbol"], how="left")
    return merged, {
        "status": "passed",
        "cached_factors_path": str(factor_path),
        "columns_added": len(kept),
        "columns_dropped": len(dropped),
        "min_finite_ratio": min_finite_ratio,
    }


def _append_factor_library(
    features: pd.DataFrame,
    market: pd.DataFrame,
    *,
    library: str,
    synthesized_factors_path: str | None,
    min_finite_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = library.strip().lower()
    if selected in {"", "basic", "none", "off"}:
        return features, {"status": "skipped", "library": selected or "basic"}
    if selected == "alpha101":
        from quantagent.factors.alpha101 import compute_alpha101

        factors = compute_alpha101(market)
    elif selected == "alpha181":
        from quantagent.factors.alpha181 import compute_alpha181

        factors = compute_alpha181(market, synthesized_definitions_path=synthesized_factors_path)
    elif selected in {"cicc80", "cicc_ashare80"}:
        from quantagent.factors.cicc_ashare80 import compute_cicc_ashare80_factors

        factors = compute_cicc_ashare80_factors(market)
    else:
        raise ValueError("factor_library must be basic, alpha101, alpha181, or cicc_ashare80")
    if factors.empty:
        return features, {"status": "empty", "library": selected, "columns_added": 0}
    wide = factors.pivot_table(
        index=["trade_date", "symbol"],
        columns="factor_name",
        values="factor_value",
        aggfunc="last",
    ).reset_index()
    wide.columns = [str(column) for column in wide.columns]
    kept = []
    dropped = []
    for column in [c for c in wide.columns if c not in {"trade_date", "symbol"}]:
        values = pd.to_numeric(wide[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_ratio = float(values.notna().mean())
        if finite_ratio >= min_finite_ratio and values.nunique(dropna=True) > 1:
            wide[column] = values
            kept.append(column)
        else:
            dropped.append(column)
    wide = wide[["trade_date", "symbol", *kept]]
    merged = features.merge(wide, on=["trade_date", "symbol"], how="left")
    return merged, {
        "status": "passed",
        "library": selected,
        "columns_added": len(kept),
        "columns_dropped": len(dropped),
        "min_finite_ratio": min_finite_ratio,
        "synthesized_factors_path": synthesized_factors_path,
    }


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
            try:
                import polars as pl
            except ImportError:
                path = path.with_suffix(".csv")
            else:
                try:
                    pl.from_pandas(frame).write_parquet(str(path))
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


def _asof_merge_timeseries(
    base: pd.DataFrame,
    extra_wide: pd.DataFrame,
    prefix: str,
) -> tuple[pd.DataFrame, int]:
    """Broadcast a pure time series (indexed by available_at) onto every symbol row.

    Returns the new frame and the count of columns added.
    """
    if extra_wide is None or extra_wide.empty or "available_at" not in extra_wide.columns:
        return base, 0
    left = base.copy()
    left["available_at"] = pd.to_datetime(left["available_at"], errors="coerce")
    right = extra_wide.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], errors="coerce")
    right = right.dropna(subset=["available_at"]).sort_values("available_at")
    new_columns = [c for c in right.columns if c != "available_at"]
    if not new_columns:
        return base, 0
    renamed = {c: f"{prefix}{c}" if not c.startswith(prefix) else c for c in new_columns}
    right = right.rename(columns=renamed)
    new_columns = [renamed[c] for c in new_columns]
    left_sorted = left.sort_values("available_at").reset_index(drop=True)
    merged = pd.merge_asof(left_sorted, right, on="available_at", direction="backward")
    # Filter to numeric, finite columns only.
    kept: list[str] = []
    for column in new_columns:
        if column not in merged.columns:
            continue
        values = pd.to_numeric(merged[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().any() and values.nunique(dropna=True) > 1:
            merged[column] = values
            kept.append(column)
        else:
            merged = merged.drop(columns=[column])
    return merged, len(kept)


def _load_macro_wide(macro_root: str | Path) -> pd.DataFrame:
    """Read every macro PIT parquet under ``macro_root`` and merge to one wide frame."""
    root = Path(macro_root)
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for spec in ("yield_curve", "shibor", "repo", "central_bank_omo",
                 "aggregate_financing", "money_supply", "cpi", "ppi"):
        path = root / f"{spec}.parquet"
        frame = load_table(path)
        if frame.empty:
            continue
        if spec == "yield_curve":
            wide = frame.pivot_table(index="available_at", columns="maturity",
                                     values="yield_pct", aggfunc="last")
            wide.columns = [f"macro_yield_{str(c).lower()}" for c in wide.columns]
        elif spec == "shibor":
            wide = frame.pivot_table(index="available_at", columns="tenor",
                                     values="rate_pct", aggfunc="last")
            wide.columns = [f"macro_shibor_{str(c).lower().replace('/', '')}" for c in wide.columns]
        elif spec == "repo":
            wide = frame.pivot_table(index="available_at", columns="tenor",
                                     values="rate_pct", aggfunc="last")
            wide.columns = [f"macro_repo_{str(c).lower()}" for c in wide.columns]
        elif spec == "central_bank_omo":
            wide = frame.set_index("available_at")[
                [c for c in ("inject_amount_cny", "expire_amount_cny", "net_amount_cny")
                 if c in frame.columns]
            ]
            wide.columns = [f"macro_omo_{c.replace('_cny', '')}" for c in wide.columns]
        elif spec == "aggregate_financing":
            wide = frame.set_index("available_at")[["aggregate_financing_cny"]]
            wide.columns = ["macro_afre"]
        elif spec == "money_supply":
            wide = frame.set_index("available_at")[
                [c for c in ("m0_cny", "m1_cny", "m2_cny") if c in frame.columns]
            ]
            wide.columns = [f"macro_{c.replace('_cny', '')}" for c in wide.columns]
        elif spec == "cpi":
            wide = frame.set_index("available_at")[["cpi_yoy_pct"]]
            wide.columns = ["macro_cpi_yoy"]
        elif spec == "ppi":
            wide = frame.set_index("available_at")[["ppi_yoy_pct"]]
            wide.columns = ["macro_ppi_yoy"]
        else:
            continue
        wide = wide.reset_index()
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for piece in frames[1:]:
        combined = combined.merge(piece, on="available_at", how="outer")
    combined = combined.sort_values("available_at").reset_index(drop=True)
    return combined


def _load_flow_wide(flow_root: str | Path) -> pd.DataFrame:
    root = Path(flow_root)
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    nb_path = root / "northbound_flow.parquet"
    nb = load_table(nb_path)
    if not nb.empty:
        wide = nb.pivot_table(index="available_at", columns="channel",
                              values="net_inflow_cny", aggfunc="last")
        wide.columns = [f"flow_{str(c).lower()}" for c in wide.columns]
        frames.append(wide.reset_index())
    mb_path = root / "margin_balance.parquet"
    mb = load_table(mb_path)
    if not mb.empty:
        wide = mb.pivot_table(index="available_at", columns="market",
                              values="margin_balance_cny", aggfunc="last")
        wide.columns = [f"flow_margin_{str(c).lower()}" for c in wide.columns]
        frames.append(wide.reset_index())
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for piece in frames[1:]:
        combined = combined.merge(piece, on="available_at", how="outer")
    return combined.sort_values("available_at").reset_index(drop=True)


def _load_index_wide(index_root: str | Path) -> pd.DataFrame:
    root = Path(index_root)
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for table in ("equity_index", "commodity_main", "treasury_future"):
        frame = load_table(root / f"{table}.parquet")
        if frame.empty:
            continue
        for value_col, suffix in (("close", "close"),):
            if value_col not in frame.columns:
                continue
            wide = frame.pivot_table(index="available_at", columns="label",
                                     values=value_col, aggfunc="last")
            wide.columns = [f"idx_{str(c).lower()}_{suffix}" for c in wide.columns]
            wide = wide.reset_index()
            # 5-day log return as a more stationary derived feature
            for col in [c for c in wide.columns if c != "available_at"]:
                series = pd.to_numeric(wide[col], errors="coerce")
                wide[f"{col[:-len('_close')]}_ret5"] = np.log(series / series.shift(5))
            frames.append(wide)
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for piece in frames[1:]:
        combined = combined.merge(piece, on="available_at", how="outer")
    return combined.sort_values("available_at").reset_index(drop=True)


def _append_macro_features(features: pd.DataFrame, macro_root: str) -> tuple[pd.DataFrame, dict[str, object]]:
    wide = _load_macro_wide(macro_root)
    if wide.empty:
        return features, {"status": "empty", "root": str(macro_root), "columns_added": 0}
    merged, added = _asof_merge_timeseries(features, wide, prefix="macro_")
    return merged, {"status": "passed", "root": str(macro_root), "columns_added": int(added)}


def _append_flow_features(features: pd.DataFrame, flow_root: str) -> tuple[pd.DataFrame, dict[str, object]]:
    wide = _load_flow_wide(flow_root)
    if wide.empty:
        return features, {"status": "empty", "root": str(flow_root), "columns_added": 0}
    merged, added = _asof_merge_timeseries(features, wide, prefix="flow_")
    return merged, {"status": "passed", "root": str(flow_root), "columns_added": int(added)}


def _append_index_features(features: pd.DataFrame, index_root: str) -> tuple[pd.DataFrame, dict[str, object]]:
    wide = _load_index_wide(index_root)
    if wide.empty:
        return features, {"status": "empty", "root": str(index_root), "columns_added": 0}
    merged, added = _asof_merge_timeseries(features, wide, prefix="idx_")
    return merged, {"status": "passed", "root": str(index_root), "columns_added": int(added)}


__all__ = [
    "V7TrainingDatasetConfig",
    "V7TrainingDatasetResult",
    "build_v7_training_dataset_artifact",
    "load_table",
    "load_fundamentals_root",
    "FORBIDDEN_INFERENCE_COLUMNS",
    "REQUIRED_ENTITY_COLUMNS",
]
