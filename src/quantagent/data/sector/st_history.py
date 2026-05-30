"""ST / *ST flag table builder with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.io import read_frame
from quantagent.data.manifest import build_manifest_for_frame, utc_now_iso


ST_FLAG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "is_st",
    "st_known",
    "block_weight",
    "st_source",
    "source_version",
    "effective_date",
    "fetched_at",
    "available_at",
    "coverage_status",
)


def _to_naive_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True).tz_convert(None)


def _to_naive_series(values: object) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


@dataclass(frozen=True)
class StFlagConfig:
    symbols: tuple[str, ...] = ()
    as_of_date: str | None = None
    fetched_at: str | None = None
    source: str = "local_st_mapping"
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    st_block_weight: float = 0.90
    suspended_block_weight: float = 1.00
    unknown_st_block_weight: float = 0.00
    min_st_coverage: float = 0.85


@dataclass(frozen=True)
class StFlagResult:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


def _normalise_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))


def _coerce_st_source(frame: pd.DataFrame, *, config: StFlagConfig) -> pd.DataFrame:
    data = frame.copy()
    if "symbol" not in data.columns:
        raise ValueError("ST mapping source must contain a 'symbol' column")
    if "is_st" not in data.columns:
        for candidate in ("st", "is_ST", "risk_warning"):
            if candidate in data.columns:
                data = data.rename(columns={candidate: "is_st"})
                break
    if "is_st" not in data.columns:
        data["is_st"] = True
    if "st_known" not in data.columns:
        data["st_known"] = True
    fetched_at = config.fetched_at or utc_now_iso()
    as_of = _to_naive_timestamp(config.as_of_date) if config.as_of_date else pd.Timestamp.now(tz=None).normalize()
    if "st_source" not in data.columns:
        data["st_source"] = config.source
    if "source_version" not in data.columns:
        data["source_version"] = config.source_version
    if "effective_date" not in data.columns:
        data["effective_date"] = data["available_at"] if "available_at" in data.columns else as_of
    if "fetched_at" not in data.columns:
        data["fetched_at"] = fetched_at
    if "available_at" not in data.columns:
        data["available_at"] = as_of
    if "coverage_status" not in data.columns:
        data["coverage_status"] = "pit_historical"
    data["symbol"] = data["symbol"].astype(str).str.strip()
    data["is_st"] = data["is_st"].fillna(False).astype(bool)
    data["st_known"] = data["st_known"].fillna(True).astype(bool)
    if "block_weight" not in data.columns:
        data["block_weight"] = data["is_st"].map(lambda v: float(config.st_block_weight) if bool(v) else 0.0)
    data["block_weight"] = pd.to_numeric(data["block_weight"], errors="coerce").fillna(0.0)
    data["st_source"] = data["st_source"].astype(str)
    data["source_version"] = data["source_version"].astype(str)
    data["effective_date"] = _to_naive_series(data["effective_date"])
    data["fetched_at"] = _to_naive_series(data["fetched_at"])
    data["available_at"] = _to_naive_series(data["available_at"])
    data["coverage_status"] = data["coverage_status"].astype(str)
    return data.dropna(subset=["symbol", "available_at"]).reset_index(drop=True)


def _missing_rows(symbols: tuple[str, ...], present: set[str], *, config: StFlagConfig) -> pd.DataFrame:
    missing = [symbol for symbol in symbols if symbol not in present]
    fetched_at = _to_naive_timestamp(config.fetched_at or utc_now_iso())
    available_at = _to_naive_timestamp(config.as_of_date) if config.as_of_date else fetched_at
    return pd.DataFrame(
        {
            "symbol": missing,
            "is_st": False,
            "st_known": False,
            "block_weight": float(config.unknown_st_block_weight),
            "st_source": "unresolved",
            "source_version": "none",
            "effective_date": available_at,
            "fetched_at": fetched_at,
            "available_at": available_at,
            "coverage_status": "missing",
        }
    )


def coverage_report_st(frame: pd.DataFrame, *, symbols: Iterable[str] = ()) -> dict[str, object]:
    expected = _normalise_symbols(symbols)
    total = len(expected) if expected else int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0
    known_mask = frame.get("st_known", pd.Series(dtype=bool)).fillna(False).astype(bool) if not frame.empty else pd.Series(dtype=bool)
    covered = int(frame.loc[known_mask, "symbol"].nunique()) if not frame.empty and "symbol" in frame.columns else 0
    st_count = int(frame.loc[frame.get("is_st", pd.Series(dtype=bool)).fillna(False), "symbol"].nunique()) if not frame.empty and "symbol" in frame.columns else 0
    unknown = int(max(total - covered, 0))
    return {
        "total_expected_symbols": int(total),
        "covered_symbols": int(covered),
        "missing_symbols": unknown,
        "coverage_rate": float(covered / total) if total else 0.0,
        "unknown_st_rate": float(unknown / total) if total else 0.0,
        "st_symbol_count": int(st_count),
    }


def validate_st_table(frame: pd.DataFrame, *, symbols: Iterable[str] = ()) -> dict[str, object]:
    missing_cols = [c for c in ST_FLAG_REQUIRED_COLUMNS if c not in frame.columns]
    duplicate_rows = int(frame.duplicated(subset=["symbol", "available_at"]).sum()) if {"symbol", "available_at"}.issubset(frame.columns) else 0
    pit_violations = 0
    effective_after_available = 0
    if {"available_at", "fetched_at"}.issubset(frame.columns):
        pit_violations = int((_to_naive_series(frame["available_at"]) > _to_naive_series(frame["fetched_at"])).fillna(False).sum())
    if {"effective_date", "available_at"}.issubset(frame.columns):
        effective_after_available = int((_to_naive_series(frame["effective_date"]) > _to_naive_series(frame["available_at"])).fillna(False).sum())
    expected = set(_normalise_symbols(symbols))
    missing_expected = sorted(expected - set(frame["symbol"].astype(str))) if expected and "symbol" in frame.columns else []
    status = "passed" if not missing_cols and not duplicate_rows and not pit_violations and not effective_after_available and not missing_expected else "failed"
    return {
        "status": status,
        "row_count": int(len(frame)),
        "missing_columns": missing_cols,
        "duplicate_symbol_available_at_count": duplicate_rows,
        "pit_violation_count": pit_violations,
        "effective_after_available_count": effective_after_available,
        "missing_expected_symbols": missing_expected,
    }


def st_coverage_gate(frame: pd.DataFrame, *, symbols: Iterable[str] = (), config: StFlagConfig | None = None) -> dict[str, object]:
    cfg = config or StFlagConfig()
    coverage = coverage_report_st(frame, symbols=symbols)
    validation = validate_st_table(frame, symbols=symbols)
    reasons: list[str] = []
    if validation["status"] != "passed":
        reasons.append("st_validation_failed")
    if float(coverage["coverage_rate"]) < float(cfg.min_st_coverage):
        reasons.append("st_coverage_below_threshold")
    usable_for_risk_filter = not reasons
    return {
        "st_usable_for_risk_filter": bool(usable_for_risk_filter),
        "reason": "passed" if usable_for_risk_filter else ",".join(reasons),
        "policy": {
            "st_block_weight": float(cfg.st_block_weight),
            "suspended_block_weight": float(cfg.suspended_block_weight),
            "unknown_st_block_weight": float(cfg.unknown_st_block_weight),
            "suspended_source": "derived separately from OHLCV zero volume/amount; not part of ST flag table",
        },
        "thresholds": {"st_coverage": float(cfg.min_st_coverage)},
        "observed": {
            "st_coverage": float(coverage["coverage_rate"]),
            "unknown_st_rate": float(coverage["unknown_st_rate"]),
        },
    }


class StFlagBuilder:
    def __init__(self, config: StFlagConfig | None = None) -> None:
        self.config = config or StFlagConfig()

    def build(self, source_frame: pd.DataFrame | None = None) -> StFlagResult:
        symbols = _normalise_symbols(self.config.symbols)
        if source_frame is not None and not source_frame.empty:
            data = _coerce_st_source(source_frame, config=self.config)
            as_of = _to_naive_timestamp(self.config.as_of_date) if self.config.as_of_date else pd.Timestamp.now(tz=None).normalize()
            data = data[data["available_at"] <= as_of].sort_values(["symbol", "available_at", "fetched_at"])
            data = data.drop_duplicates(["symbol"], keep="last").reset_index(drop=True)
        else:
            data = pd.DataFrame(columns=ST_FLAG_REQUIRED_COLUMNS)
        if symbols:
            data = data[data["symbol"].isin(symbols)].copy()
            missing = _missing_rows(symbols, set(data["symbol"].astype(str)), config=self.config)
            data = pd.concat([data, missing], ignore_index=True)
        for col in ST_FLAG_REQUIRED_COLUMNS:
            if col not in data.columns:
                data[col] = pd.NA
        data = data[list(ST_FLAG_REQUIRED_COLUMNS)].sort_values("symbol").reset_index(drop=True)
        coverage = coverage_report_st(data, symbols=symbols)
        coverage["gate"] = st_coverage_gate(data, symbols=symbols, config=self.config)
        return StFlagResult(
            frame=data,
            coverage=coverage,
            validation=validate_st_table(data, symbols=symbols),
        )

    def build_from_path(self, path: str | Path) -> StFlagResult:
        return self.build(read_frame(path))

    def write(self, result: StFlagResult) -> StFlagResult:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "st_flags"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "st_flags.parquet"
        coverage_json = out_dir / "coverage_report.json"
        validation_json = out_dir / "validation_report.json"
        result.frame.to_parquet(output, index=False)
        import json
        coverage_json.write_text(json.dumps(result.coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        validation_json.write_text(json.dumps(result.validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        manifest = build_manifest_for_frame(
            dataset_name="st_flags",
            vendor="local",
            frame=result.frame,
            output_paths=[output],
            symbols=self.config.symbols,
            required_columns=ST_FLAG_REQUIRED_COLUMNS,
            pit_violation_count=int(result.validation.get("pit_violation_count", 0)),
            extra={"coverage_report": result.coverage, "validation_report": result.validation},
        )
        manifest.write(root / "manifests" / "st_flags.json")
        paths = {
            "st_flags": str(output),
            "coverage_report": str(coverage_json),
            "validation_report": str(validation_json),
            "manifest": str(root / "manifests" / "st_flags.json"),
        }
        return StFlagResult(result.frame, result.coverage, result.validation, paths)


__all__ = [
    "ST_FLAG_REQUIRED_COLUMNS",
    "StFlagBuilder",
    "StFlagConfig",
    "StFlagResult",
    "coverage_report_st",
    "st_coverage_gate",
    "validate_st_table",
]
