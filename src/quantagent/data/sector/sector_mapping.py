"""PIT-safe sector mapping data layer.

This module materialises ``silver/sector_map/sector_map.parquet`` as a
data product, not as a portfolio signal. Current-snapshot industry
classifications are allowed for live/current diagnostics, but historical
OOS joins must only use rows whose ``available_at`` is no later than the
prediction date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.io import read_frame
from quantagent.data.manifest import build_manifest_for_frame, utc_now_iso


SECTOR_MAP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "sector_level_1",
    "sector_level_2",
    "source",
    "source_version",
    "effective_date",
    "fetched_at",
    "available_at",
    "coverage_status",
)

VALID_COVERAGE_STATUS: frozenset[str] = frozenset(
    {"pit_historical", "current_snapshot", "missing"}
)

SOURCE_PRIORITY: tuple[str, ...] = (
    "manual_vendor_sector",
    "exchange_sector",
    "csrc_sector",
    "board_proxy",
    "unresolved",
)

BOARD_PROXY_SOURCE = "board_proxy"


def _to_naive_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True).tz_convert(None)


def _to_naive_series(values: object) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


@dataclass(frozen=True)
class SectorMapConfig:
    symbols: tuple[str, ...] = ()
    as_of_date: str | None = None
    fetched_at: str | None = None
    source: str = "manual_vendor_sector"
    source_version: str = "unknown"
    coverage_status: str = "pit_historical"
    output_root: str | Path = "runtime/data/v7"
    min_level_1_coverage: float = 0.85
    min_level_2_coverage: float = 0.70
    max_unknown_rate: float = 0.15
    max_stale_available_at_rate: float = 0.05
    max_staleness_days: int = 1095


@dataclass(frozen=True)
class SectorMapResult:
    frame: pd.DataFrame
    coverage: dict[str, object] = field(default_factory=dict)
    missing_symbols: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_symbols: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_priority: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: dict[str, object] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)


def _normalise_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))


def source_priority_rank(source: object) -> int:
    text = str(source or "unresolved")
    for idx, prefix in enumerate(SOURCE_PRIORITY):
        if text == prefix or text.startswith(f"{prefix}:"):
            return idx
    return len(SOURCE_PRIORITY)


def board_proxy_rows(symbols: Iterable[str], *, as_of_date: str | None = None, fetched_at: str | None = None) -> pd.DataFrame:
    """Create market-segment fallback rows.

    These rows are explicitly labelled ``board_proxy``. They are not an
    industry classification and must not be treated as a real sector in
    optimization.
    """
    from quantagent.diagnostics.stratified_ic import board_of

    symbols_tuple = _normalise_symbols(symbols)
    if not symbols_tuple:
        return pd.DataFrame(columns=SECTOR_MAP_REQUIRED_COLUMNS)
    fetched = _to_naive_timestamp(fetched_at or utc_now_iso())
    available = _to_naive_timestamp(as_of_date) if as_of_date else fetched
    return pd.DataFrame(
        {
            "symbol": symbols_tuple,
            "sector_level_1": [board_of(symbol) for symbol in symbols_tuple],
            "sector_level_2": [board_of(symbol) for symbol in symbols_tuple],
            "source": BOARD_PROXY_SOURCE,
            "source_version": "symbol_prefix_v1",
            "effective_date": available,
            "fetched_at": fetched,
            "available_at": available,
            "coverage_status": "current_snapshot",
        }
    )


def _coerce_source_frame(
    frame: pd.DataFrame,
    *,
    config: SectorMapConfig,
) -> pd.DataFrame:
    data = frame.copy()
    if "symbol" not in data.columns:
        raise ValueError("sector mapping source must contain a 'symbol' column")

    rename = {}
    if "sector_level_1" not in data.columns:
        for candidate in ("industry", "sector", "sector_name", "申万一级行业", "行业"):
            if candidate in data.columns:
                rename[candidate] = "sector_level_1"
                break
    if "sector_level_2" not in data.columns:
        for candidate in ("sub_industry", "industry_level_2", "申万二级行业", "细分行业"):
            if candidate in data.columns:
                rename[candidate] = "sector_level_2"
                break
    if rename:
        data = data.rename(columns=rename)

    fetched_at = config.fetched_at or utc_now_iso()
    as_of = _to_naive_timestamp(config.as_of_date) if config.as_of_date else pd.Timestamp.now(tz=None).normalize()
    if "sector_level_1" not in data.columns:
        data["sector_level_1"] = pd.NA
    if "sector_level_2" not in data.columns:
        data["sector_level_2"] = data["sector_level_1"]
    if "source" not in data.columns:
        data["source"] = config.source
    if "source_version" not in data.columns:
        data["source_version"] = config.source_version
    if "effective_date" not in data.columns:
        data["effective_date"] = data["available_at"] if "available_at" in data.columns else as_of
    if "fetched_at" not in data.columns:
        data["fetched_at"] = fetched_at
    if "available_at" not in data.columns:
        data["available_at"] = as_of
    if "coverage_status" not in data.columns:
        data["coverage_status"] = config.coverage_status

    data["symbol"] = data["symbol"].astype(str).str.strip()
    data["sector_level_1"] = data["sector_level_1"].astype("string")
    data["sector_level_2"] = data["sector_level_2"].astype("string")
    data["source"] = data["source"].astype(str)
    data["source_version"] = data["source_version"].astype(str)
    data["effective_date"] = _to_naive_series(data["effective_date"])
    data["fetched_at"] = _to_naive_series(data["fetched_at"])
    data["available_at"] = _to_naive_series(data["available_at"])
    data["coverage_status"] = data["coverage_status"].astype(str)
    missing_l1 = data["sector_level_1"].isna() | (data["sector_level_1"].astype(str).str.strip() == "")
    data.loc[missing_l1, "coverage_status"] = "missing"
    data = data.dropna(subset=["symbol", "available_at"]).reset_index(drop=True)
    return data


def normalize_sector_source(
    frame: pd.DataFrame,
    *,
    source: str = "manual_vendor_sector",
    source_version: str = "unknown",
    as_of_date: str | None = None,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    """Normalize one vendor/manual source to canonical bronze schema."""
    config = SectorMapConfig(
        source=source,
        source_version=source_version,
        as_of_date=as_of_date,
        fetched_at=fetched_at,
    )
    data = _coerce_source_frame(frame, config=config)
    return data[list(SECTOR_MAP_REQUIRED_COLUMNS)].reset_index(drop=True)


def _select_latest_pit_rows(data: pd.DataFrame, *, as_of_date: str | None) -> pd.DataFrame:
    if data.empty:
        return data
    as_of = _to_naive_timestamp(as_of_date) if as_of_date else pd.Timestamp.now(tz=None).normalize()
    eligible = data[data["available_at"] <= as_of].copy()
    if eligible.empty:
        return pd.DataFrame(columns=data.columns)
    eligible["_source_priority"] = eligible["source"].map(source_priority_rank)
    eligible = eligible.sort_values(
        ["symbol", "available_at", "_source_priority", "fetched_at"],
        ascending=[True, True, False, True],
    )
    return eligible.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def _missing_rows(symbols: tuple[str, ...], present: set[str], *, config: SectorMapConfig) -> pd.DataFrame:
    missing = [symbol for symbol in symbols if symbol not in present]
    fetched_at = _to_naive_timestamp(config.fetched_at or utc_now_iso())
    available_at = _to_naive_timestamp(config.as_of_date) if config.as_of_date else fetched_at
    return pd.DataFrame(
        {
            "symbol": missing,
            "sector_level_1": pd.NA,
            "sector_level_2": pd.NA,
            "source": "unresolved",
            "source_version": "none",
            "effective_date": available_at,
            "fetched_at": fetched_at,
            "available_at": available_at,
            "coverage_status": "missing",
        }
    )


def duplicate_symbol_report(source_frame: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate source rows by ``(symbol, available_at)``."""
    if source_frame.empty or not {"symbol", "available_at"}.issubset(source_frame.columns):
        return pd.DataFrame(columns=["symbol", "available_at", "row_count"])
    work = source_frame.copy()
    work["available_at"] = _to_naive_series(work["available_at"])
    dup = (
        work.groupby(["symbol", "available_at"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    return dup[dup["row_count"] > 1].reset_index(drop=True)


def source_conflict_report(source_frame: pd.DataFrame) -> pd.DataFrame:
    """Report same-symbol/same-available_at conflicting sector labels."""
    if source_frame.empty or not {"symbol", "available_at", "sector_level_1", "sector_level_2"}.issubset(source_frame.columns):
        return pd.DataFrame(columns=["symbol", "available_at", "sector_level_1_count", "sector_level_2_count"])
    work = source_frame.copy()
    work["available_at"] = _to_naive_series(work["available_at"])
    out = (
        work.groupby(["symbol", "available_at"], dropna=False)
        .agg(
            sector_level_1_count=("sector_level_1", lambda s: int(s.dropna().astype(str).nunique())),
            sector_level_2_count=("sector_level_2", lambda s: int(s.dropna().astype(str).nunique())),
        )
        .reset_index()
    )
    return out[(out["sector_level_1_count"] > 1) | (out["sector_level_2_count"] > 1)].reset_index(drop=True)


def source_priority_report(source_frame: pd.DataFrame) -> pd.DataFrame:
    if source_frame.empty or "source" not in source_frame.columns:
        return pd.DataFrame(columns=["source", "source_priority", "row_count", "symbol_count"])
    work = source_frame.copy()
    work["source"] = work["source"].astype(str)
    out = (
        work.groupby("source", dropna=False)
        .agg(row_count=("symbol", "size"), symbol_count=("symbol", "nunique"))
        .reset_index()
    )
    out["source_priority"] = out["source"].map(source_priority_rank)
    return out.sort_values(["source_priority", "source"]).reset_index(drop=True)


def sector_distribution_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["sector_level_1", "row_count", "symbol_count", "share"])
    covered = frame[frame["coverage_status"] != "missing"].copy()
    if covered.empty:
        return pd.DataFrame(columns=["sector_level_1", "row_count", "symbol_count", "share"])
    out = (
        covered.groupby("sector_level_1", dropna=False)
        .agg(row_count=("symbol", "size"), symbol_count=("symbol", "nunique"))
        .reset_index()
        .sort_values(["symbol_count", "sector_level_1"], ascending=[False, True])
    )
    total = max(int(covered["symbol"].nunique()), 1)
    out["share"] = out["symbol_count"] / total
    return out.reset_index(drop=True)


def coverage_report(frame: pd.DataFrame, *, symbols: Iterable[str] = ()) -> dict[str, object]:
    expected = _normalise_symbols(symbols)
    total_expected = len(expected) if expected else int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0
    status = frame.get("coverage_status", pd.Series(dtype=str)).astype(str) if not frame.empty else pd.Series(dtype=str)
    is_board_proxy = frame.get("source", pd.Series(dtype=str)).astype(str).str.startswith(BOARD_PROXY_SOURCE) if not frame.empty else pd.Series(dtype=bool)
    level1_known = frame.get("sector_level_1", pd.Series(dtype=object)).notna() if not frame.empty else pd.Series(dtype=bool)
    level2_known = frame.get("sector_level_2", pd.Series(dtype=object)).notna() if not frame.empty else pd.Series(dtype=bool)
    covered = frame[(status != "missing") & ~is_board_proxy] if not frame.empty else frame
    board_proxy_symbols = int(frame.loc[is_board_proxy, "symbol"].nunique()) if not frame.empty else 0
    covered_symbols = int(covered["symbol"].nunique()) if not covered.empty else 0
    level_1_covered_symbols = int(frame.loc[(status != "missing") & ~is_board_proxy & level1_known, "symbol"].nunique()) if not frame.empty else 0
    level_2_covered_symbols = int(frame.loc[(status != "missing") & ~is_board_proxy & level2_known, "symbol"].nunique()) if not frame.empty else 0
    unknown_symbols = int(frame.loc[(status == "missing") | is_board_proxy, "symbol"].nunique()) if not frame.empty else 0
    status_counts = frame["coverage_status"].value_counts(dropna=False).to_dict() if "coverage_status" in frame.columns else {}
    distribution = sector_distribution_report(frame)
    concentration_warning = bool(not distribution.empty and float(distribution["share"].max()) > 0.60 and covered_symbols >= 20)
    return {
        "total_expected_symbols": int(total_expected),
        "covered_symbols": int(covered_symbols),
        "missing_symbols": int(max(total_expected - covered_symbols, 0)),
        "coverage_rate": float(covered_symbols / total_expected) if total_expected else 0.0,
        "sector_level_1_coverage": float(level_1_covered_symbols / total_expected) if total_expected else 0.0,
        "sector_level_2_coverage": float(level_2_covered_symbols / total_expected) if total_expected else 0.0,
        "unknown_rate": float(unknown_symbols / total_expected) if total_expected else 0.0,
        "board_proxy_symbols": int(board_proxy_symbols),
        "coverage_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "sector_count": int(distribution["sector_level_1"].nunique()) if not distribution.empty else 0,
        "largest_sector_share": float(distribution["share"].max()) if not distribution.empty else 0.0,
        "distribution_anomaly": concentration_warning,
    }


def validate_sector_map(frame: pd.DataFrame, *, symbols: Iterable[str] = ()) -> dict[str, object]:
    missing_cols = [c for c in SECTOR_MAP_REQUIRED_COLUMNS if c not in frame.columns]
    duplicate_final_rows = int(frame.duplicated(subset=["symbol"]).sum()) if "symbol" in frame.columns else 0
    invalid_status = []
    if "coverage_status" in frame.columns:
        invalid_status = sorted(set(frame["coverage_status"].dropna().astype(str)) - set(VALID_COVERAGE_STATUS))
    pit_violations = 0
    effective_after_available = 0
    if {"available_at", "fetched_at"}.issubset(frame.columns):
        avail = _to_naive_series(frame["available_at"])
        fetched = _to_naive_series(frame["fetched_at"])
        pit_violations = int((avail > fetched).fillna(False).sum())
    if {"effective_date", "available_at"}.issubset(frame.columns):
        effective = _to_naive_series(frame["effective_date"])
        avail = _to_naive_series(frame["available_at"])
        effective_after_available = int((effective > avail).fillna(False).sum())
    missing_expected = []
    expected = set(_normalise_symbols(symbols))
    if expected and "symbol" in frame.columns:
        missing_expected = sorted(expected - set(frame["symbol"].astype(str)))
    status = "passed"
    if missing_cols or duplicate_final_rows or invalid_status or pit_violations or effective_after_available or missing_expected:
        status = "failed"
    return {
        "status": status,
        "row_count": int(len(frame)),
        "missing_columns": missing_cols,
        "duplicate_symbol_count": duplicate_final_rows,
        "invalid_coverage_status": invalid_status,
        "pit_violation_count": pit_violations,
        "effective_after_available_count": effective_after_available,
        "missing_expected_symbols": missing_expected,
    }


def sector_coverage_gate(
    frame: pd.DataFrame,
    *,
    symbols: Iterable[str] = (),
    as_of_date: str | None = None,
    config: SectorMapConfig | None = None,
) -> dict[str, object]:
    cfg = config or SectorMapConfig()
    coverage = coverage_report(frame, symbols=symbols)
    validation = validate_sector_map(frame, symbols=symbols)
    total = max(int(coverage.get("total_expected_symbols", 0)), 1)
    stale_rate = 0.0
    if not frame.empty and "available_at" in frame.columns:
        as_of = _to_naive_timestamp(as_of_date) if as_of_date else pd.Timestamp.now(tz=None).normalize()
        status = frame.get("coverage_status", pd.Series(dtype=str)).astype(str)
        source = frame.get("source", pd.Series(dtype=str)).astype(str)
        available = _to_naive_series(frame["available_at"])
        eligible = (status != "missing") & ~source.str.startswith(BOARD_PROXY_SOURCE)
        stale_symbols = frame.loc[eligible & ((as_of - available).dt.days > int(cfg.max_staleness_days)), "symbol"].nunique()
        stale_rate = float(stale_symbols / total)
    reasons: list[str] = []
    if validation["status"] != "passed":
        reasons.append("sector_validation_failed")
    if float(coverage["sector_level_1_coverage"]) < float(cfg.min_level_1_coverage):
        reasons.append("sector_level_1_coverage_below_threshold")
    if float(coverage["sector_level_2_coverage"]) < float(cfg.min_level_2_coverage):
        reasons.append("sector_level_2_coverage_below_threshold")
    if float(coverage["unknown_rate"]) > float(cfg.max_unknown_rate):
        reasons.append("unknown_rate_above_threshold")
    if stale_rate > float(cfg.max_stale_available_at_rate):
        reasons.append("stale_available_at_rate_above_threshold")
    usable_for_optimization = not reasons
    return {
        "sector_usable_for_diagnostics": True,
        "sector_usable_for_optimization": bool(usable_for_optimization),
        "reason": "passed" if usable_for_optimization else ",".join(reasons),
        "thresholds": {
            "sector_level_1_coverage": float(cfg.min_level_1_coverage),
            "sector_level_2_coverage": float(cfg.min_level_2_coverage),
            "unknown_rate": float(cfg.max_unknown_rate),
            "stale_available_at_rate": float(cfg.max_stale_available_at_rate),
            "max_staleness_days": int(cfg.max_staleness_days),
        },
        "observed": {
            "sector_level_1_coverage": float(coverage["sector_level_1_coverage"]),
            "sector_level_2_coverage": float(coverage["sector_level_2_coverage"]),
            "unknown_rate": float(coverage["unknown_rate"]),
            "stale_available_at_rate": stale_rate,
        },
    }


class SectorMapBuilder:
    """Build sector map parquet plus coverage/quality side reports."""

    def __init__(self, config: SectorMapConfig | None = None) -> None:
        self.config = config or SectorMapConfig()

    def build(
        self,
        source_frame: pd.DataFrame | None = None,
    ) -> SectorMapResult:
        symbols = _normalise_symbols(self.config.symbols)
        raw = (
            _coerce_source_frame(source_frame, config=self.config)
            if source_frame is not None and not source_frame.empty
            else pd.DataFrame(columns=SECTOR_MAP_REQUIRED_COLUMNS)
        )
        duplicates = duplicate_symbol_report(raw)
        conflicts = source_conflict_report(raw)
        source_priority = source_priority_report(raw)
        selected = _select_latest_pit_rows(raw, as_of_date=self.config.as_of_date)
        if symbols:
            selected = selected[selected["symbol"].isin(symbols)].copy()
            missing = _missing_rows(symbols, set(selected["symbol"].astype(str)), config=self.config)
            final = pd.concat([selected, missing], ignore_index=True)
        else:
            final = selected
        for col in SECTOR_MAP_REQUIRED_COLUMNS:
            if col not in final.columns:
                final[col] = pd.NA
        final = final[list(SECTOR_MAP_REQUIRED_COLUMNS)].sort_values("symbol").reset_index(drop=True)
        validation = validate_sector_map(final, symbols=symbols)
        coverage = coverage_report(final, symbols=symbols)
        coverage["gate"] = sector_coverage_gate(final, symbols=symbols, as_of_date=self.config.as_of_date, config=self.config)
        coverage["source_conflict_count"] = int(len(conflicts))
        missing_symbols = final[final["coverage_status"] == "missing"][["symbol"]].reset_index(drop=True)
        distribution = sector_distribution_report(final)
        return SectorMapResult(
            frame=final,
            coverage=coverage,
            missing_symbols=missing_symbols,
            duplicate_symbols=duplicates,
            source_priority=source_priority,
            sector_distribution=distribution,
            validation=validation,
        )

    def build_from_path(self, path: str | Path) -> SectorMapResult:
        return self.build(read_frame(path))

    def write(self, result: SectorMapResult) -> SectorMapResult:
        root = Path(self.config.output_root)
        out_dir = root / "silver" / "sector_map"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "sector_map.parquet"
        coverage_json = out_dir / "coverage_report.json"
        missing_csv = out_dir / "missing_symbols.csv"
        duplicate_csv = out_dir / "duplicate_symbols.csv"
        source_priority_json = out_dir / "source_priority_report.json"
        distribution_csv = out_dir / "sector_distribution.csv"
        validation_json = out_dir / "validation_report.json"

        result.frame.to_parquet(output, index=False)
        import json
        coverage_json.write_text(json.dumps(result.coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        validation_json.write_text(json.dumps(result.validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        source_priority_json.write_text(
            json.dumps(result.source_priority.to_dict("records"), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        result.missing_symbols.to_csv(missing_csv, index=False)
        result.duplicate_symbols.to_csv(duplicate_csv, index=False)
        result.sector_distribution.to_csv(distribution_csv, index=False)
        manifest = build_manifest_for_frame(
            dataset_name="sector_map",
            vendor="local",
            frame=result.frame,
            output_paths=[output],
            symbols=self.config.symbols,
            required_columns=SECTOR_MAP_REQUIRED_COLUMNS,
            pit_violation_count=int(result.validation.get("pit_violation_count", 0)),
            warnings=("sector_distribution_anomaly",) if result.coverage.get("distribution_anomaly") else (),
            extra={
                "coverage_report": result.coverage,
                "validation_report": result.validation,
                "source_priority_report": result.source_priority.to_dict("records"),
                "pit_policy": "join only where sector.available_at <= trade_date; current_snapshot must not backfill history",
            },
        )
        manifest.write(root / "manifests" / "sector_map.json")
        paths = {
            "sector_map": str(output),
            "coverage_report": str(coverage_json),
            "missing_symbols": str(missing_csv),
            "duplicate_symbols": str(duplicate_csv),
            "source_priority_report": str(source_priority_json),
            "sector_distribution": str(distribution_csv),
            "validation_report": str(validation_json),
            "manifest": str(root / "manifests" / "sector_map.json"),
        }
        return SectorMapResult(
            frame=result.frame,
            coverage=result.coverage,
            missing_symbols=result.missing_symbols,
            duplicate_symbols=result.duplicate_symbols,
            source_priority=result.source_priority,
            sector_distribution=result.sector_distribution,
            validation=result.validation,
            output_paths=paths,
        )


__all__ = [
    "SECTOR_MAP_REQUIRED_COLUMNS",
    "VALID_COVERAGE_STATUS",
    "BOARD_PROXY_SOURCE",
    "SOURCE_PRIORITY",
    "SectorMapBuilder",
    "SectorMapConfig",
    "SectorMapResult",
    "board_proxy_rows",
    "coverage_report",
    "duplicate_symbol_report",
    "normalize_sector_source",
    "sector_distribution_report",
    "sector_coverage_gate",
    "source_conflict_report",
    "source_priority_rank",
    "source_priority_report",
    "validate_sector_map",
]
