"""AKShare daily market-panel bootstrap for the V7 silver lake."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.akshare_calendar import load_akshare_research_calendar
from quantagent.data.providers.akshare_live_provider import (
    AKSHARE_MARKET_REQUIRED_COLUMNS,
    AkShareLiveProvider,
    akshare_market_schema_report,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.trading_calendar import TradingCalendar
from quantagent.data.v7_auto_range import V7ResolvedDateRange, resolve_akshare_market_fetch_range


@dataclass(frozen=True)
class AkShareMarketPanelConfig:
    symbols: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    output_root: str | None = None
    output_path: str | None = None
    allow_network: bool = False
    adjust: str = ""
    provider_uri_for_range: str | None = None
    as_of_date: str | None = None


def build_akshare_market_panel(config: AkShareMarketPanelConfig) -> dict[str, object]:
    if not config.symbols:
        raise ValueError("AKShare market panel requires at least one symbol")
    resolved_root = Path(config.output_root) if config.output_root else quant_paths().data_root / "v7"
    lake = v7_lake_paths(resolved_root).ensure()
    heuristic_range = resolve_akshare_market_fetch_range(
        start_date=config.start_date,
        end_date=config.end_date,
        provider_uri=config.provider_uri_for_range,
        lake_root=resolved_root,
        as_of_date=config.as_of_date,
    )
    calendar_evidence = load_akshare_research_calendar(allow_network=config.allow_network)
    research_calendar = calendar_evidence.calendar
    calendar_meta = calendar_evidence.metadata
    calendar_warnings = calendar_evidence.warnings
    resolved_range = _snap_range_to_calendar(heuristic_range, research_calendar)
    resolved_output = (
        Path(config.output_path)
        if config.output_path
        else lake.silver_market_panel / "market_panel.parquet"
    )
    request = ProviderRequest(
        start_date=resolved_range.start_date,
        end_date=resolved_range.end_date,
        symbols=config.symbols,
    )

    result = AkShareLiveProvider(
        allow_network=config.allow_network,
        adjust=config.adjust,
        trading_calendar=research_calendar,
        calendar_source=str(calendar_meta.get("source") or ""),
    ).daily_ohlcv(request)
    merged_frame, merge_info = _merge_with_existing_panel(result.frame, resolved_output)
    normalised = _normalise_dtypes(merged_frame)
    schema_report = akshare_market_schema_report(normalised)
    failed_symbols = list(result.metadata.get("failed_symbols", []))
    warnings = [*result.warnings, *calendar_warnings]

    blockers: list[str] = []
    if result.frame.empty:
        blockers.append("akshare_fetch_empty")
    if failed_symbols:
        blockers.append("akshare_requested_symbol_coverage_incomplete")
    if not result.point_in_time:
        blockers.append("akshare_market_not_pit_certified")
    if schema_report["status"] != "passed":
        blockers.append("akshare_market_schema_or_pit_failed")
    if config.adjust:
        blockers.append("akshare_adjusted_history_has_no_vintaged_adjustment_evidence")
    if research_calendar.empty:
        blockers.append("akshare_independent_research_calendar_unavailable")

    # Never replace the canonical silver panel with a candidate that failed its
    # own source/PIT contract. The caller gets complete diagnostics and may retry
    # or repair evidence; the last known panel stays untouched.
    if blockers:
        return {
            "status": "blocked",
            "output": None,
            "manifest": None,
            "rows": int(len(result.frame)),
            "candidate_rows": int(len(normalised)),
            "symbols": list(request.symbols),
            "failed_symbols": failed_symbols,
            "warnings": list(dict.fromkeys(warnings)),
            "blockers": blockers,
            "schema_report": schema_report,
            "calendar": calendar_meta,
            "heuristic_range": heuristic_range.to_dict(),
            "resolved_range": resolved_range.to_dict(),
            "existing_panel_preserved": bool(resolved_output.exists()),
        }

    written = _write_frame(normalised, resolved_output)
    panel_start = (
        str(pd.to_datetime(normalised["trade_date"]).min().date())
        if not normalised.empty
        else resolved_range.start_date
    )
    panel_end = (
        str(pd.to_datetime(normalised["trade_date"]).max().date())
        if not normalised.empty
        else resolved_range.end_date
    )
    manifest_path = lake.manifests / "market_panel.json"
    prior_manifest = _read_prior_manifest(manifest_path)
    prior_extra = (
        prior_manifest.get("extra", {})
        if isinstance(prior_manifest.get("extra"), dict)
        else {}
    )
    extra = {
        "source": result.source,
        "adjust": config.adjust or "raw",
        "function_name": result.metadata.get("function_name"),
        "akshare_version": result.metadata.get("akshare_version"),
        "source_order": result.metadata.get("source_order", []),
        "source_counts": result.metadata.get("source_counts", {}),
        "source_by_symbol": result.metadata.get("source_by_symbol", {}),
        "failed_symbols": failed_symbols,
        "requested_symbol_count": result.metadata.get("requested_symbol_count"),
        "fetched_symbol_count": result.metadata.get("fetched_symbol_count"),
        "canonical_volume_unit": result.metadata.get("canonical_volume_unit", "shares"),
        "canonical_amount_unit": result.metadata.get("canonical_amount_unit", "CNY"),
        "raw_volume_unit_by_source": result.metadata.get("raw_volume_unit_by_source", {}),
        "schema_report": schema_report,
        "availability_rule": "daily_bar_available_next_explicit_trading_session",
        "calendar": calendar_meta,
        "heuristic_range": heuristic_range.to_dict(),
        "resolved_range": resolved_range.to_dict(),
        "akshare_fetched_rows": int(len(result.frame)),
        "merge_info": merge_info,
        "config": asdict(config),
        "production_integrity_certified": False,
    }
    if "adjustment_repair" in prior_extra:
        extra["adjustment_repair"] = prior_extra["adjustment_repair"]
    vendor = "qlib+akshare" if merge_info["merged_with_existing"] else "akshare"
    manifest = build_manifest_for_frame(
        dataset_name="market_panel",
        vendor=vendor,
        frame=normalised,
        output_paths=[written],
        start_date=panel_start,
        end_date=panel_end,
        symbols=request.symbols,
        required_columns=AKSHARE_MARKET_REQUIRED_COLUMNS,
        pit_violation_count=int(schema_report.get("pit_violation_count", 0)),
        warnings=tuple(dict.fromkeys(warnings)),
        extra=extra,
    )
    manifest.write(manifest_path)
    return {
        "status": "passed",
        "output": str(written),
        "manifest": str(manifest_path),
        "rows": int(len(result.frame)),
        "symbols": list(request.symbols),
        "warnings": list(dict.fromkeys(warnings)),
        "schema_report": schema_report,
        "calendar": calendar_meta,
        "heuristic_range": heuristic_range.to_dict(),
        "resolved_range": resolved_range.to_dict(),
    }


def _snap_range_to_calendar(
    value: V7ResolvedDateRange,
    calendar: TradingCalendar,
) -> V7ResolvedDateRange:
    """Snap heuristic weekday dates to the actual research session set.

    ``v7_auto_range`` remains provider-neutral and may produce a weekday bridge
    when a local Qlib calendar ends. Before an AKShare request is issued, this
    boundary converts that heuristic to the first/last real session contained in
    the independently fetched calendar. No holiday is manufactured as a fetch
    or availability date.
    """
    if calendar.empty:
        return value
    sessions = pd.DatetimeIndex(calendar.trading_days).normalize()
    start = pd.Timestamp(value.start_date).normalize()
    end = pd.Timestamp(value.end_date).normalize()
    starts = sessions[sessions >= start]
    ends = sessions[sessions <= end]
    if starts.empty or ends.empty:
        raise ValueError(
            "AKShare research calendar has no session covering the resolved fetch window"
        )
    snapped_start = starts[0]
    snapped_end = ends[-1]
    if snapped_start > snapped_end:
        raise ValueError(
            "AKShare fetch window contains no trading session after calendar snapping"
        )
    notes = list(value.notes)
    if snapped_start != start:
        notes.append(f"start_snapped_to_session={snapped_start.date()}")
    if snapped_end != end:
        notes.append(f"end_snapped_to_session={snapped_end.date()}")
    return V7ResolvedDateRange(
        start_date=str(snapped_start.date()),
        end_date=str(snapped_end.date()),
        source=value.source + "+akshare_session_calendar",
        notes=tuple(notes),
    )


# Compatibility wrapper for callers/tests that imported the former bootstrap-
# local helper. New code should use providers.akshare_calendar directly.
def _load_akshare_research_calendar(
    *, allow_network: bool
) -> tuple[TradingCalendar, dict[str, object], tuple[str, ...]]:
    evidence = load_akshare_research_calendar(allow_network=allow_network)
    return evidence.calendar, evidence.metadata, evidence.warnings


def _read_prior_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        normalised = _normalise_dtypes(frame)
        try:
            normalised.to_parquet(path, index=False)
        except Exception:
            import polars as pl

            pl.from_pandas(normalised).write_parquet(str(path))
        return path
    frame.to_csv(path, index=False)
    return path


def _normalise_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed dtypes without inventing missing PIT evidence."""
    out = frame.copy()
    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source_reliability",
    ):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("trade_date", "available_at"):
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "trade_date" in out and "available_at" not in out:
        out["available_at"] = pd.NaT
    for col in (
        "symbol",
        "source",
        "source_type",
        "volume_unit",
        "raw_volume_unit",
        "amount_unit",
        "price_adjustment",
    ):
        if col in out:
            out[col] = out[col].astype("string")
    if "point_in_time_valid" in out:
        out["point_in_time_valid"] = (
            out["point_in_time_valid"].fillna(False).astype("bool")
        )
    return out


def _merge_with_existing_panel(
    new_frame: pd.DataFrame, output_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Concat new rows with an existing panel and dedup on (symbol, trade_date)."""
    info: dict[str, object] = {
        "merged_with_existing": False,
        "existing_rows": 0,
        "new_rows": int(len(new_frame)),
        "final_rows": int(len(new_frame)),
    }
    if not output_path.exists():
        return new_frame, info
    try:
        existing = pd.read_parquet(output_path)
    except Exception:
        try:
            import polars as pl
        except ImportError:
            return new_frame, info
        existing = pl.read_parquet(str(output_path)).to_pandas()
    if existing.empty:
        return new_frame, info
    if new_frame.empty:
        info["merged_with_existing"] = True
        info["existing_rows"] = int(len(existing))
        info["final_rows"] = int(len(existing))
        return existing, info
    info["merged_with_existing"] = True
    info["existing_rows"] = int(len(existing))
    aligned_cols = list(dict.fromkeys([*existing.columns, *new_frame.columns]))
    existing = existing.reindex(columns=aligned_cols)
    new_aligned = new_frame.reindex(columns=aligned_cols)
    combined = pd.concat([existing, new_aligned], ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    combined = combined.drop_duplicates(
        subset=["symbol", "trade_date"], keep="last"
    )
    combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    info["final_rows"] = int(len(combined))
    return combined, info
