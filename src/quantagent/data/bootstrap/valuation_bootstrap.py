"""V7 valuation bootstrap with explicit historical/current semantics.

Past as-of dates use AKShare's source-dated Baidu valuation history. A current
spot snapshot is accepted only for the current Asia/Shanghai date **and** an
explicit A-share trading session. Local files must carry explicit
availability/PIT evidence; missing provenance is never invented from
``trade_date``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.bootstrap.akshare_market_bootstrap import _load_akshare_research_calendar
from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.akshare_valuation_provider import (
    AKSHARE_VALUATION_REQUIRED_COLUMNS,
    AkShareValuationProvider,
    akshare_valuation_schema_report,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.trading_calendar import TradingCalendar


@dataclass(frozen=True)
class ValuationBootstrapConfig:
    as_of_dates: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    lake_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7"))
    allow_network: bool = False
    csv_snapshot: str | None = None
    output_name: str = "valuation.parquet"


def build_valuation_cache(config: ValuationBootstrapConfig) -> dict[str, object]:
    if not config.as_of_dates and not config.csv_snapshot:
        raise ValueError("valuation bootstrap requires either as_of_dates or csv_snapshot")
    lake = v7_lake_paths(config.lake_root).ensure()
    output_path = lake.silver_valuation / config.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    blockers: list[str] = []
    provider_metadata: list[dict[str, object]] = []

    if config.csv_snapshot:
        frame = _load_local_snapshot(config.csv_snapshot)
        schema_report = akshare_valuation_schema_report(frame)
        if schema_report["status"] != "passed":
            blockers.append("valuation_local_snapshot_schema_or_pit_failed")
        if "point_in_time_valid" not in frame.columns or not frame[
            "point_in_time_valid"
        ].fillna(False).astype(bool).all():
            blockers.append("valuation_local_snapshot_missing_explicit_pit_evidence")
        vendor = "local_csv"
        calendar_meta: dict[str, object] = {
            "source": "local_snapshot_supplied_evidence",
            "production_certified": False,
            "status": "not_rederived",
        }
    else:
        frame, provider_metadata, network_warnings, network_blockers, calendar_meta = (
            _fetch_network_valuation(config)
        )
        warnings.extend(network_warnings)
        blockers.extend(network_blockers)
        schema_report = akshare_valuation_schema_report(frame)
        if schema_report["status"] != "passed":
            blockers.append("valuation_network_schema_or_pit_failed")
        vendor = "akshare"

    if config.symbols and not frame.empty:
        requested_symbols = {str(symbol) for symbol in config.symbols}
        observed_symbols = (
            set(frame["symbol"].astype(str).unique()) if "symbol" in frame else set()
        )
        missing_symbols = sorted(requested_symbols - observed_symbols)
        if missing_symbols:
            blockers.append("valuation_requested_symbol_coverage_incomplete")
            warnings.append("valuation_missing_symbols:" + ",".join(missing_symbols))

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    if blockers:
        return {
            "status": "blocked",
            "config": asdict(config),
            "output_path": None,
            "manifest_path": None,
            "rows": int(len(frame)),
            "warnings": warnings,
            "blockers": blockers,
            "schema_report": schema_report,
            "calendar": calendar_meta,
            "provider_metadata": provider_metadata,
            "existing_output_preserved": bool(
                output_path.exists() or output_path.with_suffix(".csv").exists()
            ),
        }

    if frame.empty:
        return {
            "status": "empty",
            "config": asdict(config),
            "output_path": None,
            "manifest_path": None,
            "rows": 0,
            "warnings": [*warnings, "valuation_snapshot_empty"],
            "blockers": [],
            "schema_report": schema_report,
            "calendar": calendar_meta,
        }

    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    manifest_target = _write_frame(frame, output_path)
    manifest = build_manifest_for_frame(
        dataset_name="valuation",
        vendor=vendor,
        frame=frame,
        output_paths=[manifest_target],
        start_date=str(pd.to_datetime(frame["trade_date"]).min().date()),
        end_date=str(pd.to_datetime(frame["trade_date"]).max().date()),
        symbols=config.symbols,
        required_columns=AKSHARE_VALUATION_REQUIRED_COLUMNS,
        pit_violation_count=int(schema_report.get("pit_violation_count", 0)),
        warnings=tuple(warnings),
        extra={
            "as_of_dates": list(config.as_of_dates),
            "csv_snapshot": config.csv_snapshot,
            "schema_report": schema_report,
            "calendar": calendar_meta,
            "provider_metadata": provider_metadata,
            "historical_current_snapshot_backdating_blocked": True,
            "non_session_valuation_dates_blocked": True,
            "canonical_market_cap_unit": "CNY",
            "production_integrity_certified": False,
        },
    )
    manifest_path = lake.manifests / "valuation.json"
    manifest.write(manifest_path)
    return {
        "status": "passed",
        "config": asdict(config),
        "output_path": str(manifest_target),
        "manifest_path": str(manifest_path),
        "rows": int(len(frame)),
        "warnings": warnings,
        "blockers": [],
        "schema_report": schema_report,
        "calendar": calendar_meta,
    }


def _assert_requested_dates_are_sessions(
    dates: pd.DatetimeIndex,
    calendar: TradingCalendar,
) -> None:
    """Reject weekend/holiday valuation as-of labels instead of silently snapping.

    AKShare's Baidu valuation endpoint can expose calendar-day observations. The
    canonical silver table is keyed by ``trade_date`` and therefore only accepts
    dates explicitly present in the bound A-share session set.
    """
    if calendar.empty:
        raise ValueError(
            "historical/current AKShare valuation requires an explicit A-share trading calendar"
        )
    invalid = [
        str(pd.Timestamp(value).date())
        for value in dates
        if not calendar.contains(pd.Timestamp(value))
    ]
    if invalid:
        raise ValueError(
            "valuation as_of_dates must be actual A-share trading sessions; invalid="
            + ",".join(invalid)
        )


def _fetch_network_valuation(
    config: ValuationBootstrapConfig,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[str],
    list[str],
    dict[str, object],
]:
    requested_dates = pd.DatetimeIndex(
        pd.to_datetime(list(config.as_of_dates), errors="raise")
    ).normalize()
    if requested_dates.empty:
        raise ValueError("network valuation requires at least one as_of_date")
    if requested_dates.has_duplicates:
        requested_dates = requested_dates.drop_duplicates()
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    if bool((requested_dates > today).any()):
        future = [str(value.date()) for value in requested_dates[requested_dates > today]]
        raise ValueError(
            "valuation as_of_dates cannot be in the future: " + ",".join(future)
        )

    research_calendar, calendar_meta, calendar_warnings = _load_akshare_research_calendar(
        allow_network=config.allow_network
    )
    _assert_requested_dates_are_sessions(requested_dates, research_calendar)
    provider = AkShareValuationProvider(
        allow_network=config.allow_network,
        trading_calendar=research_calendar,
    )
    frames: list[pd.DataFrame] = []
    warnings: list[str] = list(calendar_warnings)
    blockers: list[str] = []
    metadata: list[dict[str, object]] = []

    past_dates = requested_dates[requested_dates < today]
    if len(past_dates):
        if not config.symbols:
            raise ValueError(
                "historical AKShare valuation requires explicit symbols; current universe "
                "membership cannot be projected backward"
            )
        request = ProviderRequest(
            start_date=str(past_dates.min().date()),
            end_date=str(past_dates.max().date()),
            symbols=config.symbols,
        )
        result = provider.historical(request)
        warnings.extend(result.warnings)
        metadata.append(dict(result.metadata) | {"mode": "historical"})
        if not result.point_in_time:
            blockers.append("akshare_historical_valuation_not_pit_certified")
        historical = result.frame.copy()
        if not historical.empty:
            historical["trade_date"] = pd.to_datetime(
                historical["trade_date"], errors="coerce"
            ).dt.normalize()
            # Source can expose weekend/calendar-day valuations. Only exact
            # requested session dates survive into canonical silver evidence.
            historical = historical[historical["trade_date"].isin(past_dates)]
            frames.append(historical)
            _assert_requested_key_coverage(
                historical,
                symbols=config.symbols,
                dates=past_dates,
                blockers=blockers,
                warnings=warnings,
                label="historical",
            )
        else:
            blockers.append("akshare_historical_valuation_empty")

    if today in requested_dates:
        request = (
            ProviderRequest(
                str(today.date()), str(today.date()), symbols=config.symbols
            )
            if config.symbols
            else None
        )
        result = provider.snapshot(str(today.date()), request=request)
        warnings.extend(result.warnings)
        metadata.append(dict(result.metadata) | {"mode": "current_snapshot"})
        if not result.point_in_time:
            blockers.append("akshare_current_valuation_snapshot_not_pit_certified")
        current = result.frame.copy()
        if not current.empty:
            frames.append(current)
            if config.symbols:
                _assert_requested_key_coverage(
                    current,
                    symbols=config.symbols,
                    dates=pd.DatetimeIndex([today]),
                    blockers=blockers,
                    warnings=warnings,
                    label="current",
                )
        else:
            blockers.append("akshare_current_valuation_snapshot_empty")

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, metadata, warnings, list(dict.fromkeys(blockers)), calendar_meta


def _assert_requested_key_coverage(
    frame: pd.DataFrame,
    *,
    symbols: tuple[str, ...],
    dates: pd.DatetimeIndex,
    blockers: list[str],
    warnings: list[str],
    label: str,
) -> None:
    if frame.empty:
        blockers.append(f"valuation_{label}_key_coverage_empty")
        return
    observed = set(
        zip(
            frame["symbol"].astype(str),
            pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize(),
        )
    )
    expected = {
        (str(symbol), pd.Timestamp(date).normalize())
        for symbol in symbols
        for date in dates
    }
    missing = sorted(expected - observed, key=lambda item: (item[0], item[1]))
    if missing:
        blockers.append(f"valuation_{label}_requested_key_coverage_incomplete")
        sample = [f"{symbol}@{date.date()}" for symbol, date in missing[:10]]
        warnings.append(f"valuation_{label}_missing_keys:" + ",".join(sample))


def _load_local_snapshot(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"valuation csv snapshot not found: {path}")
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "trade_date" not in frame.columns:
        raise ValueError("local valuation snapshot must include trade_date")
    if "available_at" not in frame.columns:
        frame["available_at"] = pd.NaT
    if "point_in_time_valid" not in frame.columns:
        frame["point_in_time_valid"] = False
    else:
        frame["point_in_time_valid"] = (
            frame["point_in_time_valid"].fillna(False).astype(bool)
        )
    return frame


def _write_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    try:
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback
