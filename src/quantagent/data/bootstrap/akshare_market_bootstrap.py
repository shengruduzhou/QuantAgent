"""AkShare daily market-panel bootstrap for the V7 silver lake."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from quantagent.config.paths import quant_paths
from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.akshare_live_provider import (
    AKSHARE_MARKET_REQUIRED_COLUMNS,
    AkShareLiveProvider,
    akshare_market_schema_report,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.v7_auto_range import resolve_akshare_market_fetch_range


@dataclass(frozen=True)
class AkShareMarketPanelConfig:
    symbols: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    output_root: str | None = None
    output_path: str | None = None
    allow_network: bool = False
    adjust: str = "qfq"
    provider_uri_for_range: str | None = None
    as_of_date: str | None = None


def build_akshare_market_panel(config: AkShareMarketPanelConfig) -> dict[str, object]:
    if not config.symbols:
        raise ValueError("AkShare market panel requires at least one symbol")
    resolved_root = Path(config.output_root) if config.output_root else quant_paths().data_root / "v7"
    lake = v7_lake_paths(resolved_root).ensure()
    resolved_range = resolve_akshare_market_fetch_range(
        start_date=config.start_date,
        end_date=config.end_date,
        provider_uri=config.provider_uri_for_range,
        lake_root=resolved_root,
        as_of_date=config.as_of_date,
    )
    resolved_output = Path(config.output_path) if config.output_path else lake.silver_market_panel / "market_panel.parquet"
    request = ProviderRequest(
        start_date=resolved_range.start_date,
        end_date=resolved_range.end_date,
        symbols=config.symbols,
    )
    result = AkShareLiveProvider(allow_network=config.allow_network, adjust=config.adjust).daily_ohlcv(request)
    written = _write_frame(result.frame, resolved_output)
    schema_report = akshare_market_schema_report(result.frame)
    manifest = build_manifest_for_frame(
        dataset_name="market_panel",
        vendor="akshare",
        frame=result.frame,
        output_paths=[written],
        start_date=resolved_range.start_date,
        end_date=resolved_range.end_date,
        symbols=request.symbols,
        required_columns=AKSHARE_MARKET_REQUIRED_COLUMNS,
        pit_violation_count=int(schema_report.get("pit_violation_count", 0)),
        warnings=result.warnings,
        extra={
            "source": result.source,
            "adjust": config.adjust,
            "function_name": result.metadata.get("function_name"),
            "failed_symbols": result.metadata.get("failed_symbols", []),
            "schema_report": schema_report,
            "availability_rule": "daily_ohlcv_available_next_business_day",
            "resolved_range": resolved_range.to_dict(),
            "config": asdict(config),
        },
    )
    manifest_path = lake.manifests / "market_panel.json"
    manifest.write(manifest_path)
    return {
        "status": "passed" if not result.frame.empty and schema_report["status"] == "passed" else "empty",
        "output": str(written),
        "manifest": str(manifest_path),
        "rows": int(len(result.frame)),
        "symbols": list(request.symbols),
        "warnings": list(result.warnings),
        "schema_report": schema_report,
        "resolved_range": resolved_range.to_dict(),
    }


def _write_frame(frame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path
