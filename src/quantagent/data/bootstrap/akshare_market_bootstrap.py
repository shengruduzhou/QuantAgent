"""AkShare daily market-panel bootstrap for the V7 silver lake."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

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
    merged_frame, merge_info = _merge_with_existing_panel(result.frame, resolved_output)
    written = _write_frame(merged_frame, resolved_output)
    schema_report = akshare_market_schema_report(merged_frame)
    panel_start = str(merged_frame["trade_date"].min())[:10] if not merged_frame.empty else resolved_range.start_date
    panel_end = str(merged_frame["trade_date"].max())[:10] if not merged_frame.empty else resolved_range.end_date
    manifest_path = lake.manifests / "market_panel.json"
    prior_manifest = _read_prior_manifest(manifest_path)
    prior_extra = prior_manifest.get("extra", {}) if isinstance(prior_manifest.get("extra"), dict) else {}
    extra = {
        "source": result.source,
        "adjust": config.adjust,
        "function_name": result.metadata.get("function_name"),
        "failed_symbols": result.metadata.get("failed_symbols", []),
        "schema_report": schema_report,
        "availability_rule": "daily_ohlcv_available_next_business_day",
        "resolved_range": resolved_range.to_dict(),
        "akshare_fetched_rows": int(len(result.frame)),
        "merge_info": merge_info,
        "config": asdict(config),
    }
    if "adjustment_repair" in prior_extra:
        extra["adjustment_repair"] = prior_extra["adjustment_repair"]
    vendor = "qlib+akshare" if merge_info["merged_with_existing"] else "akshare"
    if result.frame.empty and prior_manifest.get("vendor"):
        vendor = str(prior_manifest["vendor"])
    manifest = build_manifest_for_frame(
        dataset_name="market_panel",
        vendor=vendor,
        frame=merged_frame,
        output_paths=[written],
        start_date=panel_start,
        end_date=panel_end,
        symbols=request.symbols,
        required_columns=AKSHARE_MARKET_REQUIRED_COLUMNS,
        pit_violation_count=int(schema_report.get("pit_violation_count", 0)),
        warnings=result.warnings,
        extra=extra,
    )
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
    """Coerce mixed dtypes (qlib+akshare concat) into a parquet-safe schema."""
    out = frame.copy()
    for col in ("open", "high", "low", "close", "volume", "amount", "source_reliability"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("trade_date", "available_at"):
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "trade_date" in out:
        if "available_at" not in out:
            out["available_at"] = pd.NaT
        missing_available = out["available_at"].isna()
        out.loc[missing_available, "available_at"] = (
            pd.to_datetime(out.loc[missing_available, "trade_date"], errors="coerce") + pd.offsets.BDay(1)
        )
    for col in ("symbol", "source", "source_type"):
        if col in out:
            out[col] = out[col].astype("string")
    if "point_in_time_valid" in out:
        out["point_in_time_valid"] = out["point_in_time_valid"].fillna(True).astype("bool")
    return out


def _merge_with_existing_panel(new_frame: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Concat new AkShare rows with any existing panel and dedup on (symbol, trade_date).

    Existing rows (typically the qlib 1999→2020 base) are preserved; AkShare rows win
    on overlap because they reflect the latest adjusted close. The merged panel becomes
    the contiguous 1999→today market_panel used downstream.
    """
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
    combined = combined.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
    combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    info["final_rows"] = int(len(combined))
    return combined, info
