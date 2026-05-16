"""Qlib CN bootstrap and PIT market-panel export for V7.

This bootstrap wraps the optional pyqlib provider so V7 can:

* verify the local provider_uri before attempting any read,
* document the official CN download command in a single place,
* materialise a canonical PIT-tagged market panel into the silver tier
  (``data/v7/silver/market_panel/``),
* derive close-available-next-day technical features, and
* emit a ``DataManifest`` that records vendor, range, schema, PIT
  violations and content hashes for downstream consumers.

Missing pyqlib or an invalid provider_uri raises ``ProviderUnavailable``
with a clear, actionable message — there is no synthetic fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.qlib_provider import (
    QLIB_MARKET_COLUMNS,
    QlibProvider,
    validate_qlib_market_schema,
)
from quantagent.data.v7_dataset_builder import build_market_features


QLIB_CN_DOWNLOAD_COMMAND = (
    "python scripts/get_data.py qlib_data "
    "--target_dir ~/.qlib/qlib_data/cn_data --region cn"
)


@dataclass(frozen=True)
class QlibBootstrapConfig:
    provider_uri: str
    start_date: str
    end_date: str
    symbols: tuple[str, ...] = ()
    universe: str | None = None
    region: str = "cn"
    output_root: str = "data/v7"
    build_features: bool = True
    require_optional_flags: bool = False
    use_lake_layout: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


def build_qlib_market_panel(config: QlibBootstrapConfig) -> dict[str, object]:
    provider_path = Path(config.provider_uri).expanduser()
    if not provider_path.exists():
        raise ProviderUnavailable(
            "Qlib provider_uri does not exist. Prepare CN data with: "
            f"{QLIB_CN_DOWNLOAD_COMMAND}"
        )
    request = ProviderRequest(
        start_date=config.start_date,
        end_date=config.end_date,
        symbols=config.symbols,
        universe=config.universe,
    )
    result = QlibProvider(str(provider_path), config.region).daily_ohlcv(request)
    report = validate_qlib_market_schema(result.frame, as_of_date=config.end_date)
    if report["status"] != "passed":
        raise ValueError(f"Qlib market schema failed: {report}")
    if config.require_optional_flags and report["optional_columns_missing"]:
        raise ValueError(
            f"Qlib market panel missing tradability flags: {report['optional_columns_missing']}"
        )

    lake = v7_lake_paths(config.output_root).ensure() if config.use_lake_layout else None
    legacy_root = Path(config.output_root)
    legacy_root.mkdir(parents=True, exist_ok=True)
    market_path = (lake.silver_market_panel / "market_panel.parquet") if lake else (legacy_root / "market_panel.parquet")
    _write_frame(result.frame, market_path)

    feature_path: Path | None = None
    feature_rows = 0
    if config.build_features:
        features = build_market_features(result.frame)
        feature_path = (lake.silver_market_panel / "market_features.parquet") if lake else (legacy_root / "market_features.parquet")
        _write_frame(features, feature_path)
        feature_rows = len(features)

    # legacy mirror so existing CLI/tests pointing at ``data/v7/market_panel.parquet`` keep working
    if lake is not None:
        mirror = legacy_root / "market_panel.parquet"
        if mirror.resolve() != market_path.resolve():
            _write_frame(result.frame, mirror)

    manifest = build_manifest_for_frame(
        dataset_name="market_panel",
        vendor="qlib",
        frame=result.frame,
        output_paths=[_existing_written_path(market_path), _existing_written_path(feature_path)] if feature_path else [_existing_written_path(market_path)],
        start_date=config.start_date,
        end_date=config.end_date,
        symbols=config.symbols,
        universe=config.universe,
        required_columns=QLIB_MARKET_COLUMNS,
        pit_violation_count=int(report.get("pit_violation_count", 0)),
        warnings=tuple(result.warnings),
        extra={
            "provider_uri": str(provider_path),
            "region": config.region,
            "feature_rows": int(feature_rows),
            "schema_report": report,
            "available_at_policy": "next-trading-row availability for close-derived features",
        },
    )
    manifest_path = (lake.manifests / "market_panel.json") if lake else (legacy_root / "market_panel_manifest.json")
    manifest.write(manifest_path)

    return {
        "status": "passed",
        "download_command": QLIB_CN_DOWNLOAD_COMMAND,
        "config": asdict(config),
        "market_path": str(_existing_written_path(market_path)),
        "market_rows": int(len(result.frame)),
        "feature_path": str(_existing_written_path(feature_path)) if feature_path else None,
        "feature_rows": int(feature_rows),
        "manifest_path": str(manifest_path),
        "schema_report": report,
        "warnings": list(result.warnings),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def _existing_written_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path
