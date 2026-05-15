"""Qlib CN bootstrap and market-panel export for V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.qlib_provider import QlibProvider, validate_qlib_market_schema
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
    metadata: dict[str, object] = field(default_factory=dict)


def build_qlib_market_panel(config: QlibBootstrapConfig) -> dict[str, object]:
    provider_path = Path(config.provider_uri)
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
    result = QlibProvider(config.provider_uri, config.region).daily_ohlcv(request)
    report = validate_qlib_market_schema(result.frame, as_of_date=config.end_date)
    if report["status"] != "passed":
        raise ValueError(f"Qlib market schema failed: {report}")

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    market_path = root / "market_panel.parquet"
    _write_frame(result.frame, market_path)
    feature_path: Path | None = None
    feature_rows = 0
    if config.build_features:
        features = build_market_features(result.frame)
        feature_path = root / "market_features.parquet"
        _write_frame(features, feature_path)
        feature_rows = len(features)

    return {
        "status": "passed",
        "download_command": QLIB_CN_DOWNLOAD_COMMAND,
        "config": asdict(config),
        "market_path": str(_existing_written_path(market_path)),
        "market_rows": int(len(result.frame)),
        "feature_path": str(_existing_written_path(feature_path)) if feature_path else None,
        "feature_rows": int(feature_rows),
        "schema_report": report,
        "warnings": list(result.warnings),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
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
