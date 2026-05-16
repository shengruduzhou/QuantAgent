"""V7 valuation bootstrap.

Pulls one or more AkShare daily valuation snapshots into the silver
valuation tier and emits a manifest. The bootstrap also accepts a
pre-collected CSV/parquet snapshot for environments without network
access — callers point ``--csv-snapshot`` at the local file and the
bootstrap copies it into the lake while still emitting a manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.akshare_valuation_provider import (
    AKSHARE_VALUATION_REQUIRED_COLUMNS,
    AkShareValuationProvider,
)
from quantagent.data.providers.base import ProviderRequest


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
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    if config.csv_snapshot:
        path = Path(config.csv_snapshot)
        if not path.exists():
            raise FileNotFoundError(f"valuation csv snapshot not found: {path}")
        snapshot = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        if "trade_date" not in snapshot.columns:
            raise ValueError("csv snapshot must include trade_date")
        if "available_at" not in snapshot.columns:
            snapshot["available_at"] = snapshot["trade_date"]
        frames.append(snapshot)
    else:
        provider = AkShareValuationProvider(allow_network=config.allow_network)
        for as_of in config.as_of_dates:
            request = ProviderRequest("", as_of, symbols=config.symbols) if config.symbols else None
            result = provider.snapshot(as_of, request=request)
            warnings.extend(result.warnings)
            if not result.frame.empty:
                frames.append(result.frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = lake.silver_valuation / config.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        warnings.append("valuation_snapshot_empty")
        frame.to_csv(output_path.with_suffix(".csv"), index=False)
        manifest_target = output_path.with_suffix(".csv")
    else:
        try:
            frame.to_parquet(output_path, index=False)
            manifest_target = output_path
        except Exception:
            output_path = output_path.with_suffix(".csv")
            frame.to_csv(output_path, index=False)
            manifest_target = output_path
    manifest = build_manifest_for_frame(
        dataset_name="valuation",
        vendor="akshare" if not config.csv_snapshot else "local_csv",
        frame=frame,
        output_paths=[manifest_target],
        symbols=config.symbols,
        required_columns=AKSHARE_VALUATION_REQUIRED_COLUMNS,
        warnings=tuple(warnings),
        extra={
            "as_of_dates": list(config.as_of_dates),
            "csv_snapshot": config.csv_snapshot,
        },
    )
    manifest_path = lake.manifests / "valuation.json"
    manifest.write(manifest_path)
    return {
        "status": "passed" if not frame.empty else "empty",
        "config": asdict(config),
        "output_path": str(manifest_target),
        "manifest_path": str(manifest_path),
        "rows": int(len(frame)),
        "warnings": warnings,
    }
