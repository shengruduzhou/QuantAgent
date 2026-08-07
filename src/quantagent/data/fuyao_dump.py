"""Bounded, auditable downloader for Fuyao full-market Parquet dumps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Final

import pyarrow.parquet as pq
import requests

from quantagent.data.manifest import DataManifest, hash_file
from quantagent.data.providers.fuyao_provider import FuyaoProvider


DUMP_ENDPOINTS: Final[dict[str, str]] = {
    "daily-k": "/api/dump/market-dumps/daily-k/download-url",
    "daily-k-10d": "/api/dump/market-dumps/daily-k-10d/download-url",
    "adjustment-factors": "/api/dump/market-dumps/adjustment-factors/download-url",
}

_DAILY_REQUIRED = frozenset(
    {"thscode", "date_ms", "open_price", "high_price", "low_price", "close_price", "volume", "turnover"}
)
_ACTION_REQUIRED = frozenset(
    {"thscode", "ex_date_ms", "dividend_per_share", "per_share_bonus", "allotment_ratio", "allotment_price"}
)


@dataclass(frozen=True)
class FuyaoDumpResult:
    dataset: str
    output: Path
    manifest: Path
    rows: int
    columns: tuple[str, ...]
    sha256: str


def _stream_https(url: str, output: Path, timeout: float) -> None:
    if not url.startswith("https://"):
        raise ValueError("Fuyao presigned dump URL must use HTTPS")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".partial",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8 << 20):
                    if chunk:
                        handle.write(chunk)
            handle.flush()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)


def download_fuyao_market_dump(
    dataset: str,
    output: str | Path,
    *,
    manifest_path: str | Path | None = None,
    allow_network: bool = False,
    timeout: float = 180.0,
    provider: FuyaoProvider | None = None,
) -> FuyaoDumpResult:
    """Download one official full-market dump without persisting the signed URL.

    The downloaded object is a **raw** artifact. It is schema-checked and hashed,
    but it deliberately receives ``pit_violation_count=None`` because raw dumps
    do not carry QuantAgent's canonical ``available_at`` column yet. Downstream
    silver normalisation owns that PIT audit.
    """
    if dataset not in DUMP_ENDPOINTS:
        raise ValueError(f"unsupported Fuyao market dump: {dataset}")
    if not allow_network:
        raise PermissionError("Fuyao market dump network access is fail-closed")

    target = Path(output)
    client = provider or FuyaoProvider(allow_network=True)
    signed = client.get_capability(DUMP_ENDPOINTS[dataset])
    presigned_url = str(signed.get("presigned_url") or "").strip()
    if not presigned_url:
        raise RuntimeError("Fuyao did not return presigned_url")

    # The URL is short-lived capability data. Use it immediately and then let it
    # fall out of scope; never include it in logs, manifests or return objects.
    _stream_https(presigned_url, target, timeout)

    parquet = pq.ParquetFile(target)
    columns = tuple(parquet.schema_arrow.names)
    rows = int(parquet.metadata.num_rows)
    required = _DAILY_REQUIRED if dataset.startswith("daily-k") else _ACTION_REQUIRED
    missing = tuple(sorted(required - set(columns)))
    if rows <= 0 or missing:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fuyao dump failed schema validation: rows={rows} missing={list(missing)}"
        )

    digest = hash_file(target)
    manifest_target = Path(manifest_path) if manifest_path else target.with_suffix(target.suffix + ".manifest.json")
    warnings = ["raw_market_dump_requires_canonical_normalization_before_model_use"]
    warnings.append(
        "raw_daily_prices_are_unadjusted"
        if dataset.startswith("daily-k")
        else "adjustment_events_require_asof_application"
    )
    manifest = DataManifest(
        dataset_name=f"fuyao_{dataset.replace('-', '_')}_raw",
        vendor="hithink_fuyao",
        fetch_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_paths=(str(target),),
        output_paths=(str(target),),
        row_count=rows,
        column_count=len(columns),
        content_hashes={str(target): digest},
        missing_columns=(),
        pit_violation_count=None,
        warnings=tuple(warnings),
        quality_status="warning",
        extra={
            "dataset": dataset,
            "schema": list(columns),
            "source_endpoint": DUMP_ENDPOINTS[dataset],
            "presigned_url_persisted": False,
        },
    )
    manifest.write(manifest_target)
    return FuyaoDumpResult(
        dataset=dataset,
        output=target,
        manifest=manifest_target,
        rows=rows,
        columns=columns,
        sha256=digest,
    )


__all__ = ["DUMP_ENDPOINTS", "FuyaoDumpResult", "download_fuyao_market_dump"]
