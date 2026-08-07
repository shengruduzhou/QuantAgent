#!/usr/bin/env python3
"""Download official Fuyao/HiThink full-market Parquet dumps.

Use this path for full-universe history.  The upstream contract explicitly says
not to loop over ~5000 securities with ``prices/historical`` when three market
dump endpoints provide the full 10-year daily panel, the recent 10-day delta,
and all adjustment-factor events.

The presigned object URL is intentionally never printed or persisted.  API keys
are read from ``HITHINK_FINANCE_API_KEY`` through :class:`FuyaoProvider`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pyarrow.parquet as pq
import requests

from quantagent.data.manifest import DataManifest, hash_file
from quantagent.data.providers.fuyao_provider import FuyaoProvider


DUMP_ENDPOINTS = {
    "daily-k": "/api/dump/market-dumps/daily-k/download-url",
    "daily-k-10d": "/api/dump/market-dumps/daily-k-10d/download-url",
    "adjustment-factors": "/api/dump/market-dumps/adjustment-factors/download-url",
}


def _download(url: str, output: Path, timeout: float) -> None:
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


def _write_manifest(dataset: str, output: Path, manifest_path: Path) -> None:
    parquet = pq.ParquetFile(output)
    schema = list(parquet.schema_arrow.names)
    expected = (
        {"thscode", "date_ms", "open_price", "high_price", "low_price", "close_price", "volume", "turnover"}
        if dataset.startswith("daily-k")
        else {"thscode", "ex_date_ms", "dividend_per_share", "per_share_bonus", "allotment_ratio", "allotment_price"}
    )
    missing = tuple(sorted(expected - set(schema)))
    status = "failed" if missing or parquet.metadata.num_rows <= 0 else "warning"
    manifest = DataManifest(
        dataset_name=f"fuyao_{dataset.replace('-', '_')}_raw",
        vendor="hithink_fuyao",
        fetch_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_paths=(str(output),),
        output_paths=(str(output),),
        row_count=int(parquet.metadata.num_rows),
        column_count=len(schema),
        content_hashes={str(output): hash_file(output)},
        missing_columns=missing,
        pit_violation_count=None,
        warnings=(
            "raw_market_dump_requires_canonical_normalization_before_model_use",
            "raw_daily_prices_are_unadjusted" if dataset.startswith("daily-k") else "adjustment_events_require_asof_application",
        ),
        quality_status=status,
        extra={
            "dataset": dataset,
            "schema": schema,
            "source_endpoint": DUMP_ENDPOINTS[dataset],
            "presigned_url_persisted": False,
        },
    )
    manifest.write(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DUMP_ENDPOINTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    if not args.allow_network:
        raise SystemExit("network access is fail-closed; pass --allow-network explicitly")

    provider = FuyaoProvider(allow_network=True)
    data = provider.get_capability(DUMP_ENDPOINTS[args.dataset])
    presigned_url = str(data.get("presigned_url") or "").strip()
    if not presigned_url.startswith("https://"):
        raise SystemExit("Fuyao did not return a valid presigned HTTPS download URL")

    _download(presigned_url, args.output, timeout=args.timeout)
    # Do not print or store the presigned URL: it is short-lived capability data.
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    _write_manifest(args.dataset, args.output, manifest_path)
    parquet = pq.ParquetFile(args.output)
    print(
        f"dataset={args.dataset} rows={parquet.metadata.num_rows} "
        f"output={args.output} manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
