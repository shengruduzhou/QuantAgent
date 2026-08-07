#!/usr/bin/env python3
"""Download official Fuyao/HiThink full-market Parquet dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantagent.data.fuyao_dump import DUMP_ENDPOINTS, download_fuyao_market_dump


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DUMP_ENDPOINTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    result = download_fuyao_market_dump(
        args.dataset,
        args.output,
        manifest_path=args.manifest,
        allow_network=args.allow_network,
        timeout=args.timeout,
    )
    print(
        f"dataset={result.dataset} rows={result.rows} "
        f"output={result.output} manifest={result.manifest} sha256={result.sha256[:12]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
