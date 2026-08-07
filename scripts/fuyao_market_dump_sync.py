#!/usr/bin/env python3
"""Download and validate Fuyao full-market Parquet artifacts.

This is the preferred bulk path for the initial A-share data load and daily
refresh. It never prints or persists the short-lived presigned download URL.

Examples:
  AI_quant_venv/bin/python3 scripts/fuyao_market_dump_sync.py --allow-network --kind all
  AI_quant_venv/bin/python3 scripts/fuyao_market_dump_sync.py --allow-network --kind daily-k-10d

Credential:
  HITHINK_FINANCE_API_KEY=<Fuyao API key>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.fuyao import FUYAO_API_KEY_ENV, FuyaoClient  # noqa: E402
from quantagent.data.ashare.fuyao_dump import (  # noqa: E402
    DEFAULT_DUMP_ROOT,
    available_dump_kinds,
    download_dump,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="required: performs authenticated Fuyao download calls")
    parser.add_argument(
        "--kind",
        choices=[*available_dump_kinds(), "all"],
        default="all",
        help="artifact family; all downloads full daily, 10-session delta and adjustment events",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / DEFAULT_DUMP_ROOT,
        help="local artifact directory",
    )
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to download: --allow-network was not confirmed")
        return 2

    load_repo_env()
    api = FuyaoClient()
    if not api.configured:
        print(
            f"missing {FUYAO_API_KEY_ENV}; put it in repo .env or the process environment "
            "(never commit the key)"
        )
        return 3

    kinds = list(available_dump_kinds()) if args.kind == "all" else [args.kind]
    artifacts = []
    for kind in kinds:
        print(f"syncing Fuyao {kind} ...", flush=True)
        artifact = download_dump(kind, api=api, root=args.root)
        artifacts.append(artifact.as_dict())
        print(
            f"  {kind}: rows={artifact.rows:,} bytes={artifact.bytes:,} "
            f"path={artifact.path}",
            flush=True,
        )

    manifest = {
        "provider": "fuyao",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "credential_env": FUYAO_API_KEY_ENV,
        "presigned_url_persisted": False,
        "artifacts": artifacts,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
