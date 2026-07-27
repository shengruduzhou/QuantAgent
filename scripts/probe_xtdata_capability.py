#!/usr/bin/env python3
"""Probe QMT / xtquant (XtData) for tick and Level-2 entitlements.

Read-only: this script never imports the trader module and never touches the
execution gateway.

    python scripts/probe_xtdata_capability.py \
        --symbols 600000.SH,000001.SZ \
        --output runtime/data/capabilities/qmt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantagent.data.providers import xtdata_market_provider as xt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="600000.SH,000001.SZ")
    parser.add_argument("--output", default="runtime/data/capabilities/qmt")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    runtime, matrix = xt.probe_capability(symbols)
    written = xt.write_artifacts(runtime, matrix, args.output)

    summary = {
        "package_importable": runtime.package_importable,
        "xtdata_importable": runtime.xtdata_importable,
        "platform_supported": runtime.platform_supported,
        "client_connected": runtime.client_connected,
        "import_error": runtime.import_error,
        "connect_error": runtime.connect_error,
        "authorized_markets": runtime.authorized_markets,
        "native_extensions_sample": runtime.native_extensions[:8],
        "capability_summary": matrix.summary(),
        "artifacts": written,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
