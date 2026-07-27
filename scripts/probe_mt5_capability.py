#!/usr/bin/env python3
"""Probe the local MetaTrader 5 runtime for genuine A-share market data.

Fail-closed: when no terminal answers, the probe still writes artifacts, and
every capability cell is CLIENT_UNAVAILABLE rather than a guess about brokers.

    python scripts/probe_mt5_capability.py \
        --output runtime/data/capabilities/mt5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantagent.data.providers import mt5_capability as probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="runtime/data/capabilities/mt5",
        help="directory for machine-readable probe artifacts",
    )
    parser.add_argument(
        "--dom-reads", type=int, default=5,
        help="how many times to read the market book before concluding it is empty",
    )
    args = parser.parse_args()

    result = probe.run_probe()
    written = probe.write_artifacts(result, args.output)

    summary = {
        "overall_classification": result.overall_classification,
        "genuine_a_share_symbols": result.genuine_a_share_symbols,
        "terminal_classification": result.terminal.classification,
        "os": f"{result.terminal.os_name} {result.terminal.os_release}",
        "package_importable": result.terminal.package_importable,
        "import_error": result.terminal.import_error,
        "account_available": result.account.available,
        "symbols_total": result.symbols_total,
        "artifacts": written,
        "notes": result.notes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
