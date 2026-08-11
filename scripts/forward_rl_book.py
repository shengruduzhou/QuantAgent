#!/usr/bin/env python3
"""Fail-closed tombstone for the retired RL forward-target exporter.

The historical exporter used a superseded reward/execution clock and could leave
``runtime/paper/forward/C_rl/targets_latest.csv`` plus ``last_update.json`` behind.
Disabling generation without retiring those files is unsafe because downstream
readers can keep consuming them indefinitely.

Every invocation therefore moves all legacy root-level ``targets_*.csv`` and
``last_update.json`` into a timestamped quarantine directory, writes an explicit
``BLOCKED.json`` tombstone, emits no replacement target, and exits non-zero.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


OUT_DIR = Path("runtime/paper/forward/C_rl")
_BLOCK_REASON = (
    "legacy RL forward targets were produced under a superseded reward/execution "
    "clock; target consumption is blocked until a separately audited current-"
    "signal inference path, contract-matched retraining, and fresh forward-shadow "
    "acceptance exist"
)


def _retire_stale_outputs(out_dir: Path) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    quarantine = out_dir / "quarantine" / f"pre_clock_audit_{stamp}"
    candidates = sorted(out_dir.glob("targets_*.csv")) if out_dir.exists() else []
    last_update = out_dir / "last_update.json"
    if last_update.exists():
        candidates.append(last_update)

    moved: list[str] = []
    if candidates:
        quarantine.mkdir(parents=True, exist_ok=True)
        for source in candidates:
            destination = quarantine / source.name
            source.replace(destination)
            moved.append(str(destination))

    out_dir.mkdir(parents=True, exist_ok=True)
    tombstone = {
        "schema_version": "quantagent.rl.forward_tombstone.v1",
        "status": "RL_FORWARD_BLOCKED",
        "generated_at": timestamp.isoformat(timespec="seconds"),
        "reason": _BLOCK_REASON,
        "stale_outputs_quarantined": moved,
        "replacement_target_emitted": False,
        "production_eligible": False,
    }
    (out_dir / "BLOCKED.json").write_text(
        json.dumps(tombstone, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tombstone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-start", default=None)
    parser.add_argument("--warmup-start", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    tombstone = _retire_stale_outputs(Path(args.output_dir))
    print(json.dumps(tombstone, ensure_ascii=False), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
