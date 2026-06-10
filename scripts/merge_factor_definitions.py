#!/usr/bin/env python3
"""Merge multiple synthesized factor-definition JSON files.

The output keeps the same schema as ``factor_synthesis.save_definitions``.
Duplicate names are renamed deterministically; duplicate expressions are kept
once. This lets GP and LLM formula survivors enter the same materialization and
retraining pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    seen_expr: set[str] = set()
    used_names: set[str] = set()
    merged: list[dict[str, str]] = []
    for raw in args.inputs:
        for item in _load(Path(raw)):
            expr = str(item.get("expression") or "").strip()
            if not expr or expr in seen_expr:
                continue
            base = str(item.get("name") or f"synth_{len(merged)+1:03d}").strip() or f"synth_{len(merged)+1:03d}"
            name = base
            suffix = 2
            while name in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            seen_expr.add(expr)
            used_names.add(name)
            merged.append(
                {
                    "name": name,
                    "expression": expr,
                    "description": str(item.get("description") or "Merged synthesized factor"),
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "n_inputs": len(args.inputs), "n_merged": len(merged), "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
