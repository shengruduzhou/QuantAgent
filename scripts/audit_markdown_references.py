#!/usr/bin/env python3
"""Report Markdown files that appear unreferenced; never delete automatically.

The script is intentionally read-only.  A document is a deletion candidate only
when its basename and repository-relative path are not referenced by any other
text file.  Audit/evidence documents can still be retained by policy even when
unreferenced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SKIP_PARTS = {
    ".git", ".venv", "venv", "runtime", "node_modules", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
TEXT_SUFFIXES = {
    ".md", ".py", ".toml", ".yml", ".yaml", ".json", ".txt", ".sh",
    ".ps1", ".service", ".timer", ".ts", ".tsx", ".js", ".jsx",
}
PROTECTED_NAMES = {
    "README.md", "AGENTS.md", "ACCEPTANCE_RULES.md", "EXPERIMENT_LEDGER.md",
    "HYPOTHESIS_REGISTRY.md", "BASELINE_TRUST_CLASSIFICATION.md",
    "HOLDOUT_CONTAMINATION_AUDIT.md", "EVALUATION_PROTOCOL_V2.md",
}


def _eligible(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in SKIP_PARTS for part in relative.parts)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def audit(root: Path) -> dict[str, object]:
    markdown = sorted(
        path for path in root.rglob("*.md") if path.is_file() and _eligible(path, root)
    )
    text_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and _eligible(path, root) and path.suffix.lower() in TEXT_SUFFIXES
    )
    corpus = {path: _read_text(path) for path in text_files}
    rows: list[dict[str, object]] = []
    for document in markdown:
        relative = document.relative_to(root).as_posix()
        basename = document.name
        references: list[str] = []
        for source, text in corpus.items():
            if source == document:
                continue
            if relative in text or basename in text:
                references.append(source.relative_to(root).as_posix())
        protected = basename in PROTECTED_NAMES
        rows.append(
            {
                "path": relative,
                "reference_count": len(references),
                "references": references,
                "protected": protected,
                "candidate": not references and not protected,
            }
        )
    return {
        "root": str(root),
        "markdown_count": len(rows),
        "candidate_count": sum(bool(row["candidate"]) for row in rows),
        "documents": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()
    payload = audit(args.root.resolve())
    if args.candidates_only:
        payload["documents"] = [
            row for row in payload["documents"] if row["candidate"]
        ]
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
