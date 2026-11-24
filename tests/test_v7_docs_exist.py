"""The authoritative documentation set exists, states the safety boundary, and
does not link to files that are gone (Round 31 consolidated 185 Markdown files
into this small set)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

AUTHORITATIVE = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "docs/ARCHITECTURE.md",
    "docs/DATA.md",
    "docs/GOVERNANCE.md",
    "docs/OPERATIONS.md",
    "docs/RESEARCH.md",
    "docs/ROADMAP.md",
    "docs/audits/HISTORY.md",
]


def test_authoritative_docs_exist_and_cover_required_boundaries():
    texts = []
    for name in AUTHORITATIVE:
        path = REPO / name
        assert path.exists(), name
        text = path.read_text(encoding="utf-8")
        assert len(text) > 200, name
        texts.append(text)
    combined = "\n".join(texts)
    for term in [
        "EvidenceRecord",
        "Point-in-Time",
        "target_weights",
        "OrderManager",
        "T+1",
        "LIVE_DISABLED",
        "configs/quarantined_windows.json",
        "scripts/baseline_protocol.py",
        "DO NOT ENABLE",
    ]:
        assert term in combined, term
    assert (REPO / "configs/v7.default.yaml").exists()


_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_local_links_in_authoritative_docs_resolve():
    broken = []
    for name in AUTHORITATIVE:
        path = REPO / name
        for target in _LINK.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{name} -> {target}")
    assert broken == []
