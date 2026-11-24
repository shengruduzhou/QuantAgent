"""The versioned agent organization (.claude/) stays declarative and secret-free.

Only Markdown definitions and the project ``settings.json`` may be tracked under
``.claude/`` (the .gitignore negations make whole directories trackable, so a
stray credential file there would otherwise be committable).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_claude_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", ".claude"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def test_only_markdown_and_settings_are_tracked():
    tracked = _tracked_claude_files()
    bad = [p for p in tracked if not (p.endswith(".md") or p == ".claude/settings.json")]
    assert not bad, f"unexpected tracked files under .claude/: {bad}"


def test_settings_deny_secrets_and_force_push():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    deny = set(settings["permissions"]["deny"])
    assert {"Read(**/.env)", "Bash(git push * --force*)", "Bash(git stash*)"} <= deny


def test_agent_definitions_have_required_frontmatter():
    agents = sorted((ROOT / ".claude" / "agents").glob("*.md"))
    assert agents
    for path in agents:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), path
        front = text.split("---\n", 2)[1]
        assert f"name: {path.stem}\n" in front, path
        assert "description:" in front, path


def test_auditors_cannot_edit_or_delegate():
    for name in ("code-security-auditor", "math-stat-auditor", "qa-release"):
        front = (ROOT / ".claude" / "agents" / f"{name}.md").read_text(encoding="utf-8").split("---\n", 2)[1]
        line = next(l for l in front.splitlines() if l.startswith("disallowedTools:"))
        for tool in ("Edit", "Write", "Agent"):
            assert tool in line, (name, tool)
