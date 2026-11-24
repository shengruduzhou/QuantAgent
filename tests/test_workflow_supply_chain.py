"""Supply-chain policy for GitHub Actions workflows.

* Every third-party action is pinned to a full 40-hex commit SHA (a moving tag
  such as ``@v6`` can be re-pointed by whoever controls the action repo).
* Every ``pip install`` outside the Dependency Drift workflow resolves against
  ``requirements/ci-constraints.txt`` so scheduled smokes run the same tested
  versions as required CI (Cycle 32: Market Data Smoke and Factor Research
  installed unconstrained, so a release inside the declared ranges could break
  them independently of required CI).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))
USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)", re.MULTILINE)
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")
DRIFT_WORKFLOW = "dependency-drift.yml"


def test_workflows_exist():
    assert WORKFLOWS, "no workflow files found"


def test_actions_are_pinned_to_commit_sha():
    unpinned = []
    for path in WORKFLOWS:
        for ref in USES.findall(path.read_text(encoding="utf-8")):
            if ref.startswith("./"):
                continue
            if not PINNED.match(ref):
                unpinned.append(f"{path.name}: {ref}")
    assert not unpinned, f"actions not pinned to a commit SHA: {unpinned}"


def test_pip_installs_use_ci_constraints():
    offenders = []
    for path in WORKFLOWS:
        if path.name == DRIFT_WORKFLOW:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bpip install\b", line) and "-c requirements/ci-constraints.txt" not in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, f"unconstrained pip installs: {offenders}"


def test_extra_packages_are_exact_pins():
    loose = []
    for path in WORKFLOWS:
        if path.name == DRIFT_WORKFLOW:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'"([A-Za-z0-9_.-]+)\s*(>=|~=|>|<)[^"]*"', text):
            loose.append(f"{path.name}: {match.group(0)}")
    assert not loose, f"extra packages must be exact pins (==): {loose}"
