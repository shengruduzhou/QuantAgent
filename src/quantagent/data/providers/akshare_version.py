"""The akshare release is part of the data contract, so it is pinned and checked.

Measured consequence of drift (Round 29): akshare 1.18.60 returns Tencent daily
bars as six columns with volume in a column named ``amount`` and no turnover,
while 1.18.69+ returns volume + turnover + CNY amount and skips its own x100 for
``sz000`` symbols. The same call therefore produced different units depending
on which release happened to be installed, and the venv (1.18.60) silently
differed from the pin in pyproject.toml (1.18.84).

``require_pinned_akshare`` fails loudly on any mismatch. An operator who has
deliberately audited a different release may acknowledge it by setting
``QUANTAGENT_AKSHARE_VERSION_OVERRIDE`` to the *installed* version string; the
acknowledgement is recorded in provenance, and a stale override (naming a
version that is not the installed one) does not unlock anything.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from quantagent.data.providers.base import ProviderUnavailable

AKSHARE_VERSION_OVERRIDE_ENV = "QUANTAGENT_AKSHARE_VERSION_OVERRIDE"
#: Mirrors pyproject.toml; used only when the package runs without its source
#: tree. tests/data/test_akshare_version_pin.py keeps the two in sync.
AKSHARE_PINNED_VERSION_FALLBACK = "1.18.84"

_PIN_PATTERN = re.compile(r'"akshare==([0-9][0-9A-Za-z.\-+]*)"')


class AkShareVersionMismatch(ProviderUnavailable):
    """Installed akshare differs from the release the data contract was audited on."""


def _find_pyproject() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                return None
            if 'name = "quantagent"' in text:
                return candidate
    return None


def pinned_akshare_version(pyproject: Path | None = None) -> str:
    """The single ``akshare==X`` pin declared in pyproject.toml."""
    path = pyproject or _find_pyproject()
    if path is None:
        return AKSHARE_PINNED_VERSION_FALLBACK
    pins = set(_PIN_PATTERN.findall(path.read_text(encoding="utf-8")))
    if len(pins) != 1:
        raise AkShareVersionMismatch(
            f"pyproject.toml must pin exactly one akshare version, found {sorted(pins)}"
        )
    return pins.pop()


def installed_akshare_version(ak: object) -> str:
    return str(getattr(ak, "__version__", "") or "unknown")


def require_pinned_akshare(ak: object) -> dict[str, object]:
    """Return version provenance, or raise with an actionable message."""
    installed = installed_akshare_version(ak)
    pinned = pinned_akshare_version()
    override = os.environ.get(AKSHARE_VERSION_OVERRIDE_ENV, "").strip()
    provenance: dict[str, object] = {
        "akshare_version": installed,
        "akshare_pinned_version": pinned,
        "akshare_version_matches_pin": installed == pinned,
        "akshare_version_drift_acknowledged": False,
    }
    if installed == pinned:
        return provenance
    if override and override == installed:
        provenance["akshare_version_drift_acknowledged"] = True
        return provenance
    hint = (
        f"set {AKSHARE_VERSION_OVERRIDE_ENV}={installed} only after auditing that "
        "release's units/shapes (tests/data/test_akshare_unit_truth_per_response.py)"
    )
    if override:
        hint = (f"{AKSHARE_VERSION_OVERRIDE_ENV}={override!r} does not name the installed "
                f"version {installed!r}; " + hint)
    raise AkShareVersionMismatch(
        f"installed akshare {installed} differs from the pinned akshare=={pinned} "
        "(pyproject.toml [data]); vendor payload shapes and volume units change "
        f"between releases. Fix: `python -m pip install akshare=={pinned}`, or {hint}."
    )
