"""The akshare release is checked against the pyproject pin before any call.

Round 29 found the venv on 1.18.60 while pyproject pinned 1.18.84; the two
return different Tencent shapes (volume named ``amount`` vs volume + turnover +
CNY amount), i.e. different units for the same call.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
from quantagent.data.providers.akshare_version import (
    AKSHARE_PINNED_VERSION_FALLBACK,
    AKSHARE_VERSION_OVERRIDE_ENV,
    AkShareVersionMismatch,
    pinned_akshare_version,
    require_pinned_akshare,
)
from quantagent.data.providers.base import ProviderRequest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(AKSHARE_VERSION_OVERRIDE_ENV, raising=False)


def test_pyproject_pins_one_version_and_the_fallback_mirrors_it():
    pins = set(re.findall(r'"akshare==([^"]+)"', PYPROJECT.read_text(encoding="utf-8")))
    assert len(pins) == 1
    assert pinned_akshare_version(PYPROJECT) == pins.pop() == AKSHARE_PINNED_VERSION_FALLBACK


def test_matching_version_returns_provenance():
    ak = types.SimpleNamespace(__version__=AKSHARE_PINNED_VERSION_FALLBACK)
    provenance = require_pinned_akshare(ak)
    assert provenance["akshare_version_matches_pin"] is True
    assert provenance["akshare_version_drift_acknowledged"] is False


def test_mismatch_fails_loudly_with_the_fix():
    ak = types.SimpleNamespace(__version__="1.18.60")
    with pytest.raises(AkShareVersionMismatch) as info:
        require_pinned_akshare(ak)
    message = str(info.value)
    assert "1.18.60" in message and f"akshare=={AKSHARE_PINNED_VERSION_FALLBACK}" in message
    assert "pip install" in message and AKSHARE_VERSION_OVERRIDE_ENV in message


def test_explicit_override_naming_the_installed_version_is_recorded(monkeypatch):
    monkeypatch.setenv(AKSHARE_VERSION_OVERRIDE_ENV, "1.18.60")
    provenance = require_pinned_akshare(types.SimpleNamespace(__version__="1.18.60"))
    assert provenance["akshare_version_matches_pin"] is False
    assert provenance["akshare_version_drift_acknowledged"] is True


def test_stale_override_does_not_unlock_a_different_release(monkeypatch):
    monkeypatch.setenv(AKSHARE_VERSION_OVERRIDE_ENV, "1.18.60")
    with pytest.raises(AkShareVersionMismatch, match="does not name the installed"):
        require_pinned_akshare(types.SimpleNamespace(__version__="1.19.1"))


def test_provider_refuses_before_any_vendor_call(monkeypatch):
    calls: list[str] = []
    module = types.SimpleNamespace(
        __version__="1.18.60",
        stock_zh_a_hist=lambda **kw: calls.append("east_money"),
    )
    monkeypatch.setitem(sys.modules, "akshare", module)
    with pytest.raises(AkShareVersionMismatch):
        AkShareLiveProvider(allow_network=True).daily_ohlcv(
            ProviderRequest("2024-01-02", "2024-01-03", symbols=("600519.SH",))
        )
    assert calls == []
    health = AkShareLiveProvider(allow_network=True).health_check()
    assert health["status"] == "version_mismatch"
    assert health["akshare_pinned_version"] == AKSHARE_PINNED_VERSION_FALLBACK
