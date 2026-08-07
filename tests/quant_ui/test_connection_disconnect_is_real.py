"""An explicit disconnect must actually disconnect.

The repository ships a `.env` holding real provider credentials, and
`quantagent.agents.llm_skill_client._load_dotenv_once` loads it into
`os.environ` process-wide. `ConnectionManager` treated "the variable exists in
os.environ" as "connected", so after an operator disconnected a provider in the
UI the manager still reported `connected: true` and still handed the credential
to jobs. The disconnect button did nothing an operator could rely on.

An operator's explicit action has to win over ambient configuration.
"""

from __future__ import annotations

import pytest

from services.quant_api.services.connections import ConnectionManager

_PROVIDER = "tickflow"
_VARIABLE = "TICKFLOW_API_KEY"


@pytest.fixture
def ambient_credential(monkeypatch):
    """Simulate the `.env` key that dotenv injects process-wide."""
    monkeypatch.setenv(_VARIABLE, "ambient-key-from-dotenv")
    monkeypatch.setenv("TICKFLOW_API_ENDPOINT", "https://ambient.example")


def _status(manager: ConnectionManager) -> dict:
    return next(item for item in manager.list() if item["id"] == _PROVIDER)


def test_ambient_environment_still_counts_as_connected_until_disconnected(ambient_credential):
    manager = ConnectionManager()

    status = _status(manager)
    assert status["connected"] is True
    assert status["source"] == "environment"


def test_disconnect_overrides_an_ambient_environment_credential(ambient_credential):
    manager = ConnectionManager()
    assert _status(manager)["connected"] is True

    manager.disconnect(_PROVIDER)

    status = _status(manager)
    assert status["connected"] is False, "disconnect must win over the ambient .env value"
    assert status["source"] == "none"


def test_a_disconnected_provider_hands_out_no_credentials(ambient_credential):
    manager = ConnectionManager()
    assert manager.environment_for({_PROVIDER}).get(_VARIABLE)
    assert manager.has_variable(_VARIABLE) is True

    manager.disconnect(_PROVIDER)

    # A job launched after the disconnect must not receive the credential.
    assert manager.environment_for({_PROVIDER}) == {}
    assert manager.has_variable(_VARIABLE) is False


def test_reconnecting_restores_the_provider(ambient_credential):
    manager = ConnectionManager()
    manager.disconnect(_PROVIDER)
    assert _status(manager)["connected"] is False

    manager.connect(_PROVIDER, {_VARIABLE: "session-secret-value"})

    status = _status(manager)
    assert status["connected"] is True
    assert status["source"] == "session"
    assert manager.environment_for({_PROVIDER})[_VARIABLE] == "session-secret-value"


def test_disconnecting_one_provider_leaves_others_alone(ambient_credential, monkeypatch):
    manager = ConnectionManager()
    others = [item["id"] for item in manager.list() if item["id"] != _PROVIDER]

    manager.disconnect(_PROVIDER)

    for provider_id in others:
        entry = next(item for item in manager.list() if item["id"] == provider_id)
        # Untouched providers keep whatever state they had; only the
        # disconnected one is suppressed.
        assert entry["id"] != _PROVIDER
    assert _status(manager)["connected"] is False
