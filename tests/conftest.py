"""Keep the test suite hermetic.

The repository ships a `.env` holding real credentials and
``QUANTAGENT_LLM_ENABLED=1`` / ``QUANTAGENT_LLM_ALLOW_NETWORK=1``.
``quantagent.agents.llm_skill_client._load_dotenv_once`` loads that file into
``os.environ`` process-wide, so importing an unrelated LLM helper anywhere in a
run silently arms live network access and injects provider API keys for every
test that follows.

Two things went wrong because of it:

* ``tests/test_v7_theme_research_pipeline.py`` made real calls to the Gemini
  endpoint. A full-suite run sat in ``ssl.py`` for ten minutes with an open
  socket and had to be interrupted, so roughly 172 tests never executed.
* ``test_connection_vault_never_returns_or_persists_secret`` failed because
  ``ConnectionManager`` reports a provider as connected when its variable is
  present in ``os.environ``, so a disconnected session still read as connected.

Disarming the flags here fixes both without touching how the application
behaves outside tests. Set ``QUANTAGENT_TEST_ALLOW_NETWORK=1`` to opt a run back
in when deliberately exercising a live provider.
"""

from __future__ import annotations

import os

#: Variables the shipped `.env` injects that must not reach a test run. They are
#: blanked rather than deleted: ``load_dotenv(override=False)`` skips names that
#: already exist in ``os.environ``, so an empty value is what actually keeps the
#: file from re-injecting them later in the session. Every consumer reads these
#: through ``os.getenv`` truthiness, for which "" and unset are equivalent.
_DISARMED = (
    "QUANTAGENT_LLM_ENABLED",
    "QUANTAGENT_LLM_ALLOW_NETWORK",
    "TICKFLOW_API_KEY",
    "TICKFLOW_API_ENDPOINT",
)


def _disarm_ambient_environment() -> None:
    if os.environ.get("QUANTAGENT_TEST_ALLOW_NETWORK") == "1":
        return
    for name in _DISARMED:
        os.environ[name] = ""


# Runs at conftest import, before collection — a fixture would be too late,
# because importing a test module can already trigger the dotenv load.
_disarm_ambient_environment()


def _isolate_runtime_home() -> None:
    """Point ``QUANTAGENT_HOME`` at a throwaway directory for the whole run.

    Without this, any code path that resolves the default storage layout
    (``quant_paths()`` / ``ApiSettings`` defaults) writes into the operator's
    real ``runtime/``: Round 31 found ``auto-train-v7`` tests leaving a 14-row
    synthetic ``runtime/data/v7/labels.parquet`` that the Strategy Workbench
    then offered as its default labels, and importing ``services.quant_api.app``
    taking the real paper writer lock. Tests that genuinely read certified
    runtime artifacts reference them by explicit path, which this does not
    change. ``QUANTAGENT_TEST_USE_REAL_RUNTIME=1`` opts out.
    """
    if os.environ.get("QUANTAGENT_TEST_USE_REAL_RUNTIME") == "1":
        return
    import tempfile

    home = tempfile.mkdtemp(prefix="quantagent-test-home-")
    os.environ["QUANTAGENT_HOME"] = home
    os.environ.pop("QUANTAGENT_DATA_ROOT", None)


_isolate_runtime_home()
