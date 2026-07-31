"""Pause / resume control plane and the T+1 research command contract."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from services.quant_api.app import create_app
from services.quant_api.services.container import ServiceContainer
from services.quant_api.services.jobs import COMMANDS, JobManager


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def _wait_for(manager: JobManager, job_id: str, status: str, timeout: float = 12.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record and record["status"] == status:
            return record
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} never reached {status}; last status "
        f"{(manager.get(job_id) or {}).get('status')}"
    )


@pytest.fixture
def long_running_job(quant_ui_settings, monkeypatch):
    """A governed command rewired to a slow no-op so pause/resume is observable."""
    spec = dict(COMMANDS["audit-u0-adjustment-forensics"])
    script = quant_ui_settings.project_root / "scripts" / "slow_fixture.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import sys, time\n"
        "for index in range(400):\n"
        "    print(f'[{index} / 400] working', flush=True)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    spec["entrypoint"] = "scripts/slow_fixture.py"
    spec["fixed_outputs"] = ()
    monkeypatch.setitem(COMMANDS, "audit-u0-adjustment-forensics", spec)
    container = ServiceContainer.create(quant_ui_settings)
    job = container.jobs.submit("data", "audit-u0-adjustment-forensics", {})
    _wait_for(container.jobs, job["id"], "running")
    yield container, job["id"]
    # Teardown must never leave a SIGSTOP'd child behind: a stopped process
    # ignores SIGTERM, and the log reader would block on its stdout forever.
    try:
        container.jobs.purge(job["id"], delete_outputs=False)
    except (KeyError, ValueError):
        pass


# --------------------------------------------------------------------------- #
# Pause / resume                                                              #
# --------------------------------------------------------------------------- #


def test_pause_suspends_a_running_job(long_running_job) -> None:
    container, job_id = long_running_job
    paused = container.jobs.pause(job_id)
    assert paused["status"] == "paused"
    assert "memory retained" in paused["message"]


def test_a_paused_job_stops_making_progress(long_running_job) -> None:
    container, job_id = long_running_job
    container.jobs.pause(job_id)
    first = container.jobs.get(job_id)["progress"]
    time.sleep(0.6)
    second = container.jobs.get(job_id)["progress"]
    assert first == second, "a paused process must not advance"


def test_resume_continues_the_same_process(long_running_job) -> None:
    container, job_id = long_running_job
    container.jobs.pause(job_id)
    before = container.jobs.get(job_id)["progress"]
    resumed = container.jobs.resume(job_id)
    assert resumed["status"] == "running"
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if (container.jobs.get(job_id)["progress"] or 0) > (before or 0):
            break
        time.sleep(0.05)
    else:  # pragma: no cover
        raise AssertionError("resumed job never advanced")


def test_pausing_a_job_twice_is_rejected(long_running_job) -> None:
    container, job_id = long_running_job
    container.jobs.pause(job_id)
    with pytest.raises(ValueError, match="only a running job"):
        container.jobs.pause(job_id)


def test_resuming_a_job_that_is_not_paused_is_rejected(long_running_job) -> None:
    container, job_id = long_running_job
    with pytest.raises(ValueError, match="only a paused job"):
        container.jobs.resume(job_id)


def test_cancelling_a_paused_job_actually_terminates_it(long_running_job) -> None:
    """SIGTERM alone never reaches a suspended process; cancel must SIGCONT first.

    Without the continue, the process stays stopped indefinitely and the log
    reader blocks on its stdout pipe — the job never reaches a terminal state.
    """
    container, job_id = long_running_job
    container.jobs.pause(job_id)
    container.jobs.cancel(job_id)
    record = _wait_for(container.jobs, job_id, "cancelled", timeout=10.0)
    assert record["status"] == "cancelled"


def test_purging_a_paused_job_does_not_block(long_running_job) -> None:
    container, job_id = long_running_job
    container.jobs.pause(job_id)
    started = time.monotonic()
    result = container.jobs.purge(job_id, delete_outputs=False)
    assert result["status"] == "purged"
    # purge() waits on the process; if the continue is missing it burns the
    # full 5s timeout and then has to SIGKILL.
    assert time.monotonic() - started < 4.0


def test_pause_and_resume_are_reachable_over_http(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    assert request(app, "POST", "/api/jobs/does-not-exist/pause").status_code == 404
    assert request(app, "POST", "/api/jobs/does-not-exist/resume").status_code == 404


def test_pausing_an_unknown_job_raises_key_error(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    with pytest.raises(KeyError):
        container.jobs.pause("job_missing")


# --------------------------------------------------------------------------- #
# T+1 intraday research command                                               #
# --------------------------------------------------------------------------- #


def test_t_plus_one_research_requires_the_sellable_inventory_input() -> None:
    """Without holdings, the backtest would invent T+0 capability A-share lacks."""
    spec = COMMANDS["research-intraday-t-trading"]
    assert "holdings_csv" in spec["required"]
    assert "holdings_csv" in spec["path_inputs"]
    assert spec["type"] == "t-plus-one-research"


def test_t_plus_one_research_declares_a_cost_model() -> None:
    allowed = COMMANDS["research-intraday-t-trading"]["allowed"]
    for name in ("slippage_bps", "spread_bps", "commission_rate", "maker_only"):
        assert name in allowed, f"{name} must be configurable; a single cost point is not evidence"


def test_t_plus_one_research_rejects_a_missing_holdings_file(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    with pytest.raises(ValueError, match="holdings_csv"):
        container.jobs.validate(
            "t-plus-one-research",
            "research-intraday-t-trading",
            {
                "minute_dir": "runtime/data/v7/silver/market_panel",
                "market_panel": "runtime/data/v7/silver/market_panel/market_panel.parquet",
                "output_dir": "runtime/reports/t_plus_one/run",
            },
        )


def test_t_plus_one_research_backend_choice_is_constrained(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    holdings = quant_ui_settings.runtime_root / "paper" / "holdings.csv"
    holdings.parent.mkdir(parents=True, exist_ok=True)
    holdings.write_text("trade_date,symbol,quantity\n2026-01-05,000001.SZ,1000\n", encoding="utf-8")
    parameters = {
        "minute_dir": "runtime/data/v7/silver/market_panel",
        "holdings_csv": "runtime/paper/holdings.csv",
        "market_panel": "runtime/data/v7/silver/market_panel/market_panel.parquet",
        "output_dir": "runtime/reports/t_plus_one/run",
        "backend": "definitely_not_a_backend",
    }
    with pytest.raises(ValueError):
        container.jobs.validate("t-plus-one-research", "research-intraday-t-trading", parameters)
    parameters["backend"] = "lightgbm"
    assert container.jobs.validate(
        "t-plus-one-research", "research-intraday-t-trading", parameters
    )["valid"] is True
