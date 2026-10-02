"""API honesty for unknown ids and unmeasured risk (round-29 R5, live GET sweep)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from services.quant_api.adapters.risk import RiskAdapter
from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings


def _settings(tmp_path: Path) -> ApiSettings:
    # Reuse the repository's own runtime fixture so the app has real runs.
    import importlib.util

    wt = Path(__import__("services").__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quant_ui_conftest_fixture", wt / "tests" / "quant_ui" / "conftest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    project_root = tmp_path / "QuantAgent"
    runtime = project_root / "runtime"
    settings = ApiSettings(
        project_root=project_root,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
        index_ttl_seconds=300,
    ).ensure()
    mod._write_runtime_fixture(settings)
    return settings


def _get(app, url: str) -> httpx.Response:
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(url)

    return asyncio.run(run())


def test_risk_overview_unknown_backtest_is_not_silently_another_run(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    result = _get(app, "/api/risk/overview?backtestId=backtest_DOES_NOT_EXIST")
    body = result.json()
    # Requested identity must be honoured: either 404 or an explicit non-ready
    # payload for the requested id -- never another run's numbers as "ready".
    assert result.status_code == 404 or (
        body["status"] != "ready" and body["data"]["backtestId"] in (None, "backtest_DOES_NOT_EXIST")
    ), (result.status_code, body["status"], body["data"].get("backtestId"))


def test_decision_chain_unknown_run_is_404_not_500(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    result = _get(app, "/api/selection/runs/NO_SUCH_RUN/stocks/000001.SZ/decision-chain")
    assert result.status_code == 404, result.status_code


class _NoEquityBacktests:
    """Backtest adapter stub: one run with NO equity curve and no events."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list(self):
        return [{"id": "bt_no_equity", "maxDrawdown": None, "volatility": None}]

    def equity(self, backtest_id):
        return []

    def risk_events(self, backtest_id, page=1, page_size=100):
        return {"items": []}

    def _resolve(self, backtest_id):
        return self.root


def test_consecutive_loss_days_is_unknown_without_equity(tmp_path: Path) -> None:
    overview = RiskAdapter(_NoEquityBacktests(tmp_path)).overview("bt_no_equity")
    # No daily returns were measured, so the streak is unknown, not "0 days".
    assert overview["consecutiveLossDays"] is None, overview["consecutiveLossDays"]


def test_factor_ic_with_no_measurement_is_not_ready(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    listing = _get(app, "/api/factors").json()["data"]
    items = listing if isinstance(listing, list) else listing.get("items", [])
    if not items:
        return
    name = items[0].get("name") or items[0].get("factorName")
    body = _get(app, f"/api/factors/{name}/ic").json()
    data = body["data"]
    statistics = [data.get(k) for k in ("ic", "rankIc", "icir", "rankIcir")]
    if all(v is None for v in statistics) and not data.get("icSeries"):
        assert body["status"] != "ready"
