"""API contract for the factor-fusion workstation surface."""

from __future__ import annotations

import asyncio

import httpx

from services.quant_api.app import create_app
from services.quant_api.services.container import ServiceContainer
from services.quant_api.services.jobs import COMMANDS


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def _first_run_id(app) -> str:
    payload = request(app, "GET", "/api/fusion/runs").json()
    assert payload["status"] == "ready"
    return payload["data"][0]["id"]


def test_fusion_runs_list_reports_search_provenance(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    payload = request(app, "GET", "/api/fusion/runs").json()
    assert payload["status"] == "ready"
    run = payload["data"][0]
    assert run["nTrials"] == 4
    assert run["contentHash"] == "abcdef0123456789"
    assert run["benchmarkMode"] == "index:000300.SH"
    assert run["frontierSize"] == 2
    assert run["factorNames"] == ["alpha001", "alpha002"]


def test_fusion_run_detail_marks_frontier_and_preference_rank(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _first_run_id(app)
    payload = request(app, "GET", f"/api/fusion/runs/{run_id}").json()
    assert payload["status"] == "ready"
    by_id = {item["id"]: item for item in payload["data"]["candidates"]}
    assert by_id["ic_weighted"]["onFrontier"] is True
    assert by_id["ic_weighted"]["preferenceRank"] == 0
    assert by_id["random_00"]["onFrontier"] is False
    assert by_id["random_00"]["preferenceRank"] is None
    assert by_id["random_00"]["isControl"] is True


def test_fusion_nav_returns_one_column_per_candidate(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _first_run_id(app)
    payload = request(app, "GET", f"/api/fusion/runs/{run_id}/nav").json()
    assert payload["status"] == "ready"
    assert set(payload["data"][0]) == {"trade_date", "ic_weighted", "equal", "random_00"}


def test_fusion_compare_rejects_more_than_four_candidates(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _first_run_id(app)
    response = request(
        app,
        "GET",
        f"/api/fusion/runs/{run_id}/compare?candidates=a,b,c,d,e",
    )
    assert response.status_code == 422


def test_fusion_compare_returns_only_requested_candidates(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _first_run_id(app)
    payload = request(
        app,
        "GET",
        f"/api/fusion/runs/{run_id}/compare?candidates=ic_weighted,equal",
    ).json()
    assert [item["id"] for item in payload["data"]["candidates"]] == ["ic_weighted", "equal"]


def test_fusion_compare_404s_on_unknown_candidate(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _first_run_id(app)
    response = request(
        app, "GET", f"/api/fusion/runs/{run_id}/compare?candidates=does_not_exist"
    )
    assert response.status_code == 404


def test_fusion_run_404s_on_unknown_id(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    assert request(app, "GET", "/api/fusion/runs/nope").status_code == 404


def test_empty_runtime_reports_actionable_empty_state(empty_quant_ui_settings) -> None:
    app = create_app(empty_quant_ui_settings)
    payload = request(app, "GET", "/api/fusion/runs").json()
    assert payload["status"] == "empty"
    assert payload["issues"][0]["code"] == "no_fusion_runs"


# --------------------------------------------------------------------------- #
# Governed launch contract                                                    #
# --------------------------------------------------------------------------- #


def test_fusion_search_is_a_governed_command() -> None:
    spec = COMMANDS["search-factor-fusion"]
    assert spec["type"] == "fusion-search"
    assert "entrypoint" not in spec, "must run through the quantagent CLI"
    assert spec["path_outputs"] == {"output_dir"}


def test_operator_cannot_declare_the_trial_count() -> None:
    """n_trials must stay derived; a caller-supplied value would break the DSR."""
    allowed = COMMANDS["search-factor-fusion"]["allowed"]
    assert not any("trial" in name for name in allowed)


def test_fusion_search_rejects_unknown_parameters(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    panel = quant_ui_settings.runtime_root / "data" / "v7" / "silver" / "market_panel" / "market_panel.parquet"
    relative = "runtime/data/v7/silver/market_panel/market_panel.parquet"
    assert panel.exists()
    try:
        container.jobs.validate(
            "fusion-search",
            "search-factor-fusion",
            {
                "factor_panel_path": relative,
                "forward_returns_path": relative,
                "factor_names": "alpha001",
                "output_dir": "runtime/reports/fusion/new_run",
                "n_trials": 3,
            },
        )
    except ValueError as exc:
        assert "n_trials" in str(exc)
    else:  # pragma: no cover - guard against a silently permissive allowlist
        raise AssertionError("expected the allowlist to reject n_trials")


def test_fusion_search_validates_a_well_formed_request(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    relative = "runtime/data/v7/silver/market_panel/market_panel.parquet"
    result = container.jobs.validate(
        "fusion-search",
        "search-factor-fusion",
        {
            "factor_panel_path": relative,
            "forward_returns_path": relative,
            "factor_names": "alpha001,alpha002",
            "output_dir": "runtime/reports/fusion/new_run",
            "horizon_days": 5,
            "top_k": 30,
            "include_genetic": False,
        },
    )
    assert result["valid"] is True


def test_a_real_search_round_trips_through_the_adapter(quant_ui_settings) -> None:
    """The writer and the reader must agree on the artifact schema.

    Every other fusion API test reads a hand-written fixture, which would keep
    passing if `save_fusion_artifacts` changed its field names. This one runs a
    genuine (small) search, saves it, and reads it back through the adapter.
    """
    import numpy as np
    import pandas as pd

    from quantagent.fusion import FusionSearchConfig, run_fusion_search, save_fusion_artifacts

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=300)
    symbols = [f"{600000 + index}.SH" for index in range(25)]
    frames = []
    for date in dates:
        signal = rng.normal(size=len(symbols))
        frames.append(pd.DataFrame({
            "trade_date": date,
            "symbol": symbols,
            "alpha_a": signal,
            "alpha_b": rng.normal(size=len(symbols)),
            "forward_return": 0.02 * signal + 0.01 * rng.normal(size=len(symbols)),
        }))
    panel = pd.concat(frames, ignore_index=True)

    result = run_fusion_search(
        factor_panel=panel[["trade_date", "symbol", "alpha_a", "alpha_b"]],
        forward_panel=panel[["trade_date", "symbol", "forward_return"]],
        config=FusionSearchConfig(
            factor_names=("alpha_a", "alpha_b"),
            horizon_days=5, top_k=5, n_folds=2, embargo_days=5,
            min_train_days=100, min_test_days=40,
            include_genetic=False, random_controls=2, single_factor_baselines=2,
        ),
    )
    output = quant_ui_settings.runtime_root / "reports" / "fusion" / "round_trip"
    save_fusion_artifacts(result, output_dir=output)

    app = create_app(quant_ui_settings)
    runs = request(app, "GET", "/api/fusion/runs").json()["data"]
    run = next(item for item in runs if item["name"] == "round_trip")
    assert run["nTrials"] == result.n_trials
    assert run["factorNames"] == ["alpha_a", "alpha_b"]

    detail = request(app, "GET", f"/api/fusion/runs/{run['id']}").json()["data"]
    assert len(detail["candidates"]) == result.n_trials
    for candidate in detail["candidates"]:
        # Field names the workstation renders must survive the round trip.
        assert set(candidate["metrics"]) >= {
            "excessReturn", "annualReturn", "maxDrawdown", "robustness", "observations",
        }
        assert set(candidate["robustnessBreakdown"]) >= {
            "foldConsistency", "overfittingResistance", "deflatedSharpeProbability",
        }
    frontier_ids = {item["id"] for item in detail["candidates"] if item["onFrontier"]}
    assert frontier_ids == set(result.frontier)
