from __future__ import annotations

import json
import inspect

import pandas as pd

from services.quant_api.app import create_app
from services.quant_api.services.connections import ConnectionManager
from services.quant_api.services.jobs import COMMANDS, JobManager, JobRecord
from quantagent.data.v7_quality_gates import V7ModelAcceptanceGateConfig, evaluate_model_acceptance_gates
from tests.quant_ui.test_api import request
from quantagent.cli.v7_train import run_full_real_training_v7


def _write_labels(path, horizons: tuple[int, ...] = (1, 5, 20)) -> None:
    frame = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        **{
            f"forward_return_{horizon}d": [0.01, -0.01]
            for horizon in horizons
        },
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _strategy_payload() -> dict:
    return {
        "name": "Reviewed A-share multi-factor",
        "hypothesis": "Reviewed factors retain cross-sectional excess return after A-share costs.",
        "invalidationCriteria": "Reject when OOS gates fail or drawdown exceeds the declared limit.",
        "marketPanelPath": "runtime/data/v7/silver/market_panel/market_panel.parquet",
        "labelsPath": "runtime/data/v7/gold/labels/labels.parquet",
        "outputDir": "runtime/reports/strategy_studio/fixture",
        "factorLibrary": "all_reviewed",
        "model": "ft_transformer",
        "researchPreset": "stable_alpha",
        "horizons": "1,5,20",
        "primaryHorizon": 5,
        "horizonBlendMethod": "adaptive_oos",
        "splitMode": "rolling",
        "nSplits": 4,
        "requireGpu": True,
        "topK": 30,
        "topKCandidates": [10, 20, 30, 50],
        "stockSelectionModes": ["none", "fundamental"],
        "fundamentalSelectionMode": "auto",
        "fundamentalSelectionThreshold": 0.5,
        "fundamentalBlendWeight": 0.4,
        "fundamentalThresholdCandidates": [0.25, 0.4, 0.55],
        "fundamentalBlendCandidates": [0.2, 0.4, 0.6],
        "selectionMaxCandidates": 64,
        "selectionMinOosDays": 80,
        "selectionMinHoldoutDays": 20,
        "maxPbo": 0.25,
        "minDsrProbability": 0.95,
        "maxSpaPValue": 0.05,
        "factorScreeningMode": "pretrain",
        "doTMode": "daily_swing",
        "maxWeightPerName": 0.08,
        "maxSectorWeight": 0.30,
        "maxTurnover": 0.50,
        "objective": "max_expected_alpha",
        "weighting": "rank",
        "initialCash": 1_000_000,
        "objectiveWeights": {
            "excessReturn": 0.45,
            "annualReturn": 0.30,
            "drawdownControl": 0.25,
        },
        "riskLimits": {"maxDrawdown": 0.15, "maxTurnover": 0.50, "minSharpe": 1.0},
        "humanApproved": True,
    }


def test_strategy_contract_validates_saves_and_builds_allowlisted_launch(quant_ui_settings) -> None:
    labels = quant_ui_settings.runtime_root / "data" / "v7" / "gold" / "labels"
    _write_labels(labels / "labels.parquet")
    app = create_app(quant_ui_settings)

    defaults = request(app, "GET", "/api/strategies/defaults")
    assert defaults.status_code == 200
    assert defaults.json()["data"]["selected"]["marketPanelPath"].endswith("market_panel.parquet")

    validation = request(app, "POST", "/api/strategies/validate", json=_strategy_payload())

    assert validation.status_code == 200
    data = validation.json()["data"]
    assert data["valid"] is True
    assert data["launch"]["commandId"] == "run-full-real-training-v7"
    assert data["launch"]["armed"] is True
    assert data["launch"]["parameters"]["market_panel_path"].endswith("market_panel.parquet")
    assert data["launch"]["parameters"]["horizon_blend_method"] == "adaptive_oos"
    assert data["launch"]["parameters"]["acceptance_max_drawdown"] == 0.15
    assert data["launch"]["parameters"]["acceptance_min_sharpe"] == 1.0
    assert data["launch"]["parameters"]["factor_library"] == "all_reviewed"
    assert data["launch"]["parameters"]["top_k_candidates"] == [10, 20, 30, 50]
    assert data["launch"]["parameters"]["stock_selection_modes"] == ["none", "fundamental"]
    assert data["launch"]["parameters"]["fundamental_selection_mode"] == "auto"
    assert data["launch"]["parameters"]["fundamental_blend_candidates"] == [0.2, 0.4, 0.6]
    assert data["launch"]["parameters"]["selection_min_oos_days"] == 80
    assert data["launch"]["parameters"]["max_pbo"] == 0.25
    assert any(member["id"] == "risk" and member["veto"] for member in data["decisionCouncil"])

    saved = request(app, "POST", "/api/strategies", json=_strategy_payload())
    assert saved.status_code == 200
    manifest = saved.json()["data"]
    assert manifest["trustClass"] == "research_only"
    assert manifest["path"].startswith("runtime/strategies/")
    persisted = quant_ui_settings.runtime_root / manifest["path"].removeprefix("runtime/")
    payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == "quantagent.strategy.v1"
    assert payload["contentHash"]

    spec = COMMANDS["run-full-real-training-v7"]
    assert spec["type"] == "strategy-pipeline"
    assert spec["option_aliases"]["market_panel_path"] == "market-panel"
    assert "mark_production_ready" not in spec["allowed"]


def test_strategy_launch_is_atomic_and_direct_job_route_is_not_exposed(quant_ui_settings, monkeypatch) -> None:
    labels = quant_ui_settings.runtime_root / "data" / "v7" / "gold" / "labels"
    _write_labels(labels / "labels.parquet")
    app = create_app(quant_ui_settings)
    observed: dict = {}

    def fake_validate(job_type, command_id, parameters):
        observed["validated"] = (job_type, command_id, parameters)
        return {"valid": True}

    def fake_submit(job_type, command_id, parameters):
        observed["submitted"] = (job_type, command_id, parameters)
        return {
            "id": "job_strategy",
            "type": job_type,
            "status": "queued",
            "commandId": command_id,
            "createdAt": "2026-01-01T00:00:00+00:00",
            "progress": 0.0,
            "outputPaths": [],
        }

    monkeypatch.setattr(app.state.services.jobs, "validate", fake_validate)
    monkeypatch.setattr(app.state.services.jobs, "submit", fake_submit)

    launched = request(app, "POST", "/api/strategies/launch", json=_strategy_payload())

    assert launched.status_code == 200
    assert launched.json()["data"]["job"]["id"] == "job_strategy"
    assert launched.json()["data"]["strategy"]["trustClass"] == "research_only"
    assert observed["validated"][0] == "strategy-pipeline"
    assert observed["submitted"][2]["acceptance_min_sharpe"] == 1.0

    direct = request(
        app,
        "POST",
        "/api/jobs/strategy-pipeline",
        json={"commandId": "run-full-real-training-v7", "parameters": {}},
    )
    assert direct.status_code in {404, 405}


def test_strategy_horizon_contract_fails_before_launch_and_checks_label_schema(
    quant_ui_settings,
) -> None:
    labels = quant_ui_settings.runtime_root / "data" / "v7" / "gold" / "labels" / "labels.parquet"
    _write_labels(labels, horizons=(1, 5, 20))
    app = create_app(quant_ui_settings)

    invalid_primary = request(
        app,
        "POST",
        "/api/strategies/validate",
        json={**_strategy_payload(), "primaryHorizon": 10},
    )
    assert invalid_primary.status_code == 422
    assert "primaryHorizon must be included in horizons" in invalid_primary.text

    missing_label = request(
        app,
        "POST",
        "/api/strategies/validate",
        json={
            **_strategy_payload(),
            "horizons": "1,5,10,20",
            "primaryHorizon": 10,
        },
    )
    assert missing_label.status_code == 200
    result = missing_label.json()["data"]
    assert result["valid"] is False
    assert any("forward_return_10d" in error for error in result["errors"])
    issue = next(
        item
        for item in result["issues"]
        if item["code"] == "missing_horizon_columns"
    )
    assert issue["severity"] == "blocking"
    assert issue["evidence"]["availableHorizons"] == [1, 5, 20]
    assert issue["evidence"]["missingHorizons"] == [10]
    council = {member["id"]: member for member in result["decisionCouncil"]}
    assert council["data_quality"]["status"] == "blocked"
    assert council["model_validation"]["status"] == "blocked"
    assert council["risk"]["status"] == "ready"


def test_strategy_auto_search_is_bounded_and_gpu_training_is_fail_closed(
    quant_ui_settings,
) -> None:
    app = create_app(quant_ui_settings)
    too_many = request(
        app,
        "POST",
        "/api/strategies/validate",
        json={
            **_strategy_payload(),
            "topKCandidates": [10, 20, 30, 40, 50, 60, 70, 80],
            "fundamentalThresholdCandidates": [0.1, 0.2, 0.3, 0.4],
            "fundamentalBlendCandidates": [0.2, 0.4, 0.6, 0.8],
            "selectionMaxCandidates": 64,
        },
    )
    assert too_many.status_code == 422
    assert "selectionMaxCandidates" in too_many.text

    cpu_deep = request(
        app,
        "POST",
        "/api/strategies/validate",
        json={**_strategy_payload(), "requireGpu": False},
    )
    assert cpu_deep.status_code == 422
    assert "requires GPU" in cpu_deep.text


def test_strategy_cli_signature_contains_every_allowlisted_search_parameter() -> None:
    parameters = inspect.signature(run_full_real_training_v7).parameters
    expected = {
        "top_k_candidates",
        "stock_selection_modes",
        "fundamental_selection_mode",
        "fundamental_selection_threshold",
        "fundamental_blend_weight",
        "fundamental_threshold_candidates",
        "fundamental_blend_candidates",
        "selection_max_candidates",
        "selection_min_oos_days",
        "selection_min_holdout_days",
        "max_pbo",
        "min_dsr_probability",
        "max_spa_pvalue",
        "objective_excess_weight",
        "objective_annual_weight",
        "objective_drawdown_weight",
        "horizon_blend_method",
    }
    assert expected <= set(parameters)


def test_strategy_defaults_expose_selectable_inputs_and_prefer_label_coverage(
    quant_ui_settings,
) -> None:
    narrow = (
        quant_ui_settings.runtime_root
        / "data"
        / "gold"
        / "full_universe"
        / "labels.parquet"
    )
    wide = quant_ui_settings.runtime_root / "data" / "v7" / "labels.parquet"
    _write_labels(narrow, horizons=(1, 5))
    _write_labels(wide, horizons=(1, 5, 20, 60, 120))
    app = create_app(quant_ui_settings)

    response = request(app, "GET", "/api/strategies/defaults")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["selected"]["labelsPath"] == "runtime/data/v7/labels.parquet"
    label_options = data["options"]["labelsPath"]
    assert {item["path"] for item in label_options} == {
        "runtime/data/gold/full_universe/labels.parquet",
        "runtime/data/v7/labels.parquet",
    }
    assert next(
        item
        for item in label_options
        if item["path"] == "runtime/data/v7/labels.parquet"
    )["availableHorizons"] == [1, 5, 20, 60, 120]
    assert len(data["options"]["fundamentalsRoot"]) == 2


def test_label_rebuild_and_horizon_blend_are_allowlisted_without_free_shell() -> None:
    labels = COMMANDS["build-labels-v7"]
    assert labels["type"] == "data"
    assert labels["required"] == {"market_panel_path", "output_path", "horizons"}
    assert labels["option_aliases"] == {
        "market_panel_path": "market-panel",
        "output_path": "output",
    }
    strategy = COMMANDS["run-full-real-training-v7"]
    assert "horizon_blend_method" in strategy["allowed"]
    assert strategy["choices"]["horizon_blend_method"] == {
        "adaptive_oos",
        "balanced",
        "short_tactical",
        "long_fundamental",
        "primary_only",
    }


def test_excess_return_objective_requires_a_benchmark(quant_ui_settings) -> None:
    labels = (
        quant_ui_settings.runtime_root
        / "data"
        / "v7"
        / "gold"
        / "labels"
        / "labels.parquet"
    )
    _write_labels(labels)
    app = create_app(quant_ui_settings)

    response = request(
        app,
        "POST",
        "/api/strategies/validate",
        json={**_strategy_payload(), "benchmarkSymbol": ""},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["valid"] is False
    issue = next(
        item for item in data["issues"] if item["code"] == "benchmark_missing"
    )
    assert issue["severity"] == "blocking"
    assert issue["action"]["kind"] == "set_benchmark"


def test_strategy_launch_fails_closed_without_human_gate_or_real_inputs(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    payload = {**_strategy_payload(), "humanApproved": False, "labelsPath": "runtime/missing.parquet"}

    validation = request(app, "POST", "/api/strategies/validate", json=payload)

    assert validation.status_code == 200
    data = validation.json()["data"]
    assert data["valid"] is False
    assert data["launch"]["armed"] is False
    assert any("does not exist" in error for error in data["errors"])
    blocked = request(app, "POST", "/api/strategies/launch", json=payload)
    assert blocked.status_code == 422


def test_connection_vault_never_returns_or_persists_secret(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    secret = "tickflow-super-secret-value"

    connected = request(
        app,
        "POST",
        "/api/connections/tickflow",
        json={"credentials": {"TICKFLOW_API_KEY": secret}},
    )

    assert connected.status_code == 200
    body = connected.text
    assert secret not in body
    assert connected.json()["data"]["source"] == "session"
    assert connected.json()["data"]["persistence"] == "process_memory"
    assert app.state.services.connections.environment_for({"tickflow"})["TICKFLOW_API_KEY"] == secret
    state_path = quant_ui_settings.jobs_root / "jobs.json"
    assert not state_path.exists() or secret not in state_path.read_text(encoding="utf-8")

    disconnected = request(app, "DELETE", "/api/connections/tickflow")
    assert disconnected.status_code == 200
    assert disconnected.json()["data"]["connected"] is False


def test_connection_manager_rejects_unknown_fields() -> None:
    manager = ConnectionManager()
    try:
        manager.connect("tickflow", {"TICKFLOW_API_KEY": "valid-secret", "SHELL": "bash"})
    except ValueError as exc:
        assert "unsupported credential fields" in str(exc)
    else:
        raise AssertionError("unknown credential field was accepted")


def test_strategy_pipeline_uses_cli_aliases_and_omits_false_single_flag(
    quant_ui_settings,
    monkeypatch,
) -> None:
    observed: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak-to-strategy-job")

    class Process:
        stdout = []

        def __init__(self, command, **kwargs) -> None:
            observed["command"] = command
            observed["env"] = kwargs["env"]

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

    monkeypatch.setattr("services.quant_api.services.jobs.subprocess.Popen", Process)
    manager = JobManager(quant_ui_settings)
    manager._jobs["job_alias"] = JobRecord(
        id="job_alias",
        type="strategy-pipeline",
        status="queued",
        commandId="run-full-real-training-v7",
        createdAt="2026-01-01T00:00:00+00:00",
    )
    market = quant_ui_settings.runtime_root / "data/v7/silver/market_panel/market_panel.parquet"
    labels = quant_ui_settings.runtime_root / "labels.parquet"
    labels.write_bytes(b"fixture")
    output = quant_ui_settings.runtime_root / "reports/strategy"
    parameters = {
        "market_panel_path": str(market),
        "labels_path": str(labels),
        "output_dir": str(output),
        "require_gpu": False,
        "max_weight_per_name": 0.08,
    }

    manager._run(
        "job_alias",
        "run-full-real-training-v7",
        parameters,
        COMMANDS["run-full-real-training-v7"],
        quant_ui_settings.jobs_root / "job_alias.log",
    )

    assert "--market-panel" in observed["command"]
    assert "--labels" in observed["command"]
    assert "--max-weight" in observed["command"]
    assert "--no-require-gpu" not in observed["command"]
    assert "OPENAI_API_KEY" not in observed["env"]


def test_declared_sharpe_and_drawdown_acceptance_limits_are_enforced() -> None:
    report = evaluate_model_acceptance_gates(
        {
            "rank_ic_mean": 0.01,
            "rank_ic_stability": 0.1,
            "turnover_adjusted_net_return": 0.05,
            "max_drawdown": -0.16,
            "sharpe": 0.8,
            "adverse_regime_passed": True,
            "excess_return_after_costs": 0.02,
            "selection_pressure_min": 4.0,
            "training_dataset_symbol_count": 100,
            "prediction_symbol_count": 100,
            "effective_universe_min": 100,
        },
        V7ModelAcceptanceGateConfig(
            max_drawdown=0.15,
            min_sharpe=1.0,
            require_paper_report=False,
            require_benchmark=False,
        ),
    )

    assert report.passed is False
    assert "max_drawdown_exceeded" in report.failures
    assert "sharpe_below_minimum" in report.failures
