from quantagent.replay.scenario_registry import ScenarioRegistry
from quantagent.services.v6_pipeline_service import run_historical_live_replay_v6


def test_replay_scenario_registry_loads_v6_yaml():
    registry = ScenarioRegistry.from_yaml("configs/replay_scenarios.v6.yaml")
    assert "2020_covid_volatility_replay" in registry.names()
    scenario = registry.get("mock_recent_replay")
    assert scenario.data_mode == "mock"


def test_mock_historical_live_replay_runs(tmp_path):
    cfg = {
        "data": {"provider": "mock"},
        "market": {"benchmark": "000300.SH", "universe": "CSI300"},
        "execution": {"audit_log_dir": str(tmp_path / "logs"), "initial_cash": 1000000, "dry_run": True},
        "reporting": {"output_dir": str(tmp_path / "reports")},
    }
    result = run_historical_live_replay_v6(cfg, "mock_recent_replay")
    assert result["days"] > 0

