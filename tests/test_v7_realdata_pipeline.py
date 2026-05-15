import json

import pandas as pd
from typer.testing import CliRunner

from quantagent.backtest.ashare_execution_simulator import simulate_ashare_target_weights
from quantagent.cli import app
from quantagent.data.v7_dataset_builder import V7DatasetBuildConfig, build_market_features, build_v7_training_dataset
from quantagent.data.v7_label_builder import build_forward_return_labels
from quantagent.data.v7_quality_gates import evaluate_model_acceptance_gates
from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment


def _market_panel(days: int = 36, symbols: tuple[str, ...] = ("600001.SH", "000001.SZ", "300750.SZ")) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="B")
    rows = []
    for sidx, symbol in enumerate(symbols):
        for didx, date in enumerate(dates):
            close = 10.0 + sidx + didx * (0.02 + sidx * 0.01)
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + sidx * 100_000,
                    "amount": close * 1_000_000,
                    "available_at": date.strftime("%Y-%m-%d"),
                    "is_suspended": False,
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_builder_shifts_close_features_and_builds_labels():
    market = _market_panel()
    result = build_v7_training_dataset(
        market,
        config=V7DatasetBuildConfig(horizons=(1, 5), min_rows=20, min_symbols=2, min_dates=10),
    )

    assert not result.dataset.empty
    assert "momentum_5d" in result.feature_schema["feature_columns"]
    assert "forward_return_5d" in result.label_schema["label_columns"]
    features = build_market_features(market)
    first = features.sort_values(["symbol", "trade_date"]).iloc[0]
    assert pd.Timestamp(first["available_at"]) > pd.Timestamp(first["trade_date"])


def test_label_builder_keeps_multi_horizon_schema():
    labels = build_forward_return_labels(_market_panel(), horizons=(1, 5))

    assert {"forward_return_1d", "forward_return_5d"}.issubset(labels.frame.columns)
    assert labels.label_schema["horizons"] == [1, 5]


def test_v7_training_experiment_writes_validation_artifacts(tmp_path):
    dataset = build_v7_training_dataset(
        _market_panel(days=50),
        config=V7DatasetBuildConfig(horizons=(1, 5), min_rows=40, min_symbols=2, min_dates=20),
    ).dataset
    result = run_v7_training_experiment(
        dataset,
        V7TrainingConfig(horizons=(1, 5), min_train_rows=40, output_dir=str(tmp_path / "alpha")),
    )

    assert result.status == "validation_only"
    assert "model_artifact" in result.artifact_paths
    assert (tmp_path / "alpha" / "metrics.json").exists()


def test_acceptance_gate_requires_paper_report_for_live_readiness(tmp_path):
    report = evaluate_model_acceptance_gates(
        {
            "rank_ic_mean": 0.01,
            "rank_ic_stability": 0.5,
            "turnover_adjusted_net_return": 0.01,
            "max_drawdown": -0.05,
            "single_factor_dominance": 0.2,
            "adverse_regime_passed": True,
            "uses_mock_or_synthetic": False,
        },
        paper_report_path=tmp_path / "missing.json",
    )

    assert not report.passed
    assert "paper_trading_report_missing" in report.failures


def test_ashare_execution_simulator_uses_order_manager_and_blocks_limit_up_buy():
    market = _market_panel(days=3, symbols=("600001.SH",))
    market.loc[market["trade_date"] == market["trade_date"].min(), "is_limit_up"] = True
    weights = pd.DataFrame({"600001.SH": [0.5, 0.5, 0.0]}, index=pd.to_datetime(sorted(market["trade_date"].unique())))
    result = simulate_ashare_target_weights(weights, market)

    assert not result.order_audit.empty
    assert "limit_up_no_buy" in set(result.failed_order_audit["last_message"])


def test_new_v7_cli_commands_smoke(tmp_path):
    market = _market_panel(days=12)
    market_path = tmp_path / "market.csv"
    labels_path = tmp_path / "labels.csv"
    market.to_csv(market_path, index=False)
    metrics_path = tmp_path / "metrics.json"
    paper_path = tmp_path / "paper.json"
    metrics_path.write_text(
        json.dumps(
            {
                "rank_ic_mean": 0.01,
                "rank_ic_stability": 0.5,
                "turnover_adjusted_net_return": 0.01,
                "max_drawdown": -0.05,
                "single_factor_dominance": 0.2,
                "adverse_regime_passed": True,
                "uses_mock_or_synthetic": False,
            }
        ),
        encoding="utf-8",
    )
    paper_path.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    download = runner.invoke(app, ["download-qlib-v7"])
    labels = runner.invoke(app, ["build-labels-v7", "--market-panel", str(market_path), "--output", str(labels_path), "--horizons", "1,5"])
    readiness = runner.invoke(app, ["v7-live-readiness-report", "--metrics", str(metrics_path), "--paper-report", str(paper_path), "--output", str(tmp_path / "ready.json")])

    assert download.exit_code == 0, download.output
    assert "scripts/get_data.py" in download.output
    assert labels.exit_code == 0, labels.output
    assert labels_path.exists()
    assert readiness.exit_code == 0, readiness.output
    assert '"passed": true' in readiness.output
