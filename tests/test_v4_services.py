import importlib.util

import pandas as pd
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.services.build_features_service import build_features_v4
from quantagent.services.daily_signal_service import infer_v4_alpha
from quantagent.services.paper_trading_service import generate_dry_run_order_intents
from quantagent.services.portfolio_build_service import build_portfolio_v4


def test_v4_synthetic_pipeline_reaches_dry_run_order_intents():
    features = build_features_v4().frame
    signals = infer_v4_alpha(features)
    portfolio = build_portfolio_v4(signals)
    latest = features.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
    intents = generate_dry_run_order_intents(portfolio.target_weights, latest["close"])
    assert not signals.empty
    assert "target_weights" not in signals.columns
    assert all(intent.risk_check_result == "dry_run_pass" for intent in intents)


def test_v4_cli_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build-features-v4" in result.output
    assert "paper-trade-v4" in result.output


def test_v4_train_service_can_run_when_torch_available():
    if importlib.util.find_spec("torch") is None:
        return
    from quantagent.services.train_v4_service import train_v4_synthetic

    metadata = train_v4_synthetic()
    assert pd.notna(metadata.train_loss)
