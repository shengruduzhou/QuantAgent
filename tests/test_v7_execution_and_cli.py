import pandas as pd
from typer.testing import CliRunner

from quantagent.backtest.tplus1_engine import TPlusOneExecutionSimulator
from quantagent.cli import app


def test_tplus1_engine_blocks_same_day_sell_and_allows_next_day_sell():
    intents = pd.DataFrame(
        [
            {"trade_date": "2026-05-14", "symbol": "600001.SH", "side": "buy", "quantity": 100, "price": 10.0, "volume": 1000000},
            {"trade_date": "2026-05-14", "symbol": "600001.SH", "side": "sell", "quantity": 100, "price": 10.1, "volume": 1000000},
            {"trade_date": "2026-05-15", "symbol": "600001.SH", "side": "sell", "quantity": 100, "price": 10.2, "volume": 1000000},
        ]
    )
    result = TPlusOneExecutionSimulator().run(intents)

    assert "t_plus_one_insufficient_available_shares" in set(result.rejects["reason"])
    assert len(result.fills) == 2
    assert result.positions["600001.SH"] == 0


def test_tplus1_engine_blocks_limit_down_sell():
    intents = pd.DataFrame(
        [
            {"trade_date": "2026-05-14", "symbol": "600001.SH", "side": "buy", "quantity": 100, "price": 10.0, "volume": 1000000},
            {"trade_date": "2026-05-15", "symbol": "600001.SH", "side": "sell", "quantity": 100, "price": 9.0, "volume": 1000000, "is_limit_down": True},
        ]
    )
    result = TPlusOneExecutionSimulator().run(intents)

    assert result.rejects.iloc[-1]["reason"] == "limit_down_no_sell"
    assert result.positions["600001.SH"] == 100


def test_v7_cli_validate_and_daily_smoke(tmp_path):
    runner = CliRunner()
    validate = runner.invoke(app, ["validate-v7", "--config", "configs/v7.default.yaml"])
    daily = runner.invoke(app, ["run-daily-v7", "--config", "configs/v7.default.yaml", "--date", "2026-05-14", "--output-dir", str(tmp_path)])

    assert validate.exit_code == 0, validate.output
    assert "passed" in validate.output
    assert daily.exit_code == 0, daily.output
    assert "status=ok" in daily.output
    assert (tmp_path / "v7_daily_research_report.json").exists()
