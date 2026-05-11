from typer.testing import CliRunner

from quantagent.cli import app


def test_v6_cli_validate_smoke(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["validate-v6", "--config", "configs/v6.default.yaml", "--output-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "passed" in result.output


def test_v6_cli_paper_trade_smoke(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["paper-trade-v6", "--config", "configs/v6.default.yaml", "--output-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "status=ok" in result.output

