import pandas as pd
from typer.testing import CliRunner

from quantagent.cli import app






def test_v7_cli_validate_and_daily_smoke(tmp_path):
    runner = CliRunner()
    validate = runner.invoke(app, ["validate-v7", "--config", "configs/v7.default.yaml"])
    daily = runner.invoke(app, ["run-daily-v7", "--config", "configs/v7.mock.yaml", "--date", "2026-05-14", "--output-dir", str(tmp_path)])

    assert validate.exit_code == 0, validate.output
    assert "passed" in validate.output
    assert daily.exit_code == 0, daily.output
    assert "status=ok" in daily.output
    assert (tmp_path / "v7_daily_research_report.json").exists()


def test_v7_cli_qlib_check_reports_missing_provider_uri(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["check-qlib-v7", "--provider-uri", str(tmp_path / "missing_qlib")])

    assert result.exit_code == 0, result.output
    assert "provider_uri_not_found" in result.output
