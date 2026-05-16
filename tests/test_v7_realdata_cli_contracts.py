from __future__ import annotations

import json
import sys
import types

import pandas as pd
from typer.testing import CliRunner

from quantagent.cli import app


def test_setup_qlib_v7_prints_windows_chain_and_manual_fallback(tmp_path):
    runner = CliRunner()
    target = tmp_path / "cn_data"
    result = runner.invoke(app, ["setup-qlib-v7", "--target-dir", str(target), "--region", "cn", "--interval", "1d"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "manual_step_required"
    assert "windows_powershell_command_chain" in payload
    assert any("quantagent check-qlib-v7" in item for item in payload["windows_powershell_command_chain"])
    assert payload["official_command"].startswith("python scripts/get_data.py qlib_data")


def test_smoke_akshare_v7_requires_explicit_network():
    runner = CliRunner()
    result = runner.invoke(app, ["smoke-akshare-v7", "--symbols", "600519.SH,000001.SZ"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["allow_network"] is False
    assert "market" in payload["probes"]
    assert "financial_indicator" in payload["probes"]


def test_akshare_financial_provider_emits_indicator_dividend_metadata(monkeypatch):
    from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
    from quantagent.data.providers.base import ProviderRequest

    def financial_report_sina(stock: str, symbol: str | None = None) -> pd.DataFrame:
        if symbol == "利润表":
            return pd.DataFrame([{"报告日期": "2024-12-31", "公告日期": "2025-03-30", "营业总收入": 100.0}])
        if symbol == "资产负债表":
            return pd.DataFrame([{"报告日期": "2024-12-31", "公告日期": "2025-03-30", "资产总计": 500.0}])
        if symbol == "现金流量表":
            return pd.DataFrame([{"报告日期": "2024-12-31", "公告日期": "2025-03-30", "经营活动产生的现金流量净额": 20.0}])
        return pd.DataFrame()

    fake_akshare = types.SimpleNamespace(
        stock_financial_report_sina=financial_report_sina,
        stock_financial_analysis_indicator=lambda symbol: pd.DataFrame(
            [{"报告日期": "2024-12-31", "公告日期": "2025-03-30", "净资产收益率": 12.0}]
        ),
        stock_history_dividend_detail=lambda symbol, indicator: pd.DataFrame(
            [{"报告日期": "2024-12-31", "公告日期": "2025-04-15", "派息": 1.5}]
        ),
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    request = ProviderRequest("2024-01-01", "2025-12-31", symbols=("600519.SH",))
    statements = AkShareFinancialProvider(allow_network=True, rate_limit_seconds=0).all_statements(request)

    assert {"income", "balance_sheet", "cashflow", "financial_indicator", "dividend"}.issubset(statements)
    for result in statements.values():
        assert result.metadata["function_name"]
        assert result.metadata["schema_hash"]
        assert result.metadata["failed_symbols"] == []
        assert {"report_period", "ann_date", "available_at"}.issubset(result.frame.columns)


def test_factor_manifest_includes_richer_factor_metadata():
    from quantagent.factors.expr import build_factor_manifest

    manifest = build_factor_manifest(backend="polars")
    names = {entry["factor_name"] for entry in manifest}
    assert {"momentum_20", "valuation_pe_inverse", "quality_roe", "cashflow_operating_cash_flow"}.issubset(names)
    first = manifest[0]
    assert {"factor_name", "expression", "lookback", "required_columns", "backend", "created_at", "no_lookahead_check"}.issubset(first)


def test_llm_skill_config_from_env_keeps_key_indirect(monkeypatch):
    from quantagent.agents.llm_skill_client import LLMSkillConfig, LLMSkillClient

    monkeypatch.setenv("QUANTAGENT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("QUANTAGENT_LLM_ENABLED", "true")
    monkeypatch.setenv("QUANTAGENT_LLM_ALLOW_NETWORK", "false")
    monkeypatch.setenv("QUANTAGENT_LLM_ENDPOINT", "https://example.com/v1/chat/completions")
    monkeypatch.setenv("QUANTAGENT_LLM_MODEL", "local-model")
    monkeypatch.setenv("QUANTAGENT_LLM_API_KEY_ENV", "OPENAI_API_KEY")

    config = LLMSkillConfig.from_env()
    assert config.provider == "openai-compatible"
    assert config.api_key_env == "OPENAI_API_KEY"
    result = LLMSkillClient(config).invoke("risk_explain", system_prompt="", user_text="", fallback={"ok": True})
    assert result.used_fallback is True
    assert result.fallback_reason == "network_blocked"
