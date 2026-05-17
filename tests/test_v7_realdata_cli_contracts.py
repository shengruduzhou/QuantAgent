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


def test_llm_skill_client_builds_gemini_payload_without_printing_key(monkeypatch):
    from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig

    monkeypatch.setenv("QUANTAGENT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("QUANTAGENT_LLM_ENABLED", "true")
    monkeypatch.setenv("QUANTAGENT_LLM_ALLOW_NETWORK", "false")
    monkeypatch.setenv("QUANTAGENT_LLM_MODEL", "gemma-test-model")
    config = LLMSkillConfig.from_env()
    client = LLMSkillClient(config)

    assert config.api_key_env == "GOOGLE_API_KEY"
    assert client._resolved_endpoint().endswith("/models/gemma-test-model:generateContent")
    payload = client._request_payload("system", "user")
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    result = client.invoke("schema", system_prompt="system", user_text="user", fallback={"ok": True})
    assert result.fallback_reason == "network_blocked"


def test_build_akshare_market_panel_writes_manifest(monkeypatch, tmp_path):
    from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
    from quantagent.data.providers.base import ProviderResult

    def fake_daily(self, request):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "600519.SH",
                    "trade_date": "2024-01-02",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000,
                    "amount": 10500.0,
                    "available_at": "2024-01-03",
                }
            ]
        )
        return ProviderResult(frame, source="akshare_live_provider:stock_zh_a_hist", point_in_time=True)

    monkeypatch.setattr(AkShareLiveProvider, "daily_ohlcv", fake_daily)
    output_root = tmp_path / "lake"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-akshare-market-panel-v7",
            "--symbols",
            "600519.SH",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--output-root",
            str(output_root),
            "--allow-network",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert (output_root / "manifests" / "market_panel.json").exists()


def test_build_akshare_market_panel_auto_dates_after_qlib_calendar(monkeypatch, tmp_path):
    from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
    from quantagent.data.providers.base import ProviderResult

    qlib_root = tmp_path / "qlib" / "cn_data"
    calendar = qlib_root / "calendars"
    calendar.mkdir(parents=True)
    (calendar / "day.txt").write_text("2020-09-24\n2020-09-25\n", encoding="utf-8")

    def fake_daily(self, request):
        assert request.start_date == "2020-09-28"
        assert request.end_date == "2026-05-15"
        frame = pd.DataFrame(
            [
                {
                    "symbol": "600519.SH",
                    "trade_date": "2021-01-04",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000,
                    "amount": 10500.0,
                    "available_at": "2021-01-05",
                }
            ]
        )
        return ProviderResult(frame, source="akshare_live_provider:stock_zh_a_hist", point_in_time=True)

    monkeypatch.setattr(AkShareLiveProvider, "daily_ohlcv", fake_daily)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-akshare-market-panel-v7",
            "--symbols",
            "600519.SH",
            "--output-root",
            str(tmp_path / "lake"),
            "--provider-uri-for-range",
            str(qlib_root),
            "--as-of-date",
            "2026-05-17",
            "--allow-network",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["resolved_range"]["source"] == "after_qlib_calendar"
    assert payload["resolved_range"]["start_date"] == "2020-09-28"
    assert payload["resolved_range"]["end_date"] == "2026-05-15"
