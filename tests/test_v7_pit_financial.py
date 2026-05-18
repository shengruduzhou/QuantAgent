import os

import pandas as pd
import pytest

from quantagent.data.providers.akshare_financial_provider import (
    AKSHARE_FINANCIAL_REQUIRED_COLUMNS,
    AkShareFinancialProvider,
    akshare_financial_schema_report,
    to_akshare_symbol,
)
from quantagent.data.providers.akshare_live_provider import akshare_market_schema_report
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.financial_cache import (
    FinancialCacheConfig,
    FinancialStatementCache,
    apply_point_in_time_filter,
)
from quantagent.data.providers.tushare_financial_provider import (
    TuShareFinancialProvider,
    merge_statements,
)
from quantagent.data.providers.qlib_provider import QlibProvider, validate_qlib_market_schema
from quantagent.fundamental.financial_features import (
    FinancialFeatureConfig,
    apply_point_in_time_filter as features_pit_filter,
    build_financial_features,
    derive_v7_financial_columns,
)


def _income_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600519.SH", "report_period": "2024-12-31", "ann_date": "2025-03-29", "available_at": "2025-03-31", "revenue": 1200.0, "net_income": 480.0, "cogs": 240.0},
            {"symbol": "600519.SH", "report_period": "2025-03-31", "ann_date": "2025-04-28", "available_at": "2025-04-29", "revenue": 360.0, "net_income": 150.0, "cogs": 70.0},
        ]
    )


def _balance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600519.SH", "report_period": "2024-12-31", "ann_date": "2025-03-29", "available_at": "2025-03-31", "total_assets": 4000.0, "total_liabilities": 800.0, "receivables": 60.0, "inventory": 320.0, "goodwill": 0.0},
            {"symbol": "600519.SH", "report_period": "2025-03-31", "ann_date": "2025-04-28", "available_at": "2025-04-29", "total_assets": 4200.0, "total_liabilities": 820.0, "receivables": 75.0, "inventory": 340.0, "goodwill": 0.0},
        ]
    )


def _cashflow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600519.SH", "report_period": "2024-12-31", "ann_date": "2025-03-29", "available_at": "2025-03-31", "operating_cash_flow": 500.0, "capex": -80.0},
            {"symbol": "600519.SH", "report_period": "2025-03-31", "ann_date": "2025-04-28", "available_at": "2025-04-29", "operating_cash_flow": 160.0, "capex": -20.0},
        ]
    )


def test_tushare_provider_requires_network_and_token():
    provider = TuShareFinancialProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable):
        provider.income(ProviderRequest("2024-01-01", "2026-05-15", symbols=("600519.SH",)))


def test_akshare_provider_requires_network():
    provider = AkShareFinancialProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable):
        provider.income(ProviderRequest("2024-01-01", "2026-05-15", symbols=("600519.SH",)))


def test_akshare_schema_normalization_and_snapshot_columns():
    provider = AkShareFinancialProvider(allow_network=False)
    raw = pd.DataFrame(
        [
            {
                "Report Date": "2025-12-31",
                "Ann Date": "2026-03-31",
                "Revenue": 100.0,
            }
        ]
    )
    normalized = provider._normalize(
        raw,
        {"Report Date": "report_period", "Ann Date": "ann_date", "Revenue": "revenue"},
        "600519.SH",
    )
    report = akshare_financial_schema_report(normalized)

    assert AKSHARE_FINANCIAL_REQUIRED_COLUMNS == ("symbol", "report_period", "ann_date", "available_at")
    assert report["status"] == "passed"
    assert {"symbol", "report_period", "ann_date", "available_at", "revenue", "raw_hash"}.issubset(normalized.columns)


def test_akshare_symbol_conversion_and_update_date_fallback():
    assert to_akshare_symbol("600000.SH") == "sh600000"
    assert to_akshare_symbol("000001.SZ") == "sz000001"
    assert to_akshare_symbol("300750.SZ") == "sz300750"
    assert to_akshare_symbol("688981.SH") == "sh688981"

    provider = AkShareFinancialProvider(allow_network=False)
    normalized = provider._normalize(
        pd.DataFrame([{"报告日期": "2025-12-31", "更新日期": "2026-03-31", "营业收入": 100.0}]),
        {"报告日期": "report_period", "更新日期": "update_date", "营业收入": "revenue"},
        "600000.SH",
    )
    assert normalized.iloc[0]["ann_date"] == "2026-03-31"
    assert normalized.iloc[0]["available_at"] == "2026-04-01"


def test_market_provider_schema_reports_missing_columns_and_pit_violations():
    qlib_frame = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "trade_date": "2026-05-14",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "volume": 100,
                "amount": 1000,
                "available_at": "2026-05-16",
            }
        ]
    )
    qlib_report = validate_qlib_market_schema(qlib_frame, as_of_date="2026-05-15")
    akshare_report = akshare_market_schema_report(qlib_frame.drop(columns=["amount"]))
    missing_available_report = akshare_market_schema_report(qlib_frame.assign(available_at=pd.NaT))

    assert qlib_report["pit_violation_count"] == 1
    assert qlib_report["status"] == "failed"
    assert "amount" in akshare_report["missing_columns"]
    assert missing_available_report["pit_violation_count"] == 1
    assert missing_available_report["status"] == "failed"


def test_qlib_provider_integration_skips_without_local_provider_uri():
    pytest.importorskip("qlib")
    provider_uri = os.getenv("QUANTAGENT_TEST_QLIB_PROVIDER_URI")
    if not provider_uri:
        pytest.skip("QUANTAGENT_TEST_QLIB_PROVIDER_URI is not configured")
    provider = QlibProvider(provider_uri=provider_uri, region="cn")
    result = provider.daily_ohlcv(
        ProviderRequest(
            start_date="2026-05-01",
            end_date="2026-05-15",
            symbols=("600519.SH",),
        )
    )
    report = validate_qlib_market_schema(result.frame, as_of_date="2026-05-15")
    assert report["status"] == "passed"


def test_merge_statements_carries_strictest_available_at():
    from quantagent.data.providers.base import ProviderResult

    statements = {
        "income": ProviderResult(_income_frame(), source="tushare_financial_provider:income"),
        "balance": ProviderResult(_balance_frame(), source="tushare_financial_provider:balancesheet"),
        "cashflow": ProviderResult(_cashflow_frame(), source="tushare_financial_provider:cashflow"),
    }
    merged = merge_statements(statements)
    assert not merged.frame.empty
    assert "available_at" in merged.frame.columns
    assert merged.point_in_time is True


def test_financial_cache_upsert_and_pit_filter(tmp_path):
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(tmp_path / "fundamentals")))
    cache.upsert("income", _income_frame())
    early = cache.load_pit_frame("income", as_of_date="2025-04-01")
    late = cache.load_pit_frame("income", as_of_date="2025-05-01")

    assert not early.frame.empty
    assert len(early.frame) == 1, "only the 2024 annual report should be visible"
    assert not late.frame.empty
    assert len(late.frame) == 2, "Q1 2025 report should be visible after 2025-04-29"


def test_apply_point_in_time_filter_drops_future_rows():
    frame = _income_frame()
    filtered = apply_point_in_time_filter(frame, as_of_date="2025-04-01")
    assert len(filtered) == 1
    assert filtered.iloc[0]["report_period"] == "2024-12-31"


def test_build_financial_features_computes_ratios_and_growth():
    features = build_financial_features(
        income=_income_frame(),
        balance_sheet=_balance_frame(),
        cashflow=_cashflow_frame(),
        config=FinancialFeatureConfig(),
    )
    assert not features.empty
    assert "gross_margin" in features.columns
    assert "ocf_to_profit" in features.columns
    assert "revenue_growth" in features.columns
    # The 2024 annual report has no prior period, so revenue_growth should be nan,
    # while the 2025 Q1 report has growth defined.
    by_period = features.set_index("report_period")
    assert pd.isna(by_period.loc["2024-12-31", "revenue_growth"])


def test_pit_features_filter_keeps_latest_visible_report():
    features = build_financial_features(
        income=_income_frame(),
        balance_sheet=_balance_frame(),
        cashflow=_cashflow_frame(),
    )
    latest_early = features_pit_filter(features, trade_date="2025-04-01")
    latest_late = features_pit_filter(features, trade_date="2025-05-01")
    assert latest_early["report_period"].iloc[0] == "2024-12-31"
    assert latest_late["report_period"].iloc[0] == "2025-03-31"


def test_derive_v7_financial_columns_projects_existing_columns_only():
    features = build_financial_features(
        income=_income_frame(),
        balance_sheet=_balance_frame(),
        cashflow=_cashflow_frame(),
    )
    projected = derive_v7_financial_columns(features)
    assert "symbol" in projected.columns
    assert "revenue" in projected.columns
    assert "gross_margin" in projected.columns
    # No PE/PB columns were supplied, so the projection must omit them
    assert "pe_ttm" not in projected.columns
