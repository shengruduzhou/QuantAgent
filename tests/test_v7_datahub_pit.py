import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache
from quantagent.data.v7_datahub import V7DataHub, V7DataHubConfig, V7DataQualityError


def _seed_minimum_local_inputs(root) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"document_id": "p1", "title": "AI policy", "body": "support AI compute", "source": "ministry", "source_level": "ministry", "published_at": "2026-04-01", "available_at": "2026-04-01"}
        ]
    ).to_csv(root / "policies.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "600519.SH", "company_name": "Test Liquor", "sector": "consumer", "industry": "liquor", "liquidity_score": 80.0, "is_st": False, "available_at": "2026-04-01"}
        ]
    ).to_csv(root / "base_universe.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "600519.SH", "liquidity_score": 80.0, "market_attention_score": 70.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False, "available_at": "2026-04-01"}
        ]
    ).to_csv(root / "market_state.csv", index=False)
    pd.DataFrame(
        [
            {"trade_date": "2026-04-01", "symbol": "600519.SH", "open": 1500.0, "high": 1520.0, "low": 1490.0, "close": 1510.0, "volume": 1000000, "amount": 1_500_000_000.0, "available_at": "2026-04-01"}
        ]
    ).to_csv(root / "market_panel.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "600519.SH", "theme": "ai_compute", "exposure_score": 60.0, "available_at": "2026-04-01"}
        ]
    ).to_csv(root / "company_theme_map.csv", index=False)


def test_v7_datahub_rejects_missing_required_tables(tmp_path):
    hub = V7DataHub(
        V7DataHubConfig(
            root=str(tmp_path / "v7_missing"),
            fundamentals_root=str(tmp_path / "v7_missing" / "fundamentals"),
            provider_mode="strict_local",
            required_tables=("policies", "base_universe", "market_state", "market_panel", "fundamentals"),
            use_financial_cache=False,
        )
    )
    with pytest.raises(V7DataQualityError) as excinfo:
        hub.load(ProviderRequest("2026-01-01", "2026-05-15"), as_of_date="2026-05-15")
    assert "fundamentals" in str(excinfo.value)


def test_v7_datahub_enriches_fundamentals_from_cache(tmp_path):
    root = tmp_path / "v7_cache"
    _seed_minimum_local_inputs(root)
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(root / "fundamentals")))
    cache.upsert(
        "income",
        pd.DataFrame(
            [
                {"symbol": "600519.SH", "report_period": "2025-12-31", "ann_date": "2026-03-29", "available_at": "2026-03-31", "revenue": 1200.0, "net_income": 480.0, "cogs": 240.0},
            ]
        ),
    )
    cache.upsert(
        "balance_sheet",
        pd.DataFrame(
            [
                {"symbol": "600519.SH", "report_period": "2025-12-31", "ann_date": "2026-03-29", "available_at": "2026-03-31", "total_assets": 4000.0, "total_liabilities": 800.0, "receivables": 60.0, "inventory": 320.0, "goodwill": 0.0},
            ]
        ),
    )
    cache.upsert(
        "cashflow",
        pd.DataFrame(
            [
                {"symbol": "600519.SH", "report_period": "2025-12-31", "ann_date": "2026-03-29", "available_at": "2026-03-31", "operating_cash_flow": 500.0, "capex": -80.0},
            ]
        ),
    )

    hub = V7DataHub(
        V7DataHubConfig(
            root=str(root),
            fundamentals_root=str(root / "fundamentals"),
            provider_mode="strict_local",
            required_tables=("policies", "base_universe", "market_state", "market_panel", "fundamentals"),
            use_financial_cache=True,
            enforce_pit_fundamentals=True,
        )
    )
    result = hub.load(ProviderRequest("2026-01-01", "2026-05-15"), as_of_date="2026-05-15")
    frame = result.bundle.fundamentals.frame

    assert not frame.empty
    assert "available_at" in frame.columns
    assert (pd.to_datetime(frame["available_at"]) <= pd.Timestamp("2026-05-15")).all()
    assert "gross_margin" in frame.columns


def test_v7_datahub_drops_future_fundamentals_when_pit_enforced(tmp_path):
    root = tmp_path / "v7_future"
    _seed_minimum_local_inputs(root)
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(root / "fundamentals")))
    cache.upsert(
        "income",
        pd.DataFrame(
            [
                {"symbol": "600519.SH", "report_period": "2025-12-31", "ann_date": "2026-03-29", "available_at": "2026-03-31", "revenue": 1200.0, "net_income": 480.0, "cogs": 240.0},
                # Future-dated report that must not leak in
                {"symbol": "600519.SH", "report_period": "2026-06-30", "ann_date": "2026-08-15", "available_at": "2026-08-16", "revenue": 1500.0, "net_income": 600.0, "cogs": 280.0},
            ]
        ),
    )

    hub = V7DataHub(
        V7DataHubConfig(
            root=str(root),
            fundamentals_root=str(root / "fundamentals"),
            provider_mode="strict_local",
            required_tables=("policies", "base_universe", "market_state", "market_panel", "fundamentals"),
            use_financial_cache=True,
            enforce_pit_fundamentals=True,
        )
    )
    result = hub.load(ProviderRequest("2026-01-01", "2026-05-15"), as_of_date="2026-05-15")
    frame = result.bundle.fundamentals.frame
    assert not frame.empty
    assert (pd.to_datetime(frame["available_at"]) <= pd.Timestamp("2026-05-15")).all()


def test_industry_chain_reasoner_strict_mode_keeps_empty_when_no_evidence():
    from quantagent.themes.industry_chain_reasoner import (
        IndustryChainReasonerConfig,
        reason_industry_chain,
    )
    from quantagent.v7.schemas import ThemeLifecycleStage, ThemeProfile

    profile = ThemeProfile(
        theme_name="speculative_theme",
        theme_category="rumor",
        theme_strength=0.10,
        policy_strength=0.05,
        market_strength=0.05,
        industry_fundamental_strength=0.05,
        capital_flow_strength=0.05,
        news_sentiment_strength=0.05,
        lifecycle_stage=ThemeLifecycleStage.NARRATIVE_FORMATION,
        expected_horizon_days=20,
        theme_confidence=0.10,
        bubble_risk=0.10,
        crowding_score=0.10,
        expiry_date="2026-09-01",
        update_frequency="daily",
    )
    result = reason_industry_chain(profile, evidence=[], config=IndustryChainReasonerConfig(strict_no_template_fallback=True))
    assert result.nodes == [] or len(result.nodes) == 0
