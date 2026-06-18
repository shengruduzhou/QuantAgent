from __future__ import annotations

import math

import pandas as pd


def test_forward_pe_peg_and_digestion_formulas():
    from quantagent.fundamental.peg import forward_pe, pe_digestion_years, peg_ratio

    assert forward_pe(30.0, 1.5) == 20.0
    assert peg_ratio(20.0, 0.25) == 0.8
    years = pe_digestion_years(60.0, 0.30, target_pe=30.0)
    assert years is not None
    assert math.isclose(years, math.log(2.0) / math.log(1.30))


def test_estimate_peg_uses_forward_eps_growth_and_flags_low_coverage():
    from quantagent.fundamental.peg import PegInputs, estimate_peg

    result = estimate_peg(
        PegInputs(
            symbol="688001.SH",
            price=40.0,
            eps_current_year=2.0,
            eps_next_year=2.8,
            analyst_count=1,
        )
    )

    assert result.forward_pe == 20.0
    assert math.isclose(result.growth_rate or 0.0, 0.40)
    assert result.growth_source == "eps_forecast_growth"
    assert result.rating in {"undervalued", "deep_undervalued"}
    assert "low_analyst_coverage" in result.risk_flags
    assert 0.0 <= result.score <= 100.0


def test_estimate_peg_falls_back_to_ttm_pe_but_records_risk():
    from quantagent.fundamental.peg import PegInputs, estimate_peg

    result = estimate_peg(PegInputs(symbol="600001.SH", pe_ttm=45.0, growth_rate=20.0, analyst_count=5))

    assert result.pe_used == 45.0
    assert result.forward_pe is None
    assert result.growth_rate == 0.20
    assert result.peg == 2.25
    assert "used_ttm_pe_fallback" in result.risk_flags
    assert "high_peg" in result.risk_flags


def test_enrich_peg_valuation_adds_stable_overlay_columns():
    from quantagent.fundamental.peg import enrich_peg_valuation

    frame = pd.DataFrame(
        [
            {
                "symbol": "300001.SZ",
                "price": 24.0,
                "eps_forward": 1.2,
                "net_income_cagr": 0.30,
                "analyst_count": 4,
            },
            {
                "symbol": "300002.SZ",
                "price": 24.0,
                "eps_forward": 1.2,
                "net_income_cagr": -0.10,
                "analyst_count": 4,
            },
        ]
    )

    enriched = enrich_peg_valuation(frame)
    assert "peg_ratio" in enriched.columns
    assert "peg_score" in enriched.columns
    assert enriched.loc[0, "peg_ratio"] < 1.0
    assert enriched.loc[0, "peg_score"] > enriched.loc[1, "peg_score"]
    assert "non_positive_growth" in enriched.loc[1, "peg_risk_flags"]


def test_market_valuation_adds_peg_overlay_when_forecast_inputs_exist():
    from quantagent.fundamental.market_valuation import enrich_market_valuation

    frame = pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "close": 30.0,
                "total_shares": 100_000_000,
                "free_float_shares": 80_000_000,
                "net_income": 100_000_000,
                "book_value": 1_000_000_000,
                "eps_forward": 1.5,
                "eps_next_year": 2.0,
                "eps_current_year": 1.5,
                "analyst_count": 4,
            }
        ]
    )

    enriched = enrich_market_valuation(frame)

    assert "peg_ratio" in enriched.columns
    assert enriched.loc[0, "peg_forward_pe"] == 20.0
    assert enriched.loc[0, "peg_confidence"] > 0.5
