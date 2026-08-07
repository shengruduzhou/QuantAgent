from __future__ import annotations

from services.quant_api.services.fuyao_analytics import attention_price_resonance, cashflow_quality


def test_cashflow_quality_uses_official_formulas_and_disclosure_date() -> None:
    health = {
        "symbol": "600519.SH",
        "source": "hithink_fuyao",
        "provenance": {"income": "income", "balance": "balance", "cashflow": "cashflow"},
        "statements": {
            "income": [{"period_end_ms": 100, "report_date_ms": 120, "net_profit": 50, "operating_income": 200}],
            "balance": [{"period_end_ms": 100, "report_date_ms": 125, "assets_total": 500, "accounts_receivable": 40, "cash": 100, "total_debt": 60}],
            "cashflow": [{"period_end_ms": 100, "report_date_ms": 130, "act_cash_flow_net": 60, "pay_fixed_assets_etc_cash": 20}],
        },
    }
    result = cashflow_quality(health)
    row = result["rows"][0]
    assert row["reportDateMs"] == 130
    assert row["cashConversion"] == 1.2
    assert row["freeCashFlow"] == 40.0
    assert row["freeCashFlowMargin"] == 0.2
    assert row["accrualRatio"] == -0.02
    assert row["receivablePressure"] == 0.2
    assert row["netCashRatio"] == 0.08
    assert row["fieldCompleteness"] == 1.0


def test_cashflow_quality_leaves_zero_denominator_ratios_null() -> None:
    result = cashflow_quality({
        "symbol": "X",
        "statements": {
            "income": [{"period_end_ms": 1, "report_date_ms": 2, "net_profit": 0, "operating_income": 0}],
            "balance": [{"period_end_ms": 1, "report_date_ms": 2, "assets_total": 0}],
            "cashflow": [{"period_end_ms": 1, "report_date_ms": 2, "act_cash_flow_net": 1, "pay_fixed_assets_etc_cash": 1}],
        },
    })
    row = result["rows"][0]
    assert row["cashConversion"] is None
    assert row["freeCashFlowMargin"] is None
    assert row["accrualRatio"] is None


def test_attention_price_aligns_only_shared_dates() -> None:
    rank = [
        {"date": "2026-01-01", "rank": 100},
        {"date": "2026-01-02", "rank": 80},
        {"date": "2026-01-03", "rank": 60},
        {"date": "2026-01-04", "rank": 50},
    ]
    stock = [
        {"datetime": "2026-01-01", "close": 10},
        {"datetime": "2026-01-02", "close": 10.5},
        {"datetime": "2026-01-03", "close": 10.2},
        {"datetime": "2026-01-04", "close": 10.8},
    ]
    benchmark = [
        {"datetime": "2026-01-01", "close": 100},
        {"datetime": "2026-01-02", "close": 101},
        {"datetime": "2026-01-03", "close": 102},
        {"datetime": "2026-01-04", "close": 103},
    ]
    result = attention_price_resonance(
        symbol="600519.SH", benchmark="000300.SH", rank_rows=rank, stock_bars=stock, benchmark_bars=benchmark
    )
    assert result["sampleSize"] == 3
    assert len(result["rows"]) == 4
    assert result["rankAxis"].startswith("raw rank")
    assert result["rows"][0]["stockIndexed"] == 100.0
    assert result["rows"][0]["benchmarkIndexed"] == 100.0
