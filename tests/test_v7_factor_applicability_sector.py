import numpy as np
import pandas as pd

from quantagent.factors.factor_applicability_agent import validate_factor_applicability
from quantagent.v7.schemas import (
    ChainRelationType,
    MarketRegime,
    ThematicUniverseMember,
    ThemeLifecycleStage,
    UniverseBucket,
)


def _member(symbol: str, sector: str, chain_node: str) -> ThematicUniverseMember:
    return ThematicUniverseMember(
        symbol=symbol,
        company_name=symbol,
        theme="ai_compute",
        sub_theme="server",
        chain_node=chain_node,
        exposure_type=ChainRelationType.DIRECT_EXPOSURE,
        exposure_score=80.0,
        revenue_exposure_estimate=0.4,
        profit_exposure_estimate=0.3,
        evidence_count=3,
        source_confidence=0.8,
        fundamental_score=75.0,
        valuation_score=55.0,
        quality_score=70.0,
        fraud_risk_score=25.0,
        liquidity_score=70.0,
        market_attention_score=60.0,
        theme_lifecycle_stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        entry_date="2026-05-01",
        expiry_date="2026-09-01",
        last_validated_at="2026-05-14",
        watchlist_status=UniverseBucket.CORE_BENEFICIARY,
        sector=sector,
    )


def test_factor_applicability_uses_real_sector_not_chain_node():
    """Regression test: ``member.sector`` (industry classification) must
    drive the per-sector applicability slice, not ``member.chain_node``
    (which is the position inside an industry chain).
    """

    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    rng = np.random.default_rng(42)
    rows = []
    members = []
    # Two real sectors but the same chain_node "server" — the old bug
    # would group everyone under sector="server" instead.
    config = (
        ("S0", "semiconductor", "gpu"),
        ("S1", "semiconductor", "gpu"),
        ("S2", "semiconductor", "gpu"),
        ("S3", "telecom_equipment", "server"),
        ("S4", "telecom_equipment", "server"),
        ("S5", "telecom_equipment", "server"),
    )
    for symbol, sector, chain_node in config:
        members.append(_member(symbol, sector, chain_node))
    for date_index, date in enumerate(dates):
        for symbol_index, (symbol, sector, chain_node) in enumerate(config):
            close = 10.0 * (1.0 + 0.01 * symbol_index) ** date_index
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": close,
                    "amount": 1_000_000 + symbol_index + rng.integers(0, 100),
                    "theme_momentum": symbol_index + rng.normal(0.0, 0.1),
                }
            )
    frame = pd.DataFrame(rows)
    reports = validate_factor_applicability(
        frame,
        ["theme_momentum"],
        members,
        MarketRegime.POLICY_DRIVEN,
    )
    assert reports
    sectors_seen: set[str] = set()
    for report in reports:
        sectors_seen.update(report.applicable_sector)
    # If the bug still existed, applicable_sector would contain "server" /
    # "gpu" (chain nodes). With the fix it must surface the real sector tags.
    assert "server" not in sectors_seen
    assert "gpu" not in sectors_seen
