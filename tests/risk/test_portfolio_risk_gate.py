from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.risk.portfolio_risk import (
    build_portfolio_risk_snapshot,
    compute_realized_tracking_error,
)
from quantagent.risk.risk_gate import RiskGate
from quantagent.risk.risk_limits import V6RiskLimits


def _limits(**overrides) -> V6RiskLimits:
    base = dict(
        max_name_weight=1.0,
        max_sector_weight=1.0,
        max_turnover=2.0,
        max_leverage=1.0,
        beta_exposure_limit=1.2,
        min_beta_coverage=0.95,
        min_sector_coverage=0.95,
        min_style_coverage=0.90,
        max_risk_evidence_age_days=5.0,
    )
    base.update(overrides)
    return V6RiskLimits(**base)


def _market(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "is_suspended": False,
            "is_st": False,
            "is_limit_up": False,
            "is_limit_down": False,
        }
    )


def _snapshot(
    weights: pd.Series,
    *,
    beta: pd.Series | None = None,
    sector: pd.Series | None = None,
    style: pd.DataFrame | None = None,
    beta_pit_safe: bool = True,
    sector_pit_safe: bool = True,
    style_pit_safe: bool = True,
    portfolio_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    min_tracking_overlap: int = 3,
):
    symbols = list(weights.index)
    if beta is None:
        beta = pd.Series(1.0, index=symbols)
    if sector is None:
        sector = pd.Series([f"sector-{i}" for i in range(len(symbols))], index=symbols)
    return build_portfolio_risk_snapshot(
        weights,
        beta=beta,
        sector=sector,
        style_loadings=style,
        beta_pit_safe=beta_pit_safe,
        sector_pit_safe=sector_pit_safe,
        style_pit_safe=style_pit_safe if style is not None else None,
        beta_freshness_days=0.0,
        sector_freshness_days=0.0,
        style_freshness_days=0.0 if style is not None else None,
        beta_source="pit-beta-test",
        sector_source="pit-sector-test",
        style_source="pit-style-test" if style is not None else None,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        min_tracking_overlap=min_tracking_overlap,
        benchmark_symbol="000300.SH" if benchmark_returns is not None else None,
        tracking_frequency="daily" if benchmark_returns is not None else None,
        as_of="2026-08-08",
    )


def _production_inputs(weights: pd.Series) -> dict[str, object]:
    return {
        "market_state": _market(list(weights.index)),
        "conformal_width": pd.Series(0.01, index=weights.index),
        "production_mode": True,
    }


def test_gross_leverage_is_a_deterministic_hard_gate() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.6})
    result = RiskGate(_limits()).check_target_weights(weights, risk_snapshot=_snapshot(weights))
    assert result.passed is False
    assert result.status == "blocked"
    assert "max_leverage" in result.violations


def test_beta_limit_is_enforced_when_evidence_is_complete() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    snapshot = _snapshot(weights, beta=pd.Series({"A": 2.0, "B": 2.0}))
    result = RiskGate(_limits()).check_target_weights(weights, risk_snapshot=snapshot)
    assert result.passed is False
    assert "beta_exposure_limit" in result.violations


def test_missing_beta_is_unknown_in_research_but_blocked_in_production() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    snapshot = _snapshot(weights, beta=pd.Series({"A": 1.0}))
    research = RiskGate(_limits()).check_target_weights(weights, risk_snapshot=snapshot)
    assert research.passed is False
    assert research.status == "unknown"
    assert "beta_coverage_below_threshold" in research.unknowns
    assert "beta_coverage_below_threshold" not in research.violations

    production = RiskGate(_limits()).check_target_weights(
        weights,
        risk_snapshot=snapshot,
        **_production_inputs(weights),
    )
    assert production.passed is False
    assert production.status == "blocked"
    assert "beta_coverage_below_threshold" in production.violations


def test_non_pit_sector_and_style_evidence_cannot_pass_production() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    style = pd.DataFrame({"size": [0.2, 0.3]}, index=weights.index)
    snapshot = _snapshot(
        weights,
        style=style,
        sector_pit_safe=False,
        style_pit_safe=False,
    )
    result = RiskGate(_limits(style_exposure_limits=(("size", 1.0),))).check_target_weights(
        weights,
        risk_snapshot=snapshot,
        **_production_inputs(weights),
    )
    assert "sector_pit_evidence_missing" in result.violations
    assert "style_pit_evidence_missing" in result.violations


def test_configured_style_limit_is_enforced_only_with_valid_metadata() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    style = pd.DataFrame({"size": [1.5, 1.5]}, index=weights.index)
    snapshot = _snapshot(weights, style=style)
    result = RiskGate(_limits(style_exposure_limits=(("size", 0.5),))).check_target_weights(
        weights,
        risk_snapshot=snapshot,
    )
    assert "style_exposure_limit:size" in result.violations


def test_tracking_error_aligns_timestamps_and_fails_closed_on_overlap() -> None:
    p = pd.Series(
        [0.01, 0.02, -0.01, 0.005],
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
    )
    b = pd.Series(
        [0.0, 0.01, 0.0, 0.02],
        index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
    )
    too_short, overlap = compute_realized_tracking_error(p, b, min_overlap=4)
    assert too_short is None
    assert overlap == 3
    value, overlap = compute_realized_tracking_error(p, b, min_overlap=3)
    assert overlap == 3
    assert value is not None and np.isfinite(value) and value > 0


def test_tracking_error_limit_requires_aligned_overlap_and_benchmark_metadata() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    idx = pd.date_range("2026-01-01", periods=4, freq="B")
    snapshot = _snapshot(
        weights,
        portfolio_returns=pd.Series([0.02, 0.01, -0.01, 0.03], index=idx),
        benchmark_returns=pd.Series([0.00, 0.00, 0.00, 0.00], index=idx),
        min_tracking_overlap=3,
    )
    result = RiskGate(_limits(tracking_error_limit=0.01, min_tracking_overlap=3)).check_target_weights(
        weights,
        risk_snapshot=snapshot,
    )
    assert "tracking_error_limit" in result.violations


def test_missing_conformal_and_market_state_rows_are_not_safe_in_production() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.4})
    snapshot = _snapshot(weights)
    result = RiskGate(_limits()).check_target_weights(
        weights,
        risk_snapshot=snapshot,
        conformal_width=pd.Series({"A": 0.01}),
        market_state=_market(["A"]),
        production_mode=True,
    )
    assert "conformal_evidence_missing:B" in result.violations
    assert "market_state_missing:B" in result.violations


def test_market_state_without_symbol_column_blocks_without_crashing() -> None:
    weights = pd.Series({"A": 0.4})
    result = RiskGate(_limits()).check_target_weights(
        weights,
        risk_snapshot=_snapshot(weights),
        conformal_width=pd.Series({"A": 0.01}),
        market_state=pd.DataFrame({"is_st": [False]}),
        production_mode=True,
    )
    assert result.status == "blocked"
    assert "market_state_symbol_column_missing" in result.violations


def test_snapshot_is_cryptographically_bound_to_target_weights() -> None:
    original = pd.Series({"A": 0.4, "B": 0.4})
    changed = pd.Series({"A": 0.3, "B": 0.5})
    result = RiskGate(_limits()).check_target_weights(
        changed,
        risk_snapshot=_snapshot(original),
    )
    assert "portfolio_risk_snapshot_target_mismatch" in result.violations
