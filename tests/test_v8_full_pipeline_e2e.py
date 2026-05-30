"""End-to-end smoke for the *complete* v8 pipeline (P0–P8).

Threads:

    canonical evidence → PIT lint → capital_flow_thesis → validation →
    decision-axis sector_pool_v8 → extended fundamental ranker →
    market_regime_detector → position_policy →
    horizon model bundles → GA factor weights →
    strict_v8 backtest with cost model + risk_events →
    daily decision report

Every stage's contract is asserted. The fixture data is synthetic
(this is integration, not strategy validation) — what we are
proving is that the seams between modules hold under the spec
shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.data.evidence import (
    to_canonical_evidence_frame,
    validate_pit_safety,
)
from quantagent.data.fundamental.extended_ranker import (
    build_extended_fundamental_ranker,
    ExtendedFundamentalConfig,
)
from quantagent.data.sector.decision_pool import build_sector_pool_v8
from quantagent.data.thesis import (
    CapitalFlowThesisBuilder,
    theses_to_frame,
    validate_theses,
)
from quantagent.diagnostics.daily_decision_report import (
    DailyDecisionInputs,
    build_daily_decision_report,
)
from quantagent.optimization.ga_weight_optimizer import (
    GAConfig, WalkForwardConfig, optimize_factor_weights_ga,
)
from quantagent.portfolio.market_regime_detector import (
    DEFAULT_BENCHMARK_INDICES,
    detect_market_regime,
)
from quantagent.portfolio.position_policy import (
    HeldPosition,
    PositionCandidate,
    PositionClass,
    PositionPolicy,
)
from quantagent.training.horizon_models import (
    build_all_horizon_bundles,
    HorizonClass,
)


def _policy_silver(n: int = 4) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01")
    return pd.DataFrame([
        {
            "event_id": f"p{i}", "source": "csrc",
            "url": f"https://csrc.gov.cn/{i}",
            "announced_at": base + pd.Timedelta(days=i),
            "effective_at": base + pd.Timedelta(days=i),
            "available_at": base + pd.Timedelta(days=i, hours=1),
            "fetched_at": base + pd.Timedelta(days=i, hours=1),
            "title": f"政策{i}", "body_summary": "",
            "themes": ["tech_innovation"], "sectors_hint": ["Semi"],
            "policy_strength": 0.7, "source_version": "v1",
        }
        for i in range(n)
    ])


def _broker_silver(n: int = 3) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-05")
    return pd.DataFrame([
        {
            "event_id": f"br{i}", "symbol": "600519.SH",
            "broker": "中信证券", "broker_tier": "tier_1",
            "announced_at": base + pd.Timedelta(days=i),
            "available_at": base + pd.Timedelta(days=i, hours=1),
            "rating": "buy", "rating_change": "upgrade",
            "target_price": 1900.0, "prev_target_price": 1600.0,
            "target_price_pct_change": 0.1875, "summary": "",
            "broker_credibility": 0.85, "source": "wind",
            "source_version": "v1",
            "fetched_at": base + pd.Timedelta(days=i, hours=2),
        }
        for i in range(n)
    ])


def _metrics_frame(n_symbols: int = 10) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01")
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_symbols):
        rows.append({
            "symbol": f"60000{i:02d}.SH",
            "available_at": base,
            "pe_ttm": 10 + rng.uniform(0, 30),
            "pb": 1.0 + rng.uniform(0, 4),
            "ps_ttm": 0.5 + rng.uniform(0, 5),
            "roe": rng.uniform(-0.1, 0.30),
            "roa": rng.uniform(-0.05, 0.15),
            "gross_margin": rng.uniform(0.05, 0.60),
            "net_margin": rng.uniform(-0.05, 0.25),
            "revenue_yoy": rng.uniform(-0.30, 0.50),
            "net_income_yoy": rng.uniform(-0.50, 0.80),
            "operating_cashflow": rng.uniform(-0.10, 0.20),
            "accruals_quality": rng.uniform(0.0, 1.0),
            "earnings_surprise": rng.uniform(-0.20, 0.20),
            "debt_to_asset": rng.uniform(0.10, 0.80),
            "interest_coverage": rng.uniform(0.5, 20.0),
            "inventory_turnover": rng.uniform(2.0, 15.0),
            "accounts_receivable_growth": rng.uniform(-0.30, 0.80),
            "goodwill_risk": rng.uniform(0.0, 0.30),
            "dividend": rng.uniform(0.0, 0.05),
            "repurchase": rng.uniform(0.0, 0.03),
        })
    return pd.DataFrame(rows)


def _ga_factor_dataset(n_dates: int = 180, n_symbols: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    syms = [f"60000{i:02d}.SH" for i in range(n_symbols)]
    rows = []
    fwd_rows = []
    for d in dates:
        a = rng.uniform(-1.0, 1.0, n_symbols)
        b = rng.uniform(-1.0, 1.0, n_symbols)
        fwd = 0.004 * a + rng.normal(0.0, 0.003, n_symbols)
        for sym, ai, bi, fi in zip(syms, a, b, fwd):
            rows.append({"trade_date": d, "symbol": sym, "factor_a": ai, "factor_b": bi})
            fwd_rows.append({"trade_date": d, "symbol": sym, "forward_return": float(fi)})
    return pd.DataFrame(rows), pd.DataFrame(fwd_rows)


def _market_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-03-01", periods=8)
    rows = []
    for d in dates:
        for sym in ("600519.SH", "600000.SH"):
            rows.append({
                "trade_date": d, "symbol": sym, "close": 100.0,
                "volume": 1_000_000.0, "amount": 10_000_000.0,
                "is_suspended": False, "is_st": False,
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def _target_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {"600519.SH": [0.02] * 8, "600000.SH": [0.02] * 8},
        index=pd.bdate_range("2024-03-01", periods=8),
    )


# ---------------------------------------------------------------------------
# Pipeline e2e
# ---------------------------------------------------------------------------

def test_v8_full_pipeline_e2e(tmp_path: Path):
    # ─── 1. Canonical evidence + PIT lint ───────────────────────────────
    canonical = to_canonical_evidence_frame(
        policy_events=_policy_silver(),
        broker_reports=_broker_silver(),
    )
    assert len(canonical) > 0
    pit = validate_pit_safety(canonical)
    assert pit.passed

    # ─── 2. Capital-flow thesis + validation ────────────────────────────
    theses = CapitalFlowThesisBuilder().build(canonical)
    assert theses
    val_panel = pd.DataFrame([
        {"trade_date": d, "sector_level_1": "Semi", "theme": "Semi",
         "sector_return": 0.004, "benchmark_return": 0.0005}
        for d in pd.bdate_range("2024-03-08", periods=130)
    ])
    updated, val_results = validate_theses(theses, val_panel)
    assert any(r.new_status in ("verified", "partially_verified") for r in val_results)
    thesis_frame = theses_to_frame(updated)

    # ─── 3. Decision-axis sector pool v8 ────────────────────────────────
    sectors = pd.DataFrame([
        {"sector_code": "Semi", "sector_name": "半导体"},
        {"sector_code": "Bank", "sector_name": "银行"},
    ])
    pool_result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-15"),
        sectors=sectors,
        capital_flow_theses=thesis_frame,
    )
    assert len(pool_result.frame) == 2
    assert "final_sector_rank" in pool_result.frame.columns

    # ─── 4. Extended fundamental ranker ─────────────────────────────────
    metrics = _metrics_frame(10)
    sector_map = pd.DataFrame([
        {"symbol": f"60000{i:02d}.SH", "sector_level_1": "Semi"}
        for i in range(10)
    ])
    fr_result = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-03-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    assert "composite_score" in fr_result.frame.columns
    assert (fr_result.frame["composite_score"].dropna().between(0, 1)).all()

    # ─── 5. Market regime detector ──────────────────────────────────────
    idx_rows = []
    for idx_code in DEFAULT_BENCHMARK_INDICES:
        price = 100.0
        for d in pd.bdate_range("2024-01-01", periods=30):
            price *= 1.002
            idx_rows.append({"trade_date": d, "index_code": idx_code,
                              "close": price, "volume": 1e6})
    regime_snap = detect_market_regime(
        trade_date=pd.bdate_range("2024-01-01", periods=30)[-1],
        index_panel=pd.DataFrame(idx_rows),
    )
    assert regime_snap.regime in (
        "bull_expansion", "bull_consolidation", "normal", "caution",
        "bear_capitulation", "crisis",
    )

    # ─── 6. Position policy ─────────────────────────────────────────────
    policy = PositionPolicy()
    cand = PositionCandidate(
        symbol="600519.SH", proposed_weight=0.02,
        proposed_class=PositionClass.MID, confidence=0.85,
    )
    verdict = policy.evaluate_candidate(
        cand, held=[], global_conviction=0.85, regime=regime_snap.regime,
    )
    assert verdict.allowed

    # ─── 7. Horizon model bundles ───────────────────────────────────────
    horizon_panel = pd.DataFrame([
        {"symbol": s, "trade_date": d,
         "available_at": d,
         "rsi_14": 50, "macd_signal": 0,
         "sector_strength_20d": 0.5,
         "pe_ttm": 20, "roe": 0.15,
         "forward_return_1d": 0.001,
         "forward_return_5d": 0.005,
         "forward_return_20d": 0.02,
         "forward_return_60d": 0.05,
         "forward_return_120d": 0.10}
        for d in pd.bdate_range("2024-01-01", periods=15)
        for s in ("A.SH", "B.SH")
    ])
    bundles = build_all_horizon_bundles(horizon_panel)
    assert HorizonClass.SHORT in bundles
    assert HorizonClass.MID in bundles
    assert HorizonClass.LONG in bundles

    # ─── 8. GA factor weights with walk-forward ─────────────────────────
    fp, fwd = _ga_factor_dataset()
    ga = optimize_factor_weights_ga(
        factor_panel=fp, forward_returns=fwd,
        factor_names=["factor_a", "factor_b"],
        ga_config=GAConfig(population_size=8, generations=3, top_k=5, random_seed=11),
        wf_config=WalkForwardConfig(n_folds=2, embargo_days=2,
                                     min_train_days=60, min_test_days=20),
    )
    assert len(ga.fold_results) >= 1
    assert abs(sum(ga.best_weights.values()) - 1.0) < 1e-6

    # ─── 9. Strict V8 backtest ──────────────────────────────────────────
    bt = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        sector_map=sector_map,
        factor_weights=ga.best_weights,
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    bt_dir = tmp_path / "bt"
    paths = bt.write(bt_dir)
    for k in ("metrics", "trades", "selected_stocks", "pnl",
              "failed_orders", "risk_events", "profit_by_stock",
              "profit_by_sector", "factor_weights"):
        assert paths[k].exists()
    factor_w = json.loads(paths["factor_weights"].read_text(encoding="utf-8"))
    assert set(factor_w) == {"factor_a", "factor_b"}

    # ─── 10. Daily decision report ──────────────────────────────────────
    decision_inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-15"),
        target_weights=pd.Series({"600519.SH": 0.02, "600000.SH": 0.02}),
        sector_map=pd.DataFrame([
            {"symbol": "600519.SH", "sector_level_1": "Bank"},
            {"symbol": "600000.SH", "sector_level_1": "Bank"},
        ]),
        sector_pool=pool_result.frame.rename(columns={"sector_code": "sector_level_1"})[["sector_level_1", "final_sector_rank"]].assign(pool_tier="core"),
        capital_flow_theses=thesis_frame,
        risk_events=bt.risk_events,
        market_regime=regime_snap.regime,
        global_conviction=0.85,
        gross_exposure=0.55,
    )
    report = build_daily_decision_report(decision_inputs)
    md_path = tmp_path / "daily.md"
    report.write(md_path)
    md = md_path.read_text(encoding="utf-8")
    assert "Daily Decision Report" in md
    assert "Bank" in md
