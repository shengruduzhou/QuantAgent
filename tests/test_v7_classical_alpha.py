import numpy as np
import pandas as pd

from quantagent.models.v7_classical_alpha import (
    ClassicalAlphaConfig,
    ClassicalAlphaModel,
    predict_v7_classical_alpha,
)
from quantagent.v7.schemas import (
    ChainRelationType,
    ThematicUniverseMember,
    ThemeLifecycleStage,
    UniverseBucket,
)


def _member(symbol: str) -> ThematicUniverseMember:
    return ThematicUniverseMember(
        symbol=symbol,
        company_name=symbol,
        theme="ai_compute",
        sub_theme="server",
        chain_node="server",
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
    )


def _build_frame(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02", periods=40, freq="B")
    symbols = [f"S{i}" for i in range(8)]
    rows = []
    for index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            momentum = rng.normal(0.0, 0.02) + 0.001 * symbol_index
            theme_strength = 50.0 + 5.0 * symbol_index + rng.normal(0.0, 1.0)
            fundamental = 50.0 + symbol_index * 4.0 + rng.normal(0.0, 1.0)
            forward = momentum * 1.2 + theme_strength / 1500.0 + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 10.0 + index * 0.05 + symbol_index,
                    "ret_1d": momentum,
                    "ret_5d": momentum * 1.1,
                    "momentum_5d": momentum * 1.2,
                    "momentum_20d": momentum * 1.3,
                    "theme_strength": theme_strength,
                    "fundamental_score": fundamental,
                    "exposure_score": 60.0 + symbol_index,
                    "policy_strength": 50.0 + symbol_index * 2.0,
                    "forward_return_5d": forward,
                    "forward_return_20d": forward * 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_classical_alpha_trains_and_predicts_per_symbol():
    frame = _build_frame()
    members = [_member(symbol) for symbol in frame["symbol"].unique()]
    feature_columns = ["momentum_20d", "theme_strength", "fundamental_score", "exposure_score", "policy_strength"]
    model = ClassicalAlphaModel(ClassicalAlphaConfig(model="ridge", min_train_rows=20, feature_columns=tuple(feature_columns)))
    model.fit(frame, feature_columns)
    alphas = model.predict(frame, members)
    assert set(alphas.keys()) == set(member.symbol for member in members)
    # The ridge baseline should produce non-trivial expected returns
    assert any(abs(alpha.expected_return) > 1e-4 for alpha in alphas.values())


def test_elastic_net_baseline_handles_small_panels():
    frame = _build_frame(seed=7)
    members = [_member(symbol) for symbol in frame["symbol"].unique()]
    feature_columns = ["momentum_20d", "theme_strength", "fundamental_score", "exposure_score"]
    alphas = predict_v7_classical_alpha(
        frame,
        members,
        config=ClassicalAlphaConfig(model="elastic_net", min_train_rows=20, max_iter=50, learning_rate=0.05),
        feature_columns=feature_columns,
    )
    assert alphas
    for alpha in alphas.values():
        assert -1.0 <= alpha.alpha_5d <= 1.0
        assert -1.0 <= alpha.alpha_120d <= 1.0


def test_classical_alpha_falls_back_when_no_labels():
    rows = []
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    for index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(["A", "B", "C"]):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "momentum_20d": 0.01 * symbol_index,
                    "theme_strength": 60.0,
                    "fundamental_score": 55.0,
                }
            )
    frame = pd.DataFrame(rows)
    members = [_member(symbol) for symbol in ("A", "B", "C")]
    alphas = predict_v7_classical_alpha(frame, members, config=ClassicalAlphaConfig(min_train_rows=10))
    assert set(alphas.keys()) == {"A", "B", "C"}
