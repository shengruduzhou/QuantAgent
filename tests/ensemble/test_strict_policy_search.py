from __future__ import annotations

import pandas as pd

from quantagent.ensemble.strict_policy_search import (
    ExpertPolicy,
    RegimePolicy,
    StrictPolicySearchConfig,
    build_regime_policy_composite,
    scale_target_weights_by_regime,
)


def test_regime_policy_composite_uses_date_specific_expert():
    dates = pd.bdate_range("2024-01-01", periods=2)
    symbols = ["S001", "S002"]
    per_horizon = {
        "short_5d": pd.DataFrame([
            {"trade_date": dates[0], "symbol": symbols[0], "alpha_score": 2.0},
            {"trade_date": dates[0], "symbol": symbols[1], "alpha_score": 1.0},
            {"trade_date": dates[1], "symbol": symbols[0], "alpha_score": 2.0},
            {"trade_date": dates[1], "symbol": symbols[1], "alpha_score": 1.0},
        ]),
        "mid_5d_30d": pd.DataFrame([
            {"trade_date": d, "symbol": s, "alpha_score": 0.0}
            for d in dates for s in symbols
        ]),
        "long_30d_120d": pd.DataFrame([
            {"trade_date": dates[0], "symbol": symbols[0], "alpha_score": 1.0},
            {"trade_date": dates[0], "symbol": symbols[1], "alpha_score": 2.0},
            {"trade_date": dates[1], "symbol": symbols[0], "alpha_score": 1.0},
            {"trade_date": dates[1], "symbol": symbols[1], "alpha_score": 2.0},
        ]),
    }
    policy = RegimePolicy(
        global_policy=ExpertPolicy((0.0, 1.0, 0.0)),
        regime_policies={
            "bull": ExpertPolicy((1.0, 0.0, 0.0)),
            "bear": ExpertPolicy((0.0, 0.0, 1.0)),
        },
    )
    regimes = pd.Series(["bull", "bear"], index=dates)

    composite = build_regime_policy_composite(per_horizon, policy, regimes)
    day0 = composite[composite["trade_date"] == dates[0]].set_index("symbol")
    day1 = composite[composite["trade_date"] == dates[1]].set_index("symbol")

    assert day0.loc["S001", "composite_score"] > day0.loc["S002", "composite_score"]
    assert day1.loc["S002", "composite_score"] > day1.loc["S001", "composite_score"]


def test_scale_target_weights_by_regime_uses_policy_gross_scale():
    dates = pd.bdate_range("2024-01-01", periods=2)
    weights = pd.DataFrame({"S001": [0.5, 0.5], "S002": [0.5, 0.5]}, index=dates)
    policy = RegimePolicy(
        global_policy=ExpertPolicy((1.0, 0.0, 0.0), gross_scale=1.0),
        regime_policies={
            "bull": ExpertPolicy((1.0, 0.0, 0.0), gross_scale=1.0),
            "bear": ExpertPolicy((0.0, 0.0, 1.0), gross_scale=0.2),
        },
    )
    regimes = pd.Series(["bull", "bear"], index=dates)

    scaled = scale_target_weights_by_regime(weights, policy, regimes)

    assert scaled.loc[dates[0]].sum() == 1.0
    assert scaled.loc[dates[1]].sum() == 0.2


def test_strict_policy_config_defaults_are_return_first():
    cfg = StrictPolicySearchConfig()

    assert cfg.return_weight >= cfg.drawdown_penalty
    assert cfg.excess_weight >= cfg.turnover_penalty
