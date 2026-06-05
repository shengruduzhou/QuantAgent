from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantagent.ensemble import blend_optimizer
from quantagent.ensemble.blend_optimizer import BlendObjective


def test_topk_excess_return_rewards_selected_names_beating_cross_section():
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=3):
        for i in range(40):
            rows.append({
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "composite_score": 100 - i,
                "forward_return_20d": 0.10 if i < 30 else -0.10,
            })
    frame = pd.DataFrame(rows)
    objective = BlendObjective(metric="topk_excess_return", target_label="forward_return_20d")

    score = objective.score(frame[["trade_date", "symbol", "composite_score"]], frame)

    assert score > 0


def test_topk_excess_return_penalises_bad_selected_names():
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=3):
        for i in range(40):
            rows.append({
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "composite_score": 100 - i,
                "forward_return_20d": -0.10 if i < 30 else 0.10,
            })
    frame = pd.DataFrame(rows)
    objective = BlendObjective(metric="topk_excess_return", target_label="forward_return_20d")

    score = objective.score(frame[["trade_date", "symbol", "composite_score"]], frame)

    assert score < 0


def test_topk_utility_penalises_selected_set_churn():
    rows = []
    stable_scores = []
    churn_scores = []
    dates = pd.bdate_range("2024-01-01", periods=4)
    for date_idx, d in enumerate(dates):
        for i in range(40):
            rows.append({
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "forward_return_20d": 0.10 if i < 20 else -0.10,
            })
            stable_scores.append({
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "composite_score": 100 - i,
            })
            if date_idx % 2 == 0:
                score = 100 - i
            else:
                score = 100 - abs(i - 10)
            churn_scores.append({
                "trade_date": d,
                "symbol": f"S{i:03d}",
                "composite_score": score,
            })
    realized = pd.DataFrame(rows)
    stable = pd.DataFrame(stable_scores)
    churn = pd.DataFrame(churn_scores)
    objective = BlendObjective(
        metric="topk_utility",
        target_label="forward_return_20d",
        top_k=10,
        drawdown_penalty=0.0,
        turnover_penalty=1.0,
    )

    stable_score = objective.score(stable, realized)
    churn_score = objective.score(churn, realized)

    assert stable_score > churn_score


def test_regime_aware_blend_learns_different_horizon_weights(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=4)
    symbols = [f"S{i:03d}" for i in range(40)]
    per_horizon = {hz: [] for hz in blend_optimizer.HORIZONS}
    realized = []
    for date in dates:
        is_bull = date in dates[:2]
        for i, symbol in enumerate(symbols):
            label = 0.10 if i < 30 else -0.10
            realized.append({
                "trade_date": date,
                "symbol": symbol,
                "forward_return_20d": label,
            })
            per_horizon["short_5d"].append({
                "trade_date": date,
                "symbol": symbol,
                "alpha_score": (100 - i) if is_bull else (2 * i),
            })
            per_horizon["mid_5d_30d"].append({
                "trade_date": date,
                "symbol": symbol,
                "alpha_score": i,
            })
            per_horizon["long_30d_120d"].append({
                "trade_date": date,
                "symbol": symbol,
                "alpha_score": (2 * i) if is_bull else (100 - i),
            })
    fake_predictions = {
        horizon: pd.DataFrame(rows)
        for horizon, rows in per_horizon.items()
    }
    fake_realized = pd.DataFrame(realized)
    monkeypatch.setattr(blend_optimizer, "load_predictions", lambda _path: fake_predictions)
    monkeypatch.setattr(
        blend_optimizer,
        "load_realized_returns",
        lambda *_args, **_kwargs: fake_realized,
    )
    regime_by_date = pd.Series(
        ["bull", "bull", "bear", "bear"],
        index=dates,
    )

    result = blend_optimizer.optimize_regime_aware_blend_weights(
        Path("unused"),
        gold_path=Path("unused"),
        regime_by_date=regime_by_date,
        objective=BlendObjective(metric="topk_excess_return", target_label="forward_return_20d"),
        step=0.5,
        n_folds=1,
        min_regime_days=2,
    )

    bull_weights = result.regime_results["bull"].best_weights
    bear_weights = result.regime_results["bear"].best_weights
    assert bull_weights[0] > bull_weights[2]
    assert bear_weights[2] > bear_weights[0]
