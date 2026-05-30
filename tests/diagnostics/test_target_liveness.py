from __future__ import annotations

import pandas as pd


def test_target_weights_not_all_zero_when_predictions_exist_and_gates_open():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights
    from quantagent.diagnostics.target_liveness import build_target_weights_liveness

    date = pd.Timestamp("2024-01-02")
    preds = pd.DataFrame(
        {
            "trade_date": [date] * 6,
            "symbol": [f"S{i}" for i in range(6)],
            "prediction": [0.9, 0.8, 0.7, 0.1, 0.0, -0.1],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": [date] * 6,
            "symbol": [f"S{i}" for i in range(6)],
            "close": [10.0] * 6,
            "amount": [pd.NA] * 6,
            "is_suspended": [False] * 6,
            "is_st": [False] * 6,
            "is_limit_up": [False] * 6,
            "is_limit_down": [False] * 6,
        }
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="top_k",
            top_k=3,
            top_k_ratio=None,
            min_selection_pressure=1.0,
            fail_if_top_k_covers_universe=False,
            optimizer_backend="deterministic",
            max_turnover=0.0,
        ),
    )
    report = build_target_weights_liveness(result.target_weights, predictions=preds, diagnostics=result.diagnostics)

    assert report["summary"]["all_zero"] is False
    assert report["summary"]["target_weights_liveness"] is True
    assert float(report["summary"]["mean_gross"]) > 0
    assert any(row.get("warning") == "liquidity_amount_all_missing_cap_disabled" for row in result.diagnostics["warnings"])


def test_all_zero_weights_require_explicit_kill_reason():
    from quantagent.diagnostics.target_liveness import build_target_weights_liveness

    date = pd.Timestamp("2024-01-02")
    weights = pd.DataFrame({"trade_date": [date], "A": [0.0], "B": [0.0]})
    preds = pd.DataFrame({"trade_date": [date, date], "symbol": ["A", "B"], "prediction": [0.1, 0.2]})
    diagnostics = {"daily_selection": [{"trade_date": str(date), "eligible_count": 2, "selected_count": 2}]}

    report = build_target_weights_liveness(weights, predictions=preds, diagnostics=diagnostics)

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["unexplained_zero_days"] == 1
    assert set(report["zero_weight_days"]["liveness_reason"]) == {"unknown_all_zero_after_candidates"}


def test_all_zero_without_generation_trace_reports_missing_diagnostics():
    from quantagent.diagnostics.target_liveness import build_target_weights_liveness

    date = pd.Timestamp("2024-01-02")
    weights = pd.DataFrame({"trade_date": [date], "A": [0.0], "B": [0.0]})
    preds = pd.DataFrame({"trade_date": [date, date], "symbol": ["A", "B"], "prediction": [0.1, 0.2]})

    report = build_target_weights_liveness(weights, predictions=preds, diagnostics={})

    assert report["summary"]["status"] == "failed"
    assert set(report["zero_weight_days"]["liveness_reason"]) == {"missing_weight_generation_diagnostics"}


def test_all_zero_weights_with_missing_amount_reason_is_explained_but_not_live():
    from quantagent.diagnostics.target_liveness import build_target_weights_liveness

    date = pd.Timestamp("2024-01-02")
    weights = pd.DataFrame({"trade_date": [date], "A": [0.0], "B": [0.0]})
    preds = pd.DataFrame({"trade_date": [date, date], "symbol": ["A", "B"], "prediction": [0.1, 0.2]})
    diagnostics = {
        "daily_selection": [{"trade_date": str(date), "eligible_count": 2, "selected_count": 2}],
        "warnings": [{"trade_date": str(date), "warning": "liquidity_amount_all_missing_cap_disabled"}],
    }

    report = build_target_weights_liveness(weights, predictions=preds, diagnostics=diagnostics)

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["reason"] == "all_zero_but_explained"
    assert report["summary"]["unexplained_zero_days"] == 0


def test_suspended_hard_block_can_zero_specific_symbols_not_entire_portfolio():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights
    from quantagent.diagnostics.target_liveness import build_target_weights_liveness

    date = pd.Timestamp("2024-01-02")
    preds = pd.DataFrame(
        {
            "trade_date": [date, date, date],
            "symbol": ["A", "B", "C"],
            "prediction": [0.9, 0.8, 0.7],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": [date, date, date],
            "symbol": ["A", "B", "C"],
            "close": [10.0, 10.0, 10.0],
            "amount": [1e8, 1e8, 1e8],
            "is_suspended": [True, False, False],
            "is_st": [False, False, False],
            "is_limit_up": [False, False, False],
            "is_limit_down": [False, False, False],
        }
    )
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(
            selection_mode="top_k",
            top_k=2,
            top_k_ratio=None,
            min_selection_pressure=1.0,
            fail_if_top_k_covers_universe=False,
            optimizer_backend="deterministic",
            max_turnover=0.0,
        ),
    )
    frame = result.target_weights
    report = build_target_weights_liveness(frame, predictions=preds, diagnostics=result.diagnostics)

    assert "A" not in frame.columns or float(frame["A"].abs().sum()) == 0.0
    assert report["summary"]["target_weights_liveness"] is True
    assert report["summary"]["nonzero_weight_days"] == 1
