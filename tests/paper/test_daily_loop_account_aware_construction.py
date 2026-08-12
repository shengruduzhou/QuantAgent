"""Account-state propagation contract for the daily paper target builder."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import quantagent.paper.daily_loop as daily_loop
from quantagent.paper.account_target_state import PaperAccountTargetState
from quantagent.portfolio.v7_target_weights import V7TargetWeightsResult


def test_daily_loop_uses_same_recovered_nav_and_weights_for_target_construction_and_reconciliation(
    tmp_path,
    monkeypatch,
):
    as_of = "2026-08-07"
    feature_path = tmp_path / "features.parquet"
    market_path = tmp_path / "market.parquet"
    feature_path.touch()
    market_path.touch()

    features = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp(as_of)],
            "symbol": ["600000.SH"],
            "feature_x": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp(as_of)],
            "symbol": ["600000.SH"],
            "close": [10.0],
            "amount": [10_000_000.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp(as_of)],
            "symbol": ["600000.SH"],
            "prediction": [0.1],
            "confidence": [0.9],
        }
    )
    account_state = PaperAccountTargetState(
        as_of_date=as_of,
        current_weights=pd.Series({"600000.SH": 0.20}),
        quantities=pd.Series({"600000.SH": 100_000.0}),
        cash=4_000_000.0,
        nav=5_000_000.0,
        canonical_records=7,
        canonical_head_hash="b" * 64,
        account_state_sha256="a" * 64,
    )
    identity = SimpleNamespace(
        schema_version="quantagent.paper.account_identity.v1",
        account_instance_id="test-account",
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        initial_cash_cny="1000000.00",
        payload_sha256="c" * 64,
    )

    monkeypatch.setattr(daily_loop, "ensure_paper_account_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        daily_loop,
        "DailyEvidenceJob",
        lambda: SimpleNamespace(
            run=lambda _cfg: SimpleNamespace(frame=pd.DataFrame([{"ok": True}]), warnings=())
        ),
    )
    monkeypatch.setattr(
        daily_loop,
        "read_frame",
        lambda path: features.copy() if str(path) == str(feature_path) else market.copy(),
    )
    monkeypatch.setattr(
        daily_loop,
        "predict_v7_alpha",
        lambda *args, **kwargs: SimpleNamespace(predictions=predictions.copy()),
    )
    monkeypatch.setattr(
        daily_loop,
        "blend_multi_horizon_predictions",
        lambda *args, **kwargs: SimpleNamespace(
            blended=predictions.copy(), diagnostics={"test": True}
        ),
    )

    recovered = {"count": 0}

    def fake_recover(**kwargs):
        recovered["count"] += 1
        assert kwargs["market_panel"].equals(market)
        assert kwargs["as_of_date"] == as_of
        return account_state

    monkeypatch.setattr(daily_loop, "recover_paper_account_target_state", fake_recover)

    captured: dict[str, object] = {}
    desired = V7TargetWeightsResult(
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(as_of)],
                "600000.SH": [0.30],
            }
        ),
        {"status": "passed"},
    )

    def fake_build(*args, **kwargs):
        del args
        captured["config"] = kwargs["config"]
        captured["initial_weights"] = kwargs["initial_weights"].copy()
        return desired

    def fake_reconcile(result, *, account_state: PaperAccountTargetState, max_turnover: float):
        captured["reconcile_result"] = result
        captured["reconcile_state"] = account_state
        captured["reconcile_turnover"] = max_turnover
        return result

    monkeypatch.setattr(daily_loop, "build_v7_target_weights", fake_build)
    monkeypatch.setattr(daily_loop, "reconcile_target_to_canonical_account", fake_reconcile)

    def fake_write_frame(frame: pd.DataFrame, path):
        output = tmp_path / ("written_" + pd.io.common.stringify_path(path).split("/")[-1].replace(".parquet", ".csv"))
        frame.to_csv(output, index=False)
        return output

    monkeypatch.setattr(daily_loop, "write_frame", fake_write_frame)
    monkeypatch.setattr(
        daily_loop,
        "write_v7_target_weights",
        lambda result, path: fake_write_frame(result.target_weights, path),
    )

    config = daily_loop.DailyPaperLoopConfig(
        as_of_date=as_of,
        model_dir=str(tmp_path / "model"),
        feature_dataset_path=str(feature_path),
        market_panel_path=str(market_path),
        output_root=str(tmp_path / "reports"),
        paper_book_path=str(tmp_path / "paper_book.parquet"),
        pending_signal_dir=str(tmp_path / "pending"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        account_identity_path=str(tmp_path / "identity.json"),
        max_turnover=0.40,
        dry_run_evidence=True,
    )

    result = daily_loop.run_once(config)

    assert result.status == "signal_recorded_pending_execution"
    assert recovered["count"] == 1
    target_config = captured["config"]
    assert target_config.capital_yuan == 5_000_000.0
    pd.testing.assert_series_equal(
        captured["initial_weights"],
        account_state.current_weights,
    )
    assert captured["reconcile_result"] is desired
    assert captured["reconcile_state"] is account_state
    assert captured["reconcile_turnover"] == 0.40
