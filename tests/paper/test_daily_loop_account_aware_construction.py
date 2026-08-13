"""Account-state propagation contract for the daily paper target builder."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

import quantagent.paper.daily_loop as daily_loop
from quantagent.paper.account_target_state import (
    PaperAccountStateRefused,
    PaperAccountTargetState,
)
from quantagent.paper.pending_signal import PendingPaperSignalStore
from quantagent.portfolio.v7_target_weights import V7TargetWeightsResult


def _identity() -> SimpleNamespace:
    return SimpleNamespace(
        schema_version="quantagent.paper.account_identity.v1",
        account_instance_id="test-account",
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        initial_cash_cny="1000000.00",
        payload_sha256="c" * 64,
    )


def _frames(as_of: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return features, market, predictions


def _install_common_doubles(
    *,
    tmp_path,
    monkeypatch,
    feature_path,
    market_path,
    features: pd.DataFrame,
    market: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    monkeypatch.setattr(daily_loop, "ensure_paper_account_identity", lambda **_kwargs: _identity())
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

    def fake_write_frame(frame: pd.DataFrame, path):
        output = tmp_path / (
            "written_"
            + pd.io.common.stringify_path(path).split("/")[-1].replace(".parquet", ".csv")
        )
        frame.to_csv(output, index=False)
        return output

    monkeypatch.setattr(daily_loop, "write_frame", fake_write_frame)
    monkeypatch.setattr(
        daily_loop,
        "write_v7_target_weights",
        lambda result, path: fake_write_frame(result.target_weights, path),
    )


def _desired(as_of: str) -> V7TargetWeightsResult:
    return V7TargetWeightsResult(
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(as_of)],
                "600000.SH": [0.30],
            }
        ),
        {"status": "passed"},
    )


def _config(tmp_path, as_of: str, feature_path, market_path) -> daily_loop.DailyPaperLoopConfig:
    return daily_loop.DailyPaperLoopConfig(
        as_of_date=as_of,
        model_dir=str(tmp_path / "model"),
        feature_dataset_path=str(feature_path),
        market_panel_path=str(market_path),
        output_root=str(tmp_path / "reports"),
        paper_book_path=str(tmp_path / "paper_book.parquet"),
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        account_identity_path=str(tmp_path / "identity.json"),
        max_turnover=0.40,
        dry_run_evidence=True,
    )


def test_daily_loop_uses_same_recovered_nav_and_weights_for_target_construction_and_reconciliation(
    tmp_path,
    monkeypatch,
):
    as_of = "2026-08-07"
    feature_path = tmp_path / "features.parquet"
    market_path = tmp_path / "market.parquet"
    feature_path.touch()
    market_path.touch()
    features, market, predictions = _frames(as_of)
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

    _install_common_doubles(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        feature_path=feature_path,
        market_path=market_path,
        features=features,
        market=market,
        predictions=predictions,
    )

    lock_state: dict[str, object] = {"held": False, "enters": 0}

    @contextmanager
    def fake_account_lock(path, **_kwargs):
        lock_state["path"] = str(path)
        lock_state["enters"] = int(lock_state["enters"]) + 1
        lock_state["held"] = True
        try:
            yield path
        finally:
            lock_state["held"] = False

    monkeypatch.setattr(daily_loop, "paper_account_lock", fake_account_lock)

    recovered = {"count": 0, "fresh_under_lock": False}

    def fake_recover(**kwargs):
        recovered["count"] += 1
        assert kwargs["market_panel"].equals(market)
        assert kwargs["as_of_date"] == as_of
        if recovered["count"] == 2:
            recovered["fresh_under_lock"] = bool(lock_state["held"])
        return account_state

    monkeypatch.setattr(daily_loop, "recover_paper_account_target_state", fake_recover)

    captured: dict[str, object] = {}
    desired = _desired(as_of)

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

    real_store = PendingPaperSignalStore

    class LockCheckingStore:
        def __init__(self, root):
            self._delegate = real_store(root)

        def read(self, signal_date):
            return self._delegate.read(signal_date)

        def record(self, **kwargs):
            captured["record_lock_held"] = bool(lock_state["held"])
            return self._delegate.record(**kwargs)

    monkeypatch.setattr(daily_loop, "PendingPaperSignalStore", LockCheckingStore)

    real_write_frame = daily_loop.write_frame
    real_write_target = daily_loop.write_v7_target_weights

    def lock_checked_write_frame(frame, path):
        captured["prediction_write_lock_held"] = bool(lock_state["held"])
        return real_write_frame(frame, path)

    def lock_checked_write_target(result, path):
        captured["target_write_lock_held"] = bool(lock_state["held"])
        return real_write_target(result, path)

    monkeypatch.setattr(daily_loop, "write_frame", lock_checked_write_frame)
    monkeypatch.setattr(daily_loop, "write_v7_target_weights", lock_checked_write_target)

    config = _config(tmp_path, as_of, feature_path, market_path)
    result = daily_loop.run_once(config)

    assert result.status == "signal_recorded_pending_execution"
    assert recovered["count"] == 2
    assert recovered["fresh_under_lock"] is True
    assert captured["prediction_write_lock_held"] is True
    assert captured["target_write_lock_held"] is True
    assert captured["record_lock_held"] is True
    assert lock_state["enters"] == 1
    assert lock_state["path"] == config.canonical_ledger_path
    assert lock_state["held"] is False
    target_config = captured["config"]
    assert target_config.capital_yuan == 5_000_000.0
    pd.testing.assert_series_equal(
        captured["initial_weights"],
        account_state.current_weights,
    )
    assert captured["reconcile_result"] is desired
    assert captured["reconcile_state"] is account_state
    assert captured["reconcile_turnover"] == 0.40


def test_daily_loop_discards_target_if_canonical_account_changes_before_freeze(
    tmp_path,
    monkeypatch,
):
    as_of = "2026-08-07"
    feature_path = tmp_path / "features.parquet"
    market_path = tmp_path / "market.parquet"
    feature_path.touch()
    market_path.touch()
    features, market, predictions = _frames(as_of)

    initial_state = PaperAccountTargetState(
        as_of_date=as_of,
        current_weights=pd.Series({"600000.SH": 0.20}),
        quantities=pd.Series({"600000.SH": 100_000.0}),
        cash=4_000_000.0,
        nav=5_000_000.0,
        canonical_records=7,
        canonical_head_hash="b" * 64,
        account_state_sha256="a" * 64,
    )
    changed_state = PaperAccountTargetState(
        as_of_date=as_of,
        current_weights=pd.Series({"600000.SH": 0.25}),
        quantities=pd.Series({"600000.SH": 125_000.0}),
        cash=3_750_000.0,
        nav=5_000_000.0,
        canonical_records=8,
        canonical_head_hash="d" * 64,
        account_state_sha256="e" * 64,
    )

    _install_common_doubles(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        feature_path=feature_path,
        market_path=market_path,
        features=features,
        market=market,
        predictions=predictions,
    )

    @contextmanager
    def fake_account_lock(path, **_kwargs):
        yield path

    monkeypatch.setattr(daily_loop, "paper_account_lock", fake_account_lock)

    states = iter([initial_state, changed_state])
    monkeypatch.setattr(
        daily_loop,
        "recover_paper_account_target_state",
        lambda **_kwargs: next(states),
    )

    desired = _desired(as_of)
    monkeypatch.setattr(daily_loop, "build_v7_target_weights", lambda *args, **kwargs: desired)
    monkeypatch.setattr(
        daily_loop,
        "reconcile_target_to_canonical_account",
        lambda result, **_kwargs: result,
    )

    config = _config(tmp_path, as_of, feature_path, market_path)
    with pytest.raises(PaperAccountStateRefused, match="changed during target construction"):
        daily_loop.run_once(config)

    assert not (tmp_path / "written_predictions.csv").exists()
    assert not (tmp_path / "written_target_weights.csv").exists()
    assert not (tmp_path / "pending" / f"{as_of}.json").exists()


def test_daily_loop_refuses_same_date_writer_before_overwriting_bound_artifacts(
    tmp_path,
    monkeypatch,
):
    as_of = "2026-08-07"
    feature_path = tmp_path / "features.parquet"
    market_path = tmp_path / "market.parquet"
    feature_path.touch()
    market_path.touch()
    features, market, predictions = _frames(as_of)
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

    _install_common_doubles(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        feature_path=feature_path,
        market_path=market_path,
        features=features,
        market=market,
        predictions=predictions,
    )

    @contextmanager
    def fake_account_lock(path, **_kwargs):
        yield path

    monkeypatch.setattr(daily_loop, "paper_account_lock", fake_account_lock)
    monkeypatch.setattr(
        daily_loop,
        "recover_paper_account_target_state",
        lambda **_kwargs: account_state,
    )
    desired = _desired(as_of)
    monkeypatch.setattr(daily_loop, "build_v7_target_weights", lambda *args, **kwargs: desired)
    monkeypatch.setattr(
        daily_loop,
        "reconcile_target_to_canonical_account",
        lambda result, **_kwargs: result,
    )

    config = _config(tmp_path, as_of, feature_path, market_path)
    PendingPaperSignalStore(config.pending_signal_dir).record(
        signal_date=as_of,
        target_weights=desired.target_weights,
        source_lineage={"test": "already-frozen"},
        created_at="2026-08-13T00:00:00+00:00",
    )
    prediction_artifact = tmp_path / "written_predictions.csv"
    target_artifact = tmp_path / "written_target_weights.csv"
    prediction_artifact.write_text("original-predictions\n", encoding="utf-8")
    target_artifact.write_text("original-target\n", encoding="utf-8")

    with pytest.raises(PaperAccountStateRefused, match="already frozen"):
        daily_loop.run_once(config)

    assert prediction_artifact.read_text(encoding="utf-8") == "original-predictions\n"
    assert target_artifact.read_text(encoding="utf-8") == "original-target\n"