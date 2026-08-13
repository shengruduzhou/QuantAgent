from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import json

import pandas as pd
import pytest

import quantagent.paper.daily_loop as daily_loop
from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.paper.account_identity import PaperAccountIdentityMismatch


def _install_common_mocks(tmp_path, monkeypatch, *, targets: pd.DataFrame):
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

    evidence_calls = {"count": 0}

    def evidence_run(cfg):
        del cfg
        evidence_calls["count"] += 1
        return SimpleNamespace(frame=pd.DataFrame([{"ok": True}]), warnings=())

    monkeypatch.setattr(
        daily_loop,
        "DailyEvidenceJob",
        lambda: SimpleNamespace(run=evidence_run),
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
    monkeypatch.setattr(
        daily_loop,
        "build_v7_target_weights",
        lambda *args, **kwargs: SimpleNamespace(
            target_weights=targets.copy(), diagnostics={"status": "passed"}
        ),
    )

    def fake_write_frame(frame: pd.DataFrame, path):
        path = pd.io.common.stringify_path(path)
        target = tmp_path / ("written_" + path.split("/")[-1].replace(".parquet", ".csv"))
        frame.to_csv(target, index=False)
        return target

    monkeypatch.setattr(daily_loop, "write_frame", fake_write_frame)
    monkeypatch.setattr(
        daily_loop,
        "write_v7_target_weights",
        lambda weights, path: fake_write_frame(weights.target_weights, path),
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
        account_identity_path=str(tmp_path / "account_identity.json"),
        dry_run_evidence=True,
    )
    return as_of, config, evidence_calls


def test_daily_loop_records_pending_signal_without_paper_pnl_or_book(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-07")],
            "600000.SH": [1.0],
        }
    )
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)
    result = daily_loop.run_once(config)

    assert result.status == "signal_recorded_pending_execution"
    assert result.execution_timing_semantics == EXECUTION_TIMING_SEMANTICS
    assert result.executed_fill_count == 0
    assert result.pending_signal_path
    assert not (tmp_path / "paper_book.parquet").exists()
    assert (tmp_path / "account_identity.json").exists()

    summary_path = tmp_path / "reports" / as_of / "daily_loop_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "signal_recorded_pending_execution"
    execution = payload["execution"]
    assert execution["executed_fill_count"] == 0
    assert execution["paper_report_written"] is False
    assert execution["paper_book_appended"] is False
    assert payload["paper_report"] is None
    assert payload["account_state"]["canonical_records"] == 0
    assert payload["account_state"]["canonical_head_hash"] == "0" * 64
    assert payload["account_identity"]["portfolio_id"] == "v7-paper"
    assert payload["account_identity"]["initial_cash_cny"] == "1000000.00"
    assert len(payload["account_identity"]["payload_sha256"]) == 64
    assert payload["target_weight_diagnostics"]["canonical_account_reconciliation"]["applied_l1_weight_churn"] == 0.4

    pending = json.loads((tmp_path / "pending" / f"{as_of}.json").read_text(encoding="utf-8"))
    predictions_path = tmp_path / "written_predictions.csv"
    weights_path = tmp_path / "written_target_weights.csv"
    assert pending["source_lineage"]["predictions_file_sha256"] == sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    assert pending["source_lineage"]["target_weights_file_sha256"] == sha256(
        weights_path.read_bytes()
    ).hexdigest()
    assert pending["source_lineage"]["paper_account_identity_sha256"] == payload["account_identity"]["payload_sha256"]
    assert pending["source_lineage"]["canonical_account_state_sha256"] == payload["account_state"]["account_state_sha256"]
    assert pending["source_lineage"]["canonical_ledger_head_hash"] == "0" * 64
    assert execution["paper_account_identity_sha256"] == pending["source_lineage"]["paper_account_identity_sha256"]
    assert execution["predictions_file_sha256"] == pending["source_lineage"]["predictions_file_sha256"]
    assert execution["target_weights_file_sha256"] == pending["source_lineage"]["target_weights_file_sha256"]
    assert execution["canonical_account_state_sha256"] == pending["source_lineage"]["canonical_account_state_sha256"]


def test_daily_loop_no_target_is_not_pending_liquidation_or_paper_evidence(tmp_path, monkeypatch) -> None:
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=pd.DataFrame())
    result = daily_loop.run_once(config)

    assert result.status == "no_target_generated"
    assert result.pending_signal_path == ""
    assert result.executed_fill_count == 0
    assert not (tmp_path / "paper_book.parquet").exists()
    assert not (tmp_path / "pending" / f"{as_of}.json").exists()
    # Account identity is a genesis invariant and is created even if the day's
    # strategy produces no target.
    assert (tmp_path / "account_identity.json").exists()

    payload = json.loads(
        (tmp_path / "reports" / as_of / "daily_loop_summary.json").read_text(encoding="utf-8")
    )
    assert payload["execution"]["status"] == "no_target_generated"
    assert payload["execution"]["paper_report_written"] is False
    assert payload["execution"]["paper_book_appended"] is False
    assert payload["execution"]["paper_account_identity_sha256"] == payload["account_identity"]["payload_sha256"]


def test_daily_loop_mismatched_genesis_fails_before_evidence_or_prediction(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-07")],
            "600000.SH": [0.5],
        }
    )
    _, config, evidence_calls = _install_common_mocks(tmp_path, monkeypatch, targets=targets)
    # First run establishes the immutable account identity.
    daily_loop.run_once(config)
    calls_after_first = evidence_calls["count"]

    with pytest.raises(PaperAccountIdentityMismatch, match="initial_cash mismatch"):
        daily_loop.run_once(replace(config, initial_cash=2_000_000.0))
    assert evidence_calls["count"] == calls_after_first

    with pytest.raises(PaperAccountIdentityMismatch, match="portfolio_id mismatch"):
        daily_loop.run_once(replace(config, portfolio_id="different-paper-book"))
    assert evidence_calls["count"] == calls_after_first


def test_daily_loop_source_has_no_same_call_strict_execution_path() -> None:
    source = __import__("inspect").getsource(daily_loop.run_once)
    assert "simulate_ashare_target_weights" not in source
    assert "write_paper_report" not in source
    assert "_market_for_dates" not in source


def test_target_date_validation_failure_does_not_permanently_freeze_day(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2026-08-06")], "600000.SH": [0.5]}
    )
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)
    with pytest.raises(ValueError, match="requires exactly the current signal date"):
        daily_loop.run_once(config)
    journal = daily_loop.PendingExecutionJournal(daily_loop._execution_journal_path(config))
    assert journal.daily_decision(as_of) is None


def test_artifact_write_failure_does_not_permanently_freeze_day(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2026-08-07")], "600000.SH": [0.5]}
    )
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated artifact write failure")

    monkeypatch.setattr(daily_loop, "write_frame", fail_write)
    with pytest.raises(OSError, match="simulated artifact write failure"):
        daily_loop.run_once(config)
    journal = daily_loop.PendingExecutionJournal(daily_loop._execution_journal_path(config))
    assert journal.daily_decision(as_of) is None


def test_no_target_stale_prediction_date_does_not_freeze_day(tmp_path, monkeypatch) -> None:
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=pd.DataFrame())
    stale = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-06")],
            "symbol": ["600000.SH"],
            "prediction": [0.1],
            "confidence": [0.9],
        }
    )
    monkeypatch.setattr(
        daily_loop,
        "predict_v7_alpha",
        lambda *args, **kwargs: SimpleNamespace(predictions=stale.copy()),
    )
    monkeypatch.setattr(
        daily_loop,
        "blend_multi_horizon_predictions",
        lambda *args, **kwargs: SimpleNamespace(
            blended=stale.copy(), diagnostics={"test": "stale-no-target"}
        ),
    )
    with pytest.raises(daily_loop.PaperAccountStateRefused, match="no-target predictions must belong exactly"):
        daily_loop.run_once(config)
    journal = daily_loop.PendingExecutionJournal(daily_loop._execution_journal_path(config))
    assert journal.daily_decision(as_of) is None


def test_freeze_failure_discards_uncommitted_current_protocol_staging(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2026-08-07")], "600000.SH": [0.5]}
    )
    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)
    original_append = daily_loop.PendingExecutionJournal.append

    def fail_daily_freeze(self, **kwargs):
        if kwargs.get("status") == daily_loop.DAILY_DECISION_STATUS:
            raise OSError("simulated freeze fsync failure")
        return original_append(self, **kwargs)

    monkeypatch.setattr(daily_loop.PendingExecutionJournal, "append", fail_daily_freeze)
    with pytest.raises(OSError, match="freeze fsync failure"):
        daily_loop.run_once(config)
    assert not (tmp_path / "pending" / f"{as_of}.json").exists()
    assert daily_loop.PendingExecutionJournal(
        daily_loop._execution_journal_path(config)
    ).daily_decision(as_of) is None
