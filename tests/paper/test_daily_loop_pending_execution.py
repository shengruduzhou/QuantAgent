from __future__ import annotations

from types import SimpleNamespace
import json

import pandas as pd

import quantagent.paper.daily_loop as daily_loop
from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS


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

    monkeypatch.setattr(
        daily_loop,
        "DailyEvidenceJob",
        lambda: SimpleNamespace(
            run=lambda cfg: SimpleNamespace(frame=pd.DataFrame([{"ok": True}]), warnings=())
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
        dry_run_evidence=True,
    )
    return as_of, config


def test_daily_loop_records_pending_signal_without_paper_pnl_or_book(tmp_path, monkeypatch) -> None:
    targets = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-07")],
            "600000.SH": [1.0],
        }
    )
    as_of, config = _install_common_mocks(tmp_path, monkeypatch, targets=targets)
    result = daily_loop.run_once(config)

    assert result.status == "signal_recorded_pending_execution"
    assert result.execution_timing_semantics == EXECUTION_TIMING_SEMANTICS
    assert result.executed_fill_count == 0
    assert result.pending_signal_path
    assert not (tmp_path / "paper_book.parquet").exists()

    summary_path = tmp_path / "reports" / as_of / "daily_loop_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "signal_recorded_pending_execution"
    execution = payload["execution"]
    assert execution["executed_fill_count"] == 0
    assert execution["paper_report_written"] is False
    assert execution["paper_book_appended"] is False
    assert payload["paper_report"] is None


def test_daily_loop_no_target_is_not_pending_liquidation_or_paper_evidence(tmp_path, monkeypatch) -> None:
    as_of, config = _install_common_mocks(tmp_path, monkeypatch, targets=pd.DataFrame())
    result = daily_loop.run_once(config)

    assert result.status == "no_target_generated"
    assert result.pending_signal_path == ""
    assert result.executed_fill_count == 0
    assert not (tmp_path / "paper_book.parquet").exists()
    assert not (tmp_path / "pending" / f"{as_of}.json").exists()

    payload = json.loads(
        (tmp_path / "reports" / as_of / "daily_loop_summary.json").read_text(encoding="utf-8")
    )
    assert payload["execution"]["status"] == "no_target_generated"
    assert payload["execution"]["paper_report_written"] is False
    assert payload["execution"]["paper_book_appended"] is False


def test_daily_loop_source_has_no_same_call_strict_execution_path() -> None:
    source = __import__("inspect").getsource(daily_loop.run_once)
    assert "simulate_ashare_target_weights" not in source
    assert "write_paper_report" not in source
    assert "_market_for_dates" not in source
