from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import quantagent.paper.daily_loop as daily_loop
from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS


def test_daily_loop_records_pending_signal_without_paper_pnl_or_book(tmp_path, monkeypatch) -> None:
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
    targets = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp(as_of)],
            "600000.SH": [1.0],
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

    paper_book = tmp_path / "paper_book.parquet"
    result = daily_loop.run_once(
        daily_loop.DailyPaperLoopConfig(
            as_of_date=as_of,
            model_dir=str(tmp_path / "model"),
            feature_dataset_path=str(feature_path),
            market_panel_path=str(market_path),
            output_root=str(tmp_path / "reports"),
            paper_book_path=str(paper_book),
            pending_signal_dir=str(tmp_path / "pending"),
            dry_run_evidence=True,
        )
    )

    assert result.status == "signal_recorded_pending_execution"
    assert result.execution_timing_semantics == EXECUTION_TIMING_SEMANTICS
    assert result.executed_fill_count == 0
    assert result.pending_signal_path
    assert not paper_book.exists()

    summary_path = tmp_path / "reports" / as_of / "daily_loop_summary.json"
    payload = pd.read_json(summary_path, typ="series")
    assert payload["status"] == "signal_recorded_pending_execution"
    execution = payload["execution"]
    assert execution["executed_fill_count"] == 0
    assert execution["paper_report_written"] is False
    assert execution["paper_book_appended"] is False
    assert payload["paper_report"] is None


def test_daily_loop_source_has_no_same_call_strict_execution_path() -> None:
    source = __import__("inspect").getsource(daily_loop.run_once)
    assert "simulate_ashare_target_weights" not in source
    assert "write_paper_report" not in source
    assert "_market_for_dates" not in source
