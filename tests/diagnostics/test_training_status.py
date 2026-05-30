from __future__ import annotations

import json


def _complete_fold(path, horizons=(5, 20, 60, 120, 126)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "ft_transformer_metrics.json").write_text("{}", encoding="utf-8")
    for horizon in horizons:
        (path / f"fold_{horizon:03d}d_oos_predictions.parquet").write_text("x", encoding="utf-8")
        (path / f"fold_{horizon:03d}d_strategy_metrics.json").write_text("{}", encoding="utf-8")


def test_v10_training_status_detects_missing_folds(tmp_path):
    from quantagent.diagnostics.training_status import V10StatusConfig, scan_v10_training_status

    base = tmp_path / "models" / "v10"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    seed_dir = tmp_path / "models" / "v10_seed4096" / "walk_forward"
    _complete_fold(seed_dir / "fold_000")
    _complete_fold(seed_dir / "fold_001")
    (log_dir / "v10_seed4096.log").write_text("running until interrupted", encoding="utf-8")

    status = scan_v10_training_status(
        V10StatusConfig(base_output=base, log_dir=log_dir, seeds=(4096,), expected_folds=4)
    )
    seed = status["seeds"]["4096"]

    assert seed["status"] == "interrupted_or_stopped"
    assert seed["completed_folds"] == ["fold_000", "fold_001"]
    assert seed["missing_folds"] == ["fold_002", "fold_003"]
    assert seed["resume_from"] == "fold_002"
    assert seed["aggregate_ready"] is False


def test_v10_training_status_completed_marker_wins(tmp_path):
    from quantagent.diagnostics.training_status import V10StatusConfig, scan_v10_training_status

    base = tmp_path / "models" / "v10"
    out = tmp_path / "models" / "v10_seed1729"
    out.mkdir(parents=True)
    (out / "_seed_completed.txt").write_text("done", encoding="utf-8")
    status = scan_v10_training_status(
        V10StatusConfig(base_output=base, log_dir=tmp_path / "logs", seeds=(1729,), expected_folds=1)
    )

    assert status["seeds"]["1729"]["status"] == "completed"
    assert status["seeds"]["1729"]["exit_reason"] == "completed_marker"


def test_write_training_status_outputs_json_and_markdown(tmp_path):
    from quantagent.diagnostics.training_status import write_training_status

    status = {"generated_at": "now", "active_training_process": False, "aggregate_ready": False, "expected_folds": 12, "seeds": {}}
    paths = write_training_status(status, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["aggregate_ready"] is False
