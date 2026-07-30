from __future__ import annotations

from services.quant_api.services.jobs import JobManager, JobRecord


def test_job_stream_reads_log_incrementally_and_finishes(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    job_id = "job_fixture"
    manager._jobs[job_id] = JobRecord(
        id=job_id,
        type="train",
        status="succeeded",
        commandId="train-v8-deep",
        createdAt="2026-01-01T00:00:00+00:00",
        startedAt="2026-01-01T00:00:01+00:00",
        finishedAt="2026-01-01T00:00:02+00:00",
    )
    log_path = quant_ui_settings.jobs_root / f"{job_id}.log"
    log_path.write_text("epoch 1\ncompleted", encoding="utf-8")

    events = list(manager.stream(job_id))

    assert any('"line": "epoch 1"' in event for event in events)
    assert any('"line": "completed"' in event for event in events)
    assert events[-1].startswith("event: status")


def test_purge_removes_trace_and_only_owned_runtime_outputs(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    owned = quant_ui_settings.runtime_root / "reports" / "owned_run"
    shared = quant_ui_settings.runtime_root / "data" / "shared.parquet"
    owned.mkdir(parents=True)
    (owned / "result.json").write_text("{}", encoding="utf-8")
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text("shared", encoding="utf-8")
    job_id = "job_purge_fixture"
    manager._jobs[job_id] = JobRecord(
        id=job_id,
        type="train",
        status="failed",
        commandId="train-v8-deep",
        createdAt="2026-01-01T00:00:00+00:00",
        outputPaths=["runtime/reports/owned_run", "runtime/data/shared.parquet"],
        ownedOutputPaths=["runtime/reports/owned_run"],
    )
    log_path = quant_ui_settings.jobs_root / f"{job_id}.log"
    log_path.write_text("failed", encoding="utf-8")

    result = manager.purge(job_id, delete_outputs=True)

    assert result["status"] == "purged"
    assert not owned.exists()
    assert shared.read_text(encoding="utf-8") == "shared"
    assert not log_path.exists()
    assert manager.get(job_id) is None
