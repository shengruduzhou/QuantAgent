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
