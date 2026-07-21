from __future__ import annotations

from fastapi.testclient import TestClient

from services.quant_api.app import create_app
from services.quant_api.events import EVENT_SCHEMA_VERSION, EventBroker
from services.quant_api.services.jobs import JobRecord


def test_event_broker_emits_versioned_envelopes_and_reports_backpressure() -> None:
    broker = EventBroker(default_queue_size=2)
    broker.start()
    subscription = broker.subscribe({"jobs"})

    for index in range(3):
        broker.publish(
            topic=f"jobs:job_{index}",
            event_type="job.status",
            payload={"job": {"id": f"job_{index}"}},
            source="test",
            correlation_id=f"job_{index}",
        )

    assert subscription.take_dropped() == 1
    first = subscription.get(timeout=0.1)
    second = subscription.get(timeout=0.1)
    assert [first.sequence, second.sequence] == [2, 3]
    assert first.schema_version == EVENT_SCHEMA_VERSION
    assert first.public()["eventType"] == "job.status"
    assert first.public()["correlationId"] == "job_1"

    broker.unsubscribe(subscription)
    assert broker.stats()["subscribers"] == 0


def test_job_status_websocket_reconnects_with_snapshot_and_live_update(
    quant_ui_settings,
) -> None:
    app = create_app(quant_ui_settings)
    with TestClient(app) as client:
        with client.websocket_connect("/api/events/ws?topics=jobs") as websocket:
            first_snapshot = websocket.receive_json()
            assert first_snapshot["schemaVersion"] == EVENT_SCHEMA_VERSION
            assert first_snapshot["eventType"] == "system.snapshot"
            assert first_snapshot["payload"] == {"jobs": []}

        with client.websocket_connect("/api/events/ws?topics=jobs") as websocket:
            second_snapshot = websocket.receive_json()
            assert second_snapshot["eventType"] == "system.snapshot"

            job_id = "job_reconnect_fixture"
            app.state.services.jobs._jobs[job_id] = JobRecord(
                id=job_id,
                type="train",
                status="queued",
                commandId="train-v8-deep",
                createdAt="2026-01-01T00:00:00+00:00",
            )
            app.state.services.jobs._update(
                job_id,
                status="running",
                startedAt="2026-01-01T00:00:01+00:00",
                message="running",
            )

            event = websocket.receive_json()
            assert event["eventType"] == "job.status"
            assert event["topic"] == f"jobs:{job_id}"
            assert event["correlationId"] == job_id
            assert event["payload"]["job"]["status"] == "running"

    assert app.state.services.events.stats()["running"] is False
    assert app.state.services.events.stats()["subscribers"] == 0
