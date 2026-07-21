from __future__ import annotations

import asyncio
from queue import Empty

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from services.quant_api.events.broker import SubscriptionClosed


router = APIRouter(prefix="/api/events")
ALLOWED_TOPICS = frozenset({"jobs"})
HEARTBEAT_SECONDS = 15.0


@router.websocket("/ws")
async def event_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    requested = {
        topic.strip()
        for topic in websocket.query_params.get("topics", "jobs").split(",")
        if topic.strip()
    }
    if not requested or not requested.issubset(ALLOWED_TOPICS):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unsupported event topic")
        return

    services = websocket.app.state.services
    try:
        subscription = services.events.subscribe(requested)
    except (RuntimeError, ValueError):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="event service unavailable")
        return

    snapshot = services.events.create_event(
        topic="jobs",
        event_type="system.snapshot",
        payload={"jobs": services.jobs.list()},
        source="quant_api.events",
    )
    disconnect_task = asyncio.create_task(websocket.receive())
    try:
        await websocket.send_json(snapshot.public())
        while True:
            event_task = asyncio.create_task(
                asyncio.to_thread(subscription.get, HEARTBEAT_SECONDS),
            )
            done, _ = await asyncio.wait(
                {event_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                event_task.cancel()
                message = disconnect_task.result()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", status.WS_1000_NORMAL_CLOSURE))
                disconnect_task = asyncio.create_task(websocket.receive())
                continue
            try:
                event = event_task.result()
            except Empty:
                event = services.events.create_event(
                    topic="system",
                    event_type="system.heartbeat",
                    payload=services.events.stats(),
                    source="quant_api.events",
                )
            dropped = subscription.take_dropped()
            if dropped:
                gap = services.events.create_event(
                    topic="system",
                    event_type="stream.gap",
                    payload={"droppedEvents": dropped, "recovery": "refresh_snapshot"},
                    source="quant_api.events",
                )
                await websocket.send_json(gap.public())
            await websocket.send_json(event.public())
    except (SubscriptionClosed, WebSocketDisconnect, RuntimeError):
        pass
    finally:
        disconnect_task.cancel()
        services.events.unsubscribe(subscription)


__all__ = ["router"]
