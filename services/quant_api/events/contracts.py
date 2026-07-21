from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from services.quant_api.schemas.models import ApiModel


EVENT_SCHEMA_VERSION = "quantagent.event.v1"


class EventEnvelope(ApiModel):
    """Versioned event contract shared by API services and WebSocket clients."""

    schema_version: Literal["quantagent.event.v1"] = Field(
        EVENT_SCHEMA_VERSION,
        alias="schemaVersion",
    )
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}", alias="eventId")
    event_type: str = Field(alias="eventType")
    topic: str
    occurred_at: str = Field(alias="occurredAt")
    source: str
    sequence: int = Field(ge=1)
    correlation_id: str | None = Field(None, alias="correlationId")
    payload: dict[str, Any] = Field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


__all__ = ["EVENT_SCHEMA_VERSION", "EventEnvelope"]
