from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from quantagent.data.providers.base import ProviderUnavailable


@dataclass(frozen=True)
class SchemaExtractionConfig:
    enabled: bool = False
    allow_network: bool = False
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_input_chars: int = 16000


class OpenAICompatibleSchemaExtractor:
    """Remote schema extraction client; disabled by default and never emits orders."""

    def __init__(self, config: SchemaExtractionConfig | dict[str, Any] | None = None) -> None:
        self.config = _coerce_config(config)

    def extract_json(self, *, system_prompt: str, user_text: str) -> dict[str, Any]:
        if not self.config.enabled:
            raise ProviderUnavailable("remote schema extraction is disabled")
        if not self.config.allow_network:
            raise ProviderUnavailable("remote schema extraction requires allow_network=true")
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ProviderUnavailable(f"{self.config.api_key_env} is required for remote schema extraction")
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text[: self.config.max_input_chars]},
            ],
        }
        request = Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
        except URLError as exc:  # pragma: no cover - network disabled in unit tests
            raise ProviderUnavailable("remote schema extraction request failed") from exc
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}


def _coerce_config(config: SchemaExtractionConfig | dict[str, Any] | None) -> SchemaExtractionConfig:
    if config is None:
        return SchemaExtractionConfig()
    if isinstance(config, SchemaExtractionConfig):
        return config
    return SchemaExtractionConfig(
        enabled=bool(config.get("enabled", False)),
        allow_network=bool(config.get("allow_network", False)),
        endpoint=str(config.get("endpoint", "https://api.openai.com/v1/chat/completions")),
        model=str(config.get("model", "gpt-4.1-mini")),
        api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        max_input_chars=int(config.get("max_input_chars", 16000)),
    )
