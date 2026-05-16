from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.data.providers.base import ProviderUnavailable


@dataclass(frozen=True)
class SchemaExtractionConfig:
    provider: str = "disabled"
    enabled: bool = False
    allow_network: bool = False
    endpoint: str = "https://api.openai.com/v1/responses"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_input_chars: int = 16000


class OpenAICompatibleSchemaExtractor:
    """Remote schema extraction client; disabled by default and never emits orders."""

    def __init__(self, config: SchemaExtractionConfig | dict[str, Any] | None = None) -> None:
        self.config = _coerce_config(config)

    def extract_json(self, *, system_prompt: str, user_text: str) -> dict[str, Any]:
        result = LLMSkillClient(
            LLMSkillConfig(
                provider=self.config.provider,
                enabled=self.config.enabled,
                allow_network=self.config.allow_network,
                endpoint=self.config.endpoint,
                model=self.config.model,
                api_key_env=self.config.api_key_env,
                timeout_seconds=self.config.timeout_seconds,
                max_input_chars=self.config.max_input_chars,
            )
        ).invoke(
            "schema_extraction",
            system_prompt=system_prompt,
            user_text=user_text,
            fallback={},
        )
        if result.used_fallback:
            raise ProviderUnavailable(f"remote schema extraction unavailable: {result.fallback_reason}")
        return result.output


def _coerce_config(config: SchemaExtractionConfig | dict[str, Any] | None) -> SchemaExtractionConfig:
    if config is None:
        return SchemaExtractionConfig()
    if isinstance(config, SchemaExtractionConfig):
        return config
    return SchemaExtractionConfig(
        provider=str(config.get("provider", "disabled")),
        enabled=bool(config.get("enabled", False)),
        allow_network=bool(config.get("allow_network", False)),
        endpoint=str(config.get("endpoint", "https://api.openai.com/v1/responses")),
        model=str(config.get("model", "gpt-4.1-mini")),
        api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        max_input_chars=int(config.get("max_input_chars", 16000)),
    )
