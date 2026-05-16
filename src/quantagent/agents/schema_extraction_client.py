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
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str | None = None
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
                endpoint=self.config.endpoint or _default_endpoint(self.config.provider),
                model=self.config.model or _default_model(self.config.provider),
                api_key_env=self.config.api_key_env or _default_api_key_env(self.config.provider),
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
        endpoint=str(config["endpoint"]) if config.get("endpoint") else None,
        model=str(config["model"]) if config.get("model") else None,
        api_key_env=str(config["api_key_env"]) if config.get("api_key_env") else None,
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        max_input_chars=int(config.get("max_input_chars", 16000)),
    )


def _default_endpoint(provider: str) -> str:
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    if provider == "openai-compatible":
        return "https://api.openai.com/v1/chat/completions"
    return "https://api.openai.com/v1/responses"


def _default_model(provider: str) -> str:
    if provider == "gemini":
        return "gemini-1.5-flash"
    return "gpt-4.1-mini"


def _default_api_key_env(provider: str) -> str:
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    return "OPENAI_API_KEY"
