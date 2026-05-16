from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from quantagent.data.providers.base import ProviderUnavailable


@dataclass(frozen=True)
class LLMSkillConfig:
    provider: str = "disabled"
    enabled: bool = False
    allow_network: bool = False
    endpoint: str = "https://api.openai.com/v1/responses"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_input_chars: int = 16000
    temperature: float = 0.0
    response_format: str = "json_object"


@dataclass(frozen=True)
class LLMSkillResult:
    skill_name: str
    output: dict[str, Any]
    raw_text: str
    used_fallback: bool
    fallback_reason: str | None = None


class LLMSkillClient:
    """OpenAI-compatible client that invokes named skills with curated prompts.

    Never emits orders. Returns structured JSON only. Falls back to deterministic
    heuristics when the network is unavailable or the skill is disabled.
    """

    def __init__(self, config: LLMSkillConfig | dict[str, Any] | None = None) -> None:
        self.config = _coerce_config(config)

    def with_overrides(self, **overrides: Any) -> "LLMSkillClient":
        return LLMSkillClient(replace(self.config, **overrides))

    def invoke(
        self,
        skill_name: str,
        *,
        system_prompt: str,
        user_text: str,
        fallback: dict[str, Any] | None = None,
    ) -> LLMSkillResult:
        if self.config.provider == "disabled" or not self.config.enabled:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "skill_disabled")
        if not self.config.allow_network:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "network_blocked")
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "api_key_missing")
        payload = self._request_payload(system_prompt, user_text)
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
        except URLError as exc:
            return LLMSkillResult(skill_name, fallback or {}, "", True, f"network_error:{exc}")
        try:
            data = json.loads(raw)
            content = self._extract_text(data)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return LLMSkillResult(skill_name, fallback or {}, raw, True, "non_dict_response")
            return LLMSkillResult(skill_name, parsed, raw, False, None)
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            return LLMSkillResult(skill_name, fallback or {}, raw, True, f"parse_error:{exc}")

    def _request_payload(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        text = user_text[: self.config.max_input_chars]
        if self.config.provider == "openai":
            return {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "text": {"format": {"type": self.config.response_format}},
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": text}]},
                ],
            }
        if self.config.provider == "openai-compatible":
            return {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "response_format": {"type": self.config.response_format},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            }
        raise ProviderUnavailable(f"unsupported LLM provider: {self.config.provider}")

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.config.provider == "openai-compatible":
            return data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if isinstance(data.get("output_text"), str):
            return str(data["output_text"])
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    return str(content["text"])
        return "{}"


def _coerce_config(config: LLMSkillConfig | dict[str, Any] | None) -> LLMSkillConfig:
    if config is None:
        return LLMSkillConfig()
    if isinstance(config, LLMSkillConfig):
        return config
    return LLMSkillConfig(
        provider=str(config.get("provider", "disabled")),
        enabled=bool(config.get("enabled", False)),
        allow_network=bool(config.get("allow_network", False)),
        endpoint=str(config.get("endpoint", "https://api.openai.com/v1/responses")),
        model=str(config.get("model", "gpt-4.1-mini")),
        api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        max_input_chars=int(config.get("max_input_chars", 16000)),
        temperature=float(config.get("temperature", 0.0)),
        response_format=str(config.get("response_format", "json_object")),
    )
