from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
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

    @classmethod
    def from_env(cls, prefix: str = "QUANTAGENT_LLM_") -> "LLMSkillConfig":
        """Build a config from environment variables without reading secrets."""
        provider = os.getenv(f"{prefix}PROVIDER", "disabled")
        endpoint = os.getenv(f"{prefix}ENDPOINT") or _default_endpoint(provider)
        api_key_env = os.getenv(f"{prefix}API_KEY_ENV") or _default_api_key_env(provider)
        return cls(
            provider=provider,
            enabled=_env_bool(os.getenv(f"{prefix}ENABLED"), default=provider != "disabled"),
            allow_network=_env_bool(os.getenv(f"{prefix}ALLOW_NETWORK"), default=False),
            endpoint=endpoint,
            model=os.getenv(f"{prefix}MODEL", _default_model(provider)),
            api_key_env=api_key_env,
            timeout_seconds=float(os.getenv(f"{prefix}TIMEOUT_SECONDS", "30")),
            max_input_chars=int(os.getenv(f"{prefix}MAX_INPUT_CHARS", "16000")),
            temperature=float(os.getenv(f"{prefix}TEMPERATURE", "0")),
            response_format=os.getenv(f"{prefix}RESPONSE_FORMAT", "json_object"),
        )


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
        if self.config.provider not in {"disabled", "openai", "openai-compatible", "gemini"}:
            return LLMSkillResult(skill_name, fallback or {}, "", True, f"unsupported_provider:{self.config.provider}")
        if self.config.provider == "disabled" or not self.config.enabled:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "skill_disabled")
        if not self.config.allow_network:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "network_blocked")
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return LLMSkillResult(skill_name, fallback or {}, "", True, "api_key_missing")
        payload = self._request_payload(system_prompt, user_text)
        request = Request(
            self._resolved_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(api_key),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            return LLMSkillResult(skill_name, fallback or {}, body, True, f"http_error:{exc.code}")
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
        if self.config.provider == "gemini":
            return {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"{system_prompt}\n\n"
                                    "Return exactly one JSON object. Do not emit orders or trading instructions.\n\n"
                                    f"{text}"
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "responseMimeType": "application/json"
                    if self.config.response_format == "json_object"
                    else "text/plain",
                },
            }
        raise ProviderUnavailable(f"unsupported LLM provider: {self.config.provider}")

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.config.provider == "openai-compatible":
            return data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if self.config.provider == "gemini":
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)) or "{}"
        if isinstance(data.get("output_text"), str):
            return str(data["output_text"])
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    return str(content["text"])
        return "{}"

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.provider == "gemini":
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolved_endpoint(self) -> str:
        if self.config.provider != "gemini":
            return self.config.endpoint
        endpoint = self.config.endpoint or _default_endpoint("gemini")
        if endpoint in {
            "https://api.openai.com/v1/responses",
            "https://api.openai.com/v1/chat/completions",
        }:
            endpoint = _default_endpoint("gemini")
        model_name = self._resolved_gemini_model()
        if "{model}" in endpoint:
            return endpoint.format(model=model_name)
        if endpoint.endswith(":generateContent"):
            return endpoint
        return f"{endpoint.rstrip('/')}/models/{model_name}:generateContent"

    def _resolved_gemini_model(self) -> str:
        if self.config.model == "gpt-4.1-mini":
            return _default_model("gemini")
        return self.config.model.removeprefix("models/")


def _coerce_config(config: LLMSkillConfig | dict[str, Any] | None) -> LLMSkillConfig:
    if config is None:
        return LLMSkillConfig()
    if isinstance(config, LLMSkillConfig):
        return config
    provider = str(config.get("provider", "disabled"))
    return LLMSkillConfig(
        provider=provider,
        enabled=bool(config.get("enabled", False)),
        allow_network=bool(config.get("allow_network", False)),
        endpoint=str(config.get("endpoint") or _default_endpoint(provider)),
        model=str(config.get("model") or _default_model(provider)),
        api_key_env=str(config.get("api_key_env") or _default_api_key_env(provider)),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        max_input_chars=int(config.get("max_input_chars", 16000)),
        temperature=float(config.get("temperature", 0.0)),
        response_format=str(config.get("response_format", "json_object")),
    )


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
