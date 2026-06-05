from __future__ import annotations

from dataclasses import dataclass, replace
import ast
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMSkillConfig:
    provider: str = "disabled"
    enabled: bool = False
    allow_network: bool = False
    endpoint: str = "https://api.openai.com/v1/responses"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    # Thinking models (gemma-4, gemini-2.5/3) spend significant latency on
    # chain-of-thought before emitting the answer, so the default must be
    # generous; override with QUANTAGENT_LLM_TIMEOUT_SECONDS.
    timeout_seconds: float = 90.0
    max_input_chars: int = 16000
    temperature: float = 0.0
    response_format: str = "json_object"

    @classmethod
    def from_env(cls, prefix: str = "QUANTAGENT_LLM_") -> "LLMSkillConfig":
        """Build a config from environment variables without reading secrets."""
        _load_dotenv_once()
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
            timeout_seconds=float(os.getenv(f"{prefix}TIMEOUT_SECONDS", "90")),
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
        if not api_key and self.config.provider == "gemini":
            api_key = os.getenv("google_API_KEY") or os.getenv("GEMINI_API_KEY")
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
        except (TimeoutError, OSError) as exc:
            return LLMSkillResult(skill_name, fallback or {}, "", True, f"network_error:{exc}")
        except URLError as exc:
            return LLMSkillResult(skill_name, fallback or {}, "", True, f"network_error:{exc}")
        try:
            data = json.loads(raw)
            content = self._extract_text(data)
            parsed = _parse_json_object(content)
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
        raise ValueError(f"unsupported LLM provider: {self.config.provider}")

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.config.provider == "openai-compatible":
            return data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if self.config.provider == "gemini":
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            # Thinking models (gemma-4, gemini-2.5/3, ...) split the response
            # into chain-of-thought parts flagged ``"thought": true`` and the
            # final answer parts. Concatenating the thoughts pollutes JSON
            # extraction (e.g. a template ``{"tickers":[...]}`` in the draft
            # would be parsed before the real answer). Keep only answer parts,
            # falling back to all parts if the model emitted thoughts only.
            answer_parts = [p for p in parts if isinstance(p, dict) and not p.get("thought")]
            selected = answer_parts or [p for p in parts if isinstance(p, dict)]
            return "".join(str(part.get("text", "")) for part in selected) or "{}"
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
        return _normalize_google_model_name(self.config.model)


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


_DOTENV_LOADED = False


def _load_dotenv_once(path: str | Path = ".env") -> None:
    """Load local .env without printing or persisting secret values.

    Existing process environment variables win over .env values. This lets
    production/systemd override local developer settings while still making
    CLI smoke runs work from the repo root.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_endpoint(provider: str) -> str:
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    if provider == "openai-compatible":
        return "https://api.openai.com/v1/chat/completions"
    return "https://api.openai.com/v1/responses"


def _default_model(provider: str) -> str:
    if provider == "gemini":
        return "gemini-2.5-flash"
    return "gpt-4.1-mini"


def _default_api_key_env(provider: str) -> str:
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    return "OPENAI_API_KEY"


def _normalize_google_model_name(model: str) -> str:
    """Normalize user-facing Google/Gemma names to API model IDs.

    Google model IDs are lowercase and, for instruction-tuned Gemma chat
    models, usually include the ``-it`` suffix.  This keeps the CLI tolerant
    of operator input such as ``gemma-4-26B-A4B`` while still resolving to the
    actual ``models/{id}:generateContent`` endpoint.
    """
    name = str(model or "").strip().strip('"').strip("'").removeprefix("models/")
    name = name.lower()
    if name.startswith("gemma-") and any(ch.isdigit() for ch in name) and not name.endswith("-it"):
        name = f"{name}-it"
    return name


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from strict or lightly wrapped model output."""
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    extracted = _extract_first_json_object(content)
    if extracted is None:
        raise json.JSONDecodeError("no JSON object found", content, 0)
    parsed = _loads_lenient_object(extracted)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("JSON value is not an object", extracted, 0)
    return parsed


def _loads_lenient_object(text: str) -> dict[str, Any]:
    """Parse strict JSON, Python-literal JSON, or YAML-like object text."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    try:
        import yaml

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return json.loads(text)


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring outside strings."""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: idx + 1]
        start = text.find("{", start + 1)
    return None
