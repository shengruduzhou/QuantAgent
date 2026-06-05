"""Tests for ``LLMSkillClient`` response handling.

The Gemma/Gemini "thinking" models (e.g. ``gemma-4-26b-a4b-it``) split the
response into chain-of-thought parts flagged ``"thought": true`` and the final
answer parts. A naive concatenation of every part lets a *template* object
written inside the model's scratchpad (``{"tickers":[...]}``) win JSON
extraction over the real answer — which previously produced garbage such as
``{"tickers": [Ellipsis]}``. These tests pin the correct behaviour.
"""

from __future__ import annotations

from quantagent.agents.llm_skill_client import (
    LLMSkillClient,
    LLMSkillConfig,
    _parse_json_object,
)


def _gemini_client() -> LLMSkillClient:
    return LLMSkillClient(LLMSkillConfig(provider="gemini", enabled=True, allow_network=False))


def _thinking_response() -> dict:
    """A response shaped exactly like the live gemma-4 thinking model output."""
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "Plan: return one JSON object.\n"
                                '```json\n{"tickers":[...],"rationale":"..."}\n```\n'
                            ),
                            "thought": True,
                        },
                        {"text": "```", "thought": True},
                        {
                            "text": (
                                '{"tickers": ["600519.SH", "600900.SH"], '
                                '"rationale": "blue-chip leaders"}'
                            )
                        },
                    ],
                }
            }
        ]
    }


def test_extract_text_drops_thought_parts():
    client = _gemini_client()
    text = client._extract_text(_thinking_response())
    # Only the final (non-thought) answer part should survive.
    assert "blue-chip leaders" in text
    assert "Plan: return one JSON object" not in text
    assert "[...]" not in text


def test_thinking_response_parses_to_real_answer_not_template():
    client = _gemini_client()
    parsed = _parse_json_object(client._extract_text(_thinking_response()))
    # The regression: previously this returned {"tickers": [Ellipsis], ...}
    assert parsed["tickers"] == ["600519.SH", "600900.SH"]
    assert parsed["rationale"] == "blue-chip leaders"
    assert Ellipsis not in parsed["tickers"]


def test_extract_text_falls_back_when_only_thoughts_present():
    client = _gemini_client()
    response = {
        "candidates": [
            {"content": {"parts": [{"text": '{"a": 1}', "thought": True}]}}
        ]
    }
    # If a model emits thoughts only, we still surface them rather than "{}".
    assert client._extract_text(response) == '{"a": 1}'
