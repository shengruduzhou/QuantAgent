"""LLM factor proposer — the generative half of the RD-Agent-style factor loop.

RD-Agent's power is a *closed loop*: an LLM proposes genuinely new factor
formulations each round, conditioned on the accumulated trace of what was
accepted or rejected and why. This module ports that idea into QuantAgent
while keeping the one production-critical constraint the original RD-Agent
does **not** have: proposals must stay inside the audited, point-in-time-safe
expression DSL (``quantagent.factors.expr``). The LLM is a *researcher* that
emits DSL formulas; it never writes free-form Python that touches real data,
and it never emits orders.

The orchestration loop lives in :func:`factor_synthesis.synthesize_factors_rd_agent`.
This module only handles LLM I/O: build a prompt from the current hypothesis +
escalating research directive + persisted accept/reject memory, call the model
(with the same backoff/fallback chain proven in
``scripts/llm_formula_alpha_candidates.py``), parse the JSON response, and turn
each formula string into a parsed, deduplicated :class:`ProposedFactor`.

The DSL node catalogue and A-share structure hints here are the single source
of truth; the standalone ``llm_formula_alpha_candidates`` script imports them.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Sequence

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.factors import expr as E
from quantagent.factors.factor_loop_memory import (
    ALLOWED_NODES,
    A_SHARE_STRUCTURES,
    FALLBACK_MODELS,
    classify_structure,
)
from quantagent.factors.factor_synthesis import (
    LLMProposalResult,
    ProposedFactor,
    RDAgentFactorHypothesis,
    _node_count,
    parse_expression,
)


# --------------------------------------------------------------------------- #
# Proposer                                                                    #
# --------------------------------------------------------------------------- #


def _sanitize_name(raw: str, ordinal: int) -> str:
    base = str(raw or f"llm_factor_{ordinal:03d}")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base.lower()).strip("_")
    cleaned = cleaned or f"llm_factor_{ordinal:03d}"
    return cleaned if cleaned.startswith("llm_") else f"llm_{cleaned}"


def _complexity_tier(expr: E.Expr) -> int:
    nodes = _node_count(expr)
    if nodes <= 4:
        return 1
    if nodes <= 8:
        return 2
    return 3


class LLMFactorProposer:
    """Proposes new DSL factor tasks from the loop's accumulated trace.

    Satisfies the ``FactorProposer`` protocol consumed by
    :func:`factor_synthesis.synthesize_factors_rd_agent`. Network access is
    opt-in via the injected :class:`LLMSkillConfig` (``allow_network``); when
    the model is unavailable the proposer returns an empty batch with
    ``used_fallback=True`` and the loop falls back to its blueprint slice.
    """

    def __init__(
        self,
        *,
        config: LLMSkillConfig | None = None,
        model: str | None = None,
        allow_network: bool | None = None,
        timeout_seconds: float = 360.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 12.0,
        fallback_models: Sequence[str] = FALLBACK_MODELS,
        max_nodes: int = 12,
    ) -> None:
        base = config or LLMSkillConfig.from_env()
        overrides: dict[str, Any] = {"timeout_seconds": max(base.timeout_seconds, float(timeout_seconds))}
        if allow_network is not None:
            overrides["allow_network"] = bool(allow_network)
        if model:
            overrides["model"] = model
        self.config = replace(base, **overrides)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.fallback_models = tuple(fallback_models)
        self.max_nodes = int(max_nodes)

    # -- prompt -------------------------------------------------------------- #

    def _build_prompt(
        self,
        *,
        hypothesis: RDAgentFactorHypothesis,
        rag_directive: str,
        memory_digest_payload: dict[str, Any],
        n_candidates: int,
        seen_expr_reprs: Sequence[str],
    ) -> tuple[str, str]:
        system = (
            "You are a quantitative researcher running an RD-Agent-style factor R&D loop for "
            "China A-shares. You propose cross-sectional stock-ranking formulas in a restricted, "
            "point-in-time-safe DSL. Return exactly one JSON object and nothing else. "
            "Never emit orders, trades, or financial advice. Use only the DSL nodes the user lists."
        )
        user = {
            "goal": (
                "Propose cross-sectional A-share ranking formulas with stable 5-day forward "
                "rank-IC out of sample. Each formula encodes ONE clear economic hypothesis and "
                "must be novel versus the factors already accepted into the SOTA library."
            ),
            "current_hypothesis": {
                "hypothesis": hypothesis.hypothesis,
                "reason": hypothesis.reason,
                "concise_knowledge": hypothesis.concise_knowledge,
            },
            "research_directive": rag_directive,
            "hypothesis_specification": (
                "Refine or replace the current hypothesis based on the feedback, then propose "
                "factors that directly test it. State the refined hypothesis in the 'hypothesis' "
                "field and one factor per economic idea in 'candidates'."
            ),
            "required_output_schema": {
                "hypothesis": {
                    "hypothesis": "one-sentence refined research hypothesis for this round",
                    "reason": "why this direction, given the feedback",
                    "concise_knowledge": "transferable lesson to carry forward",
                },
                "candidates": [
                    {
                        "name": "short_snake_case_name",
                        "expression": "Rank(Returns(Column('close'), 5))",
                        "description": "one line on what it measures",
                        "formulation": "human-readable formula, e.g. Rank(C_t / C_{t-5} - 1)",
                        "variables": {"C_t": "close at trade date t"},
                        "hypothesis": "one sentence of economic logic",
                        "horizon": "short_5d|mid_5d_30d|long_30d_120d",
                        "expected_direction": "positive|negative",
                    }
                ],
            },
            "allowed_expression_nodes": list(ALLOWED_NODES),
            "prefer_structures": list(A_SHARE_STRUCTURES),
            "constraints": [
                "Use only information available at or before trade_date (no future data, no labels).",
                f"Windows must be <= 120 trading days; keep each formula parseable and under ~{self.max_nodes} nodes.",
                "Every formula must have a clear economic meaning; no random operator soup.",
                "Avoid formulas equivalent to plain size/volatility/turnover ranks already in the library.",
                "Do NOT repeat any expression listed in already_attempted_expressions.",
                "Prioritise structures in 'uncovered_economic_structures'; AVOID the (structure, horizon) "
                "cells listed in 'crowded_but_failing' — they have been mined repeatedly without surviving.",
                f"Return exactly {n_candidates} candidates.",
            ],
            "feedback_from_previous_rounds": memory_digest_payload,
            "sota_coverage_map": memory_digest_payload.get("coverage_map") if isinstance(memory_digest_payload, dict) else None,
            "crowded_but_failing": (
                memory_digest_payload.get("coverage_map", {}).get("crowded_but_failing")
                if isinstance(memory_digest_payload, dict) else None
            ),
            "uncovered_economic_structures": (
                memory_digest_payload.get("uncovered_directions") if isinstance(memory_digest_payload, dict) else None
            ),
            "already_attempted_expressions": list(seen_expr_reprs)[-40:],
        }
        return system, json.dumps(user, ensure_ascii=False)

    # -- invocation ---------------------------------------------------------- #

    def _invoke(self, system: str, user_text: str):
        models = [self.config.model, *[m for m in self.fallback_models if m != self.config.model]]
        last = None
        for attempt, model in enumerate(models[: self.max_attempts]):
            cfg = replace(self.config, model=model)
            result = LLMSkillClient(cfg).invoke(
                "rd_agent_factor_proposer",
                system_prompt=system,
                user_text=user_text,
                fallback={"candidates": []},
            )
            last = result
            payload = result.output if isinstance(result.output, dict) else {}
            if not result.used_fallback and isinstance(payload.get("candidates"), list) and payload["candidates"]:
                return result
            if attempt + 1 < min(self.max_attempts, len(models)):
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
        return last

    # -- public API ---------------------------------------------------------- #

    def propose(
        self,
        *,
        round_idx: int,
        hypothesis: RDAgentFactorHypothesis,
        rag_directive: str,
        memory_digest_payload: dict[str, Any],
        n_candidates: int,
        seen_expr_reprs: Sequence[str],
    ) -> LLMProposalResult:
        system, user_text = self._build_prompt(
            hypothesis=hypothesis,
            rag_directive=rag_directive,
            memory_digest_payload=memory_digest_payload,
            n_candidates=n_candidates,
            seen_expr_reprs=seen_expr_reprs,
        )
        result = self._invoke(system, user_text)
        if result is None:
            return LLMProposalResult(hypothesis=hypothesis, factors=[], used_fallback=True, fallback_reason="no_result")

        payload = result.output if isinstance(result.output, dict) else {}
        refined = hypothesis
        raw_hyp = payload.get("hypothesis")
        if isinstance(raw_hyp, dict) and raw_hyp.get("hypothesis"):
            refined = RDAgentFactorHypothesis(
                hypothesis=str(raw_hyp.get("hypothesis") or hypothesis.hypothesis),
                reason=str(raw_hyp.get("reason") or hypothesis.reason),
                concise_observation=str(raw_hyp.get("concise_observation") or hypothesis.concise_observation),
                concise_justification=str(raw_hyp.get("concise_justification") or hypothesis.concise_justification),
                concise_knowledge=str(raw_hyp.get("concise_knowledge") or hypothesis.concise_knowledge),
            )

        factors = self._parse_factors(payload.get("candidates", []), seen_expr_reprs)
        return LLMProposalResult(
            hypothesis=refined,
            factors=factors,
            used_fallback=bool(result.used_fallback),
            fallback_reason=result.fallback_reason,
        )

    def _parse_factors(
        self,
        items: Any,
        seen_expr_reprs: Sequence[str],
    ) -> list[ProposedFactor]:
        if not isinstance(items, list):
            return []
        seen = set(seen_expr_reprs)
        factors: list[ProposedFactor] = []
        for ordinal, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            expr_text = str(item.get("expression") or "").strip()
            if not expr_text:
                continue
            try:
                expr = parse_expression(expr_text)
            except Exception:
                continue  # outside the DSL → silently dropped (loop logs the gap)
            key = repr(expr)
            if key in seen:
                continue
            if _node_count(expr) > self.max_nodes:
                continue
            seen.add(key)
            variables = item.get("variables")
            hypothesis_text = str(item.get("hypothesis") or "")
            description = str(item.get("description") or item.get("hypothesis") or "LLM-proposed factor")
            factors.append(
                ProposedFactor(
                    name=_sanitize_name(item.get("name"), ordinal),
                    expr=expr,
                    description=description,
                    formulation=str(item.get("formulation") or expr_text),
                    variables=variables if isinstance(variables, dict) else {},
                    hypothesis=hypothesis_text,
                    complexity_tier=_complexity_tier(expr),
                    horizon=str(item.get("horizon") or ""),
                    structure=classify_structure(f"{hypothesis_text} {description}"),
                )
            )
        return factors


__all__ = [
    "LLMFactorProposer",
]
