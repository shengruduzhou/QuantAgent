"""Genetic-programming symbolic regression for factor discovery.

Evolves expression trees in the :mod:`quantagent.factors.expr` DSL to
maximise rank-IC against forward-return labels. Output: top-K
discovered :class:`~quantagent.factors.expr.FactorDefinition`s that
can be registered alongside Alpha101.

Unlike :mod:`quantagent.optimization.factor_evolution`, which evolves
binary masks over **existing** columns, this module discovers **new
formulas** the human authors did not write.

Operators are restricted to the DSL primitives in
:mod:`quantagent.factors.expr` so the no-look-ahead and per-symbol
time-series guarantees are preserved automatically.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from quantagent.factors import expr as E
from quantagent.factors.factor_loop_memory import (
    RAG_EASY,
    RAG_HIGH_IC,
    append_memory,
    classify_horizon,
    classify_structure,
    coverage_map,
    load_memory,
    memory_digest,
    uncovered_directions,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Operator and terminal pools                                                 #
# --------------------------------------------------------------------------- #

_BINARY_OPS: tuple = (E.Add, E.Sub, E.Mul, E.Div)
_UNARY_OPS: tuple = (E.Abs, E.Sign, E.Log, E.CsZscore)
# Time-series operators are listed as (class_or_factory, takes_window: bool).
_TS_OPS: tuple = (E.TsMean, E.TsStd, E.TsSum, E.TsMax, E.TsMin, E.TsRank, E.DecayLinear)
_TS_BINARY_OPS: tuple = (E.TsCorr, E.TsCov)
_DELAY_DELTA_OPS: tuple = (E.Delay, E.Delta)
_TS_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 30, 60)
_DELAY_PERIODS: tuple[int, ...] = (1, 2, 3, 5, 10)

_PRICE_TERMINALS: tuple = (E.Open, E.High, E.Low, E.Close, E.Vwap)
_VOLUME_TERMINALS: tuple = (E.Volume, E.Amount)
_CONSTANTS: tuple = tuple(E.Constant(c) for c in (0.5, 1.0, 2.0, 5.0))

_MARKET_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SymbolicGAConfig:
    population: int = 80
    generations: int = 20
    max_depth: int = 4
    min_depth: int = 2
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.25
    elitism: int = 4
    top_k: int = 20
    label_column: str = "forward_return_5d"
    complexity_penalty: float = 5e-4
    min_finite_ratio: float = 0.3
    max_correlation: float = 0.85
    validation_fraction: float = 0.25
    min_validation_rank_ic: float = 0.0
    fitness_sample_dates: int = 400
    fitness_sample_symbols: int = 500
    seed: int = 1729
    # Fraction of the initial population seeded from Alpha101-style templates
    # (and mutated variants of them) instead of pure random trees.
    warm_start_fraction: float = 0.4
    # Weight of |daily ICIR| in fitness, alongside |mean rank-IC|.
    icir_weight: float = 0.05
    # Existing factor columns (must be present in the merged panel) that
    # candidates are decorrelated against at selection time.
    reference_columns: tuple[str, ...] = ()
    max_reference_correlation: float = 0.7
    # Fraction of each new generation replaced by fresh random trees to
    # keep diversity (quality-diversity style anti-premature-convergence).
    random_injection_rate: float = 0.10
    # Tradability guard: when these flag columns are present in the panel, the
    # validation/fitness IC drops names that are untradable at signal time so a
    # factor cannot be accepted on ranking power over positions you could never
    # take (the phantom-edge mechanism from honest-baseline-truth).
    tradability_columns: tuple[str, ...] = ("is_suspended", "is_limit_up", "is_limit_down")
    exclude_st: bool = False


@dataclass(frozen=True)
class SynthesisResult:
    definitions: list[E.FactorDefinition]
    leaderboard: pd.DataFrame  # name, expression, train/validation_rank_ic, icir, complexity, finite_ratio,
                               # + OOS economic profile: oos_top_quantile_return, oos_long_short_return,
                               #   oos_monotonicity, oos_top_quantile_turnover (RD-Agent path only)
    history: pd.DataFrame      # generation, best_fitness, mean_fitness, mean_complexity


@dataclass(frozen=True)
class RDAgentFactorHypothesis:
    """RD-Agent-style hypothesis record for one factor research loop."""

    hypothesis: str
    reason: str
    concise_observation: str = ""
    concise_justification: str = ""
    concise_knowledge: str = ""


@dataclass(frozen=True)
class RDAgentFactorTask:
    """RD-Agent-style factor task.

    RD-Agent asks a coder to implement these tasks as ``factor.py``. QuantAgent
    keeps the same task contract but maps the implementation to the safe DSL.
    """

    factor_name: str
    factor_description: str
    factor_formulation: str
    variables: dict[str, str]
    factor_implementation: bool = False

    def get_task_information(self) -> str:
        return (
            f"factor_name: {self.factor_name}\n"
            f"factor_description: {self.factor_description}\n"
            f"factor_formulation: {self.factor_formulation}\n"
            f"variables: {self.variables}"
        )

    def get_task_information_and_implementation_result(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": self.variables,
            "factor_implementation": self.factor_implementation,
        }


@dataclass(frozen=True)
class RDAgentFactorLoopConfig:
    """Configuration for the RD-Agent-style factor discovery loop.

    The loop mirrors RD-Agent's finance factor workflow: propose a hypothesis,
    create one to five factor tasks, implement them in a constrained interface,
    evaluate output/value quality, deduplicate against the SOTA factor library,
    and feed the result into the next loop.
    """

    rounds: int = 4
    factors_per_round: int = 3
    top_k: int = 20
    label_column: str = "forward_return_5d"
    validation_fraction: float = 0.25
    min_validation_rank_ic: float = 0.0
    min_finite_ratio: float = 0.3
    max_sota_correlation: float = 0.99
    complexity_penalty: float = 5e-4
    icir_weight: float = 0.05
    fitness_sample_dates: int = 400
    fitness_sample_symbols: int = 500
    seed: int = 1729
    reference_columns: tuple[str, ...] = ()
    max_reference_correlation: float = 0.7
    # Tradability guard (see SymbolicGAConfig): validation IC / fitness drop
    # names untradable at signal time so the acceptance gate cannot be fooled
    # by phantom edge over limit-up-sealed / limit-down / suspended / ST names.
    tradability_columns: tuple[str, ...] = ("is_suspended", "is_limit_up", "is_limit_down")
    exclude_st: bool = False
    # Stability floor: reject factors whose tradable validation ICIR is below
    # this even if the mean rank-IC clears ``min_validation_rank_ic``.
    min_validation_icir: float = 0.0
    # --- LLM proposal (the generative half of the closed loop) ------------- #
    # When ``use_llm`` is set, rounds at/after ``llm_start_round`` ask an LLM
    # to propose NEW DSL factor tasks conditioned on the accumulated trace and
    # the persistent accept/reject memory, instead of replaying the fixed
    # blueprint slice. Round 0 always uses blueprints as a warm start. If the
    # model is unavailable the round transparently falls back to blueprints.
    use_llm: bool = False
    llm_start_round: int = 1
    llm_candidates_per_round: int = 4
    # After this many rounds the research directive escalates from "easy,
    # attributable factors" to "richer, higher-IC interaction structures".
    rag_escalation_round: int = 3
    # Persistent JSONL of accept/reject knowledge digested into each LLM prompt.
    memory_path: str | None = None
    allow_network: bool = False
    llm_model: str | None = None
    llm_timeout_seconds: float = 360.0
    max_llm_attempts: int = 3


@dataclass(frozen=True)
class RDAgentSynthesisResult(SynthesisResult):
    """RD-Agent-style synthesis output with loop trace and task feedback."""

    trace: pd.DataFrame
    task_feedback: pd.DataFrame


@dataclass(frozen=True)
class _RDAgentCandidate:
    task: RDAgentFactorTask
    expr: E.Expr
    complexity_tier: int
    structure: str = ""
    horizon: str = ""


@dataclass(frozen=True)
class ProposedFactor:
    """An LLM-proposed factor, already parsed into the safe expression DSL.

    Produced by a :class:`FactorProposer`; the loop wraps it into an
    ``_RDAgentCandidate`` and runs it through the same value/IC/novelty gates
    that the hand-written blueprints face.
    """

    name: str
    expr: E.Expr
    description: str = ""
    formulation: str = ""
    variables: dict[str, str] = field(default_factory=dict)
    hypothesis: str = ""
    complexity_tier: int = 1
    horizon: str = ""
    structure: str = ""


@dataclass(frozen=True)
class LLMProposalResult:
    """Output of one LLM proposal round: a refined hypothesis plus DSL factors."""

    hypothesis: "RDAgentFactorHypothesis"
    factors: list[ProposedFactor]
    used_fallback: bool = False
    fallback_reason: str | None = None


class FactorProposer(Protocol):
    """Contract for the generative half of the RD-Agent-style factor loop.

    Implemented by ``llm_factor_proposer.LLMFactorProposer``; tests inject a
    deterministic fake so the closed loop runs fully offline.
    """

    def propose(
        self,
        *,
        round_idx: int,
        hypothesis: "RDAgentFactorHypothesis",
        rag_directive: str,
        memory_digest_payload: dict[str, object],
        n_candidates: int,
        seen_expr_reprs: Sequence[str],
    ) -> LLMProposalResult:
        ...


# --------------------------------------------------------------------------- #
# Tree generation                                                             #
# --------------------------------------------------------------------------- #


def _random_terminal(rng: random.Random) -> E.Expr:
    pool: list = list(_PRICE_TERMINALS) + list(_VOLUME_TERMINALS)
    if rng.random() < 0.15:
        return rng.choice(_CONSTANTS)
    return rng.choice(pool)


def _random_tree(rng: random.Random, depth: int, *, force_internal: bool = False) -> E.Expr:
    """Sample a random expression tree of bounded depth."""
    if depth <= 0:
        return _random_terminal(rng)
    if not force_internal and rng.random() < 0.18:
        return _random_terminal(rng)

    roll = rng.random()
    if roll < 0.28:
        op = rng.choice(_BINARY_OPS)
        left = _random_tree(rng, depth - 1)
        right = _random_tree(rng, depth - 1)
        return op(left, right)
    if roll < 0.38:
        op = rng.choice(_UNARY_OPS)
        return op(_random_tree(rng, depth - 1))
    if roll < 0.68:
        op = rng.choice(_TS_OPS)
        window = rng.choice(_TS_WINDOWS)
        return op(_random_tree(rng, depth - 1), window)
    if roll < 0.78:
        op = rng.choice(_TS_BINARY_OPS)
        window = rng.choice(_TS_WINDOWS)
        return op(_random_tree(rng, depth - 1), _random_tree(rng, depth - 1), window)
    if roll < 0.90:
        op = rng.choice(_DELAY_DELTA_OPS)
        period = rng.choice(_DELAY_PERIODS)
        return op(_random_tree(rng, depth - 1), period)
    return E.Rank(_random_tree(rng, depth - 1))


def _warm_start_templates() -> list[E.Expr]:
    """Alpha101/191-style structural seeds known to carry signal in A-shares.

    These encode reversal, volume-price divergence, decayed momentum,
    liquidity, intraday positioning and volatility structures so the GA
    starts from financially meaningful regions of formula space instead
    of pure noise.
    """
    O, H, L, C, V, A = E.Open, E.High, E.Low, E.Close, E.Volume, E.Amount
    vwap = E.Vwap
    ret1 = E.Returns(C, 1)
    ret5 = E.Returns(C, 5)
    return [
        # Short-term reversal family.
        E.Mul(E.Constant(-1.0), E.Rank(ret5)),
        E.Mul(E.Constant(-1.0), E.Rank(E.Delta(C, 3))),
        E.Mul(E.Constant(-1.0), E.Rank(E.TsMean(ret1, 5))),
        # Volume-price divergence / correlation (alpha101 staples).
        E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(C), E.Rank(V), 10)),
        E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(O), E.Rank(V), 10)),
        E.Sub(E.Rank(E.Delta(C, 5)), E.Rank(E.Delta(V, 5))),
        E.TsCorr(C, V, 20),
        # Decayed momentum / smoothed reversal.
        E.Mul(E.Constant(-1.0), E.DecayLinear(E.Delta(C, 3), 10)),
        E.DecayLinear(E.Rank(ret5), 10),
        E.Rank(E.Sub(C, E.TsMean(C, 20))),
        # Intraday positioning.
        E.Div(E.Sub(C, L), E.Sub(H, L)),
        E.Div(E.Sub(C, O), E.Sub(H, L)),
        E.Rank(E.Div(E.Sub(C, vwap), vwap)),
        E.Mul(E.Constant(-1.0), E.Rank(E.Div(E.Sub(H, C), E.Sub(H, L)))),
        # Volatility / range structure.
        E.Mul(E.Constant(-1.0), E.Rank(E.TsStd(ret1, 20))),
        E.Rank(E.TsMean(E.Div(E.Sub(H, L), C), 5)),
        E.TsRank(E.Div(E.Sub(H, L), C), 10),
        # Liquidity / turnover discount.
        E.Mul(E.Constant(-1.0), E.Rank(E.Log(E.TsMean(A, 20)))),
        E.Mul(E.Constant(-1.0), E.Rank(E.Div(V, E.TsMean(V, 20)))),
        E.Div(E.TsStd(A, 20), E.TsMean(A, 20)),
        # Volume shock / crowding.
        E.Div(E.Sub(V, E.TsMean(V, 5)), E.TsStd(V, 5)),
        E.Mul(E.Constant(-1.0), E.Rank(E.TsRank(V, 5))),
        # Gap / overnight behaviour.
        E.Rank(E.Sub(O, E.Delay(C, 1))),
        E.Mul(E.Constant(-1.0), E.Rank(E.Sub(E.TsMax(H, 5), C))),
        # Price level vs anchor (52w-high style on shorter horizon).
        E.Div(C, E.TsMax(H, 60)),
        E.CsZscore(E.Div(C, E.TsMean(C, 60))),
    ]


def _rd_agent_factor_blueprints() -> list[_RDAgentCandidate]:
    """Hand the RD-Agent planner a broad but safe factor task space.

    RD-Agent's original Qlib loop lets the LLM propose formulas and then asks
    a coder to implement them. In QuantAgent, formulas must remain inside the
    PIT-safe expression DSL, so the planner emits explicit task metadata plus
    the corresponding DSL implementation.
    """
    O, H, L, C, V, A = E.Open, E.High, E.Low, E.Close, E.Volume, E.Amount
    ret1 = E.Returns(C, 1)
    ret5 = E.Returns(C, 5)
    hl_range = E.Sub(H, L)
    return [
        _candidate(
            "rd_momentum_5d",
            "[Momentum Factor] Five-day cross-sectional price momentum.",
            r"Rank(C_t / C_{t-5} - 1)",
            {"C_t": "close price at trade date t"},
            E.Rank(ret5),
            1,
        ),
        _candidate(
            "rd_reversal_5d",
            "[Reversal Factor] Negative five-day return rank.",
            r"-Rank(C_t / C_{t-5} - 1)",
            {"C_t": "close price at trade date t"},
            E.Mul(E.Constant(-1.0), E.Rank(ret5)),
            1,
        ),
        _candidate(
            "rd_reversal_1d",
            "[Reversal Factor] Negative one-day return rank.",
            r"-Rank(C_t / C_{t-1} - 1)",
            {"C_t": "close price at trade date t"},
            E.Mul(E.Constant(-1.0), E.Rank(ret1)),
            1,
        ),
        _candidate(
            "rd_volume_price_corr_10d",
            "[Volume-Price Factor] Negative rolling correlation between price rank and volume rank.",
            r"-Corr(Rank(C), Rank(V), 10)",
            {"C": "close price", "V": "trading volume"},
            E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(C), E.Rank(V), 10)),
            1,
        ),
        _candidate(
            "rd_open_volume_corr_10d",
            "[Volume-Price Factor] Negative rolling correlation between open rank and volume rank.",
            r"-Corr(Rank(O), Rank(V), 10)",
            {"O": "open price", "V": "trading volume"},
            E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(O), E.Rank(V), 10)),
            1,
        ),
        _candidate(
            "rd_volume_shock_5d",
            "[Liquidity Factor] Five-day volume surprise standardized by recent volume volatility.",
            r"(V_t - Mean(V, 5)) / Std(V, 5)",
            {"V_t": "trading volume at trade date t"},
            E.Div(E.Sub(V, E.TsMean(V, 5)), E.TsStd(V, 5)),
            1,
        ),
        _candidate(
            "rd_intraday_close_position",
            "[Intraday Position Factor] Close location within the daily high-low range.",
            r"(C_t - L_t) / (H_t - L_t)",
            {"H_t": "daily high", "L_t": "daily low", "C_t": "daily close"},
            E.Div(E.Sub(C, L), hl_range),
            1,
        ),
        _candidate(
            "rd_close_open_range",
            "[Intraday Position Factor] Close-open move normalized by daily range.",
            r"(C_t - O_t) / (H_t - L_t)",
            {"O_t": "daily open", "H_t": "daily high", "L_t": "daily low", "C_t": "daily close"},
            E.Div(E.Sub(C, O), hl_range),
            1,
        ),
        _candidate(
            "rd_close_vwap_gap",
            "[Price-Volume Factor] Close premium over VWAP.",
            r"Rank((C_t - VWAP_t) / VWAP_t)",
            {"VWAP_t": "amount divided by volume"},
            E.Rank(E.Div(E.Sub(C, E.Vwap), E.Vwap)),
            1,
        ),
        _candidate(
            "rd_range_volatility_5d",
            "[Volatility Factor] Mean high-low range over close.",
            r"Mean((H_t - L_t) / C_t, 5)",
            {"H_t": "daily high", "L_t": "daily low", "C_t": "daily close"},
            E.TsMean(E.Div(hl_range, C), 5),
            2,
        ),
        _candidate(
            "rd_low_volatility_20d",
            "[Volatility Factor] Negative rank of 20-day return volatility.",
            r"-Rank(Std(C_t / C_{t-1} - 1, 20))",
            {"C_t": "daily close"},
            E.Mul(E.Constant(-1.0), E.Rank(E.TsStd(ret1, 20))),
            2,
        ),
        _candidate(
            "rd_decayed_reversal_3d_10d",
            "[Reversal Factor] Decayed three-day price change reversal.",
            r"-DecayLinear(C_t - C_{t-3}, 10)",
            {"C_t": "daily close"},
            E.Mul(E.Constant(-1.0), E.DecayLinear(E.Delta(C, 3), 10)),
            2,
        ),
        _candidate(
            "rd_volume_momentum_divergence",
            "[Volume-Price Factor] Price change rank minus volume change rank.",
            r"Rank(C_t - C_{t-5}) - Rank(V_t - V_{t-5})",
            {"C_t": "daily close", "V_t": "daily volume"},
            E.Sub(E.Rank(E.Delta(C, 5)), E.Rank(E.Delta(V, 5))),
            2,
        ),
        _candidate(
            "rd_amount_cv_20d",
            "[Liquidity Factor] Amount coefficient of variation.",
            r"Std(A, 20) / Mean(A, 20)",
            {"A": "daily traded amount"},
            E.Div(E.TsStd(A, 20), E.TsMean(A, 20)),
            2,
        ),
        _candidate(
            "rd_liquidity_discount_20d",
            "[Liquidity Factor] Negative rank of average traded amount.",
            r"-Rank(log(Mean(A, 20)))",
            {"A": "daily traded amount"},
            E.Mul(E.Constant(-1.0), E.Rank(E.Log(E.TsMean(A, 20)))),
            2,
        ),
        _candidate(
            "rd_high_anchor_60d",
            "[Anchor Factor] Close relative to 60-day high.",
            r"C_t / Max(H, 60)",
            {"C_t": "daily close", "H": "daily high"},
            E.Div(C, E.TsMax(H, 60)),
            2,
        ),
        _candidate(
            "rd_open_gap_1d",
            "[Gap Factor] Opening gap from the previous close.",
            r"Rank(O_t - C_{t-1})",
            {"O_t": "daily open", "C_{t-1}": "previous daily close"},
            E.Rank(E.Sub(O, E.Delay(C, 1))),
            2,
        ),
        _candidate(
            "rd_turnover_pressure_20d",
            "[Liquidity Factor] Turnover pressure relative to its 20-day mean.",
            r"Turnover_t / Mean(Turnover, 20)",
            {"Turnover_t": "turnover rate if available"},
            E.Div(E.TurnoverRate, E.TsMean(E.TurnoverRate, 20)),
            3,
        ),
        _candidate(
            "rd_value_pe_discount",
            "[Valuation Factor] Negative rank of PE TTM.",
            r"-Rank(PE_{TTM})",
            {"PE_TTM": "point-in-time trailing PE if available"},
            E.Mul(E.Constant(-1.0), E.Rank(E.PeTtm)),
            3,
        ),
        _candidate(
            "rd_value_pb_discount",
            "[Valuation Factor] Negative rank of price-to-book.",
            r"-Rank(PB)",
            {"PB": "point-in-time price-to-book if available"},
            E.Mul(E.Constant(-1.0), E.Rank(E.Pb)),
            3,
        ),
        _candidate(
            "rd_quality_roe",
            "[Quality Factor] Cross-sectional rank of ROE.",
            r"Rank(ROE)",
            {"ROE": "point-in-time return on equity if available"},
            E.Rank(E.Roe),
            3,
        ),
        _candidate(
            "rd_cashflow_quality",
            "[Quality Factor] Operating cash flow scaled by revenue.",
            r"Rank(OperatingCashFlow / Revenue)",
            {"OperatingCashFlow": "PIT operating cash flow", "Revenue": "PIT revenue"},
            E.Rank(E.Div(E.OperatingCashFlow, E.Revenue)),
            3,
        ),
    ]


def _candidate(
    name: str,
    description: str,
    formulation: str,
    variables: dict[str, str],
    expr: E.Expr,
    complexity_tier: int,
) -> _RDAgentCandidate:
    return _RDAgentCandidate(
        task=RDAgentFactorTask(
            factor_name=name,
            factor_description=description,
            factor_formulation=formulation,
            variables=variables,
        ),
        expr=expr,
        complexity_tier=complexity_tier,
        structure=classify_structure(f"{name} {description} {formulation}"),
        horizon=_expr_horizon(expr),
    )


def _seed_population(rng: random.Random, cfg: SymbolicGAConfig) -> list[E.Expr]:
    """Build the initial population: warm-start templates + mutants + random."""
    population: list[E.Expr] = []
    warm_n = max(0, min(cfg.population, int(round(cfg.population * cfg.warm_start_fraction))))
    templates = _warm_start_templates()
    rng.shuffle(templates)
    for i in range(warm_n):
        base = templates[i % len(templates)]
        # First pass keeps the pristine template, later passes mutate it.
        population.append(base if i < len(templates) else _mutate(base, rng, cfg.max_depth))
    while len(population) < cfg.population:
        population.append(_random_tree(rng, cfg.max_depth, force_internal=True))
    return population


# --------------------------------------------------------------------------- #
# Tree introspection                                                          #
# --------------------------------------------------------------------------- #


def _node_count(node: E.Expr) -> int:
    """Count nodes in a frozen-dataclass tree."""
    return 1 + sum(_node_count(child) for child in _children(node))


def _max_window(node: E.Expr) -> int:
    """Largest lookback window/period anywhere in the expression (0 if none)."""
    best = 0
    stack = [node]
    while stack:
        cur = stack.pop()
        for attr in ("window", "periods"):
            val = getattr(cur, attr, None)
            if isinstance(val, int):
                best = max(best, val)
        stack.extend(_children(cur))
    return best


def _expr_horizon(node: E.Expr) -> str:
    """Map an expression's dominant lookback to a canonical horizon bucket."""
    return classify_horizon(_max_window(node))


def _children(node: E.Expr) -> list[E.Expr]:
    return E.expr_children(node)


def _uses_market_terminal(node: E.Expr) -> bool:
    """Return True when an expression depends on real market input columns."""
    if isinstance(node, E.Column):
        return node.name in set(_MARKET_COLUMNS)
    return any(_uses_market_terminal(child) for child in _children(node))


def _rebuild(node: E.Expr, new_children: list[E.Expr]) -> E.Expr:
    """Reconstruct ``node`` with its Expr-valued fields replaced in order."""
    from dataclasses import fields

    iterator = iter(new_children)
    kwargs = {}
    for f in fields(node):
        value = getattr(node, f.name)
        kwargs[f.name] = next(iterator) if isinstance(value, E.Expr) else value
    return type(node)(**kwargs)


def _replace_subtree(node: E.Expr, target_id: int, replacement: E.Expr, counter: list[int]) -> E.Expr:
    """Return a copy of ``node`` with the ``target_id``-th visited subtree swapped out."""
    counter[0] += 1
    if counter[0] == target_id:
        return replacement
    children = _children(node)
    if not children:
        return node
    new_children = [_replace_subtree(child, target_id, replacement, counter) for child in children]
    return _rebuild(node, new_children)


# --------------------------------------------------------------------------- #
# Genetic operators                                                           #
# --------------------------------------------------------------------------- #


def _crossover(t1: E.Expr, t2: E.Expr, rng: random.Random) -> tuple[E.Expr, E.Expr]:
    n1, n2 = _node_count(t1), _node_count(t2)
    if n1 < 2 or n2 < 2:
        return t1, t2
    a = rng.randrange(2, n1 + 1)
    b = rng.randrange(2, n2 + 1)
    sub1 = _extract_subtree(t1, a, [0])
    sub2 = _extract_subtree(t2, b, [0])
    if sub1 is None or sub2 is None:
        return t1, t2
    c1 = _replace_subtree(t1, a, sub2, [0])
    c2 = _replace_subtree(t2, b, sub1, [0])
    return c1, c2


def _extract_subtree(node: E.Expr, target_id: int, counter: list[int]) -> E.Expr | None:
    counter[0] += 1
    if counter[0] == target_id:
        return node
    for child in _children(node):
        result = _extract_subtree(child, target_id, counter)
        if result is not None:
            return result
    return None


def _mutate(tree: E.Expr, rng: random.Random, max_depth: int) -> E.Expr:
    n = _node_count(tree)
    if n < 2:
        return _random_tree(rng, max_depth)
    target = rng.randrange(2, n + 1)
    replacement = _random_tree(rng, max(1, max_depth - 2))
    return _replace_subtree(tree, target, replacement, [0])


# --------------------------------------------------------------------------- #
# Fitness                                                                     #
# --------------------------------------------------------------------------- #


def _effective_tradability_columns(cfg: "SymbolicGAConfig | RDAgentFactorLoopConfig") -> tuple[str, ...]:
    """Resolve the tradability flag columns a config wants the IC to honour."""
    cols = list(getattr(cfg, "tradability_columns", ()) or ())
    if getattr(cfg, "exclude_st", False) and "is_st" not in cols:
        cols.append("is_st")
    return tuple(cols)


def _tradable_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series | None:
    """Row-aligned bool: ``True`` = investable at signal time.

    A name is dropped from IC computation when any requested flag is set at
    signal time — suspended, limit-up-sealed, limit-down (you cannot
    establish/adjust the long position) and optionally ST. Returns ``None``
    when none of the requested columns are present, so callers transparently
    fall back to the all-names IC (backward compatible).

    This is the production-critical guard against *phantom edge*: a factor must
    not be accepted because it ranks names you could never actually trade.
    """
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return None
    untradable = pd.Series(False, index=frame.index)
    for c in cols:
        untradable = untradable | frame[c].fillna(0).astype(bool)
    return ~untradable


def _daily_rank_ic(
    factor_values: pd.Series,
    labels: pd.Series,
    trade_date: pd.Series,
    tradable_mask: pd.Series | None = None,
) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation series.

    When ``tradable_mask`` (row-aligned bool) is given, untradable rows are
    dropped before ranking so the IC reflects only investable names.
    """
    data = {
        "trade_date": trade_date.values,
        "factor": factor_values.values,
        "label": labels.values,
    }
    if tradable_mask is not None:
        data["tradable"] = np.asarray(tradable_mask, dtype=bool)
    df = pd.DataFrame(data)
    if tradable_mask is not None:
        df = df[df["tradable"]].drop(columns="tradable")
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float)
    grouped = df.groupby("trade_date")
    rank_factor = grouped["factor"].rank(method="average")
    rank_label = grouped["label"].rank(method="average")
    df = df.assign(rank_factor=rank_factor.values, rank_label=rank_label.values)
    daily = (
        df.groupby("trade_date")[["rank_factor", "rank_label"]]
        .corr()
        .unstack()
        .iloc[:, 1]
        .dropna()
    )
    return daily


def _rank_ic(
    factor_values: pd.Series,
    labels: pd.Series,
    trade_date: pd.Series,
    tradable_mask: pd.Series | None = None,
) -> float:
    """Mean of daily cross-sectional Spearman rank correlations."""
    daily = _daily_rank_ic(factor_values, labels, trade_date, tradable_mask)
    if daily.empty:
        return 0.0
    return float(daily.mean())


def _ic_stats(
    factor_values: pd.Series,
    labels: pd.Series,
    trade_date: pd.Series,
    tradable_mask: pd.Series | None = None,
) -> tuple[float, float]:
    """Return (mean rank-IC, daily ICIR)."""
    daily = _daily_rank_ic(factor_values, labels, trade_date, tradable_mask)
    if daily.empty:
        return 0.0, 0.0
    mean = float(daily.mean())
    std = float(daily.std(ddof=0))
    icir = mean / std if std > 1e-12 else 0.0
    return mean, icir


def _factor_economic_profile(
    factor_values: pd.Series,
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    tradable_mask: pd.Series | None = None,
    *,
    quantiles: int = 5,
) -> dict[str, float]:
    """OOS economic profile *beyond* IC for an accepted factor.

    Records top-quantile mean return, long-short (Q_top − Q_bottom) spread,
    cross-sectional monotonicity, and top-quantile turnover, computed on the
    *tradable* OOS sample — the same rows the acceptance rank-IC is measured on,
    so the numbers are methodologically consistent with the gate (no phantom
    edge from untradable names). This is the "严禁只看 IC" requirement: an
    accepted factor must carry a return/turnover profile, not just an IC.

    Horizon IC-decay is intentionally *not* computed here: the acceptance panel
    is sub-sampled (``_subsample_panel`` keeps a contiguous date block but a
    random symbol subset), so re-deriving multi-horizon forward returns by
    shifting ``close`` would be unreliable. Decay belongs to the full-panel
    evaluation step (``scripts/evaluate_discovered_factors.py``).

    Best-effort: any failure returns NaNs and never breaks the acceptance loop.
    """
    out: dict[str, float] = {
        "oos_top_quantile_return": float("nan"),
        "oos_long_short_return": float("nan"),
        "oos_monotonicity": float("nan"),
        "oos_top_quantile_turnover": float("nan"),
    }
    try:
        from quantagent.factors import evaluation as _eval

        frame = pd.DataFrame({
            "trade_date": np.asarray(trade_date.values),
            "symbol": np.asarray(panel["symbol"].values),
            "factor": pd.to_numeric(pd.Series(factor_values.values), errors="coerce").to_numpy(),
            "fwd": pd.to_numeric(pd.Series(labels.values), errors="coerce").to_numpy(),
        })
        if tradable_mask is not None:
            frame = frame[np.asarray(tradable_mask, dtype=bool)]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor", "fwd"])
        if frame.empty or frame["trade_date"].nunique() < 2:
            return out
        qb = _eval.quantile_group_backtest(frame, "factor", "fwd", quantiles=quantiles)
        gr = qb.group_returns
        if quantiles in gr.columns:
            out["oos_top_quantile_return"] = float(gr[quantiles].mean())
        if not qb.long_short.dropna().empty:
            out["oos_long_short_return"] = float(qb.long_short.mean())
        if np.isfinite(qb.monotonicity):
            out["oos_monotonicity"] = float(qb.monotonicity)
        if not qb.turnover.dropna().empty:
            out["oos_top_quantile_turnover"] = float(qb.turnover.mean())
    except Exception:  # noqa: BLE001 — profiling must never break acceptance
        logger.debug("economic profile failed for an accepted factor", exc_info=True)
    return out


def _evaluate_ic(
    tree: E.Expr,
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    tradable_mask: pd.Series | None = None,
) -> tuple[float, float]:
    """Return (rank-IC, finite_ratio) for one tree."""
    try:
        values = tree.evaluate(panel)
    except Exception:
        return 0.0, 0.0
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_ratio = float(values.notna().mean())
    ic = _rank_ic(values, labels, trade_date, tradable_mask)
    return ic, finite_ratio


def _evaluate_fitness(
    tree: E.Expr,
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    cfg: SymbolicGAConfig,
    tradable_mask: pd.Series | None = None,
) -> tuple[float, float, float]:
    """Return (fitness, raw_rank_ic, finite_ratio) for one tree.

    Fitness rewards both the level of the rank-IC and its day-to-day
    stability (daily ICIR), and penalises formula complexity. Absolute
    values are used because a stable negative IC is recoverable by sign
    inversion at selection time. When ``tradable_mask`` is supplied the IC
    is computed over investable names only, steering the search away from
    factors whose apparent edge lives in untradable names.
    """
    if not _uses_market_terminal(tree):
        return -1.0, 0.0, 0.0
    try:
        values = tree.evaluate(panel)
    except Exception:
        return -1.0, 0.0, 0.0
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_ratio = float(values.notna().mean())
    if finite_ratio < cfg.min_finite_ratio:
        return -1.0, 0.0, finite_ratio
    ic_mean, icir = _ic_stats(values, labels, trade_date, tradable_mask)
    score = abs(ic_mean) + cfg.icir_weight * abs(icir) - cfg.complexity_penalty * _node_count(tree)
    return score, ic_mean, finite_ratio


def _chronological_split(
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    cfg: SymbolicGAConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.Series(pd.to_datetime(trade_date).dropna().unique()).sort_values().to_list()
    if len(dates) < 4 or cfg.validation_fraction <= 0:
        return panel, labels, trade_date, panel, labels, trade_date
    valid_count = max(1, int(round(len(dates) * min(cfg.validation_fraction, 0.5))))
    cutoff_dates = set(dates[-valid_count:])
    mask = pd.to_datetime(trade_date).isin(cutoff_dates)
    train_panel = panel.loc[~mask].reset_index(drop=True)
    valid_panel = panel.loc[mask].reset_index(drop=True)
    train_labels = labels.loc[~mask].reset_index(drop=True)
    valid_labels = labels.loc[mask].reset_index(drop=True)
    train_dates = trade_date.loc[~mask].reset_index(drop=True)
    valid_dates = trade_date.loc[mask].reset_index(drop=True)
    if train_panel.empty or valid_panel.empty:
        return panel, labels, trade_date, panel, labels, trade_date
    return train_panel, train_labels, train_dates, valid_panel, valid_labels, valid_dates


def _merge_factor_panel(
    panel: pd.DataFrame,
    labels: pd.DataFrame | None,
    label_column: str,
    reference_columns: Sequence[str],
) -> pd.DataFrame:
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if labels is None or labels.empty:
        return panel
    labels = labels.copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
    keep = ["symbol", "trade_date"]
    if label_column in labels.columns:
        keep.append(label_column)
        labels = labels[labels[label_column].notna()]
    for ref in reference_columns:
        if ref in labels.columns and ref not in panel.columns and ref not in keep:
            keep.append(ref)
    return panel.merge(labels[[c for c in keep if c in labels.columns]], on=["symbol", "trade_date"], how="inner")


def _rd_agent_hypothesis(round_idx: int, trace_rows: list[dict[str, object]]) -> RDAgentFactorHypothesis:
    accepted = sum(int(row.get("accepted_count", 0)) for row in trace_rows)
    if round_idx == 0:
        return RDAgentFactorHypothesis(
            hypothesis=(
                "Start with simple price, volume, and intraday-position factors before testing "
                "more complex combinations."
            ),
            reason=(
                "RD-Agent's finance factor loop prioritizes easy-to-implement factors in early "
                "rounds so failures are attributable and the SOTA library can accumulate cleanly."
            ),
            concise_observation="No previous QuantAgent factor-loop feedback is available.",
            concise_justification="Simple factors first.",
            concise_knowledge="Use PIT OHLCV only and avoid duplicate SOTA factors.",
        )
    if accepted == 0:
        return RDAgentFactorHypothesis(
            hypothesis=(
                "Switch to a new factor direction because earlier candidates did not survive "
                "the output, novelty, or validation gates."
            ),
            reason=(
                "RD-Agent changes direction after unsuccessful iterations instead of repeatedly "
                "implementing near-duplicates."
            ),
            concise_observation="No factor has entered the SOTA library yet.",
            concise_justification="Change direction after failures.",
            concise_knowledge="Reject invalid output and highly correlated factors.",
        )
    return RDAgentFactorHypothesis(
        hypothesis=(
            "Extend the accumulated SOTA factor library with factors from a different economic "
            "mechanism while preserving novelty against accepted factors."
        ),
        reason=(
            "RD-Agent keeps all factors that improve the iterative library and feeds their "
            "feedback into the next proposal."
        ),
        concise_observation=f"{accepted} factors have been accepted so far.",
        concise_justification="Optimize around accepted SOTA factors without reimplementing them.",
        concise_knowledge="New factors must pass validation IC and SOTA correlation gates.",
    )


def _rd_agent_task_slice(
    candidates: Sequence[_RDAgentCandidate],
    round_idx: int,
    cfg: RDAgentFactorLoopConfig,
) -> list[_RDAgentCandidate]:
    factors_per_round = max(1, min(5, int(cfg.factors_per_round)))
    ordered = sorted(candidates, key=lambda c: (c.complexity_tier, c.task.factor_name))
    start = round_idx * factors_per_round
    selected = ordered[start : start + factors_per_round]
    if len(selected) == factors_per_round:
        return selected
    selected = list(selected)
    selected.extend(ordered[: factors_per_round - len(selected)])
    return selected


def _daily_output_check(frame: pd.DataFrame) -> tuple[bool, str]:
    if "trade_date" not in frame.columns or "symbol" not in frame.columns:
        return False, "Output frame misses trade_date or symbol."
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if dates.isna().any():
        return False, "trade_date contains unparsable values."
    has_intraday_time = (dates.dt.normalize() != dates).any()
    if has_intraday_time:
        return False, "Generated factor frame is not daily; rd-agent qlib factor loop expects daily bars."
    return True, "Generated factor frame is daily."


def _rd_agent_value_feedback(
    expr: E.Expr,
    frame: pd.DataFrame,
    min_finite_ratio: float,
) -> tuple[bool, str, pd.Series | None, float]:
    daily_ok, daily_feedback = _daily_output_check(frame)
    try:
        values = expr.evaluate(frame)
    except Exception as exc:
        return False, f"Execution failed: {exc}", None, 0.0
    if not isinstance(values, pd.Series):
        return False, "Implementation did not return a pandas Series.", None, 0.0
    if len(values) != len(frame):
        return False, f"Output row count mismatch: got {len(values)} values for {len(frame)} input rows.", None, 0.0
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_ratio = float(values.notna().mean())
    feedback = [
        "Execution succeeded without error.",
        "The source dataframe has only one column which is correct.",
        "The source dataframe does not have any infinite values.",
        daily_feedback,
        f"The finite value ratio is {finite_ratio:.4f}.",
    ]
    if not daily_ok:
        return False, "\n".join(feedback), values, finite_ratio
    if finite_ratio < min_finite_ratio:
        feedback.append(
            f"The finite value ratio is below the required threshold {min_finite_ratio:.4f}."
        )
        return False, "\n".join(feedback), values, finite_ratio
    return True, "\n".join(feedback), values, finite_ratio


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    corr = left.corr(right, method="spearman")
    return float(corr) if np.isfinite(corr) else 0.0


def _rd_agent_feedback_row(
    hypothesis: RDAgentFactorHypothesis,
    task: RDAgentFactorTask,
    decision: bool,
    reason: str,
    train_ic: float = 0.0,
    validation_ic: float = 0.0,
    max_sota_corr: float = 0.0,
) -> dict[str, object]:
    return {
        "hypothesis": hypothesis.hypothesis,
        "reason": hypothesis.reason,
        "factor_name": task.factor_name,
        "factor_description": task.factor_description,
        "factor_formulation": task.factor_formulation,
        "variables": json.dumps(task.variables, ensure_ascii=False, sort_keys=True),
        "factor_implementation": bool(task.factor_implementation),
        "decision": bool(decision),
        "feedback": reason,
        "train_rank_ic": float(train_ic),
        "validation_rank_ic": float(validation_ic),
        "max_sota_corr": float(max_sota_corr),
    }


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def _sample_contiguous_dates(dates: list, n_dates: int, rng: random.Random) -> set:
    """Pick a random contiguous block of ``n_dates`` trading dates.

    Contiguity matters: rolling/delay operators computed over randomly
    sampled (gapped) dates silently produce wrong window contents.
    """
    if n_dates >= len(dates):
        return set(dates)
    start = rng.randrange(0, len(dates) - n_dates + 1)
    return set(dates[start : start + n_dates])


def _subsample_panel(
    panel: pd.DataFrame,
    label_col: str,
    cfg: SymbolicGAConfig,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sub-sample dates and symbols to keep fitness evaluation tractable."""
    if label_col not in panel.columns:
        raise KeyError(f"label column '{label_col}' missing from panel")
    valid = panel[panel[label_col].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    dates = sorted(valid["trade_date"].unique())
    if cfg.fitness_sample_dates and len(dates) > cfg.fitness_sample_dates:
        chosen_dates = _sample_contiguous_dates(dates, cfg.fitness_sample_dates, rng)
        valid = valid[valid["trade_date"].isin(chosen_dates)]
    symbols = valid["symbol"].unique()
    if cfg.fitness_sample_symbols and len(symbols) > cfg.fitness_sample_symbols:
        chosen_symbols = rng.sample(list(symbols), cfg.fitness_sample_symbols)
        valid = valid[valid["symbol"].isin(chosen_symbols)]
    valid = valid.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return valid, valid[label_col], valid["trade_date"]


def _horizon_diverse_names(leaderboard: pd.DataFrame, k: int) -> list[str]:
    """Pick up to ``k`` factor names round-robin across horizon buckets.

    ``leaderboard`` must already be sorted best-first. Within each horizon the
    order is preserved (best IC first); buckets are visited in the order their
    best member appears, so the global #1 is always picked first and weaker
    horizons are only reached once stronger ones have contributed a pick.
    """
    if leaderboard.empty or k <= 0:
        return []
    if "horizon" not in leaderboard.columns:
        return leaderboard["name"].head(k).astype(str).tolist()
    buckets: dict[str, list[str]] = {}
    for _, row in leaderboard.iterrows():
        buckets.setdefault(str(row.get("horizon") or "unspecified"), []).append(str(row["name"]))
    bucket_order = list(buckets.keys())
    order: list[str] = []
    depth = 0
    while len(order) < k:
        progressed = False
        for b in bucket_order:
            pool = buckets[b]
            if depth < len(pool):
                order.append(pool[depth])
                progressed = True
                if len(order) >= k:
                    break
        if not progressed:
            break
        depth += 1
    return order[:k]


def synthesize_factors_rd_agent(
    panel: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    config: RDAgentFactorLoopConfig | None = None,
    proposer: "FactorProposer | None" = None,
) -> RDAgentSynthesisResult:
    """Run an RD-Agent-style factor R&D loop in QuantAgent's safe DSL.

    This is intentionally not a free-form code execution system. It mirrors
    RD-Agent's loop and artifact semantics while preserving QuantAgent's
    production constraint that generated factors must stay inside the audited
    PIT expression DSL.

    The loop is a *closed* feedback cycle when ``cfg.use_llm`` is set: a warm
    start round seeds the SOTA library from hand-written blueprints, then later
    rounds ask ``proposer`` (an LLM, by default) to propose NEW DSL factor
    tasks conditioned on the accumulated trace and the persistent accept/reject
    memory. Every proposal — blueprint or LLM — passes the same value, finite,
    validation-IC, and novelty gates. Inject ``proposer`` to run fully offline
    (tests do this); otherwise an LLM proposer is built lazily when the model
    is reachable, and any round where it is unavailable falls back to the
    blueprint slice.
    """
    cfg = config or RDAgentFactorLoopConfig()
    rng = random.Random(cfg.seed)
    if proposer is None and cfg.use_llm and cfg.allow_network:
        try:
            from quantagent.factors.llm_factor_proposer import LLMFactorProposer

            proposer = LLMFactorProposer(
                model=cfg.llm_model,
                allow_network=cfg.allow_network,
                timeout_seconds=cfg.llm_timeout_seconds,
                max_attempts=cfg.max_llm_attempts,
            )
        except Exception as exc:  # pragma: no cover - import/config guard
            logger.warning("[rd-agent-factor] LLM proposer unavailable (%s); using blueprints only", exc)
            proposer = None
    use_llm = bool(cfg.use_llm and proposer is not None)
    persisted_memory = load_memory(cfg.memory_path)
    knowledge_rows: list[dict[str, object]] = []
    merged = _merge_factor_panel(panel, labels, cfg.label_column, cfg.reference_columns)
    if cfg.label_column not in merged.columns:
        raise KeyError(
            f"synthesize_factors_rd_agent needs label column '{cfg.label_column}' in panel or labels"
        )
    missing_market = [c for c in _MARKET_COLUMNS if c not in merged.columns]
    if missing_market:
        raise KeyError(
            "merged RD-Agent factor panel lost market columns "
            f"{missing_market}; available: {sorted(merged.columns)[:40]}"
        )

    valid_panel, label_series, date_series = _subsample_panel(merged, cfg.label_column, cfg, rng)
    train_panel, train_labels, train_dates, oos_panel, oos_labels, oos_dates = _chronological_split(
        valid_panel,
        label_series.reset_index(drop=True),
        date_series.reset_index(drop=True),
        cfg,
    )
    logger.info("[rd-agent-factor] sample: %d rows, %d dates, %d symbols",
                len(valid_panel), valid_panel["trade_date"].nunique(), valid_panel["symbol"].nunique())
    logger.info("[rd-agent-factor] split: train=%d rows, validation=%d rows", len(train_panel), len(oos_panel))

    # Tradability guard: when the panel carries flag columns, fitness and the
    # validation IC honour only names investable at signal time, so a factor
    # cannot be accepted on phantom edge over untradable names.
    tradability_cols = _effective_tradability_columns(cfg)
    train_tradable_mask = _tradable_mask(train_panel, tradability_cols)
    oos_tradable_mask = _tradable_mask(oos_panel, tradability_cols)
    if oos_tradable_mask is not None:
        logger.info(
            "[rd-agent-factor] tradability guard active %s: OOS tradable rows=%d/%d",
            list(tradability_cols), int(oos_tradable_mask.sum()), len(oos_tradable_mask),
        )

    candidates = _rd_agent_factor_blueprints()
    accepted: list[tuple[E.FactorDefinition, E.Expr, pd.Series]] = []
    leaderboard_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    task_feedback_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    seen_exprs: set[str] = set()
    # Persistent-loop novelty: seed the attempted-set from prior runs' memory so
    # a fresh run does not re-mine expressions already explored — each iteration
    # is steered to NEW orthogonal space. (Empty memory => no-op.) Per-run
    # outputs are unioned across iterations by the closed-loop driver.
    for _row in persisted_memory:
        for _key in ("expression", "raw_expression"):
            _e = _row.get(_key)
            if _e:
                seen_exprs.add(str(_e))
    reference_values: dict[str, pd.Series] = {
        ref: pd.to_numeric(oos_panel[ref], errors="coerce")
        for ref in cfg.reference_columns
        if ref in oos_panel.columns
    }

    for round_idx in range(max(0, int(cfg.rounds))):
        hypothesis = _rd_agent_hypothesis(round_idx, trace_rows)
        round_source = "blueprint"
        if use_llm and round_idx >= cfg.llm_start_round:
            all_knowledge = [*persisted_memory, *knowledge_rows]
            digest = memory_digest(all_knowledge)
            # Structured coverage feedback (lightweight RD-Agent knowledge port):
            # steer the proposer at the orthogonal whitespace, away from crowded
            # dead ends, instead of only showing it a flat recent-list.
            digest["coverage_map"] = coverage_map(all_knowledge)
            digest["uncovered_directions"] = uncovered_directions(all_knowledge)
            rag_directive = RAG_HIGH_IC if round_idx >= cfg.rag_escalation_round else RAG_EASY
            try:
                proposal = proposer.propose(
                    round_idx=round_idx,
                    hypothesis=hypothesis,
                    rag_directive=rag_directive,
                    memory_digest_payload=digest,
                    n_candidates=max(1, int(cfg.llm_candidates_per_round)),
                    seen_expr_reprs=sorted(seen_exprs),
                )
            except Exception as exc:  # pragma: no cover - network/parse guard
                logger.warning("[rd-agent-factor] round %d LLM proposal failed (%s); using blueprints", round_idx + 1, exc)
                proposal = None
            llm_candidates = [
                _RDAgentCandidate(
                    task=RDAgentFactorTask(
                        factor_name=pf.name,
                        factor_description=pf.description,
                        factor_formulation=pf.formulation,
                        variables=dict(pf.variables),
                    ),
                    expr=pf.expr,
                    complexity_tier=int(pf.complexity_tier),
                    structure=pf.structure or classify_structure(f"{pf.hypothesis} {pf.description}"),
                    horizon=classify_horizon(pf.horizon) if pf.horizon else _expr_horizon(pf.expr),
                )
                for pf in (proposal.factors if proposal is not None else [])
            ]
            if llm_candidates:
                hypothesis = proposal.hypothesis or hypothesis
                round_candidates = llm_candidates
                round_source = "llm"
            else:
                round_candidates = _rd_agent_task_slice(candidates, round_idx, cfg)
                logger.info("[rd-agent-factor] round %d LLM yielded no usable factors; using blueprints", round_idx + 1)
        else:
            round_candidates = _rd_agent_task_slice(candidates, round_idx, cfg)
        round_accepted = 0
        round_best_ic = -np.inf
        round_feedback: list[str] = []

        for candidate in round_candidates:
            task = candidate.task
            expr_key = repr(candidate.expr)
            if expr_key in seen_exprs:
                task = RDAgentFactorTask(
                    factor_name=task.factor_name,
                    factor_description=task.factor_description,
                    factor_formulation=task.factor_formulation,
                    variables=task.variables,
                    factor_implementation=False,
                )
                feedback = "Rejected before implementation because this expression was already attempted."
                task_feedback_rows.append(_rd_agent_feedback_row(hypothesis, task, False, feedback))
                round_feedback.append(f"{task.factor_name}: duplicate expression")
                continue
            seen_exprs.add(expr_key)

            value_ok, value_feedback, _, finite_ratio = _rd_agent_value_feedback(
                candidate.expr,
                train_panel,
                cfg.min_finite_ratio,
            )
            task = RDAgentFactorTask(
                factor_name=task.factor_name,
                factor_description=task.factor_description,
                factor_formulation=task.factor_formulation,
                variables=task.variables,
                factor_implementation=bool(value_ok),
            )
            if not value_ok:
                task_feedback_rows.append(_rd_agent_feedback_row(hypothesis, task, False, value_feedback))
                round_feedback.append(f"{task.factor_name}: value gate failed")
                continue

            fitness, train_ic, train_finite = _evaluate_fitness(
                candidate.expr,
                train_panel,
                train_labels,
                train_dates,
                SymbolicGAConfig(
                    label_column=cfg.label_column,
                    complexity_penalty=cfg.complexity_penalty,
                    min_finite_ratio=cfg.min_finite_ratio,
                    icir_weight=cfg.icir_weight,
                ),
                train_tradable_mask,
            )
            if fitness <= -1.0 or train_finite < cfg.min_finite_ratio:
                feedback = (
                    value_feedback
                    + "\nRejected because the implementation did not produce a viable training fitness."
                )
                task_feedback_rows.append(_rd_agent_feedback_row(hypothesis, task, False, feedback, train_ic, 0.0))
                round_feedback.append(f"{task.factor_name}: training fitness failed")
                continue

            oriented_expr = candidate.expr if train_ic >= 0 else E.Mul(E.Constant(-1.0), candidate.expr)
            try:
                oos_values = pd.to_numeric(oriented_expr.evaluate(oos_panel), errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            except Exception as exc:
                feedback = value_feedback + f"\nRejected because validation execution failed: {exc}"
                task_feedback_rows.append(_rd_agent_feedback_row(hypothesis, task, False, feedback, train_ic, 0.0))
                round_feedback.append(f"{task.factor_name}: validation execution failed")
                continue
            validation_finite = float(oos_values.notna().mean())
            # The acceptance gate uses the TRADABLE IC (untradable names dropped);
            # the raw all-names IC is recorded so the phantom-edge gap is visible.
            validation_ic, validation_icir = _ic_stats(oos_values, oos_labels, oos_dates, oos_tradable_mask)
            validation_ic_raw = _rank_ic(oos_values, oos_labels, oos_dates)

            max_sota_corr = 0.0
            max_reference_corr = 0.0
            for _, _, prev_values in accepted:
                max_sota_corr = max(max_sota_corr, abs(_safe_spearman(oos_values, prev_values)))
            for ref_series in reference_values.values():
                max_reference_corr = max(max_reference_corr, abs(_safe_spearman(oos_values, ref_series)))
            max_novelty_corr = max(max_sota_corr, max_reference_corr)

            if validation_finite < cfg.min_finite_ratio:
                feedback = (
                    value_feedback
                    + f"\nRejected because validation finite ratio {validation_finite:.4f} "
                    f"is below {cfg.min_finite_ratio:.4f}."
                )
                task_feedback_rows.append(
                    _rd_agent_feedback_row(
                        hypothesis,
                        task,
                        False,
                        feedback,
                        train_ic,
                        validation_ic,
                        max_novelty_corr,
                    )
                )
                round_feedback.append(f"{task.factor_name}: validation finite gate failed")
                continue
            if validation_ic < cfg.min_validation_rank_ic:
                feedback = (
                    value_feedback
                    + f"\nRejected because validation RankIC {validation_ic:.6f} "
                    f"is below {cfg.min_validation_rank_ic:.6f}."
                )
                task_feedback_rows.append(
                    _rd_agent_feedback_row(
                        hypothesis,
                        task,
                        False,
                        feedback,
                        train_ic,
                        validation_ic,
                        max_novelty_corr,
                    )
                )
                round_feedback.append(f"{task.factor_name}: validation IC gate failed")
                continue
            if cfg.min_validation_icir > 0.0 and validation_icir < cfg.min_validation_icir:
                feedback = (
                    value_feedback
                    + f"\nRejected because validation ICIR {validation_icir:.4f} "
                    f"is below {cfg.min_validation_icir:.4f} (unstable day-to-day IC)."
                )
                task_feedback_rows.append(
                    _rd_agent_feedback_row(
                        hypothesis,
                        task,
                        False,
                        feedback,
                        train_ic,
                        validation_ic,
                        max_novelty_corr,
                    )
                )
                round_feedback.append(f"{task.factor_name}: validation ICIR gate failed")
                continue
            if max_sota_corr > cfg.max_sota_correlation:
                feedback = (
                    value_feedback
                    + f"\nRejected because max SOTA/reference Spearman correlation {max_sota_corr:.6f} "
                    f"exceeds {cfg.max_sota_correlation:.6f}."
                )
                task_feedback_rows.append(
                    _rd_agent_feedback_row(
                        hypothesis,
                        task,
                        False,
                        feedback,
                        train_ic,
                        validation_ic,
                        max_novelty_corr,
                    )
                )
                round_feedback.append(f"{task.factor_name}: SOTA duplicate gate failed")
                continue
            if max_reference_corr > cfg.max_reference_correlation:
                feedback = (
                    value_feedback
                    + f"\nRejected because max reference Spearman correlation {max_reference_corr:.6f} "
                    f"exceeds {cfg.max_reference_correlation:.6f}."
                )
                task_feedback_rows.append(
                    _rd_agent_feedback_row(
                        hypothesis,
                        task,
                        False,
                        feedback,
                        train_ic,
                        validation_ic,
                        max_novelty_corr,
                    )
                )
                round_feedback.append(f"{task.factor_name}: reference duplicate gate failed")
                continue

            definition = E.FactorDefinition(
                name=task.factor_name,
                expr=oriented_expr,
                description=f"RD-Agent-style factor. {task.factor_description}",
            )
            accepted.append((definition, oriented_expr, oos_values))
            round_accepted += 1
            round_best_ic = max(round_best_ic, validation_ic)
            feedback = value_feedback + "\nAccepted into the SOTA factor library for subsequent rounds."
            task_feedback_rows.append(
                _rd_agent_feedback_row(hypothesis, task, True, feedback, train_ic, validation_ic, max_novelty_corr)
            )
            round_feedback.append(f"{task.factor_name}: accepted")
            # Beyond-IC economic profile (top-bucket/long-short return, monotonicity,
            # turnover) on the same tradable OOS sample the IC gate used — so each
            # accepted factor carries a return/turnover signature, not just an IC.
            profile = _factor_economic_profile(
                oos_values, oos_panel, oos_labels, oos_dates, oos_tradable_mask,
            )
            leaderboard_rows.append(
                {
                    "name": definition.name,
                    "expression": repr(oriented_expr),
                    "train_rank_ic": float(abs(train_ic)),
                    "validation_rank_ic": float(validation_ic),
                    "validation_rank_ic_raw": float(validation_ic_raw),
                    "validation_rank_icir": float(validation_icir),
                    "fitness": float(fitness),
                    "complexity": int(_node_count(oriented_expr)),
                    "finite_ratio": float(min(finite_ratio, validation_finite)),
                    "max_reference_corr": float(max_reference_corr),
                    "max_sota_corr": float(max_sota_corr),
                    "oos_top_quantile_return": profile["oos_top_quantile_return"],
                    "oos_long_short_return": profile["oos_long_short_return"],
                    "oos_monotonicity": profile["oos_monotonicity"],
                    "oos_top_quantile_turnover": profile["oos_top_quantile_turnover"],
                    "round": int(round_idx + 1),
                    "structure": candidate.structure,
                    "horizon": candidate.horizon,
                    "hypothesis": hypothesis.hypothesis,
                    "factor_description": task.factor_description,
                    "factor_formulation": task.factor_formulation,
                    "factor_implementation": True,
                }
            )
            if len(accepted) >= cfg.top_k:
                break

        # Distil this round's outcomes into accept/reject knowledge that feeds
        # the next LLM proposal and persists across runs (RD-Agent's evolving
        # memory). The short status strings double as machine-readable reasons.
        round_expr_by_name = {c.task.factor_name: repr(c.expr) for c in round_candidates}
        round_struct_by_name = {c.task.factor_name: c.structure for c in round_candidates}
        round_horizon_by_name = {c.task.factor_name: c.horizon for c in round_candidates}
        accepted_ic_by_name = {
            row["name"]: row["validation_rank_ic"]
            for row in leaderboard_rows
            if int(row.get("round", 0)) == round_idx + 1
        }
        round_knowledge: list[dict[str, object]] = []
        for entry in round_feedback:
            name, _, status_text = entry.partition(": ")
            status_text = status_text.strip()
            selected = status_text == "accepted"
            round_knowledge.append(
                {
                    "round": int(round_idx + 1),
                    "source": round_source,
                    "name": name,
                    "expression": round_expr_by_name.get(name, ""),
                    "raw_expression": round_expr_by_name.get(name, ""),
                    "structure": round_struct_by_name.get(name, "other"),
                    "horizon": round_horizon_by_name.get(name, "unspecified"),
                    "concise_knowledge": hypothesis.concise_knowledge,
                    "status": "selected" if selected else (status_text or "rejected"),
                    "validation_rank_ic": (accepted_ic_by_name.get(name) if selected else None),
                    "hypothesis": hypothesis.hypothesis,
                }
            )
        knowledge_rows.extend(round_knowledge)
        append_memory(cfg.memory_path, round_knowledge)

        trace_rows.append(
            {
                "round": int(round_idx + 1),
                "source": round_source,
                "hypothesis": hypothesis.hypothesis,
                "reason": hypothesis.reason,
                "candidate_count": int(len(round_candidates)),
                "accepted_count": int(round_accepted),
                "sota_size": int(len(accepted)),
                "best_validation_rank_ic": float(round_best_ic) if np.isfinite(round_best_ic) else 0.0,
                "observations": "; ".join(round_feedback),
                "decision": bool(round_accepted > 0),
            }
        )
        history_rows.append(
            {
                "round": int(round_idx + 1),
                "candidate_count": int(len(round_candidates)),
                "accepted_count": int(round_accepted),
                "sota_size": int(len(accepted)),
                "best_validation_rank_ic": float(round_best_ic) if np.isfinite(round_best_ic) else 0.0,
            }
        )
        logger.info(
            "[rd-agent-factor] round %d accepted=%d sota_size=%d best_valid_ic=%.4f",
            round_idx + 1,
            round_accepted,
            len(accepted),
            float(round_best_ic) if np.isfinite(round_best_ic) else 0.0,
        )
        if len(accepted) >= cfg.top_k:
            break

    definitions = [definition for definition, _, _ in accepted[: cfg.top_k]]
    leaderboard = pd.DataFrame(
        leaderboard_rows,
        columns=[
            "name",
            "expression",
            "train_rank_ic",
            "validation_rank_ic",
            "validation_rank_ic_raw",
            "validation_rank_icir",
            "fitness",
            "complexity",
            "finite_ratio",
            "max_reference_corr",
            "max_sota_corr",
            "oos_top_quantile_return",
            "oos_long_short_return",
            "oos_monotonicity",
            "oos_top_quantile_turnover",
            "round",
            "structure",
            "horizon",
            "hypothesis",
            "factor_description",
            "factor_formulation",
            "factor_implementation",
        ],
    )
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["validation_rank_ic", "train_rank_ic"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        definitions_by_name = {definition.name: definition for definition in definitions}
        # Horizon-diverse top_k: round-robin across horizon buckets (best-first
        # within each) so one horizon cannot monopolise the library while a
        # strictly-orthogonal horizon goes unrepresented.
        selected_names = _horizon_diverse_names(leaderboard, cfg.top_k)
        definitions = [
            definitions_by_name[name] for name in selected_names if name in definitions_by_name
        ]
    return RDAgentSynthesisResult(
        definitions=definitions,
        leaderboard=leaderboard,
        history=pd.DataFrame(history_rows),
        trace=pd.DataFrame(trace_rows),
        task_feedback=pd.DataFrame(task_feedback_rows),
    )


def synthesize_factors(
    panel: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    config: SymbolicGAConfig | None = None,
) -> SynthesisResult:
    """Run the symbolic regression GA and return top-K discovered factors.

    Parameters
    ----------
    panel
        Wide market-panel frame with at minimum ``symbol``, ``trade_date``,
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``amount``.
    labels
        Optional labels frame keyed on (symbol, trade_date) containing
        ``forward_return_*`` columns. If absent, the function looks for
        ``forward_return_5d`` (or ``cfg.label_column``) directly in ``panel``.
    """
    cfg = config or SymbolicGAConfig()
    rng = random.Random(cfg.seed)
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if labels is not None and not labels.empty:
        labels = labels.copy()
        labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
        # Keep only the join keys, the label and any requested reference
        # factor columns. Carrying the full labels frame into the merge
        # used to suffix overlapping OHLCV columns (close -> close_x/_y),
        # which silently killed every expression evaluation.
        keep = ["symbol", "trade_date"]
        if cfg.label_column in labels.columns:
            keep.append(cfg.label_column)
            labels = labels[labels[cfg.label_column].notna()]
        for ref in cfg.reference_columns:
            if ref in labels.columns and ref not in panel.columns and ref not in keep:
                keep.append(ref)
        labels = labels[[c for c in keep if c in labels.columns]]
        if cfg.fitness_sample_dates and labels["trade_date"].nunique() > cfg.fitness_sample_dates:
            sampled_dates = _sample_contiguous_dates(
                sorted(labels["trade_date"].dropna().unique()), cfg.fitness_sample_dates, rng
            )
            labels = labels[labels["trade_date"].isin(sampled_dates)]
            panel = panel[panel["trade_date"].isin(sampled_dates)]
        if cfg.fitness_sample_symbols and labels["symbol"].nunique() > cfg.fitness_sample_symbols:
            sampled_symbols = set(rng.sample(list(labels["symbol"].dropna().astype(str).unique()), cfg.fitness_sample_symbols))
            labels = labels[labels["symbol"].astype(str).isin(sampled_symbols)]
            panel = panel[panel["symbol"].astype(str).isin(sampled_symbols)]
        merged = panel.merge(labels, on=["symbol", "trade_date"], how="inner")
    else:
        merged = panel
    if cfg.label_column not in merged.columns:
        raise KeyError(
            f"synthesize_factors needs label column '{cfg.label_column}' in panel or labels"
        )
    missing_market = [c for c in _MARKET_COLUMNS if c not in merged.columns]
    if missing_market:
        raise KeyError(
            "merged GA panel lost market columns "
            f"{missing_market} (suffix collision or wrong --market-panel input); "
            f"available: {sorted(merged.columns)[:40]}"
        )
    valid_panel, label_series, date_series = _subsample_panel(merged, cfg.label_column, cfg, rng)
    train_panel, train_labels, train_dates, oos_panel, oos_labels, oos_dates = _chronological_split(
        valid_panel,
        label_series.reset_index(drop=True),
        date_series.reset_index(drop=True),
        cfg,
    )
    logger.info("[ga] fitness sample: %d rows, %d dates, %d symbols",
                len(valid_panel), valid_panel["trade_date"].nunique(), valid_panel["symbol"].nunique())
    logger.info("[ga] split: train=%d rows, validation=%d rows", len(train_panel), len(oos_panel))

    tradability_cols = _effective_tradability_columns(cfg)
    train_tradable_mask = _tradable_mask(train_panel, tradability_cols)
    oos_tradable_mask = _tradable_mask(oos_panel, tradability_cols)

    eval_cache: dict[str, tuple[float, float, float]] = {}

    def _cached_fitness(tree: E.Expr) -> tuple[float, float, float]:
        key = repr(tree)
        if key not in eval_cache:
            eval_cache[key] = _evaluate_fitness(
                tree, train_panel, train_labels, train_dates, cfg, train_tradable_mask
            )
        return eval_cache[key]

    population: list[E.Expr] = _seed_population(rng, cfg)
    fitness: list[float] = [-1.0] * cfg.population
    raw_ics: list[float] = [0.0] * cfg.population
    finite_ratios: list[float] = [0.0] * cfg.population
    for i, tree in enumerate(population):
        fitness[i], raw_ics[i], finite_ratios[i] = _cached_fitness(tree)

    if max(fitness) <= -1.0:
        # Every tree was rejected; surface the underlying error instead of
        # silently burning generations on a dead population.
        probe = E.Rank(E.Returns(E.Close, 5))
        probe.evaluate(train_panel)  # raises with the real cause if data is broken
        raise RuntimeError(
            "symbolic GA: generation 0 produced no viable tree although the probe "
            "expression evaluates; check label alignment and min_finite_ratio "
            f"(finite ratios seen: {sorted(set(round(f, 2) for f in finite_ratios))[:10]})"
        )

    history_rows: list[dict[str, float]] = []
    for generation in range(cfg.generations):
        best = max(fitness)
        mean = float(np.mean(fitness))
        mean_complexity = float(np.mean([_node_count(t) for t in population]))
        history_rows.append({
            "generation": generation,
            "best_fitness": best,
            "mean_fitness": mean,
            "mean_complexity": mean_complexity,
        })
        logger.info("[ga] gen %d  best=%.4f  mean=%.4f  mean_complexity=%.1f",
                    generation, best, mean, mean_complexity)

        # Elitism + tournament selection + crossover/mutation.
        order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        elites = [population[i] for i in order[: cfg.elitism]]
        new_pop = list(elites)
        while len(new_pop) < cfg.population:
            p1 = _tournament_pick(rng, population, fitness, cfg.tournament_size)
            p2 = _tournament_pick(rng, population, fitness, cfg.tournament_size)
            if rng.random() < cfg.crossover_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            if rng.random() < cfg.mutation_prob:
                c1 = _mutate(c1, rng, cfg.max_depth)
            if rng.random() < cfg.mutation_prob:
                c2 = _mutate(c2, rng, cfg.max_depth)
            new_pop.extend([c1, c2])
        new_pop = new_pop[: cfg.population]
        inject_n = int(round(cfg.population * cfg.random_injection_rate))
        for j in range(inject_n):
            new_pop[-(j + 1)] = _random_tree(rng, cfg.max_depth, force_internal=True)
        population = new_pop
        fitness = [-1.0] * cfg.population
        raw_ics = [0.0] * cfg.population
        finite_ratios = [0.0] * cfg.population
        for i, tree in enumerate(population):
            fitness[i], raw_ics[i], finite_ratios[i] = _cached_fitness(tree)

    # Final leaderboard.
    order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
    rows: list[dict[str, object]] = []
    chosen: list[E.Expr] = []
    chosen_values: list[pd.Series] = []
    seen_exprs: set[str] = set()
    reference_values: dict[str, pd.Series] = {
        ref: pd.to_numeric(oos_panel[ref], errors="coerce")
        for ref in cfg.reference_columns
        if ref in oos_panel.columns
    }
    for idx in order:
        tree = population[idx]
        if fitness[idx] <= 0:
            break
        expr_key = repr(tree)
        if expr_key in seen_exprs:
            continue
        seen_exprs.add(expr_key)
        train_ic = raw_ics[idx]
        oriented_tree = tree if train_ic >= 0 else E.Mul(E.Constant(-1.0), tree)
        validation_ic, validation_finite = _evaluate_ic(
            oriented_tree, oos_panel, oos_labels, oos_dates, oos_tradable_mask
        )
        if validation_finite < cfg.min_finite_ratio or validation_ic < cfg.min_validation_rank_ic:
            continue
        try:
            values = pd.to_numeric(oriented_tree.evaluate(oos_panel), errors="coerce")
        except Exception:
            continue
        validation_ic_raw = _rank_ic(values, oos_labels, oos_dates)
        # Decorrelate against already-chosen survivors.
        if any(
            abs(values.corr(prev, method="spearman")) > cfg.max_correlation
            for prev in chosen_values
            if prev is not None
        ):
            continue
        # Decorrelate against the existing factor library (novelty gate).
        ref_corr = 0.0
        for ref_series in reference_values.values():
            corr = abs(values.corr(ref_series, method="spearman"))
            if np.isfinite(corr):
                ref_corr = max(ref_corr, float(corr))
        if reference_values and ref_corr > cfg.max_reference_correlation:
            continue
        chosen.append(oriented_tree)
        chosen_values.append(values)
        rows.append({
            "name": f"synth_{len(chosen):03d}",
            "expression": repr(oriented_tree),
            "train_rank_ic": float(abs(train_ic)),
            "validation_rank_ic": float(validation_ic),
            "validation_rank_ic_raw": float(validation_ic_raw),
            "fitness": float(fitness[idx]),
            "complexity": int(_node_count(oriented_tree)),
            "finite_ratio": float(finite_ratios[idx]),
            "max_reference_corr": ref_corr,
        })
        if len(chosen) >= cfg.top_k:
            break

    definitions = [
        E.FactorDefinition(name=row["name"], expr=tree, description="GA-synthesized factor")
        for row, tree in zip(rows, chosen)
    ]
    leaderboard = pd.DataFrame(rows, columns=["name", "expression", "train_rank_ic", "validation_rank_ic", "validation_rank_ic_raw", "fitness", "complexity", "finite_ratio", "max_reference_corr"])
    history = pd.DataFrame(history_rows)
    return SynthesisResult(definitions=definitions, leaderboard=leaderboard, history=history)


def _tournament_pick(
    rng: random.Random,
    population: Sequence[E.Expr],
    fitness: Sequence[float],
    tournament_size: int,
) -> E.Expr:
    indices = rng.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = max(indices, key=lambda i: fitness[i])
    return population[best_idx]


# --------------------------------------------------------------------------- #
# Persistence helpers                                                         #
# --------------------------------------------------------------------------- #


def save_definitions(definitions: Iterable[E.FactorDefinition], path: str | Path) -> Path:
    """Dump discovered factors as JSON (name + repr-of-expression + description)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"name": d.name, "expression": repr(d.expr), "description": d.description}
        for d in definitions
    ]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_result(result: SynthesisResult, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    defs_path = save_definitions(result.definitions, out / "synthesized_definitions.json")
    lb_path = out / "synthesized_leaderboard.parquet"
    try:
        result.leaderboard.to_parquet(lb_path, index=False)
    except Exception:
        lb_path = lb_path.with_suffix(".csv")
        result.leaderboard.to_csv(lb_path, index=False)
    hist_path = out / "synthesized_history.parquet"
    try:
        result.history.to_parquet(hist_path, index=False)
    except Exception:
        hist_path = hist_path.with_suffix(".csv")
        result.history.to_csv(hist_path, index=False)
    paths = {
        "definitions": str(defs_path),
        "leaderboard": str(lb_path),
        "history": str(hist_path),
    }
    if isinstance(result, RDAgentSynthesisResult):
        trace_path = out / "rd_agent_trace.json"
        trace_path.write_text(
            json.dumps(result.trace.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        feedback_path = out / "rd_agent_task_feedback.json"
        feedback_path.write_text(
            json.dumps(result.task_feedback.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["rd_agent_trace"] = str(trace_path)
        paths["rd_agent_task_feedback"] = str(feedback_path)
    return paths


# --------------------------------------------------------------------------- #
# Load discovered factors back into the live feature pipeline                 #
# --------------------------------------------------------------------------- #


_PARSE_NAMESPACE = {
    "Column": E.Column,
    "OptionalColumn": E.OptionalColumn,
    "Constant": E.Constant,
    "Add": E.Add,
    "Sub": E.Sub,
    "Mul": E.Mul,
    "Div": E.Div,
    "Abs": E.Abs,
    "Sign": E.Sign,
    "Log": E.Log,
    "Delay": E.Delay,
    "Delta": E.Delta,
    "Returns": E.Returns,
    "Rank": E.Rank,
    "TsRank": E.TsRank,
    "TsCorr": E.TsCorr,
    "TsCov": E.TsCov,
    "DecayLinear": E.DecayLinear,
    "CsZscore": E.CsZscore,
    "TsMean": E.TsMean,
    "TsStd": E.TsStd,
    "TsSum": E.TsSum,
    "TsMax": E.TsMax,
    "TsMin": E.TsMin,
    "_RollingReduction": E._RollingReduction,
    # OptionalColumn(default=nan) and any nan/inf constants must round-trip:
    # repr() emits bare ``nan``/``inf`` tokens, so they need to resolve here
    # (builtins are stripped from the eval namespace for safety).
    "nan": float("nan"),
    "inf": float("inf"),
}


def parse_expression(expr_repr: str) -> E.Expr:
    """Reconstruct an :class:`Expr` from its ``repr()``.

    Safe because the eval namespace is restricted to the DSL classes above —
    no builtins, no arbitrary import paths.
    """
    return eval(expr_repr, {"__builtins__": {}}, _PARSE_NAMESPACE)  # noqa: S307


def load_definitions(path: str | Path) -> list[E.FactorDefinition]:
    """Load top-K factors saved by :func:`save_definitions`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        E.FactorDefinition(name=item["name"], expr=parse_expression(item["expression"]),
                           description=item.get("description", ""))
        for item in payload
    ]


def compute_synthesized_factors(frame: pd.DataFrame, definitions_path: str | Path) -> pd.DataFrame:
    """Evaluate GA-synthesised factors against ``frame`` and return long format.

    Returns an empty DataFrame if the definitions file is missing or empty so
    callers can chain this safely into feature pipelines.
    """
    path = Path(definitions_path)
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    definitions = load_definitions(path)
    if not definitions:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    rows: list[pd.DataFrame] = []
    base = frame[["symbol", "trade_date"]].copy()
    for definition in definitions:
        try:
            values = pd.to_numeric(definition.expr.evaluate(frame), errors="coerce")
        except Exception:
            continue
        piece = base.copy()
        piece["factor_name"] = definition.name
        piece["factor_value"] = values.replace([np.inf, -np.inf], np.nan).to_numpy()
        rows.append(piece)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    return pd.concat(rows, ignore_index=True)


__all__ = [
    "SymbolicGAConfig",
    "SynthesisResult",
    "RDAgentFactorHypothesis",
    "RDAgentFactorTask",
    "RDAgentFactorLoopConfig",
    "RDAgentSynthesisResult",
    "ProposedFactor",
    "LLMProposalResult",
    "FactorProposer",
    "synthesize_factors_rd_agent",
    "synthesize_factors",
    "save_definitions",
    "save_result",
    "load_definitions",
    "parse_expression",
    "compute_synthesized_factors",
]
