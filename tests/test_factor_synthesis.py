"""Validate that the symbolic-GA factor synthesiser actually discovers signal.

We inject a known cross-sectional alpha (5-day momentum) into synthetic OHLCV
and confirm the GA produces at least one factor whose train/validation IC have
the same sign and reach a meaningful magnitude.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantagent.factors import expr as E
from quantagent.factors.factor_synthesis import (
    RDAgentFactorLoopConfig,
    SymbolicGAConfig,
    _evaluate_fitness,
    _evaluate_ic,
    _rank_ic,
    _tradable_mask,
    save_result,
    synthesize_factors,
    synthesize_factors_rd_agent,
)


def _make_panel_with_momentum_signal(
    *,
    n_symbols: int = 6,
    n_days: int = 220,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate OHLCV where forward_return_5d ~ rank(close_t / close_{t-5})."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    closes: dict[str, list[float]] = {}
    for j in range(n_symbols):
        symbol = f"S{j:02d}"
        # Mean-reverting random walk so closes do not blow up.
        path = [10.0 + j]
        for _ in range(1, n_days):
            shock = rng.normal(0.0, 0.02)
            path.append(max(0.5, path[-1] * (1.0 + shock)))
        closes[symbol] = path

    for i, date in enumerate(dates):
        # Compute cross-sectional 5d momentum at i to drive label at i.
        momentum: dict[str, float] = {}
        for symbol, path in closes.items():
            if i >= 5:
                momentum[symbol] = path[i] / path[i - 5] - 1.0
            else:
                momentum[symbol] = 0.0
        if i >= 5:
            ranked = pd.Series(momentum).rank(method="average")
            # forward_return_5d ~ this rank (with small noise)
            normed = (ranked - ranked.mean()) / max(ranked.std(), 1e-9)
            label = {sym: float(normed[sym]) * 0.01 + rng.normal(0.0, 0.003) for sym in closes}
        else:
            label = {sym: 0.0 for sym in closes}
        for symbol, path in closes.items():
            close = path[i]
            volume = 1_000_000.0 + 50_000.0 * (j + 1) + rng.normal(0, 30_000)
            volume = float(max(volume, 1_000.0))
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": close * 0.997,
                    "high": close * 1.012,
                    "low": close * 0.985,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "forward_return_5d": label[symbol],
                }
            )
    return pd.DataFrame(rows)


def test_synthesize_factors_recovers_injected_momentum_signal():
    panel = _make_panel_with_momentum_signal()
    cfg = SymbolicGAConfig(
        population=60,
        generations=12,
        max_depth=3,
        min_depth=2,
        tournament_size=5,
        elitism=4,
        top_k=10,
        label_column="forward_return_5d",
        complexity_penalty=2e-4,
        min_finite_ratio=0.3,
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=400,
        fitness_sample_symbols=20,
        seed=2024,
    )
    result = synthesize_factors(panel, config=cfg)

    assert not result.leaderboard.empty, "GA produced no surviving factors"
    top = result.leaderboard.iloc[0]
    # The injected signal is monotone in 5-day momentum; even a coarse GA
    # should find an expression with at least mild positive IC on both
    # train and validation, and not flip sign between them.
    assert top["train_rank_ic"] >= 0.10, (
        f"top train_rank_ic too low: {top['train_rank_ic']:.4f}"
    )
    assert top["validation_rank_ic"] >= 0.0, (
        f"top validation_rank_ic flipped negative: {top['validation_rank_ic']:.4f}"
    )

    # GA history must show monotone (best ≥ first generation) progress over
    # the run — i.e., evolution is actually doing something.
    assert not result.history.empty
    last_best = float(result.history.iloc[-1]["best_fitness"])
    first_best = float(result.history.iloc[0]["best_fitness"])
    assert last_best >= first_best - 1e-9


def test_rd_agent_synthesis_loop_accumulates_sota_factor_library(tmp_path):
    panel = _make_panel_with_momentum_signal(n_symbols=6, n_days=180)
    cfg = RDAgentFactorLoopConfig(
        rounds=3,
        factors_per_round=4,
        top_k=5,
        label_column="forward_return_5d",
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=2026,
    )

    result = synthesize_factors_rd_agent(panel, config=cfg)

    assert not result.trace.empty
    assert result.trace["candidate_count"].max() <= 4
    assert result.trace["sota_size"].max() == len(result.definitions)
    assert len(result.definitions) > 0
    assert not result.leaderboard.empty
    assert result.leaderboard["factor_implementation"].all()
    assert result.task_feedback["factor_name"].nunique() >= len(result.definitions)

    paths = save_result(result, tmp_path)
    assert "rd_agent_trace" in paths
    assert "rd_agent_task_feedback" in paths
    assert (tmp_path / "rd_agent_trace.json").exists()
    assert (tmp_path / "rd_agent_task_feedback.json").exists()


def test_synthesize_factors_rejects_pure_noise():
    """Random returns should yield few/no factors above the validation gate."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-02", periods=160, freq="B")
    rows: list[dict[str, object]] = []
    for j in range(5):
        sym = f"N{j:02d}"
        close = 10.0
        for date in dates:
            close *= 1.0 + rng.normal(0.0, 0.015)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": date,
                    "open": close * 0.998,
                    "high": close * 1.011,
                    "low": close * 0.986,
                    "close": close,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    # No signal — label is pure noise, uncorrelated with anything.
                    "forward_return_5d": float(rng.normal(0.0, 0.01)),
                }
            )
    panel = pd.DataFrame(rows)
    cfg = SymbolicGAConfig(
        population=40,
        generations=6,
        top_k=10,
        label_column="forward_return_5d",
        validation_fraction=0.3,
        min_validation_rank_ic=0.10,  # demand non-trivial OOS IC
        seed=11,
    )
    result = synthesize_factors(panel, config=cfg)
    # Either nothing survives, or whatever survives has very small magnitude.
    assert (
        result.leaderboard.empty
        or result.leaderboard["validation_rank_ic"].abs().max() < 0.25
    )


def test_synthesize_factors_rejects_constant_only_expression():
    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    labels = panel["forward_return_5d"]
    dates = panel["trade_date"]
    cfg = SymbolicGAConfig(label_column="forward_return_5d")

    fitness, raw_ic, finite_ratio = _evaluate_fitness(
        E.TsStd(E.Rank(E.Constant(2.0)), 5),
        panel,
        labels,
        dates,
        cfg,
    )

    assert fitness == -1.0
    assert raw_ic == 0.0
    assert finite_ratio == 0.0


def test_synthesize_factors_normalizes_trade_date_before_label_merge():
    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    labels = panel[["symbol", "trade_date", "forward_return_5d"]].copy()
    panel = panel.drop(columns=["forward_return_5d"])
    panel["trade_date"] = panel["trade_date"].dt.strftime("%Y-%m-%d")
    cfg = SymbolicGAConfig(
        population=8,
        generations=1,
        top_k=2,
        label_column="forward_return_5d",
        fitness_sample_dates=30,
        fitness_sample_symbols=4,
        seed=99,
    )

    result = synthesize_factors(panel, labels=labels, config=cfg)

    assert result.history is not None


def test_labels_with_overlapping_ohlcv_columns_do_not_kill_population():
    """Regression: full training datasets carry their own OHLCV columns; the
    merge must not suffix the panel's market columns (which made every tree
    raise KeyError and every fitness -1.0)."""
    panel = _make_panel_with_momentum_signal(n_symbols=5, n_days=120)
    labels = panel[["symbol", "trade_date", "forward_return_5d"]].copy()
    # Simulate a training-dataset labels file that also contains OHLCV.
    labels["close"] = 1.0
    labels["volume"] = 2.0
    labels["open"] = 3.0
    panel_no_label = panel.drop(columns=["forward_return_5d"])
    cfg = SymbolicGAConfig(
        population=12,
        generations=1,
        max_depth=3,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=3,
    )
    result = synthesize_factors(panel_no_label, labels=labels, config=cfg)
    assert (result.history["best_fitness"] > -1.0).any()


def test_new_dsl_operators_evaluate_and_roundtrip():
    from quantagent.factors.factor_synthesis import parse_expression

    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=80)
    candidates = [
        E.TsCorr(E.Rank(E.Close), E.Rank(E.Volume), 10),
        E.TsCov(E.Returns(E.Close, 1), E.Returns(E.Volume, 1), 10),
        E.DecayLinear(E.Delta(E.Close, 3), 10),
        E.CsZscore(E.Div(E.Close, E.TsMean(E.Close, 20))),
    ]
    for tree in candidates:
        values = tree.evaluate(panel)
        assert values.notna().sum() > 0, repr(tree)
        rebuilt = parse_expression(repr(tree))
        pd.testing.assert_series_equal(
            rebuilt.evaluate(panel), values, check_names=False
        )


def test_decay_linear_weights_recent_bars_most():
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=10, freq="B")):
        rows.append({"symbol": "S", "trade_date": d, "open": 1.0, "high": 1.0,
                     "low": 1.0, "close": float(i), "volume": 1.0, "amount": 1.0})
    frame = pd.DataFrame(rows)
    out = E.DecayLinear(E.Close, 3).evaluate(frame)
    # window [1,2,3] with ascending weights 1/6,2/6,3/6 over closes 1,2,3 -> (1*1+2*2+3*3)/6
    expected = (1 * 1 + 2 * 2 + 3 * 3) / 6.0
    assert abs(out.iloc[3] - expected) < 1e-9


def test_ts_rank_matches_pandas_reference():
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("A", "B"):
        for i, d in enumerate(pd.date_range("2024-01-01", periods=40, freq="B")):
            rows.append({"symbol": sym, "trade_date": d, "open": 1.0, "high": 1.0,
                         "low": 1.0, "close": float(rng.normal()), "volume": 1.0, "amount": 1.0})
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    fast = E.TsRank(E.Close, 10).evaluate(frame)
    ref = (
        frame.assign(trade_date=pd.to_datetime(frame["trade_date"]))
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol")["close"]
        .rolling(10, min_periods=10)
        .apply(lambda b: pd.Series(b).rank(method="average").iloc[-1] / len(b), raw=True)
        .reset_index(level=0, drop=True)
    )
    aligned = fast.reindex(ref.index)
    mask = ref.notna()
    assert np.allclose(aligned[mask], ref[mask])


def test_optional_column_with_nan_default_round_trips():
    """OptionalColumn(default=nan) reprs must survive save -> parse_expression.

    Regression: the saved repr emits a bare ``nan`` token; parse_expression
    eval-s with builtins stripped, so ``nan`` must resolve in the namespace or
    every discovered valuation/quality factor fails to load.
    """
    from quantagent.factors.factor_synthesis import parse_expression

    expr = E.Mul(
        E.Constant(-1.0),
        E.Rank(E.Mul(E.OptionalColumn("pb"), E.TsMean(E.Volume, 20))),
    )
    text = repr(expr)
    assert "nan" in text  # OptionalColumn default serializes as nan
    assert repr(parse_expression(text)) == text


# --------------------------------------------------------------------------- #
# RD-Agent closed LLM loop (offline via an injected fake proposer)            #
# --------------------------------------------------------------------------- #


class _RecordingProposer:
    """Deterministic offline stand-in for the LLM factor proposer.

    Records every call so the test can prove the loop feeds the proposer the
    accumulated hypothesis, escalating research directive, persisted memory
    digest, and the running set of attempted expressions.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.proposed_names: set[str] = set()

    def propose(
        self,
        *,
        round_idx,
        hypothesis,
        rag_directive,
        memory_digest_payload,
        n_candidates,
        seen_expr_reprs,
    ):
        from quantagent.factors.factor_synthesis import (
            LLMProposalResult,
            ProposedFactor,
            RDAgentFactorHypothesis,
        )

        self.calls.append(
            {
                "round_idx": round_idx,
                "rag_directive": rag_directive,
                "memory_digest": memory_digest_payload,
                "n_candidates": n_candidates,
                "seen_expr_reprs": list(seen_expr_reprs),
            }
        )
        # A structurally-novel but momentum-correlated factor (passes gates) and
        # a low-vol factor; names mirror the real proposer's ``llm_`` prefix.
        factors = [
            ProposedFactor(
                name=f"llm_mean_momentum_r{round_idx}",
                expr=E.Rank(E.TsMean(E.Returns(E.Close, 1), 5)),
                description="mean of 1-day returns over 5 days",
                formulation="Rank(mean(ret1, 5))",
                hypothesis="cross-sectional short momentum",
                complexity_tier=2,
            ),
            ProposedFactor(
                name=f"llm_low_vol_r{round_idx}",
                expr=E.Mul(E.Constant(-1.0), E.Rank(E.TsStd(E.Returns(E.Close, 1), 20))),
                description="negative 20-day return volatility rank",
                formulation="-Rank(std(ret1, 20))",
                hypothesis="low volatility premium",
                complexity_tier=2,
            ),
        ]
        self.proposed_names.update(f.name for f in factors)
        return LLMProposalResult(
            hypothesis=RDAgentFactorHypothesis(
                hypothesis=f"round {round_idx} refined hypothesis",
                reason="fake proposer",
                concise_knowledge="prefer novel momentum-orthogonal structures",
            ),
            factors=factors,
        )


def test_rd_agent_loop_runs_closed_llm_cycle_with_injected_proposer(tmp_path):
    panel = _make_panel_with_momentum_signal(n_symbols=6, n_days=200)
    memory_path = tmp_path / "factor_memory.jsonl"
    proposer = _RecordingProposer()
    cfg = RDAgentFactorLoopConfig(
        rounds=3,
        factors_per_round=4,
        top_k=8,
        label_column="forward_return_5d",
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=2026,
        use_llm=True,
        allow_network=False,  # irrelevant: an explicit proposer is injected
        llm_start_round=1,
        llm_candidates_per_round=2,
        rag_escalation_round=2,
        memory_path=str(memory_path),
        # The synthetic panel's only signal is momentum, which the blueprint
        # round already captures; relax the near-duplicate gate so this test
        # exercises the acceptance/value flow rather than the (separately
        # tested) SOTA dedup behaviour.
        max_sota_correlation=1.0,
    )

    result = synthesize_factors_rd_agent(panel, config=cfg, proposer=proposer)

    sources = list(result.trace["source"])
    # Round 0 is a blueprint warm start; later rounds are LLM-driven.
    assert sources[0] == "blueprint"
    assert "llm" in sources

    # The proposer was actually invoked for rounds >= llm_start_round.
    called_rounds = {c["round_idx"] for c in proposer.calls}
    assert called_rounds == {1, 2}

    # Feedback flows: by the first LLM round the SOTA library already has
    # blueprint acceptances, so the memory digest carries accepted examples and
    # the running set of attempted expressions is non-empty.
    first_llm_call = next(c for c in proposer.calls if c["round_idx"] == 1)
    assert len(first_llm_call["seen_expr_reprs"]) > 0
    assert first_llm_call["memory_digest"]["recent_accepted_examples"]

    # Research directive escalates at rag_escalation_round.
    from quantagent.factors.factor_loop_memory import RAG_EASY, RAG_HIGH_IC

    directive_by_round = {c["round_idx"]: c["rag_directive"] for c in proposer.calls}
    assert directive_by_round[1] == RAG_EASY
    assert directive_by_round[2] == RAG_HIGH_IC

    # An LLM-proposed factor survived the same gates as the blueprints.
    definition_names = {d.name for d in result.definitions}
    assert definition_names & proposer.proposed_names

    # Knowledge persisted to the JSONL memory, including at least one acceptance.
    assert memory_path.exists()
    persisted = [json.loads(line) for line in memory_path.read_text().splitlines() if line.strip()]
    assert persisted
    assert any(row.get("status") == "selected" for row in persisted)
    assert any(row.get("source") == "llm" for row in persisted)


# --------------------------------------------------------------------------- #
# Tradability guard (Phase 0): no phantom edge over untradable names           #
# --------------------------------------------------------------------------- #


def _make_phantom_edge_panel(
    *,
    n_days: int = 80,
    n_tradable: int = 20,
    n_sealed: int = 20,
    seed: int = 17,
) -> pd.DataFrame:
    """Panel where ``Rank(close)`` predicts forward return ONLY on limit-up-sealed
    names; tradable names get a pure-noise label.

    A tradability-blind IC sees a strong factor (the sealed block dominates the
    cross-section); a tradability-aware IC — which drops the limit-up-sealed
    names you cannot actually buy — sees ~nothing. This is exactly the
    phantom-edge mechanism the guard must defeat.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    tradable = [f"T{j:02d}" for j in range(n_tradable)]
    sealed = [f"U{j:02d}" for j in range(n_sealed)]
    rows: list[dict[str, object]] = []
    for date in dates:
        closes = {s: float(rng.uniform(5.0, 50.0)) for s in tradable + sealed}
        sealed_close = pd.Series({s: closes[s] for s in sealed})
        sr = sealed_close.rank()
        sr = (sr - sr.mean()) / max(sr.std(ddof=0), 1e-9)
        for s in tradable + sealed:
            is_up = s in sealed
            label = (
                float(sr[s]) * 0.01 + rng.normal(0.0, 0.0005)
                if is_up
                else float(rng.normal(0.0, 0.01))
            )
            c = closes[s]
            rows.append(
                {
                    "symbol": s,
                    "trade_date": date,
                    "open": c * 0.997,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * c,
                    "is_limit_up": is_up,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "forward_return_5d": label,
                }
            )
    return pd.DataFrame(rows)


def test_tradable_mask_drops_untradable_rows_and_kills_phantom_ic():
    panel = _make_phantom_edge_panel(n_days=60)
    factor = E.Rank(E.Close).evaluate(panel)
    labels = panel["forward_return_5d"]
    dates = panel["trade_date"]

    mask = _tradable_mask(panel, ("is_suspended", "is_limit_up", "is_limit_down"))
    assert mask is not None
    # Exactly the limit-up-sealed names are excluded.
    assert bool((~mask).equals(panel["is_limit_up"].astype(bool))) or int((~mask).sum()) == int(
        panel["is_limit_up"].sum()
    )

    raw_ic = _rank_ic(factor, labels, dates)
    tradable_ic = _rank_ic(factor, labels, dates, mask)
    assert raw_ic > 0.2, f"phantom raw IC should be strong, got {raw_ic:.4f}"
    assert abs(tradable_ic) < 0.1, f"tradable IC should be ~0, got {tradable_ic:.4f}"


def test_tradable_mask_none_when_no_flag_columns():
    panel = _make_panel_with_momentum_signal(n_symbols=4, n_days=40)
    assert _tradable_mask(panel, ("is_suspended", "is_limit_up", "is_limit_down")) is None
    # _evaluate_ic with mask=None must equal today's behaviour (backward compat).
    factor = E.Rank(E.Returns(E.Close, 5))
    ic_a, _ = _evaluate_ic(factor, panel, panel["forward_return_5d"], panel["trade_date"], None)
    ic_b, _ = _evaluate_ic(factor, panel, panel["forward_return_5d"], panel["trade_date"])
    assert ic_a == ic_b


class _SingleFactorProposer:
    """Injected proposer that emits exactly one DSL factor (no network)."""

    def __init__(self, expr, name: str = "llm_phantom_close") -> None:
        self.expr = expr
        self.name = name

    def propose(self, *, round_idx, hypothesis, rag_directive, memory_digest_payload,
                n_candidates, seen_expr_reprs):
        from quantagent.factors.factor_synthesis import (
            LLMProposalResult,
            ProposedFactor,
            RDAgentFactorHypothesis,
        )

        return LLMProposalResult(
            hypothesis=RDAgentFactorHypothesis(hypothesis="phantom probe", reason="test"),
            factors=[
                ProposedFactor(
                    name=self.name,
                    expr=self.expr,
                    description="cross-sectional close level",
                    formulation="Rank(close)",
                    hypothesis="price level",
                    complexity_tier=1,
                )
            ],
        )


def test_rd_agent_loop_rejects_phantom_edge_with_guard_and_accepts_without():
    panel = _make_phantom_edge_panel(n_days=90)
    proposer = _SingleFactorProposer(E.Rank(E.Close))
    base_kwargs = dict(
        rounds=2,
        factors_per_round=3,
        top_k=8,
        label_column="forward_return_5d",
        validation_fraction=0.25,
        min_validation_rank_ic=0.15,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=2026,
        use_llm=True,
        allow_network=False,
        llm_start_round=1,
        llm_candidates_per_round=1,
        max_sota_correlation=1.0,  # isolate the IC gate from the dedup gate
    )

    # Guard ON (default tradability columns present in the panel): the phantom
    # factor's tradable validation IC is ~0, so it is rejected.
    guarded = synthesize_factors_rd_agent(
        panel, config=RDAgentFactorLoopConfig(**base_kwargs), proposer=proposer
    )
    assert proposer.name not in {d.name for d in guarded.definitions}

    # Guard OFF (no tradability columns honoured): the tradability-blind IC is
    # strong, so the very same factor is accepted — proving the guard is what
    # makes the difference, not some unrelated rejection.
    unguarded = synthesize_factors_rd_agent(
        panel,
        config=RDAgentFactorLoopConfig(tradability_columns=(), **base_kwargs),
        proposer=proposer,
    )
    assert proposer.name in {d.name for d in unguarded.definitions}
    row = unguarded.leaderboard.loc[unguarded.leaderboard["name"] == proposer.name].iloc[0]
    assert row["validation_rank_ic_raw"] > 0.2
    # The raw-vs-tradable gap is exactly the phantom edge the guard removes.
    assert row["validation_rank_ic"] >= row["validation_rank_ic_raw"] - 1e-9


# --------------------------------------------------------------------------- #
# Factor-gen quality (Phase 1): ICIR floor, horizon diversity, coverage map    #
# --------------------------------------------------------------------------- #


def test_rd_agent_icir_gate_rejects_below_floor():
    """Raising the validation-ICIR floor above every realized ICIR drops all
    survivors — proving the stability gate is wired into acceptance."""
    panel = _make_panel_with_momentum_signal(n_symbols=8, n_days=200)
    base = dict(
        rounds=2,
        factors_per_round=4,
        top_k=8,
        label_column="forward_return_5d",
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=7,
        max_sota_correlation=1.0,
    )
    loose = synthesize_factors_rd_agent(panel, config=RDAgentFactorLoopConfig(**base))
    assert not loose.leaderboard.empty
    assert "validation_rank_icir" in loose.leaderboard.columns
    realized = float(loose.leaderboard["validation_rank_icir"].max())

    strict = synthesize_factors_rd_agent(
        panel, config=RDAgentFactorLoopConfig(min_validation_icir=realized + 1.0, **base)
    )
    assert strict.leaderboard.empty


def test_horizon_diverse_names_round_robins_buckets():
    from quantagent.factors.factor_synthesis import _horizon_diverse_names

    lb = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d", "e"],
            "horizon": ["short_5d", "short_5d", "short_5d", "long_30d_120d", "mid_5d_30d"],
        }
    )
    picked = _horizon_diverse_names(lb, 3)
    assert picked[0] == "a"  # global best is still picked first
    # ...but diversity is reached immediately rather than filling with b, c.
    assert set(picked) == {"a", "d", "e"}


def test_rd_agent_loop_feeds_coverage_map_to_proposer():
    """The proposer must receive the structured coverage map + uncovered
    structures, not just a flat recent-list."""
    panel = _make_panel_with_momentum_signal(n_symbols=6, n_days=200)
    proposer = _RecordingProposer()
    cfg = RDAgentFactorLoopConfig(
        rounds=3,
        factors_per_round=4,
        top_k=8,
        validation_fraction=0.25,
        min_validation_rank_ic=0.0,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=2026,
        use_llm=True,
        allow_network=False,
        llm_start_round=1,
        llm_candidates_per_round=2,
        max_sota_correlation=1.0,
    )
    result = synthesize_factors_rd_agent(panel, config=cfg, proposer=proposer)

    first_llm_call = next(c for c in proposer.calls if c["round_idx"] == 1)
    digest = first_llm_call["memory_digest"]
    assert "coverage_map" in digest and "cells" in digest["coverage_map"]
    assert "uncovered_directions" in digest
    # Blueprints accepted in round 0 must show up as covered cells with a structure.
    assert any(cell["accepted"] > 0 for cell in digest["coverage_map"]["cells"])

    # Accepted factors carry a structure + horizon tag on the leaderboard.
    assert "structure" in result.leaderboard.columns
    assert "horizon" in result.leaderboard.columns
    assert result.leaderboard["horizon"].notna().all()


def test_rd_agent_loop_default_path_never_calls_proposer(tmp_path):
    panel = _make_panel_with_momentum_signal(n_symbols=6, n_days=180)
    proposer = _RecordingProposer()
    cfg = RDAgentFactorLoopConfig(
        rounds=2,
        factors_per_round=4,
        top_k=5,
        fitness_sample_dates=0,
        fitness_sample_symbols=0,
        seed=2026,
        use_llm=False,  # default: blueprint-only screener, proposer ignored
    )

    result = synthesize_factors_rd_agent(panel, config=cfg, proposer=proposer)

    assert proposer.calls == []
    assert set(result.trace["source"]) == {"blueprint"}
