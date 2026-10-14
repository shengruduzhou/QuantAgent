"""LLM factor expressions must not read label-side / next-session columns (round-29 R10-F07)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quantagent.factors import expr as E
from quantagent.factors.factor_synthesis import RDAgentFactorLoopConfig, parse_expression, synthesize_factors_rd_agent
from quantagent.factors.expression_safety import expression_leakage_reasons


class _One:
    def __init__(self, expr, name):
        self.expr, self.name = expr, name
    def propose(self, *, round_idx, hypothesis, rag_directive, memory_digest_payload, n_candidates, seen_expr_reprs):
        from quantagent.factors.factor_synthesis import LLMProposalResult, ProposedFactor, RDAgentFactorHypothesis
        return LLMProposalResult(hypothesis=RDAgentFactorHypothesis(hypothesis="h", reason="r"),
                                 factors=[ProposedFactor(name=self.name, expr=self.expr, description="d",
                                                         formulation="f", hypothesis="h", complexity_tier=1)])


def _panel(n_sym=30, n_days=160, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_sym):
        px = 10 * np.cumprod(1 + rng.normal(0, 0.02, n_days))
        for i, d in enumerate(pd.bdate_range("2023-01-02", periods=n_days)):
            rows.append({"symbol": f"S{j:02d}", "trade_date": d, "open": px[i], "high": px[i] * 1.01,
                         "low": px[i] * 0.99, "close": px[i], "volume": 1e6, "amount": 1e7 * px[i]})
    f = pd.DataFrame(rows).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = f.groupby("symbol")["close"]
    f["entry_close_t1"] = g.shift(-1)
    f["forward_return_5d"] = g.shift(-6) / g.shift(-1) - 1
    return f.dropna(subset=["forward_return_5d"]).reset_index(drop=True)


def test_label_side_column_is_refused_before_it_is_ever_evaluated():
    text = "Rank(expr=Div(numerator=Column(name='entry_close_t1'), denominator=Column(name='close')), method='average', pct=True)"
    expr = parse_expression(text)  # syntactically valid DSL
    from quantagent.factors.expression_safety import post_decision_column_reasons
    assert post_decision_column_reasons(expr) == ("post_decision_column:entry_close_t1",)
    panel = _panel()
    proposer = _One(expr, "llm_entry_close_lead")
    res = synthesize_factors_rd_agent(panel, config=RDAgentFactorLoopConfig(
        rounds=2, factors_per_round=2, top_k=8, label_column="forward_return_5d", validation_fraction=0.25,
        min_validation_rank_ic=0.01, fitness_sample_dates=0, fitness_sample_symbols=0, seed=1, use_llm=True,
        allow_network=False, llm_start_round=1, llm_candidates_per_round=1, max_sota_correlation=1.0,
        tradability_columns=()), proposer=proposer)
    lb = res.leaderboard
    row = lb.loc[lb["name"] == proposer.name]

    # A guard would refuse the expression before it is ever evaluated.
    assert row.empty, "label-side column evaluated as a factor"


def test_column_leaves_are_allowlisted_not_pattern_matched():
    """Round-30: the round-29 name pattern missed fwd_* / next_* spellings.

    A leaf must be OHLCV / a blueprint fundamental or explicitly declared;
    the label-side pattern still applies to declared names.
    """
    from quantagent.factors.expression_safety import post_decision_column_reasons

    def reasons(column, **kwargs):
        expr = parse_expression(f"Rank(expr=Column(name='{column}'), method='average', pct=True)")
        return post_decision_column_reasons(expr, **kwargs)

    assert reasons("fwd_ret_5") == ("post_decision_column:fwd_ret_5",)
    assert reasons("next_open") == ("post_decision_column:next_open",)
    assert "post_decision_column:lead_close" in reasons("lead_close")
    assert reasons("alpha_secret") == ("undeclared_column:alpha_secret",)
    assert reasons("alpha_secret", declared_columns=("alpha_secret",)) == ()
    assert reasons("next_open", declared_columns=("next_open",)) == ("post_decision_column:next_open",)
    for column in ("close", "volume", "amount", "turnover_rate", "pe_ttm", "roe"):
        assert reasons(column) == (), column


def test_llm_proposer_drops_undeclared_columns_and_lists_allowed_ones():
    import json

    from quantagent.factors.llm_factor_proposer import LLMFactorProposer
    from quantagent.factors.factor_synthesis import RDAgentFactorHypothesis

    proposer = LLMFactorProposer(allow_network=False, declared_columns=("northbound_ratio",))
    items = [
        {"name": "a", "expression": "Rank(expr=Column(name='fwd_ret_5'), method='average', pct=True)"},
        {"name": "b", "expression": "Rank(expr=Column(name='mystery'), method='average', pct=True)"},
        {"name": "c", "expression": "Rank(expr=Column(name='northbound_ratio'), method='average', pct=True)"},
    ]
    kept = proposer._parse_factors(items, ())
    assert [factor.name for factor in kept] == ["llm_c"]
    _, user_text = proposer._build_prompt(
        hypothesis=RDAgentFactorHypothesis(hypothesis="h", reason="r"),
        rag_directive="",
        memory_digest_payload={},
        n_candidates=1,
        seen_expr_reprs=(),
    )
    assert "northbound_ratio" in json.loads(user_text)["allowed_columns"]
