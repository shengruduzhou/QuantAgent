"""Lineage must be traceable, immutable and content-derived.

The UI drill-down the phase requires — strategy -> signal -> target -> intent ->
risk decision -> order -> events -> fills -> position -> cash -> NAV — is only
possible if every object carries an unbroken chain. These pin the properties
that make the chain worth trusting.
"""

from __future__ import annotations

import pytest

from quantagent.domain.lineage import LINEAGE_FIELDS, Lineage, content_id


def test_the_chain_covers_every_identifier_the_phase_requires():
    required = {
        "research_id", "experiment_id", "strategy_id", "strategy_version_id",
        "model_version_id", "run_id", "signal_id", "order_intent_id", "order_id",
        "parent_order_id", "broker_order_id", "execution_id", "position_lot_id",
        "risk_decision_id",
    }
    assert required == set(LINEAGE_FIELDS)


def test_identifiers_are_content_derived_not_random():
    """Replay depends on this: the same inputs must yield the same id."""
    first = content_id("sig", strategy="alpha", symbol="600000.SH", date="2026-08-03")
    second = content_id("sig", symbol="600000.SH", date="2026-08-03", strategy="alpha")

    assert first == second, "key order must not change the identity"
    assert first.startswith("sig_")

    different = content_id("sig", strategy="alpha", symbol="600000.SH", date="2026-08-04")
    assert first != different


def test_deriving_a_child_preserves_the_parents_chain():
    strategy = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_3")
    signal = strategy.derive(signal_id="sig_1")
    intent = signal.derive(order_intent_id="oi_1")
    order = intent.derive(order_id="ord_1")

    assert order.research_id == "res_1"
    assert order.strategy_version_id == "sv_3"
    assert order.signal_id == "sig_1"
    assert order.order_intent_id == "oi_1"


def test_lineage_is_immutable():
    original = Lineage(strategy_id="str_1")
    derived = original.derive(order_id="ord_1")

    assert original.order_id is None, "deriving must not mutate the parent"
    assert derived.strategy_id == "str_1"
    with pytest.raises(AttributeError):
        original.strategy_id = "str_2"  # type: ignore[misc]


def test_an_unknown_field_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unknown lineage fields"):
        Lineage().derive(not_a_real_id="x")


def test_ancestry_detects_a_chain_that_does_not_belong():
    intent = Lineage(strategy_id="str_1", signal_id="sig_1", order_intent_id="oi_1")
    own_fill = intent.derive(order_id="ord_1", execution_id="ex_1")
    foreign_fill = Lineage(strategy_id="str_1", signal_id="sig_2", order_intent_id="oi_9",
                           order_id="ord_9", execution_id="ex_9")

    assert intent.is_ancestor_of(own_fill)
    assert not intent.is_ancestor_of(foreign_fill)


def test_describe_walks_the_chain_broadest_first():
    chain = Lineage(research_id="res_1", strategy_id="str_1", order_id="ord_1")

    described = chain.describe()

    assert described.index("research_id") < described.index("strategy_id")
    assert described.index("strategy_id") < described.index("order_id")


def test_round_trips_through_a_mapping():
    original = Lineage(research_id="res_1", signal_id="sig_1", order_id="ord_1")

    restored = Lineage.from_mapping(original.as_dict())

    assert restored == original
    assert Lineage.from_mapping(None) == Lineage()
    assert original.depth == 3
