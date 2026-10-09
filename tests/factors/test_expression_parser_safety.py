"""LLM factor text must be parsed, never executed, and must not encode a lead (round-29 R3-F05)."""

import numpy as np
import pandas as pd
import pytest

from quantagent.factors import expr as E
from quantagent.factors.expression_safety import expression_leakage_reasons
from quantagent.factors.factor_synthesis import parse_expression


def _frame():
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame({"symbol": ["A"] * 6, "trade_date": dates,
                         "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})


@pytest.mark.parametrize(
    "text",
    [
        "Delay(expr=Column(name='close'), periods=-1)",
        "Delta(expr=Column(name='close'), periods=0)",
        "Returns(expr=Column(name='close'), periods=-5)",
        "_RollingReduction(expr=Column(name='close'), window=-3, op='mean')",
        "TsCorr(left=Column(name='close'), right=Column(name='volume'), window=0)",
    ],
)
def test_non_positive_lag_or_window_is_rejected(text):
    with pytest.raises(ValueError, match=">= 1"):
        parse_expression(text)
    assert expression_leakage_reasons(text)


@pytest.mark.parametrize(
    "payload",
    [
        "().__class__.__base__.__subclasses__()",
        "__import__('os').system('true')",
        "Column(name='close').__class__",
        "Rank(expr=[x for x in ()])",
        "Column(*['close'])",
    ],
)
def test_parser_refuses_anything_outside_the_dsl(payload):
    with pytest.raises((ValueError, SyntaxError)):
        parse_expression(payload)


def test_dsl_constructors_refuse_a_lead_directly():
    with pytest.raises(ValueError):
        E.Delay(E.Column("close"), -1)


def test_valid_expression_round_trips_and_stays_causal():
    expr = E.Rank(E.TsMean(E.Returns(E.Column("close"), 1), 3))
    rebuilt = parse_expression(repr(expr))
    assert repr(rebuilt) == repr(expr)
    lagged = parse_expression("Delay(expr=Column(name='close'), periods=1)").evaluate(_frame())
    assert np.isnan(lagged.iloc[0]) and lagged.iloc[1:].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_nan_constants_round_trip():
    expr = parse_expression("OptionalColumn(name='pe', default=nan)")
    assert isinstance(expr, E.OptionalColumn)
