from __future__ import annotations

import re
from collections.abc import Iterable

_FUTURE_NAME = re.compile(r"(?:^|[^a-z0-9])(?:forward[_-]?returns?|future[_-]?returns?|future|lead|label|target)(?:[^a-z0-9]|$)", re.IGNORECASE)
_Q_LIB_NEGATIVE_REF = re.compile(r"\b(?:Ref|Shift)\s*\([^,]+,\s*-(?:0*[1-9]\d*)\s*\)", re.IGNORECASE)
_EXPLICIT_LEAD = re.compile(r"\b(?:Lead|Future|LookAhead)\s*\(", re.IGNORECASE)
# QuantAgent DSL keyword form, e.g. ``Delay(expr=..., periods=-1)``.
_DSL_NON_POSITIVE_LAG = re.compile(r"\b(?:periods|window)\s*=\s*(?:-\s*\d|0+(?:\.0*)?\s*[,)])")


def expression_leakage_reasons(expression: str) -> tuple[str, ...]:
    """Return fail-closed reasons for a formulaic-alpha feature expression."""
    text = str(expression).strip()
    if not text:
        return ("empty_expression",)
    reasons: list[str] = []
    if _FUTURE_NAME.search(text):
        reasons.append("future_or_label_token")
    if _Q_LIB_NEGATIVE_REF.search(text):
        reasons.append("negative_ref_is_future")
    if _EXPLICIT_LEAD.search(text):
        reasons.append("explicit_lead_operator")
    if _DSL_NON_POSITIVE_LAG.search(text):
        reasons.append("non_positive_lag_or_window")
    return tuple(dict.fromkeys(reasons))


#: Column names that hold information from AFTER the decision close: label
#: windows, next-session entry/exit prices, eligibility masks built from them.
_POST_DECISION_COLUMN = re.compile(
    r"(?:^|_)(?:forward|fwd|future|next|lead|label|target|entry|exit)(?:_|$)"
    r"|_t\+?1(?:_|$)|^mask_|eligible|feasible",
    re.IGNORECASE,
)

def is_post_decision_column(name: str) -> bool:
    """True when a column name denotes information from after the decision close.

    Shared by the factor DSL and the training-dataset builder so the two cannot
    drift apart: a panel column such as ``entry_close_t1`` or
    ``forward_return_20d`` must never become either a factor leaf or a model
    feature.
    """
    return bool(_POST_DECISION_COLUMN.search(str(name)))


#: Inputs a synthesized factor may read without being declared: daily OHLCV
#: plus the point-in-time fundamentals the built-in blueprints use. A name
#: pattern can only reject spellings someone thought of (round 29 missed
#: ``fwd_*`` / ``next_*``), so leaves are checked against this allowlist and
#: anything else must be declared by the caller.
PIT_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "open", "high", "low", "close", "volume", "amount",
        "turnover_rate", "pb", "pe_ttm", "revenue", "roe", "operating_cash_flow",
    }
)


def expression_column_names(expr: object) -> tuple[str, ...]:
    """Every column a DSL expression reads (Column / OptionalColumn leaves)."""
    import dataclasses

    names: list[str] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if type(node).__name__ in ("Column", "OptionalColumn") and hasattr(node, "name"):
            names.append(str(node.name))
        if dataclasses.is_dataclass(node):
            stack.extend(getattr(node, field.name) for field in dataclasses.fields(node))
    return tuple(dict.fromkeys(names))


def post_decision_column_reasons(
    expr: object,
    *,
    label_column: str | None = None,
    declared_columns: Iterable[str] = (),
) -> tuple[str, ...]:
    """Reasons a factor expression reads data unknown at the decision close.

    The DSL's operators are causal, but a Column leaf can still name a
    label-side field present in the merged panel (``entry_close_t1`` is the
    next session's close) - round-29 R10-F07. A leaf must be in
    :data:`PIT_INPUT_COLUMNS` or ``declared_columns``; the name pattern is a
    second layer that also refuses a label-side name someone declared.
    """
    allowed = PIT_INPUT_COLUMNS | frozenset(str(name) for name in declared_columns)
    reasons: list[str] = []
    for name in expression_column_names(expr):
        if _POST_DECISION_COLUMN.search(name) or (label_column and name == label_column):
            reasons.append(f"post_decision_column:{name}")
        elif name not in allowed:
            reasons.append(f"undeclared_column:{name}")
    reasons.extend(expression_leakage_reasons(repr(expr)))
    return tuple(dict.fromkeys(reasons))


def validate_feature_expression(expression: str) -> str:
    reasons = expression_leakage_reasons(expression)
    if reasons:
        raise ValueError(
            "formulaic alpha expression is not PIT-safe for a feature: "
            f"{expression!r}; reasons={list(reasons)}"
        )
    return expression


def validate_feature_expressions(expressions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(validate_feature_expression(item) for item in expressions)


__all__ = [
    "expression_column_names",
    "expression_leakage_reasons",
    "is_post_decision_column",
    "post_decision_column_reasons",
    "validate_feature_expression",
    "validate_feature_expressions",
]
