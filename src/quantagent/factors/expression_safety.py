from __future__ import annotations

import re

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
    r"(?:^|_)(?:forward|future|label|target|entry|exit)(?:_|$)|_t\+?1(?:_|$)|^mask_|eligible|feasible",
    re.IGNORECASE,
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
    expr: object, *, label_column: str | None = None
) -> tuple[str, ...]:
    """Reasons a factor expression reads data unknown at the decision close.

    The DSL's operators are causal, but a Column leaf can still name a
    label-side field present in the merged panel (``entry_close_t1`` is the
    next session's close) - round-29 R10-F07.
    """
    reasons = [
        f"post_decision_column:{name}"
        for name in expression_column_names(expr)
        if _POST_DECISION_COLUMN.search(name) or (label_column and name == label_column)
    ]
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
    "post_decision_column_reasons",
    "validate_feature_expression",
    "validate_feature_expressions",
]
