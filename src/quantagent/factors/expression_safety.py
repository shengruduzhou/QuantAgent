from __future__ import annotations

import re

_FUTURE_NAME = re.compile(r"(?:^|[^a-z0-9])(?:forward[_-]?returns?|future[_-]?returns?|future|lead|label|target)(?:[^a-z0-9]|$)", re.IGNORECASE)
_Q_LIB_NEGATIVE_REF = re.compile(r"\b(?:Ref|Shift)\s*\([^,]+,\s*-(?:0*[1-9]\d*)\s*\)", re.IGNORECASE)
_EXPLICIT_LEAD = re.compile(r"\b(?:Lead|Future|LookAhead)\s*\(", re.IGNORECASE)


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


__all__ = ["expression_leakage_reasons", "validate_feature_expression", "validate_feature_expressions"]
