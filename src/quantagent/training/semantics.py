"""Lightweight canonical semantics identifiers for governed model artifacts.

Keep these constants free of optional ML framework imports so execution-time
trust verification can compare artifact semantics without importing PyTorch or
instantiating any training code.
"""

FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION = (
    "ft_transformer_objective_v2_per_date_listwise_validation"
)


__all__ = ["FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION"]
