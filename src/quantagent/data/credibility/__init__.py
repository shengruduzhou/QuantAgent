"""Stage 5.2 — news / event source credibility weighting.

A small utility that assigns a 0..1 credibility score to a source name
(news outlet, regulator, social media handle, etc.).  Consumers can
multiply this into sentiment/policy/broker signal strengths to weight
high-credibility sources more heavily and discount social-media noise.

Use the ``apply_credibility_column`` helper to attach a credibility
score to any event frame; or call ``lookup_source_credibility`` for
ad-hoc lookups.
"""

from quantagent.data.credibility.source_table import (
    SOURCE_CREDIBILITY_TABLE,
    SOURCE_TIER_TABLE,
    apply_credibility_column,
    apply_credibility_weight_to_strength,
    lookup_source_credibility,
    lookup_source_tier,
)

__all__ = [
    "SOURCE_CREDIBILITY_TABLE",
    "SOURCE_TIER_TABLE",
    "apply_credibility_column",
    "apply_credibility_weight_to_strength",
    "lookup_source_credibility",
    "lookup_source_tier",
]
