"""Theme-lifecycle-driven trading rules.

The :class:`ThemeProfile` reports a lifecycle stage. This module turns
that stage into concrete trading constraints:

* ``max_position_weight`` — per-theme cap on individual names.
* ``max_total_weight`` — total exposure cap for the theme.
* ``allowed_actions`` — opening / adding / trimming / exit permissions.
* ``required_evidence`` — which evidence types are needed to enter.
* ``exit_triggers`` — conditions that force a reduction or full exit.

This is the rule layer the user requested:

    POLICY_SEED           : only watchlist
    NARRATIVE_FORMATION   : small position, theme tilt
    CAPITAL_INFLOW        : medium, technical timing
    FUNDAMENTAL_VALIDATION: larger, fundamentals heavy
    EARNINGS_REALIZATION  : core long
    VALUATION_BUBBLE      : trim, take profit, hedge
    DECAY / INVALIDATED   : no new opens, exit only
"""

from __future__ import annotations

from dataclasses import dataclass

from quantagent.v7.schemas import ThemeLifecycleStage


@dataclass(frozen=True)
class LifecycleTradingRule:
    stage: ThemeLifecycleStage
    max_position_weight: float
    max_total_theme_weight: float
    allow_open: bool
    allow_add: bool
    require_trim: bool
    require_full_exit: bool
    required_evidence: tuple[str, ...]
    exit_triggers: tuple[str, ...]
    recommended_sleeve: str
    rationale: str


_RULES: dict[ThemeLifecycleStage, LifecycleTradingRule] = {
    ThemeLifecycleStage.POLICY_SEED: LifecycleTradingRule(
        stage=ThemeLifecycleStage.POLICY_SEED,
        max_position_weight=0.0,
        max_total_theme_weight=0.0,
        allow_open=False,
        allow_add=False,
        require_trim=False,
        require_full_exit=False,
        required_evidence=("policy", "industry_chain"),
        exit_triggers=(),
        recommended_sleeve="watchlist",
        rationale="Theme is still policy seed; only watch and gather evidence",
    ),
    ThemeLifecycleStage.NARRATIVE_FORMATION: LifecycleTradingRule(
        stage=ThemeLifecycleStage.NARRATIVE_FORMATION,
        max_position_weight=0.02,
        max_total_theme_weight=0.10,
        allow_open=True,
        allow_add=False,
        require_trim=False,
        require_full_exit=False,
        required_evidence=("policy", "news"),
        exit_triggers=("contradicting_policy", "narrative_breakdown"),
        recommended_sleeve="medium_theme",
        rationale="Narrative is forming; only small theme tilt allowed",
    ),
    ThemeLifecycleStage.CAPITAL_INFLOW: LifecycleTradingRule(
        stage=ThemeLifecycleStage.CAPITAL_INFLOW,
        max_position_weight=0.04,
        max_total_theme_weight=0.20,
        allow_open=True,
        allow_add=True,
        require_trim=False,
        require_full_exit=False,
        required_evidence=("policy", "news", "market_flow"),
        exit_triggers=("flow_reversal", "breadth_collapse"),
        recommended_sleeve="medium_theme",
        rationale="Capital inflow phase; use technical timing for entries",
    ),
    ThemeLifecycleStage.FUNDAMENTAL_VALIDATION: LifecycleTradingRule(
        stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        max_position_weight=0.06,
        max_total_theme_weight=0.30,
        allow_open=True,
        allow_add=True,
        require_trim=False,
        require_full_exit=False,
        required_evidence=("policy", "earnings_growth", "order_confirmed"),
        exit_triggers=("earnings_miss", "audit_opinion_downgrade", "regulatory_penalty"),
        recommended_sleeve="long_fundamental",
        rationale="Fundamentals confirm the theme; tilt toward long fundamental sleeve",
    ),
    ThemeLifecycleStage.EARNINGS_REALIZATION: LifecycleTradingRule(
        stage=ThemeLifecycleStage.EARNINGS_REALIZATION,
        max_position_weight=0.08,
        max_total_theme_weight=0.35,
        allow_open=True,
        allow_add=True,
        require_trim=False,
        require_full_exit=False,
        required_evidence=("earnings_growth", "order_confirmed", "policy"),
        exit_triggers=("earnings_miss", "valuation_overshoot", "demand_inflection"),
        recommended_sleeve="long_fundamental",
        rationale="Earnings are realised; theme is core long",
    ),
    ThemeLifecycleStage.VALUATION_BUBBLE: LifecycleTradingRule(
        stage=ThemeLifecycleStage.VALUATION_BUBBLE,
        max_position_weight=0.03,
        max_total_theme_weight=0.15,
        allow_open=False,
        allow_add=False,
        require_trim=True,
        require_full_exit=False,
        required_evidence=("earnings_growth",),
        exit_triggers=("valuation_overshoot", "crowding_peak", "breadth_divergence"),
        recommended_sleeve="hedge",
        rationale="Valuation bubble; trim, take profit, raise hedge",
    ),
    ThemeLifecycleStage.DIVERGENCE: LifecycleTradingRule(
        stage=ThemeLifecycleStage.DIVERGENCE,
        max_position_weight=0.04,
        max_total_theme_weight=0.20,
        allow_open=False,
        allow_add=False,
        require_trim=True,
        require_full_exit=False,
        required_evidence=("earnings_growth",),
        exit_triggers=("leader_breakdown", "false_breakout"),
        recommended_sleeve="medium_theme",
        rationale="Leaders diverging from followers; hold only the strongest names",
    ),
    ThemeLifecycleStage.DECAY: LifecycleTradingRule(
        stage=ThemeLifecycleStage.DECAY,
        max_position_weight=0.01,
        max_total_theme_weight=0.05,
        allow_open=False,
        allow_add=False,
        require_trim=True,
        require_full_exit=False,
        required_evidence=(),
        exit_triggers=("trend_break", "fundamental_decay"),
        recommended_sleeve="cash_buffer",
        rationale="Theme is decaying; reduce exposure",
    ),
    ThemeLifecycleStage.INVALIDATED: LifecycleTradingRule(
        stage=ThemeLifecycleStage.INVALIDATED,
        max_position_weight=0.0,
        max_total_theme_weight=0.0,
        allow_open=False,
        allow_add=False,
        require_trim=False,
        require_full_exit=True,
        required_evidence=(),
        exit_triggers=("invalidation_confirmed",),
        recommended_sleeve="cash_buffer",
        rationale="Theme has been invalidated; exit fully",
    ),
}


def lifecycle_rule(stage: ThemeLifecycleStage) -> LifecycleTradingRule:
    return _RULES.get(stage, _RULES[ThemeLifecycleStage.NARRATIVE_FORMATION])


def apply_lifecycle_caps(
    target_weights: dict[str, float],
    member_lifecycle: dict[str, ThemeLifecycleStage],
    member_theme: dict[str, str],
) -> tuple[dict[str, float], list[str]]:
    """Apply per-stage caps to a target_weights dict.

    Returns the (possibly trimmed) weights and a list of human-readable
    rationale strings explaining every adjustment.
    """

    notes: list[str] = []
    out = dict(target_weights)
    # First pass: enforce per-name cap and explicit full exit
    for symbol, weight in list(out.items()):
        stage = member_lifecycle.get(symbol)
        if stage is None:
            continue
        rule = lifecycle_rule(stage)
        if rule.require_full_exit or not rule.allow_open and not rule.allow_add and weight > 0:
            if rule.require_full_exit:
                out[symbol] = 0.0
                notes.append(f"{symbol}:full_exit_due_to_{stage.value}")
                continue
        if weight > rule.max_position_weight:
            out[symbol] = rule.max_position_weight
            notes.append(f"{symbol}:capped_at_{rule.max_position_weight:.3f}_for_{stage.value}")
    # Second pass: per-theme total cap
    theme_totals: dict[str, float] = {}
    for symbol, weight in out.items():
        theme = member_theme.get(symbol)
        if theme is None:
            continue
        theme_totals[theme] = theme_totals.get(theme, 0.0) + max(0.0, weight)
    for theme, total in theme_totals.items():
        members = [s for s in out if member_theme.get(s) == theme]
        if not members:
            continue
        # Pick the tightest cap among stages of this theme's members
        cap = min(
            (lifecycle_rule(member_lifecycle[s]).max_total_theme_weight for s in members if s in member_lifecycle),
            default=1.0,
        )
        if total <= cap or cap <= 0.0:
            if cap <= 0.0:
                for symbol in members:
                    if out.get(symbol, 0.0) > 0:
                        out[symbol] = 0.0
                        notes.append(f"{symbol}:zeroed_for_theme_{theme}_cap_0")
            continue
        scale = cap / total
        for symbol in members:
            out[symbol] = float(max(0.0, out.get(symbol, 0.0) * scale))
        notes.append(f"theme_{theme}:scaled_by_{scale:.3f}_to_cap_{cap:.3f}")
    return out, notes
