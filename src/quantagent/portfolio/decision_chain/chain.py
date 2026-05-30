"""Core implementation of the 14-step decision chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gate result + ordering
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    gate_name: str
    passed: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "reason": self.reason,
            "detail": self.detail,
        }


GATE_ORDER: tuple[str, ...] = (
    "alpha_threshold",
    "liquidity",
    "tradeable_today",
    "price_limit_block",
    "st_status",
    "sector_pool",
    "hard_market_gate",
    "regime_alignment",
    "fundamental_filter",
    "policy_aligned",
    "broker_consensus",
    "drawdown_kill",
    "concentration_limit",
    "risk_budget",
    "gross_exposure_budget",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionChainConfig:
    # Gate thresholds
    min_alpha: float = 0.0  # raw alpha; usually overridden externally with rank
    min_avg_amount_cny: float = 50_000_000.0  # ≥ 5000 万日均额
    block_limit_up_pct: float = 0.095   # limit-up if same-day ret >= this
    block_limit_down_pct: float = -0.095
    allow_st: bool = False
    excluded_sector_tiers: tuple[str, ...] = ("excluded",)
    regime_setup_compatibility: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "normal": ("lowbuy", "breakout"),
            "caution": ("lowbuy",),
            "bear": ("lowbuy",),
            "crisis": (),  # nothing trades in crisis
        }
    )
    fundamental_rank_min_pct: float = 0.30  # ≥ 30% percentile
    policy_signal_min: float = -0.20  # block when policy_signal < this for sector
    broker_consensus_min: float = -0.30
    stock_drawdown_kill_pct: float = -0.20
    max_sector_weight: float = 0.30
    max_name_weight: float = 0.03
    # Spec section 7 — total-position budget. Default cap 60%; only
    # extendable to ``high_conviction_cap`` when global conviction
    # clears the threshold AND the market regime is friendly.
    default_gross_exposure_cap: float = 0.60
    high_conviction_gross_exposure_cap: float = 0.80
    high_conviction_threshold: float = 0.80
    high_conviction_friendly_regimes: tuple[str, ...] = ("normal", "bull")
    # Gate enablement (set False to skip a gate entirely)
    enabled_gates: tuple[str, ...] = GATE_ORDER


# ---------------------------------------------------------------------------
# Candidate + context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    trade_date: pd.Timestamp
    symbol: str
    alpha_score: float
    setup_label: str | None = None  # e.g. "lowbuy" / "breakout"
    target_weight: float = 0.0  # planned weight if eligible


@dataclass
class DecisionContext:
    """All optional inputs the gates may consult. Missing inputs cause the
    associated gate to PASS (we don't reject for lack of data, only for
    confirmed-bad signals).
    """

    market_panel: pd.DataFrame | None = None       # cols: trade_date, symbol, amount, suspension, daily_return
    st_flags: pd.DataFrame | None = None           # cols: trade_date, symbol, is_st
    sector_map: pd.DataFrame | None = None         # cols: symbol, sector_level_1
    sector_pool: pd.DataFrame | None = None        # cols: sector_level_1, pool_tier
    hard_gate_frame: pd.DataFrame | None = None    # cols: trade_date, hard_gate_active
    regime_state: pd.Series | None = None          # index trade_date, value = regime tag
    fundamental_ranker: pd.DataFrame | None = None  # cols: symbol, composite_rank, as_of_date
    policy_signal_by_sector: pd.DataFrame | None = None  # cols: trade_date, sector, policy_signal
    broker_consensus: pd.DataFrame | None = None   # cols: trade_date, symbol, broker_consensus_score
    stock_drawdown: pd.DataFrame | None = None     # cols: trade_date, symbol, dd_20d
    current_weights: dict[str, float] = field(default_factory=dict)   # symbol → weight
    sector_weights: dict[str, float] = field(default_factory=dict)    # sector → weight
    # Spec section 7 — global gross exposure context. The caller sums
    # the current portfolio's |w| into ``current_gross_exposure``; the
    # global conviction is the optimiser's confidence in *today's*
    # positioning. The gate uses these to enforce 60%/80% caps.
    current_gross_exposure: float = 0.0
    global_conviction: float = 0.0


@dataclass
class DecisionTrace:
    candidate_id: str
    trade_date: pd.Timestamp
    symbol: str
    final_decision: str    # "eligible" or "rejected"
    failed_gate: str | None
    gate_results: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trade_date": self.trade_date.isoformat() if isinstance(self.trade_date, pd.Timestamp) else str(self.trade_date),
            "symbol": self.symbol,
            "final_decision": self.final_decision,
            "failed_gate": self.failed_gate,
            "gate_results": [g.to_dict() for g in self.gate_results],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_id(candidate: Candidate) -> str:
    dt = candidate.trade_date.isoformat() if isinstance(candidate.trade_date, pd.Timestamp) else str(candidate.trade_date)
    return f"{dt}|{candidate.symbol}"


def _lookup_market_row(
    market_panel: pd.DataFrame,
    trade_date: pd.Timestamp,
    symbol: str,
) -> pd.Series | None:
    mask = (market_panel["trade_date"] == trade_date) & (market_panel["symbol"] == symbol)
    rows = market_panel[mask]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _lookup_sector(sector_map: pd.DataFrame | None, symbol: str) -> str | None:
    if sector_map is None or sector_map.empty:
        return None
    rows = sector_map[sector_map["symbol"] == symbol]
    if rows.empty:
        return None
    return str(rows.iloc[-1].get("sector_level_1"))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _gate_alpha_threshold(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if candidate.alpha_score < config.min_alpha:
        return GateResult(
            "alpha_threshold", False,
            f"alpha_{candidate.alpha_score:.4f}_below_{config.min_alpha:.4f}",
            {"alpha": candidate.alpha_score},
        )
    return GateResult("alpha_threshold", True, "passed", {"alpha": candidate.alpha_score})


def _gate_liquidity(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if context.market_panel is None or context.market_panel.empty:
        return GateResult("liquidity", True, "skipped_no_panel")
    row = _lookup_market_row(context.market_panel, candidate.trade_date, candidate.symbol)
    if row is None or "amount" not in row.index:
        return GateResult("liquidity", True, "skipped_no_row")
    amount = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
    if amount < config.min_avg_amount_cny:
        return GateResult(
            "liquidity", False,
            f"amount_{amount:.0f}_below_{config.min_avg_amount_cny:.0f}",
            {"amount_cny": amount},
        )
    return GateResult("liquidity", True, "passed", {"amount_cny": amount})


def _gate_tradeable_today(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if context.market_panel is None or context.market_panel.empty:
        return GateResult("tradeable_today", True, "skipped_no_panel")
    row = _lookup_market_row(context.market_panel, candidate.trade_date, candidate.symbol)
    if row is None:
        return GateResult("tradeable_today", True, "skipped_no_row")
    if "suspension" in row.index and bool(row["suspension"]):
        return GateResult("tradeable_today", False, "suspended", {"suspension": True})
    return GateResult("tradeable_today", True, "passed")


def _gate_price_limit_block(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if context.market_panel is None or context.market_panel.empty:
        return GateResult("price_limit_block", True, "skipped_no_panel")
    row = _lookup_market_row(context.market_panel, candidate.trade_date, candidate.symbol)
    if row is None or "daily_return" not in row.index:
        return GateResult("price_limit_block", True, "skipped_no_row")
    ret = float(row["daily_return"]) if pd.notna(row["daily_return"]) else 0.0
    if ret >= config.block_limit_up_pct:
        return GateResult("price_limit_block", False, "at_limit_up", {"daily_return": ret})
    if ret <= config.block_limit_down_pct:
        return GateResult("price_limit_block", False, "at_limit_down", {"daily_return": ret})
    return GateResult("price_limit_block", True, "passed", {"daily_return": ret})


def _gate_st_status(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if context.st_flags is None or context.st_flags.empty:
        return GateResult("st_status", True, "skipped_no_st_flags")
    flags = context.st_flags
    mask = (flags["trade_date"] == candidate.trade_date) & (flags["symbol"] == candidate.symbol)
    rows = flags[mask]
    if rows.empty and "symbol" in flags.columns:
        # Try symbol-only fallback (some ST tables are not date-keyed)
        rows = flags[flags["symbol"] == candidate.symbol]
    if rows.empty:
        return GateResult("st_status", True, "skipped_no_match")
    is_st = bool(rows.iloc[-1].get("is_st", False))
    if is_st and not config.allow_st:
        return GateResult("st_status", False, "is_st", {"is_st": True})
    return GateResult("st_status", True, "passed", {"is_st": is_st})


def _gate_sector_pool(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    sector = _lookup_sector(context.sector_map, candidate.symbol)
    if sector is None:
        return GateResult("sector_pool", True, "skipped_no_sector_for_symbol")
    pool = context.sector_pool
    if pool is None or pool.empty or "pool_tier" not in pool.columns:
        return GateResult("sector_pool", True, "skipped_no_pool")
    rows = pool[pool["sector_level_1"] == sector]
    if rows.empty:
        return GateResult("sector_pool", True, "skipped_sector_not_in_pool", {"sector": sector})
    tier = str(rows.iloc[-1]["pool_tier"])
    if tier in config.excluded_sector_tiers:
        return GateResult(
            "sector_pool", False, f"sector_tier_{tier}",
            {"sector": sector, "tier": tier},
        )
    return GateResult("sector_pool", True, "passed", {"sector": sector, "tier": tier})


def _gate_hard_market_gate(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    frame = context.hard_gate_frame
    if frame is None or frame.empty:
        return GateResult("hard_market_gate", True, "skipped_no_hard_gate")
    series = frame.set_index("trade_date")["hard_gate_active"]
    if candidate.trade_date not in series.index:
        # Fall back to nearest backward
        try:
            active = bool(series.reindex([candidate.trade_date], method="ffill").iloc[0])
        except (KeyError, IndexError):
            return GateResult("hard_market_gate", True, "skipped_no_date")
    else:
        active = bool(series.loc[candidate.trade_date])
    if active:
        return GateResult("hard_market_gate", False, "hard_gate_active")
    return GateResult("hard_market_gate", True, "passed")


def _gate_regime_alignment(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if context.regime_state is None or len(context.regime_state) == 0 or candidate.setup_label is None:
        return GateResult("regime_alignment", True, "skipped_no_regime_or_setup")
    regime = str(
        context.regime_state.reindex([candidate.trade_date], method="ffill").iloc[0]
        if candidate.trade_date in context.regime_state.index or len(context.regime_state) > 0
        else "normal"
    )
    compatible = config.regime_setup_compatibility.get(regime, ())
    if compatible == ():
        return GateResult(
            "regime_alignment", False, f"regime_{regime}_blocks_all",
            {"regime": regime},
        )
    if candidate.setup_label not in compatible:
        return GateResult(
            "regime_alignment", False,
            f"setup_{candidate.setup_label}_not_compatible_with_{regime}",
            {"regime": regime, "setup": candidate.setup_label},
        )
    return GateResult(
        "regime_alignment", True, "passed",
        {"regime": regime, "setup": candidate.setup_label},
    )


def _gate_fundamental_filter(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    fr = context.fundamental_ranker
    if fr is None or fr.empty or "composite_rank" not in fr.columns:
        return GateResult("fundamental_filter", True, "skipped_no_ranker")
    rows = fr[fr["symbol"] == candidate.symbol]
    if rows.empty:
        return GateResult("fundamental_filter", True, "skipped_no_match")
    rank = float(rows.iloc[-1]["composite_rank"])
    if rank < config.fundamental_rank_min_pct:
        return GateResult(
            "fundamental_filter", False,
            f"rank_{rank:.3f}_below_{config.fundamental_rank_min_pct:.3f}",
            {"composite_rank": rank},
        )
    return GateResult("fundamental_filter", True, "passed", {"composite_rank": rank})


def _gate_policy_aligned(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    ps = context.policy_signal_by_sector
    if ps is None or ps.empty:
        return GateResult("policy_aligned", True, "skipped_no_policy")
    sector = _lookup_sector(context.sector_map, candidate.symbol)
    if sector is None:
        return GateResult("policy_aligned", True, "skipped_no_sector_for_symbol")
    mask = (ps["trade_date"] == candidate.trade_date) & (ps["sector"] == sector)
    rows = ps[mask]
    if rows.empty:
        # Fall back to most-recent prior signal
        prior = ps[(ps["sector"] == sector) & (ps["trade_date"] <= candidate.trade_date)]
        if prior.empty:
            return GateResult("policy_aligned", True, "skipped_no_signal_for_sector_date")
        rows = prior.tail(1)
    signal = float(rows.iloc[-1].get("policy_signal", 0.0))
    if signal < config.policy_signal_min:
        return GateResult(
            "policy_aligned", False,
            f"signal_{signal:.3f}_below_{config.policy_signal_min:.3f}",
            {"sector": sector, "signal": signal},
        )
    return GateResult("policy_aligned", True, "passed", {"sector": sector, "signal": signal})


def _gate_broker_consensus(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    bc = context.broker_consensus
    if bc is None or bc.empty or "broker_consensus_score" not in bc.columns:
        return GateResult("broker_consensus", True, "skipped_no_broker")
    mask = (bc["trade_date"] == candidate.trade_date) & (bc["symbol"] == candidate.symbol)
    rows = bc[mask]
    if rows.empty:
        prior = bc[(bc["symbol"] == candidate.symbol) & (bc["trade_date"] <= candidate.trade_date)]
        if prior.empty:
            return GateResult("broker_consensus", True, "skipped_no_match")
        rows = prior.tail(1)
    score = float(rows.iloc[-1]["broker_consensus_score"])
    if score < config.broker_consensus_min:
        return GateResult(
            "broker_consensus", False,
            f"score_{score:.3f}_below_{config.broker_consensus_min:.3f}",
            {"score": score},
        )
    return GateResult("broker_consensus", True, "passed", {"score": score})


def _gate_drawdown_kill(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    dd = context.stock_drawdown
    if dd is None or dd.empty or "dd_20d" not in dd.columns:
        return GateResult("drawdown_kill", True, "skipped_no_dd")
    mask = (dd["trade_date"] == candidate.trade_date) & (dd["symbol"] == candidate.symbol)
    rows = dd[mask]
    if rows.empty:
        return GateResult("drawdown_kill", True, "skipped_no_match")
    dd_val = float(rows.iloc[-1]["dd_20d"])
    if dd_val < config.stock_drawdown_kill_pct:
        return GateResult(
            "drawdown_kill", False,
            f"dd_{dd_val:.3f}_below_{config.stock_drawdown_kill_pct:.3f}",
            {"dd_20d": dd_val},
        )
    return GateResult("drawdown_kill", True, "passed", {"dd_20d": dd_val})


def _gate_concentration_limit(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    sector = _lookup_sector(context.sector_map, candidate.symbol)
    if sector is None:
        return GateResult("concentration_limit", True, "skipped_no_sector")
    current = float(context.sector_weights.get(sector, 0.0))
    proposed = current + max(0.0, candidate.target_weight)
    if proposed > config.max_sector_weight:
        return GateResult(
            "concentration_limit", False,
            f"sector_weight_{proposed:.3f}_above_{config.max_sector_weight:.3f}",
            {"sector": sector, "current": current, "proposed": proposed},
        )
    return GateResult(
        "concentration_limit", True, "passed",
        {"sector": sector, "current": current, "proposed": proposed},
    )


def _gate_risk_budget(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    if abs(candidate.target_weight) > config.max_name_weight:
        return GateResult(
            "risk_budget", False,
            f"target_weight_{candidate.target_weight:.4f}_above_{config.max_name_weight:.4f}",
            {"target_weight": candidate.target_weight},
        )
    return GateResult(
        "risk_budget", True, "passed",
        {"target_weight": candidate.target_weight},
    )


def _resolve_regime_label(context: DecisionContext, trade_date: pd.Timestamp) -> str | None:
    """Pull the regime label for ``trade_date`` from ``context.regime_state``.

    Returns ``None`` when the regime series is unavailable so callers
    can treat absent context as 'unknown regime'.
    """
    regime = context.regime_state
    if regime is None or len(regime) == 0:
        return None
    try:
        idx = pd.Index(regime.index)
        if trade_date in idx:
            return str(regime.loc[trade_date])
        # backward-fill nearest prior date
        nearest = regime.reindex([trade_date], method="ffill")
        if nearest.empty or pd.isna(nearest.iloc[0]):
            return None
        return str(nearest.iloc[0])
    except (KeyError, IndexError):
        return None


def _gate_gross_exposure_budget(
    candidate: Candidate, context: DecisionContext, config: DecisionChainConfig
) -> GateResult:
    """Enforce 60% default / 80% high-conviction global gross exposure.

    The candidate's ``target_weight`` would *raise* the portfolio's
    gross exposure to ``current_gross_exposure + |target_weight|``.
    If this exceeds the default cap (60%), the gate checks whether
    the high-conviction conditions hold:

    * ``context.global_conviction`` ≥ ``high_conviction_threshold``
    * regime in ``high_conviction_friendly_regimes``

    Only when both hold can the gate let the candidate push exposure
    up to the high-conviction cap (80%). Anything above that, or
    above 60% without the high-conviction conditions, is rejected.

    Selling (target_weight ≤ 0) always passes — the gate is about
    growing total exposure, not trimming it.
    """
    if candidate.target_weight <= 0:
        return GateResult(
            "gross_exposure_budget", True, "passed_sell_or_trim",
            {"target_weight": candidate.target_weight},
        )
    current = float(max(0.0, context.current_gross_exposure))
    proposed = current + float(candidate.target_weight)
    default_cap = float(config.default_gross_exposure_cap)
    hc_cap = float(config.high_conviction_gross_exposure_cap)
    detail = {
        "current_gross": current,
        "proposed_gross": proposed,
        "default_cap": default_cap,
        "high_conviction_cap": hc_cap,
        "global_conviction": float(context.global_conviction),
    }
    if proposed <= default_cap:
        return GateResult("gross_exposure_budget", True, "within_default_cap", detail)
    if proposed > hc_cap:
        return GateResult(
            "gross_exposure_budget", False,
            f"proposed_gross_{proposed:.3f}_above_high_conviction_cap_{hc_cap:.3f}",
            detail,
        )
    # default_cap < proposed <= hc_cap → high-conviction conditions required
    regime = _resolve_regime_label(context, candidate.trade_date)
    detail["regime"] = regime
    conviction_ok = float(context.global_conviction) >= float(config.high_conviction_threshold)
    regime_ok = regime in config.high_conviction_friendly_regimes
    if conviction_ok and regime_ok:
        return GateResult(
            "gross_exposure_budget", True,
            "high_conviction_extension_allowed", detail,
        )
    reason_parts: list[str] = []
    if not conviction_ok:
        reason_parts.append(
            f"global_conviction_{float(context.global_conviction):.3f}_below_{float(config.high_conviction_threshold):.3f}"
        )
    if not regime_ok:
        reason_parts.append(f"regime_{regime}_not_in_friendly")
    return GateResult(
        "gross_exposure_budget", False,
        f"above_default_{default_cap:.2f}_and_(" + ",".join(reason_parts) + ")",
        detail,
    )


GATE_FUNCTIONS: dict[str, Callable[[Candidate, DecisionContext, DecisionChainConfig], GateResult]] = {
    "alpha_threshold": _gate_alpha_threshold,
    "liquidity": _gate_liquidity,
    "tradeable_today": _gate_tradeable_today,
    "price_limit_block": _gate_price_limit_block,
    "st_status": _gate_st_status,
    "sector_pool": _gate_sector_pool,
    "hard_market_gate": _gate_hard_market_gate,
    "regime_alignment": _gate_regime_alignment,
    "fundamental_filter": _gate_fundamental_filter,
    "policy_aligned": _gate_policy_aligned,
    "broker_consensus": _gate_broker_consensus,
    "drawdown_kill": _gate_drawdown_kill,
    "concentration_limit": _gate_concentration_limit,
    "risk_budget": _gate_risk_budget,
    "gross_exposure_budget": _gate_gross_exposure_budget,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_decision_chain(
    candidate: Candidate,
    context: DecisionContext,
    config: DecisionChainConfig | None = None,
) -> DecisionTrace:
    """Run one Candidate through the 14-gate chain. First-fail short-circuits."""
    cfg = config or DecisionChainConfig()
    trace = DecisionTrace(
        candidate_id=_candidate_id(candidate),
        trade_date=candidate.trade_date,
        symbol=candidate.symbol,
        final_decision="eligible",  # optimistic default; overwritten on fail
        failed_gate=None,
    )
    for gate_name in GATE_ORDER:
        if gate_name not in cfg.enabled_gates:
            continue
        gate_fn = GATE_FUNCTIONS[gate_name]
        result = gate_fn(candidate, context, cfg)
        trace.gate_results.append(result)
        if not result.passed:
            trace.final_decision = "rejected"
            trace.failed_gate = gate_name
            break
    return trace


def run_decision_chain_batch(
    candidates: Iterable[Candidate],
    context: DecisionContext,
    config: DecisionChainConfig | None = None,
) -> list[DecisionTrace]:
    return [run_decision_chain(c, context, config) for c in candidates]


def traces_to_frame(traces: Sequence[DecisionTrace]) -> pd.DataFrame:
    """Long-form audit frame: one row per (trace, gate_result)."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for gate in trace.gate_results:
            rows.append(
                {
                    "candidate_id": trace.candidate_id,
                    "trade_date": trace.trade_date,
                    "symbol": trace.symbol,
                    "final_decision": trace.final_decision,
                    "failed_gate": trace.failed_gate,
                    "gate_name": gate.gate_name,
                    "gate_passed": gate.passed,
                    "gate_reason": gate.reason,
                }
            )
    return pd.DataFrame(rows)
