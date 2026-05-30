"""Position state machine implementation.

Why this exists
---------------
The deployed sleeve currently treats each symbol as binary: it's either
in the top-K (BUY) or it's not (NOT BUY). That is too coarse — the
user's v4 spec §7-8 calls for an 11-state machine that:

* Distinguishes a *watching* state from an *entry-ready* state from
  an *active position*.
* Tracks holding period so the short-line model doesn't get tossed at
  3 days just because mid-horizon score dropped.
* Allows graceful state transitions: HOLD_SHORT → HOLD_MID when the
  signal stays valid past the short window.
* Implements explicit DO_T (做 T), REDUCE, TAKE_PROFIT, STOP_LOSS
  paths instead of "drop from portfolio".
* Hard-bans names that hit ST / suspended / repeated failure conditions.

This module exposes the machine as a pure stateless transition function
plus a tiny per-symbol state log that callers can persist. There is no
I/O here — wiring into the sleeve backtest is done in v7_experiment.

The transition rules are intentionally deterministic and unit-testable.
There is no learned policy here; learned signals come in through the
``PositionContext`` (predictions, regime, etc.). The machine just turns
heterogeneous signals into a single decision per (date, symbol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class PositionState(Enum):
    """11 states from user spec §8.

    Values are auto-assigned ints; use ``.name`` for display.
    """

    BAN = auto()                # never trade (ST hard, structural ban)
    WATCH = auto()              # tracking, no entry yet
    LOW_BUY_READY = auto()      # signal valid + price near support → low-buy
    OPEN_POSITION = auto()      # just entered; pre-confirmation hold
    HOLD_SHORT = auto()         # in short-horizon hold (≤ 5 days)
    HOLD_MID = auto()           # in mid-horizon hold (6-30 days)
    HOLD_LONG = auto()          # in long-horizon hold (31-120 days)
    DO_T = auto()               # do-T (intra-day add-then-sell to lower cost)
    REDUCE = auto()             # partial reduce (1/3 ~ 1/2 of position)
    TAKE_PROFIT = auto()        # signal weakened but still positive — exit
    STOP_LOSS = auto()          # hard stop hit — exit
    EXIT = auto()               # graceful exit (signal invalidated)


@dataclass(frozen=True)
class StateMachineConfig:
    """Transition thresholds. All values dimensionless / fractional.

    ``short_hold_max_days`` / ``mid_hold_max_days`` / ``long_hold_max_days``
    bound the holding window for each tier. Default (5 / 30 / 120)
    matches the user's "短/中/长" spec.

    ``low_buy_pred_quantile`` — prediction must be in the top X
    quantile of the day to be considered low-buy ready (default 0.10
    = top 10%).

    ``take_profit_pred_drop`` — if the rolling prediction drops by
    ≥ this fraction from entry, take profit. Default 0.40 (40%
    deterioration).

    ``stop_loss_unrealized_dd`` — hard stop at this absolute
    unrealized drawdown on the position. Default 0.08 = 8%.

    ``reduce_pred_drop`` — partial-reduce trigger (gentler than
    take-profit). Default 0.20.

    ``do_t_pred_threshold`` — when in active hold and intraday
    prediction strength exceeds this fractile of the day's universe,
    DO_T (one-shot add-then-trim). Default 0.95 (top 5%).
    """

    short_hold_max_days: int = 5
    mid_hold_max_days: int = 30
    long_hold_max_days: int = 120
    low_buy_pred_quantile: float = 0.10
    take_profit_pred_drop: float = 0.40
    stop_loss_unrealized_dd: float = 0.08
    reduce_pred_drop: float = 0.20
    do_t_pred_threshold: float = 0.95


@dataclass(frozen=True)
class PositionContext:
    """Per (date, symbol) snapshot the machine needs to decide.

    All fields are optional — missing values are treated as
    "unknown" and the machine errs on the safe side.
    """

    symbol: str
    trade_date: object  # pandas.Timestamp or str
    current_state: PositionState
    current_weight: float = 0.0          # 0..1 (or signed for long-short)
    entry_weight: float = 0.0            # weight at most recent entry
    entry_prediction: Optional[float] = None
    current_prediction: Optional[float] = None
    pred_quantile: Optional[float] = None  # this prediction's percentile in today's universe
    days_held: int = 0
    unrealized_return: Optional[float] = None  # since entry
    unrealized_drawdown: Optional[float] = None  # min cumulative return since entry, capped at 0
    is_suspended: bool = False
    is_st: bool = False
    is_limit_up_at_close: bool = False
    is_high_chase: bool = False
    market_regime: Optional[str] = None   # normal / caution / bear / crisis


@dataclass(frozen=True)
class PositionDecision:
    """Output of a transition step.

    ``target_state`` is what the symbol should be in after this
    decision. ``weight_action`` is one of: ``buy`` (open or add),
    ``hold`` (keep current), ``reduce`` (cut partial), ``exit``
    (sell all), ``do_t`` (buy then sell same day), ``skip`` (do not
    interact this period). ``target_weight_multiplier`` is applied
    to ``current_weight`` for ``hold`` / ``reduce`` actions.
    """

    target_state: PositionState
    weight_action: str
    target_weight_multiplier: float
    rationale: str


class PositionStateMachine:
    """Deterministic state machine.

    Usage:
        sm = PositionStateMachine(StateMachineConfig())
        decision = sm.transition(context)
    """

    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()

    # --- public API -------------------------------------------------

    def transition(self, ctx: PositionContext) -> PositionDecision:
        """Decide the next state + weight action for a single symbol.

        Order of precedence (highest priority first):

        1. BAN state → stay BAN
        2. hard universe blocks (ST hard / suspended) → EXIT or BAN
        3. position-level hard stops (stop-loss DD) → STOP_LOSS
        4. natural state progression by days_held within current tier
        5. entry / re-entry conditions from WATCH / LOW_BUY_READY
        6. signal deterioration triggers REDUCE / TAKE_PROFIT
        7. otherwise: hold or skip
        """

        # 1) BAN — re-evaluate per spec "ST 不要那么绝对" (review fix #5).
        # If currently BAN-ed BUT the cause has cleared (e.g. stock un-ST's),
        # let it back into WATCH so future signals can act. BAN persists only
        # while the underlying block condition (ST today) holds.
        if ctx.current_state == PositionState.BAN:
            if ctx.is_st or ctx.is_suspended:
                return PositionDecision(PositionState.BAN, "skip", 0.0, "state=BAN and condition still active")
            return PositionDecision(PositionState.WATCH, "skip", 0.0, "BAN cleared (no longer ST / suspended) → back to WATCH")

        # 2) Hard universe blocks
        if ctx.is_suspended and ctx.current_weight > 0:
            return PositionDecision(
                PositionState.HOLD_SHORT if ctx.days_held <= self.config.short_hold_max_days else PositionState.HOLD_MID,
                "hold",
                1.0,
                "suspended: hold existing, cannot trade",
            )
        if ctx.is_suspended and ctx.current_weight == 0:
            return PositionDecision(PositionState.WATCH, "skip", 0.0, "suspended: no entry possible")
        if ctx.is_st and ctx.current_state != PositionState.HOLD_LONG:
            # ST is soft-blocked; if not already in a long-held position, exit
            if ctx.current_weight > 0:
                return PositionDecision(PositionState.EXIT, "exit", 0.0, "is_st: soft block, exit pending")
            return PositionDecision(PositionState.BAN, "skip", 0.0, "is_st: do not enter")

        # 3) Position-level hard stops
        if ctx.current_weight > 0 and ctx.unrealized_drawdown is not None:
            if abs(ctx.unrealized_drawdown) >= self.config.stop_loss_unrealized_dd:
                return PositionDecision(PositionState.STOP_LOSS, "exit", 0.0, f"unrealized DD {ctx.unrealized_drawdown:.2%} >= stop")

        # 4) Natural state aging
        if ctx.current_weight > 0:
            decision = self._decide_for_holder(ctx)
            if decision is not None:
                return decision

        # 5) Entry from WATCH or LOW_BUY_READY
        if ctx.current_weight == 0:
            decision = self._decide_for_non_holder(ctx)
            if decision is not None:
                return decision

        return PositionDecision(ctx.current_state, "hold", 1.0, "no transition triggered")

    # --- internal helpers -------------------------------------------

    def _decide_for_holder(self, ctx: PositionContext) -> Optional[PositionDecision]:
        # First decide which HOLD tier we belong in by days_held
        if ctx.days_held > self.config.long_hold_max_days:
            # Beyond long-horizon — take profit (signal has had its full window)
            return PositionDecision(
                PositionState.TAKE_PROFIT, "exit", 0.0,
                f"days_held {ctx.days_held} > long_hold_max_days, profit-take",
            )
        natural_state = self._hold_tier_for_age(ctx.days_held)

        # Check signal deterioration vs entry — handle negative entries
        # safely (review fix #6). The fraction drop is computed against
        # |entry| so it stays meaningful regardless of entry sign. Tiny
        # |entry| (<1e-6) is skipped to avoid div-by-near-zero noise.
        if ctx.entry_prediction is not None and ctx.current_prediction is not None:
            entry = float(ctx.entry_prediction)
            now = float(ctx.current_prediction)
            if abs(entry) > 1e-6:
                drop = (entry - now) / abs(entry)
                if drop >= self.config.take_profit_pred_drop and ctx.unrealized_return is not None and ctx.unrealized_return > 0:
                    return PositionDecision(
                        PositionState.TAKE_PROFIT, "exit", 0.0,
                        f"pred dropped {drop:.0%}, in profit → take",
                    )
                if drop >= self.config.reduce_pred_drop:
                    return PositionDecision(
                        PositionState.REDUCE, "reduce", 0.5,
                        f"pred dropped {drop:.0%}, partial reduce",
                    )

        # DO_T opportunity — top-quantile prediction in held position
        if ctx.pred_quantile is not None and ctx.pred_quantile >= self.config.do_t_pred_threshold:
            return PositionDecision(
                natural_state, "do_t", 1.0,
                f"pred quantile {ctx.pred_quantile:.2f} ≥ DO_T threshold, intra-day add-then-trim",
            )

        return PositionDecision(natural_state, "hold", 1.0, "in active hold")

    def _decide_for_non_holder(self, ctx: PositionContext) -> Optional[PositionDecision]:
        if ctx.is_high_chase:
            return PositionDecision(PositionState.WATCH, "skip", 0.0, "high-chase blocked")
        if ctx.is_limit_up_at_close:
            return PositionDecision(PositionState.WATCH, "skip", 0.0, "limit-up: cannot fill")
        if ctx.market_regime == "crisis":
            return PositionDecision(PositionState.WATCH, "skip", 0.0, "regime=crisis: no new entries")

        if ctx.pred_quantile is not None and ctx.pred_quantile >= (1.0 - self.config.low_buy_pred_quantile):
            return PositionDecision(
                PositionState.LOW_BUY_READY, "buy", 1.0,
                f"pred quantile {ctx.pred_quantile:.2f} in top {self.config.low_buy_pred_quantile:.0%} → low-buy",
            )
        return PositionDecision(PositionState.WATCH, "skip", 0.0, "no entry trigger")

    def _hold_tier_for_age(self, days_held: int) -> PositionState:
        if days_held <= self.config.short_hold_max_days:
            return PositionState.HOLD_SHORT
        if days_held <= self.config.mid_hold_max_days:
            return PositionState.HOLD_MID
        return PositionState.HOLD_LONG


__all__ = [
    "PositionContext",
    "PositionDecision",
    "PositionState",
    "PositionStateMachine",
    "StateMachineConfig",
]
