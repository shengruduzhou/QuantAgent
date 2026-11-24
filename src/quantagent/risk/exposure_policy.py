"""Gross-exposure policy: drawdown ladder + volatility targeting -> multiplier in [0, 1].

Round-29 R2-F10 / R7: the paper account could *stop* (20% all-time-peak latch,
5% daily-loss buy halt) but nothing *managed* drawdown, and the only ladder in
the repository lived inside the training-time sleeve backtest with a rolling
252-day peak. This module is the single place a gross multiplier is decided.
The strict backtest (``scripts/baseline_protocol.py --exposure-policy``) and the
paper target path (``paper.daily_loop``) call the same :func:`decide_exposure`.

Contract
--------
* **Pure and deterministic.** The multiplier at decision date ``t`` is a
  function of the account observations with date ``<= t`` and the policy only.
  Rows after ``t`` are dropped before anything is computed (and counted in the
  record), so a change to the NAV path after ``t`` cannot change the decision
  at ``t``. The decision is applied to orders executed at the *next* session.
* **Cadence independent.** The ladder state machine advances on every account
  observation (session close), not on decision dates, so a weekly backtest and
  a daily paper loop that see the same NAV path decide the same multiplier on
  the same date.
* **Fail closed.** Missing / non-finite / non-positive NAV, a missing NAV at the
  decision date, unsorted or duplicate dates, or a required regime flag that is
  absent produce ``status="fail_closed"`` with ``multiplier=0.0`` and an
  explicit reason. Never a silent 1.0. The two shipped callers refuse to
  proceed on a fail-closed decision (backtest raises, paper refuses to freeze a
  target) rather than liquidating on a data gap.
* **No leverage.** Every component is capped at 1.0.

Components
----------
1. Volatility targeting (Moreira-Muir 2017; Harvey et al. 2018). Realised vol is
   estimated from *unit-exposure* returns ``r_t = (NAV_t - NAV_{t-1}) /
   invested_{t-1}`` -- the return on the capital that was actually at risk --
   with an EWMA (zero mean, RiskMetrics style). Using the account's own return
   would feed the multiplier back into its own vol estimate (a de-grossed book
   looks calm and is re-levered). ``m_vol = min(1, target / sigma)``. Until
   ``vol_min_observations`` unit returns exist in the trailing window the
   component is in declared warm-up and uses ``vol_warmup_multiplier``
   (recorded as ``vol_status="warmup"``).
2. Drawdown ladder vs the account's ALL-TIME peak (Grossman-Zhou 1993 drawdown
   constraint, discretised). Step DOWN immediately when drawdown from the
   ladder anchor crosses a deeper threshold; step UP one tier only after the
   drawdown has recovered to ``threshold - hysteresis`` AND the current tier has
   been held ``ladder_min_hold_sessions`` -- so the tier cannot flip daily.
   The anchor is the all-time peak until a re-anchor.
3. Re-anchor (explicit answer to the cash-lock problem). A monotone ladder on
   the all-time peak is a discretised CPPI: once a book is deep under water it
   is pinned at the floor and, at floor exposure, can almost never earn back to
   the old peak (round-29 R7 measured exactly this). After
   ``ladder_reanchor_sessions`` in a reduced tier with a strictly positive NAV
   change over that same window -- evidence the losses stopped; the SIGN of
   the account return at a positive floor exposure equals the sign of the
   strategy return, so the evidence is observable (Kaminski-Lo: re-entry must
   be judged on the risky asset, not on a flat account) -- the anchor is reset
   to the current NAV and the book steps up ONE tier. Trade-off (documented,
   not hidden): a re-anchored book can lose again from a lower base; the
   all-time-peak drawdown is still reported in every record and still drives
   the venue's hard 20% latch, which this policy never overrides. A floor
   multiplier of 0 is refused by validation because re-entry evidence would be
   unobservable.
4. Optional regime multiplier: when configured, ``regime_risk_off`` must be
   supplied (None -> fail closed).

Final multiplier = floor-to-grid(m_vol * m_ladder * m_regime), never rounded to
zero when positive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EXPOSURE_DECISION_SCHEMA = "quantagent.risk.exposure_decision.v1"
STATUS_OK = "ok"
STATUS_FAIL_CLOSED = "fail_closed"


class ExposurePolicyError(ValueError):
    """Invalid policy configuration (programming / config error, not data)."""


class ExposureDecisionRefused(RuntimeError):
    """Raised by callers that refuse to act on a fail-closed decision."""

    def __init__(self, decision: "ExposureDecision") -> None:
        super().__init__(
            f"exposure policy {decision.policy_name!r} failed closed at "
            f"{decision.decision_date}: {', '.join(decision.reasons) or 'unknown'}"
        )
        self.decision = decision


@dataclass(frozen=True)
class ExposurePolicy:
    """Exposure policy configuration. All fields are explicit and recorded."""

    name: str
    # -- volatility targeting (None disables) -----------------------------
    vol_target_annual: float | None = None
    vol_halflife_sessions: float = 20.0
    vol_window_sessions: int = 126
    vol_min_observations: int = 20
    #: A unit return is measured only when the invested fraction at the prior
    #: close was at least this; below it the day is unmeasured (not zero).
    vol_min_invested_fraction: float = 0.05
    #: Declared warm-up multiplier while realised vol is not yet measurable.
    vol_warmup_multiplier: float = 0.5
    periods_per_year: int = 244
    # -- drawdown ladder (empty thresholds disables) ------------------------
    ladder_thresholds: tuple[float, ...] = ()
    ladder_multipliers: tuple[float, ...] = ()
    ladder_hysteresis: float = 0.04
    ladder_min_hold_sessions: int = 10
    #: None disables the re-anchor rule (monotone ladder on the all-time peak).
    ladder_reanchor_sessions: int | None = 63
    # -- regime (None disables) --------------------------------------------
    regime_risk_off_multiplier: float | None = None
    # -- output --------------------------------------------------------------
    multiplier_step: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "ladder_thresholds", tuple(float(x) for x in self.ladder_thresholds))
        object.__setattr__(self, "ladder_multipliers", tuple(float(x) for x in self.ladder_multipliers))
        self.validate()

    # -- validation -----------------------------------------------------------
    def validate(self) -> None:
        if not str(self.name).strip():
            raise ExposurePolicyError("policy name must be non-empty")
        if self.vol_target_annual is not None:
            if not (math.isfinite(self.vol_target_annual) and self.vol_target_annual > 0):
                raise ExposurePolicyError("vol_target_annual must be finite and > 0 (or None)")
            if not (self.vol_halflife_sessions > 0):
                raise ExposurePolicyError("vol_halflife_sessions must be > 0")
            if int(self.vol_window_sessions) < 2:
                raise ExposurePolicyError("vol_window_sessions must be >= 2")
            if not (2 <= int(self.vol_min_observations) <= int(self.vol_window_sessions)):
                raise ExposurePolicyError("vol_min_observations must be in [2, vol_window_sessions]")
            if not (0.0 < self.vol_min_invested_fraction <= 1.0):
                raise ExposurePolicyError("vol_min_invested_fraction must be in (0, 1]")
            if not (0.0 < self.vol_warmup_multiplier <= 1.0):
                raise ExposurePolicyError(
                    "vol_warmup_multiplier must be in (0, 1]: 0 would never let a new "
                    "account invest, so vol could never become measurable"
                )
            if int(self.periods_per_year) <= 0:
                raise ExposurePolicyError("periods_per_year must be > 0")
        th, mu = self.ladder_thresholds, self.ladder_multipliers
        if len(th) != len(mu):
            raise ExposurePolicyError("ladder_thresholds and ladder_multipliers must have equal length")
        if th:
            if any(not (0.0 < x < 1.0) for x in th):
                raise ExposurePolicyError("ladder thresholds must be drawdown fractions in (0, 1)")
            if any(b <= a for a, b in zip(th, th[1:])):
                raise ExposurePolicyError("ladder thresholds must be strictly increasing")
            if any(not (0.0 < x <= 1.0) for x in mu):
                raise ExposurePolicyError(
                    "ladder multipliers must be in (0, 1]: a 0 floor is a cash-lock -- "
                    "re-entry evidence becomes unobservable"
                )
            if any(b > a for a, b in zip(mu, mu[1:])):
                raise ExposurePolicyError("ladder multipliers must be non-increasing with depth")
            if not (0.0 <= self.ladder_hysteresis < th[0]):
                raise ExposurePolicyError("ladder_hysteresis must be in [0, first threshold)")
            if int(self.ladder_min_hold_sessions) < 1:
                raise ExposurePolicyError("ladder_min_hold_sessions must be >= 1")
            if self.ladder_reanchor_sessions is not None and int(self.ladder_reanchor_sessions) < 1:
                raise ExposurePolicyError("ladder_reanchor_sessions must be >= 1 or None")
        if self.regime_risk_off_multiplier is not None and not (
            0.0 <= self.regime_risk_off_multiplier <= 1.0
        ):
            raise ExposurePolicyError("regime_risk_off_multiplier must be in [0, 1] or None")
        if not (0.0 < self.multiplier_step <= 1.0):
            raise ExposurePolicyError("multiplier_step must be in (0, 1]")

    def validate_backstop(self, hard_drawdown_limit: float) -> None:
        """Refuse a ladder that does not act strictly before the hard latch.

        The venue's all-time-peak drawdown latch (paper ``RiskLimits.max_drawdown``)
        is the backstop ABOVE this ladder. A ladder threshold at or beyond it
        would make the soft layer redundant exactly when it matters.
        """
        limit = float(hard_drawdown_limit)
        if not (math.isfinite(limit) and limit > 0):
            raise ExposurePolicyError("hard drawdown limit must be finite and > 0")
        deep = [x for x in self.ladder_thresholds if x >= limit]
        if deep:
            raise ExposurePolicyError(
                f"ladder thresholds {deep} are not below the hard drawdown latch {limit}"
            )

    # -- identity ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ladder_thresholds"] = list(self.ladder_thresholds)
        payload["ladder_multipliers"] = list(self.ladder_multipliers)
        return payload

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def vol_enabled(self) -> bool:
        return self.vol_target_annual is not None

    @property
    def ladder_enabled(self) -> bool:
        return bool(self.ladder_thresholds)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExposurePolicy":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ExposurePolicyError(f"unknown exposure policy fields: {unknown}")
        if "name" not in payload:
            raise ExposurePolicyError("exposure policy requires a name")
        data = dict(payload)
        for key in ("ladder_thresholds", "ladder_multipliers"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


#: Pre-registered parameter sets (docs/research/preregistrations/ENG_RISK30_preregistration.md).
#: Both are counted as trials. ``identity`` is a no-op policy used to prove the
#: wiring itself is inert (multiplier always 1.0, history still validated).
PRESETS: dict[str, ExposurePolicy] = {
    "identity": ExposurePolicy(name="identity"),
    "vt20": ExposurePolicy(name="vt20", vol_target_annual=0.20),
    "vt20_ddl": ExposurePolicy(
        name="vt20_ddl",
        vol_target_annual=0.20,
        ladder_thresholds=(0.08, 0.12, 0.16),
        ladder_multipliers=(0.75, 0.50, 0.25),
        ladder_hysteresis=0.04,
        ladder_min_hold_sessions=10,
        ladder_reanchor_sessions=63,
    ),
}


def load_exposure_policy(spec: str | Mapping[str, Any] | ExposurePolicy) -> ExposurePolicy:
    """Resolve a preset name, a JSON file path, inline JSON or a mapping."""
    if isinstance(spec, ExposurePolicy):
        return spec
    if isinstance(spec, Mapping):
        return ExposurePolicy.from_dict(spec)
    text = str(spec).strip()
    if not text:
        raise ExposurePolicyError("empty exposure policy spec")
    if text in PRESETS:
        return PRESETS[text]
    if text.startswith("{"):
        return ExposurePolicy.from_dict(json.loads(text))
    path = Path(text)
    if path.suffix.lower() == ".json" and path.exists():
        return ExposurePolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
    raise ExposurePolicyError(
        f"unknown exposure policy {text!r}: use a preset {sorted(PRESETS)}, inline JSON or a .json path"
    )


@dataclass(frozen=True)
class ExposureDecision:
    """One decision plus every input it used (the audit record)."""

    policy_name: str
    policy_sha256: str
    decision_date: str
    status: str
    multiplier: float
    reasons: tuple[str, ...] = ()
    raw_multiplier: float | None = None
    observations_used: int = 0
    observations_after_decision_ignored: int = 0
    first_observation_date: str | None = None
    nav: float | None = None
    invested_fraction: float | None = None
    all_time_peak_nav: float | None = None
    all_time_peak_date: str | None = None
    drawdown_all_time: float | None = None
    ladder_status: str = "disabled"
    ladder_anchor_nav: float | None = None
    drawdown_vs_anchor: float | None = None
    tier: int | None = None
    tier_multiplier: float | None = None
    sessions_in_tier: int | None = None
    reanchor_count: int | None = None
    last_reanchor_date: str | None = None
    vol_status: str = "disabled"
    realised_vol_annual: float | None = None
    vol_observations: int | None = None
    vol_multiplier: float | None = None
    regime_risk_off: bool | None = None
    regime_multiplier: float | None = None
    schema: str = EXPOSURE_DECISION_SCHEMA

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


# ---------------------------------------------------------------------------
# history normalisation
# ---------------------------------------------------------------------------

def account_history_frame(
    dates: Sequence[Any], nav: Sequence[float], invested_value: Sequence[float]
) -> pd.DataFrame:
    """Build the canonical ``(trade_date, nav, invested_value)`` history frame."""
    return pd.DataFrame({
        "trade_date": pd.to_datetime(list(dates)),
        "nav": np.asarray(list(nav), dtype=float),
        "invested_value": np.asarray(list(invested_value), dtype=float),
    })


def _fail(policy: ExposurePolicy, decision_date: str, reason: str, **extra: Any) -> ExposureDecision:
    return ExposureDecision(
        policy_name=policy.name,
        policy_sha256=policy.sha256,
        decision_date=decision_date,
        status=STATUS_FAIL_CLOSED,
        multiplier=0.0,
        reasons=(reason,),
        **extra,
    )


def _quantise(value: float, step: float) -> float:
    if value <= 0:
        return 0.0
    units = math.floor(value / step + 1e-9)
    return float(min(1.0, max(step, round(units * step, 10))))


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------

def _unit_returns(nav: np.ndarray, invested: np.ndarray, min_fraction: float) -> np.ndarray:
    """Return on capital at risk; NaN (unmeasured) when the prior book was ~cash."""
    out = np.full(len(nav), np.nan)
    if len(nav) < 2:
        return out
    prior_nav = nav[:-1]
    prior_inv = invested[:-1]
    frac = np.divide(prior_inv, prior_nav, out=np.zeros_like(prior_inv), where=prior_nav > 0)
    measured = frac >= min_fraction
    delta = nav[1:] - nav[:-1]
    out[1:][measured] = delta[measured] / prior_inv[measured]
    return out


def _vol_component(
    policy: ExposurePolicy, nav: np.ndarray, invested: np.ndarray
) -> tuple[str, float, float | None, int]:
    """(status, multiplier, realised annual vol, observations)."""
    returns = _unit_returns(nav, invested, policy.vol_min_invested_fraction)
    window = returns[-int(policy.vol_window_sessions):]
    ages = np.arange(len(window))[::-1]  # 0 = decision-date return
    measured = np.isfinite(window)
    count = int(measured.sum())
    if count < int(policy.vol_min_observations):
        return "warmup", float(policy.vol_warmup_multiplier), None, count
    weights = np.power(0.5, ages[measured] / float(policy.vol_halflife_sessions))
    values = window[measured]
    variance = float(np.dot(weights, values * values) / weights.sum())
    sigma = math.sqrt(max(variance, 0.0) * float(policy.periods_per_year))
    if sigma <= 0:
        return "measured", 1.0, sigma, count
    return "measured", float(min(1.0, float(policy.vol_target_annual) / sigma)), sigma, count


@dataclass
class _LadderState:
    tier: int = 0
    anchor: float = float("nan")
    held: int = 0
    reanchors: int = 0
    last_reanchor_index: int | None = None


def _run_ladder(policy: ExposurePolicy, nav: np.ndarray) -> _LadderState:
    """Advance the ladder state machine over every observation (cadence-free)."""
    th = policy.ladder_thresholds
    hyst = float(policy.ladder_hysteresis)
    min_hold = int(policy.ladder_min_hold_sessions)
    reanchor_n = policy.ladder_reanchor_sessions
    state = _LadderState(anchor=float(nav[0]))
    for i, value in enumerate(nav):
        if value > state.anchor:
            state.anchor = float(value)
        dd = 1.0 - float(value) / state.anchor
        target_tier = sum(1 for x in th if dd >= x)
        if i == 0:
            state.tier = target_tier
            state.held = 0
            continue
        if target_tier > state.tier:
            state.tier = target_tier  # de-risk immediately, possibly several tiers
            state.held = 0
            continue
        state.held += 1
        if state.tier == 0:
            continue
        if (
            reanchor_n is not None
            and state.held >= int(reanchor_n)
            and i - int(reanchor_n) >= 0
            and float(value) > float(nav[i - int(reanchor_n)])
        ):
            state.anchor = float(value)
            state.tier -= 1
            state.held = 0
            state.reanchors += 1
            state.last_reanchor_index = i
            continue
        if state.held >= min_hold and dd <= th[state.tier - 1] - hyst:
            state.tier -= 1
            state.held = 0
    return state


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def decide_exposure(
    history: pd.DataFrame,
    policy: ExposurePolicy,
    *,
    decision_date: Any,
    regime_risk_off: bool | None = None,
) -> ExposureDecision:
    """Decide the gross multiplier at ``decision_date`` (a session close).

    ``history`` holds one row per account observation (session close) with
    ``trade_date``, ``nav`` (marked account equity) and ``invested_value``
    (marked value of positions). The account's inception (initial capital, zero
    invested) should be the first row so the all-time peak includes it. Rows
    after ``decision_date`` are ignored.
    """
    if not isinstance(policy, ExposurePolicy):
        raise ExposurePolicyError("policy must be an ExposurePolicy")
    try:
        when = pd.Timestamp(decision_date).normalize()
    except (TypeError, ValueError):
        return _fail(policy, str(decision_date), "decision_date_invalid")
    if pd.isna(when):
        return _fail(policy, str(decision_date), "decision_date_invalid")
    stamp = when.date().isoformat()
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return _fail(policy, stamp, "nav_history_missing")
    missing = sorted({"trade_date", "nav", "invested_value"} - set(history.columns))
    if missing:
        return _fail(policy, stamp, f"nav_history_columns_missing:{','.join(missing)}")
    dates = pd.to_datetime(history["trade_date"], errors="coerce")
    if dates.isna().any():
        return _fail(policy, stamp, "nav_history_date_invalid")
    dates = dates.dt.normalize()
    keep = (dates <= when).to_numpy()
    ignored = int((~keep).sum())
    dates = dates[keep]
    if dates.empty:
        return _fail(policy, stamp, "nav_history_missing", observations_after_decision_ignored=ignored)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        return _fail(policy, stamp, "nav_history_not_strictly_increasing",
                     observations_after_decision_ignored=ignored)
    nav = pd.to_numeric(history["nav"], errors="coerce").to_numpy(dtype=float)[keep]
    invested = pd.to_numeric(history["invested_value"], errors="coerce").to_numpy(dtype=float)[keep]
    common = dict(
        observations_used=int(len(nav)),
        observations_after_decision_ignored=ignored,
        first_observation_date=dates.iloc[0].date().isoformat(),
    )
    if not np.isfinite(nav).all() or (nav <= 0).any():
        return _fail(policy, stamp, "nav_non_finite_or_non_positive", **common)
    if not np.isfinite(invested).all() or (invested < 0).any():
        return _fail(policy, stamp, "invested_value_non_finite_or_negative", **common)
    if dates.iloc[-1] != when:
        return _fail(policy, stamp, "nav_at_decision_date_missing", **common)

    peak_index = int(np.flatnonzero(nav == nav.max())[0])
    all_time_peak = float(nav.max())
    measured = dict(
        common,
        nav=float(nav[-1]),
        invested_fraction=float(invested[-1] / nav[-1]),
        all_time_peak_nav=all_time_peak,
        all_time_peak_date=dates.iloc[peak_index].date().isoformat(),
        drawdown_all_time=float(1.0 - nav[-1] / all_time_peak),
    )

    reasons: list[str] = []
    multiplier = 1.0

    # regime ----------------------------------------------------------------
    regime_multiplier: float | None = None
    if policy.regime_risk_off_multiplier is not None:
        if regime_risk_off is None:
            return _fail(policy, stamp, "regime_flag_missing", **measured)
        regime_multiplier = float(policy.regime_risk_off_multiplier) if regime_risk_off else 1.0
        multiplier *= regime_multiplier
    elif regime_risk_off is not None:
        reasons.append("regime_flag_ignored_policy_has_no_regime_component")

    # volatility ------------------------------------------------------------
    vol_status, vol_multiplier, sigma, vol_obs = "disabled", None, None, None
    if policy.vol_enabled:
        vol_status, vol_multiplier, sigma, vol_obs = _vol_component(policy, nav, invested)
        if vol_status == "warmup":
            reasons.append("vol_warmup_declared_multiplier")
        multiplier *= vol_multiplier

    # ladder ----------------------------------------------------------------
    ladder: dict[str, Any] = {"ladder_status": "disabled"}
    if policy.ladder_enabled:
        state = _run_ladder(policy, nav)
        tier_multiplier = 1.0 if state.tier == 0 else policy.ladder_multipliers[state.tier - 1]
        multiplier *= tier_multiplier
        ladder = {
            "ladder_status": "active",
            "ladder_anchor_nav": float(state.anchor),
            "drawdown_vs_anchor": float(1.0 - nav[-1] / state.anchor),
            "tier": int(state.tier),
            "tier_multiplier": float(tier_multiplier),
            "sessions_in_tier": int(state.held),
            "reanchor_count": int(state.reanchors),
            "last_reanchor_date": (
                None if state.last_reanchor_index is None
                else dates.iloc[state.last_reanchor_index].date().isoformat()
            ),
        }

    return ExposureDecision(
        policy_name=policy.name,
        policy_sha256=policy.sha256,
        decision_date=stamp,
        status=STATUS_OK,
        multiplier=_quantise(float(multiplier), float(policy.multiplier_step)),
        reasons=tuple(reasons),
        raw_multiplier=float(multiplier),
        vol_status=vol_status,
        realised_vol_annual=sigma,
        vol_observations=vol_obs,
        vol_multiplier=vol_multiplier,
        regime_risk_off=regime_risk_off,
        regime_multiplier=regime_multiplier,
        **measured,
        **ladder,
    )


__all__ = [
    "EXPOSURE_DECISION_SCHEMA",
    "ExposureDecision",
    "ExposureDecisionRefused",
    "ExposurePolicy",
    "ExposurePolicyError",
    "PRESETS",
    "STATUS_FAIL_CLOSED",
    "STATUS_OK",
    "account_history_frame",
    "decide_exposure",
    "load_exposure_policy",
]
