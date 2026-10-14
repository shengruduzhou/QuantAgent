"""Round-30 ENG_RISK30: ExposurePolicy (drawdown ladder + vol targeting).

Every test here fails before the module existed (ImportError) and pins one
property the round-29 R2-F10 / R7 findings asked for: no look-ahead,
hysteresis (no daily whipsaw), an explicit answer to the all-time-peak cash-lock,
fail-closed on missing inputs, and no leverage.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from quantagent.risk.exposure_policy import (
    PRESETS,
    STATUS_FAIL_CLOSED,
    STATUS_OK,
    ExposurePolicy,
    ExposurePolicyError,
    account_history_frame,
    decide_exposure,
    load_exposure_policy,
)

LADDER = ExposurePolicy(
    name="ladder_only",
    ladder_thresholds=(0.08, 0.12, 0.16),
    ladder_multipliers=(0.75, 0.50, 0.25),
    ladder_hysteresis=0.04,
    ladder_min_hold_sessions=10,
    ladder_reanchor_sessions=63,
)


def _history(navs, invested_fraction=0.95, start="2020-01-01"):
    navs = np.asarray(navs, dtype=float)
    dates = pd.bdate_range(start, periods=len(navs))
    invested = navs * invested_fraction
    invested[0] = 0.0  # inception: initial capital, nothing invested yet
    return account_history_frame(dates, navs, invested)


def _tier_path(policy, navs):
    hist = _history(navs)
    return [
        decide_exposure(hist, policy, decision_date=d).tier for d in hist["trade_date"]
    ]


def _random_nav(n=400, seed=0, drift=0.0, vol=0.02):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n - 1)
    return 1_000_000.0 * np.cumprod(np.r_[1.0, 1.0 + r])


# ---------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_nav_change_after_t_does_not_change_decision_at_t(name):
    navs = _random_nav(seed=3)
    hist = _history(navs)
    t = hist["trade_date"].iloc[250]
    base = decide_exposure(hist, PRESETS[name], decision_date=t)

    altered = hist.copy()
    after = altered["trade_date"] > t
    altered.loc[after, "nav"] = altered.loc[after, "nav"] * 0.3  # crash after t
    altered.loc[after, "invested_value"] = 0.0
    changed = decide_exposure(altered, PRESETS[name], decision_date=t)

    truncated = decide_exposure(hist[~after], PRESETS[name], decision_date=t)
    assert base.status == STATUS_OK
    assert base.multiplier == changed.multiplier == truncated.multiplier
    a, b = base.to_dict(), changed.to_dict()
    assert a == b  # every recorded input identical, not just the output
    assert base.observations_after_decision_ignored == int(after.sum())
    assert truncated.observations_after_decision_ignored == 0


def test_decision_uses_close_of_decision_date():
    navs = np.r_[np.full(30, 1_000_000.0), 850_000.0]  # 15% loss on the last day
    hist = _history(navs)
    before = decide_exposure(hist, LADDER, decision_date=hist["trade_date"].iloc[-2])
    at = decide_exposure(hist, LADDER, decision_date=hist["trade_date"].iloc[-1])
    assert before.tier == 0 and before.multiplier == 1.0
    assert at.tier == 2 and at.multiplier == 0.5  # de-risk decided at t close, for t+1


# ---------------------------------------------------------------- hysteresis
def test_step_down_immediately_step_up_only_after_recovery_and_min_hold():
    navs = [1_000_000.0] * 5 + [910_000.0]  # 9% drawdown -> tier 1
    navs += [940_000.0] * 30  # recovered to 6%: above 8% - 4% = 4% -> must stay
    navs += [970_000.0] * 3   # 3%: recovered, but only 3 sessions in a row...
    tiers = _tier_path(LADDER, navs)
    assert tiers[5] == 1
    assert all(t == 1 for t in tiers[5:36])  # partial recovery is not enough
    # held counter already exceeds min_hold, so recovery below 4% steps up at once
    assert tiers[36] == 0


def test_step_up_waits_for_min_hold_even_after_full_recovery():
    navs = [1_000_000.0] * 5 + [910_000.0] + [1_000_000.0] * 15
    tiers = _tier_path(LADDER, navs)
    # entered tier 1 at index 5; recovered fully at index 6 but must hold 10 sessions
    assert tiers[5:15] == [1] * 10
    assert tiers[15] == 0


def test_oscillation_around_threshold_cannot_whipsaw_daily():
    navs = [1_000_000.0] * 5
    for i in range(200):
        navs.append(1_000_000.0 * (0.919 if i % 2 == 0 else 0.921))  # 8.1% / 7.9% daily
    tiers = _tier_path(LADDER, navs)
    change_at = [i for i, (a, b) in enumerate(zip(tiers, tiers[1:])) if a != b]
    # one step down on the first breach; the only later change is the quarterly
    # re-anchor (a +0.2% quarter counts as a positive slope) -- never daily flips
    assert tiers[5] == 1
    assert len(change_at) <= 2
    assert all(b - a >= LADDER.ladder_reanchor_sessions for a, b in zip(change_at, change_at[1:]))


def test_multi_tier_drop_is_immediate_and_recovery_is_one_tier_at_a_time():
    navs = [1_000_000.0] * 5 + [820_000.0] + [1_000_000.0] * 40  # -18% -> tier 3
    tiers = _tier_path(LADDER, navs)
    assert tiers[5] == 3
    steps = [t for t in tiers[5:]]
    # each step up is a single tier, separated by >= min_hold sessions
    ups = [i for i, (a, b) in enumerate(zip(steps, steps[1:])) if b < a]
    assert all(steps[i] - steps[i + 1] == 1 for i in ups)
    assert all(b - a >= 10 for a, b in zip(ups, ups[1:]))
    assert tiers[-1] == 0


# ---------------------------------------------------------------- cash-lock
def test_deep_drawdown_does_not_pin_exposure_forever_with_reanchor():
    # -40% then a slow positive drift: never gets back to the all-time peak
    navs = [1_000_000.0] * 5 + list(np.linspace(1_000_000.0, 600_000.0, 40))
    navs += list(600_000.0 * np.cumprod(np.full(300, 1.0003)))
    hist = _history(navs)
    last = decide_exposure(hist, LADDER, decision_date=hist["trade_date"].iloc[-1])
    assert last.drawdown_all_time > 0.30  # still deep under the all-time peak
    assert last.reanchor_count >= 1
    assert last.tier == 0 and last.multiplier == 1.0

    monotone = ExposurePolicy(**{**LADDER.to_dict(), "name": "monotone",
                                 "ladder_reanchor_sessions": None})
    pinned = decide_exposure(hist, monotone, decision_date=hist["trade_date"].iloc[-1])
    assert pinned.tier == 3 and pinned.multiplier == 0.25  # the R7 cash-lock


def test_reanchor_requires_positive_slope():
    navs = [1_000_000.0] * 5 + list(1_000_000.0 * np.cumprod(np.full(400, 0.9995)))
    hist = _history(navs)
    last = decide_exposure(hist, LADDER, decision_date=hist["trade_date"].iloc[-1])
    assert last.reanchor_count == 0
    assert last.tier == 3 and last.multiplier == 0.25
    assert last.drawdown_all_time == pytest.approx(1 - 0.9995 ** 400)


def test_ladder_measures_all_time_peak_not_rolling_window():
    # peak 600 sessions ago; a rolling-252 peak would forget it
    navs = [1_000_000.0] + [1_200_000.0] + [1_000_000.0 * 0.9] * 600
    hist = _history(navs)
    d = decide_exposure(hist, PRESETS["vt20_ddl"], decision_date=hist["trade_date"].iloc[-1])
    assert d.all_time_peak_nav == 1_200_000.0
    assert d.drawdown_all_time == pytest.approx(0.25)


# ---------------------------------------------------------------- fail closed
@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda h: h.iloc[0:0], "nav_history_missing"),
        (lambda h: h.assign(nav=h["nav"].where(h.index != 10, np.nan)), "nav_non_finite_or_non_positive"),
        (lambda h: h.assign(nav=h["nav"].where(h.index != 10, 0.0)), "nav_non_finite_or_non_positive"),
        (lambda h: h.assign(invested_value=h["invested_value"].where(h.index != 10, -1.0)),
         "invested_value_non_finite_or_negative"),
        (lambda h: h.drop(index=h.index[-1]), "nav_at_decision_date_missing"),
        (lambda h: pd.concat([h, h.iloc[[-1]]]), "nav_history_not_strictly_increasing"),
        (lambda h: h.iloc[::-1], "nav_history_not_strictly_increasing"),
        (lambda h: h.drop(columns=["invested_value"]), "nav_history_columns_missing:invested_value"),
    ],
)
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_missing_or_invalid_inputs_fail_closed_never_default_to_one(mutate, reason, name):
    hist = _history(_random_nav(n=60, seed=1))
    t = hist["trade_date"].iloc[-1]
    d = decide_exposure(mutate(hist), PRESETS[name], decision_date=t)
    assert d.status == STATUS_FAIL_CLOSED
    assert d.multiplier == 0.0
    assert d.reasons == (reason,)


def test_none_history_fails_closed():
    d = decide_exposure(None, PRESETS["vt20"], decision_date="2020-01-01")
    assert d.status == STATUS_FAIL_CLOSED and d.multiplier == 0.0


def test_required_regime_flag_missing_fails_closed():
    policy = ExposurePolicy(name="regime", regime_risk_off_multiplier=0.5)
    hist = _history(_random_nav(n=40))
    t = hist["trade_date"].iloc[-1]
    assert decide_exposure(hist, policy, decision_date=t).reasons == ("regime_flag_missing",)
    assert decide_exposure(hist, policy, decision_date=t, regime_risk_off=True).multiplier == 0.5
    assert decide_exposure(hist, policy, decision_date=t, regime_risk_off=False).multiplier == 1.0


# ---------------------------------------------------------------- vol targeting
def test_vol_multiplier_is_capped_at_one_and_never_levers():
    for seed in range(20):
        for vol in (0.001, 0.005, 0.02, 0.05):
            hist = _history(_random_nav(n=200, seed=seed, vol=vol))
            for name in ("vt20", "vt20_ddl"):
                d = decide_exposure(hist, PRESETS[name], decision_date=hist["trade_date"].iloc[-1])
                assert 0.0 < d.multiplier <= 1.0
                assert d.vol_multiplier <= 1.0
    calm = _history(_random_nav(n=200, seed=1, vol=0.001))
    assert decide_exposure(calm, PRESETS["vt20"], decision_date=calm["trade_date"].iloc[-1]).multiplier == 1.0


def test_vol_targets_unit_exposure_return_not_account_return():
    """A half-invested account must not look half as risky (feedback loop)."""
    rng = np.random.default_rng(7)
    strat = rng.normal(0.0, 0.03, 199)  # ~47% annualised strategy vol
    full = [1_000_000.0]
    half = [1_000_000.0]
    for r in strat:
        full.append(full[-1] * (1 + r))
        half.append(half[-1] * (1 + 0.5 * r))
    dates = pd.bdate_range("2021-01-01", periods=200)
    h_full = account_history_frame(dates, full, [0.0] + [v for v in full[:-1]])
    # invested value at close t-1 = 50% of NAV ⇒ account return = 0.5 * strategy return
    h_half = account_history_frame(dates, half, [0.0] + [0.5 * v for v in half[:-1]])
    # make invested_value at close t equal what earned the return over (t, t+1]
    h_full["invested_value"] = np.r_[np.array(full[:-1]), full[-1]]
    h_half["invested_value"] = np.r_[0.5 * np.array(half[:-1]), 0.5 * half[-1]]
    t = dates[-1]
    a = decide_exposure(h_full, PRESETS["vt20"], decision_date=t)
    b = decide_exposure(h_half, PRESETS["vt20"], decision_date=t)
    assert a.realised_vol_annual == pytest.approx(b.realised_vol_annual, rel=1e-9)
    assert a.realised_vol_annual > 0.3
    assert a.multiplier == b.multiplier < 1.0


def test_vol_warmup_is_declared_not_silent():
    hist = _history(_random_nav(n=10))
    d = decide_exposure(hist, PRESETS["vt20"], decision_date=hist["trade_date"].iloc[-1])
    assert d.vol_status == "warmup"
    assert d.multiplier == PRESETS["vt20"].vol_warmup_multiplier
    assert "vol_warmup_declared_multiplier" in d.reasons
    inception = _history([1_000_000.0])
    first = decide_exposure(inception, PRESETS["vt20_ddl"], decision_date=inception["trade_date"].iloc[0])
    assert first.status == STATUS_OK and first.vol_observations == 0 and first.multiplier == 0.5


def test_cash_days_are_unmeasured_not_zero_returns():
    navs = np.full(100, 1_000_000.0)
    dates = pd.bdate_range("2020-01-01", periods=100)
    hist = account_history_frame(dates, navs, np.zeros(100))  # all cash
    d = decide_exposure(hist, PRESETS["vt20"], decision_date=dates[-1])
    assert d.vol_status == "warmup" and d.vol_observations == 0


# ---------------------------------------------------------------- config
def test_zero_floor_and_unordered_ladders_are_refused():
    with pytest.raises(ExposurePolicyError, match="cash-lock"):
        ExposurePolicy(name="x", ladder_thresholds=(0.1,), ladder_multipliers=(0.0,))
    with pytest.raises(ExposurePolicyError):
        ExposurePolicy(name="x", ladder_thresholds=(0.12, 0.08), ladder_multipliers=(0.5, 0.25))
    with pytest.raises(ExposurePolicyError):
        ExposurePolicy(name="x", ladder_thresholds=(0.08, 0.12), ladder_multipliers=(0.5, 0.75))
    with pytest.raises(ExposurePolicyError):
        ExposurePolicy(name="x", vol_target_annual=0.2, vol_warmup_multiplier=0.0)


def test_ladder_must_sit_below_the_hard_latch():
    for policy in PRESETS.values():
        policy.validate_backstop(0.20)  # shipped paper RiskLimits.max_drawdown
    deep = ExposurePolicy(name="deep", ladder_thresholds=(0.10, 0.20), ladder_multipliers=(0.5, 0.25))
    with pytest.raises(ExposurePolicyError, match="hard drawdown latch"):
        deep.validate_backstop(0.20)


def test_policy_loading_and_identity():
    assert load_exposure_policy("vt20_ddl") is PRESETS["vt20_ddl"]
    inline = load_exposure_policy(json.dumps(PRESETS["vt20_ddl"].to_dict()))
    assert inline == PRESETS["vt20_ddl"] and inline.sha256 == PRESETS["vt20_ddl"].sha256
    with pytest.raises(ExposurePolicyError, match="unknown exposure policy fields"):
        load_exposure_policy({"name": "x", "vol_target": 0.2})
    with pytest.raises(ExposurePolicyError):
        load_exposure_policy("no_such_preset")


def test_decisions_are_deterministic_and_json_safe():
    hist = _history(_random_nav(n=300, seed=11))
    t = hist["trade_date"].iloc[-1]
    a = decide_exposure(hist, PRESETS["vt20_ddl"], decision_date=t).to_dict()
    b = decide_exposure(hist.copy(), PRESETS["vt20_ddl"], decision_date=t).to_dict()
    assert a == b
    text = json.dumps(a, allow_nan=False)
    assert json.loads(text)["policy_sha256"] == PRESETS["vt20_ddl"].sha256
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in a.values())
