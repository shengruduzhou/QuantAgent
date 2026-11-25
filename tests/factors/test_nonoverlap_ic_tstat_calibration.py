"""ROADMAP P1-6 / GOVERNANCE R2: size of the IC t-stat under a true null.

Null: a persistent AR(1) factor (phi = 0.98, i.e. like a slow price/fundamental
signal) that is independent of i.i.d. daily returns; the label is the h-day
forward return, so consecutive per-date ICs share h-1 days of returns.

Measured with this seed (600 panels of 750 sessions x 50 names):

    h   HAC(lag>=h-1) |t|>1.96   non-overlap cohort-avg |t|>1.96   cohort |t|>=3
    5         7.7%                       3.2%                        0.17%
    20       11.8%                       3.3%                        0.17%

The HAC t is oversized (known since Round 20); the non-overlapping cohort
average is at or below nominal, which is why screens use it with |t| >= 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.evaluation import cross_sectional_rank_ic, information_coefficient
from quantagent.factors.governance_metrics import FactorGateConfig, evaluate_factor_candidate
from quantagent.factors.lifecycle import recommend_factor_status
from quantagent.quant_math.performance import newey_west_t_stat, nonoverlapping_cohort_t_stat

SEED = 20261125
SESSIONS = 750
NAMES = 50
PHI = 0.98
PANELS = 600


def _null_panel(rng: np.random.Generator, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    total = SESSIONS + horizon
    factor = np.empty((total, NAMES))
    factor[0] = rng.normal(size=NAMES)
    shocks = rng.normal(size=(total, NAMES)) * np.sqrt(1.0 - PHI**2)
    for t in range(1, total):
        factor[t] = PHI * factor[t - 1] + shocks[t]
    cumulative = np.cumsum(rng.normal(size=(total, NAMES)), axis=0)
    label = cumulative[horizon:total] - cumulative[0:SESSIONS]  # r_{t+1..t+h}
    return factor[:SESSIONS], label


def _rank_ic_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = x.argsort(1).argsort(1).astype(float)
    ry = y.argsort(1).argsort(1).astype(float)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx**2).sum(1) * (ry**2).sum(1))


def _rejection_rates(horizon: int) -> dict[str, float]:
    rng = np.random.default_rng(SEED + horizon)
    hac, cohort = [], []
    for _ in range(PANELS):
        x, y = _null_panel(rng, horizon)
        ic = pd.Series(_rank_ic_rows(x, y))
        hac.append(newey_west_t_stat(ic, horizon_days=horizon))
        cohort.append(nonoverlapping_cohort_t_stat(ic, horizon))
    hac_t, cohort_t = np.abs(hac), np.abs(cohort)
    return {
        "hac_5pct": float((hac_t > 1.96).mean()),
        "cohort_5pct": float((cohort_t > 1.96).mean()),
        "cohort_t3": float((cohort_t >= 3.0).mean()),
    }


def test_vectorised_ic_matches_authoritative_rank_ic() -> None:
    rng = np.random.default_rng(SEED)
    x, y = _null_panel(rng, 5)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(np.arange(SESSIONS), NAMES),
            "f": x.ravel(),
            "y": y.ravel(),
        }
    )
    expected = cross_sectional_rank_ic(frame, "f", "y", sort=True).to_numpy()
    assert np.max(np.abs(_rank_ic_rows(x, y) - expected)) < 1e-12


@pytest.mark.parametrize(
    ("horizon", "hac_floor"),
    [(5, 0.06), (20, 0.085)],
)
def test_null_rejection_rate_hac_vs_nonoverlap(horizon: int, hac_floor: float) -> None:
    rates = _rejection_rates(horizon)
    # Non-overlapping cohort average: never oversized (binomial SE ~0.9pp).
    assert 0.015 <= rates["cohort_5pct"] <= 0.065, rates
    assert rates["cohort_t3"] <= 0.01, rates
    # HAC at lag h-1 stays oversized; if this starts failing the HAC estimator
    # changed and the documented calibration must be re-measured.
    assert rates["hac_5pct"] >= hac_floor, rates
    assert rates["hac_5pct"] > rates["cohort_5pct"] + 0.02, rates


def test_cohort_t_keeps_session_alignment_with_nan_placeholders() -> None:
    values = pd.Series([1.0, np.nan, 1.2, 0.0, 0.9, 0.1, 1.1, -0.1, 1.0, 0.05])
    # h=2: cohort 0 = [1.0, 1.2, 0.9, 1.1, 1.0]; cohort 1 = [0.0, 0.1, -0.1, 0.05]
    c0 = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
    c1 = np.array([0.0, 0.1, -0.1, 0.05])
    expected = np.mean([c.mean() / c.std(ddof=1) * np.sqrt(c.size) for c in (c0, c1)])
    assert nonoverlapping_cohort_t_stat(values, 2) == pytest.approx(expected)
    # Dropping the NaN first would shift every later session into the other cohort.
    assert nonoverlapping_cohort_t_stat(values.dropna().reset_index(drop=True), 2) != pytest.approx(expected)
    assert np.isnan(nonoverlapping_cohort_t_stat(pd.Series([1.0, 2.0]), 1))
    with pytest.raises(ValueError):
        nonoverlapping_cohort_t_stat(values, 0)


def test_information_coefficient_exposes_nonoverlap_t() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-02", periods=120)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 30),
            "f": rng.normal(size=120 * 30),
        }
    )
    frame["forward_return_5d"] = 0.3 * frame["f"] + rng.normal(size=len(frame))
    shuffled = frame.sample(frac=1.0, random_state=3)
    summary = information_coefficient(shuffled, "f", "forward_return_5d").summary
    ordered = information_coefficient(frame, "f", "forward_return_5d").summary
    assert summary.rank_nonoverlap_t_stat > 3.0
    # Cohorts follow the session order, not the row order of the input.
    assert summary.rank_nonoverlap_t_stat == pytest.approx(ordered.rank_nonoverlap_t_stat)
    with pytest.warns(RuntimeWarning):
        unnamed = information_coefficient(frame.rename(columns={"forward_return_5d": "ret"}), "f", "ret")
    assert np.isnan(unnamed.summary.rank_nonoverlap_t_stat)


def test_lifecycle_validation_requires_nonoverlap_t_of_three() -> None:
    kwargs = dict(
        rank_icir=0.8,
        positive_ratio=0.8,
        monotonicity=0.9,
        live_drift=0.2,
        max_existing_correlation=0.1,
        effective_dates=200,
        median_symbols_per_date=50,
        newey_west_rank_t_stat=4.0,
    )
    assert recommend_factor_status(**kwargs, nonoverlap_rank_t_stat=3.1) == "validated"
    assert recommend_factor_status(**kwargs, nonoverlap_rank_t_stat=2.9) == "watch"
    assert recommend_factor_status(**kwargs) == "watch"  # unknown is not a pass


def test_governance_gate_reports_and_applies_nonoverlap_t() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=160)
    rows = len(dates) * 40
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 40),
            "symbol": np.tile([f"S{i:03d}" for i in range(40)], len(dates)),
            "factor": rng.normal(size=rows),
            "adv20_cny": 1e9,
        }
    )
    frame["fwd_20d"] = 0.02 * frame["factor"] + rng.normal(size=rows)
    loose = dict(
        min_dates=30,
        min_symbols_per_date=10,
        min_mean_rank_ic=-1.0,
        min_ic_information_ratio=-10.0,
        min_newey_west_rank_t_stat=-10.0,
        min_positive_ic_ratio=0.0,
        max_losing_period_rate=1.0,
        max_recent_predictive_drift_z=1e9,
        min_decay_retention=-1e9,
        max_decay_reversal=-1e9,
        min_capacity_multiple=0.0,
    )
    report = evaluate_factor_candidate(
        frame,
        factor_name="factor",
        target_return_col="fwd_20d",
        decay_return_columns={20: "fwd_20d"},
        target_horizon_days=20,
        config=FactorGateConfig(**loose),
    )
    payload = report.to_dict()
    assert np.isfinite(payload["nonoverlap_rank_t_stat"])
    assert payload["nonoverlap_rank_t_stat"] < 3.0
    assert any(reason.startswith("nonoverlap_rank_t_stat=") for reason in report.rejection_reasons)
    relaxed = evaluate_factor_candidate(
        frame,
        factor_name="factor",
        target_return_col="fwd_20d",
        decay_return_columns={20: "fwd_20d"},
        target_horizon_days=20,
        config=FactorGateConfig(**loose, min_nonoverlap_rank_t_stat=-10.0),
    )
    assert not any(r.startswith("nonoverlap_rank_t_stat=") for r in relaxed.rejection_reasons)


def _frozen_rd_agent_nonoverlapping_ic_tstat(daily_ic: pd.Series, horizon: int) -> float:
    values = pd.to_numeric(pd.Series(daily_ic), errors="coerce").to_numpy(dtype=float)
    step = max(1, int(horizon))
    stats: list[float] = []
    for offset in range(step):
        cohort = values[offset::step]
        cohort = cohort[np.isfinite(cohort)]
        if cohort.size < 3:
            continue
        std = float(cohort.std(ddof=1))
        if std <= 1e-12:
            continue
        stats.append(float(cohort.mean()) / std * float(np.sqrt(cohort.size)))
    return float(np.mean(stats)) if stats else 0.0


@pytest.mark.parametrize("seed", range(20))
def test_rd_agent_gate_statistic_unchanged_by_delegation(seed: int) -> None:
    from quantagent.factors.factor_synthesis import _nonoverlapping_ic_tstat

    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 80))
    values = pd.Series(rng.normal(0.02, 0.1, n))
    values[rng.random(n) < 0.1] = np.nan
    if seed % 4 == 0:
        values[:] = 0.05  # zero spread -> skipped cohorts -> 0.0 fallback
    for horizon in (0, 1, 5, 20):
        assert _nonoverlapping_ic_tstat(values, horizon) == _frozen_rd_agent_nonoverlapping_ic_tstat(values, horizon)


def test_lifecycle_nonoverlap_t_keeps_ineligible_sessions_as_gaps() -> None:
    """Review item: dropping ineligible sessions shifted later ones across cohorts."""
    from quantagent.factors.lifecycle import build_factor_lifecycle_report

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-02", periods=150)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 30),
            "symbol": np.tile([f"S{i:03d}" for i in range(30)], len(dates)),
            "f": rng.normal(size=len(dates) * 30),
        }
    )
    frame["forward_return_5d"] = 0.15 * frame["f"] + rng.normal(size=len(frame))
    dropped = set(rng.choice(dates, size=45, replace=False))  # 30% ineligible
    frame.loc[frame["trade_date"].isin(dropped), "f"] = np.nan
    report = build_factor_lifecycle_report(frame, "f", "forward_return_5d")

    ic = cross_sectional_rank_ic(frame, "f", "forward_return_5d", sort=True)
    ic.index = pd.DatetimeIndex(ic.index)
    aligned = nonoverlapping_cohort_t_stat(ic.reindex(dates), 5)
    shifted = nonoverlapping_cohort_t_stat(ic.dropna().reset_index(drop=True), 5)
    assert aligned != pytest.approx(shifted)  # cohort membership matters here
    assert report.nonoverlap_t_stat == pytest.approx(aligned)
