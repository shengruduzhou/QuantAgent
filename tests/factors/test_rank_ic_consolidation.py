"""Cycle 32 P1-3: every per-date rank-IC call site delegates to
``factors.evaluation.cross_sectional_rank_ic`` and reproduces its former output.

Each ``_frozen_*`` function below is the private implementation as it stood
before consolidation (copied verbatim, only renamed).  The migrated call sites
are compared against them on randomized panels that contain NaNs, +/-inf, ties,
constant cross-sections, 0..6-name dates, all-NaN dates and unsorted rows.  A
difference larger than 1e-12 would mean the consolidation changed a number.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.evaluation import cross_sectional_rank_ic

pytestmark = pytest.mark.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-12
SEEDS = range(12)


# --------------------------------------------------------------------------
# Frozen references (pre-consolidation code, verbatim)
# --------------------------------------------------------------------------


def _frozen_evaluation_corr_by_date(frame, factor_column, return_column, date_column, method):
    values = {}
    for date, group in frame.groupby(date_column, sort=False):
        clean = group[[factor_column, return_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3 or clean[factor_column].nunique() <= 1 or clean[return_column].nunique() <= 1:
            values[pd.Timestamp(date)] = np.nan
        else:
            values[pd.Timestamp(date)] = float(clean[factor_column].corr(clean[return_column], method=method))
    return pd.Series(values, dtype=float)


def _frozen_interactions_daily_rank_ic(scores, labels, dates, *, min_names=5):
    frame = pd.DataFrame({"score": scores, "label": labels, "date": dates}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    values = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        if group["score"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        correlation = group["score"].rank().corr(group["label"].rank())
        if pd.notna(correlation):
            values[date] = float(correlation)
    return pd.Series(values, dtype=float)


def _frozen_stratified_rank_ic(group, label_col, min_n):
    if len(group) < min_n:
        return float("nan")
    p = pd.to_numeric(group["prediction"], errors="coerce")
    y = pd.to_numeric(group[label_col], errors="coerce")
    mask = p.notna() & y.notna()
    if mask.sum() < min_n:
        return float("nan")
    return float(p[mask].rank().corr(y[mask].rank()))


def _frozen_blender_daily_ic(subset, target_column):
    daily_ic = []
    for _, day in subset.dropna(subset=[target_column]).groupby("trade_date", sort=True):
        if len(day) < 3:
            continue
        value = day["prediction"].corr(day[target_column], method="spearman")
        if pd.notna(value):
            daily_ic.append(float(value))
    return daily_ic


def _frozen_blend_optimizer_per_date_rank_ic(merged, score_col, label_col):
    if score_col not in merged.columns or label_col not in merged.columns:
        return float("nan")
    sub = merged[["trade_date", score_col, label_col]].dropna()
    if sub.empty:
        return float("nan")
    grouped = sub.groupby("trade_date", sort=False)
    rows = []
    for _, g in grouped:
        if len(g) < 5:
            continue
        s = g[score_col].rank(method="average")
        r = g[label_col].rank(method="average")
        c = s.corr(r, method="pearson")
        if pd.notna(c):
            rows.append(float(c))
    return float(np.mean(rows)) if rows else float("nan")


def _frozen_governance_rank_ic_by_date(frame, factor_col, return_col):
    if frame.empty:
        return pd.Series(dtype=float)
    values = {}
    for date, group in frame.groupby("trade_date", sort=True):
        valid = group[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 3 or valid[factor_col].nunique() < 2 or valid[return_col].nunique() < 2:
            values[pd.Timestamp(date)] = float("nan")
        else:
            values[pd.Timestamp(date)] = float(valid[factor_col].corr(valid[return_col], method="spearman"))
    return pd.Series(values, dtype=float).dropna().sort_index()


def _frozen_synthesis_daily_rank_ic(factor_values, labels, trade_date, tradable_mask=None):
    data = {"trade_date": trade_date.values, "factor": factor_values.values, "label": labels.values}
    if tradable_mask is not None:
        data["tradable"] = np.asarray(tradable_mask, dtype=bool)
    df = pd.DataFrame(data)
    if tradable_mask is not None:
        df = df[df["tradable"]].drop(columns="tradable")
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float)
    grouped = df.groupby("trade_date")
    rank_factor = grouped["factor"].rank(method="average")
    rank_label = grouped["label"].rank(method="average")
    df = df.assign(rank_factor=rank_factor.values, rank_label=rank_label.values)
    return df.groupby("trade_date")[["rank_factor", "rank_label"]].corr().unstack().iloc[:, 1].dropna()


def _frozen_model_comparison_daily_rank_ic(frame, label_column, min_names):
    values = {}
    for date, group in frame.groupby("trade_date", sort=True):
        usable = group[["prediction", label_column]].dropna()
        if len(usable) < min_names:
            continue
        if usable["prediction"].nunique() < 2 or usable[label_column].nunique() < 2:
            continue
        correlation = usable["prediction"].rank().corr(usable[label_column].rank())
        if pd.notna(correlation):
            values[pd.Timestamp(date)] = float(correlation)
    return pd.Series(values, dtype=float).sort_index()


def _frozen_quality_gates_adverse_ic(subset, label_column):
    return subset.groupby("trade_date").apply(
        lambda f: float(f["prediction"].rank().corr(f[label_column].rank()))
        if len(f) >= 2 and f["prediction"].nunique() >= 2 and f[label_column].nunique() >= 2
        else float("nan")
    ).dropna()


def _frozen_v8_gated_mean_rank_ic(mj, target_label):
    ics = []
    for _, g in mj.groupby("trade_date"):
        if len(g) >= 20:
            c = g["composite_score"].rank().corr(g[target_label].rank())
            if pd.notna(c):
                ics.append(c)
    return float(sum(ics) / len(ics)) if ics else float("nan")


def _frozen_m5_per_date_rank_ic(frame, feature, label, min_names=2):
    valid = frame.dropna(subset=[feature, label])
    if valid.empty:
        return None
    per_date = valid.groupby("trade_date").apply(
        lambda f: f[feature].rank().corr(f[label].rank()) if len(f) >= min_names else np.nan,
        include_groups=False,
    ).dropna()
    return None if per_date.empty else round(float(per_date.mean()), 6)


def _frozen_discovered_daily_ic(values, labels, dates):
    df = pd.DataFrame({"d": dates.values, "f": values.values, "y": labels.values}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby("d")
    df = df.assign(rf=g["f"].rank(), ry=g["y"].rank())
    return df.groupby("d")[["rf", "ry"]].corr().unstack().iloc[:, 1].dropna()


def _frozen_full_universe_rank_ic(frame, factor, label):
    work = frame[["trade_date", factor, label]].dropna()
    if work.empty:
        return float("nan"), float("nan"), 0
    per_date = work.groupby("trade_date").apply(
        lambda g: g[factor].corr(g[label], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    )
    per_date = per_date.dropna()
    if per_date.empty:
        return float("nan"), float("nan"), 0
    return float(per_date.mean()), float(per_date.std(ddof=1)), int(len(per_date))


# --------------------------------------------------------------------------
# Randomized adversarial panels
# --------------------------------------------------------------------------


def _panel(seed: int, *, big: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = [0, 1, 2, 3, 4, 5, 6, 7, 25] + ([30, 40] if big else [])
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=14)
    for i, date in enumerate(dates):
        n = int(rng.choice(sizes))
        if i == 3:
            n = 8  # guaranteed constant-factor date
        for j in range(n):
            rows.append({"trade_date": date, "symbol": f"S{j:03d}"})
    frame = pd.DataFrame(rows, columns=["trade_date", "symbol"])
    m = len(frame)
    x = rng.normal(size=m)
    y = 0.3 * x + rng.normal(size=m)
    x = np.where(rng.random(m) < 0.3, np.round(x, 1), x)  # ties
    y = np.where(rng.random(m) < 0.2, np.round(y, 1), y)
    x[rng.random(m) < 0.12] = np.nan
    y[rng.random(m) < 0.12] = np.nan
    x[rng.random(m) < 0.03] = np.inf
    y[rng.random(m) < 0.03] = -np.inf
    frame["prediction"] = x
    frame["label"] = y
    const = frame["trade_date"] == dates[3]
    frame.loc[const, "prediction"] = 0.5
    allnan = frame["trade_date"] == dates[5]
    frame.loc[allnan, "label"] = np.nan
    frame["tradable"] = rng.random(m) > 0.15
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _assert_series_close(new: pd.Series, old: pd.Series) -> None:
    assert list(new.index) == list(old.index)
    a = new.to_numpy(dtype=float)
    b = old.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    finite = ~np.isnan(a)
    assert np.all(np.abs(a[finite] - b[finite]) <= TOL), np.max(np.abs(a[finite] - b[finite]))


def _assert_float_close(new: float | None, old: float | None) -> None:
    if old is None or new is None:
        assert old is None and new is None
        return
    if np.isnan(old):
        assert np.isnan(new)
    else:
        assert abs(new - old) <= TOL, (new, old)


def _load_script(relative: str):
    path = REPO / relative
    name = "cycle32_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Equivalence per migrated call site
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_information_coefficient_series_unchanged(seed: int) -> None:
    from quantagent.factors.evaluation import _corr_by_date

    frame = _panel(seed)
    for method in ("spearman", "pearson"):
        new = _corr_by_date(frame, "prediction", "label", "trade_date", method=method)
        old = _frozen_evaluation_corr_by_date(frame, "prediction", "label", "trade_date", method)
        _assert_series_close(new, old)


@pytest.mark.parametrize("seed", SEEDS)
def test_interactions_daily_rank_ic_unchanged(seed: int) -> None:
    from quantagent.models.interactions import _daily_rank_ic

    frame = _panel(seed)
    for min_names in (1, 3, 5):
        new = _daily_rank_ic(frame["prediction"], frame["label"], frame["trade_date"], min_names=min_names)
        old = _frozen_interactions_daily_rank_ic(
            frame["prediction"], frame["label"], frame["trade_date"], min_names=min_names
        )
        _assert_series_close(new, old)


@pytest.mark.parametrize("seed", SEEDS)
def test_stratified_bucket_ic_unchanged(seed: int) -> None:
    from quantagent.diagnostics.stratified_ic import _per_date_rank_ic

    frame = _panel(seed)
    for min_n in (2, 5):
        new = _per_date_rank_ic(frame, "label", min_n)
        old = frame.groupby("trade_date", group_keys=False).apply(
            lambda g: _frozen_stratified_rank_ic(g, "label", min_n)
        ).dropna()
        _assert_series_close(new, old)


@pytest.mark.parametrize("seed", SEEDS)
def test_multi_horizon_blender_daily_ic_unchanged(seed: int) -> None:
    from quantagent.portfolio.multi_horizon_blender import _daily_rank_ic_values

    frame = _panel(seed).dropna(subset=["prediction"])
    new = _daily_rank_ic_values(frame, "label")
    old = _frozen_blender_daily_ic(frame, "label")
    assert len(new) == len(old)
    assert np.all(np.abs(np.asarray(new) - np.asarray(old)) <= TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_blend_optimizer_rank_ic_unchanged(seed: int) -> None:
    from quantagent.ensemble.blend_optimizer import _per_date_rank_ic

    frame = _panel(seed, big=True)
    _assert_float_close(
        _per_date_rank_ic(frame, "prediction", "label"),
        _frozen_blend_optimizer_per_date_rank_ic(frame, "prediction", "label"),
    )
    assert np.isnan(_per_date_rank_ic(frame, "missing", "label"))


@pytest.mark.parametrize("seed", SEEDS)
def test_governance_rank_ic_by_date_unchanged(seed: int) -> None:
    from quantagent.factors.governance_metrics import _rank_ic_by_date

    frame = _panel(seed)
    _assert_series_close(
        _rank_ic_by_date(frame, "prediction", "label"),
        _frozen_governance_rank_ic_by_date(frame, "prediction", "label"),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_factor_synthesis_daily_rank_ic_unchanged(seed: int) -> None:
    from quantagent.factors.factor_synthesis import _daily_rank_ic

    frame = _panel(seed)
    for mask in (None, frame["tradable"]):
        new = _daily_rank_ic(frame["prediction"], frame["label"], frame["trade_date"], mask)
        old = _frozen_synthesis_daily_rank_ic(frame["prediction"], frame["label"], frame["trade_date"], mask)
        _assert_series_close(new, old)


@pytest.mark.parametrize("seed", SEEDS)
def test_model_comparison_daily_rank_ic_unchanged(seed: int) -> None:
    from quantagent.research.model_comparison import _daily_rank_ic

    frame = _panel(seed)
    for min_names in (2, 5):
        _assert_series_close(
            _daily_rank_ic(frame, "label", min_names),
            _frozen_model_comparison_daily_rank_ic(frame, "label", min_names),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_quality_gates_adverse_ic_unchanged(seed: int) -> None:
    from quantagent.data.v7_quality_gates import _adverse_dates_rank_ic

    frame = _panel(seed).dropna(subset=["trade_date", "prediction", "label"])
    _assert_series_close(_adverse_dates_rank_ic(frame, "label"), _frozen_quality_gates_adverse_ic(frame, "label"))


@pytest.mark.parametrize("seed", SEEDS)
def test_v8_gated_headline_rank_ic_unchanged(seed: int) -> None:
    from quantagent.cli.v8_gated import _mean_rank_ic

    frame = _panel(seed, big=True).rename(columns={"prediction": "composite_score"}).dropna()
    _assert_float_close(_mean_rank_ic(frame, "label"), _frozen_v8_gated_mean_rank_ic(frame, "label"))


@pytest.mark.parametrize("seed", SEEDS)
def test_m5_leakage_audit_rank_ic_unchanged(seed: int) -> None:
    module = _load_script("scripts/m5_leakage_audit.py")
    frame = _panel(seed, big=True)
    for min_names in (2, 20):
        _assert_float_close(
            module.per_date_rank_ic(frame, "prediction", "label", min_names=min_names),
            _frozen_m5_per_date_rank_ic(frame, "prediction", "label", min_names=min_names),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_evaluate_discovered_factors_daily_ic_unchanged(seed: int) -> None:
    module = _load_script("scripts/evaluate_discovered_factors.py")
    frame = _panel(seed)
    _assert_series_close(
        module._daily_ic(frame["prediction"], frame["label"], frame["trade_date"]),
        _frozen_discovered_daily_ic(frame["prediction"], frame["label"], frame["trade_date"]),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_full_universe_price_factor_rank_ic_unchanged(seed: int) -> None:
    module = _load_script("scripts/evaluate_full_universe_price_factors.py")
    frame = _panel(seed, big=True)
    new = module._rank_ic(frame, "prediction", "label")
    old = _frozen_full_universe_rank_ic(frame, "prediction", "label")
    _assert_float_close(new[0], old[0])
    _assert_float_close(new[1], old[1])
    assert new[2] == old[2]


def test_semantic_knobs_are_explicit() -> None:
    """The knobs that distinguish the historic copies change results as documented."""
    frame = pd.DataFrame(
        {
            "trade_date": ["d"] * 5,
            "x": [1.0, 2.0, 3.0, 4.0, np.inf],
            "y": [1.0, 3.0, 2.0, 5.0, 0.0],
        }
    )
    with_inf = cross_sectional_rank_ic(frame, "x", "y", inf_as_missing=False).iloc[0]
    without_inf = cross_sectional_rank_ic(frame, "x", "y").iloc[0]
    assert with_inf == pytest.approx(-0.1)  # inf ranked as the largest value
    assert without_inf == pytest.approx(0.8)  # inf row dropped
    assert np.isnan(cross_sectional_rank_ic(frame, "x", "y", min_names=6).iloc[0])
    assert cross_sectional_rank_ic(frame, "x", "y", min_names=6, drop_undefined=True).empty
