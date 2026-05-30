"""Tests for the Stage 4.2 policy time-lag estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.policy import (
    TimeLagConfig,
    apply_policy_lag_features,
    estimate_policy_lag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "event_id" not in df.columns:
        df["event_id"] = [f"e{i:03d}" for i in range(len(df))]
    if "policy_strength" not in df.columns:
        df["policy_strength"] = 0.7
    return df


def _sector_panel(n_days: int = 60, sectors: tuple[str, ...] = ("Bank", "Tech")) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    rng = np.random.default_rng(0)
    for s in sectors:
        for d in dates:
            rows.append({"trade_date": d, "sector": s, "ret": float(rng.normal(0.0, 0.005))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Setup detection / single-theme curve
# ---------------------------------------------------------------------------

def test_single_theme_curve_peaks_near_engineered_lag():
    """Inject a +1% sector return exactly 3 business days after each
    announcement; the estimator should pick lag=3 as the peak.
    """
    base = _sector_panel(n_days=80, sectors=("Bank",))
    base["ret"] = 0.0
    events = _make_events(
        [
            {"announced_at": pd.Timestamp("2024-01-10"), "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": pd.Timestamp("2024-01-25"), "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": pd.Timestamp("2024-02-12"), "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": pd.Timestamp("2024-03-04"), "themes": ["monetary"], "sectors_hint": ["Bank"]},
        ]
    )
    # On t+3 of each event, give Bank a +1% jump
    for _, ev in events.iterrows():
        target_dt = ev["announced_at"] + pd.tseries.offsets.BDay(3)
        mask = (base["trade_date"] == target_dt) & (base["sector"] == "Bank")
        base.loc[mask, "ret"] = 0.01

    result = estimate_policy_lag(events, base, market_returns=None,
                                  config=TimeLagConfig(max_lag_days=10, min_events_per_theme=2))
    assert "monetary" in result.best_lag
    best_k, _ = result.best_lag["monetary"]
    assert best_k == 3


def test_lag_curve_returns_one_row_per_lag_per_theme():
    panel = _sector_panel(n_days=80)
    events = _make_events(
        [
            {"announced_at": "2024-01-10", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-01-15", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-02-01", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-02-15", "themes": ["monetary"], "sectors_hint": ["Bank"]},
        ]
    )
    result = estimate_policy_lag(events, panel, config=TimeLagConfig(max_lag_days=8, min_events_per_theme=2))
    assert len(result.lag_curves) == 8
    assert set(result.lag_curves["theme"]) == {"monetary"}


def test_themes_below_min_events_dropped():
    panel = _sector_panel(n_days=60)
    events = _make_events(
        [
            {"announced_at": "2024-01-10", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-01-15", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-01-20", "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": "2024-01-25", "themes": ["fiscal"], "sectors_hint": ["Bank"]},  # only 1 fiscal event
        ]
    )
    result = estimate_policy_lag(events, panel, config=TimeLagConfig(min_events_per_theme=3))
    assert "monetary" in result.best_lag
    assert "fiscal" not in result.best_lag


def test_empty_events_returns_empty_result():
    panel = _sector_panel()
    result = estimate_policy_lag(pd.DataFrame(), panel)
    assert result.lag_curves.empty
    assert result.best_lag == {}


def test_empty_panel_returns_empty_result():
    events = _make_events(
        [{"announced_at": "2024-01-10", "themes": ["monetary"], "sectors_hint": ["Bank"]}]
    )
    result = estimate_policy_lag(events, pd.DataFrame())
    assert result.lag_curves.empty


# ---------------------------------------------------------------------------
# Excess vs market
# ---------------------------------------------------------------------------

def test_excess_returns_subtract_market():
    """When market_returns is provided, the estimator must work on
    sector minus market — so a sector that just tracks the market has
    zero excess and no lag signal.
    """
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    panel = pd.DataFrame({"trade_date": dates, "sector": "Bank", "ret": 0.005})
    market = pd.Series(0.005, index=dates)  # market matches sector exactly
    events = _make_events(
        [
            {"announced_at": dates[10], "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": dates[20], "themes": ["monetary"], "sectors_hint": ["Bank"]},
            {"announced_at": dates[30], "themes": ["monetary"], "sectors_hint": ["Bank"]},
        ]
    )
    result = estimate_policy_lag(
        events,
        panel,
        market_returns=market,
        config=TimeLagConfig(min_events_per_theme=2, apply_event_count_weight=False),
    )
    # All excess = 0 → curve is flat at 0
    if not result.lag_curves.empty:
        assert (result.lag_curves["mean_excess"].abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# Per-event policy-strength weighting
# ---------------------------------------------------------------------------

def test_policy_strength_weighted_events_contribute_more():
    panel = _sector_panel(n_days=80, sectors=("Bank",))
    panel["ret"] = 0.0
    events = _make_events(
        [
            {"announced_at": "2024-01-10", "themes": ["monetary"], "sectors_hint": ["Bank"], "policy_strength": 1.0},
            {"announced_at": "2024-01-25", "themes": ["monetary"], "sectors_hint": ["Bank"], "policy_strength": 0.1},
            {"announced_at": "2024-02-12", "themes": ["monetary"], "sectors_hint": ["Bank"], "policy_strength": 1.0},
        ]
    )
    # Inject a +2% jump 5 days after the high-strength events only
    for _, ev in events.iterrows():
        if ev["policy_strength"] == 1.0:
            dt = pd.Timestamp(ev["announced_at"]) + pd.tseries.offsets.BDay(5)
            panel.loc[(panel["trade_date"] == dt) & (panel["sector"] == "Bank"), "ret"] = 0.02
    result = estimate_policy_lag(events, panel, config=TimeLagConfig(min_events_per_theme=2))
    best_k, _ = result.best_lag["monetary"]
    # Strength weighting should let lag=5 still win even though one weak
    # event contributed noise at other lags.
    assert best_k == 5


# ---------------------------------------------------------------------------
# Feature application
# ---------------------------------------------------------------------------

def test_apply_lag_features_adds_policy_signal_columns():
    panel = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=40).repeat(2),
            "symbol": ["B.SZ", "T.SH"] * 40,
            "sector_level_1": ["Bank", "Tech"] * 40,
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "announced_at": pd.Timestamp("2024-01-10"),
                "themes": ["monetary"],
                "sectors_hint": ["Bank"],
                "policy_strength": 1.0,
            },
            {
                "event_id": "e2",
                "announced_at": pd.Timestamp("2024-01-20"),
                "themes": ["fiscal"],
                "sectors_hint": ["Tech"],
                "policy_strength": 0.7,
            },
        ]
    )
    out = apply_policy_lag_features(
        panel,
        events,
        lag_table={"monetary": (3, 0.01), "fiscal": (5, 0.005)},
    )
    assert "policy_signal_monetary" in out.columns
    assert "policy_signal_fiscal" in out.columns


def test_apply_lag_features_pit_safe_no_future_leak():
    """A Bank row on 2024-01-10 must NOT see a policy_signal for an
    event announced 2024-01-15 (PIT safety).
    """
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-25")],
            "symbol": ["B.SZ", "B.SZ"],
            "sector_level_1": ["Bank", "Bank"],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "announced_at": pd.Timestamp("2024-01-15"),
                "themes": ["monetary"],
                "sectors_hint": ["Bank"],
                "policy_strength": 1.0,
            }
        ]
    )
    out = apply_policy_lag_features(
        panel, events, lag_table={"monetary": (3, 0.01)}
    )
    # 2024-01-10 row: announce 2024-01-15 + 3bd lag = ~2024-01-18 → not seen → 0
    # 2024-01-25 row: signal effective by 2024-01-18 → should be non-zero
    by_date = out.set_index("trade_date")["policy_signal_monetary"].astype(float)
    assert by_date.loc["2024-01-10"] == 0.0
    assert by_date.loc["2024-01-25"] > 0.0


def test_apply_lag_features_uses_default_lag_for_unknown_theme():
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-25")],
            "symbol": ["B.SZ"],
            "sector_level_1": ["Bank"],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "announced_at": pd.Timestamp("2024-01-15"),
                "themes": ["unknown_theme"],
                "sectors_hint": ["Bank"],
                "policy_strength": 1.0,
            }
        ]
    )
    out = apply_policy_lag_features(panel, events, lag_table=None, default_lag=5)
    assert "policy_signal_unknown_theme" in out.columns
    assert float(out["policy_signal_unknown_theme"].iloc[0]) > 0.0


def test_apply_lag_features_empty_inputs_pass_through():
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-10")],
            "symbol": ["B.SZ"],
            "sector_level_1": ["Bank"],
        }
    )
    out = apply_policy_lag_features(panel, pd.DataFrame())
    # No events → no new columns added; original panel returned
    assert "policy_signal_monetary" not in out.columns
    out2 = apply_policy_lag_features(pd.DataFrame(), pd.DataFrame())
    assert out2.empty


def test_apply_lag_features_all_sector_event_broadcasts_to_every_symbol():
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-25")] * 2,
            "symbol": ["B.SZ", "T.SH"],
            "sector_level_1": ["Bank", "Tech"],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "announced_at": pd.Timestamp("2024-01-15"),
                "themes": ["macro"],
                "sectors_hint": [],
                "policy_strength": 0.9,
            }
        ]
    )
    out = apply_policy_lag_features(panel, events, lag_table={"macro": (3, 0.01)})
    # _ALL_ event hits both symbols
    assert (out["policy_signal_macro"] > 0).all()
