"""ExtendedFundamentalRanker tests — 19-axis v8 spec section 4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.fundamental.extended_ranker import (
    EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS,
    ExtendedFundamentalConfig,
    ExtendedFundamentalRankerBuilder,
    build_extended_fundamental_ranker,
)


def _metrics_frame(n_symbols: int = 10) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-15")
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_symbols):
        rows.append({
            "symbol": f"60000{i:02d}.SH",
            "available_at": base,
            # valuation
            "pe_ttm": 10 + rng.uniform(0, 30),
            "pb": 1.0 + rng.uniform(0, 4),
            "ps_ttm": 0.5 + rng.uniform(0, 5),
            # profitability
            "roe": rng.uniform(-0.1, 0.30),
            "roa": rng.uniform(-0.05, 0.15),
            "gross_margin": rng.uniform(0.05, 0.60),
            "net_margin": rng.uniform(-0.05, 0.25),
            # growth
            "revenue_yoy": rng.uniform(-0.30, 0.50),
            "net_income_yoy": rng.uniform(-0.50, 0.80),
            # quality
            "operating_cashflow": rng.uniform(-0.10, 0.20),
            "accruals_quality": rng.uniform(0.0, 1.0),
            "earnings_surprise": rng.uniform(-0.20, 0.20),
            # leverage
            "debt_to_asset": rng.uniform(0.10, 0.80),
            "interest_coverage": rng.uniform(0.5, 20.0),
            # efficiency
            "inventory_turnover": rng.uniform(2.0, 15.0),
            "accounts_receivable_growth": rng.uniform(-0.30, 0.80),
            "goodwill_risk": rng.uniform(0.0, 0.30),
            # capital actions
            "dividend": rng.uniform(0.0, 0.05),
            "repurchase": rng.uniform(0.0, 0.03),
        })
    return pd.DataFrame(rows)


def _sector_map(n_symbols: int = 10, sector: str = "Semi") -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": f"60000{i:02d}.SH", "sector_level_1": sector}
         for i in range(n_symbols)]
    )


def test_empty_input_returns_canonical_schema():
    res = build_extended_fundamental_ranker(
        pd.DataFrame(), as_of_dates=[pd.Timestamp("2024-01-15")]
    )
    assert list(res.frame.columns) == list(EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS)
    assert len(res.frame) == 0


def test_canonical_columns_match_spec():
    metrics = _metrics_frame(8)
    sector_map = _sector_map(8)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    assert list(res.frame.columns) == list(EXTENDED_FUNDAMENTAL_REQUIRED_COLUMNS)


def test_composite_score_in_unit_interval():
    metrics = _metrics_frame(10)
    sector_map = _sector_map(10)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    composite = res.frame["composite_score"].dropna()
    assert (composite >= 0.0).all() and (composite <= 1.0).all()


def test_group_scores_emitted_for_each_block():
    metrics = _metrics_frame(10)
    sector_map = _sector_map(10)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    for group in ("valuation_score", "profitability_score", "growth_score",
                   "leverage_score", "efficiency_score", "quality_score",
                   "capital_action_score"):
        assert group in res.frame.columns
        # at least some non-null entries (since fixture supplies all axes)
        assert res.frame[group].notna().any()


def test_skips_bucket_below_min_universe():
    """Bucket size 3 → below default min 5 → skip."""
    metrics = _metrics_frame(3)
    sector_map = _sector_map(3)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
    )
    assert len(res.frame) == 0


def test_winsorization_dampens_outliers():
    # Inject a huge outlier in pe_ttm
    metrics = _metrics_frame(10)
    metrics.loc[0, "pe_ttm"] = 10_000.0
    sector_map = _sector_map(10)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    # Despite the outlier, valuation_score still in [0, 1]
    val_scores = res.frame["valuation_score"].dropna()
    assert (val_scores >= 0.0).all() and (val_scores <= 1.0).all()


def test_pit_discipline_only_visible_rows_pass():
    metrics = _metrics_frame(10)
    # Push half the rows' availability date past the as_of cutoff
    metrics.loc[:4, "available_at"] = pd.Timestamp("2030-01-01")
    sector_map = _sector_map(10)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=3),
    )
    # only 5 (the not-future-leaking) rows should survive — bucket may still
    # need at least min_universe_per_bucket items to score
    assert res.frame["symbol"].nunique() <= 5


def test_metric_completeness_lower_when_axes_missing():
    metrics = _metrics_frame(10)
    # drop two whole axis columns
    metrics = metrics.drop(columns=["dividend", "repurchase", "earnings_surprise"])
    sector_map = _sector_map(10)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
        sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    # capital_action_score (using dividend + repurchase) and quality (using earnings_surprise)
    # have fewer non-null axes → completeness < 1
    assert res.frame["metric_completeness"].max() < 1.0


def test_higher_roe_yields_higher_profitability_score():
    # Two-symbol pair: same everything except ROE
    base = pd.Timestamp("2024-01-15")
    metrics = pd.DataFrame([
        {"symbol": f"6000{i:02d}.SH", "available_at": base,
         "pe_ttm": 20.0, "pb": 2.0, "ps_ttm": 2.0,
         "roe": 0.05 + i * 0.05, "roa": 0.05, "gross_margin": 0.30, "net_margin": 0.10,
         "revenue_yoy": 0.10, "net_income_yoy": 0.10,
         "operating_cashflow": 0.10, "accruals_quality": 0.7, "earnings_surprise": 0.0,
         "debt_to_asset": 0.40, "interest_coverage": 5.0,
         "inventory_turnover": 8.0, "accounts_receivable_growth": 0.0, "goodwill_risk": 0.05,
         "dividend": 0.02, "repurchase": 0.01}
        for i in range(6)
    ])
    sector_map = _sector_map(6)
    res = build_extended_fundamental_ranker(
        metrics, as_of_dates=[base], sector_map=sector_map,
        config=ExtendedFundamentalConfig(min_universe_per_bucket=5),
    )
    res = res.frame.sort_values("symbol")
    prof = res["profitability_score"].tolist()
    assert prof[0] < prof[-1]  # higher ROE → higher profitability_score


def test_builder_writes_files(tmp_path):
    metrics = _metrics_frame(10)
    sector_map = _sector_map(10)
    builder = ExtendedFundamentalRankerBuilder(
        ExtendedFundamentalConfig(min_universe_per_bucket=5, output_root=tmp_path,
                                   source_version="t1")
    )
    res = builder.build(metrics, as_of_dates=[pd.Timestamp("2024-01-15")],
                         sector_map=sector_map)
    builder.write(res)
    assert (tmp_path / "silver" / "fundamental_extended" / "fundamental_extended.parquet").exists()
    assert (tmp_path / "manifests" / "fundamental_extended.json").exists()
