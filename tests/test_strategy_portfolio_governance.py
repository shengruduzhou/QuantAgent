from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.cli.v7_train import _apply_stock_selection_mode
from quantagent.research.selection_governance import (
    NestedSelectionConfig,
    evaluate_frozen_candidate,
)


def test_fundamental_blend_uses_pit_rank_and_declared_weight() -> None:
    dates = pd.to_datetime(["2025-03-03", "2025-03-04"])
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    predictions = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "prediction": float(symbol_index + 1),
            }
            for date in dates
            for symbol_index, symbol in enumerate(symbols)
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "trade_date": dates[0],
                "available_at": "2025-02-28",
                "symbol": symbol,
                "pe_ttm": float(8 + symbol_index * 4),
                "pb": float(1 + symbol_index),
                "ps_ttm": float(1 + symbol_index / 2),
                "roe": float(0.24 - symbol_index * 0.03),
                "gross_margin": float(0.50 - symbol_index * 0.04),
                "operating_cf_to_net_income": float(1.2 - symbol_index * 0.1),
                "revenue_yoy": float(0.20 - symbol_index * 0.02),
                "net_income_yoy": float(0.18 - symbol_index * 0.02),
            }
            for symbol_index, symbol in enumerate(symbols)
        ]
    )
    # A future disclosure must never affect either requested as-of date.
    fundamentals = pd.concat(
        [
            fundamentals,
            pd.DataFrame(
                [{
                    **fundamentals.iloc[0].to_dict(),
                    "available_at": "2025-03-20",
                    "pe_ttm": 0.1,
                    "roe": 9.0,
                }]
            ),
        ],
        ignore_index=True,
    )

    selected, evidence = _apply_stock_selection_mode(
        predictions,
        fundamentals,
        "fundamental",
        threshold=0.25,
        blend_weight=0.60,
    )

    assert not selected.empty
    assert selected["prediction"].between(0.0, 1.0).all()
    assert evidence["fundamentalBlendWeight"] == 0.60
    assert evidence["modelRankWeight"] == pytest.approx(0.40)
    assert evidence["pitPolicy"] == "latest available_at <= prediction trade_date"
    assert evidence["rowsAfter"] <= evidence["rowsBefore"]


def test_frozen_candidate_gate_records_and_rejects_overfit_evidence() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-02", periods=160, freq="B")
    returns = pd.DataFrame(
        {
            "champion": rng.normal(0.0002, 0.012, len(index)),
            "challenger_a": rng.normal(0.0001, 0.012, len(index)),
            "challenger_b": rng.normal(0.0001, 0.012, len(index)),
        },
        index=index,
    )

    report = evaluate_frozen_candidate(
        returns,
        selected_candidate="champion",
        config=NestedSelectionConfig(
            max_pbo=0.0,
            min_dsr_probability=1.0,
            max_spa_pvalue=0.0,
            max_losing_outer_fold_rate=0.0,
        ),
        cumulative_trials=50,
        minimum_observed_days=80,
    )

    payload = report.to_dict()
    assert payload["selected_candidate"] == "champion"
    assert payload["observed_days"] == 160
    assert payload["cumulative_trials"] == 50
    assert payload["accepted"] is False
    assert payload["rejection_reasons"]


def test_frozen_candidate_gate_requires_a_real_early_oos_window() -> None:
    returns = pd.DataFrame(
        {"a": [0.01] * 20, "b": [0.0] * 20},
        index=pd.date_range("2025-01-02", periods=20, freq="B"),
    )
    with pytest.raises(ValueError, match="insufficient early OOS days"):
        evaluate_frozen_candidate(
            returns,
            selected_candidate="a",
            minimum_observed_days=80,
        )
