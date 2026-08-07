from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.models.interactions import _candidate_pairs, cross_sectional_rank_normalise


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=20)
    rows = []
    for date in dates:
        for symbol in range(20):
            row = {"trade_date": date, "symbol": f"S{symbol:02d}", "label": rng.normal()}
            for index in range(20):
                # f19 is deliberately all-NaN: an invalid standalone IC must not
                # poison the ordering or make results depend on input column order.
                row[f"f{index:02d}"] = np.nan if index == 19 else rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_candidate_budget_is_strict_and_order_independent():
    panel = _panel()
    columns = [f"f{index:02d}" for index in range(20)]
    ranked = cross_sectional_rank_normalise(panel, columns)
    labels = panel["label"]
    dates = panel["trade_date"]

    first = _candidate_pairs(ranked, labels, dates, columns, max_candidates=17)
    second = _candidate_pairs(ranked, labels, dates, list(reversed(columns)), max_candidates=17)

    assert len(first) == 17
    assert first == second
    assert len(set(first)) == 17


def test_zero_candidate_budget_means_zero_search_trials():
    panel = _panel()
    columns = ["f00", "f01", "f02"]
    ranked = cross_sectional_rank_normalise(panel, columns)

    assert _candidate_pairs(ranked, panel["label"], panel["trade_date"], columns, 0) == []
