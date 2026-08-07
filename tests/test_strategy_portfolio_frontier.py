from __future__ import annotations

import pandas as pd

from quantagent.cli.v7_train import (
    _select_portfolio_frontier,
    _split_portfolio_selection_holdout,
)


def test_portfolio_selection_and_final_holdout_are_chronologically_disjoint() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=20).repeat(2),
        "symbol": ["000001.SZ", "600000.SH"] * 20,
        "prediction": range(40),
    })

    selection, holdout, evidence = _split_portfolio_selection_holdout(frame)

    assert selection["trade_date"].max() < holdout["trade_date"].min()
    assert evidence["selectionEnd"] < evidence["holdoutStart"]
    assert len(set(selection["trade_date"]) & set(holdout["trade_date"])) == 0


def test_label_horizon_is_purged_at_the_selection_holdout_seam() -> None:
    """Adjacent segments leak: the last selection label resolves in the holdout.

    Every other train/test boundary in the pipeline purges the label horizon.
    This one did not, so a 20-day forward label attached to the final selection
    day was still resolving twenty days into the frozen holdout — and the book
    the selector chose was still being held there.
    """
    frame = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=60).repeat(2),
        "symbol": ["000001.SZ", "600000.SH"] * 60,
        "prediction": range(120),
        "horizon": 20,
    })

    selection, holdout, evidence = _split_portfolio_selection_holdout(frame)

    assert evidence["purgeDays"] == 20
    gap_days = (holdout["trade_date"].min() - selection["trade_date"].max()).days
    assert gap_days > 20
    # The purged span belongs to neither segment.
    purged = set(pd.date_range(evidence["purgeStart"], evidence["purgeEnd"]))
    assert not purged & set(selection["trade_date"])
    assert not purged & set(holdout["trade_date"])


def test_split_without_a_horizon_column_keeps_the_legacy_gap_free_behaviour() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=20).repeat(2),
        "symbol": ["000001.SZ", "600000.SH"] * 20,
        "prediction": range(40),
    })

    _, _, evidence = _split_portfolio_selection_holdout(frame)

    assert evidence["purgeDays"] == 0
    assert evidence["purgeStart"] is None


def test_pareto_frontier_removes_dominated_candidate_and_respects_preferences() -> None:
    candidates = [
        {"id": "dominated", "metrics": {"excess_return": 0.08, "annualized_return": 0.10, "max_drawdown": -0.18}},
        {"id": "balanced", "metrics": {"excess_return": 0.12, "annualized_return": 0.14, "max_drawdown": -0.10}},
        {"id": "return", "metrics": {"excess_return": 0.18, "annualized_return": 0.22, "max_drawdown": -0.16}},
        {"id": "defensive", "metrics": {"excess_return": 0.09, "annualized_return": 0.11, "max_drawdown": -0.05}},
    ]

    champion, frontier = _select_portfolio_frontier(
        candidates,
        objective_weights=(0.45, 0.35, 0.20),
    )

    assert champion["id"] in {"balanced", "return", "defensive"}
    assert "dominated" not in {item["id"] for item in frontier}
    assert all(item["paretoOptimal"] is True for item in frontier)
