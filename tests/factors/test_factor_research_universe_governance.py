from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from quantagent.factors.universe_membership import UniverseMembershipError
from scripts.run_factor_research_cycle_governed import run_governed_cycle


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-02", periods=45)
    for day_idx, date in enumerate(dates):
        for symbol_idx in range(12):
            close = 10.0 + symbol_idx * 0.2 + day_idx * (0.01 + symbol_idx * 0.001)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"S{symbol_idx:03d}",
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000.0,
                    "amount": close * 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _base_args(panel, calendar, output) -> argparse.Namespace:
    return argparse.Namespace(
        market_panel=str(panel),
        market_calendar=str(calendar),
        provider="none",
        symbols="",
        start_date="2025-01-01",
        end_date="2025-12-31",
        factors="rsi_14",
        max_factors=1,
        horizons="1,5,10",
        target_horizon=5,
        target_book_cny=10_000_000.0,
        max_adv_participation=0.10,
        cluster_correlation=0.85,
        output_dir=str(output),
        universe_mode="research_universe_explicit_static",
        universe_membership="",
        universe_id="",
    )


def _write_inputs(tmp_path):
    panel_frame = _panel()
    panel = tmp_path / "market.csv"
    panel_frame.to_csv(panel, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"trade_date": sorted(panel_frame["trade_date"].unique())}).to_csv(
        calendar,
        index=False,
    )
    return panel_frame, panel, calendar


def _write_membership(path, panel_frame: pd.DataFrame) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(panel_frame["trade_date"].unique()))
    pivot = pd.Timestamp(dates[20]).normalize()
    before = pd.Timestamp(dates[0]).normalize()
    rows: list[dict[str, object]] = []
    for symbol_idx in range(12):
        symbol = f"S{symbol_idx:03d}"
        if symbol_idx == 0:
            start, finish, reason = pivot, None, "entered_index"
        elif symbol_idx == 1:
            start, finish, reason = before, pivot - pd.Timedelta(days=1), "delisted"
        else:
            start, finish, reason = before, None, "continuous_member"
        rows.append(
            {
                "symbol": symbol,
                "effective_from": start.date().isoformat(),
                "effective_to": None if finish is None else finish.date().isoformat(),
                "universe_id": "test_pit_universe",
                "source": "test_membership_master",
                "source_version": "v2025-01",
                "available_at": (start - pd.Timedelta(days=1)).strftime("%Y-%m-%dT18:00:00+08:00"),
                "membership_reason": reason,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return pivot


def test_static_universe_is_explicitly_non_pit_and_non_generalising(tmp_path) -> None:
    _panel_frame, panel, calendar = _write_inputs(tmp_path)
    output = tmp_path / "static"
    args = _base_args(panel, calendar, output)
    manifest = run_governed_cycle(args)

    universe = manifest["research_universe"]
    assert manifest["schema_version"] == "factor_research_cycle_v4_pit_universe"
    assert universe["mode"] == "research_universe_explicit_static"
    assert universe["point_in_time_membership"] is False
    assert universe["survivorship_bias_possible"] is True
    assert universe["broad_market_generalization_certified"] is False
    assert manifest["economic_live_eligible"] is False
    assert manifest["automatic_factor_activation"] is False

    validity = json.loads((output / "factor_validity.json").read_text())
    assert validity[0]["research_universe_mode"] == "research_universe_explicit_static"
    assert validity[0]["research_universe_contract_sha256"] == universe["membership_contract_sha256"]


def test_pit_universe_filters_entry_and_exit_before_factor_calculation(tmp_path) -> None:
    panel_frame, panel, calendar = _write_inputs(tmp_path)
    membership = tmp_path / "membership.csv"
    pivot = _write_membership(membership, panel_frame)

    output = tmp_path / "pit"
    args = _base_args(panel, calendar, output)
    args.universe_mode = "point_in_time_membership"
    args.universe_membership = str(membership)
    args.universe_id = "test_pit_universe"
    manifest = run_governed_cycle(args)

    universe = manifest["research_universe"]
    assert universe["point_in_time_membership"] is True
    assert universe["survivorship_bias_from_current_membership_blocked"] is True
    assert universe["membership_source_sha256"]
    assert universe["membership_contract_sha256"]
    assert universe["market_membership_coverage"]["complete"] is True

    filtered = pd.read_csv(output / "market_panel.csv", parse_dates=["trade_date"])
    assert filtered.loc[filtered["trade_date"] < pivot, "symbol"].ne("S000").all()
    assert filtered.loc[filtered["trade_date"] >= pivot, "symbol"].ne("S001").all()
    assert (filtered.loc[filtered["trade_date"] < pivot, "symbol"] == "S001").any()
    assert (filtered.loc[filtered["trade_date"] >= pivot, "symbol"] == "S000").any()

    membership_artifact = pd.read_csv(output / "universe_membership.csv")
    assert set(membership_artifact["universe_id"]) == {"test_pit_universe"}
    assert manifest["research_degrees_of_freedom"]["universe_choice_counted"] is True
    assert manifest["universe_membership_sha_required_for_comparison"] is True


def test_pit_universe_rejects_missing_active_member_market_rows(tmp_path) -> None:
    panel_frame, panel, calendar = _write_inputs(tmp_path)
    membership = tmp_path / "membership.csv"
    _write_membership(membership, panel_frame)

    broken = pd.read_csv(panel, parse_dates=["trade_date"])
    target_date = broken["trade_date"].min()
    broken = broken[
        ~((broken["trade_date"] == target_date) & (broken["symbol"] == "S002"))
    ]
    broken.to_csv(panel, index=False)

    args = _base_args(panel, calendar, tmp_path / "broken")
    args.universe_mode = "point_in_time_membership"
    args.universe_membership = str(membership)
    args.universe_id = "test_pit_universe"

    with pytest.raises(UniverseMembershipError, match="market coverage is incomplete"):
        run_governed_cycle(args)
