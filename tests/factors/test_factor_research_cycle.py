from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

from scripts.run_factor_research_cycle import run_cycle


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


def _args(panel, output, *, market_calendar: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        market_panel=str(panel),
        market_calendar=market_calendar,
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
    )


def test_provided_panel_requires_independent_market_calendar(tmp_path) -> None:
    panel = tmp_path / "market.csv"
    _panel().to_csv(panel, index=False)

    with pytest.raises(ValueError, match="--market-calendar"):
        run_cycle(_args(panel, tmp_path / "out"))


def test_provided_panel_cycle_never_self_certifies_or_activates_factor(tmp_path) -> None:
    panel_frame = _panel()
    panel = tmp_path / "market.csv"
    panel_frame.to_csv(panel, index=False)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"trade_date": sorted(panel_frame["trade_date"].unique())}).to_csv(
        calendar,
        index=False,
    )
    output = tmp_path / "out"
    args = _args(panel, output, market_calendar=str(calendar))
    manifest = run_cycle(args)

    assert manifest["schema_version"] == "factor_research_cycle_v3_direction_provenance"
    assert manifest["research_only"] is True
    assert manifest["economic_live_eligible"] is False
    assert manifest["automatic_factor_activation"] is False
    assert manifest["source"]["production_integrity_certified"] is False
    assert manifest["market_calendar"]["production_integrity_certified"] is False
    assert manifest["materialised_factors"] == ["rsi_14"]
    assert manifest["factor_direction_contracts"] == {"rsi_14": "negative"}
    assert manifest["factor_values_direction_aligned"] is True

    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    validity = json.loads((output / "factor_validity.json").read_text(encoding="utf-8"))
    lifecycle = json.loads(
        (output / "factor_lifecycle_diagnostics.json").read_text(encoding="utf-8")
    )
    assert saved["research_only"] is True
    assert saved["factor_direction_contracts"] == {"rsi_14": "negative"}
    assert validity[0]["promotion_candidate_ready"] is False
    assert validity[0]["activation_authorized"] is False
    assert "promotion_context_missing" in validity[0]["promotion_blockers"]

    # The research panel is already sign-aligned before lifecycle diagnostics, so
    # its effective input direction remains positive. Preserve the registry's raw
    # factor contract separately to make the artifact auditable without applying
    # the negative sign twice.
    assert lifecycle[0]["expected_direction"] == "positive"
    assert lifecycle[0]["registry_expected_direction"] == "negative"
    assert lifecycle[0]["factor_values_direction_aligned"] is True
