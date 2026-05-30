"""Tests for the Stage 2.2 silver st_flags → universe filter loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_silver_st_flags(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_legacy_market_features(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_manifest(path: Path, *, st_usable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extra": {
            "coverage_report": {
                "gate": {
                    "st_usable_for_risk_filter": bool(st_usable),
                    "reason": "test_gate",
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _preds(dates: list[str], symbols: list[str]) -> pd.DataFrame:
    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append({"trade_date": pd.Timestamp(date), "symbol": symbol})
    return pd.DataFrame(rows)


def test_silver_path_broadcasts_one_row_per_symbol_to_every_trade_date(tmp_path):
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    primary = tmp_path / "silver" / "st_flags" / "st_flags.parquet"
    _write_silver_st_flags(
        primary,
        [
            {"symbol": "000001.SZ", "is_st": True, "available_at": pd.Timestamp("2024-01-01")},
            {"symbol": "600519.SH", "is_st": False, "available_at": pd.Timestamp("2024-01-01")},
        ],
    )

    result = _load_st_flags_for_filter(
        primary_path=primary,
        legacy_path=tmp_path / "missing_legacy.parquet",
        manifest_path=tmp_path / "missing_manifest.json",
        prediction_frame=_preds(["2024-02-01", "2024-02-02"], ["000001.SZ", "600519.SH"]),
    )

    assert result is not None
    assert set(result.columns) == {"trade_date", "symbol", "is_st"}
    by_date = result.set_index(["trade_date", "symbol"])["is_st"]
    assert bool(by_date.loc[(pd.Timestamp("2024-02-01"), "000001.SZ")]) is True
    assert bool(by_date.loc[(pd.Timestamp("2024-02-02"), "000001.SZ")]) is True
    assert bool(by_date.loc[(pd.Timestamp("2024-02-01"), "600519.SH")]) is False


def test_legacy_path_used_when_silver_missing(tmp_path):
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    legacy = tmp_path / "silver" / "market_panel" / "market_features.parquet"
    _write_legacy_market_features(
        legacy,
        [
            {"trade_date": pd.Timestamp("2024-02-01"), "symbol": "000001.SZ", "is_st": True},
            {"trade_date": pd.Timestamp("2024-02-01"), "symbol": "600519.SH", "is_st": False},
        ],
    )

    result = _load_st_flags_for_filter(
        primary_path=tmp_path / "missing_silver.parquet",
        legacy_path=legacy,
        manifest_path=tmp_path / "missing_manifest.json",
        prediction_frame=_preds(["2024-02-01"], ["000001.SZ", "600519.SH"]),
    )

    assert result is not None
    assert bool(result.set_index("symbol")["is_st"].loc["000001.SZ"]) is True


def test_closed_gate_disables_st_filter_even_if_silver_present(tmp_path):
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    primary = tmp_path / "silver" / "st_flags" / "st_flags.parquet"
    manifest = tmp_path / "manifests" / "st_flags.json"
    _write_silver_st_flags(
        primary,
        [{"symbol": "000001.SZ", "is_st": True, "available_at": pd.Timestamp("2024-01-01")}],
    )
    _write_manifest(manifest, st_usable=False)

    result = _load_st_flags_for_filter(
        primary_path=primary,
        legacy_path=tmp_path / "missing_legacy.parquet",
        manifest_path=manifest,
        prediction_frame=_preds(["2024-02-01"], ["000001.SZ"]),
    )

    assert result is None


def test_open_gate_passes_silver_through(tmp_path):
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    primary = tmp_path / "silver" / "st_flags" / "st_flags.parquet"
    manifest = tmp_path / "manifests" / "st_flags.json"
    _write_silver_st_flags(
        primary,
        [{"symbol": "000001.SZ", "is_st": True, "available_at": pd.Timestamp("2024-01-01")}],
    )
    _write_manifest(manifest, st_usable=True)

    result = _load_st_flags_for_filter(
        primary_path=primary,
        legacy_path=tmp_path / "missing_legacy.parquet",
        manifest_path=manifest,
        prediction_frame=_preds(["2024-02-01"], ["000001.SZ"]),
    )

    assert result is not None
    assert bool(result.iloc[0]["is_st"]) is True


def test_asof_uses_latest_available_at_not_after_trade_date(tmp_path):
    """A 2024-02-01 prediction must NOT pick up a 2024-03-01 ST flag."""
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    primary = tmp_path / "silver" / "st_flags" / "st_flags.parquet"
    _write_silver_st_flags(
        primary,
        [
            {"symbol": "X.SZ", "is_st": False, "available_at": pd.Timestamp("2024-01-01")},
            {"symbol": "X.SZ", "is_st": True, "available_at": pd.Timestamp("2024-03-01")},
        ],
    )

    result = _load_st_flags_for_filter(
        primary_path=primary,
        legacy_path=tmp_path / "missing_legacy.parquet",
        manifest_path=tmp_path / "missing_manifest.json",
        prediction_frame=_preds(["2024-02-01", "2024-04-01"], ["X.SZ"]),
    )

    assert result is not None
    by_date = result.set_index("trade_date")["is_st"]
    assert bool(by_date.loc[pd.Timestamp("2024-02-01")]) is False
    assert bool(by_date.loc[pd.Timestamp("2024-04-01")]) is True


def test_missing_everything_returns_none(tmp_path):
    from quantagent.training.v7_experiment import _load_st_flags_for_filter

    result = _load_st_flags_for_filter(
        primary_path=tmp_path / "missing_silver.parquet",
        legacy_path=tmp_path / "missing_legacy.parquet",
        manifest_path=tmp_path / "missing_manifest.json",
        prediction_frame=_preds(["2024-02-01"], ["X.SZ"]),
    )

    assert result is None


def test_sector_map_loader_respects_gate(tmp_path):
    from quantagent.training.v7_experiment import _load_sector_map_for_optimization

    sector_path = tmp_path / "silver" / "sector_map" / "sector_map.parquet"
    manifest = tmp_path / "manifests" / "sector_map.json"
    sector_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "sector_level_1": "Food",
                "sector_level_2": "Liquor",
                "available_at": pd.Timestamp("2024-01-01"),
                "coverage_status": "pit_historical",
                "source": "manual_vendor_sector",
            }
        ]
    ).to_parquet(sector_path, index=False)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "coverage_report": {
                        "gate": {"sector_usable_for_optimization": False, "reason": "blocked"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result_closed = _load_sector_map_for_optimization(
        sector_map_path=sector_path,
        manifest_path=manifest,
        gate_enabled=True,
    )
    assert result_closed is None

    manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "coverage_report": {
                        "gate": {"sector_usable_for_optimization": True, "reason": "passed"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result_open = _load_sector_map_for_optimization(
        sector_map_path=sector_path,
        manifest_path=manifest,
        gate_enabled=True,
    )
    assert result_open is not None
    assert "industry" in result_open.columns or "sector_level_1" in result_open.columns


def test_sector_map_loader_disabled_returns_none(tmp_path):
    from quantagent.training.v7_experiment import _load_sector_map_for_optimization

    result = _load_sector_map_for_optimization(
        sector_map_path=tmp_path / "anything.parquet",
        manifest_path=tmp_path / "anything.json",
        gate_enabled=False,
    )
    assert result is None
