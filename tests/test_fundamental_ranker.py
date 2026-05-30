"""Tests for the fundamental PIT ranker (Stage 2 fundamental ranker)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _make_metrics(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_universe_sector(n: int, sector_name: str = "Bank") -> pd.DataFrame:
    """A 10-stock bank sector with varying valuation / quality / growth."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": f"B{i:03d}.SZ",
                "available_at": pd.Timestamp("2024-01-15"),
                "pe_ttm": 5.0 + i,           # low PE = cheap = good
                "pb": 0.6 + 0.1 * i,         # low PB = cheap = good
                "ps_ttm": 1.0 + 0.2 * i,
                "roe": 0.15 - 0.005 * i,     # high ROE = good
                "gross_margin": 0.40 - 0.01 * i,
                "operating_cf_to_net_income": 1.2 - 0.05 * i,
                "revenue_yoy": 0.10 - 0.005 * i,
                "net_income_yoy": 0.12 - 0.005 * i,
            }
        )
    df = pd.DataFrame(rows)
    sector = pd.DataFrame(
        [
            {
                "symbol": f"B{i:03d}.SZ",
                "sector_level_1": sector_name,
                "available_at": pd.Timestamp("2024-01-01"),
            }
            for i in range(n)
        ]
    )
    return df, sector


def test_basic_score_and_rank_within_sector(tmp_path):
    from quantagent.data.fundamental import build_fundamental_ranker

    metrics, sector = _make_universe_sector(10)
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
    )
    frame = result.frame
    assert not frame.empty
    assert set(frame["rank_bucket"]) == {"Bank"}
    # The "cheapest, highest-quality, highest-growth" stock is B000.SZ
    # (low PE/PB/PS, high ROE/margin, high YoY). Its composite rank
    # must be the highest (closest to 1.0).
    by_symbol = frame.set_index("symbol")
    assert float(by_symbol.loc["B000.SZ", "valuation_score"]) >= float(by_symbol.loc["B009.SZ", "valuation_score"])
    assert float(by_symbol.loc["B000.SZ", "quality_score"]) >= float(by_symbol.loc["B009.SZ", "quality_score"])
    assert float(by_symbol.loc["B000.SZ", "growth_score"]) >= float(by_symbol.loc["B009.SZ", "growth_score"])
    assert float(by_symbol.loc["B000.SZ", "composite_rank"]) > float(by_symbol.loc["B009.SZ", "composite_rank"])


def test_pit_join_uses_latest_available_at_at_or_before_as_of():
    """A 2024-02-01 as_of must not see a 2024-03-01 metric row."""
    from quantagent.data.fundamental import build_fundamental_ranker

    metrics, _ = _make_universe_sector(6)
    # Add a future-dated update for B000.SZ that should be ignored.
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "symbol": "B000.SZ",
                        "available_at": pd.Timestamp("2024-03-01"),
                        "pe_ttm": 100.0,
                        "pb": 100.0,
                        "ps_ttm": 100.0,
                        "roe": -0.10,
                        "gross_margin": -0.20,
                        "operating_cf_to_net_income": -1.0,
                        "revenue_yoy": -0.40,
                        "net_income_yoy": -0.50,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=None,
    )
    by_symbol = result.frame.set_index("symbol")
    # B000.SZ should still rank near the top under its 2024-01-15 row,
    # not the future 2024-03-01 disaster row.
    assert float(by_symbol.loc["B000.SZ", "composite_score"]) > 0.5


def test_missing_dimension_drops_that_score_only():
    from quantagent.data.fundamental import build_fundamental_ranker

    metrics = _make_metrics(
        [
            {"symbol": f"X{i:03d}.SZ", "available_at": pd.Timestamp("2024-01-15"), "pe_ttm": 5.0 + i, "pb": 0.6 + 0.1 * i, "ps_ttm": 1.0 + 0.2 * i}
            for i in range(6)
        ]
    )
    sector = pd.DataFrame(
        [{"symbol": f"X{i:03d}.SZ", "sector_level_1": "Industrial", "available_at": pd.Timestamp("2024-01-01")} for i in range(6)]
    )
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
    )
    frame = result.frame
    assert frame["quality_score"].isna().all()
    assert frame["growth_score"].isna().all()
    assert frame["valuation_score"].notna().any()
    # composite = pure valuation when other dims are absent
    composite_eq_valuation = (
        frame["composite_score"].sub(frame["valuation_score"]).abs().fillna(0.0).max()
    )
    assert composite_eq_valuation < 1e-9


def test_below_min_universe_bucket_is_skipped():
    from quantagent.data.fundamental import FundamentalRankerConfig, build_fundamental_ranker

    metrics = _make_metrics(
        [
            {"symbol": "A.SZ", "available_at": pd.Timestamp("2024-01-15"), "pe_ttm": 10.0, "pb": 1.0, "ps_ttm": 1.0},
            {"symbol": "B.SZ", "available_at": pd.Timestamp("2024-01-15"), "pe_ttm": 12.0, "pb": 1.1, "ps_ttm": 1.1},
        ]
    )
    sector = pd.DataFrame(
        [
            {"symbol": "A.SZ", "sector_level_1": "TinySector", "available_at": pd.Timestamp("2024-01-01")},
            {"symbol": "B.SZ", "sector_level_1": "TinySector", "available_at": pd.Timestamp("2024-01-01")},
        ]
    )
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
        config=FundamentalRankerConfig(min_universe_per_bucket=5),
    )
    # Only 2 symbols in one bucket → below min → no rows
    assert result.frame.empty


def test_board_proxy_used_when_sector_missing():
    from quantagent.data.fundamental import build_fundamental_ranker

    rows = []
    # 6 SH main-board symbols (60xxxx.SH) — board_of returns 'SH_Main_沪主板'
    for i in range(6):
        rows.append(
            {
                "symbol": f"60000{i}.SH",
                "available_at": pd.Timestamp("2024-01-15"),
                "pe_ttm": 5.0 + i,
                "pb": 0.6 + 0.1 * i,
                "ps_ttm": 1.0 + 0.2 * i,
            }
        )
    result = build_fundamental_ranker(
        _make_metrics(rows),
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=None,
    )
    assert "board_proxy" in set(result.frame["rank_bucket_kind"])


def test_invalid_valuation_below_floor_is_excluded_from_score():
    from quantagent.data.fundamental import build_fundamental_ranker

    rows = [
        {"symbol": f"S{i:03d}.SH", "available_at": pd.Timestamp("2024-01-15"), "pe_ttm": 5.0 + i, "pb": 0.6 + 0.1 * i, "ps_ttm": 1.0 + 0.2 * i}
        for i in range(6)
    ]
    rows.append(
        {
            "symbol": "BADPE.SH",
            "available_at": pd.Timestamp("2024-01-15"),
            "pe_ttm": -50.0,  # negative — below floor → dropped from PE rank
            "pb": 0.9,
            "ps_ttm": 1.5,
        }
    )
    sector = pd.DataFrame(
        [{"symbol": row["symbol"], "sector_level_1": "Sector", "available_at": pd.Timestamp("2024-01-01")} for row in rows]
    )
    result = build_fundamental_ranker(
        _make_metrics(rows),
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
    )
    bad = result.frame.set_index("symbol").loc["BADPE.SH"]
    # PE was excluded but PB / PS still feed → valuation_score is the
    # mean of the two remaining ranks (not NaN, not based on -50 PE).
    assert pd.notna(bad["valuation_score"])


def test_multi_date_as_of_writes_one_block_per_date():
    from quantagent.data.fundamental import build_fundamental_ranker

    metrics, sector = _make_universe_sector(8)
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
        sector_map=sector,
    )
    assert set(result.frame["as_of_date"]) == {pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")}
    by_date = result.frame.groupby("as_of_date").size()
    assert int(by_date.loc[pd.Timestamp("2024-02-01")]) == 8
    assert int(by_date.loc[pd.Timestamp("2024-03-01")]) == 8


def test_dimension_weights_drive_composite():
    from quantagent.data.fundamental import FundamentalRankerConfig, build_fundamental_ranker

    metrics, sector = _make_universe_sector(8)
    # Pure-valuation weighting: composite must equal valuation_score
    pure_val = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
        config=FundamentalRankerConfig(dimension_weights={"valuation": 1.0, "quality": 0.0, "growth": 0.0}),
    )
    diff = pure_val.frame["composite_score"].sub(pure_val.frame["valuation_score"]).abs().max()
    assert diff < 1e-9


def test_metric_completeness_field():
    from quantagent.data.fundamental import build_fundamental_ranker

    metrics, sector = _make_universe_sector(6)
    result = build_fundamental_ranker(
        metrics,
        as_of_dates=[pd.Timestamp("2024-02-01")],
        sector_map=sector,
    )
    # All three dimensions populated → completeness = 1.0
    assert (result.frame["metric_completeness"] - 1.0).abs().max() < 1e-9


def test_writer_emits_parquet_and_manifest(tmp_path):
    from quantagent.data.fundamental import FundamentalRankerBuilder, FundamentalRankerConfig

    metrics, sector = _make_universe_sector(8)
    builder = FundamentalRankerBuilder(FundamentalRankerConfig(output_root=tmp_path))
    written = builder.write(
        builder.build(metrics, as_of_dates=[pd.Timestamp("2024-02-01")], sector_map=sector)
    )

    parquet_path = tmp_path / "silver" / "fundamental_ranker" / "fundamental_ranker.parquet"
    assert parquet_path.exists()
    assert (tmp_path / "silver" / "fundamental_ranker" / "coverage_report.json").exists()
    assert (tmp_path / "silver" / "fundamental_ranker" / "validation_report.json").exists()
    assert (tmp_path / "manifests" / "fundamental_ranker.json").exists()
    assert written.output_paths["fundamental_ranker"].endswith("fundamental_ranker.parquet")


def test_gate_blocks_when_real_sector_share_too_low(tmp_path):
    from quantagent.data.fundamental import FundamentalRankerBuilder, FundamentalRankerConfig

    rows = [
        {"symbol": f"60000{i}.SH", "available_at": pd.Timestamp("2024-01-15"), "pe_ttm": 5.0 + i, "pb": 0.6 + 0.1 * i, "ps_ttm": 1.0 + 0.2 * i}
        for i in range(6)
    ]
    builder = FundamentalRankerBuilder(FundamentalRankerConfig(output_root=tmp_path))
    result = builder.build(_make_metrics(rows), as_of_dates=[pd.Timestamp("2024-02-01")], sector_map=None)
    gate = result.coverage["gate"]
    # All rows are board_proxy → real_sector_share = 0 → gate closed
    assert gate["fundamental_ranker_usable_for_overlay"] is False
    assert "real_sector_share_below_threshold" in gate["reason"]


def test_gate_opens_with_full_sector_coverage_and_metrics():
    from quantagent.data.fundamental import FundamentalRankerBuilder

    metrics, sector = _make_universe_sector(8)
    builder = FundamentalRankerBuilder()
    result = builder.build(metrics, as_of_dates=[pd.Timestamp("2024-02-01")], sector_map=sector)
    gate = result.coverage["gate"]
    assert gate["fundamental_ranker_usable_for_overlay"] is True
    assert gate["reason"] == "passed"


def test_overlay_helper_respects_manifest_gate(tmp_path):
    from quantagent.data.fundamental import fundamental_ranker_for_overlay

    pool = pd.DataFrame([{"symbol": "A.SZ", "composite_score": 0.9}])

    closed = tmp_path / "closed.json"
    closed.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"fundamental_ranker_usable_for_overlay": False}}}}
        ),
        encoding="utf-8",
    )
    assert fundamental_ranker_for_overlay(pool, closed) is None

    open_path = tmp_path / "open.json"
    open_path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"fundamental_ranker_usable_for_overlay": True}}}}
        ),
        encoding="utf-8",
    )
    out = fundamental_ranker_for_overlay(pool, open_path)
    assert out is not None
    assert "composite_score" in out.columns


def test_overlay_helper_returns_none_when_missing_inputs(tmp_path):
    from quantagent.data.fundamental import fundamental_ranker_for_overlay

    assert fundamental_ranker_for_overlay(pd.DataFrame(), tmp_path / "x.json") is None
    assert fundamental_ranker_for_overlay(None, tmp_path / "x.json") is None
    assert fundamental_ranker_for_overlay(pd.DataFrame([{"x": 1}]), None) is None


def test_cli_build_fundamental_ranker(tmp_path):
    from typer.testing import CliRunner

    from quantagent.cli import app

    metrics_path = tmp_path / "metrics.parquet"
    metrics, sector = _make_universe_sector(8)
    metrics.to_parquet(metrics_path, index=False)
    sector_path = tmp_path / "sector_map.parquet"
    sector.to_parquet(sector_path, index=False)
    output_root = tmp_path / "lake"

    result = CliRunner().invoke(
        app,
        [
            "build-fundamental-ranker-v7",
            "--metrics",
            str(metrics_path),
            "--sector-map",
            str(sector_path),
            "--as-of-dates",
            "2024-02-01,2024-03-01",
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_root / "silver" / "fundamental_ranker" / "fundamental_ranker.parquet").exists()
    assert (output_root / "manifests" / "fundamental_ranker.json").exists()
