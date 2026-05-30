from __future__ import annotations

import pandas as pd


def test_sector_map_builder_is_pit_safe_and_reports_missing(tmp_path):
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig

    source = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "sector_level_1": "食品饮料",
                "sector_level_2": "白酒",
                "available_at": "2024-01-01",
                "fetched_at": "2024-01-02",
            },
            {
                "symbol": "600519.SH",
                "sector_level_1": "未来行业",
                "sector_level_2": "未来细分",
                "available_at": "2026-01-01",
                "fetched_at": "2026-01-02",
            },
            {
                "symbol": "000001.SZ",
                "sector_level_1": "银行",
                "sector_level_2": "股份制银行",
                "available_at": "2024-01-01",
                "fetched_at": "2024-01-02",
            },
        ]
    )
    cfg = SectorMapConfig(
        symbols=("600519.SH", "000001.SZ", "300750.SZ"),
        as_of_date="2025-01-01",
        fetched_at="2025-01-01",
        output_root=tmp_path,
    )
    result = SectorMapBuilder(cfg).build(source)
    frame = result.frame.set_index("symbol")

    assert frame.loc["600519.SH", "sector_level_1"] == "食品饮料"
    assert frame.loc["300750.SZ", "coverage_status"] == "missing"
    assert result.coverage["covered_symbols"] == 2
    assert result.coverage["missing_symbols"] == 1
    assert result.validation["status"] == "passed"


def test_sector_map_builder_writes_reports_and_manifest(tmp_path):
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig

    source = pd.DataFrame(
        [
            {"symbol": "A", "industry": "X", "available_at": "2024-01-01", "fetched_at": "2024-01-02"},
            {"symbol": "B", "industry": "Y", "available_at": "2024-01-01", "fetched_at": "2024-01-02"},
        ]
    )
    cfg = SectorMapConfig(symbols=("A", "B"), as_of_date="2024-02-01", output_root=tmp_path)
    builder = SectorMapBuilder(cfg)
    written = builder.write(builder.build(source))

    assert (tmp_path / "silver" / "sector_map" / "sector_map.parquet").exists()
    assert (tmp_path / "silver" / "sector_map" / "coverage_report.json").exists()
    assert (tmp_path / "silver" / "sector_map" / "missing_symbols.csv").exists()
    assert (tmp_path / "silver" / "sector_map" / "duplicate_symbols.csv").exists()
    assert (tmp_path / "silver" / "sector_map" / "source_priority_report.json").exists()
    assert (tmp_path / "silver" / "sector_map" / "sector_distribution.csv").exists()
    assert (tmp_path / "manifests" / "sector_map.json").exists()
    assert written.output_paths["sector_map"].endswith("sector_map.parquet")


def test_sector_source_normalization_and_priority_selects_best_source():
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig, normalize_sector_source

    raw = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "sector_level_1": "board",
                "sector_level_2": "board",
                "source": "board_proxy",
                "source_version": "v1",
                "effective_date": "2024-01-01",
                "available_at": "2024-01-02",
                "fetched_at": "2024-01-02",
            },
            {
                "symbol": "600519.SH",
                "sector_level_1": "食品饮料",
                "sector_level_2": "白酒",
                "source": "manual_vendor_sector",
                "source_version": "vendor_2024",
                "effective_date": "2024-01-01",
                "available_at": "2024-01-02",
                "fetched_at": "2024-01-03",
            },
        ]
    )
    normalized = normalize_sector_source(raw)
    assert {"source_version", "effective_date"}.issubset(normalized.columns)
    result = SectorMapBuilder(
        SectorMapConfig(symbols=("600519.SH",), as_of_date="2024-02-01")
    ).build(normalized)

    row = result.frame.iloc[0]
    assert row["source"] == "manual_vendor_sector"
    assert row["sector_level_1"] == "食品饮料"
    assert result.coverage["gate"]["sector_usable_for_optimization"] is True


def test_sector_coverage_gate_blocks_low_coverage_and_board_proxy():
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig, board_proxy_rows

    symbols = ("600519.SH", "300750.SZ")
    result = SectorMapBuilder(
        SectorMapConfig(symbols=symbols, as_of_date="2024-02-01")
    ).build(board_proxy_rows(symbols, as_of_date="2024-01-02"))

    assert result.coverage["board_proxy_symbols"] == 2
    assert result.coverage["sector_level_1_coverage"] == 0.0
    gate = result.coverage["gate"]
    assert gate["sector_usable_for_diagnostics"] is True
    assert gate["sector_usable_for_optimization"] is False
    assert "sector_level_1_coverage_below_threshold" in gate["reason"]


def test_import_sector_source_cli_writes_bronze(tmp_path):
    from typer.testing import CliRunner

    from quantagent.cli import app

    raw = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "industry": "食品饮料",
                "available_at": "2024-01-02",
                "fetched_at": "2024-01-03",
            }
        ]
    )
    input_path = tmp_path / "manual.csv"
    output_path = tmp_path / "bronze" / "manual_sector.parquet"
    raw.to_csv(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "import-sector-source-v7",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--source-version",
            "test_v1",
        ],
    )

    assert result.exit_code == 0, result.output
    written = pd.read_parquet(output_path)
    assert written.loc[0, "source"] == "manual_vendor_sector"
    assert written.loc[0, "source_version"] == "test_v1"


def test_stratified_ic_sector_axis_uses_available_at_asof_join():
    from quantagent.diagnostics.stratified_ic import StratifiedICConfig, compute_stratified_ic

    dates = pd.bdate_range("2024-01-02", periods=12)
    rows = []
    for date in dates:
        for i in range(12):
            symbol = f"{600000 + i}.SH"
            pred = float(i)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "horizon": 5,
                    "prediction": pred,
                    "forward_return_5d": pred / 100.0,
                }
            )
    preds = pd.DataFrame(rows)
    sector = pd.DataFrame(
        [
            {
                "symbol": f"{600000 + i}.SH",
                "sector_level_1": "visible" if i < 6 else "future_only",
                "sector_level_2": "visible_l2" if i < 6 else "future_l2",
                "source": "test",
                "source_version": "v1",
                "effective_date": "2024-01-01" if i < 6 else "2026-01-01",
                "fetched_at": "2024-01-01" if i < 6 else "2026-01-01",
                "available_at": "2024-01-01" if i < 6 else "2026-01-01",
                "coverage_status": "pit_historical",
            }
            for i in range(12)
        ]
    )
    result = compute_stratified_ic(
        preds,
        sector_map=sector,
        config=StratifiedICConfig(min_symbols_per_date=5, min_days_per_bucket=5, top_k_for_ann_return=5),
    )
    buckets = set(result.by_axis["sector_level_1"]["bucket"])
    assert "visible" in buckets
    assert "future_only" not in buckets


def test_st_flag_builder_keeps_source_and_available_at(tmp_path):
    from quantagent.data.sector import StFlagBuilder, StFlagConfig

    source = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "is_st": True,
                "st_source": "local_notice",
                "available_at": "2024-01-03",
                "fetched_at": "2024-01-04",
            }
        ]
    )
    cfg = StFlagConfig(symbols=("000001.SZ",), as_of_date="2024-02-01", output_root=tmp_path)
    builder = StFlagBuilder(cfg)
    result = builder.write(builder.build(source))
    frame = result.frame

    assert frame.loc[0, "is_st"] is True or bool(frame.loc[0, "is_st"]) is True
    assert frame.loc[0, "st_source"] == "local_notice"
    assert "available_at" in frame.columns
    assert (tmp_path / "silver" / "st_flags" / "st_flags.parquet").exists()
