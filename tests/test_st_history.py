from __future__ import annotations

import pandas as pd


def test_st_history_marks_unknown_without_false_st(tmp_path):
    from quantagent.data.sector import StFlagBuilder, StFlagConfig

    source = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "is_st": True,
                "st_source": "manual_notice",
                "source_version": "v1",
                "effective_date": "2024-01-02",
                "available_at": "2024-01-03",
                "fetched_at": "2024-01-04",
            }
        ]
    )
    cfg = StFlagConfig(
        symbols=("000001.SZ", "600519.SH"),
        as_of_date="2024-02-01",
        output_root=tmp_path,
        st_block_weight=0.90,
        unknown_st_block_weight=0.00,
    )
    result = StFlagBuilder(cfg).build(source)
    frame = result.frame.set_index("symbol")

    assert bool(frame.loc["000001.SZ", "is_st"]) is True
    assert bool(frame.loc["000001.SZ", "st_known"]) is True
    assert float(frame.loc["000001.SZ", "block_weight"]) == 0.90
    assert bool(frame.loc["600519.SH", "is_st"]) is False
    assert bool(frame.loc["600519.SH", "st_known"]) is False
    assert float(frame.loc["600519.SH", "block_weight"]) == 0.00
    assert frame.loc["600519.SH", "coverage_status"] == "missing"


def test_st_coverage_gate_blocks_low_coverage():
    from quantagent.data.sector import StFlagBuilder, StFlagConfig

    result = StFlagBuilder(
        StFlagConfig(symbols=("A", "B"), as_of_date="2024-02-01", min_st_coverage=0.85)
    ).build(pd.DataFrame([{"symbol": "A", "is_st": False, "available_at": "2024-01-01", "fetched_at": "2024-01-02"}]))

    gate = result.coverage["gate"]
    assert result.coverage["coverage_rate"] == 0.5
    assert gate["st_usable_for_risk_filter"] is False
    assert "st_coverage_below_threshold" in gate["reason"]
    assert gate["policy"]["suspended_block_weight"] == 1.0


def test_st_flag_builder_writes_manifest_and_reports(tmp_path):
    from quantagent.data.sector import StFlagBuilder, StFlagConfig

    cfg = StFlagConfig(symbols=("A",), as_of_date="2024-02-01", output_root=tmp_path)
    builder = StFlagBuilder(cfg)
    result = builder.write(
        builder.build(pd.DataFrame([{"symbol": "A", "is_st": False, "available_at": "2024-01-01", "fetched_at": "2024-01-02"}]))
    )

    assert (tmp_path / "silver" / "st_flags" / "st_flags.parquet").exists()
    assert (tmp_path / "silver" / "st_flags" / "coverage_report.json").exists()
    assert (tmp_path / "silver" / "st_flags" / "validation_report.json").exists()
    assert (tmp_path / "manifests" / "st_flags.json").exists()
    assert result.output_paths["st_flags"].endswith("st_flags.parquet")
