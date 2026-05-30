from __future__ import annotations

import json

import pandas as pd


def _weights(symbols: list[str]) -> pd.DataFrame:
    row = {"trade_date": pd.Timestamp("2024-02-01")}
    for symbol in symbols:
        row[symbol] = 1.0 / len(symbols)
    return pd.DataFrame([row])


def test_sector_audit_zero_coverage_is_unknown_and_disabled(tmp_path):
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig
    from quantagent.diagnostics.sector_audit import build_sector_audit

    symbols = ["600519.SH", "300750.SZ"]
    builder = SectorMapBuilder(SectorMapConfig(symbols=tuple(symbols), as_of_date="2024-01-01", output_root=tmp_path))
    written = builder.write(builder.build(None))
    sector = written.frame

    audit = build_sector_audit(
        _weights(symbols),
        sector_map=sector,
        sector_manifest=tmp_path / "manifests" / "sector_map.json",
    )

    status = audit["gate_status"]
    assert status["sector_usable_for_diagnostics"] is True
    assert status["sector_usable_for_optimization"] is False
    assert status["sector_unknown_rate"] == 1.0
    assert set(audit["exposure_by_sector_l1"]["sector_level_1"]) == {"UNKNOWN"}
    assert set(audit["exposure_by_board_proxy"]["board_proxy"]) == {"SH_Main_沪主板", "ChiNext_创业"}


def test_sector_audit_partial_coverage_keeps_diagnostics_but_gate_closed(tmp_path):
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig
    from quantagent.diagnostics.sector_audit import build_sector_audit

    symbols = ["600519.SH", "300750.SZ"]
    source = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "sector_level_1": "Food",
                "sector_level_2": "Liquor",
                "available_at": "2024-01-01",
                "fetched_at": "2024-01-01",
            }
        ]
    )
    builder = SectorMapBuilder(SectorMapConfig(symbols=tuple(symbols), as_of_date="2024-01-15", output_root=tmp_path))
    written = builder.write(builder.build(source))

    audit = build_sector_audit(
        _weights(symbols),
        sector_map=written.frame,
        sector_manifest=tmp_path / "manifests" / "sector_map.json",
    )

    assert audit["gate_status"]["sector_usable_for_optimization"] is False
    assert set(audit["exposure_by_sector_l1"]["sector_level_1"]) == {"Food", "UNKNOWN"}
    assert audit["gate_status"]["real_sector_coverage"] == 0.5


def test_sector_audit_high_coverage_can_open_gate_without_changing_weights(tmp_path):
    from quantagent.data.sector import SectorMapBuilder, SectorMapConfig
    from quantagent.diagnostics.sector_audit import build_sector_audit, load_sector_st_gate_status

    symbols = [f"60000{i}.SH" for i in range(10)]
    source = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "sector_level_1": "SectorA",
                "sector_level_2": "SubA",
                "available_at": "2024-01-01",
                "fetched_at": "2024-01-01",
            }
            for symbol in symbols[:9]
        ]
    )
    builder = SectorMapBuilder(SectorMapConfig(symbols=tuple(symbols), as_of_date="2024-01-15", output_root=tmp_path))
    written = builder.write(builder.build(source))
    manifest = tmp_path / "manifests" / "sector_map.json"

    gate = load_sector_st_gate_status(sector_manifest=manifest)
    audit = build_sector_audit(_weights(symbols), sector_map=written.frame, sector_manifest=manifest)

    assert gate.sector_usable_for_optimization is True
    assert audit["gate_status"]["sector_usable_for_optimization"] is True
    assert audit["gate_status"]["target_weights_contamination"] is False
    assert round(audit["gate_status"]["real_sector_coverage"], 6) == 0.9


def test_sector_audit_writes_expected_artifacts(tmp_path):
    from quantagent.diagnostics.sector_audit import build_sector_audit, write_sector_audit

    audit = build_sector_audit(_weights(["600519.SH"]))
    paths = write_sector_audit(audit, tmp_path / "sector_audit")

    expected = {
        "gate_status.json",
        "exposure_by_board_proxy.csv",
        "exposure_by_sector_l1.csv",
        "exposure_by_sector_l2.csv",
        "unknown_exposure.csv",
        "st_risk_audit.csv",
        "sector_audit.md",
    }
    assert expected == {path.name for path in paths.values()}
    gate = json.loads((tmp_path / "sector_audit" / "gate_status.json").read_text(encoding="utf-8"))
    assert gate["sector_usable_for_optimization"] is False
