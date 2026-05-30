"""Tests for sector_pool builder (Stage 2 sector-pool deliverable)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _ic_table(
    rows: list[tuple[str, int, float, float, float, int, int]],
) -> pd.DataFrame:
    """Tiny helper to build a stratified_ic-shaped frame."""
    return pd.DataFrame(
        [
            {
                "bucket": name,
                "horizon": horizon,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "n_dates": n_dates,
                "n_symbols": n_symbols,
            }
            for name, horizon, ic_mean, ic_std, ic_ir, n_dates, n_symbols in rows
        ]
    )


def test_excluded_when_sample_size_too_small():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    table = _ic_table(
        [
            ("Food", 20, 0.08, 0.04, 2.00, 200, 50),
            ("Banks", 20, 0.05, 0.03, 1.50, 5, 50),  # n_dates < min_dates
            ("Coal", 20, 0.04, 0.03, 1.30, 200, 3),  # n_symbols < min_symbols
        ]
    )
    result = build_sector_pool(table, config=SectorPoolConfig(min_dates=60, min_symbols=20))
    tiers = dict(zip(result.frame["sector_level_1"], result.frame["pool_tier"]))
    assert tiers["Banks"] == "excluded"
    assert tiers["Coal"] == "excluded"
    assert tiers["Food"] == "core"


def test_excluded_when_ic_non_positive():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    table = _ic_table(
        [
            ("Good", 20, 0.06, 0.04, 1.50, 200, 50),
            ("Bad", 20, -0.01, 0.05, -0.20, 200, 50),
            ("Flat", 20, 0.0, 0.05, 0.00, 200, 50),
        ]
    )
    result = build_sector_pool(table, config=SectorPoolConfig(min_dates=60, min_symbols=20))
    tiers = dict(zip(result.frame["sector_level_1"], result.frame["pool_tier"]))
    assert tiers["Bad"] == "excluded"
    assert tiers["Flat"] == "excluded"
    assert tiers["Good"] == "core"


def test_core_requires_both_high_ic_and_stable_ir():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    # Use a population large enough that the top-30% quantile gives us
    # at least one sector to compare against.
    table = _ic_table(
        [
            ("HighICStable", 20, 0.10, 0.03, 1.50, 200, 50),    # core candidate
            ("HighICUnstable", 20, 0.09, 0.20, 0.05, 200, 50),  # high IC, IR < threshold
            ("Mid1", 20, 0.04, 0.04, 0.50, 200, 50),
            ("Mid2", 20, 0.03, 0.04, 0.40, 200, 50),
            ("LowStable", 20, 0.02, 0.02, 0.30, 200, 50),
        ]
    )
    result = build_sector_pool(
        table,
        config=SectorPoolConfig(
            min_dates=60,
            min_symbols=20,
            core_quantile=0.30,
            core_ir_threshold=0.50,
            watch_ir_threshold=0.20,
            short_term_vol_threshold=0.10,
        ),
    )
    tiers = dict(zip(result.frame["sector_level_1"], result.frame["pool_tier"]))
    assert tiers["HighICStable"] == "core"
    # HighICUnstable has high vol AND low IR → short_term
    assert tiers["HighICUnstable"] == "short_term"


def test_watch_for_positive_but_below_core_cutoff():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    table = _ic_table(
        [
            ("Top", 20, 0.10, 0.03, 1.50, 200, 50),
            ("Middle", 20, 0.04, 0.04, 0.30, 200, 50),     # positive IR but below core cutoff
            ("Low", 20, 0.02, 0.03, 0.25, 200, 50),
            ("Lower", 20, 0.01, 0.04, 0.22, 200, 50),
        ]
    )
    result = build_sector_pool(
        table,
        config=SectorPoolConfig(
            min_dates=60,
            min_symbols=20,
            core_quantile=0.30,
            core_ir_threshold=1.00,
            watch_ir_threshold=0.20,
            short_term_vol_threshold=0.10,
        ),
    )
    tiers = dict(zip(result.frame["sector_level_1"], result.frame["pool_tier"]))
    # Top has IR=1.5 but the cutoff is 1.00 — it qualifies for core only
    # because both top quantile AND IR threshold hit. Middle has IR=0.3 <
    # 1.00 so falls through to watch.
    assert tiers["Top"] == "core"
    assert tiers["Middle"] == "watch"


def test_short_term_tier_for_unstable_high_vol_sectors():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    table = _ic_table(
        [
            ("Stable", 20, 0.08, 0.02, 2.00, 200, 50),
            ("Volatile", 20, 0.06, 0.15, 0.30, 200, 50),    # ic_std crosses vol threshold
        ]
    )
    result = build_sector_pool(
        table,
        config=SectorPoolConfig(
            min_dates=60,
            min_symbols=20,
            core_quantile=0.50,
            core_ir_threshold=0.50,
            watch_ir_threshold=0.20,
            short_term_vol_threshold=0.10,
        ),
    )
    tiers = dict(zip(result.frame["sector_level_1"], result.frame["pool_tier"]))
    assert tiers["Stable"] == "core"
    assert tiers["Volatile"] == "short_term"


def test_reference_horizon_uses_closest_available():
    from quantagent.data.sector import SectorPoolConfig, build_sector_pool

    table = _ic_table(
        [
            ("OnlyH5", 5, 0.07, 0.04, 1.50, 200, 50),       # no 20d, fall back to 5d
            ("Both", 20, 0.10, 0.04, 1.80, 200, 50),
            ("Both", 60, 0.04, 0.05, 0.50, 200, 50),
        ]
    )
    result = build_sector_pool(table, config=SectorPoolConfig(reference_horizon=20, min_dates=60, min_symbols=20))
    only_h5 = result.frame.set_index("sector_level_1").loc["OnlyH5"]
    both = result.frame.set_index("sector_level_1").loc["Both"]
    assert int(only_h5["horizon"]) == 5
    assert int(both["horizon"]) == 20


def test_unknown_bucket_is_dropped_silently():
    from quantagent.data.sector import build_sector_pool

    table = _ic_table(
        [
            ("UNKNOWN", 20, 0.08, 0.04, 2.00, 200, 50),
            ("Real", 20, 0.06, 0.04, 1.50, 200, 50),
        ]
    )
    result = build_sector_pool(table)
    assert "UNKNOWN" not in set(result.frame["sector_level_1"])


def test_writer_emits_parquet_and_manifest(tmp_path):
    from quantagent.data.sector import SectorPoolBuilder, SectorPoolConfig

    table = _ic_table([("Food", 20, 0.08, 0.04, 2.00, 200, 50)])
    builder = SectorPoolBuilder(SectorPoolConfig(min_dates=60, min_symbols=20, output_root=tmp_path))
    written = builder.write(builder.build(table))

    parquet_path = tmp_path / "silver" / "sector_pool" / "sector_pool.parquet"
    assert parquet_path.exists()
    assert (tmp_path / "silver" / "sector_pool" / "coverage_report.json").exists()
    assert (tmp_path / "silver" / "sector_pool" / "validation_report.json").exists()
    assert (tmp_path / "silver" / "sector_pool" / "tier_distribution.csv").exists()
    assert (tmp_path / "manifests" / "sector_pool.json").exists()
    assert written.output_paths["sector_pool"].endswith("sector_pool.parquet")
    written_frame = pd.read_parquet(parquet_path)
    assert "pool_tier" in written_frame.columns
    assert set(written_frame["pool_tier"]).issubset({"core", "watch", "short_term", "excluded"})


def test_gate_blocks_when_no_core_sector(tmp_path):
    from quantagent.data.sector import SectorPoolBuilder, SectorPoolConfig

    # All sectors below core cutoff → gate must close.
    table = _ic_table(
        [
            ("OnlyWatch1", 20, 0.04, 0.04, 0.20, 200, 50),
            ("OnlyWatch2", 20, 0.03, 0.04, 0.18, 200, 50),
        ]
    )
    builder = SectorPoolBuilder(
        SectorPoolConfig(
            min_dates=60,
            min_symbols=20,
            core_quantile=0.30,
            core_ir_threshold=1.00,  # impossibly high → nothing reaches core
            watch_ir_threshold=0.10,
            short_term_vol_threshold=0.20,
            output_root=tmp_path,
        )
    )
    result = builder.build(table)
    gate = result.coverage["gate"]
    assert gate["sector_pool_usable_for_overlay"] is False
    assert "no_core_sector" in gate["reason"]


def test_gate_opens_with_core_present_and_low_excluded_ratio(tmp_path):
    from quantagent.data.sector import SectorPoolBuilder, SectorPoolConfig

    table = _ic_table(
        [
            ("Top", 20, 0.10, 0.03, 1.80, 200, 50),
            ("Mid1", 20, 0.06, 0.04, 0.40, 200, 50),
            ("Mid2", 20, 0.05, 0.04, 0.35, 200, 50),
        ]
    )
    builder = SectorPoolBuilder(SectorPoolConfig(min_dates=60, min_symbols=20, output_root=tmp_path))
    result = builder.build(table)
    gate = result.coverage["gate"]
    assert gate["sector_pool_usable_for_overlay"] is True
    assert gate["reason"] == "passed"


def test_overlay_helper_returns_none_when_gate_closed(tmp_path):
    from quantagent.data.sector import sector_pool_for_weight_overlay

    pool = pd.DataFrame(
        [
            {"sector_level_1": "Food", "pool_tier": "core"},
            {"sector_level_1": "Banks", "pool_tier": "excluded"},
        ]
    )
    manifest = tmp_path / "manifests" / "sector_pool.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"sector_pool_usable_for_overlay": False}}}}
        ),
        encoding="utf-8",
    )
    assert sector_pool_for_weight_overlay(pool, manifest) is None


def test_overlay_helper_returns_weights_when_gate_open(tmp_path):
    from quantagent.data.sector import sector_pool_for_weight_overlay

    pool = pd.DataFrame(
        [
            {"sector_level_1": "Food", "pool_tier": "core"},
            {"sector_level_1": "Banks", "pool_tier": "watch"},
            {"sector_level_1": "Coal", "pool_tier": "excluded"},
        ]
    )
    manifest = tmp_path / "manifests" / "sector_pool.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {"sector_pool_usable_for_overlay": True}}}}
        ),
        encoding="utf-8",
    )
    overlay = sector_pool_for_weight_overlay(pool, manifest)
    assert overlay is not None
    by_sector = overlay.set_index("sector_level_1")["overlay_weight"]
    assert float(by_sector.loc["Food"]) == 1.00
    assert float(by_sector.loc["Banks"]) == 0.70
    assert float(by_sector.loc["Coal"]) == 0.00


def test_overlay_helper_returns_none_when_manifest_missing(tmp_path):
    from quantagent.data.sector import sector_pool_for_weight_overlay

    pool = pd.DataFrame([{"sector_level_1": "Food", "pool_tier": "core"}])
    assert sector_pool_for_weight_overlay(pool, tmp_path / "nope.json") is None
    assert sector_pool_for_weight_overlay(pool, None) is None


def test_overlay_helper_returns_none_when_pool_empty():
    from quantagent.data.sector import sector_pool_for_weight_overlay

    assert sector_pool_for_weight_overlay(pd.DataFrame(), "ignored") is None
    assert sector_pool_for_weight_overlay(None, "ignored") is None


def test_cli_build_sector_pool_from_csv(tmp_path):
    from typer.testing import CliRunner

    from quantagent.cli import app

    ic_path = tmp_path / "sector_ic.csv"
    ic_path.write_text(
        "bucket,horizon,ic_mean,ic_std,ic_ir,n_dates,n_symbols\n"
        "Food,20,0.08,0.04,2.0,200,50\n"
        "Banks,20,0.02,0.05,0.3,200,50\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "lake"
    result = CliRunner().invoke(
        app,
        [
            "build-sector-pool-v7",
            "--ic-report",
            str(ic_path),
            "--output-root",
            str(output_root),
            "--min-dates",
            "60",
            "--min-symbols",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_root / "silver" / "sector_pool" / "sector_pool.parquet").exists()
    assert (output_root / "manifests" / "sector_pool.json").exists()


def test_cli_build_sector_pool_from_stratified_ic_json(tmp_path):
    """Accept the JSON shape stratified_ic_report.py writes directly."""
    from typer.testing import CliRunner

    from quantagent.cli import app

    payload = {
        "tables": {
            "sector_level_1": [
                {"bucket": "Food", "horizon": 20, "ic_mean": 0.08, "ic_std": 0.04, "ic_ir": 2.0, "n_dates": 200, "n_symbols": 50},
                {"bucket": "Coal", "horizon": 20, "ic_mean": -0.02, "ic_std": 0.05, "ic_ir": -0.4, "n_dates": 200, "n_symbols": 50},
            ]
        }
    }
    ic_path = tmp_path / "stratified_ic.json"
    ic_path.write_text(json.dumps(payload), encoding="utf-8")
    output_root = tmp_path / "lake"
    result = CliRunner().invoke(
        app,
        [
            "build-sector-pool-v7",
            "--ic-report",
            str(ic_path),
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    written = pd.read_parquet(output_root / "silver" / "sector_pool" / "sector_pool.parquet")
    tiers = dict(zip(written["sector_level_1"], written["pool_tier"]))
    assert tiers["Food"] == "core"
    assert tiers["Coal"] == "excluded"
