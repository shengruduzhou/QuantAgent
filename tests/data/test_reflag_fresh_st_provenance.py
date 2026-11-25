"""Cycle 32: broadcast ST rows are re-derived only where dated evidence exists."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from quantagent.data.providers.st_pit import load_historical_st_evidence

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reflag_fresh_st_provenance.py"


def _module():
    spec = importlib.util.spec_from_file_location("reflag_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reflag_uses_dated_evidence_and_marks_the_rest_unknown(tmp_path) -> None:
    mod = _module()
    days = pd.to_datetime(["2026-07-23", "2026-07-24", "2026-07-27"])
    panel = pd.DataFrame({
        "symbol": ["000001.SZ"] * 3 + ["600000.SH"] * 3,
        "trade_date": list(days) * 2,
        "is_st": [True, True, True, True, True, True],          # broadcast: ST everywhere
        "is_st_provenance": pd.array(["current_snapshot_broadcast"] * 6, dtype="string"),
        "st_evidence_sha256": pd.array([None] * 6, dtype="string"),
    })
    evidence = pd.DataFrame({
        "symbol": ["000001.SZ", "600000.SH"],
        "start_date": ["2016-01-04", "2026-07-27"],
        "end_date": ["2026-11-25", "2026-11-24"],
        "is_st": [False, False],
        "available_at": ["2016-01-04T00:00:00+08:00", "2026-07-27T00:00:00+08:00"],
    })
    path = tmp_path / "ev.parquet"
    evidence.to_parquet(path)
    out, report = mod.reflag(panel, load_historical_st_evidence(path))

    sz = out[out["symbol"] == "000001.SZ"]
    assert not sz["is_st"].any()
    assert set(sz["is_st_provenance"]) == {mod.DATED}
    sh = out[out["symbol"] == "600000.SH"].set_index("trade_date")
    # SSE before the forward register: state untouched, provenance explicitly unknown
    assert sh.loc[days[:2], "is_st"].all()
    assert set(sh.loc[days[:2], "is_st_provenance"]) == {mod.UNKNOWN}
    assert sh.loc[days[2], "is_st_provenance"] == mod.DATED and not sh.loc[days[2], "is_st"]
    assert report["by_exchange"]["SH"]["unknown_rows"] == 2
    assert report["by_exchange"]["SZ"]["flipped_true_to_false"] == 3
    # the unknown label must still trip the dataset builder's broadcast refusal
    assert "current_snapshot" in mod.UNKNOWN and "broadcast" in mod.UNKNOWN


def test_reflag_refuses_live_and_existing_targets(tmp_path) -> None:
    import pytest

    mod = _module()
    panel = tmp_path / "in.parquet"
    with pytest.raises(SystemExit, match="live"):
        mod.check_output_path(mod.DEFAULT_PANEL, panel)
    with pytest.raises(SystemExit, match="live panel directory"):
        mod.check_output_path(mod.DEFAULT_PANEL.parent / "market_panel.v2.parquet", panel)
    with pytest.raises(SystemExit, match="input"):
        mod.check_output_path(panel, panel)
    existing = tmp_path / "out.parquet"
    existing.write_bytes(b"x")
    with pytest.raises(SystemExit, match="existing"):
        mod.check_output_path(existing, panel)
    mod.check_output_path(tmp_path / "new.parquet", panel)   # fresh path is fine
