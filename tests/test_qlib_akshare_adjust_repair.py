"""Regression test for scripts/fix_qlib_akshare_adjust.py.

Constructs a synthetic merged panel where the qlib leg is normalised
(first_close=1.0 per symbol) and the akshare leg is in real today-anchored
prices, then asserts that ``repair_panel`` rescales the qlib OHLC rows to
restore boundary continuity without touching volume/amount.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fix_qlib_akshare_adjust.py"


@pytest.fixture(scope="module")
def repair_module():
    spec = importlib.util.spec_from_file_location("fix_qlib_akshare_adjust", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fix_qlib_akshare_adjust"] = module
    spec.loader.exec_module(module)
    return module


def _make_panel() -> pd.DataFrame:
    """Build a 2-symbol panel with a clean qlib→akshare boundary."""
    rows: list[dict[str, object]] = []
    # ---- A: qlib 30 days normalised, then akshare 5 days real ----
    qlib_dates = pd.date_range("2020-01-02", periods=30, freq="B")
    qlib_closes = np.linspace(1.0, 1.07, 30)  # gentle drift, first=1.0 normalised
    ak_dates = pd.date_range("2020-02-13", periods=5, freq="B")
    ak_first_close = 32.10  # ratio ≈ 32.10 / 1.07 ≈ 30
    ak_closes = ak_first_close * (1.0 + np.array([0.0, 0.003, -0.002, 0.001, 0.004]))
    for d, c in zip(qlib_dates, qlib_closes):
        rows.append({
            "symbol": "A", "trade_date": d,
            "open": c * 0.998, "high": c * 1.005, "low": c * 0.995, "close": c,
            "volume": 1_000_000, "amount": c * 1_000_000,
            "source": "qlib", "available_at": d + pd.Timedelta(days=1),
        })
    for d, c in zip(ak_dates, ak_closes):
        rows.append({
            "symbol": "A", "trade_date": d,
            "open": c * 0.998, "high": c * 1.005, "low": c * 0.995, "close": c,
            "volume": 2_000_000, "amount": c * 2_000_000,
            "source": "akshare", "available_at": d + pd.Timedelta(days=1),
        })

    # ---- B: same pattern, different ratio ----
    qlib_dates_b = pd.date_range("2020-01-02", periods=20, freq="B")
    qlib_closes_b = np.linspace(1.0, 1.05, 20)
    ak_dates_b = pd.date_range("2020-01-30", periods=4, freq="B")
    ak_first_close_b = 8.40  # ratio ≈ 8.40 / 1.05 = 8.0
    ak_closes_b = ak_first_close_b * (1.0 + np.array([0.0, -0.001, 0.002, 0.0]))
    for d, c in zip(qlib_dates_b, qlib_closes_b):
        rows.append({
            "symbol": "B", "trade_date": d,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 500_000, "amount": c * 500_000,
            "source": "qlib", "available_at": d + pd.Timedelta(days=1),
        })
    for d, c in zip(ak_dates_b, ak_closes_b):
        rows.append({
            "symbol": "B", "trade_date": d,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 800_000, "amount": c * 800_000,
            "source": "akshare", "available_at": d + pd.Timedelta(days=1),
        })

    return pd.DataFrame(rows)


def test_repair_panel_rescales_qlib_to_match_akshare_boundary(tmp_path, repair_module):
    panel = _make_panel()
    panel_path = tmp_path / "market_panel.csv"
    panel.to_csv(panel_path, index=False)
    manifest_path = tmp_path / "manifests" / "market_panel.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "dataset_name": "market_panel",
        "vendor": "qlib+akshare",
        "row_count": int(len(panel)),
        "start_date": "2020-01-02",
        "end_date": "2020-02-19",
    }, indent=2), encoding="utf-8")

    summary = repair_module.repair_panel(panel_path, manifest_path, dry_run=False)
    assert summary["status"] == "repaired"
    assert summary["symbols_rescaled"] == 2
    assert summary["symbols_discarded"] == 0

    # Read the rebuilt parquet and confirm volume/amount preserved on qlib rows,
    # and close magnitudes are now close to akshare scale.
    rebuilt = pd.read_parquet(tmp_path / "market_panel.parquet")
    qlib_rows = rebuilt[rebuilt["source"] == "qlib_rescaled"]
    assert not qlib_rows.empty
    assert qlib_rows[qlib_rows["symbol"] == "A"]["volume"].iloc[0] == 1_000_000
    assert qlib_rows[qlib_rows["symbol"] == "B"]["volume"].iloc[0] == 500_000

    # Boundary continuity for A: last qlib_rescaled close ≈ first akshare close
    a = rebuilt[rebuilt["symbol"] == "A"].sort_values("trade_date").reset_index(drop=True)
    boundary_idx = a.index[a["source"] == "akshare"].min()
    assert boundary_idx > 0
    last_qlib_close = float(a.iloc[boundary_idx - 1]["close"])
    first_ak_close = float(a.iloc[boundary_idx]["close"])
    rel_jump = abs(np.log(first_ak_close / last_qlib_close))
    assert rel_jump < 0.05, f"boundary jump too large after repair: {rel_jump:.4f}"

    # Manifest refreshed with adjustment_repair record
    manifest = json.loads(manifest_path.read_text())
    assert manifest["vendor"] == "qlib_rescaled+akshare"
    assert manifest["extra"]["adjustment_repair"]["symbols_rescaled"] == 2


def test_repair_panel_dry_run(tmp_path, repair_module):
    panel = _make_panel()
    panel_path = tmp_path / "market_panel.csv"
    panel.to_csv(panel_path, index=False)
    manifest_path = tmp_path / "market_panel.json"
    summary = repair_module.repair_panel(panel_path, manifest_path, dry_run=True)
    assert summary["status"] == "dry_run"
    assert summary["symbols_bridged"] == 2
    # No parquet should be written in dry-run mode.
    assert not (tmp_path / "market_panel.parquet").exists()
