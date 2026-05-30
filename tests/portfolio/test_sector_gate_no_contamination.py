from __future__ import annotations

import json

import pandas as pd
from pandas.testing import assert_frame_equal


def _write_manifest(path, *, sector_opt=False, st_risk=False):
    payload = {
        "extra": {
            "coverage_report": {
                "gate": {
                    "sector_usable_for_diagnostics": True,
                    "sector_usable_for_optimization": bool(sector_opt),
                    "st_usable_for_risk_filter": bool(st_risk),
                    "reason": "test_gate",
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_st_manifest(path, *, st_risk=False):
    payload = {
        "extra": {
            "coverage_report": {
                "gate": {
                    "st_usable_for_risk_filter": bool(st_risk),
                    "reason": "test_gate",
                    "policy": {"st_block_weight": 0.9, "suspended_block_weight": 1.0},
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")] * 4,
            "symbol": ["600519.SH", "000001.SZ", "300750.SZ", "688001.SH"],
            "prediction": [0.9, 0.8, 0.7, 0.6],
            "confidence": [0.9, 0.9, 0.9, 0.9],
        }
    )


def _market_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")] * 4,
            "symbol": ["600519.SH", "000001.SZ", "300750.SZ", "688001.SH"],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [1_000_000] * 4,
            "amount": [100_000_000.0] * 4,
        }
    )


def test_target_weights_unchanged_when_sector_optimization_disabled(tmp_path):
    from quantagent.diagnostics.sector_audit import sector_map_for_optimization
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    manifest = tmp_path / "manifests" / "sector_map.json"
    _write_manifest(manifest, sector_opt=False)
    sector_map = pd.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ", "300750.SZ", "688001.SH"],
            "sector_level_1": ["A", "A", "B", "B"],
            "sector_level_2": ["A1", "A1", "B1", "B1"],
            "available_at": [pd.Timestamp("2024-01-01")] * 4,
            "coverage_status": ["pit_historical"] * 4,
            "source": ["manual_vendor_sector"] * 4,
        }
    )
    cfg = V7TargetWeightsConfig(
        selection_mode="top_k",
        top_k=2,
        top_k_ratio=None,
        fail_if_top_k_covers_universe=False,
        max_turnover=0.0,
        block_st=False,
        block_suspended=False,
        block_limit_up_buy=False,
        block_limit_down_sell=False,
        min_selection_pressure=1.0,
    )

    without_sector = build_v7_target_weights(_predictions(), _market_panel(), sector_map=None, config=cfg).target_weights
    guarded_sector = sector_map_for_optimization(sector_map, manifest)
    with_disabled_sector = build_v7_target_weights(_predictions(), _market_panel(), sector_map=guarded_sector, config=cfg).target_weights

    assert guarded_sector is None
    assert_frame_equal(without_sector.sort_index(axis=1), with_disabled_sector.sort_index(axis=1))


def test_st_soft_block_noop_when_st_gate_disabled(tmp_path):
    from quantagent.diagnostics.sector_audit import st_flags_for_risk_filter
    from quantagent.universe.filters import UniverseFilterConfig, apply_universe_filter

    manifest = tmp_path / "manifests" / "st_flags.json"
    _write_st_manifest(manifest, st_risk=False)
    preds = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")] * 5,
            "symbol": [f"ST{i}.SZ" for i in range(5)],
            "prediction": [1.0, 0.8, 0.6, 0.4, 0.2],
        }
    )
    st_flags = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")] * 5,
            "symbol": [f"ST{i}.SZ" for i in range(5)],
            "is_st": [True] * 5,
        }
    )
    cfg = UniverseFilterConfig(
        st_min_block_rate=0.90,
        suspended_block_new=False,
        limit_up_block_new=False,
        high_chase_enabled=False,
    )

    guarded = st_flags_for_risk_filter(st_flags, manifest)
    result = apply_universe_filter(preds, st_flags=guarded, config=cfg)

    assert guarded is None
    assert result.filtered_predictions["universe_pass"].all()


def test_suspended_hard_block_still_works_when_st_gate_disabled(tmp_path):
    from quantagent.diagnostics.sector_audit import st_flags_for_risk_filter
    from quantagent.universe.filters import UniverseFilterConfig, apply_universe_filter

    manifest = tmp_path / "manifests" / "st_flags.json"
    _write_st_manifest(manifest, st_risk=False)
    date = pd.Timestamp("2024-01-02")
    preds = pd.DataFrame({"trade_date": [date, date], "symbol": ["A.SZ", "B.SZ"], "prediction": [0.5, 0.4]})
    market_rows = [
        {"trade_date": date, "symbol": "A.SZ", "close": 10.0, "volume": 1000, "amount": 10000.0},
        {"trade_date": date, "symbol": "B.SZ", "close": 10.0, "volume": 0, "amount": 0.0},
    ]
    for idx in range(100):
        market_rows.append({"trade_date": date, "symbol": f"X{idx:03d}.SZ", "close": 10.0, "volume": 1000, "amount": 10000.0})
    cfg = UniverseFilterConfig(limit_up_block_new=False, high_chase_enabled=False)

    result = apply_universe_filter(
        preds,
        market_panel=pd.DataFrame(market_rows),
        st_flags=st_flags_for_risk_filter(pd.DataFrame(), manifest),
        config=cfg,
    )

    by_symbol = result.filtered_predictions.set_index("symbol")
    assert bool(by_symbol.loc["A.SZ", "universe_pass"]) is True
    assert bool(by_symbol.loc["B.SZ", "universe_pass"]) is False
    assert by_symbol.loc["B.SZ", "universe_reason"] == "suspended"
