import numpy as np
import pandas as pd
import pytest

import quantagent.research.intraday_dot_factor_combo as combo_mod
from quantagent.execution.broker_base import OrderSide
from quantagent.research.intraday_dot_factor_combo import (
    FactorComboConfig,
    _attach_excess_metrics,
    build_factor_combo_dataset,
    build_dot_outcomes,
    build_entry_relative_strength_features,
    _causal_entry_features,
    _evaluate_selection,
    _fast_conservative_fill,
    _fast_conservative_window_fill,
    _simulate_fixed_time_prepared,
    train_factor_combo_model,
    verdict_from_metrics,
)
from quantagent.execution.selective_dot import SelectiveDotParams


def test_train_factor_combo_uses_validation_policy_and_eod_probabilities():
    rows = []
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    for i, dt in enumerate(dates):
        for j, sym in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
            edge = (j - 1) * 20 + i
            rows.append({
                "symbol": sym,
                "trade_date": dt,
                "target_net_ret_bps": float(edge),
                "gross_ret": float(edge + 26.0) / 10_000.0,
                "net_ret": float(edge) / 10_000.0,
                "state": "closed_eod" if (i + j) % 5 == 0 else "closed_profit",
                "eod_restore": int((i + j) % 5 == 0),
                "atr_pct": 0.03 + j * 0.001,
                "mom_5d": 0.01 * (j - 1),
                "gap_open": 0.001 * i,
                "mode": "dip_buy",
                "mode_dip_buy": 1.0,
                "dip_atr_mult": 0.3,
                "target_atr_mult": 0.5,
                "stop_atr_mult": 0.5,
                "regime": "sideways",
                "regime_bull": 0.0,
                "regime_sideways": 1.0,
                "entry_minute_idx": 10 + i,
                "entry_price_vs_vwap_prev": 0.001 * (j - 1),
                "entry_return_from_open": 0.002 * j,
                "entry_range_pos_sofar": 0.5,
                "entry_rolling_return_3m": 0.001,
                "entry_volume_zscore_5m": 0.1 * j,
                "entry_cum_volume": 100_000 + i,
                "entry_participation_rate": 0.03,
                "entry_volume_capacity_ratio": 0.6,
                "exit_volume_capacity_ratio": 0.7,
                "weight": np.nan,
            })
    cfg = FactorComboConfig(
        split="2026-01-05",
        validation_split="2026-01-08",
        top_fracs=(0.5, 1.0),
        eod_restore_prob_caps=(0.5, 1.0),
        min_train_legs=5,
        min_validation_legs=1,
        min_oos_legs=1,
        selection_book_only=False,
    )
    _, feature_cols, scored, metrics = train_factor_combo_model(pd.DataFrame(rows), config=cfg, backend="sklearn")
    assert metrics["policy_selected_on"] == "validation"
    assert "validation" in metrics
    assert "pred_eod_restore_prob" in scored.columns
    assert scored["pred_eod_restore_prob"].between(0, 1).all()
    assert "pred_stop_prob" in scored.columns
    assert scored["pred_stop_prob"].between(0, 1).all()
    assert "pred_policy_score_bps" in scored.columns
    assert scored["pred_policy_score_bps"].notna().all()
    assert "chosen_max_stop_prob" in metrics
    assert "avg_stop_prob" in metrics["validation"]
    assert "avg_policy_score_bps" in metrics["validation"]
    assert metrics["book_candidate_coverage"]["validation"]["candidate_symbol_days"] > 0
    assert metrics["book_candidate_coverage"]["validation"]["book_symbol_days"] == 0
    assert "entry_minute_idx" in feature_cols

    test_metrics = metrics["test"]
    assert "test_baselines" in metrics
    assert test_metrics["daily_uplift_bps_excess"] == pytest.approx(
        test_metrics["daily_uplift_bps"] - test_metrics["baseline_daily_uplift_bps"]
    )
    assert test_metrics["excess_baseline_name"] in {
        "no_trade",
        "random_time_same_count_baseline",
        "shuffled_signal_baseline",
        "vwap_only_baseline",
    }


def test_attach_excess_metrics_uses_strongest_baseline_not_zero() -> None:
    metrics = {"daily_uplift_bps": 12.0, "n_legs": 20}
    baselines = {
        "random_time_same_count_baseline": {"daily_uplift_bps": 5.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 8.0},
        "vwap_only_baseline": {"daily_uplift_bps": 15.0},
    }

    out = _attach_excess_metrics(metrics, baselines)

    assert out["baseline_daily_uplift_bps"] == 15.0
    assert out["excess_baseline_name"] == "vwap_only_baseline"
    assert out["daily_uplift_bps_excess"] == -3.0
    assert out["daily_uplift_bps_excess_vs_shuffled_signal_baseline"] == 4.0


def test_book_only_selection_ranks_only_positive_holdings() -> None:
    df = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "pred_net_ret_bps": [100.0, 10.0],
        "pred_policy_score_bps": [100.0, 10.0],
        "net_ret": [0.02, 0.001],
        "gross_ret": [0.021, 0.002],
        "state": ["closed_profit", "closed_profit"],
        "weight": [np.nan, 0.02],
    })
    cfg = FactorComboConfig(selection_book_only=True)

    metrics = _evaluate_selection(df, frac=1.0, config=cfg, name="book_only")

    assert metrics["n_legs"] == 1
    assert metrics["book_n_legs"] == 1
    assert metrics["mean_net_bps"] == pytest.approx(10.0)


def test_fixed_time_candidate_enters_at_configured_time_without_future_data() -> None:
    day = {
        "open": np.array([10.0, 10.0, 10.0, 10.1, 10.2]),
        "high": np.array([10.0, 10.0, 10.02, 10.25, 10.3]),
        "low": np.array([10.0, 9.99, 9.98, 10.0, 10.1]),
        "close": np.array([10.0, 10.0, 10.0, 10.2, 10.2]),
        "volume": np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
        "amount": np.array([10000.0, 10000.0, 10000.0, 10200.0, 10200.0]),
        "vwap_prev": np.array([np.nan, 10.0, 10.0, 10.0, 10.1]),
        "time": np.array(["09:30:00", "09:31:00", "10:00:00", "10:01:00", "10:02:00"]),
    }
    params = SelectiveDotParams(
        mode="time_buy",
        morning_deadline="10:00:00",
        target_atr_mult=0.10,
        stop_atr_mult=0.20,
        eod_close="10:02:00",
        min_bars_before_entry=1,
    )

    state, entry_px, exit_px, ret, entry_idx, exit_idx = _simulate_fixed_time_prepared(day, 0.10, params, "time_buy")

    assert state == "closed_profit"
    assert entry_idx == 2
    assert exit_idx == 3
    assert entry_px == pytest.approx(10.0)
    assert exit_px == pytest.approx(10.1)
    assert ret > 0


def test_book_only_dataset_build_filters_contexts_to_holdings(tmp_path, monkeypatch) -> None:
    minute_dir = tmp_path / "minute"
    minute_dir.mkdir()
    for sym in ["000001.SZ", "000002.SZ"]:
        (minute_dir / f"{sym}.parquet").write_bytes(b"placeholder")
    market_panel = tmp_path / "market_panel.parquet"
    panel = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "open": [10.0, 20.0],
        "high": [10.1, 20.1],
        "low": [9.9, 19.9],
        "close": [10.0, 20.0],
    })
    panel.to_parquet(market_panel, index=False)
    holdings = tmp_path / "holdings.csv"
    pd.DataFrame({
        "trade_date": ["2026-01-02"],
        "symbol": ["000002.SZ"],
        "weight": [0.02],
    }).to_csv(holdings, index=False)

    contexts = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ"],
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "atr_pct": [0.01, 0.02],
        "prev_close": [10.0, 20.0],
        "mom_5d": [0.0, 0.0],
        "gap_open": [0.0, 0.0],
        "regime": ["sideways", "sideways"],
    })
    captured = {}

    def fake_build_day_contexts(_panel):
        return contexts.copy()

    def fake_build_dot_outcomes(*, minute_dir, symbols, contexts, start, end, config):
        captured["symbols"] = list(symbols)
        captured["contexts"] = contexts.copy()
        return pd.DataFrame({
            "symbol": ["000002.SZ"],
            "trade_date": pd.to_datetime(["2026-01-02"]),
            "combo": [1],
            "mode": ["dip_buy"],
            "dip_atr_mult": [0.3],
            "target_atr_mult": [0.4],
            "stop_atr_mult": [0.6],
            "morning_deadline": ["10:00:00"],
            "tail_exit_time": ["14:25:00"],
            "state": ["closed_profit"],
            "eod_restore": [0],
            "time_exit": [0],
            "gross_ret": [0.002],
            "net_ret": [0.001],
            "entry_idx": [3],
            "entry_fill_status": ["filled"],
            "entry_price_vs_vwap_prev": [0.0],
            "fee_cost_bps": [26.0],
            "entry_order_flow_imbalance_5m": [0.0],
            "entry_mode_adverse_risk": [0.0],
        })

    def fake_intraday_factors(**_kwargs):
        return pd.DataFrame({
            "symbol": ["000002.SZ"],
            "trade_date": pd.to_datetime(["2026-01-02"]),
        })

    monkeypatch.setattr(combo_mod, "build_day_contexts", fake_build_day_contexts)
    monkeypatch.setattr(combo_mod, "build_dot_outcomes", fake_build_dot_outcomes)
    monkeypatch.setattr(combo_mod, "load_or_build_intraday_factors", fake_intraday_factors)

    cfg = FactorComboConfig(start="2026-01-02", end="2026-01-02", selection_book_only=True)
    dataset, filtered_contexts = build_factor_combo_dataset(
        minute_dir=minute_dir,
        market_panel_path=market_panel,
        intraday_factors_path=tmp_path / "missing.parquet",
        holdings_csv=holdings,
        config=cfg,
    )

    assert captured["symbols"] == ["000002.SZ"]
    assert filtered_contexts["symbol"].tolist() == ["000002.SZ"]
    assert captured["contexts"]["symbol"].tolist() == ["000002.SZ"]
    assert dataset["symbol"].tolist() == ["000002.SZ"]
    assert dataset["weight"].tolist() == [0.02]
    assert dataset["book_only_context"].tolist() == [True]


def test_verdict_requires_validation_excess_before_enable() -> None:
    metrics = {
        "validation": {
            "n_legs": 120,
            "mean_net_bps": -1.0,
            "daily_uplift_bps_excess": -0.5,
            "eod_restore_rate": 0.1,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 0.0},
        "vwap_only_baseline": {"daily_uplift_bps": 0.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "DO_NOT_ENABLE"
    assert "validation" in reason


def test_verdict_reports_book_only_oos_candidate_coverage_when_no_legs() -> None:
    metrics = {
        "selection_book_only": True,
        "book_candidate_coverage": {
            "test": {
                "book_symbol_days": 30,
                "book_positive_pred_symbol_days": 3,
            }
        },
        "validation": {
            "n_legs": 120,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
        },
        "test": {"n_legs": 0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "DO_NOT_ENABLE"
    assert "book candidates 30" in reason
    assert "positive-pred candidates 3" in reason


def test_verdict_rejects_insufficient_validation_legs() -> None:
    metrics = {
        "min_validation_legs": 100,
        "min_oos_legs": 300,
        "validation": {
            "n_legs": 4,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 0.0},
        "vwap_only_baseline": {"daily_uplift_bps": 0.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "DO_NOT_ENABLE"
    assert reason == "validation selected legs below minimum"


def test_verdict_requires_validation_stop_gate() -> None:
    metrics = {
        "min_validation_legs": 100,
        "min_oos_legs": 300,
        "max_validation_stop_rate": 0.35,
        "validation": {
            "n_legs": 120,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.6,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 0.0},
        "vwap_only_baseline": {"daily_uplift_bps": 0.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "DO_NOT_ENABLE"
    assert reason == "validation net/excess/risk gates failed"


def test_verdict_enables_only_when_validation_and_oos_gates_pass() -> None:
    metrics = {
        "validation": {
            "n_legs": 120,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 40,
            "book_daily_uplift_bps": 1.0,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 120,
            "book_daily_uplift_bps": 1.5,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 1.0},
        "vwap_only_baseline": {"daily_uplift_bps": 2.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "ENABLE"
    assert reason == "conservative OOS excess and book gates passed"


def test_verdict_does_not_enable_without_book_executable_legs() -> None:
    metrics = {
        "require_book_for_enable": True,
        "min_validation_book_legs": 30,
        "min_oos_book_legs": 100,
        "validation": {
            "n_legs": 120,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 0,
            "book_daily_uplift_bps": 0.0,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 0,
            "book_daily_uplift_bps": 0.0,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 1.0},
        "vwap_only_baseline": {"daily_uplift_bps": 2.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "PAPER_ONLY"
    assert "book executable legs below minimum" in reason


def test_verdict_rejects_negative_book_uplift_after_oos_pass() -> None:
    metrics = {
        "require_book_for_enable": True,
        "min_validation_book_legs": 1,
        "min_oos_book_legs": 1,
        "validation": {
            "n_legs": 120,
            "mean_net_bps": 10.0,
            "daily_uplift_bps_excess": 2.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 10,
            "book_daily_uplift_bps": 1.0,
        },
        "test": {
            "n_legs": 350,
            "mean_net_bps": 10.0,
            "daily_uplift_bps": 3.0,
            "daily_uplift_bps_excess": 3.0,
            "eod_restore_rate": 0.1,
            "stop_rate": 0.1,
            "book_n_legs": 10,
            "book_daily_uplift_bps": -0.1,
        },
        "random_time_same_count_baseline": {"daily_uplift_bps": 0.0},
        "shuffled_signal_baseline": {"daily_uplift_bps": 1.0},
        "vwap_only_baseline": {"daily_uplift_bps": 2.0},
    }

    verdict, reason = verdict_from_metrics(metrics)

    assert verdict == "DO_NOT_ENABLE"
    assert reason == "book-level validation/OOS uplift gate failed"


def test_fast_conservative_fill_uses_next_bar_and_five_percent_capacity():
    day = {
        "open": np.array([10.0, 10.1, 10.2]),
        "high": np.array([10.1, 10.2, 10.3]),
        "low": np.array([9.9, 10.0, 10.1]),
        "close": np.array([10.0, 10.1, 10.2]),
        "volume": np.array([1000.0, 10_000.0, 10_000.0]),
        "time": np.array(["09:30:00", "09:31:00", "09:32:00"]),
    }
    cfg = FactorComboConfig(order_notional_yuan=5_000.0, max_minute_participation=0.05)
    fill = _fast_conservative_fill(
        day,
        signal_index=0,
        side=OrderSide.BUY,
        quantity=800,
        config=cfg,
        limit_up=np.nan,
        limit_down=np.nan,
    )
    assert fill["fill_time"] == "09:31:00"
    assert fill["filled_qty"] == 500
    assert fill["status"] == "partial"
    assert round(fill["participation_rate"], 4) == 0.05


def test_tail_exit_window_slices_across_multiple_minutes():
    day = {
        "open": np.array([10.0, 10.1, 10.2, 10.3]),
        "high": np.array([10.1, 10.2, 10.3, 10.4]),
        "low": np.array([9.9, 10.0, 10.1, 10.2]),
        "close": np.array([10.0, 10.1, 10.2, 10.3]),
        "volume": np.array([1000.0, 10_000.0, 10_000.0, 10_000.0]),
        "time": np.array(["14:09:00", "14:10:00", "14:11:00", "14:12:00"]),
    }
    cfg = FactorComboConfig(order_notional_yuan=5_000.0, max_minute_participation=0.05)
    fill = _fast_conservative_window_fill(
        day,
        signal_index=0,
        side=OrderSide.SELL,
        quantity=800,
        config=cfg,
        limit_up=np.nan,
        limit_down=np.nan,
        latest_time="14:12:00",
    )
    assert fill["filled_qty"] == 800
    assert fill["status"] == "filled"
    assert fill["reason"] == "filled_window"
    assert fill["fill_time"] == "14:11:00"


def test_causal_entry_features_include_order_flow_and_adverse_risk():
    day = {
        "open": np.array([10.0, 10.1, 10.2, 10.1, 10.0, 9.9]),
        "high": np.array([10.1, 10.2, 10.3, 10.2, 10.1, 10.0]),
        "low": np.array([9.9, 10.0, 10.1, 10.0, 9.9, 9.8]),
        "close": np.array([10.0, 10.1, 10.2, 10.1, 10.0, 9.9]),
        "volume": np.array([1000.0, 2000.0, 3000.0, 5000.0, 6000.0, 7000.0]),
        "amount": np.array([10000.0, 20200.0, 30600.0, 50500.0, 60000.0, 69300.0]),
        "vwap_prev": np.array([np.nan, 10.0, 10.05, 10.1, 10.1, 10.05]),
        "time": np.array(["09:30:00", "09:31:00", "09:32:00", "09:33:00", "09:34:00", "09:35:00"]),
    }
    features = _causal_entry_features(day, 5, mode="dip_buy")
    assert "entry_order_flow_imbalance_5m" in features
    assert "entry_price_impact_bps_5m" in features
    assert "entry_mode_adverse_risk" in features
    assert 0.0 <= features["entry_mode_adverse_risk"] <= 1.0
    assert features["entry_downtrend_persistence_10m"] > 0


def test_relative_strength_features_use_entry_minute_and_sector(tmp_path):
    minute_dir = tmp_path / "minute"
    minute_dir.mkdir()
    rows = []
    for sym, sector, step in [("000001.SZ", "银行", 0.01), ("000002.SZ", "银行", 0.0), ("000003.SZ", "地产", -0.01)]:
        times = pd.date_range("2026-01-02 09:30:00", periods=3, freq="min")
        close = np.array([10.0, 10.0 + step * 100, 10.0 + step * 200])
        df = pd.DataFrame({
            "symbol": sym,
            "trade_time": times,
            "open": [10.0, 10.0, 10.0],
            "close": close,
            "volume": [1000, 2000, 3000],
            "amount": close * np.array([1000, 2000, 3000]),
        })
        df.to_parquet(minute_dir / f"{sym}.parquet", index=False)
        rows.append({"symbol": sym, "sector_level_1": sector})
    sector_path = tmp_path / "sector.parquet"
    pd.DataFrame(rows).to_parquet(sector_path, index=False)
    outcomes = pd.DataFrame({
        "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
        "trade_date": pd.to_datetime(["2026-01-02"] * 3),
        "entry_idx": [2, 2, 2],
    })
    features = build_entry_relative_strength_features(
        outcomes,
        minute_dir=minute_dir,
        sector_map_path=sector_path,
        cache_dir=tmp_path / "feature_cache",
    )
    assert len(features) == 3
    strong = features.loc[features["symbol"] == "000001.SZ"].iloc[0]
    assert strong["entry_stock_return_minus_market"] > 0
    assert "entry_industry_return_sofar" in features.columns
    assert list((tmp_path / "feature_cache").glob("entry_relative_*.parquet"))


def test_build_dot_outcomes_writes_symbol_cache(tmp_path):
    minute_dir = tmp_path / "minute"
    minute_dir.mkdir()
    times = pd.date_range("2026-01-02 09:30:00", periods=20, freq="min")
    close = np.linspace(10.0, 10.5, len(times))
    minute = pd.DataFrame({
        "symbol": "000001.SZ",
        "trade_date": [pd.Timestamp("2026-01-02")] * len(times),
        "trade_time": times,
        "open": close,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": [100_000] * len(times),
        "amount": close * 100_000,
    })
    minute.to_parquet(minute_dir / "000001.SZ.parquet", index=False)
    contexts = pd.DataFrame({
        "symbol": ["000001.SZ"],
        "trade_date": [pd.Timestamp("2026-01-02")],
        "atr_pct": [0.02],
        "prev_close": [10.0],
        "mom_5d": [0.0],
        "gap_open": [0.0],
        "regime": ["sideways"],
    })
    cfg = FactorComboConfig(
        start="2026-01-02",
        end="2026-01-02",
        order_notional_yuan=2_000,
        outcome_cache_dir=str(tmp_path / "outcome_cache"),
    )
    build_dot_outcomes(
        minute_dir=minute_dir,
        symbols=["000001.SZ"],
        contexts=contexts,
        start=pd.Timestamp("2026-01-02"),
        end=pd.Timestamp("2026-01-02"),
        config=cfg,
    )
    assert list((tmp_path / "outcome_cache").glob("000001.SZ_*.parquet"))


def test_event_exit_uses_window_fill_before_forced_eod(tmp_path):
    minute_dir = tmp_path / "minute"
    minute_dir.mkdir()
    times = pd.date_range("2026-01-02 09:30:00", periods=20, freq="min")
    close = np.full(len(times), 10.0)
    open_ = np.full(len(times), 10.0)
    high = np.full(len(times), 10.02)
    low = np.full(len(times), 9.99)
    low[5] = 9.95
    high[6] = 10.12
    open_[8] = 10.03
    close[8] = 10.03
    volume = np.full(len(times), 100_000.0)
    volume[7] = 0.0
    minute = pd.DataFrame({
        "symbol": "000001.SZ",
        "trade_date": [pd.Timestamp("2026-01-02")] * len(times),
        "trade_time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": close * volume,
    })
    minute.to_parquet(minute_dir / "000001.SZ.parquet", index=False)
    contexts = pd.DataFrame({
        "symbol": ["000001.SZ"],
        "trade_date": [pd.Timestamp("2026-01-02")],
        "atr_pct": [0.02],
        "prev_close": [10.0],
        "mom_5d": [0.02],
        "gap_open": [0.0],
        "regime": ["bull"],
    })
    cfg = FactorComboConfig(
        start="2026-01-02",
        end="2026-01-02",
        order_notional_yuan=2_000,
        max_minute_participation=0.05,
        min_fill_ratio=1.0,
    )

    outcomes = build_dot_outcomes(
        minute_dir=minute_dir,
        symbols=["000001.SZ"],
        contexts=contexts,
        start=pd.Timestamp("2026-01-02"),
        end=pd.Timestamp("2026-01-02"),
        config=cfg,
    )

    event_window_fills = outcomes[
        outcomes["state"].eq("closed_profit")
        & outcomes["exit_fill_reason"].eq("filled_window")
        & outcomes["exit_fill_time"].eq("09:38:00")
    ]
    assert not event_window_fills.empty
