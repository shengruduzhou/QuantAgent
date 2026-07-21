from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import polars as pl
import pytest

from services.quant_api.config import ApiSettings


@pytest.fixture
def quant_ui_settings(tmp_path: Path) -> ApiSettings:
    project_root = tmp_path / "QuantAgent"
    runtime = project_root / "runtime"
    settings = ApiSettings(
        project_root=project_root,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
        index_ttl_seconds=300,
    ).ensure()
    _write_runtime_fixture(settings)
    return settings


@pytest.fixture
def empty_quant_ui_settings(tmp_path: Path) -> ApiSettings:
    project_root = tmp_path / "EmptyQuantAgent"
    runtime = project_root / "runtime"
    return ApiSettings(
        project_root=project_root,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
        index_ttl_seconds=300,
    ).ensure()


def _write_runtime_fixture(settings: ApiSettings) -> None:
    runtime = settings.runtime_root
    backtest = runtime / "reports" / "v8" / "deep" / "fixture_run" / "short_5d" / "backtest"
    backtest.mkdir(parents=True)
    metrics_path = backtest / "metrics.json"
    metrics_path.write_text(
        json.dumps({
            "total_return": 0.12,
            "annualized_return": 0.18,
            "max_drawdown": 0.07,
            "sharpe": 1.2,
            "calmar": 2.5,
            "turnover": 0.3,
            "win_rate": 0.6,
            "profit_factor": 1.4,
            "n_trades": 1,
            "n_fills": 2,
            "start_date": "2026-01-02",
            "end_date": "2026-01-05",
        }),
        encoding="utf-8",
    )
    (backtest / "metrics.json.manifest.json").write_text(
        json.dumps({
            "schema_version": "quantagent.backtest.metrics.1",
            "created_at": "2026-01-06T00:00:00+00:00",
            "trust_class": "production_ready",
            "artifact_type": "backtest_metrics",
            "run_id": "fixture_run",
            "horizon": "short_5d",
            "producer": "run-strict-a-share-backtest-v8",
            "quality_status": "passed",
            "row_count": 1,
            "date_range": {"start": "2026-01-02", "end": "2026-01-05"},
            "upstream_paths": ["runtime/reports/v8/deep/fixture_run/short_5d/backtest/nav.csv"],
            "output": str(metrics_path),
            "output_sha256": sha256(metrics_path.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    (backtest / "nav.csv").write_text(
        "trade_date,nav\n2026-01-02,1000000\n2026-01-05,1010000\n",
        encoding="utf-8",
    )
    (backtest / "pnl.csv").write_text(
        "trade_date,daily_return,nav\n2026-01-02,0,1000000\n2026-01-05,0.01,1010000\n",
        encoding="utf-8",
    )
    (backtest / "trades.csv").write_text(
        "trade_date,client_order_id,status,filled_quantity,avg_price,last_message,symbol,side,quantity,reference_price\n"
        "2026-01-02,buy-1,filled,100,10.08,filled,000001.SZ,buy,100,10.00\n"
        "2026-01-05,sell-1,filled,100,10.92,filled,000001.SZ,sell,100,11.00\n",
        encoding="utf-8",
    )
    (backtest / "realized_trades.csv").write_text(
        "symbol,buy_date,sell_date,quantity,buy_price,sell_price,gross_pnl,cost,net_pnl\n"
        "000001.SZ,2026-01-02,2026-01-05,100,10.08,10.92,84,8,76\n",
        encoding="utf-8",
    )
    (backtest / "failed_orders.csv").write_text(
        "trade_date,client_order_id,status,filled_quantity,avg_price,last_message,symbol,side,quantity,reference_price\n",
        encoding="utf-8",
    )
    (backtest / "selected_stocks.csv").write_text(
        "symbol,first_filled,last_filled,n_fills\n000001.SZ,2026-01-02,2026-01-05,2\n",
        encoding="utf-8",
    )
    (backtest / "profit_by_stock.csv").write_text(
        "symbol,n_trades,n_fills,quantity,gross_pnl,cost,net_pnl,win_rate,avg_profit_per_trade,pnl_proxy\n"
        "000001.SZ,1,2,100,84,8,76,1,76,76\n",
        encoding="utf-8",
    )
    (backtest / "risk_events.json").write_text(
        json.dumps([{
            "trade_date": "2026-01-05",
            "event_type": "order_rejected",
            "symbol": "000002.SZ",
            "reason": "limit_up_no_buy",
        }]),
        encoding="utf-8",
    )
    (backtest / "factor_weights.json").write_text("{}", encoding="utf-8")
    run_root = backtest.parent
    (run_root / "run_config.json").write_text(
        json.dumps({
            "train_start": "2024-01-01",
            "train_end": "2025-12-31",
            "test_end": "2026-01-05",
            "feature_policy": "fixture",
            "initial_cash": 1_000_000,
        }),
        encoding="utf-8",
    )

    market = runtime / "data" / "v7" / "silver" / "market_panel"
    market.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "trade_date": [datetime(2026, 1, 2), datetime(2026, 1, 5)],
        "open": [10.0, 10.8],
        "high": [10.5, 11.2],
        "low": [9.8, 10.7],
        "close": [10.4, 11.0],
        "volume": [1_000_000.0, 1_200_000.0],
        "amount": [10_400_000.0, 13_200_000.0],
        "available_at": [datetime(2026, 1, 5), datetime(2026, 1, 6)],
        "source": ["fixture", "fixture"],
        "is_st": [False, False],
        "is_suspended": [False, False],
        "is_limit_up": [False, False],
        "is_limit_down": [False, False],
    }).write_parquet(market / "market_panel.parquet")
    code_map = runtime / "data" / "v7" / "silver"
    pl.DataFrame({"code": ["000001"], "name": ["平安银行"]}).write_parquet(code_map / "code_name_map.parquet")

    model = runtime / "reports" / "v8" / "deep" / "fixture_run" / "short_5d" / "ft"
    model.mkdir(parents=True)
    (model / "ft_transformer.pt").write_bytes(b"metadata-only")
    (model / "ft_transformer_config.json").write_text(
        json.dumps({"device": "cpu", "feature_columns": ["alpha001"], "horizons": [5]}),
        encoding="utf-8",
    )
    (model / "ft_transformer_feature_schema.json").write_text(
        json.dumps({"architecture": "ft_transformer", "feature_columns": ["alpha001"], "horizons": [5], "version": "v7"}),
        encoding="utf-8",
    )
    (model / "ft_transformer_metrics.json").write_text(
        json.dumps({"device": "cpu", "training_history": [{"epoch": 0, "loss": 0.3, "val_loss": 0.2}]}),
        encoding="utf-8",
    )
    pl.DataFrame({
        "trade_date": [datetime(2026, 1, 2), datetime(2026, 1, 5)],
        "symbol": ["000001.SZ", "000001.SZ"],
        "alpha_score": [0.5, 0.7],
    }).write_parquet(run_root / "predictions.parquet")

    rl_model = runtime / "models" / "rl_fixture"
    rl_policy = rl_model / "policy"
    rl_policy.mkdir(parents=True)
    (rl_policy / "policy.zip").write_bytes(b"metadata-only")
    (rl_model / "training_summary.json").write_text(
        json.dumps({
            "status": "trained",
            "timesteps": 1000,
            "device": "cpu",
            "config": {"env": {"transaction_cost_bps": 10}},
        }),
        encoding="utf-8",
    )
    (rl_model / "verdict.json").write_text(
        json.dumps({"verdict": "PAPER_ONLY", "annualized_return": 0.08, "max_drawdown": 0.04}),
        encoding="utf-8",
    )
    pl.DataFrame({
        "trade_date": [datetime(2026, 1, 5)],
        "000001.SZ": [0.05],
    }).write_parquet(rl_model / "weights_test.parquet")

    t_plus_one_model = runtime / "reports" / "t_plus_one_fixture"
    t_plus_one_model.mkdir(parents=True)
    (t_plus_one_model / "do_t_models.joblib").write_bytes(b"metadata-only")
    (t_plus_one_model / "ev_backtest_report.json").write_text(
        json.dumps({
            "verdict": "DO_NOT_ENABLE",
            "n_train_rows": 100,
            "metrics": {"hit_rate": 0.45, "excess_return_after_costs": -0.01},
        }),
        encoding="utf-8",
    )

    selection = runtime / "reports" / "v8" / "llm_hybrid_fixture"
    selection.mkdir(parents=True)
    (selection / "summary.json").write_text(
        json.dumps({
            "as_of_date": "2026-01-05",
            "candidate_rows": 2,
            "final_stock_rows": 1,
            "used_fallback": False,
            "position_hint": {"no_orders_generated": True},
        }),
        encoding="utf-8",
    )
    pl.DataFrame({
        "trade_date": [datetime(2026, 1, 5)],
        "symbol": ["000001.SZ"],
        "prediction": [0.7],
        "model_rank": [1],
        "factor_prior_score": [0.8],
        "old_dealer_risk_score": [0.2],
        "do_t_suitability_score": [0.6],
        "llm_stock_score": [0.75],
        "llm_confidence": [0.8],
        "hybrid_score": [0.77],
        "hybrid_rank": [1],
        "action_bucket": ["research_buy"],
        "sector_level_1": ["银行"],
        "core_policy_score": [0.5],
        "no_orders_generated": [True],
    }).write_parquet(selection / "hybrid_stock_pool.parquet")

    do_t = runtime / "reports" / "intraday_dot_factor_combo_fixture"
    do_t.mkdir(parents=True)
    (do_t / "factor_combo_report.json").write_text(
        json.dumps({"verdict": "PAPER_ONLY", "metrics": {"hit_rate": 1.0}}),
        encoding="utf-8",
    )
    pl.DataFrame({
        "symbol": ["000001.SZ"],
        "trade_date": [datetime(2026, 1, 5)],
        "mode": ["dip_buy"],
        "state": ["closed_profit"],
        "gross_ret": [0.01],
        "net_ret": [0.008],
        "entry_px": [10.5],
        "exit_px": [10.7],
        "requested_qty": [100],
        "filled_qty": [100],
        "entry_fill_time": ["2026-01-05T10:00:00"],
        "exit_fill_time": ["2026-01-05T13:30:00"],
        "entry_fill_status": ["filled"],
        "exit_fill_status": ["filled"],
    }).write_parquet(do_t / "factor_combo_scored.parquet")

    factor_report = runtime / "reports" / "v8" / "factor_full_judgment"
    factor_report.mkdir(parents=True)
    (factor_report / "factor_judgment_table.csv").write_text(
        "factor,family,source,ic_5d,icir_5d,ic_20d,icir_20d,ic_60d,icir_60d,best_horizon,ic_2022,ic_2023,ic_2024,ic_2025,ic_2026,ic_bull,ic_sideways,ic_bear,capacity_ratio,verdict,years_passed,regimes_passed\n"
        "alpha001,alpha101_worldquant,alpha101,0.02,0.4,0.03,0.5,0.04,0.6,60d,0.01,0.02,0.03,0.04,0.05,0.03,0.02,0.01,0.8,all_weather,4,3\n",
        encoding="utf-8",
    )

    ignored_cache = runtime / "reports" / "intraday_dot_factor_combo_fixture" / "feature_cache"
    ignored_cache.mkdir()
    (ignored_cache / "cached.parquet").write_bytes(b"cache")

    logs = runtime / "logs"
    logs.mkdir(parents=True)
    (logs / "quant_ui_fixture.log").write_text("first line\nlast line\n", encoding="utf-8")
