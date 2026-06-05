from __future__ import annotations

import json

from quantagent.diagnostics.v8_result_table import collect_v8_result_rows


def test_collect_v8_result_rows_normalises_headline_and_metrics(tmp_path):
    root = tmp_path / "v8"
    run = root / "bull" / "candidate_a"
    bt = run / "backtest"
    bt.mkdir(parents=True)
    (run / "headline_report.json").write_text(json.dumps({
        "horizon": "mid_5d_30d",
        "strategy_ann": 0.20,
        "benchmark_equal_weight_ann": 0.15,
        "excess_return_ann": 0.05,
    }), encoding="utf-8")
    (bt / "metrics.json").write_text(json.dumps({
        "total_return": 0.30,
        "annualized_return": 0.20,
        "volatility": 0.18,
        "sharpe": 1.1,
        "max_drawdown": 0.12,
        "calmar": 1.67,
        "turnover": 0.03,
    }), encoding="utf-8")

    table = collect_v8_result_rows([root])

    assert len(table) == 1
    row = table.iloc[0]
    assert row["market_env"] == "bull"
    assert row["total_return"] == 0.30
    assert row["annualized_return"] == 0.20
    assert row["annualized_volatility"] == 0.18
    assert row["cost_after_return"] == 0.20
    assert row["excess_equal_weight_return"] == 0.05
