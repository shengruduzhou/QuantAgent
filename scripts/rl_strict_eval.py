#!/usr/bin/env python3
"""Honest RL verdict: roll the PPO policy, export its DAILY TARGET WEIGHTS,
and run them through the SAME strict A-share simulator as every other sleeve.

The PortfolioEnv's own NAV (+275%/yr, sharpe 5.2 on 2026) is an ANALYTIC
artifact: frictionless close-to-close fills, no tradability flags, and a
top-80 universe selected from the evaluation window itself. Project rule:
no sleeve is judged by its own simulator — only by baseline_protocol-grade
strict accounting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
POLICY = "runtime/models/v88_rl_overlay/policy.zip"
OUT = Path("runtime/models/v88_rl_overlay")
ANN = 244


def main() -> int:
    from stable_baselines3 import PPO

    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8
    from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig

    preds = pd.read_parquet(PREDS).rename(columns={"composite_score": "alpha_score"})
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    eval_preds = preds[preds["trade_date"] >= "2025-12-01"]
    eval_panel = panel[panel["trade_date"] >= "2025-11-20"]
    env_cfg = PortfolioEnvConfig(top_n=80, max_turnover=0.30, cost_bps=12.0)
    env = PortfolioEnv(eval_preds, eval_panel, env_cfg)
    model = PPO.load(POLICY, device="cpu")

    obs, _ = env.reset(seed=7)
    rows: dict[pd.Timestamp, pd.Series] = {}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        w = pd.Series(env._weights.copy(), index=env.symbols)
        w = w[w.abs() > 1e-6]
        if len(w):
            rows[pd.Timestamp(info["trade_date"])] = w
    tw = pd.DataFrame(rows).T.fillna(0.0)
    tw.index.name = "trade_date"
    tw = tw[tw.index >= "2026-01-02"].sort_index()
    if tw.empty:
        raise SystemExit("policy produced no 2026 weights")

    sim_panel = panel[panel["trade_date"] >= pd.Timestamp("2026-01-02") - pd.Timedelta(days=5)]
    res = run_strict_backtest_v8(tw, sim_panel,
                                 config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0,
                                                                        slippage_bps=8.0))
    m = res.metrics
    px = sim_panel[sim_panel["trade_date"].isin(tw.index)].pivot_table(
        index="trade_date", columns="symbol", values="close")
    bench = px.pct_change(fill_method=None).mean(axis=1).dropna()
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)

    out = {
        "window": f"{tw.index.min().date()}..{tw.index.max().date()}",
        "strict_annualized": round(m.annualized_return, 4),
        "strict_total": round(m.total_return, 4),
        "strict_sharpe": round(m.sharpe, 3),
        "strict_maxDD": round(m.max_drawdown, 4),
        "excess_vs_paper_bench_ann": round(m.annualized_return - bench_ann, 4),
        "bench_ann": round(bench_ann, 4),
        "mean_gross": round(float(tw.sum(axis=1).mean()), 3),
        "env_analytic_claim": "ann +275%, sharpe 5.16 — REJECTED as evaluation basis (frictionless env)",
    }
    (OUT / "strict_eval_2026.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
