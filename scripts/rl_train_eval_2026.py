#!/usr/bin/env python3
"""PPO portfolio-overlay: train on 2024-08..2025-12, evaluate on 2026 YTD.

RL's mandate (project rule): adjust WEIGHTS/exposure inside the
factor-selected universe — never pick stocks, never emit real orders.
The policy is only ENABLED if its 2026 out-of-sample excess beats the
deterministic hold-band paper account on the same signal (v8.8 ensemble).

Outputs runtime/models/v88_rl_overlay/{policy.zip, training_summary.json,
eval_2026.json}.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
OUT = Path("runtime/models/v88_rl_overlay")
ANN = 244


def main() -> int:
    from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig
    from quantagent.rl.train_ppo import PPOTrainingConfig, train_ppo_policy

    preds = pd.read_parquet(PREDS).rename(columns={"composite_score": "alpha_score"})
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close",
                                            "volume", "amount", "is_suspended", "is_st", "is_limit_up",
                                            "is_limit_down"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    train_preds = preds[(preds["trade_date"] >= "2024-08-28") & (preds["trade_date"] <= "2025-12-31")]
    eval_preds = preds[preds["trade_date"] >= "2025-12-01"]  # warm context + 2026 eval
    train_panel = panel[(panel["trade_date"] >= "2024-08-20") & (panel["trade_date"] <= "2026-01-10")]
    eval_panel = panel[panel["trade_date"] >= "2025-11-20"]

    env_cfg = PortfolioEnvConfig(top_n=80, max_turnover=0.30, cost_bps=12.0)
    cfg = PPOTrainingConfig(
        timesteps=1_000_000,
        n_envs=4,
        device="cuda",
        output_dir=str(OUT),
        env=env_cfg,
        seed=1729,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary = train_ppo_policy(train_preds, train_panel, cfg)
    print(json.dumps({k: summary[k] for k in ("status", "policy_path", "timesteps", "gpu_name")},
                     ensure_ascii=False))

    # ---- deterministic 2026 evaluation -------------------------------------
    from stable_baselines3 import PPO

    model = PPO.load(summary["policy_path"], device="cuda")
    env = PortfolioEnv(eval_preds, eval_panel, env_cfg)
    obs, _ = env.reset(seed=7)
    navs, dates_used = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        navs.append(float(info.get("nav", np.nan)))
        if "trade_date" in info:
            dates_used.append(info["trade_date"])
    nav = pd.Series(navs).dropna()
    if len(nav) < 10:
        raise SystemExit("eval episode too short")
    rets = nav.pct_change().dropna()
    n = len(rets)
    ann = float((1 + rets).prod() ** (ANN / n) - 1)
    sharpe = float(rets.mean() / (rets.std(ddof=0) + 1e-12) * np.sqrt(ANN))
    peak = nav.cummax()
    maxdd = float(((peak - nav) / peak).max())

    eval_out = {
        "window_days": int(n),
        "annualized": round(ann, 4),
        "sharpe": round(sharpe, 3),
        "maxDD": round(maxdd, 4),
        "note": "fixed top-80 universe daily-delta policy; compare vs hold-band paper replay "
                "(runtime/paper/replay_2026/summary.json) before enabling",
    }
    (OUT / "eval_2026.json").write_text(json.dumps(eval_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(eval_out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
