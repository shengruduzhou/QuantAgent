#!/usr/bin/env python3
"""PIT-RL walk-forward: train PPO on the hold-band book env, judge strictly.

Pipeline:
  1. hold-band book (enter30/exit150/hold50, t+1) over the full v8.8 window,
     warmed up from 2024-08 — the SAME deterministic book the paper account
     uses (PIT by construction).
  2. PPO trains ONLY on book dates ≤ --train-end (value-add reward, flags
     constraints inside the env).
  3. Deterministic rollout on the untouched 2026 test window exports daily
     weights; BOTH the policy book and the passive hold-band book are
     re-simulated through run_strict_backtest_v8.
  4. Verdict: ENABLE only if the policy's strict annualized beats the
     passive book's on the same window (this was the bar the old top-80
     env-policy failed — it underperformed its own universe).

Outputs runtime/models/v88_rl_pit/{policy.zip, verdict.json, weights_test.parquet}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
OUT = Path("runtime/models/v88_rl_pit")
ANN = 244


def _strict(tw: pd.DataFrame, panel: pd.DataFrame, sector: pd.DataFrame,
            slippage_bps: float) -> dict:
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    res = run_strict_backtest_v8(
        tw, panel, sector_map=sector,
        config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0,
                                               slippage_bps=slippage_bps))
    m = res.metrics
    return {"ann": round(m.annualized_return, 4), "total": round(m.total_return, 4),
            "sharpe": round(m.sharpe, 3), "maxDD": round(m.max_drawdown, 4)}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=PREDS)
    ap.add_argument("--score-column", default="composite_score")
    ap.add_argument("--warmup-start", default="2024-08-09")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--test-start", default="2026-01-02")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--cost-bps", type=float, default=12.0)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--skip-train", action="store_true", help="reuse existing policy.zip")
    ap.add_argument("--random-baselines", type=int, default=5,
                    help="N untrained-policy rollouts forming the strict-sim null distribution")
    ap.add_argument("--output-dir", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.output_dir)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights
    from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    if args.score_column != "alpha_score":
        preds = preds.rename(columns={args.score_column: "alpha_score"})

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)]
    sector = pd.read_parquet(SECTOR)

    flags = panel[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    preds_f = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel["trade_date"].unique())
    book = build_hold_band_weights(
        preds_f, config=HoldBandConfig(n_hold=50, entry_rank=30, exit_rank=150, delay_days=1),
        trade_dates=trade_dates)
    print(f"book: {book.index.min().date()}..{book.index.max().date()} "
          f"({len(book)} dates, max {int((book > 0).sum(axis=1).max())} names)", flush=True)

    train_book = book[book.index <= pd.Timestamp(args.train_end)]
    test_book = book[book.index >= pd.Timestamp(args.test_start)]
    env_cfg = PITPortfolioEnvConfig(max_book=60, cost_bps=args.cost_bps)

    policy_path = OUT / "policy.zip"
    if not args.skip_train or not policy_path.exists():
        print(f"=== training PPO ({args.timesteps} steps, {args.n_envs} envs) ===", flush=True)

        def make_env(rank: int):
            def _factory():
                env = PITPortfolioEnv(train_book, preds, panel, env_cfg)
                env.reset(seed=args.seed + rank)
                return env
            return _factory

        vec = SubprocVecEnv([make_env(i) for i in range(args.n_envs)]) if args.n_envs > 1 \
            else DummyVecEnv([make_env(0)])
        model = PPO("MlpPolicy", vec, device=args.device, seed=args.seed,
                    tensorboard_log=str(OUT / "tb"))
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(policy_path)
        vec.close()
    else:
        print("=== reusing existing policy.zip ===", flush=True)

    print("=== deterministic test rollout ===", flush=True)
    model = PPO.load(policy_path, device="cpu")
    env = PITPortfolioEnv(test_book, preds, panel, env_cfg)
    # Env-flat guard: if within-book alpha dispersion is ~0 the policy cannot
    # express stock selection, so a positive value-add would be a gross/cash
    # bet, not alpha. Surfaced in the verdict and required for ENABLE.
    dispersion = env.book_dispersion_report()
    print(f"env dispersion: {json.dumps(dispersion)}", flush=True)
    obs, _ = env.reset(seed=7)
    rows: dict[pd.Timestamp, pd.Series] = {}
    value_add = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        w = pd.Series(info["weights"])
        rows[pd.Timestamp(info["trade_date"])] = w[w > 1e-6]
        value_add.append(info["value_add"])
    tw_policy = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    tw_policy.index.name = "trade_date"
    tw_policy.to_parquet(OUT / "weights_test.parquet")
    tw_passive = test_book.loc[test_book.index.isin(tw_policy.index)]

    print("=== strict simulation: policy vs passive book ===", flush=True)
    sim_panel = panel[panel["trade_date"] >= pd.Timestamp(args.test_start) - pd.Timedelta(days=5)]
    strict_policy = _strict(tw_policy, sim_panel, sector, args.slippage_bps)
    strict_passive = _strict(tw_passive, sim_panel, sector, args.slippage_bps)

    # Null distribution: untrained policies through the SAME strict path.
    # A concentrated tilted book through the throttled simulator has high
    # variance on a 4-month window — beating passive alone is not evidence.
    null_anns = []
    for k in range(args.random_baselines):
        rnd = PPO("MlpPolicy", PITPortfolioEnv(test_book, preds, panel, env_cfg),
                  device="cpu", seed=1000 + k)
        env_k = PITPortfolioEnv(test_book, preds, panel, env_cfg)
        obs_k, _ = env_k.reset(seed=1000 + k)
        rows_k: dict[pd.Timestamp, pd.Series] = {}
        done_k = False
        while not done_k:
            act, _ = rnd.predict(obs_k, deterministic=True)
            obs_k, _, done_k, _, info_k = env_k.step(act)
            wk = pd.Series(info_k["weights"])
            rows_k[pd.Timestamp(info_k["trade_date"])] = wk[wk > 1e-6]
        tw_k = pd.DataFrame(rows_k).T.fillna(0.0).sort_index()
        null_anns.append(_strict(tw_k, sim_panel, sector, args.slippage_bps)["ann"])
        print(f"  null #{k}: ann {null_anns[-1]:+.2%}", flush=True)

    env_va = float(np.sum(value_add))
    beats_null = strict_policy["ann"] > max(null_anns) if null_anns else True
    verdict = {
        "verdict": ("ENABLE" if strict_policy["ann"] > strict_passive["ann"]
                    and strict_policy["maxDD"] <= strict_passive["maxDD"] + 0.05
                    and env_va > 0 and beats_null
                    and dispersion["env_can_select"]
                    else "DO_NOT_ENABLE"),
        "env_dispersion": dispersion,
        "selection_driven": bool(dispersion["env_can_select"]),
        "null_strict_anns_untrained": [round(a, 4) for a in null_anns],
        "beats_all_null": bool(beats_null),
        "window": f"{tw_policy.index.min().date()}..{tw_policy.index.max().date()}",
        "train_end": args.train_end,
        "strict_policy": strict_policy,
        "strict_passive_holdband": strict_passive,
        "ann_value_add_strict": round(strict_policy["ann"] - strict_passive["ann"], 4),
        "env_cum_value_add": round(env_va, 4),
        "mean_daily_turnover_policy": round(float(tw_policy.diff().abs().sum(axis=1).mean() / 2), 4),
        "timesteps": args.timesteps,
        "note": "universe = PIT hold-band book; reward = value-add vs constrained passive book",
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
