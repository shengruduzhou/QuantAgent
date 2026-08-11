#!/usr/bin/env python3
"""Governed PIT-RL walk-forward research: train on a signal-dated book, judge strictly.

Pipeline
--------
1. Build the deterministic hold-band book on **signal dates** (delay_days=0).
   The book builder does not own execution timing.
2. PPO receives close(T) information and is rewarded only on the canonical
   executable transition close(T+1)->close(T+2), using an explicit independent
   global market-session calendar.
3. Export deterministic test-window signal-dated weights.  Both policy and
   passive hold-band books are re-simulated through ``run_strict_backtest_v8``;
   that simulator is the sole owner of T-close -> next-session execution.
4. Compare against the passive book and untrained-policy strict-sim nulls.

A pass is only an incremental *research screen*.  This command never grants
production eligibility or order authority; Stage-4 lineage, an independent
one-shot holdout and the production risk/OMS chain remain mandatory.
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
MARKET_CALENDAR = "runtime/data/u0/pit/trading_calendar.parquet"
OUT = Path("runtime/models/v88_rl_pit")


def _read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(target)
    if target.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(target)
    raise ValueError(f"unsupported table format: {target}")


def _market_sessions(path: str | Path) -> pd.DatetimeIndex:
    from quantagent.factors.executable_labels import canonical_market_sessions

    frame = _read_table(path)
    if frame.empty:
        raise ValueError("market calendar is empty")
    date_column = next(
        (name for name in ("trade_date", "calendar_date", "date") if name in frame.columns),
        None,
    )
    if date_column is None:
        if len(frame.columns) != 1:
            raise ValueError("market calendar requires trade_date/calendar_date/date")
        date_column = str(frame.columns[0])
    work = frame.copy()
    if "is_trading_day" in work.columns:
        open_flag = work["is_trading_day"].astype(str).str.strip().str.lower()
        work = work[open_flag.isin({"1", "true", "yes"})]
    return canonical_market_sessions(work[date_column].tolist())


def _strict(
    target_weights: pd.DataFrame,
    panel: pd.DataFrame,
    sector: pd.DataFrame,
    slippage_bps: float,
) -> dict[str, object]:
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    result = run_strict_backtest_v8(
        target_weights,
        panel,
        sector_map=sector,
        config=AShareExecutionSimulationConfig(
            initial_cash=1_000_000.0,
            slippage_bps=slippage_bps,
        ),
    )
    metrics = result.metrics
    return {
        "annualized_return": round(metrics.annualized_return, 4),
        "total_return": round(metrics.total_return, 4),
        "sharpe": round(metrics.sharpe, 3),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
    }


def _strict_ann(report: dict[str, object]) -> float:
    return float(report["annualized_return"])


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=PREDS)
    parser.add_argument("--score-column", default="composite_score")
    parser.add_argument("--market-calendar", default=MARKET_CALENDAR)
    parser.add_argument("--warmup-start", default="2024-08-09")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-02")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing governed policy.zip")
    parser.add_argument(
        "--random-baselines",
        type=int,
        default=5,
        help="untrained policies forming a strict-simulator null distribution",
    )
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output_dir)

    from stable_baselines3 import PPO

    from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights
    from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig
    from quantagent.rl.train_ppo import PPOTrainingConfig, train_ppo_policy

    OUT.mkdir(parents=True, exist_ok=True)
    sessions = _market_sessions(args.market_calendar)

    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce").dt.normalize()
    if args.score_column != "alpha_score":
        if args.score_column not in preds.columns:
            raise ValueError(f"prediction score column not found: {args.score_column}")
        preds = preds.rename(columns={args.score_column: "alpha_score"})

    panel_cols = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "available_at",
        "is_suspended",
        "is_st",
        "is_limit_up",
        "is_limit_down",
    ]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel = panel[
        panel["trade_date"] >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)
    ]
    sector = pd.read_parquet(SECTOR)

    flags = panel[
        ["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]
    ]
    preds_f = preds.merge(flags, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    critical = ["is_st", "is_suspended", "is_limit_up"]
    unknown = preds_f[critical].isna().any(axis=1)
    if bool(unknown.any()):
        examples = preds_f.loc[unknown, ["trade_date", "symbol"]].head(5).to_dict("records")
        raise ValueError(
            "hold-band eligibility requires known PIT execution flags; "
            f"examples={examples}"
        )

    # IMPORTANT: target rows are signal-dated.  strict_v8 owns the one and only
    # T-close -> T+1 mapping, so delay_days=1 here would double-delay execution.
    book = build_hold_band_weights(
        preds_f,
        config=HoldBandConfig(
            n_hold=50,
            entry_rank=30,
            exit_rank=150,
            delay_days=0,
        ),
        trade_dates=list(sessions),
    )
    if book.empty:
        raise ValueError("hold-band produced no signal-dated book")
    print(
        f"signal book: {book.index.min().date()}..{book.index.max().date()} "
        f"({len(book)} dates, max {int((book > 0).sum(axis=1).max())} names)",
        flush=True,
    )

    train_book = book[book.index <= pd.Timestamp(args.train_end)]
    test_book = book[book.index >= pd.Timestamp(args.test_start)]
    env_cfg = PITPortfolioEnvConfig(max_book=60, cost_bps=args.cost_bps)
    if len(train_book) < 3 or len(test_book) < 3:
        raise ValueError("RL train/test signal windows are too short")

    policy_path = OUT / "policy.zip"
    if not args.skip_train or not policy_path.exists():
        print(f"=== governed PPO training ({args.timesteps} steps) ===", flush=True)
        train_summary = train_ppo_policy(
            preds,
            panel,
            PPOTrainingConfig(
                timesteps=args.timesteps,
                n_envs=args.n_envs,
                device=args.device,
                output_dir=str(OUT),
                tensorboard_log=str(OUT / "tb"),
                env=env_cfg,
                seed=args.seed,
                require_gpu=args.device.startswith("cuda"),
            ),
            book_weights=train_book,
            market_sessions=sessions,
        )
        policy_path = Path(str(train_summary["policy_path"]))
    else:
        summary_path = OUT / "training_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                "--skip-train requires training_summary.json proving the governed reward contract"
            )
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        from quantagent.rl.pit_portfolio_env import RL_REWARD_SEMANTICS
        from quantagent.factors.executable_labels import market_session_schedule_sha256

        if prior.get("reward_semantics") != RL_REWARD_SEMANTICS:
            raise RuntimeError("existing policy was trained under a stale/unknown RL reward clock")
        if prior.get("market_session_schedule_sha256") != market_session_schedule_sha256(sessions):
            raise RuntimeError("existing policy market-calendar lineage does not match this run")
        print("=== reusing governed policy.zip with matching clock lineage ===", flush=True)

    print("=== deterministic untouched-window rollout ===", flush=True)
    model = PPO.load(policy_path, device="cpu")
    env = PITPortfolioEnv(test_book, preds, panel, sessions, env_cfg)
    dispersion = env.book_dispersion_report()
    obs, _ = env.reset(seed=7)
    rows: dict[pd.Timestamp, pd.Series] = {}
    value_add: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        weights = pd.Series(info["weights"])
        rows[pd.Timestamp(info["signal_date"])] = weights[weights > 1e-6]
        value_add.append(float(info["value_add"]))
    tw_policy = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    tw_policy.index.name = "trade_date"
    tw_policy.to_parquet(OUT / "weights_test.parquet")
    tw_passive = test_book.loc[test_book.index.isin(tw_policy.index)]

    print("=== strict signal->T+1 simulation: policy vs passive ===", flush=True)
    sim_panel = panel[
        panel["trade_date"] >= pd.Timestamp(args.test_start) - pd.Timedelta(days=5)
    ]
    strict_policy = _strict(tw_policy, sim_panel, sector, args.slippage_bps)
    strict_passive = _strict(tw_passive, sim_panel, sector, args.slippage_bps)

    null_anns: list[float] = []
    for k in range(args.random_baselines):
        env_k = PITPortfolioEnv(test_book, preds, panel, sessions, env_cfg)
        random_policy = PPO("MlpPolicy", env_k, device="cpu", seed=1000 + k)
        obs_k, _ = env_k.reset(seed=1000 + k)
        rows_k: dict[pd.Timestamp, pd.Series] = {}
        done_k = False
        while not done_k:
            action_k, _ = random_policy.predict(obs_k, deterministic=True)
            obs_k, _, done_k, _, info_k = env_k.step(action_k)
            weights_k = pd.Series(info_k["weights"])
            rows_k[pd.Timestamp(info_k["signal_date"])] = weights_k[weights_k > 1e-6]
        targets_k = pd.DataFrame(rows_k).T.fillna(0.0).sort_index()
        null_report = _strict(targets_k, sim_panel, sector, args.slippage_bps)
        null_anns.append(_strict_ann(null_report))
        print(f"  null #{k}: ann {null_anns[-1]:+.2%}", flush=True)

    env_value_add = float(np.sum(value_add))
    policy_ann = _strict_ann(strict_policy)
    passive_ann = _strict_ann(strict_passive)
    beats_null = policy_ann > max(null_anns) if null_anns else True
    research_pass = bool(
        policy_ann > passive_ann
        and float(strict_policy["max_drawdown"])
        <= float(strict_passive["max_drawdown"]) + 0.05
        and env_value_add > 0
        and beats_null
        and dispersion["env_can_select"]
    )
    verdict = {
        "schema": "quantagent.rl.incremental_research_screen.v2",
        "verdict": "RESEARCH_PASS" if research_pass else "RESEARCH_REJECT",
        "researchPromotionEligible": research_pass,
        "productionEligible": False,
        "researchOnly": True,
        "reward_semantics": dispersion["reward_semantics"],
        "market_session_schedule_sha256": dispersion["market_session_schedule_sha256"],
        "target_index_semantics": "signal_date",
        "strict_execution_owner": "run_strict_backtest_v8",
        "env_dispersion": dispersion,
        "selection_driven": bool(dispersion["env_can_select"]),
        "null_strict_annualized_untrained": [round(value, 4) for value in null_anns],
        "beats_all_null": bool(beats_null),
        "window": f"{tw_policy.index.min().date()}..{tw_policy.index.max().date()}",
        "train_end": args.train_end,
        "strict_policy": strict_policy,
        "strict_passive_holdband": strict_passive,
        "annualized_value_add_strict": round(policy_ann - passive_ann, 4),
        "env_cumulative_value_add": round(env_value_add, 4),
        "mean_daily_target_turnover_policy": round(
            float(tw_policy.diff().abs().sum(axis=1).mean() / 2), 4
        ),
        "timesteps": args.timesteps,
        "productionBlockers": [
            "this is a research comparison, not a Stage-4 production certificate",
            "independent one-shot FinalHoldoutLedger evidence remains required",
            "PBO/DSR/SPA and cumulative trial accounting remain required for promotion",
            "paper/shadow broker soak and hard risk/OMS evidence remain required",
        ],
    }
    (OUT / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
