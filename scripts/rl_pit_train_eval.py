#!/usr/bin/env python3
"""PIT-RL walk-forward research with executable-clock and policy contracts.

Clock contract:
  * hold-band rows are SIGNAL-DATED (delay_days=0);
  * strict A-share execution owns signal T close -> execution T+1 close;
  * PITPortfolioEnv reward begins only after that execution:
    close(T+1) -> close(T+2);
  * execution-session tradability evidence is mandatory and fail-closed.

A policy is never production-enabled by this script. A research candidate must
beat the passive hold-band book and untrained null policies in strict replay,
show genuine within-book selection capacity, and then continue through the
repository's independent statistical/risk/shadow gates.
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
POLICY_CONTRACT_SCHEMA = "quantagent.rl.policy_contract.v1"


def _strict(
    target_weights: pd.DataFrame,
    panel: pd.DataFrame,
    sector: pd.DataFrame,
    slippage_bps: float,
) -> dict:
    from quantagent.backtest.ashare_execution_simulator import (
        AShareExecutionSimulationConfig,
    )
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8
    from quantagent.portfolio.hold_band import SIGNAL_DATE_SEMANTICS

    semantics = target_weights.attrs.get("target_index_semantics")
    if semantics != SIGNAL_DATE_SEMANTICS:
        raise RuntimeError(
            "RL strict evaluation requires explicit signal-dated targets; "
            f"got target_index_semantics={semantics!r}"
        )
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
        "ann": round(metrics.annualized_return, 4),
        "total": round(metrics.total_return, 4),
        "sharpe": round(metrics.sharpe, 3),
        "maxDD": round(metrics.max_drawdown, 4),
    }


def _mark_signal_dated(frame: pd.DataFrame) -> pd.DataFrame:
    from quantagent.portfolio.hold_band import SIGNAL_DATE_SEMANTICS

    frame.attrs["target_index_semantics"] = SIGNAL_DATE_SEMANTICS
    frame.attrs["execution_mapping_owner"] = "strict_ashare_simulator"
    return frame


def _policy_contract(
    *,
    reward_clock_semantics: str,
    target_index_semantics: str,
    max_book: int,
    train_end: str,
    test_start: str,
    score_column: str,
    cost_bps: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": POLICY_CONTRACT_SCHEMA,
        "reward_clock_semantics": reward_clock_semantics,
        "target_index_semantics": target_index_semantics,
        "execution_mapping_owner": "strict_ashare_simulator",
        "max_book": int(max_book),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "score_column": str(score_column),
        "cost_bps": float(cost_bps),
        "seed": int(seed),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_matching_policy_contract(
    path: Path,
    expected: dict[str, object],
) -> None:
    if not path.is_file():
        raise RuntimeError(
            "existing RL policy has no policy_contract.json; stale pre-clock-audit "
            "policies cannot be reused. Retrain without --skip-train."
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"cannot read RL policy contract {path}: {exc}") from exc
    if actual != expected:
        raise RuntimeError(
            "RL policy contract does not match the current executable reward clock "
            "or training configuration; retrain instead of reusing stale weights. "
            f"expected={expected}, actual={actual}"
        )


def main() -> int:
    global OUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=PREDS)
    parser.add_argument("--score-column", default="composite_score")
    parser.add_argument("--warmup-start", default="2024-08-09")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-02")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument(
        "--max-book",
        type=int,
        default=80,
        help=(
            "RL transition slots. Must fit current targets plus carried exits; "
            "the env fails closed instead of truncating when this is too small."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse policy.zip only when policy_contract.json matches exactly",
    )
    parser.add_argument(
        "--random-baselines",
        type=int,
        default=5,
        help="N untrained-policy rollouts forming the strict-sim null distribution",
    )
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output_dir)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from quantagent.portfolio.hold_band import (
        SIGNAL_DATE_SEMANTICS,
        HoldBandConfig,
        build_hold_band_weights,
    )
    from quantagent.rl.pit_portfolio_env import (
        RL_REWARD_CLOCK_SEMANTICS,
        PITPortfolioEnv,
        PITPortfolioEnvConfig,
    )

    if args.max_book < 1:
        raise ValueError("--max-book must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(args.predictions)
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    if args.score_column != "alpha_score":
        predictions = predictions.rename(
            columns={args.score_column: "alpha_score"}
        )

    panel_columns = [
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
    panel = pd.read_parquet(PANEL, columns=panel_columns)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[
        panel["trade_date"]
        >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)
    ]
    sector = pd.read_parquet(SECTOR)

    flags = panel[
        ["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]
    ]
    predictions_with_flags = predictions.merge(
        flags, on=["symbol", "trade_date"], how="left"
    )
    trade_dates = sorted(panel["trade_date"].unique())

    # The strict simulator owns the only signal -> execution delay.
    book = build_hold_band_weights(
        predictions_with_flags,
        config=HoldBandConfig(
            n_hold=50,
            entry_rank=30,
            exit_rank=150,
            delay_days=0,
        ),
        trade_dates=trade_dates,
    )
    if book.attrs.get("target_index_semantics") != SIGNAL_DATE_SEMANTICS:
        raise RuntimeError("PIT-RL hold-band book is not signal dated")
    print(
        f"book: {book.index.min().date()}..{book.index.max().date()} "
        f"({len(book)} dates, max {int((book > 0).sum(axis=1).max())} names)",
        flush=True,
    )

    train_book = book[book.index <= pd.Timestamp(args.train_end)].copy()
    test_book = book[book.index >= pd.Timestamp(args.test_start)].copy()
    _mark_signal_dated(train_book)
    _mark_signal_dated(test_book)
    env_config = PITPortfolioEnvConfig(
        max_book=args.max_book,
        cost_bps=args.cost_bps,
    )

    # Instantiate before accepting/reusing any policy. This validates the full
    # reward/tradability evidence contract and slot capacity on both windows.
    train_probe = PITPortfolioEnv(
        train_book, predictions, panel, env_config
    )
    test_probe = PITPortfolioEnv(test_book, predictions, panel, env_config)
    if (
        train_probe.reward_clock_semantics != RL_REWARD_CLOCK_SEMANTICS
        or test_probe.reward_clock_semantics != RL_REWARD_CLOCK_SEMANTICS
    ):
        raise RuntimeError("RL environment reward clock semantics mismatch")

    contract = _policy_contract(
        reward_clock_semantics=RL_REWARD_CLOCK_SEMANTICS,
        target_index_semantics=SIGNAL_DATE_SEMANTICS,
        max_book=args.max_book,
        train_end=args.train_end,
        test_start=args.test_start,
        score_column=args.score_column,
        cost_bps=args.cost_bps,
        seed=args.seed,
    )
    policy_path = OUT / "policy.zip"
    contract_path = OUT / "policy_contract.json"

    if not args.skip_train or not policy_path.exists():
        print(
            f"=== training PPO ({args.timesteps} steps, {args.n_envs} envs) ===",
            flush=True,
        )

        def make_env(rank: int):
            def _factory():
                env = PITPortfolioEnv(
                    train_book, predictions, panel, env_config
                )
                env.reset(seed=args.seed + rank)
                return env

            return _factory

        vector_env = (
            SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
            if args.n_envs > 1
            else DummyVecEnv([make_env(0)])
        )
        model = PPO(
            "MlpPolicy",
            vector_env,
            device=args.device,
            seed=args.seed,
            tensorboard_log=str(OUT / "tb"),
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(policy_path)
        vector_env.close()
        _write_json(contract_path, contract)
    else:
        _require_matching_policy_contract(contract_path, contract)
        print("=== reusing contract-matched policy.zip ===", flush=True)

    print("=== deterministic test rollout ===", flush=True)
    model = PPO.load(policy_path, device="cpu")
    env = PITPortfolioEnv(test_book, predictions, panel, env_config)
    dispersion = env.book_dispersion_report()
    print(f"env dispersion: {json.dumps(dispersion)}", flush=True)

    observation, _ = env.reset(seed=7)
    rows: dict[pd.Timestamp, pd.Series] = {}
    value_add: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        weights = pd.Series(info["weights"])
        rows[pd.Timestamp(info["signal_date"])] = weights[weights > 1e-6]
        value_add.append(float(info["value_add"]))

    target_policy = _mark_signal_dated(
        pd.DataFrame(rows).T.fillna(0.0).sort_index()
    )
    target_policy.index.name = "trade_date"
    target_policy.to_parquet(OUT / "weights_test.parquet")
    target_passive = _mark_signal_dated(
        test_book.loc[test_book.index.isin(target_policy.index)].copy()
    )

    print("=== strict simulation: policy vs passive book ===", flush=True)
    simulation_panel = panel[
        panel["trade_date"]
        >= pd.Timestamp(args.test_start) - pd.Timedelta(days=5)
    ]
    strict_policy = _strict(
        target_policy, simulation_panel, sector, args.slippage_bps
    )
    strict_passive = _strict(
        target_passive, simulation_panel, sector, args.slippage_bps
    )

    null_annualized_returns: list[float] = []
    for index in range(args.random_baselines):
        random_model = PPO(
            "MlpPolicy",
            PITPortfolioEnv(test_book, predictions, panel, env_config),
            device="cpu",
            seed=1000 + index,
        )
        random_env = PITPortfolioEnv(
            test_book, predictions, panel, env_config
        )
        observation_k, _ = random_env.reset(seed=1000 + index)
        rows_k: dict[pd.Timestamp, pd.Series] = {}
        done_k = False
        while not done_k:
            action_k, _ = random_model.predict(
                observation_k, deterministic=True
            )
            observation_k, _, done_k, _, info_k = random_env.step(action_k)
            weights_k = pd.Series(info_k["weights"])
            rows_k[pd.Timestamp(info_k["signal_date"])] = weights_k[
                weights_k > 1e-6
            ]
        target_k = _mark_signal_dated(
            pd.DataFrame(rows_k).T.fillna(0.0).sort_index()
        )
        null_ann = _strict(
            target_k, simulation_panel, sector, args.slippage_bps
        )["ann"]
        null_annualized_returns.append(null_ann)
        print(f"  null #{index}: ann {null_ann:+.2%}", flush=True)

    env_cumulative_value_add = float(np.sum(value_add))
    beats_null = (
        strict_policy["ann"] > max(null_annualized_returns)
        if null_annualized_returns
        else True
    )
    research_gate_pass = bool(
        strict_policy["ann"] > strict_passive["ann"]
        and strict_policy["maxDD"] <= strict_passive["maxDD"] + 0.05
        and env_cumulative_value_add > 0
        and beats_null
        and dispersion["env_can_select"]
    )

    verdict = {
        "verdict": (
            "RESEARCH_GATE_PASS" if research_gate_pass else "RESEARCH_REJECT"
        ),
        "researchOnly": True,
        "productionEligible": False,
        "promotionBlocked": True,
        "promotionBlockReason": (
            "A clock-correct research win is not proof of deployable alpha. "
            "Require fresh strict historical rerun, multiple-testing/overfit "
            "governance, and independent forward shadow acceptance before use."
        ),
        "clock_blocked": False,
        "clock_audit_passed": True,
        "reward_clock_semantics": RL_REWARD_CLOCK_SEMANTICS,
        "policy_contract": contract,
        "target_index_semantics": SIGNAL_DATE_SEMANTICS,
        "execution_mapping_owner": "strict_ashare_simulator",
        "env_dispersion": dispersion,
        "selection_driven": bool(dispersion["env_can_select"]),
        "null_strict_anns_untrained": [
            round(value, 4) for value in null_annualized_returns
        ],
        "beats_all_null": bool(beats_null),
        "research_gate_pass": research_gate_pass,
        "window": (
            f"{target_policy.index.min().date()}.."
            f"{target_policy.index.max().date()}"
        ),
        "train_end": args.train_end,
        "strict_policy": strict_policy,
        "strict_passive_holdband": strict_passive,
        "ann_value_add_strict": round(
            strict_policy["ann"] - strict_passive["ann"], 4
        ),
        "env_cum_value_add": round(env_cumulative_value_add, 4),
        "mean_daily_turnover_policy": round(
            float(target_policy.diff().abs().sum(axis=1).mean() / 2),
            4,
        ),
        "timesteps": args.timesteps,
        "requiresFreshHistoricalRerun": True,
        "note": (
            "signal-dated PIT hold-band universe; execution on T+1; reward "
            "T+1->T+2; execution flags fail closed; RL remains research-only"
        ),
    }
    _write_json(OUT / "verdict.json", verdict)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
