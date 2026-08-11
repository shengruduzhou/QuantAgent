#!/usr/bin/env python3
"""PIT-RL research under explicit execution, quarantine, and policy contracts.

Clock contract:
  * hold-band rows are SIGNAL-DATED (delay_days=0);
  * strict A-share execution owns signal T close -> execution T+1 close;
  * PITPortfolioEnv reward is close(T+1) -> close(T+2);
  * training censors by reward_end <= train_end, not signal date alone;
  * missing traded bars require proven SUSPENDED session-gap evidence.

Governance contract:
  * training may never intersect a quarantined/frozen holdout;
  * evaluation that intersects quarantine is blocked unless explicitly opened as
    logged forensics, and forensic output can never pass the research gate;
  * reusable PPO weights are content-bound to their effective training inputs;
  * this script never marks a policy production eligible.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SESSION_GAPS = "runtime/data/u0/panel/session_gaps.parquet"
OUT = Path("runtime/models/v88_rl_pit")
POLICY_CONTRACT_SCHEMA = "quantagent.rl.policy_contract.v2"


def _strict(
    target_weights: pd.DataFrame,
    panel: pd.DataFrame,
    sector: pd.DataFrame,
    slippage_bps: float,
) -> dict[str, float]:
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
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


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    data = frame.copy()
    metadata = json.dumps(
        {
            "shape": list(data.shape),
            "columns": [str(column) for column in data.columns],
            "index_name": str(data.index.name),
            "index_dtype": str(data.index.dtype),
            "dtypes": [str(dtype) for dtype in data.dtypes],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(metadata)
    if len(data):
        row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy(dtype="uint64")
        digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _training_input_fingerprint(
    *,
    book: pd.DataFrame,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    session_gaps: pd.DataFrame,
    train_end: str,
) -> tuple[str, dict[str, str]]:
    cutoff = pd.Timestamp(train_end)
    start = pd.Timestamp(book.index.min())
    pred_slice = predictions[
        (predictions["trade_date"] >= start)
        & (predictions["trade_date"] <= cutoff)
    ][["trade_date", "symbol", "alpha_score"]].copy()
    panel_columns = [
        column
        for column in (
            "trade_date",
            "symbol",
            "close",
            "is_limit_up",
            "is_limit_down",
            "is_suspended",
        )
        if column in panel.columns
    ]
    panel_slice = panel[panel["trade_date"] <= cutoff][panel_columns].copy()
    if session_gaps.empty:
        gap_slice = pd.DataFrame(columns=["trade_date", "symbol", "classification"])
    else:
        gap_slice = session_gaps[
            session_gaps["trade_date"] <= cutoff
        ][["trade_date", "symbol", "classification"]].copy()

    parts = {
        "book": _frame_fingerprint(book),
        "predictions": _frame_fingerprint(pred_slice),
        "panel": _frame_fingerprint(panel_slice),
        "session_gaps": _frame_fingerprint(gap_slice),
    }
    digest = sha256()
    for name, value in sorted(parts.items()):
        digest.update(f"{name}:{value}\n".encode("utf-8"))
    return digest.hexdigest(), parts


def _policy_contract(
    *,
    reward_clock_semantics: str,
    target_index_semantics: str,
    max_book: int,
    train_start: str,
    train_end: str,
    train_reward_end: str,
    test_start: str,
    warmup_start: str,
    score_column: str,
    predictions_path: str,
    panel_path: str,
    session_gaps_path: str,
    training_input_sha256: str,
    training_component_sha256: dict[str, str],
    cost_bps: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": POLICY_CONTRACT_SCHEMA,
        "reward_clock_semantics": reward_clock_semantics,
        "target_index_semantics": target_index_semantics,
        "execution_mapping_owner": "strict_ashare_simulator",
        "max_book": int(max_book),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "train_reward_end": str(train_reward_end),
        "test_start": str(test_start),
        "warmup_start": str(warmup_start),
        "score_column": str(score_column),
        "predictions_path": str(predictions_path),
        "panel_path": str(panel_path),
        "session_gaps_path": str(session_gaps_path),
        "training_input_sha256": str(training_input_sha256),
        "training_component_sha256": dict(training_component_sha256),
        "cost_bps": float(cost_bps),
        "seed": int(seed),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_matching_policy_contract(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file():
        raise RuntimeError(
            "existing RL policy has no policy_contract.json; stale pre-audit "
            "policies cannot be reused. Retrain without --skip-train."
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"cannot read RL policy contract {path}: {exc}") from exc
    if actual != expected:
        raise RuntimeError(
            "RL policy contract does not match current clock/training inputs; "
            "retrain instead of reusing stale weights. "
            f"expected={expected}, actual={actual}"
        )


def _reward_end_for_last_signal(
    signal_date: pd.Timestamp,
    sessions: list[pd.Timestamp],
) -> pd.Timestamp:
    normalized = [pd.Timestamp(item).normalize() for item in sessions]
    signal = pd.Timestamp(signal_date).normalize()
    try:
        index = normalized.index(signal)
    except ValueError as exc:
        raise RuntimeError(f"test signal {signal.date()} absent from market sessions") from exc
    if index + 2 >= len(normalized):
        raise RuntimeError(
            "test window is right-censored; two sessions after the final signal "
            "are required for execution/reward evidence"
        )
    return normalized[index + 2]


def main() -> int:
    global OUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=PREDS)
    parser.add_argument("--panel", default=PANEL)
    parser.add_argument("--sector-map", default=SECTOR)
    parser.add_argument("--session-gaps", default=SESSION_GAPS)
    parser.add_argument("--score-column", default="composite_score")
    parser.add_argument("--warmup-start", default="2024-08-09")
    parser.add_argument(
        "--train-end",
        default="2025-08-29",
        help="must remain before every quarantined/frozen holdout window",
    )
    parser.add_argument("--test-start", default="2026-01-02")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0)
    parser.add_argument("--max-book", type=int, default=80)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse policy.zip only when policy_contract.json matches exactly",
    )
    parser.add_argument("--random-baselines", type=int, default=5)
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument(
        "--allow-quarantined",
        default=None,
        metavar="JUSTIFICATION",
        help=(
            "forensics only: log quarantined evaluation access and force verdict "
            "to contaminated_holdout_forensics / RESEARCH_REJECT"
        ),
    )
    args = parser.parse_args()
    OUT = Path(args.output_dir)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from quantagent.backtest.quarantine import (
        FORENSICS_TRUST_CLASS,
        check_window,
        log_access,
        violation_message,
    )
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
        if args.score_column not in predictions.columns:
            raise ValueError(f"prediction score column {args.score_column!r} not found")
        predictions = predictions.rename(columns={args.score_column: "alpha_score"})

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
    panel = pd.read_parquet(args.panel, columns=panel_columns)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[
        panel["trade_date"]
        >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)
    ].copy()

    gaps_path = Path(args.session_gaps)
    if gaps_path.exists():
        session_gaps = pd.read_parquet(
            gaps_path,
            columns=["symbol", "trade_date", "classification"],
        )
        session_gaps["trade_date"] = pd.to_datetime(session_gaps["trade_date"])
        session_gaps = session_gaps[
            session_gaps["trade_date"]
            >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)
        ].copy()
    else:
        session_gaps = pd.DataFrame(
            columns=["symbol", "trade_date", "classification"]
        )

    sector = pd.read_parquet(args.sector_map)
    flags = panel[
        ["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]
    ]
    predictions_with_flags = predictions.merge(
        flags, on=["symbol", "trade_date"], how="left"
    )
    trade_dates = [pd.Timestamp(value) for value in sorted(panel["trade_date"].unique())]

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
    if train_book.empty or test_book.empty:
        raise RuntimeError("RL train/test book is empty after configured date split")
    _mark_signal_dated(train_book)
    _mark_signal_dated(test_book)

    training_quarantine = check_window(train_book.index.min(), args.train_end)
    if training_quarantine is not None:
        raise RuntimeError(
            "RL training is forbidden from consuming quarantined/frozen holdout "
            "outcomes. Change --train-end.\n"
            + violation_message(
                train_book.index.min(), args.train_end, training_quarantine
            )
        )

    requested_eval_end = _reward_end_for_last_signal(
        pd.Timestamp(test_book.index.max()), trade_dates
    )
    quarantine_hit = check_window(test_book.index.min(), requested_eval_end)
    forensic_access: dict[str, object] | None = None
    if quarantine_hit is not None:
        if not args.allow_quarantined:
            raise RuntimeError(
                violation_message(
                    test_book.index.min(), requested_eval_end, quarantine_hit
                )
            )
        forensic_access = log_access(
            quarantine_hit,
            args.allow_quarantined,
            test_book.index.min(),
            requested_eval_end,
        )
        trust_class = FORENSICS_TRUST_CLASS
    else:
        # This script is not the canonical variant-C evaluator and therefore
        # cannot self-award clean_oos/walk_forward_oos.
        trust_class = "unknown"

    train_env_config = PITPortfolioEnvConfig(
        max_book=args.max_book,
        cost_bps=args.cost_bps,
        reward_end_date_limit=args.train_end,
    )
    test_env_config = PITPortfolioEnvConfig(
        max_book=args.max_book,
        cost_bps=args.cost_bps,
    )
    train_probe = PITPortfolioEnv(
        train_book,
        predictions,
        panel,
        train_env_config,
        session_gaps=session_gaps,
    )
    test_probe = PITPortfolioEnv(
        test_book,
        predictions,
        panel,
        test_env_config,
        session_gaps=session_gaps,
    )
    if train_probe.reward_end_dates[-1] > pd.Timestamp(args.train_end):
        raise RuntimeError("RL training reward crosses train_end boundary")
    if (
        train_probe.reward_clock_semantics != RL_REWARD_CLOCK_SEMANTICS
        or test_probe.reward_clock_semantics != RL_REWARD_CLOCK_SEMANTICS
    ):
        raise RuntimeError("RL environment reward clock semantics mismatch")

    training_sha, training_parts = _training_input_fingerprint(
        book=train_book,
        predictions=predictions,
        panel=panel,
        session_gaps=session_gaps,
        train_end=args.train_end,
    )
    contract = _policy_contract(
        reward_clock_semantics=RL_REWARD_CLOCK_SEMANTICS,
        target_index_semantics=SIGNAL_DATE_SEMANTICS,
        max_book=args.max_book,
        train_start=train_probe.dates[0].date().isoformat(),
        train_end=args.train_end,
        train_reward_end=train_probe.reward_end_dates[-1].date().isoformat(),
        test_start=args.test_start,
        warmup_start=args.warmup_start,
        score_column=args.score_column,
        predictions_path=args.predictions,
        panel_path=args.panel,
        session_gaps_path=args.session_gaps,
        training_input_sha256=training_sha,
        training_component_sha256=training_parts,
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
                    train_book,
                    predictions,
                    panel,
                    train_env_config,
                    session_gaps=session_gaps,
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
    env = PITPortfolioEnv(
        test_book,
        predictions,
        panel,
        test_env_config,
        session_gaps=session_gaps,
    )
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
            PITPortfolioEnv(
                test_book,
                predictions,
                panel,
                test_env_config,
                session_gaps=session_gaps,
            ),
            device="cpu",
            seed=1000 + index,
        )
        random_env = PITPortfolioEnv(
            test_book,
            predictions,
            panel,
            test_env_config,
            session_gaps=session_gaps,
        )
        observation_k, _ = random_env.reset(seed=1000 + index)
        rows_k: dict[pd.Timestamp, pd.Series] = {}
        done_k = False
        while not done_k:
            action_k, _ = random_model.predict(observation_k, deterministic=True)
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

    # Strict replay marks through the final target's T+1 execution session. The
    # environment's final transition additionally consumes T+1->T+2. Exclude
    # that terminal reward from the cross-engine gate instead of comparing
    # different economic windows.
    aligned_value_add = value_add[:-1]
    if not aligned_value_add:
        raise RuntimeError("RL evaluation has no strict-window-aligned value-add steps")
    env_cumulative_value_add = float(np.sum(aligned_value_add))
    beats_null = (
        strict_policy["ann"] > max(null_annualized_returns)
        if null_annualized_returns
        else True
    )
    gate_conditions = bool(
        strict_policy["ann"] > strict_passive["ann"]
        and strict_policy["maxDD"] <= strict_passive["maxDD"] + 0.05
        and env_cumulative_value_add > 0
        and beats_null
        and dispersion["env_can_select"]
    )
    research_gate_pass = bool(gate_conditions and quarantine_hit is None)

    verdict = {
        "verdict": "RESEARCH_GATE_PASS" if research_gate_pass else "RESEARCH_REJECT",
        "researchOnly": True,
        "productionEligible": False,
        "promotionBlocked": True,
        "promotionBlockReason": (
            "Clock-correct research is not deployable alpha. Require canonical "
            "variant-C evaluation, multiple-testing/overfit governance, and "
            "independent forward shadow acceptance."
        ),
        "trust_class": trust_class,
        "quarantine_hit": (
            {
                "start": str(quarantine_hit.start.date()),
                "end": str(quarantine_hit.end.date()),
                "reason": quarantine_hit.reason,
            }
            if quarantine_hit is not None
            else None
        ),
        "forensic_access": forensic_access,
        "clock_blocked": False,
        "clock_audit_passed": True,
        "reward_clock_semantics": RL_REWARD_CLOCK_SEMANTICS,
        "policy_contract": contract,
        "target_index_semantics": SIGNAL_DATE_SEMANTICS,
        "execution_mapping_owner": "strict_ashare_simulator",
        "env_dispersion": dispersion,
        "selection_driven": bool(dispersion["env_can_select"]),
        "null_strict_anns_untrained": [round(value, 4) for value in null_annualized_returns],
        "beats_all_null": bool(beats_null),
        "raw_gate_conditions_pass": gate_conditions,
        "research_gate_pass": research_gate_pass,
        "window": f"{target_policy.index.min().date()}..{target_policy.index.max().date()}",
        "strict_window_end": str(env.execution_dates[-1].date()),
        "env_terminal_reward_end_excluded_from_gate": str(env.reward_end_dates[-1].date()),
        "train_end": args.train_end,
        "strict_policy": strict_policy,
        "strict_passive_holdband": strict_passive,
        "ann_value_add_strict": round(strict_policy["ann"] - strict_passive["ann"], 4),
        "env_cum_value_add_aligned_to_strict_window": round(env_cumulative_value_add, 4),
        "mean_daily_turnover_policy": round(
            float(target_policy.diff().abs().sum(axis=1).mean() / 2), 4
        ),
        "timesteps": args.timesteps,
        "requiresFreshHistoricalRerun": True,
        "note": (
            "signal T; execution T+1; reward T+1->T+2; train reward censored; "
            "suspension gaps proven; quarantine enforced; RL remains research-only"
        ),
    }
    _write_json(OUT / "verdict.json", verdict)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
