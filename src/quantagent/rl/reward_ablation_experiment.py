"""Runnable driver for the round-23 RL reward ablation.

``reward_ablation`` is the comparison machinery; this module is the *experiment*
-- the part that says which real artefacts were read, how the passive book was
built, where the train/test boundary sits and which arms were run. It exists as
committed code rather than a notebook because a verdict on "does the risk term
help" is only worth anything if someone else can re-run the exact thing that
produced it.

Nothing here fabricates data. Every input is a path to a real artefact and the
driver fails loudly if one is missing; there is no synthetic fallback, because a
synthetic panel would let the ablation return a confident number about nothing.

Two properties of the setup deserve to be stated where they cannot be missed:

* **The passive book must have a holding period.** With the daily top-k rebuild
  the benchmark replaces most of its names every session, and at 12 bps both
  arms drown in transaction cost long before a drawdown term could matter. See
  :mod:`quantagent.rl.books`.
* **The yardstick carries no risk term.** Every arm, including the ones trained
  with a penalty, is scored in an environment built with
  ``drawdown_lambda = volatility_lambda = 0``. Scoring a drawdown-penalised
  policy with a drawdown-penalised metric would assume the conclusion.

Symbols whose panel bar is missing on some session *without* a ``SUSPENDED``
classification in ``session_gaps`` are dropped from the candidate universe up
front, because the environment fails closed on them by design. That exclusion
uses whole-window knowledge, so it is a mild look-ahead in the *universe
definition*; it is applied identically to the passive book and to every arm, and
the count is printed so it cannot be silently large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.rl.books import hold_band_book_from_predictions
from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig
from quantagent.rl.reward_ablation import (
    AblationConfig,
    RewardArm,
    evaluate_episode,
    run_ablation,
    summarise,
    zero_policy,
)

_EXECUTION_FLAGS = ("is_limit_up", "is_limit_down", "is_suspended")


def load_inputs(
    *,
    panel_path: str | Path,
    predictions_path: str | Path,
    session_gaps_path: str | Path | None,
    score_column: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Read the real artefacts and return ``(predictions, panel, gaps, dropped)``."""
    panel = pd.read_parquet(panel_path)
    missing = sorted(
        {"trade_date", "symbol", "close", *_EXECUTION_FLAGS} - set(panel.columns)
    )
    if missing:
        raise ValueError(f"market panel {panel_path} missing columns: {missing}")
    panel = panel[["trade_date", "symbol", "close", *_EXECUTION_FLAGS]].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    for flag in _EXECUTION_FLAGS:
        panel[flag] = pd.to_numeric(panel[flag], errors="coerce").fillna(0.0) > 0.5

    predictions = pd.read_parquet(predictions_path)
    if score_column not in predictions.columns:
        raise ValueError(
            f"predictions {predictions_path} has no column {score_column!r}"
        )
    predictions = predictions[["trade_date", "symbol", score_column]].copy()
    predictions = predictions.rename(columns={score_column: "alpha_score"})
    predictions["trade_date"] = pd.to_datetime(
        predictions["trade_date"]
    ).dt.normalize()
    predictions["symbol"] = predictions["symbol"].astype(str)
    predictions = predictions[
        (predictions["trade_date"] >= pd.Timestamp(start))
        & (predictions["trade_date"] <= pd.Timestamp(end))
    ]
    if predictions.empty:
        raise ValueError(f"no predictions inside [{start}, {end}]")

    gaps = None
    if session_gaps_path is not None:
        gaps = pd.read_parquet(session_gaps_path)
        gaps["trade_date"] = pd.to_datetime(gaps["trade_date"]).dt.normalize()

    dropped = _symbols_without_proven_coverage(predictions, panel, gaps)
    predictions = predictions[~predictions["symbol"].isin(dropped)]
    return predictions, panel, gaps, dropped


def _symbols_without_proven_coverage(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    gaps: pd.DataFrame | None,
) -> list[str]:
    """Symbols with a bar missing on some session and no ``SUSPENDED`` proof.

    The environment refuses to price a position on a session it has no proven
    close for, which is the correct behaviour and must not be relaxed. Removing
    the affected names from the candidate universe keeps that guard intact
    instead of weakening it.
    """
    sessions = pd.DatetimeIndex(np.sort(panel["trade_date"].unique()))
    symbols = sorted(set(predictions["symbol"].astype(str)))
    full = pd.MultiIndex.from_product(
        [symbols, sessions], names=["symbol", "trade_date"]
    )
    present = pd.MultiIndex.from_frame(
        panel.loc[panel["symbol"].isin(symbols), ["symbol", "trade_date"]]
    )
    absent = pd.DataFrame(index=full.difference(present)).reset_index()
    if absent.empty:
        return []
    if gaps is None:
        return sorted(absent["symbol"].unique())
    merged = absent.merge(gaps, on=["symbol", "trade_date"], how="left")
    classification = merged["classification"].fillna("NOT_IN_GAPS").str.upper()
    return sorted(merged.loc[classification != "SUSPENDED", "symbol"].unique())


def daily_topk_book(predictions: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """The pre-fix passive book: the whole top-k list rebuilt every session.

    Kept here, not deleted, because the round-23 report has to quote the
    turnover of the book that was actually in use before the fix, and a number
    nobody can reproduce is not evidence.
    """
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for date, group in predictions.groupby("trade_date", sort=True):
        chosen = group.nlargest(top_k, "alpha_score")["symbol"].astype(str)
        if chosen.empty:
            continue
        weight = 1.0 / float(len(chosen))
        rows[pd.Timestamp(date)] = {symbol: weight for symbol in chosen}
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def book_turnover_report(book: pd.DataFrame) -> dict[str, float]:
    """Turnover of the *book itself*, before any execution constraint.

    Two-sided: a completely replaced 1.0-gross book scores 2.0, matching the
    environment's ``sum |dw|`` accounting. The first session is excluded because
    it measures building the book from cash, not rebalancing it.

    Holding spells still open on the last session are counted at their observed
    length, so the spell statistics are right-censored and understate a
    long-holding book. That biases *against* the hold band and in favour of the
    daily rebuild, so it cannot manufacture the difference being reported.
    """
    filled = book.fillna(0.0)
    delta = (filled - filled.shift(1).fillna(0.0)).abs().sum(axis=1)
    steady = delta.iloc[1:]
    held = filled > 1e-12
    entered = held & ~held.shift(1, fill_value=False)
    spells: list[int] = []
    values = held.to_numpy()
    for column in range(values.shape[1]):
        run = 0
        for row in range(values.shape[0]):
            if values[row, column]:
                run += 1
            elif run:
                spells.append(run)
                run = 0
        if run:
            spells.append(run)
    return {
        "sessions": int(len(book)),
        "turnover_median": float(steady.median()),
        "turnover_mean": float(steady.mean()),
        "turnover_p90": float(steady.quantile(0.90)),
        "names_entered_per_session_median": float(entered.sum(axis=1).iloc[1:].median()),
        "holding_spell_median_sessions": float(np.median(spells)) if spells else float("nan"),
        "holding_spell_mean_sessions": float(np.mean(spells)) if spells else float("nan"),
        "gross_median": float(filled.sum(axis=1).median()),
    }


def default_arms(
    drawdown_lambdas: tuple[float, ...],
    volatility_lambdas: tuple[float, ...],
) -> list[RewardArm]:
    """Controls first, then the incumbent reward, then the treatments."""
    arms = [
        RewardArm(name="zero", kind="zero"),
        RewardArm(name="random", kind="random"),
        RewardArm(name="old_reward_lambda0", kind="trained"),
    ]
    arms += [
        RewardArm(name=f"drawdown_lambda_{value:g}", drawdown_lambda=value)
        for value in drawdown_lambdas
    ]
    arms += [
        RewardArm(name=f"volatility_lambda_{value:g}", volatility_lambda=value)
        for value in volatility_lambdas
    ]
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--session-gaps", default=None)
    parser.add_argument("--score-column", default="alpha_score")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--exit-rank", type=int, default=90)
    parser.add_argument("--min-hold-sessions", type=int, default=10)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument(
        "--train-sessions",
        type=int,
        required=True,
        help="number of leading book sessions used for training; the rest is the "
        "evaluation window, and the training reward clock is censored so no "
        "training transition can read a return from it",
    )
    parser.add_argument(
        "--eval-reward-end-limit",
        default=None,
        help="censor the evaluation reward clock here; defaults to --end so no "
        "reward interval can read a return from beyond the declared window",
    )
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="1729,20260819,7,13,42")
    parser.add_argument("--drawdown-lambdas", default="0.5,1.0,2.0")
    parser.add_argument("--volatility-lambdas", default="25,100")
    parser.add_argument("--arms", default=None, help="comma-separated arm subset")
    parser.add_argument("--results", required=True, help="JSONL sink; resumable")
    parser.add_argument("--book-report", default=None)
    args = parser.parse_args(argv)

    predictions, panel, gaps, dropped = load_inputs(
        panel_path=args.panel,
        predictions_path=args.predictions,
        session_gaps_path=args.session_gaps,
        score_column=args.score_column,
        start=args.start,
        end=args.end,
    )
    print(
        f"universe: {predictions['symbol'].nunique()} symbols, "
        f"{predictions['trade_date'].nunique()} signal sessions; "
        f"dropped {len(dropped)} symbols lacking proven session coverage",
        flush=True,
    )

    book = hold_band_book_from_predictions(
        predictions,
        top_k=args.top_k,
        exit_rank=args.exit_rank,
        min_hold_sessions=args.min_hold_sessions,
        rebalance_every=args.rebalance_every,
    )
    report = {
        "before_daily_topk": book_turnover_report(
            daily_topk_book(predictions, top_k=args.top_k)
        ),
        "after_hold_band": book_turnover_report(book),
        "hold_band_params": {
            "top_k": args.top_k,
            "exit_rank": args.exit_rank,
            "min_hold_sessions": args.min_hold_sessions,
            "rebalance_every": args.rebalance_every,
        },
        "dropped_symbols": dropped,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_symbols"},
                     indent=2, sort_keys=True), flush=True)
    if args.book_report:
        Path(args.book_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.book_report).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    dates = list(book.index)
    if not 0 < args.train_sessions < len(dates):
        raise ValueError(
            f"--train-sessions must be inside (0, {len(dates)}), got {args.train_sessions}"
        )
    train_book = book.iloc[: args.train_sessions]
    test_book = book.iloc[args.train_sessions :]
    train_limit = str(pd.Timestamp(dates[args.train_sessions - 1]).date())
    print(
        f"train {dates[0].date()}..{dates[args.train_sessions - 1].date()} "
        f"({len(train_book)} sessions, reward censored at {train_limit}) | "
        f"test {dates[args.train_sessions].date()}..{dates[-1].date()} "
        f"({len(test_book)} sessions)",
        flush=True,
    )

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    arms = default_arms(
        tuple(float(v) for v in args.drawdown_lambdas.split(",") if v.strip()),
        tuple(float(v) for v in args.volatility_lambdas.split(",") if v.strip()),
    )
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
        unknown = wanted - {arm.name for arm in arms}
        if unknown:
            raise ValueError(f"unknown arm names: {sorted(unknown)}")
        arms = [arm for arm in arms if arm.name in wanted]

    # The evaluation reward clock is censored at ``--end`` too, not just the
    # signal date. A transition signalled on the last in-window session rewards
    # over close(T+1) -> close(T+2), which lands two sessions *later*; leaving
    # it uncensored would read returns from beyond the declared window -- and on
    # this repository's calendar, straight into the quarantined burned holdout
    # that starts the day after. Signal-date truncation alone is not enough.
    config = AblationConfig(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        device=args.device,
        seeds=seeds,
        base_env=PITPortfolioEnvConfig(),
        train_reward_end_limit=train_limit,
        eval_reward_end_limit=args.eval_reward_end_limit or args.end,
    )
    runs = run_ablation(
        train_book=train_book,
        test_book=test_book,
        predictions=predictions,
        panel=panel,
        arms=arms,
        config=config,
        session_gaps=gaps,
        progress=True,
        results_path=args.results,
    )
    print(summarise(runs).to_string(index=False), flush=True)
    return 0


def passive_book_probe(
    book: pd.DataFrame,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    session_gaps: pd.DataFrame | None,
    config: PITPortfolioEnvConfig | None = None,
):
    """Roll the zero action through one environment built on ``book``.

    Used to report the *environment's* passive turnover and drawdown, which
    differ from :func:`book_turnover_report` because execution constraints
    (limit-up, limit-down, suspension) can hold a position the book wanted to
    move.
    """
    env = PITPortfolioEnv(
        book, predictions, panel, config or PITPortfolioEnvConfig(),
        session_gaps=session_gaps,
    )
    return evaluate_episode(env, zero_policy(int(env.action_space.shape[0])))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
