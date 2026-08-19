"""Passive-book construction for the RL environment.

The RL reward is *value-add over a passive book*, so the book is not a detail:
it is the benchmark, and its trading behaviour is baked into every number the
environment reports. ``train_ppo.equal_weight_book_from_predictions`` rebuilds
the top-k list from scratch on every session. Measured on the round-23 dataset
(v8.9+7 ``short_5d`` scores, 257 sessions, ``top_k=30``) that book replaces a
median of 21 of its 30 names per session: two-sided turnover has median 1.40
and mean 1.40, i.e. ~70% of the book one-way *every day*. At the environment's
12 bps cost that is ~16.8 bps of cost per session on the benchmark itself.

A reward ablation run against that book does not measure what it claims to
measure. Both the policy and the passive book bleed cost far faster than any
drawdown term could matter, so the arms get ranked by how much trading they
avoid, and the experiment silently becomes a study of transaction costs.

:func:`hold_band_book_from_predictions` builds a book with an actual holding
period instead: names enter only on rebalance sessions, are protected by a
minimum holding period, and leave only when they fall out of a wider exit
band. Nothing here is a claim that this book is *good* -- it is a benchmark
with a plausible holding period, which is the precondition for the drawdown
question being askable at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["hold_band_book_from_predictions"]


def hold_band_book_from_predictions(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    exit_rank: int,
    min_hold_sessions: int,
    rebalance_every: int,
    entry_rank_limit: int | None = None,
    score_column: str | None = None,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Equal-weight top-k book with a hold band and a minimum holding period.

    Rules, applied session by session in chronological order:

    * A held name whose score is missing on a session is dropped immediately.
      This is a *forced* exit -- the environment requires a finite alpha for
      every target name, so carrying an unscored name would raise. The freed
      weight is not redistributed until the next rebalance, so the book's gross
      dips rather than the whole book being re-weighted (which would charge
      turnover on every surviving name for one missing one).
    * On rebalance sessions only (every ``rebalance_every`` sessions, counted
      from the first): held names are dropped if their rank is worse than
      ``exit_rank`` **and** they have been held at least ``min_hold_sessions``
      sessions; free slots are then refilled from the best-ranked names not
      already held.
    * On every other session the book is carried forward unchanged, so its
      turnover is exactly zero.

    ``exit_rank`` must be at least ``top_k``: an exit band no wider than the
    entry band is not a band, and would evict a name the moment it is passed by
    any other, reproducing the daily-churn behaviour this function exists to
    avoid.

    Only information dated at or before each session is read -- the loop never
    looks at a later row -- so the book is PIT-safe with respect to the
    prediction frame it is given. It inherits whatever PIT properties that frame
    has and adds none of its own.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if exit_rank < top_k:
        raise ValueError(
            f"exit_rank ({exit_rank}) must be >= top_k ({top_k}); a band that is "
            "not wider than the entry rank evicts on the first overtake and "
            "reproduces daily churn"
        )
    if min_hold_sessions < 0:
        raise ValueError("min_hold_sessions must be non-negative")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")

    frame = predictions.copy()
    if score_column is None:
        if "alpha_score" in frame.columns:
            score_column = "alpha_score"
        elif "prediction" in frame.columns:
            score_column = "prediction"
        else:
            raise ValueError(
                "predictions require an alpha_score or prediction column"
            )
    missing = sorted(
        {date_column, symbol_column, score_column} - set(frame.columns)
    )
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")

    frame = frame[[date_column, symbol_column, score_column]]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[symbol_column] = frame[symbol_column].astype(str)
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, symbol_column, score_column])
    if frame.empty:
        raise ValueError("cannot build a passive book from empty predictions")
    if frame.duplicated([date_column, symbol_column]).any():
        raise ValueError(
            "predictions contain duplicate trade_date/symbol rows; the book "
            "would depend on row order"
        )

    scores = frame.pivot(
        index=date_column, columns=symbol_column, values=score_column
    ).sort_index()
    # ``method="first"`` makes ties resolve by column order, which is stable
    # across runs; ``average`` would produce fractional ranks that compare
    # inconsistently against the integer band edges.
    ranks = scores.rank(axis=1, ascending=False, method="first")
    if entry_rank_limit is None:
        entry_rank_limit = exit_rank

    weight = 1.0 / float(top_k)
    held_age: dict[str, int] = {}
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for position, date in enumerate(scores.index):
        score_row = scores.loc[date]
        rank_row = ranks.loc[date]
        scored = score_row.notna()

        held_age = {
            symbol: age
            for symbol, age in held_age.items()
            if bool(scored.get(symbol, False))
        }

        if position % rebalance_every == 0:
            held_age = {
                symbol: age
                for symbol, age in held_age.items()
                if age < min_hold_sessions
                or float(rank_row.get(symbol, np.inf)) <= exit_rank
            }
            if len(held_age) < top_k:
                eligible = rank_row[scored & (rank_row <= entry_rank_limit)]
                for symbol in eligible.sort_values(kind="mergesort").index:
                    if len(held_age) >= top_k:
                        break
                    if symbol not in held_age:
                        held_age[str(symbol)] = 0

        rows[pd.Timestamp(date)] = {
            symbol: weight for symbol in sorted(held_age)
        }
        held_age = {symbol: age + 1 for symbol, age in held_age.items()}

    rows = {date: held for date, held in rows.items() if held}
    if not rows:
        raise ValueError("hold-band book selected no names on any session")
    book = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    return book.astype(float)
