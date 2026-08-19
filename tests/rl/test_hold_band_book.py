"""Round 23 / R11: the passive book must have a holding period.

The RL reward is value-add over a passive book, so the book *is* the benchmark
and its trading behaviour is baked into every number the environment reports.
The book in use through round 22 rebuilt the whole top-k list every session:
measured on the round-23 dataset (690 sessions, ``top_k=30``) it replaced a
median of 27 of its 30 names per session, two-sided turnover median 1.80 and
p90 2.00, median holding spell **one session**.

At the environment's 12 bps that benchmark pays ~21 bps of cost per session by
itself. Both the policy and the passive book then bleed cost far faster than a
drawdown term could matter, the arms get ranked by how little they trade, and a
reward ablation run on top of it silently measures transaction costs instead of
the thing it claims to measure. That is why this is a precondition for the
round-23 experiment and not a tidy-up.

These tests are written against the *property* -- a real holding period and a
band that does not evict on the first overtake -- rather than against the
specific numbers of one dataset, so they keep holding when the book parameters
are retuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.rl.books import hold_band_book_from_predictions

SESSIONS = pd.date_range("2026-01-05", periods=40, freq="B")


def _churning_predictions(n_symbols: int = 40) -> pd.DataFrame:
    """Scores whose *daily* top-k rotates completely every session.

    Each session the ranking is rotated by ``top_k`` positions, so a book that
    re-reads the top-k list every day replaces every single name every day.
    This is the pathology the hold band exists to survive; a book builder that
    quietly follows the ranking will fail the turnover assertions below.
    """
    symbols = [f"S{index:03d}" for index in range(n_symbols)]
    rows = []
    for position, trade_date in enumerate(SESSIONS):
        order = np.roll(np.arange(n_symbols), position * 10)
        for rank, index in enumerate(order):
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbols[index],
                    "alpha_score": float(n_symbols - rank),
                }
            )
    return pd.DataFrame(rows)


def _two_sided_turnover(book: pd.DataFrame) -> pd.Series:
    filled = book.fillna(0.0)
    return (filled - filled.shift(1).fillna(0.0)).abs().sum(axis=1).iloc[1:]


def _daily_topk(predictions: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for trade_date, group in predictions.groupby("trade_date", sort=True):
        chosen = group.nlargest(top_k, "alpha_score")["symbol"].astype(str)
        rows[pd.Timestamp(trade_date)] = {
            symbol: 1.0 / len(chosen) for symbol in chosen
        }
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def test_hold_band_book_turns_over_far_less_than_the_daily_rebuild():
    """The regression the round-23 experiment was blocked on.

    On a ranking engineered to rotate completely, the daily rebuild churns the
    entire book every session (turnover 2.0) while the hold band must not.
    """
    predictions = _churning_predictions()
    daily = _two_sided_turnover(_daily_topk(predictions, top_k=10))
    banded = _two_sided_turnover(
        hold_band_book_from_predictions(
            predictions,
            top_k=10,
            exit_rank=30,
            min_hold_sessions=10,
            rebalance_every=5,
        )
    )

    assert float(daily.median()) == pytest.approx(2.0)
    # The hold band is allowed to trade, but not on most sessions and never
    # anywhere near a full replacement on the median session.
    assert float(banded.median()) == pytest.approx(0.0)
    assert float(banded.mean()) < float(daily.mean()) / 5.0


def test_non_rebalance_sessions_have_exactly_zero_turnover():
    """A carried-forward book must be carried forward *bit-identically*.

    Re-deriving the same weights every session would be arithmetically equal
    but floating-point unequal, and the environment charges ``sum |dw|``: a
    1e-17 wobble on 30 names is a real, if tiny, cost that would accumulate
    over 690 sessions and be invisible in any summary statistic.
    """
    book = hold_band_book_from_predictions(
        _churning_predictions(),
        top_k=10,
        exit_rank=30,
        min_hold_sessions=10,
        rebalance_every=5,
    )
    turnover = _two_sided_turnover(book)
    traded = turnover[turnover > 0.0]
    # Every session that traded must be a rebalance session.
    positions = {book.index.get_loc(date) for date in traded.index}
    assert positions
    assert all(position % 5 == 0 for position in positions)
    assert float(turnover[turnover.index.map(lambda d: book.index.get_loc(d) % 5 != 0)].abs().max()) == 0.0


def test_minimum_holding_period_protects_a_name_that_falls_out_of_the_band():
    """A name may not be evicted before ``min_hold_sessions`` is satisfied.

    Without this, a rebalance landing right after entry re-creates the daily
    churn on exactly the sessions where it costs the most.
    """
    predictions = _churning_predictions()
    protected = hold_band_book_from_predictions(
        predictions,
        top_k=10,
        exit_rank=12,
        min_hold_sessions=20,
        rebalance_every=1,
    )
    unprotected = hold_band_book_from_predictions(
        predictions,
        top_k=10,
        exit_rank=12,
        min_hold_sessions=0,
        rebalance_every=1,
    )
    assert float(_two_sided_turnover(protected).mean()) < float(
        _two_sided_turnover(unprotected).mean()
    )
    held = protected > 1e-12
    spells = []
    values = held.to_numpy()
    for column in range(values.shape[1]):
        run = 0
        for row in range(values.shape[0]):
            if values[row, column]:
                run += 1
            elif run:
                spells.append(run)
                run = 0
    # Every completed spell must have reached the minimum holding period.
    assert spells
    assert min(spells) >= 20


def test_exit_rank_narrower_than_top_k_is_rejected():
    """An exit band no wider than the entry band is not a band.

    It evicts a name the moment anything overtakes it, which reproduces the
    daily-churn behaviour the builder exists to avoid -- silently, and under a
    name that claims otherwise.
    """
    with pytest.raises(ValueError, match="exit_rank"):
        hold_band_book_from_predictions(
            _churning_predictions(),
            top_k=10,
            exit_rank=9,
            min_hold_sessions=5,
            rebalance_every=5,
        )


def test_book_rows_do_not_depend_on_later_sessions():
    """Truncating the input must leave the surviving rows byte-identical.

    The book is a PIT artefact fed straight into the environment as the passive
    target. If a later session could change an earlier row, the benchmark --
    and therefore every value-add number quoted against it -- would be
    contaminated by the future.
    """
    predictions = _churning_predictions()
    cutoff = SESSIONS[25]
    full = hold_band_book_from_predictions(
        predictions,
        top_k=10,
        exit_rank=30,
        min_hold_sessions=10,
        rebalance_every=5,
    )
    truncated = hold_band_book_from_predictions(
        predictions[predictions["trade_date"] <= cutoff],
        top_k=10,
        exit_rank=30,
        min_hold_sessions=10,
        rebalance_every=5,
    )
    shared = full.loc[:cutoff]
    aligned = truncated.reindex(columns=shared.columns).fillna(0.0)
    assert shared.shape[0] == aligned.shape[0]
    np.testing.assert_array_equal(shared.to_numpy(), aligned.to_numpy())


def test_a_name_that_loses_its_score_is_dropped_without_reweighting_the_book():
    """A forced exit costs one name's weight, not a whole-book rebalance.

    The environment requires a finite alpha for every target name, so an
    unscored holding cannot be carried. Redistributing its weight across the
    survivors would charge turnover on all of them for one missing score; the
    book's gross dips instead and recovers at the next rebalance.
    """
    predictions = _churning_predictions()
    victim = predictions.loc[
        predictions["trade_date"] == SESSIONS[0]
    ].nlargest(1, "alpha_score")["symbol"].iloc[0]
    holed = predictions[
        ~(
            (predictions["symbol"] == victim)
            & (predictions["trade_date"] == SESSIONS[2])
        )
    ]
    book = hold_band_book_from_predictions(
        holed,
        top_k=10,
        exit_rank=30,
        min_hold_sessions=10,
        rebalance_every=5,
    )
    assert book.loc[SESSIONS[1], victim] > 0.0
    assert book.loc[SESSIONS[2], victim] == 0.0
    gross = book.sum(axis=1)
    assert gross.loc[SESSIONS[2]] == pytest.approx(gross.loc[SESSIONS[1]] - 0.1)
    turnover = _two_sided_turnover(book)
    # Exactly one leg moved: the dropped name. Nothing else was touched.
    assert float(turnover.loc[SESSIONS[2]]) == pytest.approx(0.1)
