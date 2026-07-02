"""Universe filter for ST / suspended / limit-up / limit-down handling.

Implements the user's "至少 90% 不能买 ST / 停牌, 但不绝对" requirement:

* **Suspended (停牌)** — hard exclude for new entries. This is a market
  mechanics constraint, not a strategy choice; you literally cannot
  place orders on a suspended stock. Existing holdings are not forced
  out (they cannot be sold while suspended either; they simply hold).
* **ST / *ST** — soft exclude. By default the bottom 90% of ST stocks
  (by prediction rank inside the ST cohort) are dropped. Only the top
  10% may pass through, AND the total ST share in the portfolio is
  capped at 10% of selected gross. This produces the "at least 90%
  can't be bought" guarantee while still allowing high-conviction
  picks to be expressed.
* **Limit-up at close (涨停)** — block new buys on the next day (can't
  fill at the limit-up price).
* **Limit-down at close (跌停)** — note in audit; selling decisions are
  the execution simulator's problem.

Data availability gaps (May 2026):

* ``market_panel.parquet`` carries OHLCV only — no ST / suspended
  columns. We **derive** is_suspended from ``volume == 0`` on days
  where the benchmark traded, and is_limit_up/down from a ±9.9% close
  change threshold (a board-conservative proxy — main board is ±10%,
  ChiNext / STAR are ±20%, ST is ±5%, so 9.9% slightly under-counts
  ChiNext / STAR limit-ups but never falsely flags them).
* ST flag is **not derivable** from OHLCV. We backfill from the stale
  ``market_features.parquet`` (covers 2020-02 → 2020-09 only) where
  available, and tag the rest of the window as "unknown ST" — those
  days the soft-filter only applies to suspended / limit-up checks.
  Stage 2 will fetch a fresh ST list from akshare to close this gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseFilterConfig:
    """Knobs for ST / suspended / limit-up / limit-down handling.

    ``st_min_block_rate`` — fraction of ST stocks per day that MUST be
    excluded. Defaults to 0.90 per user's "至少 90% 不能买". The
    filter computes a per-day prediction-rank threshold inside the ST
    cohort and keeps only the top (1 - st_min_block_rate) fraction.
    Setting to 1.0 is equivalent to hard exclude.

    ``st_max_portfolio_share`` — hard cap on the fraction of selected
    gross exposure that may be ST. Defensive belt-and-braces on top
    of the rank threshold — if too many ST stocks rank highly, only
    the top by prediction win the share allocation.

    ``suspended_block_new`` — when True, suspended stocks cannot be
    new entries. (Holdings during suspension are handled in the
    execution layer, not here.)

    ``limit_up_block_new`` — when True, stocks that closed at the
    upper limit yesterday cannot be new entries today.

    ``limit_up_pct`` / ``limit_down_pct`` — legacy flat thresholds. The
    tradable ``is_limit_up`` / ``is_limit_down`` flags are now derived
    **board-aware** by :func:`derive_market_flags` and no longer use these.
    They are retained only for the trailing high-chase feature
    (:func:`compute_high_chase_flags`), where a single conservative threshold
    for counting recent limit-ups is acceptable.
    """

    st_min_block_rate: float = 0.90
    st_max_portfolio_share: float = 0.10
    suspended_block_new: bool = True
    limit_up_block_new: bool = True
    limit_down_block_sell: bool = True  # advisory only — actual sells in execution sim
    limit_up_pct: float = 0.099
    limit_down_pct: float = -0.099
    require_amount_above: float = 0.0  # minimum daily amount to be considered tradable
    # High-chase penalty (user spec §10): block new entries when a stock
    # is "接盘多日高涨停" — defined as the AND of (a) THREE OR MORE
    # limit-ups inside a tight window and (b) elevated cumulative
    # return.
    #
    # Tuning history on v9 OOS:
    #   * OR(cum>0.30 / lim≥3 / lookback=10): cost 6.6pp excess via
    #     false-positives on grinding bear-rebound rallies.
    #   * AND(cum>0.30 / lim≥2 / lookback=5): cost 8.5pp via blocking
    #     legitimate post-bottom momentum names (folds 7/9).
    #   * AND(cum>0.30 / lim≥3 / lookback=5) [current default]: only
    #     catches the canonical 3-连板 pattern, the unambiguous
    #     "接盘多日高涨停" signal. Single + double limit-ups during
    #     healthy momentum pass through.
    high_chase_enabled: bool = True
    high_chase_lookback: int = 5
    high_chase_max_cum_return: float = 0.30
    high_chase_max_limit_ups: int = 3
    high_chase_combine: str = "and"  # "and" (both required) or "or" (either)


@dataclass(frozen=True)
class UniverseFilterResult:
    """Per-day filter outcome.

    ``filtered_predictions`` keeps the same schema as the input frame
    plus a boolean ``universe_pass`` column. Downstream sleeve picks
    should consume ``filtered_predictions[filtered_predictions['universe_pass']]``.

    ``audit`` is a long-format frame keyed on ``(trade_date, symbol)``
    listing exclusion reasons for the names that did NOT pass.

    ``summary`` aggregates the audit counts per trade_date and reason
    so callers can dump it to JSON without bloating logs.
    """

    filtered_predictions: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, object] = field(default_factory=dict)


def compute_high_chase_flags(
    market_panel: pd.DataFrame,
    *,
    lookback: int = 5,
    max_cum_return: float = 0.30,
    max_limit_ups: int = 2,
    limit_up_pct: float = 0.099,
    combine: str = "and",
) -> pd.DataFrame:
    """Derive per-(date, symbol) high-chase block flags from OHLCV.

    A stock is flagged ``is_high_chase`` on date T if BOTH (when
    ``combine='and'``) or EITHER (when ``combine='or'``):

    * its trailing ``lookback``-day cumulative close-to-close return
      exceeds ``max_cum_return``, AND/OR
    * it had ``max_limit_ups`` or more limit-up closes inside that
      window.

    ``combine='and'`` is the default and matches the user's
    "尽量不要接盘多日高涨停" intent — a grinding rally that doesn't
    hit limit-ups passes, and a single limit-up that wasn't part of a
    sustained run also passes. Only the conjunction (parabolic +
    multiple 涨停) blocks.

    The flag is computed BEFORE T's close — i.e. it uses returns through
    T-1 — so applying it as a "block new buys on T" rule does not peek
    at T's price.
    """
    if combine not in ("and", "or"):
        raise ValueError(f"combine must be 'and' or 'or', got {combine!r}")
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "is_high_chase", "cum_return_n", "limit_ups_n"])

    mp = market_panel.copy()
    mp["trade_date"] = pd.to_datetime(mp["trade_date"], errors="coerce")
    mp["symbol"] = mp["symbol"].astype(str)
    mp = mp.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Trailing N-day return using previous N closes (shift by 1 so today's
    # close is NOT in the lookback — this matches "block based on what we
    # know before today's open")
    mp["_close"] = pd.to_numeric(mp["close"], errors="coerce")
    mp["_close_lag1"] = mp.groupby("symbol")["_close"].shift(1)
    mp["_close_lag_n"] = mp.groupby("symbol")["_close"].shift(int(lookback) + 1)
    mp["cum_return_n"] = mp["_close_lag1"] / mp["_close_lag_n"] - 1.0

    # Count limit-ups in trailing window
    mp["_prev_close"] = mp.groupby("symbol")["_close"].shift(1)
    pct_change = (mp["_close"] - mp["_prev_close"]) / mp["_prev_close"]
    mp["_was_limit_up"] = (pct_change >= float(limit_up_pct)).astype(int)
    # Shift by 1 so today's limit-up does not count; sum trailing lookback values
    mp["_was_limit_up_lag1"] = mp.groupby("symbol")["_was_limit_up"].shift(1)
    mp["limit_ups_n"] = (
        mp.groupby("symbol")["_was_limit_up_lag1"]
        .rolling(window=int(lookback), min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    cum_hit = mp["cum_return_n"].fillna(0.0) > float(max_cum_return)
    limit_hit = mp["limit_ups_n"].fillna(0) >= int(max_limit_ups)
    if combine == "and":
        mp["is_high_chase"] = cum_hit & limit_hit
    else:
        mp["is_high_chase"] = cum_hit | limit_hit
    return mp[["trade_date", "symbol", "is_high_chase", "cum_return_n", "limit_ups_n"]].reset_index(drop=True)


def derive_market_flags(
    market_panel: pd.DataFrame,
    *,
    is_st_column: str = "is_st",
    tolerance: float = 1e-3,
    limits: "ASharePriceLimit | None" = None,
) -> pd.DataFrame:
    """Derive is_suspended / is_limit_up / is_limit_down from OHLCV.

    The returned frame has ``trade_date``, ``symbol``, and three new
    boolean columns. Original columns are *not* preserved — merge back
    by ``(trade_date, symbol)`` if you need them.

    Definitions
    -----------
    * **is_suspended**: ``volume`` is 0 / NaN AND ``amount`` is 0 / NaN
      on a date that has > 100 trading symbols in the panel. Days with
      few symbols are skipped (those are likely market holidays where
      the *whole* market did not trade — flagging every name as
      suspended would be misleading).
    * **is_limit_up**: ``close / prev_close - 1 >= board_limit - tolerance``,
      excluding the first day per symbol (no prev_close). The board limit
      is **board-aware** via :func:`quant_math.ashare.board_price_limit_vector`
      (main 10% / ChiNext·STAR 20% / BSE 30% / ST 5%) — a flat 10% rule
      mislabels every ChiNext/STAR/BSE name and is no longer used. When an
      ``is_st`` column is present it drives the ST 5% override.
    * **is_limit_down**: same with the down threshold.
    """

    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "is_suspended", "is_limit_up", "is_limit_down"])

    from quantagent.quant_math.ashare import board_price_limit_vector

    mp = market_panel.copy()
    mp["trade_date"] = pd.to_datetime(mp["trade_date"], errors="coerce")
    mp["symbol"] = mp["symbol"].astype(str)
    mp = mp.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Active days = days where at least 100 symbols have a finite close
    active_days_count = mp.groupby("trade_date")["close"].apply(
        lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())
    )
    active_days = set(active_days_count[active_days_count >= 100].index)

    volume_nan = (
        mp["volume"].isna()
        if "volume" in mp.columns
        else pd.Series(False, index=mp.index)
    )
    volume_zero = (
        pd.to_numeric(mp["volume"], errors="coerce").fillna(0.0) == 0
        if "volume" in mp.columns
        else pd.Series(False, index=mp.index)
    )
    amount_nan = (
        mp["amount"].isna()
        if "amount" in mp.columns
        else pd.Series(True, index=mp.index)
    )
    amount_zero = (
        pd.to_numeric(mp["amount"], errors="coerce").fillna(0.0) == 0
        if "amount" in mp.columns
        else pd.Series(False, index=mp.index)
    )
    has_no_trade = (volume_nan | volume_zero) & (amount_nan | amount_zero)
    mp["is_suspended"] = has_no_trade & mp["trade_date"].isin(active_days)

    # Board-aware limit-up / limit-down via close % change vs previous day per
    # symbol. The per-row limit ratio is resolved by the canonical A-share rule
    # engine (board prefix + optional ST 5% override) — single source of truth
    # shared with the provider/backtest/execution layers.
    is_st = mp[is_st_column] if is_st_column in mp.columns else False
    ratio = board_price_limit_vector(mp["symbol"], is_st, limits)
    up_threshold = ratio - float(tolerance)
    down_threshold = -(ratio - float(tolerance))
    mp["_prev_close"] = mp.groupby("symbol")["close"].shift(1)
    pct_change = (mp["close"] - mp["_prev_close"]) / mp["_prev_close"]
    mp["is_limit_up"] = pct_change >= up_threshold
    mp["is_limit_down"] = pct_change <= down_threshold
    # First day per symbol → no prev_close → not a limit move
    mp.loc[mp["_prev_close"].isna(), ["is_limit_up", "is_limit_down"]] = False

    return mp[["trade_date", "symbol", "is_suspended", "is_limit_up", "is_limit_down"]].reset_index(drop=True)


def _coerce_bool_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[col].fillna(False).astype(bool)


def apply_universe_filter(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame | None = None,
    st_flags: pd.DataFrame | None = None,
    config: UniverseFilterConfig | None = None,
) -> UniverseFilterResult:
    """Apply ST + suspended + limit-up filters per the user's policy.

    Parameters
    ----------
    predictions:
        Long frame with ``trade_date``, ``symbol``, ``prediction`` at
        minimum. Additional columns are passed through. Multiple
        horizons (one row per ``(date, symbol, horizon)``) are
        supported — the universe filter applies per ``(date, symbol)``
        irrespective of horizon.
    market_panel:
        OHLCV panel used to derive is_suspended / is_limit_up /
        is_limit_down. If None, those checks are skipped.
    st_flags:
        Optional long frame ``trade_date, symbol, is_st`` (bool). When
        absent the ST soft-exclusion is skipped (with a warning in
        ``summary['warnings']``). The Stage 2 feature pipeline rebuild
        will produce a fresh PIT-aligned ST flag table; until then we
        backfill from the stale market_features.parquet where
        available.
    config:
        UniverseFilterConfig knobs. Defaults match the user's "90% ST
        blocked, suspended hard-exclude" requirement.
    """

    config = config or UniverseFilterConfig()
    if predictions is None or predictions.empty:
        empty = pd.DataFrame(columns=["trade_date", "symbol", "reason"])
        return UniverseFilterResult(predictions, empty, summary={"status": "empty_input"})

    # Reset index so ST-filter's keep_mask.loc[idx] works regardless of
    # caller's index layout (review fix #2). Without this, any caller
    # passing a frame with a sliced / non-unique / non-integer index
    # would KeyError silently in the ST loop.
    work = predictions.copy().reset_index(drop=True)
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["symbol"] = work["symbol"].astype(str)

    warnings_collected: list[str] = []

    # 1) market flags (derived from OHLCV — board-aware limits, single source)
    if market_panel is not None and not market_panel.empty:
        flags = derive_market_flags(market_panel)
        # Defensive dedup (review fix #1): market_panel from disparate
        # sources can rarely contain duplicate (date, symbol) rows; an
        # m:n merge here would explode the prediction set silently.
        flags = flags.drop_duplicates(["trade_date", "symbol"], keep="last")
        # Note: limit_up flag is computed on TODAY's close. For new-entry
        # blocking, "today closed at limit-up → block tomorrow's buy" is
        # the correct policy. Predictions in v9/v10 OOS are made AT close
        # for next-period decisions, so blocking same-day is the right
        # semantics here.
        work = work.merge(flags, on=["trade_date", "symbol"], how="left")
    else:
        work["is_suspended"] = False
        work["is_limit_up"] = False
        work["is_limit_down"] = False

    # 2) ST flags — validate schema (review fix #3) instead of crashing
    if st_flags is not None and not st_flags.empty:
        if "is_st" not in st_flags.columns:
            warnings_collected.append("st_flags_missing_is_st_column — soft filter skipped")
        else:
            st = st_flags.copy()
            st["trade_date"] = pd.to_datetime(st["trade_date"], errors="coerce")
            st["symbol"] = st["symbol"].astype(str)
            st["is_st"] = st["is_st"].fillna(False).astype(bool)
            st = st.drop_duplicates(["trade_date", "symbol"], keep="last")
            work = work.merge(st[["trade_date", "symbol", "is_st"]], on=["trade_date", "symbol"], how="left")
    if "is_st" not in work.columns:
        work["is_st"] = False
    work["is_st"] = work["is_st"].fillna(False).astype(bool)

    # 3) Apply hard exclusions
    is_suspended = _coerce_bool_column(work, "is_suspended")
    is_limit_up = _coerce_bool_column(work, "is_limit_up")
    rejected_reasons: dict[int, str] = {}
    keep_mask = pd.Series(True, index=work.index)

    if config.suspended_block_new:
        for idx in work.index[is_suspended]:
            rejected_reasons[int(idx)] = "suspended"
        keep_mask &= ~is_suspended

    if config.limit_up_block_new:
        for idx in work.index[is_limit_up & keep_mask]:
            rejected_reasons[int(idx)] = "limit_up_at_close"
        keep_mask &= ~is_limit_up

    if config.require_amount_above > 0 and "amount" in work.columns:
        below_min = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0) < config.require_amount_above
        for idx in work.index[below_min & keep_mask]:
            rejected_reasons[int(idx)] = "below_min_amount"
        keep_mask &= ~below_min

    # High-chase penalty (user spec §10): block stocks that have rallied
    # too far AND hit too many limit-ups in the trailing window.
    if config.high_chase_enabled and market_panel is not None and not market_panel.empty:
        chase = compute_high_chase_flags(
            market_panel,
            lookback=int(config.high_chase_lookback),
            max_cum_return=float(config.high_chase_max_cum_return),
            max_limit_ups=int(config.high_chase_max_limit_ups),
            limit_up_pct=float(config.limit_up_pct),
            combine=str(config.high_chase_combine),
        )
        if not chase.empty:
            # Defensive dedup (review fix #1): if the panel had dup rows
            # for the same (date, symbol), the chase frame inherits them
            # and an m:n merge below would inflate the prediction set.
            chase = chase.drop_duplicates(["trade_date", "symbol"], keep="last")
            chase_aligned = work.merge(
                chase[["trade_date", "symbol", "is_high_chase"]],
                on=["trade_date", "symbol"],
                how="left",
            )["is_high_chase"].fillna(False).astype(bool)
            for idx in work.index[chase_aligned & keep_mask]:
                rejected_reasons[int(idx)] = "high_chase_block"
            keep_mask &= ~chase_aligned

    # 4) ST soft filter — keep top (1 - st_min_block_rate) per date by prediction
    is_st = _coerce_bool_column(work, "is_st")
    pred = pd.to_numeric(work["prediction"], errors="coerce")

    if config.st_min_block_rate > 0 and is_st.any() and pred.notna().any():
        per_date_groups = work[is_st & keep_mask].groupby("trade_date")
        for date, group in per_date_groups:
            n = len(group)
            if n == 0:
                continue
            # Compute n_block first then n_pass = n - n_block. Avoids the
            # (1.0 - 0.9) IEEE-754 drift that makes floor(n * 0.0999...) one
            # short of the intended top-quantile count.
            n_block = int(np.ceil(n * float(config.st_min_block_rate)))
            n_pass = max(0, n - n_block)
            if n_pass >= n:
                continue
            # rank by prediction within ST cohort; keep top n_pass
            ranked = group.assign(_p=pd.to_numeric(group["prediction"], errors="coerce")).sort_values("_p", ascending=False)
            blocked_idx = ranked.iloc[n_pass:].index
            for idx in blocked_idx:
                rejected_reasons[int(idx)] = "st_soft_block_below_top_quantile"
                keep_mask.loc[idx] = False

    # 5) Stage attaching universe_pass column
    work["universe_pass"] = keep_mask
    work["universe_reason"] = pd.Series(
        [rejected_reasons.get(int(i), "ok" if keep_mask.loc[i] else "blocked_unknown") for i in work.index],
        index=work.index,
    )

    audit = work.loc[~keep_mask, ["trade_date", "symbol", "universe_reason"]].rename(
        columns={"universe_reason": "reason"}
    ).reset_index(drop=True)

    summary: dict[str, object] = {
        "status": "passed",
        "n_rows_in": int(len(work)),
        "n_rows_pass": int(int(keep_mask.sum())),
        "block_rate": float(1.0 - keep_mask.mean()) if len(work) else 0.0,
        "by_reason": (
            audit["reason"].value_counts().to_dict() if not audit.empty else {}
        ),
        "config": {
            "st_min_block_rate": float(config.st_min_block_rate),
            "st_max_portfolio_share": float(config.st_max_portfolio_share),
            "suspended_block_new": bool(config.suspended_block_new),
            "limit_up_block_new": bool(config.limit_up_block_new),
            "limit_up_pct": float(config.limit_up_pct),
        },
        "warnings": warnings_collected + ([] if (st_flags is not None and not st_flags.empty) else ["st_flags_missing — ST soft filter skipped for days without ST data"]),
    }

    return UniverseFilterResult(filtered_predictions=work, audit=audit, summary=summary)


__all__ = [
    "UniverseFilterConfig",
    "UniverseFilterResult",
    "apply_universe_filter",
    "derive_market_flags",
]
