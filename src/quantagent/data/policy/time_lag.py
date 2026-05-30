"""Stage 4.2 — policy time-lag model.

Given a normalised policy event frame and a sector-return panel,
estimate the typical lag between an announcement and its observable
price impact in affected sectors.  The output drives a lag-shifted
feature that the FT-Transformer can learn from.

Method (deliberately simple, deliberately frequentist):

For every (event, hinted_sector) pair we compute the **sector excess
return** over [announce_t + 1, announce_t + k] for k = 1..max_lag.
Excess = sector return − market return (so the metric isolates the
*differential* sector impact, not the market-wide drift on policy
days).  We then group by theme, average across events, and report the
mean excess curve along with its argmax (best lag).

Two policy strengths are applied:

* Per-event sample weight = ``policy_strength`` from the event row.
* Per-event sample weight is further multiplied by ``min(1.0, n_obs / 5)``
  so a theme with only 1-2 events doesn't dominate the curve.

Output is meant for offline analysis + feature engineering — never for
live trading without a separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeLagConfig:
    max_lag_days: int = 20
    min_events_per_theme: int = 3
    apply_policy_strength: bool = True
    apply_event_count_weight: bool = True


@dataclass
class TimeLagResult:
    """Output of the time-lag estimator.

    * ``lag_curves`` — long-form: columns ``theme``, ``lag_k``,
      ``mean_excess``, ``n_events``.
    * ``best_lag`` — dict ``theme → (best_k, peak_excess)``.
    * ``per_event_excess`` — wide event-level diagnostic frame.
    """

    lag_curves: pd.DataFrame
    best_lag: dict[str, tuple[int, float]]
    per_event_excess: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _explode_sector_themes(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, theme, sector_hint).

    Events with empty themes are dropped — they cannot contribute to a
    theme-conditional lag.  Events with empty ``sectors_hint`` keep a
    synthetic ``"_ALL_"`` sector row so we still produce a market-wide
    curve for that event.
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["event_id", "announced_at", "theme", "sector", "policy_strength"])
    work = events.copy()
    work["sectors_hint"] = work["sectors_hint"].apply(
        lambda v: list(v) if isinstance(v, (list, tuple)) and v else ["_ALL_"]
    )
    work = work.explode("themes").rename(columns={"themes": "theme"})
    work = work[work["theme"].notna() & (work["theme"] != "")]
    work = work.explode("sectors_hint").rename(columns={"sectors_hint": "sector"})
    work["sector"] = work["sector"].astype(str)
    return work[["event_id", "announced_at", "theme", "sector", "policy_strength"]]


def _sector_excess_panel(
    sector_returns: pd.DataFrame,
    market_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Excess-return panel (long form: trade_date, sector, excess_ret).

    ``sector_returns`` may be either:
      * long-form with columns ``trade_date``, ``sector``, ``ret`` (one
        sector return per day), OR
      * wide-form with index ``trade_date`` and columns = sectors.
    """
    if "sector" in sector_returns.columns and "ret" in sector_returns.columns:
        long = sector_returns[["trade_date", "sector", "ret"]].copy()
    else:
        wide = sector_returns.copy()
        if "trade_date" in wide.columns:
            wide = wide.set_index("trade_date")
        wide.index = pd.to_datetime(wide.index)
        long = (
            wide.stack(future_stack=True)
            .rename("ret")
            .reset_index()
            .rename(columns={"level_1": "sector"})
        )
    long["trade_date"] = pd.to_datetime(long["trade_date"])

    if market_returns is None:
        long["excess_ret"] = long["ret"].astype(float)
    else:
        mkt = pd.Series(market_returns).copy()
        if isinstance(mkt.index, pd.DatetimeIndex):
            mkt.index = pd.to_datetime(mkt.index)
        mkt = mkt.rename("market_ret").reset_index().rename(columns={"index": "trade_date"})
        mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
        long = long.merge(mkt, on="trade_date", how="left")
        long["excess_ret"] = long["ret"].astype(float) - long["market_ret"].fillna(0.0).astype(float)
    return long[["trade_date", "sector", "excess_ret"]]


def estimate_policy_lag(
    events: pd.DataFrame,
    sector_returns: pd.DataFrame,
    *,
    market_returns: pd.Series | None = None,
    config: TimeLagConfig | None = None,
) -> TimeLagResult:
    """Estimate the optimal post-announcement lag per policy theme.

    Parameters
    ----------
    events : DataFrame with columns ``event_id``, ``announced_at``,
        ``themes``, ``sectors_hint``, ``policy_strength``.
    sector_returns : Long-form (trade_date, sector, ret) OR wide
        (trade_date index, sector columns).
    market_returns : Optional benchmark daily-return series used to
        compute *excess* sector returns. When None, raw sector returns
        are used (less robust to market-wide policy days).
    config : Tuning thresholds.
    """
    cfg = config or TimeLagConfig()
    exploded = _explode_sector_themes(events)
    if exploded.empty:
        return TimeLagResult(
            lag_curves=pd.DataFrame(columns=["theme", "lag_k", "mean_excess", "n_events"]),
            best_lag={},
        )
    exploded["announced_at"] = pd.to_datetime(exploded["announced_at"])

    if sector_returns is None or len(sector_returns) == 0:
        return TimeLagResult(
            lag_curves=pd.DataFrame(columns=["theme", "lag_k", "mean_excess", "n_events"]),
            best_lag={},
        )
    panel = _sector_excess_panel(sector_returns, market_returns)
    if panel.empty:
        return TimeLagResult(
            lag_curves=pd.DataFrame(columns=["theme", "lag_k", "mean_excess", "n_events"]),
            best_lag={},
        )
    # Long → fast lookup map keyed by (date, sector)
    panel_idx = panel.set_index(["trade_date", "sector"]).sort_index()

    # Use *per-day* excess (not cumulative) so the lag profile has a real
    # peak at the impact day rather than monotonically saturating.
    rows: list[dict] = []
    for _, ev in exploded.iterrows():
        announce_dt = ev["announced_at"]
        sector = ev["sector"]
        strength = float(ev.get("policy_strength") or 1.0)
        target_dates = pd.bdate_range(
            announce_dt + pd.Timedelta(days=1),
            announce_dt + pd.Timedelta(days=cfg.max_lag_days * 2),
        )[: cfg.max_lag_days]
        for k, dt in enumerate(target_dates, start=1):
            try:
                r = float(panel_idx.loc[(dt, sector), "excess_ret"])
            except KeyError:
                continue
            if not np.isfinite(r):
                continue
            rows.append(
                {
                    "event_id": ev["event_id"],
                    "theme": ev["theme"],
                    "sector": sector,
                    "lag_k": int(k),
                    "cum_excess": float(r),  # named for back-compat; actually per-day excess
                    "policy_strength": strength,
                }
            )

    per_event = pd.DataFrame(rows)
    if per_event.empty:
        return TimeLagResult(
            lag_curves=pd.DataFrame(columns=["theme", "lag_k", "mean_excess", "n_events"]),
            best_lag={},
            per_event_excess=per_event,
        )

    # Per-theme weighted curve
    if cfg.apply_policy_strength:
        per_event["_w"] = per_event["policy_strength"]
    else:
        per_event["_w"] = 1.0

    agg = (
        per_event.assign(weighted=lambda x: x["cum_excess"] * x["_w"])
        .groupby(["theme", "lag_k"])
        .agg(
            sum_weighted=("weighted", "sum"),
            sum_weight=("_w", "sum"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    agg["mean_excess"] = agg["sum_weighted"] / agg["sum_weight"].replace(0, np.nan)

    if cfg.apply_event_count_weight:
        # Shrink themes with few events toward zero so they don't dominate
        # the argmax purely by noise.
        agg["mean_excess"] = (
            agg["mean_excess"]
            * np.minimum(1.0, agg["n_events"] / 5.0)
        )

    agg = agg[agg["n_events"] >= cfg.min_events_per_theme].copy()
    if agg.empty:
        return TimeLagResult(
            lag_curves=pd.DataFrame(columns=["theme", "lag_k", "mean_excess", "n_events"]),
            best_lag={},
            per_event_excess=per_event.drop(columns=["_w"], errors="ignore"),
        )

    lag_curves = agg[["theme", "lag_k", "mean_excess", "n_events"]].sort_values(
        ["theme", "lag_k"]
    ).reset_index(drop=True)

    best_lag: dict[str, tuple[int, float]] = {}
    for theme, grp in lag_curves.groupby("theme"):
        idx_peak = grp["mean_excess"].abs().idxmax()
        row = lag_curves.loc[idx_peak]
        best_lag[str(theme)] = (int(row["lag_k"]), float(row["mean_excess"]))

    return TimeLagResult(
        lag_curves=lag_curves,
        best_lag=best_lag,
        per_event_excess=per_event.drop(columns=["_w"], errors="ignore"),
    )


# ---------------------------------------------------------------------------
# Feature application
# ---------------------------------------------------------------------------

def apply_policy_lag_features(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lag_table: dict[str, tuple[int, float]] | None = None,
    default_lag: int = 5,
    themes_to_include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Augment a training panel with policy-event feature columns.

    Adds one column per theme: ``policy_signal_{theme}``.  On any
    (trade_date, symbol) row, the signal is the sum of
    ``policy_strength`` of all events of that theme whose
    ``announced_at + best_lag_days <= trade_date`` and whose
    sectors_hint contains the symbol's industry (when available) OR
    "_ALL_".  Past-anchored merge_asof keeps the join PIT-safe.

    Parameters
    ----------
    panel : DataFrame with ``trade_date`` and ``symbol`` (and
        optionally ``sector_level_1``).
    events : Normalised policy event frame.
    lag_table : Output of :func:`estimate_policy_lag`'s ``best_lag``;
        a dict theme → (lag, peak_excess).  When omitted, ``default_lag``
        is used for every theme.
    """
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()
    if events is None or events.empty:
        return panel.copy()

    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    if "symbol" in work.columns:
        work["symbol"] = work["symbol"].astype(str)

    sector_col = "sector_level_1" if "sector_level_1" in work.columns else None

    ev = events.copy()
    ev["announced_at"] = pd.to_datetime(ev["announced_at"])
    ev = ev.explode("themes").rename(columns={"themes": "theme"})
    ev = ev[ev["theme"].notna() & (ev["theme"] != "")]

    target_themes = (
        set(themes_to_include) if themes_to_include is not None else set(ev["theme"].unique())
    )

    for theme in sorted(target_themes):
        ev_t = ev[ev["theme"] == theme].copy()
        if ev_t.empty:
            work[f"policy_signal_{theme}"] = 0.0
            continue
        lag_days = int(
            (lag_table or {}).get(theme, (default_lag, 0.0))[0]
        )
        ev_t["effective_signal_date"] = ev_t["announced_at"] + pd.tseries.offsets.BDay(lag_days)
        ev_t["sectors_hint"] = ev_t["sectors_hint"].apply(
            lambda v: list(v) if isinstance(v, (list, tuple)) and v else ["_ALL_"]
        )
        ev_t = ev_t.explode("sectors_hint").rename(columns={"sectors_hint": "sector_hint"})
        ev_t["sector_hint"] = ev_t["sector_hint"].astype(str)

        # Build a per-(effective_date, sector) cumulative strength frame
        agg = (
            ev_t.groupby(["effective_signal_date", "sector_hint"])["policy_strength"]
            .sum()
            .reset_index()
            .rename(columns={"effective_signal_date": "trade_date"})
        )

        # Sector-keyed merge_asof backward to attach the most recent
        # policy signal for that sector
        cum_per_sector = (
            agg.sort_values(["sector_hint", "trade_date"])
            .groupby("sector_hint")
            .apply(lambda g: g.assign(cum_strength=g["policy_strength"].cumsum()))
            .reset_index(drop=True)
        )

        signal = pd.Series(0.0, index=work.index)
        for sector, sub in cum_per_sector.groupby("sector_hint"):
            sub = sub.sort_values("trade_date")[["trade_date", "cum_strength"]]
            if sector == "_ALL_":
                mask = pd.Series(True, index=work.index)
            elif sector_col is None:
                continue
            else:
                mask = work[sector_col].astype(str) == sector
            if not mask.any():
                continue
            left = work.loc[mask, ["trade_date"]].sort_values("trade_date").copy()
            left["__orig_index"] = left.index
            merged = pd.merge_asof(
                left, sub, on="trade_date", direction="backward"
            )
            merged["cum_strength"] = merged["cum_strength"].fillna(0.0)
            signal.loc[merged["__orig_index"].values] = signal.loc[
                merged["__orig_index"].values
            ].values + merged["cum_strength"].values

        work[f"policy_signal_{theme}"] = signal.astype(float)

    return work
