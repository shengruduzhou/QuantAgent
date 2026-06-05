"""Turnover-controlled alpha portfolio constructor.

Converts per-date model alpha scores into target weights with explicit
turnover control — empirically the single biggest lever for harvesting a
multi-day cross-sectional signal *net of A-share trading costs*.

Why this module exists
----------------------
The v8 deep pipeline built target weights with a naive ``top_k`` equal
weight, re-emitted **every trading day**. A 20-day-horizon signal churns
40-80 % of the book per day when rebalanced daily, so realistic costs
(commission + stamp + slippage) bleed the entire edge: on the v8 OOS the
naive long book *under*performed a costless equal-weight all-A benchmark
by ~5 %/yr despite a strong rank-IC of ~0.12.

Holding the signal for its horizon instead — emitting a target row only on
a rebalance cadence (``rebalance_interval`` trading days) and letting the
execution simulator hold between — collapses turnover to ~4 % and recovers
**+11 to +14 %/yr excess** vs the same benchmark in *both* bull and bear
OOS windows. See ``docs/v8_portfolio_construction.md`` for the full grid.

Two structures
--------------
* ``long`` — top ``book_fraction`` of the cross-section (decile/quintile),
  equal- or rank-weighted, per-name capped. Wins trending / bull regimes
  (v8 bull OOS: Sharpe 2.1, +13.6 % excess).
* ``long_short`` — +top / −bottom book, market-neutral (net ≈ 0, gross 1).
  Wins choppy / bear regimes (v8 bear OOS: Sharpe ~3, max-DD ~4 %).

``gross_scale`` (scalar) or ``gross_scale_by_date`` (per-date Series) is
the hook for market-regime exposure scaling (牛市满仓 / 熊市空仓): multiply
the whole book by the regime's scale so the constructor composes with the
existing regime detector without re-implementing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaPortfolioConfig:
    """Configuration for :func:`build_alpha_portfolio`.

    * ``book_fraction`` — fraction of the daily cross-section to hold long
      (0.10 = top decile, 0.20 = top quintile).
    * ``weighting`` — ``"equal"`` or ``"rank"`` (linear rank weight, capped).
    * ``max_name_weight`` — hard per-name cap after normalisation.
    * ``rebalance_interval`` — emit a target row every N trading days; the
      execution simulator holds positions between rebalances, which is what
      keeps turnover low. ``1`` reproduces daily rebalancing.
    * ``long_short`` — build a market-neutral +top/−bottom book.
    * ``gross_scale`` — scalar multiplier on the whole book (regime hook).
    * ``min_names_per_date`` — skip thin cross-sections.
    """

    book_fraction: float = 0.10
    weighting: str = "equal"
    max_name_weight: float = 0.05
    rebalance_interval: int = 20
    long_short: bool = False
    gross_scale: float = 1.0
    min_names_per_date: int = 50
    # liquidity floor: drop names whose trailing avg amount (yuan) is below
    # this before ranking. 0 = no filter (down-cap, includes micro-caps);
    # ~5e7 = institutional liquidity. Requires the ``liquidity`` arg.
    min_avg_amount_yuan: float = 0.0


def _waterfill_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Apply a hard per-name cap, redistributing excess to uncapped names.

    Naive ``min(w, cap)`` followed by renormalisation re-breaks the cap
    (dividing by a sub-unit sum inflates the capped names back over it).
    Water-filling caps the offenders and pushes the freed mass onto the
    still-uncapped names until none exceed ``cap``. When the cap is so tight
    that ``cap * len(w) < 1`` (impossible to stay fully invested under it),
    the leg is left intentionally under-invested at the cap rather than
    silently violating it.
    """
    if cap is None or cap <= 0:
        return w
    w = w.astype(float).copy()
    if cap * len(w) <= 1.0 + 1e-12:
        return np.minimum(w, cap)
    for _ in range(1000):
        over = w > cap + 1e-15
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap - 1e-15
        if not under.any():
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w


def _date_weights(scores: pd.Series, cfg: AlphaPortfolioConfig, scale: float) -> dict[str, float]:
    """Target weights for a single date from that date's alpha scores."""
    scores = scores.dropna()
    n = len(scores)
    if n == 0:
        return {}
    k = max(1, int(round(n * cfg.book_fraction)))
    ordered = scores.sort_values(ascending=False)
    longs = list(ordered.index[:k])
    if cfg.weighting == "rank":
        raw = np.arange(k, 0, -1, dtype=float)
        w = raw / raw.sum()
    elif cfg.weighting == "equal":
        w = np.full(k, 1.0 / k)
    else:
        raise ValueError(f"unknown weighting: {cfg.weighting}")
    # hard per-name cap with excess redistributed (keeps the long leg at
    # gross 1.0 whenever the cap is feasible).
    w = _waterfill_cap(w, cfg.max_name_weight)

    out: dict[str, float] = {}
    if cfg.long_short:
        shorts = list(ordered.index[-k:])
        for s, wi in zip(longs, w):
            out[s] = out.get(s, 0.0) + 0.5 * float(wi)
        sw = 0.5 / k
        for s in shorts:
            out[s] = out.get(s, 0.0) - sw
    else:
        for s, wi in zip(longs, w):
            out[s] = float(wi)
    if scale != 1.0:
        out = {s: v * scale for s, v in out.items()}
    return out


def build_alpha_portfolio(
    predictions: pd.DataFrame,
    *,
    config: AlphaPortfolioConfig | None = None,
    score_column: str = "alpha_score",
    gross_scale_by_date: pd.Series | None = None,
    liquidity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a turnover-controlled wide ``target_weights`` frame.

    Parameters
    ----------
    predictions
        Long frame with ``trade_date``, ``symbol`` and ``score_column``.
    config
        :class:`AlphaPortfolioConfig`. Defaults to top-decile, equal-weight,
        20-day rebalance — the OOS-best net-of-cost variant.
    score_column
        Column holding the per-date cross-sectional alpha score
        (``alpha_score`` for raw horizons, ``composite_score`` for the
        blended ensemble).
    gross_scale_by_date
        Optional per-date exposure multiplier (regime scaling). Overrides
        ``config.gross_scale`` for dates present in the Series.

    Returns
    -------
    DataFrame indexed by ``trade_date`` (rebalance dates only), columns are
    symbols, values are target weights. Rows are emitted only on the
    rebalance cadence so the downstream simulator holds between them.
    """
    cfg = config or AlphaPortfolioConfig()
    if score_column not in predictions.columns:
        raise KeyError(f"predictions missing score column {score_column!r}")
    df = predictions[["trade_date", "symbol", score_column]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "symbol", score_column])

    # Optional liquidity floor: PIT-join trailing avg amount and drop names
    # below the floor before ranking (down-cap vs institutional-liquidity).
    if cfg.min_avg_amount_yuan > 0.0 and liquidity is not None and not liquidity.empty:
        liq = liquidity[["trade_date", "symbol", "avg_amount"]].copy()
        liq["trade_date"] = pd.to_datetime(liq["trade_date"], errors="coerce")
        df = df.merge(liq, on=["trade_date", "symbol"], how="left")
        df = df[df["avg_amount"].fillna(0.0) >= cfg.min_avg_amount_yuan]
        df = df.drop(columns=["avg_amount"])

    dates = sorted(df["trade_date"].unique())
    interval = max(1, int(cfg.rebalance_interval))

    scale_map: dict = {}
    if gross_scale_by_date is not None:
        s = gross_scale_by_date.copy()
        s.index = pd.to_datetime(s.index, errors="coerce")
        scale_map = s.to_dict()

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for i, d in enumerate(dates):
        if i % interval != 0:
            continue
        g = df[df["trade_date"] == d]
        if len(g) < cfg.min_names_per_date:
            continue
        scale = float(scale_map.get(d, cfg.gross_scale))
        if scale <= 0.0:
            rows[d] = {}  # 空仓 — hold cash this rebalance
            continue
        rows[d] = _date_weights(g.set_index("symbol")[score_column], cfg, scale)

    if not rows:
        return pd.DataFrame()
    # reindex onto all rebalance dates so all-cash (regime crisis) rows are
    # not dropped by from_dict when their weight dict is empty.
    wide = pd.DataFrame.from_dict(rows, orient="index")
    wide = wide.reindex(sorted(rows.keys())).fillna(0.0)
    wide.index.name = "trade_date"
    return wide


__all__ = ["AlphaPortfolioConfig", "build_alpha_portfolio"]
