"""Regime-conditional LLM-side overlay.

The Phase-2 finding: a flat evidence overlay (policy / sentiment / fundamental /
sector / dip − old_dealer) on top of the raw factor alpha *helps* in
bear/sideways markets but *drags* in a euphoric bull. The fix is to make the
evidence weight ``lambda`` a function of the market regime.

Design (deliberately low-parameter to avoid overfitting):

    score_t,i = z(alpha_t,i) + lambda(regime_t) * evidence_composite_t,i

* ``z(alpha)`` is the per-day cross-sectional z-score of the factor model.
* ``evidence_composite`` is a *fixed* weighted z-blend of the evidence signals.
* ``lambda(regime)`` is ONE scalar per regime bucket (bull/sideways/bear),
  fit on a grid that **includes 0** — so in any regime the conditional overlay
  can fall back to the pure factor ranking and is therefore never worse than
  the baseline in-sample. In strong bull lambda→0 (pure factor); in
  bear/sideways lambda ramps up (lean on evidence).

Regime labels reuse the project's :func:`quantagent.quant_math.regime.detect_regime`
on lookahead-safe trailing benchmark features, collapsed to 3 buckets.

This module is pure (DataFrame in / Series out); all I/O lives in the driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.quant_math.regime import MarketRegime, RegimeThresholds, detect_regime

REGIME_BUCKETS: tuple[str, ...] = ("bull", "sideways", "bear")

# Evidence-composite weights (relative; lambda scales the whole composite).
# lambda=1 reproduces the flat Phase-2 overlay; lambda=0 = pure factor.
DEFAULT_EVIDENCE_WEIGHTS: dict[str, float] = {
    "core_policy_score": 0.18,
    "core_sentiment_score": 0.09,
    "fundamental_quality_score": 0.27,
    "sector_resonance_score": 0.22,
    "dip_buy_flow_score": 0.09,
    "old_dealer_risk_score": -0.22,
}

DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

# detect_regime -> 3-bucket collapse (risk-off states fold into "bear").
_BUCKET_OF_REGIME: dict[MarketRegime, str] = {
    MarketRegime.BULL_TREND: "bull",
    MarketRegime.RANGE_BOUND: "sideways",
    MarketRegime.HIGH_VOLATILITY: "bear",
    MarketRegime.BEAR_TREND: "bear",
    MarketRegime.LIQUIDITY_CRISIS: "bear",
}

TRADING_DAYS_PER_YEAR = 244


@dataclass(frozen=True)
class RegimeOverlayConfig:
    alpha_col: str = "alpha_score"
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    fwd_ret_col: str = "fwd_ret"
    top_k: int = 50
    trend_window: int = 20
    vol_window: int = 20
    drawdown_window: int = 60
    evidence_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EVIDENCE_WEIGHTS))
    thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)


# ---------------------------------------------------------------------------
# Regime labelling (lookahead-safe: every feature uses only past closes)
# ---------------------------------------------------------------------------

def compute_regime_labels(benchmark_close: pd.Series, config: RegimeOverlayConfig | None = None) -> pd.Series:
    """Label each date bull/sideways/bear from trailing benchmark features."""
    cfg = config or RegimeOverlayConfig()
    px = benchmark_close.sort_index()
    ret = px.pct_change()
    market_trend = px.pct_change(cfg.trend_window)            # trailing N-day return
    market_vol = ret.rolling(cfg.vol_window).std()            # trailing daily vol
    roll_max = px.rolling(cfg.drawdown_window, min_periods=1).max()
    drawdown = px / roll_max - 1.0
    feat = pd.DataFrame({
        "market_trend": market_trend,
        "market_vol": market_vol,
        "drawdown": drawdown,
        "liquidity_change": 0.0,
    }).dropna()
    labels = feat.apply(lambda r: _BUCKET_OF_REGIME[detect_regime(r, cfg.thresholds)], axis=1)
    labels.name = "regime"
    return labels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _xs_z(df: pd.DataFrame, col: str, date_col: str) -> pd.Series:
    def z(g: pd.Series) -> pd.Series:
        s = g.std()
        return (g - g.mean()) / s if s and s > 1e-12 else g * 0.0
    return df.groupby(date_col)[col].transform(z)


def evidence_composite(df: pd.DataFrame, config: RegimeOverlayConfig | None = None) -> pd.Series:
    """Fixed weighted z-blend of evidence signals (one value per row)."""
    cfg = config or RegimeOverlayConfig()
    comp = pd.Series(0.0, index=df.index)
    for col, w in cfg.evidence_weights.items():
        if col in df.columns:
            comp = comp + float(w) * _xs_z(df, col, cfg.date_col)
    return comp


def conditional_score(
    df: pd.DataFrame,
    regime_labels: pd.Series,
    lambdas: dict[str, float],
    config: RegimeOverlayConfig | None = None,
    *,
    evidence: pd.Series | None = None,
) -> pd.Series:
    """score = z(alpha) + lambda(regime_t) * evidence_composite."""
    cfg = config or RegimeOverlayConfig()
    z_alpha = _xs_z(df, cfg.alpha_col, cfg.date_col)
    comp = evidence if evidence is not None else evidence_composite(df, cfg)
    lam = df[cfg.date_col].map(regime_labels).map(lambda r: float(lambdas.get(str(r), 0.0))).fillna(0.0)
    return z_alpha + lam.values * comp


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def portfolio_daily_return(df: pd.DataFrame, score: pd.Series, config: RegimeOverlayConfig | None = None) -> pd.Series:
    """Equal-weight daily return of the top-K by `score` (uses fwd_ret)."""
    cfg = config or RegimeOverlayConfig()
    d = pd.DataFrame({
        cfg.date_col: df[cfg.date_col].values,
        "score": np.asarray(score),
        "fwd_ret": df[cfg.fwd_ret_col].values,
    }).dropna(subset=["score", "fwd_ret"])
    d = d.sort_values([cfg.date_col, "score"], ascending=[True, False])
    d["rank"] = d.groupby(cfg.date_col).cumcount()
    top = d[d["rank"] < cfg.top_k]
    return top.groupby(cfg.date_col)["fwd_ret"].mean()


def annualized_return(daily: pd.Series) -> float:
    if daily is None or len(daily) == 0:
        return 0.0
    return float((1.0 + daily.mean()) ** TRADING_DAYS_PER_YEAR - 1.0)


def per_regime_excess(
    daily: pd.Series, bench_daily: pd.Series, regime_labels: pd.Series
) -> dict[str, float]:
    """Annualized excess (strategy − benchmark) within each regime bucket + ALL."""
    out: dict[str, float] = {}
    common = daily.index.intersection(bench_daily.index)
    reg = regime_labels.reindex(common)
    for bucket in (*REGIME_BUCKETS, "ALL"):
        idx = common if bucket == "ALL" else common[reg == bucket]
        if len(idx) == 0:
            out[bucket] = 0.0
            continue
        out[bucket] = annualized_return(daily.loc[idx]) - annualized_return(bench_daily.loc[idx])
    return out


def lambda_excess_curves(
    df: pd.DataFrame,
    regime_labels: pd.Series,
    bench_daily: pd.Series,
    config: RegimeOverlayConfig | None = None,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> dict[str, dict[float, float]]:
    """Per-regime excess for each lambda in one window. ``excess[bucket][lambda]``.

    Evidence composite is regime-independent, so a single uniform-lambda run
    yields each regime's realized return at that lambda (slice by regime).
    """
    cfg = config or RegimeOverlayConfig()
    comp = evidence_composite(df, cfg)
    z_alpha = _xs_z(df, cfg.alpha_col, cfg.date_col)
    excess: dict[str, dict[float, float]] = {b: {} for b in REGIME_BUCKETS}
    for lam in lambda_grid:
        daily = portfolio_daily_return(df, z_alpha + lam * comp, cfg)
        ex = per_regime_excess(daily, bench_daily, regime_labels)
        for b in REGIME_BUCKETS:
            excess[b][lam] = ex[b]
    return excess


def fit_regime_lambdas(
    df: pd.DataFrame,
    regime_labels: pd.Series,
    bench_daily: pd.Series,
    config: RegimeOverlayConfig | None = None,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> dict[str, float]:
    """In-sample fit: per regime, the lambda maximizing that regime's excess.

    Because the grid includes 0.0, this is >= baseline in every regime
    in-sample (but may overfit — see :func:`fit_regime_lambdas_cv`).
    """
    excess = lambda_excess_curves(df, regime_labels, bench_daily, config, lambda_grid)
    return {b: (max(excess[b], key=excess[b].get) if excess[b] else 0.0) for b in REGIME_BUCKETS}


def fit_regime_lambdas_cv(
    windows: list[tuple[pd.DataFrame, pd.Series, pd.Series]],
    config: RegimeOverlayConfig | None = None,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    aggregate: str = "min",
) -> dict[str, float]:
    """Robust cross-validated fit across multiple windows/eras.

    For each regime bucket, pick the lambda maximizing the *aggregate*
    (default ``min`` = worst-case) of that bucket's excess across all windows.
    Worst-case selection resists overfitting a single era: a lambda only wins
    if it helps (or at least does not hurt) in every window. With 0.0 in the
    grid the result never makes a regime materially worse than baseline in any
    window.

    ``windows`` is a list of ``(df, regime_labels, bench_daily)`` tuples.

    The objective is the worst-case **improvement over baseline** (lambda=0)
    per regime, NOT raw excess — otherwise the lowest-magnitude era dominates
    and the CV degenerates to an in-sample fit. A lambda only wins if it
    helps (vs pure factor) in every window; with 0.0 in the grid the robust
    choice falls back to baseline when no lambda generalizes.
    """
    curves = [lambda_excess_curves(df, reg, bench, config, lambda_grid) for df, reg, bench in windows]
    agg = min if aggregate == "min" else (np.mean if aggregate == "mean" else max)
    best: dict[str, float] = {}
    for b in REGIME_BUCKETS:
        # improvement over baseline within each window, then worst-case across windows
        scored = {}
        for lam in lambda_grid:
            improvements = [c[b].get(lam, 0.0) - c[b].get(0.0, 0.0) for c in curves]
            scored[lam] = agg(improvements)
        # prefer the smallest lambda among ties (shrink toward baseline)
        best_val = max(scored.values()) if scored else 0.0
        best[b] = min((lam for lam, v in scored.items() if v >= best_val - 1e-12), default=0.0)
    return best
