"""Stage 11 — beta-aware absolute-return decomposition.

The Stage 8-10 arc always asked "does X beat v8.9 on excess?" but never asked
the deeper question: *how much of any strategy's return is market beta vs real
selection alpha?* This module computes, for a strategy's daily returns against a
set of benchmarks (all-A eqw, CSI300/500/1000, a selected-concept basket):

  beta            OLS slope of strat-excess on bench-excess
  alpha_ann       annualised Jensen alpha (intercept) — the beta-adjusted edge
  r2 / corr       how much of the strategy is explained by that benchmark
  up/down capture episode capture vs the benchmark
  CAGR / excess   absolute & relative return
  Calmar/Sharpe/MaxDD/turnover

and classifies the strategy (user's rule): beta_strategy / research_signal /
production_candidate / window_artifact. Pure functions, returns in decimals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 244


def ann_return(daily: pd.Series) -> float:
    d = daily.dropna()
    n = len(d)
    return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def sharpe(daily: pd.Series, rf: float = 0.0) -> float:
    d = daily.dropna()
    if len(d) < 2 or d.std() < 1e-12:
        return float("nan")
    return float((d.mean() - rf / ANN) / d.std() * np.sqrt(ANN))


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    return float(abs((nav / nav.cummax() - 1).min()))


def beta_alpha(strat: pd.Series, bench: pd.Series, rf: float = 0.0) -> dict:
    """OLS strat_excess = alpha + beta*bench_excess (daily), alpha annualised."""
    idx = strat.dropna().index.intersection(bench.dropna().index)
    if len(idx) < 20:
        return {"beta": np.nan, "alpha_ann": np.nan, "r2": np.nan, "corr": np.nan}
    s = strat.reindex(idx).values - rf / ANN
    b = bench.reindex(idx).values - rf / ANN
    bvar = np.var(b)
    if bvar < 1e-18:
        return {"beta": np.nan, "alpha_ann": np.nan, "r2": np.nan, "corr": np.nan}
    beta = float(np.cov(s, b, ddof=0)[0, 1] / bvar)
    alpha_daily = float(s.mean() - beta * b.mean())
    resid = s - (alpha_daily + beta * b)
    r2 = float(1 - np.var(resid) / (np.var(s) + 1e-18))
    corr = float(np.corrcoef(s, b)[0, 1])
    # annualise alpha geometrically via daily intercept
    alpha_ann = float((1 + alpha_daily) ** ANN - 1)
    return {"beta": round(beta, 3), "alpha_ann": round(alpha_ann, 4),
            "r2": round(r2, 3), "corr": round(corr, 3)}


def capture(strat: pd.Series, bench: pd.Series) -> dict:
    idx = strat.dropna().index.intersection(bench.dropna().index)
    s, b = strat.reindex(idx), bench.reindex(idx)
    up, dn = b > 0, b < 0
    def cap(mask):
        if mask.sum() < 5:
            return np.nan
        bb = float((1 + b[mask]).prod() - 1)
        if abs(bb) < 1e-9:
            return np.nan
        return float(((1 + s[mask]).prod() - 1) / bb)
    return {"up_capture": round(cap(up), 3) if not np.isnan(cap(up)) else None,
            "down_capture": round(cap(dn), 3) if not np.isnan(cap(dn)) else None}


def full_panel(strat_daily: pd.Series, nav: pd.Series, benches: dict[str, pd.Series],
               *, turnover: float | None = None, primary: str = "all_a") -> dict:
    """Complete beta-aware metric panel for one strategy."""
    cagr = ann_return(strat_daily)
    dd = max_drawdown(nav)
    panel = {
        "cagr": round(cagr, 4),
        "maxdd": round(dd, 4),
        "calmar": round(cagr / dd, 3) if dd and dd > 1e-9 else None,
        "sharpe": round(sharpe(strat_daily), 3),
        "turnover": round(turnover, 4) if turnover is not None else None,
        "vol_ann": round(float(strat_daily.std() * np.sqrt(ANN)), 4),
    }
    for name, b in benches.items():
        ba = beta_alpha(strat_daily, b)
        idx = strat_daily.dropna().index.intersection(b.dropna().index)
        panel[f"excess_{name}"] = round(ann_return(strat_daily.reindex(idx)) - ann_return(b.reindex(idx)), 4)
        panel[f"beta_{name}"] = ba["beta"]
        panel[f"alpha_{name}"] = ba["alpha_ann"]
        panel[f"r2_{name}"] = ba["r2"]
        if name == primary:
            cp = capture(strat_daily, b)
            panel["up_capture"] = cp["up_capture"]
            panel["down_capture"] = cp["down_capture"]
    return panel


def classify_strategy(panel: dict, *, multi_window_ok: bool | None = None,
                      phase_std: float | None = None, max_weight: float | None = None,
                      primary: str = "all_a") -> tuple[str, list[str]]:
    """User's selection logic -> label + flags."""
    flags: list[str] = []
    cagr = panel.get("cagr") or 0.0
    alpha = panel.get(f"alpha_{primary}")
    beta = panel.get(f"beta_{primary}")
    calmar = panel.get("calmar") or 0.0
    dd = panel.get("maxdd") or 1.0
    if multi_window_ok is False:
        flags.append("window_artifact")
    if max_weight is not None and max_weight > 0.15:
        flags.append("high_concentration:verify_capacity")
    if phase_std is not None and panel.get(f"excess_{primary}") is not None \
            and phase_std >= abs(panel.get(f"excess_{primary}") or 0):
        flags.append("phase_unstable")

    a = alpha if alpha is not None else 0.0
    # primary categorisation
    if flags and "window_artifact" in flags:
        label = "window_artifact"
    elif cagr >= 0.15 and a > 0.03 and calmar >= 1.0 and dd <= 0.25:
        label = "production_candidate"
    elif cagr >= 0.15 and a <= 0.03:
        label = "beta_strategy"
    elif a > 0.05 and cagr < 0.15:
        label = "research_signal"
    elif a > 0.02 and cagr >= 0.10:
        label = "production_candidate"
    else:
        label = "beta_strategy" if (beta or 0) > 0.7 else "research_signal"
    return label, flags
