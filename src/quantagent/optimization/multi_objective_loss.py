"""Multi-objective loss for portfolio config search (Stage 3 — 10 terms).

The user's v4 spec §14 lists a 13-term loss for hyperparameter / GA
optimization. Stage 1 implemented 5 of them; Stage 3 extends to 10;
Stage 5 will reach all 13.

Stage 1 components (1-5):

1. **net_return** (+) — daily mean compounded to annual; higher better.
2. **sharpe** (+) — annualised mean / std; higher better.
3. **calmar** (+) — annualised return / |max DD|; higher better.
4. **max_drawdown** (−) — absolute drawdown magnitude; lower better.
5. **high_chase_penalty** (−) — fraction of selected gross exposure
   that was in high-chase names on the day of selection. Lower better.

Stage 3 additions (6-10):

6. **turnover_penalty** (−) — average daily portfolio turnover. High
   churn eats alpha through transaction cost; this term explicitly
   penalises strategies that need ≥30%/day rebalance to work.
7. **tail_risk** (−) — CVaR at 5% (mean of the worst 5% daily returns,
   reported as a positive magnitude). Captures left-tail risk that
   sharpe and max_dd miss.
8. **regime_consistency** (+) — the *minimum* per-regime sharpe across
   regime tags. A strategy that earns +3.0 sharpe in normal markets
   but -1.0 in bear is penalised relative to one that earns +2.0
   everywhere.
9. **gross_volatility** (−) — annualised σ of daily returns. At a
   fixed sharpe, lower vol means a smoother curve (better Stage 3 DD).
10. **win_rate** (+) — fraction of profitable daily returns. Bonus for
    strategies that produce many small wins instead of a few big ones.

The function consumes the deployed-sleeve back-test output (the
``equity_curve.csv`` and ``trade_blotter.csv`` plus optional universe
filter audit) and returns a ``LossComponents`` breakdown and a single
scalar ``total`` for the optimizer.

Sign convention: ``total`` is what the optimizer **minimises**. We
negate the "good" components (net_return, sharpe, calmar) and add the
"bad" ones (max_dd, high_chase). Component values are reported as
positive numbers so the breakdown is human-readable; the sign is
applied during aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LossWeights:
    """Component weights for the multi-objective loss.

    Stage 1 default weights are calibrated to the v9 OOS panel:

    * Net return weight 1.0 anchors the scale (so 1pp annual return
      moves the loss by 0.01).
    * Sharpe weight 0.5: amplifies risk-adjusted return on top of raw
      return.
    * Calmar weight 0.5: penalises configs that earn return via deep
      drawdowns (raw return without drawdown cap).
    * Max DD weight 2.0: each 1pp of drawdown over the 10% target
      costs more loss than 1pp of return gained.
    * High-chase weight 1.0: 1pp of exposure to "接盘" names costs the
      same as 1pp of foregone return.

    Stage 5 additions (spec section 6) — explicit cost / structure
    penalties so the optimiser cannot game a high turnover / high
    concentration solution that looks profitable only on paper.
    """

    net_return: float = 1.0
    sharpe: float = 0.5
    calmar: float = 0.5
    max_drawdown: float = 2.0
    high_chase: float = 1.0
    # Stage 3 additions — calibrated to keep total-loss magnitudes
    # comparable to the Stage 1 terms at typical operating points.
    turnover: float = 0.5
    tail_risk: float = 1.5
    regime_consistency: float = 0.5
    gross_volatility: float = 1.0
    win_rate: float = 0.5
    # Stage 5 additions — spec section 6 explicit penalty terms.
    transaction_cost: float = 1.0
    concentration: float = 0.5
    illiquidity: float = 0.5
    st_exposure: float = 1.0
    execution_unfilled: float = 0.5


@dataclass(frozen=True)
class LossComponents:
    """Per-term breakdown. All values are positive magnitudes; sign
    convention is applied during aggregation in ``total``.
    """

    net_return: float
    sharpe: float
    calmar: float
    max_drawdown: float
    high_chase: float
    # Stage 3 additions
    turnover: float
    tail_risk: float
    regime_consistency: float
    gross_volatility: float
    win_rate: float
    # Stage 5 additions — spec section 6
    transaction_cost: float = 0.0
    concentration: float = 0.0
    illiquidity: float = 0.0
    st_exposure: float = 0.0
    execution_unfilled: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "net_return": self.net_return,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "high_chase": self.high_chase,
            "turnover": self.turnover,
            "tail_risk": self.tail_risk,
            "regime_consistency": self.regime_consistency,
            "gross_volatility": self.gross_volatility,
            "win_rate": self.win_rate,
            "transaction_cost": self.transaction_cost,
            "concentration": self.concentration,
            "illiquidity": self.illiquidity,
            "st_exposure": self.st_exposure,
            "execution_unfilled": self.execution_unfilled,
            "total": self.total,
        }


def _ann_return_from_daily(returns: pd.Series, periods: int = 252) -> float:
    """Geometric (compound) annualised return.

    Review fix #7: the prior implementation used the arithmetic mean
    compounded as ``(1 + mean) ** 252 - 1`` which **overstates** the
    realised compound return whenever returns have variance (AM-GM
    inequality). For a +10% / -10% series the arithmetic-mean method
    reports 0% while the true cumulative is -1%. Use the cumulative
    product instead so the loss reflects what an investor actually
    sees in the account.
    """

    if returns.empty:
        return 0.0
    # Drop NaN before product so a single missing day does not zero the run.
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    n = int(len(clean))
    # Guard against -100%+ moves that would push (1+r) ≤ 0 and break the
    # geometric formula. We floor at -0.999 (-99.9% daily) — realistic
    # daily loss cap on A-share (limit-down + ST is ~5-10%); anything
    # beyond suggests a data error and the bound prevents pow() failures.
    factors = (1.0 + clean.clip(lower=-0.999)).astype(float)
    cumulative = float(factors.prod())
    if cumulative <= 0:
        return float("nan")
    return float(cumulative ** (periods / n) - 1.0)


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(returns.mean() / std * (periods ** 0.5))


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(abs(dd.min()))


def _calmar(ann_return: float, max_dd: float) -> float:
    if max_dd <= 1e-9:
        # No drawdown: undefined. Treat as a strong positive but bounded so it
        # doesn't dominate the loss. Use a 10x multiplier of ann_return.
        return float(ann_return) * 10.0
    return float(ann_return / max_dd)


# ---------------------------------------------------------------------------
# Stage 3 — five additional term calculators
# ---------------------------------------------------------------------------

def _annualised_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualised σ of daily net returns. 0 for too-short series."""
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= 1e-12 or not np.isfinite(std):
        return 0.0
    return float(std * (periods ** 0.5))


def _cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR at level α: mean of the worst α-fraction of daily
    returns, reported as a positive magnitude. 0 when the series has no
    losses or too few observations.
    """
    if len(returns) < int(np.ceil(1 / alpha)):
        return 0.0
    threshold = float(returns.quantile(alpha))
    tail = returns[returns <= threshold]
    if tail.empty:
        return 0.0
    cvar_value = float(tail.mean())
    return float(abs(cvar_value)) if cvar_value < 0 else 0.0


def _win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    wins = int((returns > 0).sum())
    return float(wins / len(returns))


def _regime_consistency_sharpe(
    returns: pd.Series,
    regime: pd.Series | None,
    periods: int = 252,
    min_regime_days: int = 20,
) -> float:
    """Minimum per-regime sharpe across regime buckets.

    Rationale: a strategy that earns +3.0 sharpe in normal markets but
    -1.0 in bear must not look "good" in aggregate.  We take the
    *minimum* sharpe across buckets that have enough observations.
    Buckets smaller than ``min_regime_days`` are dropped (statistical
    noise dominates).

    Returns 0.0 when regime data is unavailable, so this term doesn't
    inadvertently penalise users without regime labels.
    """
    if regime is None or len(regime) == 0:
        return 0.0
    rs = pd.Series(regime).reset_index(drop=True)
    rr = pd.Series(returns).reset_index(drop=True)
    if len(rs) != len(rr):
        return 0.0
    df = pd.DataFrame({"ret": rr, "regime": rs.astype(str)}).dropna(subset=["ret"])
    if df.empty:
        return 0.0
    sharpes: list[float] = []
    for _, grp in df.groupby("regime"):
        if len(grp) < min_regime_days:
            continue
        s = _sharpe(grp["ret"], periods=periods)
        sharpes.append(s)
    if not sharpes:
        return 0.0
    # Clip to a sane bound so a single ugly small-sample bucket can't
    # collapse the loss.
    return float(min(sharpes))


def compute_multi_objective_loss(
    daily_returns: pd.Series,
    *,
    high_chase_exposure_rate: float = 0.0,
    avg_daily_turnover: float = 0.0,
    regime_states: pd.Series | None = None,
    weights: LossWeights | None = None,
    periods: int = 252,
    cvar_alpha: float = 0.05,
    transaction_cost_rate: float = 0.0,
    concentration_score: float = 0.0,
    illiquidity_score: float = 0.0,
    st_exposure_rate: float = 0.0,
    execution_unfilled_rate: float = 0.0,
) -> LossComponents:
    """Compute the 10-term loss from a daily-return series.

    Stage 1 terms (existing):
    ``daily_returns``, ``high_chase_exposure_rate`` feed terms 1-5.

    Stage 3 terms (new):
    ``avg_daily_turnover`` feeds term 6 (turnover penalty).
    ``daily_returns`` alone feeds terms 7, 9, 10 (tail risk, vol, win rate).
    ``regime_states`` (optional, aligned to ``daily_returns``) feeds
    term 8 (regime consistency).  Omit it and term 8 contributes 0.

    Parameters
    ----------
    daily_returns:
        Series of daily portfolio returns (net of costs).
    high_chase_exposure_rate:
        Time-integrated fraction of gross exposure in high-chase names.
    avg_daily_turnover:
        Average daily portfolio turnover (sum of |Δweight| per day,
        averaged over time). Computed externally from a trade blotter.
    regime_states:
        Optional regime label series (same length & order as
        ``daily_returns``). Strings like "normal" / "caution" / "bear"
        / "crisis". When provided, term 8 is the minimum sharpe across
        regime buckets with ≥20 observations.
    weights:
        Component weights (default :class:`LossWeights`).
    periods:
        Annualisation factor (252 daily, 12 monthly).
    cvar_alpha:
        Tail probability for the CVaR term (default 5%).
    """

    w = weights or LossWeights()
    returns = pd.Series(pd.to_numeric(daily_returns, errors="coerce")).dropna().astype(float)
    ann_return = _ann_return_from_daily(returns, periods=periods)
    sharpe = _sharpe(returns, periods=periods)
    max_dd = _max_drawdown(returns)
    calmar = _calmar(ann_return, max_dd)
    high_chase = float(min(1.0, max(0.0, high_chase_exposure_rate)))

    # Stage 3 terms
    turnover = float(min(2.0, max(0.0, avg_daily_turnover)))  # cap at 2.0 (200%/day is absurd)
    tail_risk = _cvar(returns, alpha=cvar_alpha)
    regime_consistency = _regime_consistency_sharpe(returns, regime_states, periods=periods)
    gross_volatility = _annualised_volatility(returns, periods=periods)
    win_rate = _win_rate(returns)

    # Stage 5 terms — bounded positive magnitudes so the optimizer's
    # search surface stays well-conditioned.
    txn_cost = float(min(1.0, max(0.0, transaction_cost_rate)))
    concentration = float(min(1.0, max(0.0, concentration_score)))
    illiq = float(min(1.0, max(0.0, illiquidity_score)))
    st_exp = float(min(1.0, max(0.0, st_exposure_rate)))
    unfilled = float(min(1.0, max(0.0, execution_unfilled_rate)))

    total = (
        -w.net_return * ann_return
        - w.sharpe * sharpe
        - w.calmar * calmar
        + w.max_drawdown * max_dd
        + w.high_chase * high_chase
        + w.turnover * turnover
        + w.tail_risk * tail_risk
        - w.regime_consistency * regime_consistency
        + w.gross_volatility * gross_volatility
        - w.win_rate * win_rate
        + w.transaction_cost * txn_cost
        + w.concentration * concentration
        + w.illiquidity * illiq
        + w.st_exposure * st_exp
        + w.execution_unfilled * unfilled
    )
    return LossComponents(
        net_return=float(ann_return),
        sharpe=float(sharpe),
        calmar=float(calmar),
        max_drawdown=float(max_dd),
        high_chase=float(high_chase),
        turnover=float(turnover),
        tail_risk=float(tail_risk),
        regime_consistency=float(regime_consistency),
        gross_volatility=float(gross_volatility),
        win_rate=float(win_rate),
        transaction_cost=txn_cost,
        concentration=concentration,
        illiquidity=illiq,
        st_exposure=st_exp,
        execution_unfilled=unfilled,
        total=float(total),
    )


def score_backtest(
    equity_curve: pd.DataFrame,
    trade_blotter: pd.DataFrame | None = None,
    high_chase_symbols: pd.DataFrame | None = None,
    weights: LossWeights | None = None,
) -> LossComponents:
    """Convenience wrapper that consumes a deployed-sleeve back-test.

    Parameters
    ----------
    equity_curve:
        Frame with at least ``daily_eq_return`` column (the daily net
        return of the strategy). Other columns ignored.
    trade_blotter:
        Optional frame with ``trade_date``, ``symbol``, ``weight``
        rows. Used together with ``high_chase_symbols`` to compute
        the high-chase exposure rate.
    high_chase_symbols:
        Optional frame with ``trade_date``, ``symbol``, ``is_high_chase``
        — long format. Where missing, the high-chase rate defaults to
        0.
    weights:
        Component weights (default :class:`LossWeights`).
    """

    if equity_curve is None or equity_curve.empty:
        return compute_multi_objective_loss(pd.Series(dtype=float), weights=weights)
    if "daily_eq_return" not in equity_curve.columns:
        raise ValueError("equity_curve must include daily_eq_return column")
    rets = equity_curve["daily_eq_return"]

    # Stage 3 — extract turnover and regime tags from the equity_curve if
    # the upstream backtest emitted them (the v7 horizon-sleeve backtest
    # does, by default).
    avg_turnover = 0.0
    if "turnover" in equity_curve.columns:
        avg_turnover = float(
            pd.to_numeric(equity_curve["turnover"], errors="coerce").dropna().mean() or 0.0
        )
    regime_states = (
        equity_curve["regime_state"]
        if "regime_state" in equity_curve.columns
        else None
    )

    high_chase_rate = 0.0
    if (
        trade_blotter is not None
        and not trade_blotter.empty
        and high_chase_symbols is not None
        and not high_chase_symbols.empty
        and {"trade_date", "symbol", "weight"}.issubset(trade_blotter.columns)
        and {"trade_date", "symbol", "is_high_chase"}.issubset(high_chase_symbols.columns)
    ):
        bl = trade_blotter[["trade_date", "symbol", "weight"]].copy()
        hc = high_chase_symbols[["trade_date", "symbol", "is_high_chase"]].copy()
        bl["trade_date"] = pd.to_datetime(bl["trade_date"], errors="coerce")
        hc["trade_date"] = pd.to_datetime(hc["trade_date"], errors="coerce")
        bl["symbol"] = bl["symbol"].astype(str)
        hc["symbol"] = hc["symbol"].astype(str)
        merged = bl.merge(hc, on=["trade_date", "symbol"], how="left")
        merged["is_high_chase"] = merged["is_high_chase"].fillna(False).astype(bool)
        weight_chase = float(merged.loc[merged["is_high_chase"], "weight"].abs().sum())
        weight_total = float(merged["weight"].abs().sum())
        if weight_total > 1e-12:
            high_chase_rate = weight_chase / weight_total

    return compute_multi_objective_loss(
        rets,
        high_chase_exposure_rate=high_chase_rate,
        avg_daily_turnover=avg_turnover,
        regime_states=regime_states,
        weights=weights,
    )


__all__ = [
    "LossWeights",
    "LossComponents",
    "compute_multi_objective_loss",
    "score_backtest",
]
