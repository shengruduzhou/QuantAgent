from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_by_date(
    frame: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    date_column: str = "trade_date",
) -> pd.Series:
    """Daily Spearman rank correlation between predictions and realized returns."""
    subset = frame[[date_column, prediction_column, target_column]]
    return subset.groupby(date_column).apply(
        lambda x: x[prediction_column].rank().corr(x[target_column].rank()),
        include_groups=False,
    )


def information_coefficient_summary(rank_ic: pd.Series) -> dict[str, float]:
    clean = rank_ic.dropna()
    if clean.empty:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan, "icir": np.nan}
    std = clean.std(ddof=1)
    return {
        "rank_ic_mean": float(clean.mean()),
        "rank_ic_std": float(std),
        "icir": float(clean.mean() / std) if std and not np.isnan(std) else np.nan,
    }


def alpha_evaluation_summary(
    frame: pd.DataFrame,
    prediction_column: str = "alpha",
    target_column: str = "target",
    weight_column: str | None = None,
) -> dict[str, float]:
    rank_ic = rank_ic_by_date(frame, prediction_column, target_column)
    ic = frame[prediction_column].corr(frame[target_column])
    turnover = np.nan
    if weight_column and weight_column in frame.columns:
        wide = frame.pivot_table(index="trade_date", columns="symbol", values=weight_column, aggfunc="last").fillna(0.0)
        turnover = float(wide.diff().abs().sum(axis=1).mean())
    summary = information_coefficient_summary(rank_ic)
    summary.update(
        {
            "ic": float(ic) if ic == ic else np.nan,
            "turnover": turnover,
            "calibration_error": float((frame[prediction_column] - frame[target_column]).abs().mean()),
        }
    )
    return summary


def top_minus_bottom_spread(
    frame: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    quantile: float = 0.20,
    date_column: str = "trade_date",
) -> dict[str, float]:
    """Per-day top vs bottom quantile spread of realized returns."""
    if frame.empty:
        return {"top_minus_bottom_mean": float("nan"), "top_minus_bottom_std": float("nan"), "hit_rate": float("nan")}
    daily_spread: list[float] = []
    hits = 0
    total = 0
    for _, group in frame.groupby(date_column):
        if len(group) < 5:
            continue
        threshold_high = group[prediction_column].quantile(1 - quantile)
        threshold_low = group[prediction_column].quantile(quantile)
        top = group[group[prediction_column] >= threshold_high][target_column].mean()
        bot = group[group[prediction_column] <= threshold_low][target_column].mean()
        spread = top - bot
        if not np.isnan(spread):
            daily_spread.append(float(spread))
            total += 1
            if spread > 0:
                hits += 1
    if not daily_spread:
        return {"top_minus_bottom_mean": float("nan"), "top_minus_bottom_std": float("nan"), "hit_rate": float("nan")}
    spread_series = np.asarray(daily_spread, dtype=float)
    return {
        "top_minus_bottom_mean": float(spread_series.mean()),
        "top_minus_bottom_std": float(spread_series.std(ddof=1)) if len(spread_series) > 1 else 0.0,
        "hit_rate": float(hits / total) if total else float("nan"),
    }


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, mar: float = 0.0) -> float:
    """Annualised Sortino ratio. ``returns`` is a daily return series."""
    clean = pd.Series(returns).dropna().astype(float)
    if clean.empty:
        return float("nan")
    excess = clean - mar
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside ** 2).mean())) if not downside.empty else 0.0
    if downside_std == 0.0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    clean = pd.Series(returns).dropna().astype(float)
    if clean.empty:
        return float("nan")
    excess = clean - rf
    std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    if std == 0.0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns).dropna().astype(float)
    if clean.empty:
        return float("nan")
    nav = (1 + clean).cumprod()
    return float((nav / nav.cummax() - 1).min())


def capacity_proxy(
    frame: pd.DataFrame,
    weight_column: str,
    amount_column: str,
    participation_cap: float = 0.10,
    date_column: str = "trade_date",
) -> float:
    """Approximate strategy capacity (CNY) given a daily participation cap.

    For each day, capacity ≈ min over names of ``cap * amount / |weight|``.
    The function returns the median of daily capacities so isolated illiquid
    names do not dominate the estimate.
    """
    if frame.empty or weight_column not in frame.columns or amount_column not in frame.columns:
        return float("nan")
    daily_caps: list[float] = []
    for _, group in frame.groupby(date_column):
        weights = group[weight_column].astype(float).abs()
        amounts = group[amount_column].astype(float).abs()
        mask = weights > 0
        if not mask.any():
            continue
        capacities = (participation_cap * amounts[mask]) / weights[mask]
        capacities = capacities.replace([np.inf, -np.inf], np.nan).dropna()
        if not capacities.empty:
            daily_caps.append(float(capacities.min()))
    if not daily_caps:
        return float("nan")
    return float(np.median(daily_caps))


def compose_alpha_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    cost_bps: float = 0.0,
    weight_column: str | None = None,
    amount_column: str | None = None,
) -> dict[str, float]:
    """Single helper used by the V7 walk-forward aggregator."""
    rank_ic = rank_ic_by_date(frame, prediction_column, target_column)
    summary = information_coefficient_summary(rank_ic)
    spread = top_minus_bottom_spread(frame, prediction_column, target_column)
    daily_returns = (
        frame.groupby("trade_date").apply(
            lambda g: (g[prediction_column].rank(pct=True) - 0.5).pipe(
                lambda r: (r / max(1e-9, float(r.abs().sum())) * g[target_column]).sum()
            ),
            include_groups=False,
        )
        - cost_bps / 10_000.0
    )
    summary.update(spread)
    summary["sharpe"] = sharpe_ratio(daily_returns)
    summary["sortino"] = sortino_ratio(daily_returns)
    summary["max_drawdown"] = max_drawdown(daily_returns)
    summary["net_return"] = float(daily_returns.sum()) if not daily_returns.empty else float("nan")
    if weight_column and amount_column:
        summary["capacity_proxy"] = capacity_proxy(frame, weight_column, amount_column)
    return summary
