from __future__ import annotations

import numpy as np
import pandas as pd
from statistics import NormalDist

from quantagent.quant_math.risk_metrics import drawdown


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    if excess.empty or excess.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Bailey & Lopez de Prado 2012 PSR adjusted for skew and kurtosis."""
    clean = returns.dropna()
    n = len(clean)
    if n < 4:
        return np.nan
    std = clean.std(ddof=1)
    if std <= 1e-12:
        mean = float(clean.mean())
        return 1.0 if mean > sr_benchmark / periods_per_year else 0.0
    sr = sharpe_ratio(clean, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    with np.errstate(invalid="ignore"):
        skew_raw = clean.skew()
        kurt_raw = clean.kurt()
    skew = 0.0 if not np.isfinite(skew_raw) else float(skew_raw)
    kurt = 0.0 if not np.isfinite(kurt_raw) else float(kurt_raw)
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    denom = np.sqrt(max(1.0 - skew * sr + kurt / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(_normal_cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    candidate_sharpes: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Deflated SR: PSR threshold inflated by max-of-N selection bias."""
    clean = returns.dropna()
    n = len(clean)
    if n < 4 or candidate_sharpes.size < 2:
        return np.nan
    var_sr = float(np.var(candidate_sharpes, ddof=1))
    if var_sr <= 0:
        return np.nan
    n_trials = candidate_sharpes.size
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(var_sr) * (
        (1.0 - euler_mascheroni) * _normal_ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * _normal_ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(
        clean,
        sr_benchmark=float(expected_max) * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def newey_west_t_stat(series: pd.Series, max_lag: int | None = None) -> float:
    """Newey-West HAC t-stat for the mean of an autocorrelated series."""
    clean = series.dropna().to_numpy()
    n = len(clean)
    if n < 2:
        return np.nan
    if max_lag is None:
        max_lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mean = clean.mean()
    centered = clean - mean
    gamma0 = float(np.dot(centered, centered) / n)
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        nw_var += 2.0 * weight * gamma
    nw_var = max(nw_var, 1e-12)
    return float(mean / np.sqrt(nw_var / n))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if excess.empty or downside.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def max_drawdown(nav: pd.Series) -> float:
    dd = drawdown(nav.dropna())
    return float(dd.min()) if not dd.empty else np.nan


def calmar_ratio(nav: pd.Series, periods_per_year: int = 252) -> float:
    clean = nav.dropna()
    if len(clean) < 2:
        return np.nan
    years = len(clean) / periods_per_year
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0
    max_dd = abs(max_drawdown(clean))
    return float(cagr / max_dd) if max_dd > 0 else np.nan


def hit_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    return float((clean > 0).mean()) if not clean.empty else np.nan


def profit_factor(returns: pd.Series) -> float:
    clean = returns.dropna()
    gains = clean[clean > 0].sum()
    losses = -clean[clean < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).diff().abs().sum(axis=1)


def _normal_cdf(value: float) -> float:
    return NormalDist().cdf(value)


def _normal_ppf(value: float) -> float:
    return NormalDist().inv_cdf(value)


# --------------------------------------------------------------------------- #
# Backtest-overfitting diagnostics                                            #
# --------------------------------------------------------------------------- #


def probability_of_backtest_overfitting(
    is_oos_perf_matrix: np.ndarray | pd.DataFrame,
    n_partitions: int = 16,
    rng_seed: int = 0,
) -> float:
    """PBO via combinatorially-symmetric cross-validation (Bailey et al. 2014).

    Parameters
    ----------
    is_oos_perf_matrix
        Performance matrix with shape ``(T, N)`` where ``T`` is the number of
        equally-spaced time slices and ``N`` is the number of competing
        strategy configurations. Each cell is a performance score (e.g.
        Sharpe ratio) of strategy ``n`` on slice ``t``.
    n_partitions
        Number of slices to split the time axis into. The default of 16 yields
        ``C(16,8) = 12870`` symmetric IS/OOS combinations, the standard
        recommended in the original paper.
    rng_seed
        Used only when ``n_partitions`` does not evenly divide the row count
        (we randomise the trim positions for reproducibility).

    Returns
    -------
    float in ``[0, 1]`` — the probability that the configuration ranked best
    in-sample drops below the median out-of-sample. Values near 0.5 indicate
    no skill differentiation; values above 0.5 are evidence of overfitting.
    """
    from itertools import combinations

    arr = (
        is_oos_perf_matrix.to_numpy(dtype=float)
        if isinstance(is_oos_perf_matrix, pd.DataFrame)
        else np.asarray(is_oos_perf_matrix, dtype=float)
    )
    if arr.ndim != 2:
        raise ValueError("is_oos_perf_matrix must be 2-D (T_slices, N_strategies)")
    n_rows, n_strats = arr.shape
    if n_strats < 2:
        raise ValueError("PBO requires at least 2 competing strategies")
    if n_partitions < 4 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even and >= 4")
    if n_rows < n_partitions:
        raise ValueError(
            f"need at least {n_partitions} rows, got {n_rows}"
        )

    rng = np.random.default_rng(rng_seed)
    rows_per_chunk = n_rows // n_partitions
    if rows_per_chunk * n_partitions != n_rows:
        keep = rows_per_chunk * n_partitions
        offset = int(rng.integers(0, n_rows - keep + 1))
        arr = arr[offset : offset + keep]

    # Aggregate per chunk by mean — this is robust to within-chunk noise.
    chunks = arr.reshape(n_partitions, rows_per_chunk, n_strats).mean(axis=1)

    half = n_partitions // 2
    indices = list(range(n_partitions))
    logits: list[float] = []
    for is_idx in combinations(indices, half):
        is_set = set(is_idx)
        oos_idx = [i for i in indices if i not in is_set]
        is_score = chunks[list(is_idx)].mean(axis=0)
        oos_score = chunks[oos_idx].mean(axis=0)
        # IS argmax
        n_star = int(np.argmax(is_score))
        # OOS rank (1 = best, n_strats = worst); use average rank for ties.
        order = pd.Series(oos_score).rank(method="average", ascending=False)
        rank_n_star = float(order.iloc[n_star])
        # Map rank → relative rank in (0, 1) where 1 = best, 0 = worst.
        w = (n_strats - rank_n_star) / (n_strats - 1) if n_strats > 1 else 0.5
        # Logit; clip to avoid divide-by-zero at the boundaries.
        w = float(min(max(w, 1.0 / (n_strats + 1.0)), 1.0 - 1.0 / (n_strats + 1.0)))
        logits.append(float(np.log(w / (1.0 - w))))

    logits_arr = np.asarray(logits)
    return float((logits_arr < 0).mean())


def _politis_romano_block_length(series: np.ndarray) -> int:
    """Politis-White (2004) automatic block-length for stationary bootstrap."""
    n = len(series)
    if n < 8:
        return 1
    centered = series - series.mean()
    var = float(np.dot(centered, centered) / n)
    if var <= 1e-15:
        return 1
    # Use a generous lag cap; the kernel weighting damps long lags.
    max_lag = min(n - 1, int(np.floor(8.0 * (n / 100.0) ** (1.0 / 3.0))) + 1)
    auto = np.array(
        [float(np.dot(centered[k:], centered[:-k]) / n) / var for k in range(1, max_lag + 1)]
    )
    # Flat-top lag-window weights from Politis & White (2004).
    g_hat = 0.0
    sigma_hat = var
    for k in range(1, max_lag + 1):
        weight = 1.0 if abs(auto[k - 1]) >= 2.0 * np.sqrt(np.log10(n) / n) else 0.0
        if weight == 0.0:
            break
        g_hat += 2.0 * k * weight * auto[k - 1]
        sigma_hat += 2.0 * weight * auto[k - 1]
    if sigma_hat <= 1e-15:
        return 1
    b_opt = (2.0 * g_hat * g_hat / (sigma_hat * sigma_hat)) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    return int(max(1, min(n // 2, round(b_opt)))) if np.isfinite(b_opt) else 1


def _stationary_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index sample of size n."""
    if block_length <= 0:
        block_length = 1
    p_new = 1.0 / block_length
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    for i in range(1, n):
        if rng.random() < p_new:
            idx[i] = int(rng.integers(0, n))
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def spa_test(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Hansen (2005) Superior Predictive Ability test, consistent variant.

    Returns p-values for the null "no candidate beats benchmark". Smaller
    p-values are evidence at least one candidate strategy has genuine edge
    after correcting for multiple testing.

    Returns
    -------
    dict with keys:
      ``p_consistent``   — main SPA p-value
      ``p_lower``        — conservative lower-bound p-value
      ``p_upper``        — liberal upper-bound p-value
      ``best_strategy``  — column name of the strategy with the largest
                           standardised mean excess return
      ``test_statistic`` — observed standardised statistic value
      ``block_length``   — auto-selected (or user) block length used
    """
    rng = np.random.default_rng(rng_seed)
    aligned = candidate_returns.dropna(how="all").copy()
    bench = benchmark_returns.reindex(aligned.index).astype(float)
    mask = ~(aligned.isna().any(axis=1) | bench.isna())
    aligned = aligned.loc[mask]
    bench = bench.loc[mask]
    if len(aligned) < 8 or aligned.shape[1] == 0:
        return {
            "p_consistent": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "best_strategy": "",
            "test_statistic": float("nan"),
            "block_length": 1,
        }

    excess = aligned.sub(bench, axis=0).to_numpy(dtype=float)
    n, m = excess.shape
    mean_excess = excess.mean(axis=0)
    centered = excess - mean_excess

    if block_length is None:
        block_length = _politis_romano_block_length(
            excess.mean(axis=1) if m > 1 else excess[:, 0]
        )

    # Per-strategy variance via stationary-bootstrap-implied long-run variance.
    var_lr = np.empty(m)
    for j in range(m):
        col = centered[:, j]
        gamma0 = float(np.dot(col, col) / n)
        max_lag = max(1, int(np.floor(min(n - 1, 4.0 * (n / 100.0) ** (2.0 / 9.0)))))
        var = gamma0
        for k in range(1, max_lag + 1):
            kernel = (1.0 - k / (max_lag + 1.0))
            cov = float(np.dot(col[k:], col[:-k]) / n)
            var += 2.0 * kernel * cov
        var_lr[j] = max(var, 1e-12)

    omega = np.sqrt(var_lr / n)
    standardized = mean_excess / omega
    test_stat = float(max(standardized.max(), 0.0))
    best_idx = int(np.argmax(standardized))

    # Three recentering schemes (lower / consistent / upper) per Hansen 2005.
    threshold_consistent = -np.sqrt(2.0 * np.log(np.log(n)) / n)
    mu_consistent = np.where(standardized <= threshold_consistent, 0.0, mean_excess)
    mu_lower = np.zeros(m)
    mu_upper = np.where(mean_excess > 0.0, 0.0, mean_excess)

    boot_stats_consistent = np.empty(n_bootstrap)
    boot_stats_lower = np.empty(n_bootstrap)
    boot_stats_upper = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        sample = excess[idx]
        sample_mean = sample.mean(axis=0)
        z_consistent = (sample_mean - mu_consistent - mean_excess) / omega
        z_lower = (sample_mean - mu_lower - mean_excess) / omega
        z_upper = (sample_mean - mu_upper - mean_excess) / omega
        boot_stats_consistent[b] = max(z_consistent.max(), 0.0)
        boot_stats_lower[b] = max(z_lower.max(), 0.0)
        boot_stats_upper[b] = max(z_upper.max(), 0.0)

    return {
        "p_consistent": float((boot_stats_consistent >= test_stat).mean()),
        "p_lower": float((boot_stats_lower >= test_stat).mean()),
        "p_upper": float((boot_stats_upper >= test_stat).mean()),
        "best_strategy": str(aligned.columns[best_idx]),
        "test_statistic": test_stat,
        "block_length": int(block_length),
    }
