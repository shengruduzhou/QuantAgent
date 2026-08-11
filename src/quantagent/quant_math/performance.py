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
    """Bailey & Lopez de Prado PSR adjusted for skew and Pearson kurtosis.

    pandas ``Series.kurt`` returns excess kurtosis. The PSR denominator is
    expressed with Pearson kurtosis as ``(gamma4 - 1) / 4``; therefore the
    pandas-convention coefficient is ``(excess_kurtosis + 2) / 4``.
    """
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
    excess_kurtosis = 0.0 if not np.isfinite(kurt_raw) else float(kurt_raw)
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    denom = np.sqrt(max(1.0 - skew * sr + (excess_kurtosis + 2.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(_normal_cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    candidate_sharpes: np.ndarray,
    periods_per_year: int = 252,
    *,
    n_trials: int | None = None,
) -> float:
    """Deflated SR: PSR threshold inflated by max-of-N selection bias.

    ``candidate_sharpes`` must be **per-period**, not annualised: the expected
    maximum is re-annualised here before it becomes the PSR benchmark. Passing
    annualised Sharpes inflates the benchmark by ``sqrt(periods_per_year)`` and
    returns 0.0 for every champion. Every entry must be finite -- a placeholder
    substituted for an undefined Sharpe would set ``var_sr``, and so the whole
    benchmark, by itself.

    The two inputs to the Bailey & Lopez de Prado benchmark are separate and
    must be supplied separately:

    * ``candidate_sharpes`` estimates the cross-trial *dispersion* of Sharpes.
      Only trials whose Sharpe was actually measured belong in it.
    * ``n_trials`` is *how many* trials were run, defaulting to the size of
      that sample. Declare it whenever the search tried more configurations
      than are represented in ``candidate_sharpes``.

    Do not conflate them by padding the sample up to the trial count. A
    constant pad shrinks ``var(candidate_sharpes)`` like ``1/n_trials`` while
    the order-statistic term grows only like ``sqrt(log n_trials)``, so the
    benchmark *falls* towards zero and declaring more data mining makes the
    gate easier to pass -- backwards for a multiple-testing correction. With
    the two passed separately the benchmark is non-decreasing in ``n_trials``,
    so the returned probability is non-increasing in it.
    """
    clean = returns.dropna()
    n = len(clean)
    if n_trials is not None and int(n_trials) < candidate_sharpes.size:
        raise ValueError(
            "n_trials cannot be smaller than the measured candidate sample: "
            f"n_trials={int(n_trials)}, candidate_sharpes={candidate_sharpes.size}"
        )
    if n < 4 or candidate_sharpes.size < 2:
        return np.nan
    var_sr = float(np.var(candidate_sharpes, ddof=1))
    if var_sr <= 0:
        return np.nan
    n_trials = candidate_sharpes.size if n_trials is None else int(n_trials)
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

    chunks = arr.reshape(n_partitions, rows_per_chunk, n_strats).mean(axis=1)

    half = n_partitions // 2
    indices = list(range(n_partitions))
    logits: list[float] = []
    for is_idx in combinations(indices, half):
        is_set = set(is_idx)
        oos_idx = [i for i in indices if i not in is_set]
        is_score = chunks[list(is_idx)].mean(axis=0)
        oos_score = chunks[oos_idx].mean(axis=0)
        n_star = int(np.argmax(is_score))
        order = pd.Series(oos_score).rank(method="average", ascending=False)
        rank_n_star = float(order.iloc[n_star])
        w = (n_strats - rank_n_star) / (n_strats - 1) if n_strats > 1 else 0.5
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
    max_lag = min(n - 1, int(np.floor(8.0 * (n / 100.0) ** (1.0 / 3.0))) + 1)
    auto = np.array(
        [float(np.dot(centered[k:], centered[:-k]) / n) / var for k in range(1, max_lag + 1)]
    )
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


def _spa_recenter_means(
    mean_excess: np.ndarray,
    omega: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hansen SPA lower/consistent/upper null means for return differentials.

    Positive ``mean_excess`` means a candidate outperforms the benchmark. For
    the consistent estimator, a candidate is asymptotically relevant when its
    observed mean is no farther below zero than Hansen's ``log(log(n))`` bound.
    Since ``omega`` is the standard error of the sample mean, the raw-return
    threshold is ``-omega * sqrt(2 log log n)``.

    The three recentering rules mirror the standard SPA construction:

    * lower: do not recenter clearly inferior (negative-mean) candidates;
    * consistent: recenter only sample-dependent relevant candidates;
    * upper: recenter every candidate.

    They are returned in p-value order ``lower, consistent, upper`` together
    with the consistent relevance mask.
    """
    if n < 3:
        raise ValueError("SPA recentering requires n >= 3")
    if mean_excess.shape != omega.shape:
        raise ValueError("mean_excess and omega must have identical shape")
    if np.any(~np.isfinite(mean_excess)) or np.any(~np.isfinite(omega)):
        raise ValueError("SPA recentering requires finite means and standard errors")
    if np.any(omega <= 0.0):
        raise ValueError("SPA standard errors must be strictly positive")

    log_log = np.log(np.log(float(n)))
    bound_scale = np.sqrt(max(2.0 * log_log, 0.0))
    threshold = -omega * bound_scale
    relevant = mean_excess >= threshold

    upper_mean = mean_excess.copy()
    consistent_mean = upper_mean.copy()
    consistent_mean[~relevant] = 0.0
    lower_mean = upper_mean.copy()
    lower_mean[lower_mean < 0.0] = 0.0
    return lower_mean, consistent_mean, upper_mean, relevant


def spa_test(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Hansen (2005) Superior Predictive Ability test, consistent variant.

    Returns p-values for the one-sided null ``no candidate beats benchmark``.
    The statistic is studentized by a HAC standard error and the stationary
    bootstrap uses Hansen's lower/consistent/upper sample-dependent null
    recentering. A smaller consistent p-value is stronger evidence that at
    least one candidate has genuine positive excess return after correcting for
    data snooping.

    Notes
    -----
    ``candidate_returns - benchmark_returns`` is the performance differential,
    so positive values are better. The bootstrap sample mean is recentered
    **once** by the null mean. Subtracting the observed mean a second time would
    shift the simulated distribution and invalidate the SPA p-value.

    Returns
    -------
    dict with keys:
      ``p_consistent``   — Hansen sample-dependent SPA p-value
      ``p_lower``        — lower-bound p-value
      ``p_upper``        — upper-bound p-value
      ``best_strategy``  — largest studentized mean excess return
      ``test_statistic`` — observed one-sided studentized maximum
      ``block_length``   — stationary-bootstrap mean block length
    """
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be >= 1")
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
    block_length = int(block_length)
    if block_length < 1:
        raise ValueError("block_length must be >= 1")

    # HAC long-run variance of each performance differential. ``var_lr`` is
    # variance per observation; omega is the standard error of the sample mean.
    var_lr = np.empty(m)
    for j in range(m):
        col = centered[:, j]
        gamma0 = float(np.dot(col, col) / n)
        max_lag = max(1, int(np.floor(min(n - 1, 4.0 * (n / 100.0) ** (2.0 / 9.0)))))
        var = gamma0
        for k in range(1, max_lag + 1):
            kernel = 1.0 - k / (max_lag + 1.0)
            cov = float(np.dot(col[k:], col[:-k]) / n)
            var += 2.0 * kernel * cov
        var_lr[j] = max(var, 1e-12)

    omega = np.sqrt(var_lr / n)
    standardized = mean_excess / omega
    test_stat = float(max(float(standardized.max()), 0.0))
    best_idx = int(np.argmax(standardized))

    mu_lower, mu_consistent, mu_upper, _ = _spa_recenter_means(
        mean_excess,
        omega,
        n,
    )

    boot_stats_lower = np.empty(n_bootstrap)
    boot_stats_consistent = np.empty(n_bootstrap)
    boot_stats_upper = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        sample_mean = excess[idx].mean(axis=0)
        # One recentering only. This is the studentized analogue of the
        # standard SPA stationary-bootstrap construction.
        z_lower = (sample_mean - mu_lower) / omega
        z_consistent = (sample_mean - mu_consistent) / omega
        z_upper = (sample_mean - mu_upper) / omega
        boot_stats_lower[b] = max(float(z_lower.max()), 0.0)
        boot_stats_consistent[b] = max(float(z_consistent.max()), 0.0)
        boot_stats_upper[b] = max(float(z_upper.max()), 0.0)

    p_lower = float((boot_stats_lower >= test_stat).mean())
    p_consistent = float((boot_stats_consistent >= test_stat).mean())
    p_upper = float((boot_stats_upper >= test_stat).mean())
    # Shared bootstrap draws and nested recentering sets imply this order. Fail
    # closed if future edits violate the mathematical contract beyond rounding.
    if not (p_lower <= p_consistent + 1e-12 and p_consistent <= p_upper + 1e-12):
        raise RuntimeError(
            "SPA p-value ordering invariant failed: expected lower <= consistent <= upper, "
            f"got {p_lower:.6f}, {p_consistent:.6f}, {p_upper:.6f}"
        )

    return {
        "p_consistent": p_consistent,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "best_strategy": str(aligned.columns[best_idx]),
        "test_statistic": test_stat,
        "block_length": block_length,
    }
