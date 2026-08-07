from __future__ import annotations

from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

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
    """Probabilistic Sharpe ratio with the Bailey/Lopez-de-Prado moments term.

    ``pandas.Series.kurt`` is Fisher/excess kurtosis (Pearson kurtosis minus 3).
    The PSR denominator uses ``(Pearson kurtosis - 1) / 4``.  Therefore the
    coefficient expressed in pandas' convention is ``(excess_kurtosis + 2)/4``.
    Using ``excess_kurtosis/4`` understates sampling uncertainty for ordinary
    return distributions and can overstate PSR/DSR promotion probabilities.
    """
    clean = returns.dropna().astype(float)
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
        excess_kurt_raw = clean.kurt()
    skew = 0.0 if not np.isfinite(skew_raw) else float(skew_raw)
    excess_kurt = 0.0 if not np.isfinite(excess_kurt_raw) else float(excess_kurt_raw)
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    moment_term = 1.0 - skew * sr + ((excess_kurt + 2.0) / 4.0) * sr**2
    denom = np.sqrt(max(moment_term, 1e-12))
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

    ``candidate_sharpes`` are per-period Sharpes; ``n_trials`` is the complete
    number of configurations tried.  Unmeasured/degenerate trials must not be
    imputed into the dispersion sample.
    """
    clean = returns.dropna()
    sharpes = np.asarray(candidate_sharpes, dtype=float)
    if n_trials is not None and int(n_trials) < sharpes.size:
        raise ValueError(
            "n_trials cannot be smaller than the measured candidate sample: "
            f"n_trials={int(n_trials)}, candidate_sharpes={sharpes.size}"
        )
    if len(clean) < 4 or sharpes.size < 2 or not np.isfinite(sharpes).all():
        return np.nan
    var_sr = float(np.var(sharpes, ddof=1))
    if var_sr <= 0:
        return np.nan
    total_trials = sharpes.size if n_trials is None else int(n_trials)
    if total_trials < 2:
        return np.nan
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(var_sr) * (
        (1.0 - euler_mascheroni) * _normal_ppf(1.0 - 1.0 / total_trials)
        + euler_mascheroni * _normal_ppf(1.0 - 1.0 / (total_trials * np.e))
    )
    return probabilistic_sharpe_ratio(
        clean,
        sr_benchmark=float(expected_max) * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def newey_west_t_stat(series: pd.Series, max_lag: int | None = None) -> float:
    clean = series.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        return np.nan
    if max_lag is None:
        max_lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mean = clean.mean()
    centered = clean - mean
    gamma0 = float(np.dot(centered, centered) / n)
    nw_var = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        nw_var += 2.0 * weight * gamma
    return float(mean / np.sqrt(max(nw_var, 1e-12) / n))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if excess.empty or len(downside) < 2 or downside.std(ddof=1) == 0:
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


def probability_of_backtest_overfitting(
    is_oos_perf_matrix: np.ndarray | pd.DataFrame,
    n_partitions: int = 16,
    rng_seed: int = 0,
) -> float:
    """CSCV probability that the in-sample winner is below OOS median."""
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
        raise ValueError(f"need at least {n_partitions} rows, got {n_rows}")
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
        n_star = int(np.argmax(chunks[list(is_idx)].mean(axis=0)))
        oos_score = chunks[oos_idx].mean(axis=0)
        ranks = pd.Series(oos_score).rank(method="average", ascending=False)
        rank_n_star = float(ranks.iloc[n_star])
        w = (n_strats - rank_n_star) / (n_strats - 1)
        w = float(min(max(w, 1.0 / (n_strats + 1.0)), 1.0 - 1.0 / (n_strats + 1.0)))
        logits.append(float(np.log(w / (1.0 - w))))
    return float((np.asarray(logits) < 0).mean())


def _politis_romano_block_length(series: np.ndarray) -> int:
    n = len(series)
    if n < 8:
        return 1
    centered = series - series.mean()
    var = float(np.dot(centered, centered) / n)
    if var <= 1e-15:
        return 1
    max_lag = min(n - 1, int(np.floor(8.0 * (n / 100.0) ** (1.0 / 3.0))) + 1)
    auto = np.array([float(np.dot(centered[k:], centered[:-k]) / n) / var for k in range(1, max_lag + 1)])
    g_hat = 0.0
    sigma_hat = var
    for k in range(1, max_lag + 1):
        weight = 1.0 if abs(auto[k - 1]) >= 2.0 * np.sqrt(np.log10(n) / n) else 0.0
        if weight == 0.0:
            break
        g_hat += 2.0 * k * auto[k - 1]
        sigma_hat += 2.0 * auto[k - 1]
    if sigma_hat <= 1e-15:
        return 1
    b_opt = (2.0 * g_hat * g_hat / (sigma_hat * sigma_hat)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(max(1, min(n // 2, round(b_opt)))) if np.isfinite(b_opt) else 1


def _stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    block_length = max(1, int(block_length))
    p_new = 1.0 / block_length
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    for i in range(1, n):
        idx[i] = int(rng.integers(0, n)) if rng.random() < p_new else (idx[i - 1] + 1) % n
    return idx


def spa_test(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Hansen-style SPA bootstrap; small p means a candidate beats benchmark."""
    rng = np.random.default_rng(rng_seed)
    aligned = candidate_returns.dropna(how="all").copy()
    bench = benchmark_returns.reindex(aligned.index).astype(float)
    mask = ~(aligned.isna().any(axis=1) | bench.isna())
    aligned = aligned.loc[mask]
    bench = bench.loc[mask]
    if len(aligned) < 8 or aligned.shape[1] == 0:
        return {
            "p_consistent": float("nan"), "p_lower": float("nan"), "p_upper": float("nan"),
            "best_strategy": "", "test_statistic": float("nan"), "block_length": 1,
        }
    excess = aligned.sub(bench, axis=0).to_numpy(dtype=float)
    n, m = excess.shape
    mean_excess = excess.mean(axis=0)
    centered = excess - mean_excess
    if block_length is None:
        block_length = _politis_romano_block_length(excess.mean(axis=1) if m > 1 else excess[:, 0])
    var_lr = np.empty(m)
    for j in range(m):
        col = centered[:, j]
        gamma0 = float(np.dot(col, col) / n)
        max_lag = max(1, int(np.floor(min(n - 1, 4.0 * (n / 100.0) ** (2.0 / 9.0)))))
        var = gamma0
        for k in range(1, max_lag + 1):
            kernel = 1.0 - k / (max_lag + 1.0)
            var += 2.0 * kernel * float(np.dot(col[k:], col[:-k]) / n)
        var_lr[j] = max(var, 1e-12)
    omega = np.sqrt(var_lr / n)
    standardized = mean_excess / omega
    test_stat = float(max(standardized.max(), 0.0))
    best_idx = int(np.argmax(standardized))
    threshold_consistent = -np.sqrt(2.0 * np.log(np.log(n)) / n)
    mu_consistent = np.where(standardized <= threshold_consistent, 0.0, mean_excess)
    mu_lower = np.zeros(m)
    mu_upper = np.where(mean_excess > 0.0, 0.0, mean_excess)
    boot_consistent = np.empty(n_bootstrap)
    boot_lower = np.empty(n_bootstrap)
    boot_upper = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        sample_mean = excess[idx].mean(axis=0)
        boot_consistent[b] = max(((sample_mean - mu_consistent - mean_excess) / omega).max(), 0.0)
        boot_lower[b] = max(((sample_mean - mu_lower - mean_excess) / omega).max(), 0.0)
        boot_upper[b] = max(((sample_mean - mu_upper - mean_excess) / omega).max(), 0.0)
    return {
        "p_consistent": float((boot_consistent >= test_stat).mean()),
        "p_lower": float((boot_lower >= test_stat).mean()),
        "p_upper": float((boot_upper >= test_stat).mean()),
        "best_strategy": str(aligned.columns[best_idx]),
        "test_statistic": test_stat,
        "block_length": int(block_length),
    }
