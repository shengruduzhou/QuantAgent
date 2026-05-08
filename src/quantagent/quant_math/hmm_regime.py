from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quantagent.quant_math.regime import REGIME_MULTIPLIER, MarketRegime


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 3
    max_iter: int = 100
    tol: float = 1e-5
    seed: int = 42


@dataclass(frozen=True)
class HMMState:
    means: np.ndarray
    covariances: np.ndarray
    transitions: np.ndarray
    initial: np.ndarray
    log_likelihood: float


def fit_gaussian_hmm(observations: np.ndarray, config: HMMConfig | None = None) -> HMMState:
    """EM-fitted diagonal-covariance Gaussian HMM (no external dependency)."""
    config = config or HMMConfig()
    rng = np.random.default_rng(config.seed)
    obs = np.atleast_2d(observations)
    if obs.shape[0] < obs.shape[1]:
        obs = obs.T
    n, d = obs.shape
    k = config.n_states
    means = rng.normal(size=(k, d)) * obs.std(axis=0) + obs.mean(axis=0)
    covariances = np.tile(np.var(obs, axis=0), (k, 1)) + 1e-4
    transitions = np.full((k, k), 1.0 / k)
    initial = np.full(k, 1.0 / k)
    prev_ll = -np.inf
    for _ in range(config.max_iter):
        emissions = np.stack(
            [stats.multivariate_normal.pdf(obs, mean=means[s], cov=np.diag(covariances[s])) for s in range(k)],
            axis=1,
        )
        emissions = np.clip(emissions, 1e-300, None)
        alpha = np.zeros((n, k))
        scale = np.zeros(n)
        alpha[0] = initial * emissions[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ transitions) * emissions[t]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = transitions @ (emissions[t + 1] * beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12
        xi = np.zeros((n - 1, k, k))
        for t in range(n - 1):
            denom = (alpha[t] * (transitions @ (emissions[t + 1] * beta[t + 1]))).sum() + 1e-12
            xi[t] = (
                alpha[t][:, None]
                * transitions
                * (emissions[t + 1] * beta[t + 1])[None, :]
                / denom
            )
        initial = gamma[0]
        transitions = xi.sum(axis=0)
        transitions /= transitions.sum(axis=1, keepdims=True) + 1e-12
        gamma_sum = gamma.sum(axis=0) + 1e-12
        means = (gamma.T @ obs) / gamma_sum[:, None]
        for s in range(k):
            diff = obs - means[s]
            covariances[s] = (gamma[:, s][:, None] * diff ** 2).sum(axis=0) / gamma_sum[s] + 1e-6
        ll = float(np.log(scale + 1e-300).sum())
        if abs(ll - prev_ll) < config.tol:
            break
        prev_ll = ll
    return HMMState(
        means=means,
        covariances=covariances,
        transitions=transitions,
        initial=initial,
        log_likelihood=prev_ll,
    )


def posterior_state_probabilities(state: HMMState, observations: np.ndarray) -> np.ndarray:
    """Return [n, k] smoothed P(state_k | obs_1..n)."""
    obs = np.atleast_2d(observations)
    if obs.shape[0] < obs.shape[1]:
        obs = obs.T
    n, _ = obs.shape
    k = state.means.shape[0]
    emissions = np.stack(
        [
            stats.multivariate_normal.pdf(obs, mean=state.means[s], cov=np.diag(state.covariances[s]))
            for s in range(k)
        ],
        axis=1,
    )
    emissions = np.clip(emissions, 1e-300, None)
    alpha = np.zeros((n, k))
    scale = np.zeros(n)
    alpha[0] = state.initial * emissions[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ state.transitions) * emissions[t]
        scale[t] = alpha[t].sum()
        if scale[t] > 0:
            alpha[t] /= scale[t]
    beta = np.zeros((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = state.transitions @ (emissions[t + 1] * beta[t + 1])
        if scale[t + 1] > 0:
            beta[t] /= scale[t + 1]
    gamma = alpha * beta
    return gamma / (gamma.sum(axis=1, keepdims=True) + 1e-12)


def label_states_to_regimes(state: HMMState) -> list[MarketRegime]:
    """Sort by mean drift; assign bear / range / bull."""
    drifts = state.means[:, 0]
    order = np.argsort(drifts)
    labels: dict[int, MarketRegime] = {}
    if state.means.shape[0] >= 3:
        labels[order[0]] = MarketRegime.BEAR_TREND
        labels[order[-1]] = MarketRegime.BULL_TREND
        for s in order[1:-1]:
            vol = state.covariances[s].mean()
            labels[s] = MarketRegime.HIGH_VOLATILITY if vol > state.covariances.mean() else MarketRegime.RANGE_BOUND
    elif state.means.shape[0] == 2:
        labels[order[0]] = MarketRegime.BEAR_TREND
        labels[order[1]] = MarketRegime.BULL_TREND
    else:
        labels[order[0]] = MarketRegime.RANGE_BOUND
    return [labels[i] for i in range(state.means.shape[0])]


def hmm_regime_alpha_multiplier(
    posterior: pd.Series,
    state_labels: list[MarketRegime],
) -> float:
    """Mixture multiplier: sum_k P(state_k) * REGIME_MULTIPLIER[label_k]."""
    multipliers = np.array([REGIME_MULTIPLIER[label] for label in state_labels])
    return float(np.dot(posterior.to_numpy(), multipliers))
