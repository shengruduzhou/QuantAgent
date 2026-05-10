from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal
from quantagent.agents.views_schema import AgentView
from quantagent.quant_math.signal_fusion import black_litterman_posterior


@dataclass(frozen=True)
class BLViewConfig:
    base_view_strength: float = 0.05
    min_omega: float = 1e-6
    tau: float = 0.05


def agent_signals_to_bl_views(
    signals: list[AgentSignal],
    universe: pd.Index,
    agent_ir: pd.Series | None = None,
    expected_volatility: pd.Series | None = None,
    config: BLViewConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map AgentSignal list to BL (P, q, Omega).

    P[k, i]: 1 / -1 / 0 picking matrix.
    q[k]:    expected return view = strength * confidence * vol scaling.
    Omega:   diagonal uncertainty = 1 / (confidence * evidence_quality * IR^2).
    """
    config = config or BLViewConfig()
    valid = [s for s in signals if s.symbol in universe]
    if not valid:
        return np.zeros((0, len(universe))), np.zeros(0), np.zeros((0, 0))
    n_views = len(valid)
    n_assets = len(universe)
    p = np.zeros((n_views, n_assets))
    q = np.zeros(n_views)
    omega = np.zeros(n_views)
    sym_to_idx = {s: i for i, s in enumerate(universe)}
    for k, signal in enumerate(valid):
        i = sym_to_idx[signal.symbol]
        direction = float(np.sign(signal.signal_strength)) or 1.0
        p[k, i] = direction
        vol_i = (
            float(expected_volatility.loc[signal.symbol])
            if expected_volatility is not None and signal.symbol in expected_volatility.index
            else config.base_view_strength
        )
        magnitude = abs(signal.signal_strength) * signal.confidence
        q[k] = direction * magnitude * vol_i
        ir = (
            float(agent_ir.get(signal.agent_name, 1.0))
            if agent_ir is not None
            else 1.0
        )
        sharp = max(signal.confidence * signal.evidence_quality * (ir ** 2), 1e-3)
        omega[k] = max(vol_i ** 2 / sharp, config.min_omega)
    return p, q, omega


def agent_views_to_bl_views(
    views: list[AgentView],
    universe: pd.Index,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [view for view in views if set(view.symbols).intersection(set(universe))]
    if not valid:
        return np.zeros((0, len(universe))), np.zeros(0), np.zeros((0, 0))
    sym_to_idx = {symbol: i for i, symbol in enumerate(universe)}
    p = np.zeros((len(valid), len(universe)))
    q = np.zeros(len(valid))
    omega = np.zeros(len(valid))
    for row, view in enumerate(valid):
        for symbol, value in view.exposure.items():
            if symbol in sym_to_idx:
                p[row, sym_to_idx[symbol]] = float(value)
        q[row] = float(view.q)
        omega[row] = max(float(view.omega), 1e-12)
    return p, q, omega


def posterior_alpha_from_agent_views(
    prior_returns: pd.Series,
    covariance: pd.DataFrame,
    views: list[AgentView],
    config: BLViewConfig | None = None,
) -> pd.Series:
    config = config or BLViewConfig()
    p, q, omega = agent_views_to_bl_views(views, prior_returns.index)
    if p.shape[0] == 0:
        return prior_returns.copy()
    return black_litterman_posterior(
        prior_returns,
        covariance,
        pick_matrix=p,
        views=q,
        view_uncertainty=omega,
        tau=config.tau,
    )


def posterior_alpha_from_agents(
    prior_returns: pd.Series,
    covariance: pd.DataFrame,
    signals: list[AgentSignal],
    agent_ir: pd.Series | None = None,
    expected_volatility: pd.Series | None = None,
    config: BLViewConfig | None = None,
) -> pd.Series:
    """End-to-end pipeline: agents -> views -> Black-Litterman posterior alpha."""
    config = config or BLViewConfig()
    p, q, omega = agent_signals_to_bl_views(
        signals,
        prior_returns.index,
        agent_ir=agent_ir,
        expected_volatility=expected_volatility,
        config=config,
    )
    if p.shape[0] == 0:
        return prior_returns.copy()
    return black_litterman_posterior(
        prior_returns,
        covariance,
        pick_matrix=p,
        views=q,
        view_uncertainty=omega,
        tau=config.tau,
    )
