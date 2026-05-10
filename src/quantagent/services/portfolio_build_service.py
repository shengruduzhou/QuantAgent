from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant_math.optimizer import V4PortfolioConfig, V4PortfolioResult, solve_v4_portfolio


def build_portfolio_v4(signals: pd.DataFrame, mode: str = "long_only_enhancement") -> V4PortfolioResult:
    alpha = signals.set_index("symbol")["alpha"].astype(float)
    symbols = alpha.index
    covariance = pd.DataFrame(np.eye(len(symbols)) * 0.04, index=symbols, columns=symbols)
    current = pd.Series(0.0, index=symbols)
    cost = pd.Series(0.0005, index=symbols)
    return solve_v4_portfolio(alpha, covariance, current_weights=current, cost=cost, config=V4PortfolioConfig(mode=mode))
