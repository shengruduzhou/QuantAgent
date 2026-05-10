import numpy as np
import pandas as pd

from quantagent.quant_math.optimizer import V4PortfolioConfig, solve_v4_portfolio
from quantagent.quant_math.signal_fusion import blend_alpha_and_views


def test_v4_blend_and_long_only_optimizer_outputs_weights_not_orders():
    alpha = pd.Series({"A": 0.03, "B": 0.02, "C": -0.01})
    blended = blend_alpha_and_views(alpha, agent_posterior=alpha * 1.2)
    assert "blended_alpha" in blended.columns
    cov = pd.DataFrame(np.eye(3) * 0.04, index=alpha.index, columns=alpha.index)
    result = solve_v4_portfolio(alpha, cov, config=V4PortfolioConfig(max_name_weight=0.1, max_turnover=1.0))
    assert result.target_weights.max() <= 0.1000001
    assert not hasattr(result, "orders")


def test_v4_hedged_optimizer_and_reject_diagnostics():
    alpha = pd.Series({"A": 0.03, "B": -0.02, "C": 0.01})
    cov = pd.DataFrame(np.eye(3) * 0.04, index=alpha.index, columns=alpha.index)
    tradability = pd.DataFrame([{"symbol": "A", "is_limit_up": True}])
    result = solve_v4_portfolio(alpha, cov, tradability=tradability, config=V4PortfolioConfig(mode="hedged_alpha"))
    assert "A" in result.rejected_symbols
    assert abs(result.target_weights.sum()) < 1e-9 or result.target_weights.empty
