import numpy as np
import pandas as pd

from quantagent.quant_math.risk_metrics import historical_cvar, historical_var






def test_risk_metrics_return_positive_tail_loss():
    returns = pd.Series([0.01, -0.01, -0.03, 0.02, -0.05])

    assert historical_var(returns, 0.8) > 0
    assert historical_cvar(returns, 0.8) >= historical_var(returns, 0.8)






