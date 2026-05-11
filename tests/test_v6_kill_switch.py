import pandas as pd

from quantagent.risk.kill_switch import KillSwitch
from quantagent.risk.risk_gate import RiskGate


def test_kill_switch_blocks_risk_gate():
    kill = KillSwitch()
    kill.trigger("manual_kill")
    result = RiskGate(kill_switch=kill).check_target_weights(pd.Series({"600000.SH": 0.01}))
    assert not result.passed
    assert "kill_switch_triggered" in result.violations

