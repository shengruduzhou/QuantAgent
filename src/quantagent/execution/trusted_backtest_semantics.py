"""Canonical trusted execution/cost expectations for live-model evidence.

Values are derived from the same simulator and cost-model dataclasses used by
StrictBacktestV8.  A trust certificate therefore cannot claim generic
"costs included" while silently changing slippage, fees, participation caps or
cash-account capabilities.
"""

from __future__ import annotations

from dataclasses import asdict

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    STRICT_CASH_ACCOUNT_SEMANTICS,
)
from quantagent.execution.cost_model import AShareCostModel


TRUSTED_EXECUTION_SEMANTICS = STRICT_CASH_ACCOUNT_SEMANTICS


def trusted_simulation_config() -> dict[str, object]:
    config = AShareExecutionSimulationConfig()
    # audit_log_dir is an output location, not an economic assumption.
    payload = asdict(config)
    payload.pop("audit_log_dir", None)
    return payload


def trusted_cost_model_config() -> dict[str, object]:
    return asdict(AShareCostModel())


__all__ = [
    "TRUSTED_EXECUTION_SEMANTICS",
    "trusted_cost_model_config",
    "trusted_simulation_config",
]
