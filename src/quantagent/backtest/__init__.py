"""Event-driven backtester respecting A-share T+1 and price-limit rules."""

from quantagent.backtest.event_driven_theme_backtester import EventDrivenThemeBacktester
from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    AShareExecutionSimulationResult,
    simulate_ashare_target_weights,
)


__all__ = [
    "EventDrivenThemeBacktester",
    "AShareExecutionSimulationConfig",
    "AShareExecutionSimulationResult",
    "simulate_ashare_target_weights",
]
