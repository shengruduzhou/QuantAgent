"""Event-driven backtester respecting A-share T+1 and price-limit rules."""

from quantagent.backtest.event_driven_theme_backtester import EventDrivenThemeBacktester
from quantagent.backtest.full_pipeline_backtester import (
    FullPipelineBacktestConfig,
    FullPipelineBacktestResult,
    build_pit_evidence_slice,
    run_full_pipeline_backtest,
)
from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    AShareExecutionSimulationResult,
    simulate_ashare_target_weights,
)


__all__ = [
    "EventDrivenThemeBacktester",
    "FullPipelineBacktestConfig",
    "FullPipelineBacktestResult",
    "AShareExecutionSimulationConfig",
    "AShareExecutionSimulationResult",
    "build_pit_evidence_slice",
    "run_full_pipeline_backtest",
    "simulate_ashare_target_weights",
]
