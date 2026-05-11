from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V6RiskLimits:
    max_name_weight: float = 0.05
    max_sector_weight: float = 0.30
    max_turnover: float = 0.30
    max_order_value: float = 100_000.0
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    max_orders_per_day: int = 200
    min_lot_size: int = 100
    max_leverage: float = 1.0
    beta_exposure_limit: float = 1.2
    conformal_uncertainty_threshold: float = 0.08
    min_data_quality_score: float = 0.85
    max_model_drift_score: float = 0.30
    no_trade_st: bool = True
    no_buy_limit_up: bool = True
    no_sell_limit_down: bool = True

