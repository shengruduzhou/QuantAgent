"""V5 compatibility wrapper around the unified multi-tower model."""
from __future__ import annotations

from dataclasses import dataclass

from quantagent.models.multitower import MoEFusion, MultiTowerConfig, MultiTowerModel


@dataclass(frozen=True)
class V5MultiTowerConfig:
    sequence_input_dim: int
    snapshot_input_dim: int
    event_numeric_dim: int = 6
    regime_dim: int = 4
    lookback: int = 20
    hidden_dim: int = 64
    tower_dim: int = 64
    dropout: float = 0.1
    factor_group_dim: int = 6
    sequence_backbone: str = "simple_seq"
    num_experts: int = 3


class V5MultiTowerModel(MultiTowerModel):
    """Legacy V5 API. Implementation lives in ``models.multitower``."""

    def __init__(self, config: V5MultiTowerConfig) -> None:
        super().__init__(
            MultiTowerConfig(
                sequence_input_dim=config.sequence_input_dim,
                snapshot_input_dim=config.snapshot_input_dim,
                event_numeric_dim=config.event_numeric_dim,
                regime_dim=config.regime_dim,
                lookback=config.lookback,
                hidden_dim=config.hidden_dim,
                tower_dim=config.tower_dim,
                dropout=config.dropout,
                factor_gate_dim=config.factor_group_dim,
                sequence_backbone=config.sequence_backbone,
                num_experts=config.num_experts,
                fusion_mode="moe",
                factor_gate_activation="softmax",
            )
        )
        self.legacy_config = config


def build_tiny_v5_model(
    sequence_input_dim: int,
    snapshot_input_dim: int,
    event_numeric_dim: int = 6,
    regime_dim: int = 4,
    lookback: int = 8,
    sequence_backbone: str = "simple_seq",
    factor_group_dim: int = 4,
) -> V5MultiTowerModel:
    return V5MultiTowerModel(
        V5MultiTowerConfig(
            sequence_input_dim=sequence_input_dim,
            snapshot_input_dim=snapshot_input_dim,
            event_numeric_dim=event_numeric_dim,
            regime_dim=regime_dim,
            lookback=lookback,
            hidden_dim=16,
            tower_dim=16,
            dropout=0.0,
            factor_group_dim=factor_group_dim,
            sequence_backbone=sequence_backbone,
            num_experts=2,
        )
    )
