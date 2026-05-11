"""V4 compatibility wrapper around the unified multi-tower model."""
from __future__ import annotations

from dataclasses import dataclass

from quantagent.models.multitower import MultiTowerConfig, MultiTowerModel


@dataclass(frozen=True)
class V4MultiTowerConfig:
    sequence_input_dim: int
    snapshot_input_dim: int
    event_numeric_dim: int = 6
    lookback: int = 20
    hidden_dim: int = 64
    tower_dim: int = 64
    dropout: float = 0.1
    factor_gate_dim: int = 4


class V4MultiTowerModel(MultiTowerModel):
    """Legacy V4 API. Implementation lives in ``models.multitower``."""

    def __init__(self, config: V4MultiTowerConfig) -> None:
        super().__init__(
            MultiTowerConfig(
                sequence_input_dim=config.sequence_input_dim,
                snapshot_input_dim=config.snapshot_input_dim,
                event_numeric_dim=config.event_numeric_dim,
                lookback=config.lookback,
                hidden_dim=config.hidden_dim,
                tower_dim=config.tower_dim,
                dropout=config.dropout,
                factor_gate_dim=config.factor_gate_dim,
                sequence_backbone="simple_seq",
                fusion_mode="concat",
                factor_gate_activation="sigmoid",
            )
        )
        self.legacy_config = config

    def forward(
        self,
        sequence,
        snapshot,
        events,
        sequence_mask=None,
        event_mask=None,
    ):
        return super().forward(
            sequence=sequence,
            snapshot=snapshot,
            events=events,
            regime=None,
            sequence_mask=sequence_mask,
            event_mask=event_mask,
        )


def build_tiny_v4_model(
    sequence_input_dim: int,
    snapshot_input_dim: int,
    event_numeric_dim: int = 6,
    lookback: int = 8,
) -> V4MultiTowerModel:
    return V4MultiTowerModel(
        V4MultiTowerConfig(
            sequence_input_dim=sequence_input_dim,
            snapshot_input_dim=snapshot_input_dim,
            event_numeric_dim=event_numeric_dim,
            lookback=lookback,
            hidden_dim=16,
            tower_dim=16,
            dropout=0.0,
            factor_gate_dim=3,
        )
    )
