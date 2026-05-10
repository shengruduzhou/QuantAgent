from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from quantagent.models.backbone_base import BackboneSpec, SimpleSequenceBackbone
from quantagent.models.event_tower import StructuredEventTower, StructuredEventTowerConfig
from quantagent.models.tabular_resnet import TabularResNet, TabularResNetConfig


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


if nn is not None:

    class V4MultiTowerModel(nn.Module):
        """Three-tower V4 alpha model for A-share research."""

        def __init__(self, config: V4MultiTowerConfig) -> None:
            super().__init__()
            self.config = config
            self.sequence_tower = SimpleSequenceBackbone(
                BackboneSpec(
                    input_dim=config.sequence_input_dim,
                    lookback=config.lookback,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.tower_dim,
                    dropout=config.dropout,
                )
            )
            self.snapshot_tower = TabularResNet(
                TabularResNetConfig(
                    input_dim=config.snapshot_input_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.tower_dim,
                    dropout=config.dropout,
                )
            )
            self.event_tower = StructuredEventTower(
                StructuredEventTowerConfig(
                    numeric_dim=config.event_numeric_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.tower_dim,
                    dropout=config.dropout,
                )
            )
            fusion_dim = config.tower_dim * 3
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.alpha_head = nn.Linear(config.hidden_dim, 1)
            self.direction_head = nn.Linear(config.hidden_dim, 1)
            self.quantile_head = nn.Linear(config.hidden_dim, 2)
            self.factor_gate_head = nn.Linear(config.hidden_dim, config.factor_gate_dim)
            self.confidence_head = nn.Linear(config.hidden_dim, 1)
            self.risk_head = nn.Linear(config.hidden_dim, 1)

        def forward(
            self,
            sequence: "torch.Tensor",
            snapshot: "torch.Tensor",
            events: "torch.Tensor",
            sequence_mask: "torch.Tensor | None" = None,
            event_mask: "torch.Tensor | None" = None,
        ) -> dict[str, "torch.Tensor"]:
            seq = self.sequence_tower(sequence, sequence_mask)
            snap = self.snapshot_tower(snapshot)
            event = self.event_tower(events, event_mask)
            hidden = self.fusion(torch.cat([seq, snap, event], dim=-1))
            quantiles = self.quantile_head(hidden)
            q_low = torch.minimum(quantiles[:, 0], quantiles[:, 1])
            q_high = torch.maximum(quantiles[:, 0], quantiles[:, 1])
            return {
                "alpha": self.alpha_head(hidden).squeeze(-1),
                "direction_logit": self.direction_head(hidden).squeeze(-1),
                "q_low": q_low,
                "q_high": q_high,
                "factor_gate": torch.sigmoid(self.factor_gate_head(hidden)),
                "confidence": torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(hidden)).squeeze(-1),
            }


def build_tiny_v4_model(
    sequence_input_dim: int,
    snapshot_input_dim: int,
    event_numeric_dim: int = 6,
    lookback: int = 8,
) -> "V4MultiTowerModel":
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

else:

    class V4MultiTowerModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("V4MultiTowerModel requires PyTorch: install quantagent[training]")


    def build_tiny_v4_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("V4MultiTowerModel requires PyTorch: install quantagent[training]")
