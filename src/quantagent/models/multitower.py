"""Unified multi-tower alpha model used by V4, V5, and V6 training paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from quantagent.models.backbone_registry import BackboneRegistryConfig, build_sequence_backbone
from quantagent.models.event_tower import StructuredEventTower, StructuredEventTowerConfig
from quantagent.models.tabular_resnet import TabularResNet, TabularResNetConfig

FusionMode = Literal["concat", "moe"]
GateActivation = Literal["sigmoid", "softmax"]


@dataclass(frozen=True)
class MultiTowerConfig:
    sequence_input_dim: int
    snapshot_input_dim: int
    event_numeric_dim: int = 6
    regime_dim: int = 4
    lookback: int = 20
    hidden_dim: int = 64
    tower_dim: int = 64
    dropout: float = 0.1
    factor_gate_dim: int = 6
    sequence_backbone: str = "simple_seq"
    num_experts: int = 3
    fusion_mode: FusionMode = "moe"
    factor_gate_activation: GateActivation = "softmax"


if nn is not None:

    class MoEFusion(nn.Module):
        """Regime-conditional Mixture-of-Experts fusion of tower outputs."""

        def __init__(self, tower_dim: int, regime_dim: int, hidden_dim: int, num_experts: int, dropout: float) -> None:
            super().__init__()
            self.num_experts = num_experts
            fusion_dim = tower_dim * 3
            self.gate = nn.Sequential(
                nn.Linear(regime_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts),
            )
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(fusion_dim),
                        nn.Linear(fusion_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_experts)
                ]
            )

        def forward(self, z_seq, z_snap, z_event, regime) -> tuple["torch.Tensor", "torch.Tensor"]:
            cat = torch.cat([z_seq, z_snap, z_event], dim=-1)
            gate_weights = torch.softmax(self.gate(regime), dim=-1)
            expert_outputs = torch.stack([expert(cat) for expert in self.experts], dim=1)
            return (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1), gate_weights


    class ConcatFusion(nn.Module):
        """V4-compatible deterministic tower fusion."""

        def __init__(self, tower_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(tower_dim * 3),
                nn.Linear(tower_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, z_seq, z_snap, z_event, regime=None) -> tuple["torch.Tensor", "torch.Tensor | None"]:
            del regime
            return self.net(torch.cat([z_seq, z_snap, z_event], dim=-1)), None


    class MultiTowerModel(nn.Module):
        """Shared configurable model for legacy V4, V5, and V6 model configs."""

        def __init__(self, config: MultiTowerConfig) -> None:
            super().__init__()
            self.config = config
            self.sequence_tower = build_sequence_backbone(
                BackboneRegistryConfig(
                    name=config.sequence_backbone,
                    input_dim=config.sequence_input_dim,
                    lookback=config.lookback,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.tower_dim,
                    dropout=config.dropout,
                    num_layers=2,
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
            if config.fusion_mode == "concat":
                self.fusion = ConcatFusion(config.tower_dim, config.hidden_dim, config.dropout)
            else:
                self.fusion = MoEFusion(config.tower_dim, config.regime_dim, config.hidden_dim, config.num_experts, config.dropout)
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
            regime: "torch.Tensor | None" = None,
            sequence_mask: "torch.Tensor | None" = None,
            event_mask: "torch.Tensor | None" = None,
        ) -> dict[str, "torch.Tensor"]:
            if getattr(self.sequence_tower, "supports_mask", False):
                z_seq = self.sequence_tower(sequence, sequence_mask)
            else:
                z_seq = self.sequence_tower(sequence)
            z_snap = self.snapshot_tower(snapshot)
            z_event = self.event_tower(events, event_mask)
            if self.config.fusion_mode == "moe" and regime is None:
                regime = torch.zeros((sequence.shape[0], self.config.regime_dim), device=sequence.device, dtype=sequence.dtype)
            hidden, moe_gate = self.fusion(z_seq, z_snap, z_event, regime)
            quantiles = self.quantile_head(hidden)
            q_low = torch.minimum(quantiles[:, 0], quantiles[:, 1])
            q_high = torch.maximum(quantiles[:, 0], quantiles[:, 1])
            gate_logits = self.factor_gate_head(hidden)
            if self.config.factor_gate_activation == "softmax":
                factor_gate = torch.softmax(gate_logits, dim=-1)
            else:
                factor_gate = torch.sigmoid(gate_logits)
            outputs = {
                "alpha": self.alpha_head(hidden).squeeze(-1),
                "direction_logit": self.direction_head(hidden).squeeze(-1),
                "q_low": q_low,
                "q_high": q_high,
                "factor_gate": factor_gate,
                "confidence": torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(hidden)).squeeze(-1),
            }
            if moe_gate is not None:
                outputs["moe_gate"] = moe_gate
            return outputs


    def build_tiny_multitower_model(
        sequence_input_dim: int,
        snapshot_input_dim: int,
        event_numeric_dim: int = 6,
        regime_dim: int = 4,
        lookback: int = 8,
        sequence_backbone: str = "simple_seq",
        factor_gate_dim: int = 4,
        fusion_mode: FusionMode = "moe",
        factor_gate_activation: GateActivation = "softmax",
    ) -> "MultiTowerModel":
        return MultiTowerModel(
            MultiTowerConfig(
                sequence_input_dim=sequence_input_dim,
                snapshot_input_dim=snapshot_input_dim,
                event_numeric_dim=event_numeric_dim,
                regime_dim=regime_dim,
                lookback=lookback,
                hidden_dim=16,
                tower_dim=16,
                dropout=0.0,
                factor_gate_dim=factor_gate_dim,
                sequence_backbone=sequence_backbone,
                num_experts=2,
                fusion_mode=fusion_mode,
                factor_gate_activation=factor_gate_activation,
            )
        )

else:

    class MultiTowerModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("MultiTowerModel requires PyTorch: install quantagent[training]")


    class MoEFusion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("MoEFusion requires PyTorch: install quantagent[training]")


    def build_tiny_multitower_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("MultiTowerModel requires PyTorch: install quantagent[training]")

