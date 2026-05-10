"""V5 multi-tower alpha model.

Differences from V4MultiTowerModel:
1. Sequence backbone is configurable via BackboneRegistry (simple / alpha_transformer /
   itransformer / patchtst).
2. Tower fusion is a Mixture-of-Experts (MoE) gated by a regime embedding, rather
   than plain concatenation. This lets the model learn per-state weights for the
   sequence, snapshot, and event signals.
3. factor_gate output is a first-class signal that downstream FactorComposite is
   expected to consume; the head is now followed by a softmax (sums to 1) so the
   downstream weights are interpretable as a probability distribution over factor
   groups.
4. Conformal interval support: q_low / q_high heads are kept; the calibrator lives
   in training/conformal_calibrator.py.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from quantagent.models.backbone_registry import (
    BackboneRegistryConfig,
    build_sequence_backbone,
)
from quantagent.models.event_tower import StructuredEventTower, StructuredEventTowerConfig
from quantagent.models.tabular_resnet import TabularResNet, TabularResNetConfig


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


if nn is not None:

    class MoEFusion(nn.Module):
        """Regime-conditional Mixture-of-Experts fusion of three tower outputs."""

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
            gate_logits = self.gate(regime)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            expert_outputs = torch.stack([expert(cat) for expert in self.experts], dim=1)
            fused = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
            return fused, gate_weights


    class V5MultiTowerModel(nn.Module):
        """V5 alpha model: configurable backbone + MoE fusion + interpretable gates."""

        def __init__(self, config: V5MultiTowerConfig) -> None:
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
            self.fusion = MoEFusion(
                tower_dim=config.tower_dim,
                regime_dim=config.regime_dim,
                hidden_dim=config.hidden_dim,
                num_experts=config.num_experts,
                dropout=config.dropout,
            )
            self.alpha_head = nn.Linear(config.hidden_dim, 1)
            self.direction_head = nn.Linear(config.hidden_dim, 1)
            self.quantile_head = nn.Linear(config.hidden_dim, 2)
            self.factor_gate_head = nn.Linear(config.hidden_dim, config.factor_group_dim)
            self.confidence_head = nn.Linear(config.hidden_dim, 1)
            self.risk_head = nn.Linear(config.hidden_dim, 1)

        def forward(
            self,
            sequence: "torch.Tensor",
            snapshot: "torch.Tensor",
            events: "torch.Tensor",
            regime: "torch.Tensor",
            sequence_mask: "torch.Tensor | None" = None,
            event_mask: "torch.Tensor | None" = None,
        ) -> dict[str, "torch.Tensor"]:
            z_seq = self.sequence_tower(sequence, sequence_mask) if hasattr(self.sequence_tower, "supports_mask") and self.sequence_tower.supports_mask else self.sequence_tower(sequence)
            z_snap = self.snapshot_tower(snapshot)
            z_event = self.event_tower(events, event_mask)
            hidden, gate_weights = self.fusion(z_seq, z_snap, z_event, regime)
            quantiles = self.quantile_head(hidden)
            q_low = torch.minimum(quantiles[:, 0], quantiles[:, 1])
            q_high = torch.maximum(quantiles[:, 0], quantiles[:, 1])
            return {
                "alpha": self.alpha_head(hidden).squeeze(-1),
                "direction_logit": self.direction_head(hidden).squeeze(-1),
                "q_low": q_low,
                "q_high": q_high,
                "factor_gate": torch.softmax(self.factor_gate_head(hidden), dim=-1),
                "confidence": torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
                "risk_score": torch.sigmoid(self.risk_head(hidden)).squeeze(-1),
                "moe_gate": gate_weights,
            }


    def build_tiny_v5_model(
        sequence_input_dim: int,
        snapshot_input_dim: int,
        event_numeric_dim: int = 6,
        regime_dim: int = 4,
        lookback: int = 8,
        sequence_backbone: str = "simple_seq",
        factor_group_dim: int = 4,
    ) -> "V5MultiTowerModel":
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

else:

    class V5MultiTowerModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("V5MultiTowerModel requires PyTorch: install quantagent[training]")


    def build_tiny_v5_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("V5MultiTowerModel requires PyTorch: install quantagent[training]")
