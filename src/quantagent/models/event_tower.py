from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StructuredEventTowerConfig:
    numeric_dim: int = 6
    event_type_count: int = 32
    event_type_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.1


if nn is not None:

    class StructuredEventTower(nn.Module):
        """Structured event-policy tower without external LLM dependency."""

        def __init__(self, config: StructuredEventTowerConfig | None = None) -> None:
            super().__init__()
            self.config = config or StructuredEventTowerConfig()
            self.output_dim = self.config.output_dim
            self.type_embedding = nn.Embedding(self.config.event_type_count, self.config.event_type_dim)
            self.event_mlp = nn.Sequential(
                nn.Linear(self.config.numeric_dim + self.config.event_type_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.output_dim),
            )

        def forward(self, events: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            if events.dim() == 2:
                events = events.unsqueeze(1)
            event_type = events[..., 0].long().clamp(0, self.config.event_type_count - 1)
            numeric = torch.nan_to_num(events[..., 1 : 1 + self.config.numeric_dim].float(), nan=0.0)
            if numeric.shape[-1] < self.config.numeric_dim:
                numeric = torch.nn.functional.pad(numeric, (0, self.config.numeric_dim - numeric.shape[-1]))
            embedded = self.type_embedding(event_type)
            encoded = self.event_mlp(torch.cat([embedded, numeric], dim=-1))
            if mask is None:
                confidence = numeric[..., 2].clamp(0.0, 1.0) if numeric.shape[-1] >= 3 else torch.ones_like(event_type, dtype=torch.float32)
                recency = numeric[..., 5].clamp_min(0.0) if numeric.shape[-1] >= 6 else torch.zeros_like(confidence)
                weights = confidence * torch.exp(-recency / 20.0)
            else:
                weights = mask.float()
            weights = weights.unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

else:

    class StructuredEventTower:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("StructuredEventTower requires PyTorch: install quantagent[training]")
