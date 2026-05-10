from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class BackboneProtocol(Protocol):
    output_dim: int
    supports_mask: bool

    def forward(self, x, mask=None): ...


@dataclass(frozen=True)
class BackboneSpec:
    input_dim: int
    lookback: int
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.1


if nn is not None:

    class BackboneBase(nn.Module):
        output_dim: int
        supports_mask: bool = False

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            raise NotImplementedError


    class SimpleSequenceBackbone(BackboneBase):
        """Small sequence encoder for CPU tests."""

        supports_mask = True

        def __init__(self, spec: BackboneSpec) -> None:
            super().__init__()
            self.output_dim = spec.output_dim
            self.input_proj = nn.Linear(spec.input_dim, spec.hidden_dim)
            self.encoder = nn.GRU(
                input_size=spec.hidden_dim,
                hidden_size=spec.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(spec.hidden_dim),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.hidden_dim, spec.output_dim),
            )

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            hidden = self.input_proj(x)
            encoded, _ = self.encoder(hidden)
            if mask is not None:
                weights = mask.float().unsqueeze(-1)
                pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            else:
                pooled = encoded[:, -1, :]
            return self.head(pooled)


    class ModuleBackboneAdapter(BackboneBase):
        """Adapter for existing AlphaTransformer or iTransformer-style modules."""

        supports_mask = False

        def __init__(self, module: nn.Module, output_dim: int) -> None:
            super().__init__()
            self.module = module
            self.output_dim = output_dim

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            del mask
            return self.module(x)

else:

    class BackboneBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("BackboneBase requires PyTorch: install quantagent[training]")


    class SimpleSequenceBackbone(BackboneBase):  # type: ignore[no-redef]
        pass


    class ModuleBackboneAdapter(BackboneBase):  # type: ignore[no-redef]
        pass
