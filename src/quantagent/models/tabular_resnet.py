from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TabularResNetConfig:
    input_dim: int
    hidden_dim: int = 64
    output_dim: int = 64
    num_blocks: int = 2
    dropout: float = 0.1
    use_missing_mask: bool = True


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.net(x)


    class TabularResNet(nn.Module):
        supports_mask = True

        def __init__(self, config: TabularResNetConfig) -> None:
            super().__init__()
            self.config = config
            in_dim = config.input_dim * 2 if config.use_missing_mask else config.input_dim
            self.output_dim = config.output_dim
            self.input = nn.Sequential(nn.Linear(in_dim, config.hidden_dim), nn.GELU())
            self.blocks = nn.Sequential(*[ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.num_blocks)])
            self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.output_dim))

        def forward(self, x: "torch.Tensor", missing_mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            if missing_mask is None:
                missing_mask = torch.isnan(x)
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if self.config.use_missing_mask:
                x = torch.cat([x, missing_mask.float()], dim=-1)
            hidden = self.input(x)
            return self.output(self.blocks(hidden))

else:

    class TabularResNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("TabularResNet requires PyTorch: install quantagent[training]")
