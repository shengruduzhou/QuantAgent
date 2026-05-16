"""FT-Transformer for tabular alpha factors.

The model follows the design of Gorishniy et al., "Revisiting Deep
Learning Models for Tabular Data" (2021): every numerical feature is
projected to a token embedding via a learnable per-feature linear
layer plus bias, a learnable ``[CLS]`` token is prepended, the resulting
token sequence flows through a small Transformer encoder, and the
``[CLS]`` output is mapped to one regression head per forecast horizon.

The class is designed to plug directly into ``V7DeepAlphaTrainer`` —
constructors take the number of features and the number of horizons,
and ``forward`` accepts a ``[batch, num_features]`` tensor. When
PyTorch is not installed the import is a noop placeholder that raises
on construction, matching the convention used by the other deep
models in this directory.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FTTransformerConfig:
    num_features: int
    num_horizons: int = 1
    d_token: int = 64
    n_blocks: int = 3
    n_heads: int = 4
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    ffn_factor: float = 4 / 3
    use_missing_mask: bool = True


if nn is not None:

    class _FeatureTokenizer(nn.Module):
        """Project ``[batch, F]`` numerical features into ``[batch, F, d_token]``.

        Each feature gets its own ``(scale, bias)`` parameters, matching
        the FT-Transformer paper. If ``use_missing_mask`` is set the
        tokenizer concatenates the mask as an extra learnable embedding
        so NaN features remain identifiable.
        """

        def __init__(self, num_features: int, d_token: int, use_missing_mask: bool) -> None:
            super().__init__()
            self.num_features = num_features
            self.d_token = d_token
            self.use_missing_mask = use_missing_mask
            self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
            self.bias = nn.Parameter(torch.zeros(num_features, d_token))
            if use_missing_mask:
                self.missing_embed = nn.Parameter(torch.zeros(num_features, d_token))

        def forward(self, x: "torch.Tensor", missing_mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            if missing_mask is None:
                missing_mask = torch.isnan(x)
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            if self.use_missing_mask:
                tokens = tokens + missing_mask.float().unsqueeze(-1) * self.missing_embed.unsqueeze(0)
            return tokens

    class _CLSToken(nn.Module):
        def __init__(self, d_token: int) -> None:
            super().__init__()
            self.token = nn.Parameter(torch.zeros(1, 1, d_token))

        def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
            batch = tokens.shape[0]
            cls = self.token.expand(batch, -1, -1)
            return torch.cat([cls, tokens], dim=1)

    class _TransformerBlock(nn.Module):
        def __init__(
            self,
            d_token: int,
            n_heads: int,
            attention_dropout: float,
            ffn_dropout: float,
            residual_dropout: float,
            ffn_hidden: int,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.attn = nn.MultiheadAttention(
                d_token, num_heads=n_heads, dropout=attention_dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_token)
            self.ffn = nn.Sequential(
                nn.Linear(d_token, ffn_hidden),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_hidden, d_token),
            )
            self.residual_dropout = nn.Dropout(residual_dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x = x + self.residual_dropout(attn_out)
            x = x + self.residual_dropout(self.ffn(self.norm2(x)))
            return x

    class FTTransformer(nn.Module):
        """FT-Transformer for cross-sectional tabular alpha prediction."""

        supports_mask = True

        def __init__(self, config: FTTransformerConfig) -> None:
            super().__init__()
            self.config = config
            self.tokenizer = _FeatureTokenizer(
                num_features=config.num_features,
                d_token=config.d_token,
                use_missing_mask=config.use_missing_mask,
            )
            self.cls = _CLSToken(config.d_token)
            ffn_hidden = max(1, int(round(config.d_token * config.ffn_factor)))
            self.blocks = nn.ModuleList(
                [
                    _TransformerBlock(
                        d_token=config.d_token,
                        n_heads=config.n_heads,
                        attention_dropout=config.attention_dropout,
                        ffn_dropout=config.ffn_dropout,
                        residual_dropout=config.residual_dropout,
                        ffn_hidden=ffn_hidden,
                    )
                    for _ in range(config.n_blocks)
                ]
            )
            self.norm = nn.LayerNorm(config.d_token)
            self.head = nn.Linear(config.d_token, config.num_horizons)

        def forward(
            self,
            features: "torch.Tensor",
            missing_mask: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            tokens = self.tokenizer(features, missing_mask=missing_mask)
            tokens = self.cls(tokens)
            for block in self.blocks:
                tokens = block(tokens)
            cls_out = self.norm(tokens[:, 0, :])
            return self.head(cls_out)

else:

    class FTTransformer:  # type: ignore[no-redef]
        supports_mask = True

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "FTTransformer requires PyTorch — install quantagent[training]"
            )


__all__ = ["FTTransformer", "FTTransformerConfig"]
