from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "iTransformer requires the training extra: pip install -e .[training]"
    ) from exc


class iTransformer(nn.Module):
    """ICLR 2024 iTransformer: invert time-channel, attention across variables.

    Input  : [batch, lookback, num_features]
    Output : [batch, output_dim]
    """

    def __init__(
        self,
        num_features: int,
        lookback: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(lookback, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(num_features * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        tokens = self.token_proj(x)
        encoded = self.norm(self.encoder(tokens))
        flat = encoded.reshape(encoded.shape[0], -1)
        return self.head(flat)


class PatchTSTBackbone(nn.Module):
    """Minimal PatchTST encoder (Nie et al. ICLR 2023). Per-channel patch tokens."""

    def __init__(
        self,
        lookback: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max(1, (lookback - patch_len) // stride + 1)
        self.patch_proj = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model * self.num_patches

    def forward(self, channel_series: torch.Tensor) -> torch.Tensor:
        b, l = channel_series.shape
        patches = channel_series.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[1] < self.num_patches:
            pad = self.num_patches - patches.shape[1]
            patches = torch.nn.functional.pad(patches, (0, 0, 0, pad))
        tokens = self.patch_proj(patches)
        encoded = self.encoder(tokens)
        return encoded.reshape(b, -1)


class PatchTSTAlpha(nn.Module):
    """PatchTST head for cross-sectional alpha multi-task prediction."""

    def __init__(
        self,
        num_features: int,
        lookback: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.backbone = PatchTSTBackbone(
            lookback=lookback,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_features = num_features
        self.head = nn.Sequential(
            nn.Linear(num_features * self.backbone.out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, l, c = features.shape
        per_channel = []
        for i in range(c):
            per_channel.append(self.backbone(features[:, :, i]))
        flat = torch.cat(per_channel, dim=-1)
        return self.head(flat)
