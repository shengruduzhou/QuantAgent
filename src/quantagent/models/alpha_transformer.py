from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError(
        "AlphaTransformer requires the training extra: pip install -e .[training]"
    ) from exc


class AlphaTransformer(nn.Module):
    """Multi-horizon alpha and risk predictor for daily cross-sectional ranking."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.supports_mask = False
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features shape: [batch, lookback, num_features]."""
        hidden = self.input_proj(features)
        hidden = self.encoder(hidden)
        return self.head(hidden[:, -1, :])
