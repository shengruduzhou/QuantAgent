"""V5 sequence backbone factory.

Allows V5MultiTower to swap between SimpleSequenceBackbone,
AlphaTransformer, iTransformer, and PatchTSTAlpha through configuration
without rewriting model code.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from quantagent.models.backbone_base import (
    BackboneSpec,
    ModuleBackboneAdapter,
    SimpleSequenceBackbone,
)


@dataclass(frozen=True)
class BackboneRegistryConfig:
    name: str = "simple_seq"
    input_dim: int = 16
    lookback: int = 20
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.1
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 2
    patch_len: int = 8
    stride: int = 4


SUPPORTED_BACKBONES = ("simple_seq", "alpha_transformer", "itransformer", "patchtst")


def build_sequence_backbone(config: BackboneRegistryConfig):
    """Return a (BackboneBase-compatible) sequence encoder based on config.name."""
    if nn is None:
        raise ImportError("Sequence backbones require PyTorch: install quantagent[training]")

    name = config.name.lower()
    if name == "simple_seq":
        return SimpleSequenceBackbone(
            BackboneSpec(
                input_dim=config.input_dim,
                lookback=config.lookback,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                dropout=config.dropout,
            )
        )
    if name == "alpha_transformer":
        from quantagent.models.alpha_transformer import AlphaTransformer

        module = AlphaTransformer(
            num_features=config.input_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
        return ModuleBackboneAdapter(module, output_dim=config.output_dim)
    if name == "itransformer":
        from quantagent.models.itransformer import iTransformer

        module = iTransformer(
            num_features=config.input_dim,
            lookback=config.lookback,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
        return ModuleBackboneAdapter(module, output_dim=config.output_dim)
    if name == "patchtst":
        from quantagent.models.itransformer import PatchTSTAlpha

        module = PatchTSTAlpha(
            num_features=config.input_dim,
            lookback=config.lookback,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
        return ModuleBackboneAdapter(module, output_dim=config.output_dim)
    raise ValueError(f"Unknown sequence backbone: {config.name}. Supported: {SUPPORTED_BACKBONES}")
