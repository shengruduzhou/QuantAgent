from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

if torch is not None:
    from quantagent.models.v4_multitower import build_tiny_v4_model
    from quantagent.training.composite_loss import v4_composite_loss


@dataclass(frozen=True)
class V4TrainingMetadata:
    feature_version: str
    config_hash: str
    train_loss: float
    metrics: dict[str, float]


def build_synthetic_training_batch(
    batch_size: int = 12,
    lookback: int = 8,
    sequence_features: int = 5,
    snapshot_features: int = 7,
    event_features: int = 7,
    seed: int = 0,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("Synthetic V4 training batch requires PyTorch")
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.randn(batch_size, lookback, sequence_features, generator=generator)
    snapshot = torch.randn(batch_size, snapshot_features, generator=generator)
    events = torch.zeros(batch_size, 3, event_features)
    events[..., 0] = torch.randint(0, 5, (batch_size, 3), generator=generator).float()
    events[..., 1:] = torch.randn(batch_size, 3, event_features - 1, generator=generator) * 0.2
    alpha = sequence[:, -1, 0] * 0.02 + snapshot[:, 0] * 0.01 + events[:, :, 2].mean(dim=1) * 0.01
    return {
        "sequence": sequence,
        "snapshot": snapshot,
        "events": events,
        "targets": {
            "alpha": alpha,
            "factor_gate_target": torch.sigmoid(torch.randn(batch_size, 3, generator=generator)),
            "risk_target": torch.sigmoid(torch.randn(batch_size, generator=generator)),
            "factor_icir": torch.tensor([0.2, 0.1, -0.1]),
            "factor_turnover": torch.tensor([0.1, 0.2, 0.3]),
            "factor_corr": torch.tensor([0.1, 0.5, 0.2]),
        },
    }


def train_one_v4_step(batch: dict[str, Any] | None = None, lr: float = 1e-3) -> tuple[Any, V4TrainingMetadata]:
    if torch is None:
        raise ImportError("V4 training requires PyTorch")
    batch = batch or build_synthetic_training_batch()
    model = build_tiny_v4_model(
        sequence_input_dim=batch["sequence"].shape[-1],
        snapshot_input_dim=batch["snapshot"].shape[-1],
        event_numeric_dim=batch["events"].shape[-1] - 1,
        lookback=batch["sequence"].shape[1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["sequence"], batch["snapshot"], batch["events"])
    loss, parts = v4_composite_loss(outputs, batch["targets"])
    loss.backward()
    optimizer.step()
    metadata = V4TrainingMetadata(
        feature_version="synthetic_v4",
        config_hash="synthetic",
        train_loss=float(loss.detach().cpu()),
        metrics={"loss": parts["total"], "rank": parts["rank"], "huber": parts["huber"]},
    )
    return model, metadata


def save_training_metadata(metadata: V4TrainingMetadata, output_path: str | Path) -> Path:
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    _, metadata = train_one_v4_step()
    print(metadata)


if __name__ == "__main__":
    main()
