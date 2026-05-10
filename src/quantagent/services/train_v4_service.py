from __future__ import annotations

from quantagent.training.train_v4_multitower import V4TrainingMetadata, train_one_v4_step


def train_v4_synthetic() -> V4TrainingMetadata:
    _, metadata = train_one_v4_step()
    return metadata
