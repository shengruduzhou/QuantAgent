"""V7 training dataset builders.

Builders join PIT market features with PIT financial / valuation /
disclosure facts and multi-horizon forward-return labels into a single
gold-tier training frame. Each builder writes a DataManifest alongside
its output and refuses to silently fill gaps with synthetic data.
"""

from quantagent.data.dataset_builder.v7_training_dataset import (
    FORBIDDEN_INFERENCE_COLUMNS,
    REQUIRED_ENTITY_COLUMNS,
    V7TrainingDatasetConfig,
    V7TrainingDatasetResult,
    build_v7_training_dataset_artifact,
    load_fundamentals_root,
    load_table,
)

__all__ = [
    "FORBIDDEN_INFERENCE_COLUMNS",
    "REQUIRED_ENTITY_COLUMNS",
    "V7TrainingDatasetConfig",
    "V7TrainingDatasetResult",
    "build_v7_training_dataset_artifact",
    "load_fundamentals_root",
    "load_table",
]
