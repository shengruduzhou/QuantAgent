"""OOM-resilient wrapper around :class:`FTTransformerTrainer`.

The FT-Transformer trainer batches by trade-date, so the dominant memory
levers are ``d_token`` and ``n_blocks`` (model width / depth), with
``ffn_dropout`` / ``attention_dropout`` having minor effect.

This helper catches ``torch.cuda.OutOfMemoryError`` (or the generic
``RuntimeError("out of memory")`` raised by older torch) and retries with
a smaller model. The plan's "batch ladder" is preserved via a
``mini_batch_size`` knob: if a date holds more rows than the mini-batch
limit, the trainer splits within the date as well.
"""

from __future__ import annotations

from dataclasses import replace
import gc
import logging

import pandas as pd

from quantagent.training.ft_transformer_trainer import (
    FTTransformerArtifacts,
    FTTransformerTrainer,
    FTTransformerTrainerConfig,
)

logger = logging.getLogger(__name__)

# Ordered ladder: each entry overrides FTTransformerTrainerConfig fields.
# Applied left-to-right on successive retries.
_DEFAULT_LADDER: tuple[dict[str, object], ...] = (
    {},  # try as-is first
    {"d_token": 48},
    {"d_token": 32, "n_blocks": 2},
    {"d_token": 24, "n_blocks": 2, "ffn_dropout": 0.2, "attention_dropout": 0.2},
)


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def train_with_oom_retry(
    base_config: FTTransformerTrainerConfig,
    dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame | None = None,
    *,
    ladder: tuple[dict[str, object], ...] | None = None,
) -> FTTransformerArtifacts:
    """Run FTTransformerTrainer, retrying with smaller models on OOM."""
    steps = ladder or _DEFAULT_LADDER
    last_exc: BaseException | None = None
    for i, overrides in enumerate(steps):
        cfg = replace(base_config, **overrides) if overrides else base_config
        if overrides:
            logger.warning(
                "OOM retry %d/%d — applying overrides: %s",
                i,
                len(steps) - 1,
                overrides,
            )
        try:
            return FTTransformerTrainer(cfg).fit_and_save(dataset, validation_dataset)
        except BaseException as exc:  # noqa: BLE001 — torch raises both
            if not _is_oom(exc):
                raise
            last_exc = exc
            try:
                import torch  # type: ignore
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            continue
    assert last_exc is not None
    raise last_exc


__all__ = ["train_with_oom_retry"]
