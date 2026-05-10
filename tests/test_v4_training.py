import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="PyTorch is not installed")


def test_v4_composite_loss_and_training_step_are_finite():
    import torch

    from quantagent.training.train_v4_multitower import build_synthetic_training_batch, train_one_v4_step

    batch = build_synthetic_training_batch()
    _, metadata = train_one_v4_step(batch)
    assert torch.isfinite(torch.tensor(metadata.train_loss))
    assert metadata.train_loss >= 0 or metadata.train_loss < 0
