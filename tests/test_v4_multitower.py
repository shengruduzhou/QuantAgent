import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="PyTorch is not installed")


def test_v4_multitower_forward_shapes_and_quantile_ordering():
    import torch

    from quantagent.models.v4_multitower import build_tiny_v4_model

    model = build_tiny_v4_model(sequence_input_dim=5, snapshot_input_dim=7, event_numeric_dim=6, lookback=8)
    outputs = model(torch.randn(4, 8, 5), torch.randn(4, 7), torch.randn(4, 3, 7))
    assert outputs["alpha"].shape == (4,)
    assert outputs["factor_gate"].shape == (4, 3)
    assert torch.all(outputs["q_low"] <= outputs["q_high"])
