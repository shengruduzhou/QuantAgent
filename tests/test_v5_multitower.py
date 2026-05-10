import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="PyTorch is not installed")


def test_v5_multitower_simple_seq_backbone():
    import torch

    from quantagent.models.v5_multitower import build_tiny_v5_model

    model = build_tiny_v5_model(
        sequence_input_dim=5,
        snapshot_input_dim=7,
        event_numeric_dim=6,
        regime_dim=4,
        lookback=8,
        sequence_backbone="simple_seq",
        factor_group_dim=4,
    )
    outputs = model(
        sequence=torch.randn(4, 8, 5),
        snapshot=torch.randn(4, 7),
        events=torch.randn(4, 3, 7),
        regime=torch.randn(4, 4),
    )
    assert outputs["alpha"].shape == (4,)
    assert outputs["factor_gate"].shape == (4, 4)
    assert torch.allclose(outputs["factor_gate"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert outputs["moe_gate"].shape == (4, 2)
    assert torch.all(outputs["q_low"] <= outputs["q_high"])


def test_v5_multitower_itransformer_backbone():
    import torch

    from quantagent.models.v5_multitower import build_tiny_v5_model

    model = build_tiny_v5_model(
        sequence_input_dim=4,
        snapshot_input_dim=6,
        lookback=8,
        sequence_backbone="itransformer",
        factor_group_dim=3,
    )
    outputs = model(
        sequence=torch.randn(2, 8, 4),
        snapshot=torch.randn(2, 6),
        events=torch.randn(2, 2, 7),
        regime=torch.randn(2, 4),
    )
    assert outputs["alpha"].shape == (2,)
    assert outputs["factor_gate"].shape == (2, 3)
