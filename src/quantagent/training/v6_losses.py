from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def rank_loss(pred, target):
    import torch

    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = torch.sign(target.unsqueeze(0) - target.unsqueeze(1))
    return torch.relu(-pred_diff * target_diff).mean()


def pinball_loss(pred, target, quantile: float):
    import torch

    diff = target - pred
    return torch.maximum(quantile * diff, (quantile - 1.0) * diff).mean()


def multi_horizon_consistency_loss(alpha_1d, alpha_5d, alpha_20d):
    import torch

    return torch.mean(torch.relu(torch.abs(alpha_1d) - torch.abs(alpha_5d) + 1e-6)) + torch.mean(torch.relu(torch.abs(alpha_5d) - torch.abs(alpha_20d) + 1e-6))

