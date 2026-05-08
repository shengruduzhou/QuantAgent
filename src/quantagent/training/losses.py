from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("Training losses require: pip install -e .[training]") from exc


def alpha_multi_task_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 over return horizons plus lower-weight risk targets."""
    return_loss = F.smooth_l1_loss(pred[:, :3], target[:, :3])
    risk_loss = F.smooth_l1_loss(pred[:, 3:], target[:, 3:])
    return return_loss + 0.5 * risk_loss


def daily_rank_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiability is limited; use mainly as an auxiliary training signal."""
    pred_rank = torch.argsort(torch.argsort(pred)).float()
    target_rank = torch.argsort(torch.argsort(target)).float()
    pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-6)
    target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-6)
    return -(pred_rank * target_rank).mean()
