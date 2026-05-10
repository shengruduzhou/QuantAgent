from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("Training losses require: pip install -e .[training]") from exc


def alpha_multi_task_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 over return horizons plus lower-weight risk targets."""
    return_loss = F.smooth_l1_loss(pred[:, :3], target[:, :3])
    risk_loss = F.smooth_l1_loss(pred[:, 3:], target[:, 3:])
    return return_loss + 0.5 * risk_loss


def soft_rank(values: torch.Tensor, regularization: float = 1.0) -> torch.Tensor:
    """Differentiable rank approximation via pairwise sigmoid (O(n^2) per row).

    rank_i = 1 + sum_{j != i} sigmoid((v_j - v_i) / tau)
    """
    if values.dim() == 1:
        values = values.unsqueeze(0)
    diff = values.unsqueeze(2) - values.unsqueeze(1)
    pair = torch.sigmoid(diff / max(regularization, 1e-6))
    rank = 1.0 + pair.sum(dim=1) - 0.5
    return rank


def differentiable_spearman_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    regularization: float = 1.0,
) -> torch.Tensor:
    """Negative cross-sectional Spearman correlation using soft ranks."""
    pred_rank = soft_rank(pred, regularization)
    target_rank = soft_rank(target, regularization)
    pred_centered = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
    target_centered = target_rank - target_rank.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denom = torch.sqrt(
        (pred_centered.pow(2).sum(dim=-1) + 1e-8)
        * (target_centered.pow(2).sum(dim=-1) + 1e-8)
    )
    return -(numerator / denom).mean()


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE ranking loss (Xia et al. 2008): differentiable plackett-luce NLL."""
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    order = torch.argsort(target, dim=-1, descending=True)
    sorted_pred = torch.gather(pred, -1, order)
    flipped = torch.flip(sorted_pred, dims=[-1])
    cum_logsumexp = torch.flip(
        torch.logcumsumexp(flipped, dim=-1),
        dims=[-1],
    )
    return (cum_logsumexp - sorted_pred).sum(dim=-1).mean()


def daily_rank_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Backwards-compatible alias to differentiable Spearman loss."""
    return differentiable_spearman_loss(pred, target)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Quantile regression pinball loss."""
    diff = target - pred
    return torch.maximum(quantile * diff, (quantile - 1.0) * diff).mean()
