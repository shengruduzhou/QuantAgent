from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CompositeLossWeights:
    rank: float = 1.0
    huber: float = 1.0
    direction: float = 0.5
    quantile: float = 0.5
    factor_gate: float = 0.2
    turnover: float = 0.1
    risk: float = 0.2


if torch is not None:
    from quantagent.training.losses import differentiable_spearman_loss, pinball_loss

    def v4_composite_loss(
        outputs: dict[str, "torch.Tensor"],
        targets: dict[str, "torch.Tensor"],
        weights: CompositeLossWeights | None = None,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        cfg = weights or CompositeLossWeights()
        alpha = torch.nan_to_num(outputs["alpha"].float(), nan=0.0)
        target_alpha = torch.nan_to_num(targets["alpha"].float(), nan=0.0)
        rank_loss = differentiable_spearman_loss(alpha, target_alpha)
        huber = F.smooth_l1_loss(alpha, target_alpha)
        direction_target = (target_alpha > 0).float()
        direction = F.binary_cross_entropy_with_logits(outputs["direction_logit"].float(), direction_target)
        q_low = torch.minimum(outputs["q_low"].float(), outputs["q_high"].float())
        q_high = torch.maximum(outputs["q_low"].float(), outputs["q_high"].float())
        quantile = pinball_loss(q_low, target_alpha, 0.1) + pinball_loss(q_high, target_alpha, 0.9)
        gate = _factor_gate_loss(outputs.get("factor_gate"), targets)
        turnover = _turnover_penalty(targets)
        risk = _risk_penalty(outputs, targets)
        total = (
            cfg.rank * rank_loss
            + cfg.huber * huber
            + cfg.direction * direction
            + cfg.quantile * quantile
            + cfg.factor_gate * gate
            + cfg.turnover * turnover
            + cfg.risk * risk
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=1e6)
        parts = {
            "rank": float(rank_loss.detach().cpu()),
            "huber": float(huber.detach().cpu()),
            "direction": float(direction.detach().cpu()),
            "quantile": float(quantile.detach().cpu()),
            "factor_gate": float(gate.detach().cpu()),
            "turnover": float(turnover.detach().cpu()),
            "risk": float(risk.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total, parts


    def _factor_gate_loss(gate: "torch.Tensor | None", targets: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        if gate is None:
            return torch.tensor(0.0, device=next(iter(targets.values())).device)
        loss = torch.tensor(0.0, device=gate.device)
        if "factor_gate_target" in targets:
            target = targets["factor_gate_target"].float().to(gate.device)
            loss = loss + F.mse_loss(gate, target)
        if "factor_icir" in targets:
            reward = torch.nan_to_num(targets["factor_icir"].float().to(gate.device), nan=0.0)
            while reward.dim() < gate.dim():
                reward = reward.unsqueeze(0)
            loss = loss - (gate * reward).mean()
        if "factor_turnover" in targets:
            loss = loss + (gate * targets["factor_turnover"].float().to(gate.device)).mean()
        if "factor_corr" in targets:
            loss = loss + (gate * targets["factor_corr"].float().abs().to(gate.device)).mean()
        entropy = -(gate.clamp(1e-6, 1 - 1e-6) * gate.clamp(1e-6, 1 - 1e-6).log()).mean()
        return loss - 0.01 * entropy


    def _turnover_penalty(targets: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        if "weights" in targets and "previous_weights" in targets:
            return (targets["weights"].float() - targets["previous_weights"].float()).abs().mean()
        device = next(iter(targets.values())).device
        return torch.tensor(0.0, device=device)


    def _risk_penalty(outputs: dict[str, "torch.Tensor"], targets: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        risk = outputs.get("risk_score")
        if risk is None:
            return torch.tensor(0.0, device=next(iter(targets.values())).device)
        loss = torch.tensor(0.0, device=risk.device)
        if "risk_target" in targets:
            loss = loss + F.mse_loss(risk.float(), targets["risk_target"].float().to(risk.device))
        for key in ("style_exposure", "sector_exposure", "beta_exposure", "volatility_exposure"):
            if key in targets:
                loss = loss + targets[key].float().abs().mean().to(risk.device)
        return loss

else:

    def v4_composite_loss(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("v4_composite_loss requires PyTorch: install quantagent[training]")
