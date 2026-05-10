from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from quantagent.portfolio.sleeve import DEFAULT_SLEEVE_CONFIGS, SleeveAllocationResult, SleeveConfig, SleeveTarget, SleeveType


class SleeveAllocator:
    def __init__(self, configs: Sequence[SleeveConfig] | None = None) -> None:
        self.configs = {config.sleeve_type: config for config in (configs or DEFAULT_SLEEVE_CONFIGS)}

    def allocate(
        self,
        total_nav: float,
        market_regime: str,
        drawdown_state: float,
        volatility_state: float,
        short_alpha_quality: float,
        long_alpha_quality: float,
        sector_rotation_strength: float,
        hedge_need: float,
        sector_breadth: float = 0.5,
        flow_confirmation: float = 0.0,
    ) -> SleeveAllocationResult:
        raw = {sleeve: self.configs[sleeve].default_weight for sleeve in self.configs}
        drawdown = abs(min(drawdown_state, 0.0))
        volatility = max(volatility_state, 0.0)
        crash = market_regime in {"crash", "liquidity_shock"}

        raw[SleeveType.CASH_BUFFER] += 0.55 * volatility + 0.75 * drawdown
        raw[SleeveType.HEDGE] += 0.45 * hedge_need + 0.25 * volatility + 0.35 * drawdown
        raw[SleeveType.SHORT_EVENT] *= float(np.clip(short_alpha_quality, 0.0, 1.0))
        raw[SleeveType.LONG_FUNDAMENTAL] *= float(np.clip(long_alpha_quality, 0.0, 1.0))
        if crash:
            raw[SleeveType.LONG_FUNDAMENTAL] *= 0.35
            raw[SleeveType.CASH_BUFFER] += 0.15
        confirmation = max(sector_breadth - 0.5, 0.0) * 2.0 * max(flow_confirmation, 0.0)
        raw[SleeveType.SECTOR_ROTATION] *= float(np.clip(sector_rotation_strength * confirmation, 0.0, 1.5))

        clipped = {sleeve: self._clip(sleeve, weight) for sleeve, weight in raw.items()}
        normalized = _normalize_with_cash(clipped)
        targets = tuple(
            SleeveTarget(
                sleeve_type=sleeve,
                target_weight=weight,
                confidence=float(np.clip(1.0 - volatility * 0.3 - drawdown * 0.5, 0.0, 1.0)),
                reason=_reason(sleeve, market_regime, volatility, drawdown),
            )
            for sleeve, weight in sorted(normalized.items(), key=lambda item: item[0].value)
        )
        return SleeveAllocationResult(
            targets=targets,
            total_nav=total_nav,
            cash_weight=normalized.get(SleeveType.CASH_BUFFER, 0.0),
            diagnostics={
                "volatility_state": float(volatility),
                "drawdown_abs": float(drawdown),
                "hedge_need": float(hedge_need),
                "sector_confirmation": float(confirmation),
            },
        )

    def _clip(self, sleeve: SleeveType, weight: float) -> float:
        config = self.configs[sleeve]
        return float(np.clip(weight, config.min_weight, config.max_weight))


def _normalize_with_cash(weights: dict[SleeveType, float]) -> dict[SleeveType, float]:
    non_cash_total = sum(weight for sleeve, weight in weights.items() if sleeve != SleeveType.CASH_BUFFER)
    cash = weights.get(SleeveType.CASH_BUFFER, 0.0)
    total = non_cash_total + cash
    if total <= 0:
        return {sleeve: 0.0 for sleeve in weights}
    normalized = {sleeve: weight / total for sleeve, weight in weights.items()}
    min_cash = 0.03
    if normalized.get(SleeveType.CASH_BUFFER, 0.0) < min_cash:
        scale = (1.0 - min_cash) / max(1.0 - normalized.get(SleeveType.CASH_BUFFER, 0.0), 1e-12)
        normalized = {
            sleeve: (min_cash if sleeve == SleeveType.CASH_BUFFER else weight * scale)
            for sleeve, weight in normalized.items()
        }
    return normalized


def _reason(sleeve: SleeveType, market_regime: str, volatility: float, drawdown: float) -> str:
    if sleeve == SleeveType.CASH_BUFFER:
        return f"cash buffer reflects regime={market_regime}, volatility={volatility:.3f}, drawdown={drawdown:.3f}"
    if sleeve == SleeveType.HEDGE:
        return "hedge sleeve responds to beta, drawdown, and volatility pressure"
    if sleeve == SleeveType.SHORT_EVENT:
        return "short-event sleeve is scaled by recent factor quality"
    if sleeve == SleeveType.SECTOR_ROTATION:
        return "sector sleeve requires breadth and flow confirmation"
    return "long-fundamental sleeve is scaled by long-alpha quality and market regime"

