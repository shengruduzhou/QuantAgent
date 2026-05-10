from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.domain.schemas import TargetWeight
from quantagent.portfolio.sleeve import SleeveType


def horizon_alpha(short_weight: float, horizon_days: int) -> float:
    if horizon_days <= 10:
        return short_weight
    if horizon_days <= 60:
        return 0.5
    return 1.0 - short_weight


def short_signal_to_weight(
    short_signal: float,
    volatility: float,
    vol_target: float = 0.012,
    max_abs_weight: float = 0.04,
) -> float:
    if volatility <= 0:
        return 0.0
    raw = short_signal / volatility * vol_target
    return max(-max_abs_weight, min(max_abs_weight, raw))


def long_score_to_weight(
    long_score: float,
    margin_of_safety: float,
    quality_gate: float,
    max_weight: float = 0.08,
) -> float:
    normalized = max(0.0, min(1.0, long_score / 100.0))
    raw = normalized * max(0.0, margin_of_safety) * max(0.0, min(1.0, quality_gate))
    return max(0.0, min(max_weight, raw))


def combine_short_long_weights(
    symbol: str,
    short_weight: float,
    long_weight: float,
    horizon_days: int,
    confidence: float,
    max_abs_weight: float = 0.10,
) -> TargetWeight:
    alpha = horizon_alpha(0.8, horizon_days)
    combined = alpha * short_weight + (1.0 - alpha) * long_weight
    combined = max(-max_abs_weight, min(max_abs_weight, combined)) * max(0.0, min(1.0, confidence))
    return TargetWeight(
        symbol=symbol,
        target_weight=combined,
        horizon_days=horizon_days,
        confidence=confidence,
        source="weight_adapter",
        reason="combined short-horizon and long-horizon raw weights",
    )


def combine_sleeve_target_weights(
    targets_by_sleeve: Mapping[SleeveType | str, Sequence[TargetWeight]],
    sleeve_weights: Mapping[SleeveType | str, float],
    max_abs_weight: float = 0.10,
) -> list[TargetWeight]:
    """Combine sleeve-local target weights into portfolio-level target weights."""
    rows: dict[str, dict[str, float | int | str]] = {}
    for sleeve, targets in targets_by_sleeve.items():
        sleeve_key = sleeve.value if isinstance(sleeve, SleeveType) else str(sleeve)
        budget = float(sleeve_weights.get(sleeve, sleeve_weights.get(sleeve_key, 0.0)))
        for target in targets:
            contribution = float(np.clip(target.target_weight * budget, -max_abs_weight, max_abs_weight))
            row = rows.setdefault(
                target.symbol,
                {
                    "weight": 0.0,
                    "confidence_weight": 0.0,
                    "confidence_sum": 0.0,
                    "horizon_days": target.horizon_days,
                    "reason": "",
                },
            )
            row["weight"] = float(row["weight"]) + contribution
            row["confidence_weight"] = float(row["confidence_weight"]) + target.confidence * abs(contribution)
            row["confidence_sum"] = float(row["confidence_sum"]) + abs(contribution)
            row["horizon_days"] = min(int(row["horizon_days"]), target.horizon_days)
            row["reason"] = f"{row['reason']};{sleeve_key}:{target.source}" if row["reason"] else f"{sleeve_key}:{target.source}"
    combined: list[TargetWeight] = []
    for symbol, row in rows.items():
        weight = float(np.clip(row["weight"], -max_abs_weight, max_abs_weight))
        denom = float(row["confidence_sum"])
        confidence = float(row["confidence_weight"]) / denom if denom > 0 else 0.0
        combined.append(
            TargetWeight(
                symbol=symbol,
                target_weight=weight,
                horizon_days=int(row["horizon_days"]),
                confidence=confidence,
                source="sleeve_weight_adapter",
                reason=str(row["reason"]),
            )
        )
    return combined


def apply_lot_liquidity_constraints(
    targets: Sequence[TargetWeight],
    nav: float,
    prices: pd.Series,
    liquidity: pd.Series | None = None,
    lot_size: int = 100,
    max_single_name_cash_usage: float = 0.10,
    min_position_threshold: float = 0.001,
) -> list[TargetWeight]:
    """Round target weights to feasible A-share lot and liquidity-aware weights."""
    adjusted: list[TargetWeight] = []
    liquidity = liquidity if liquidity is not None else pd.Series(np.inf, index=prices.index)
    for target in targets:
        price = float(prices.get(target.symbol, np.nan))
        if not np.isfinite(price) or price <= 0 or nav <= 0:
            continue
        raw_cash = target.target_weight * nav
        max_cash = min(abs(raw_cash), nav * max_single_name_cash_usage, float(liquidity.get(target.symbol, np.inf)))
        share_sign = 1.0 if raw_cash >= 0 else -1.0
        shares = np.floor(max_cash / price / lot_size) * lot_size
        feasible_weight = share_sign * shares * price / nav
        if abs(feasible_weight) < min_position_threshold:
            feasible_weight = 0.0
        adjusted.append(
            TargetWeight(
                symbol=target.symbol,
                target_weight=float(feasible_weight),
                horizon_days=target.horizon_days,
                confidence=target.confidence,
                source=target.source,
                reason=f"{target.reason}; lot_liquidity_adjusted",
            )
        )
    return adjusted
