from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantagent.agents.views_schema import EvidenceRecord
from quantagent.domain.schemas import AgentSignal


def policy_specialist_signal(row: Mapping[str, Any] | pd.Series, horizon_days: int = 60) -> AgentSignal | None:
    """Convert policy evidence into a directional A-share specialist signal."""

    symbol = _symbol(row)
    if not symbol:
        return None
    strength = _num(row, "policy_strength", "magnitude", default=None)
    direction = _num(row, "policy_direction", "direction", default=1.0)
    authority = _num(row, "source_authority", "authority", default=0.5)
    lag = _num(row, "expected_lag_days", "lag_days", default=0.0)
    if strength is None:
        return None

    lag_decay = _clamp(1.0 - max(lag or 0.0, 0.0) / 180.0, 0.35, 1.0)
    signal_strength = _clamp(float(np.sign(direction or 0.0)) * strength * authority * lag_decay, -1.0, 1.0)
    confidence = _clamp(0.25 + 0.55 * strength + 0.20 * authority, 0.0, 1.0)
    return AgentSignal(
        agent_name="policy_specialist",
        symbol=symbol,
        horizon_days=int(_num(row, "horizon_days", default=horizon_days) or horizon_days),
        signal_strength=signal_strength,
        confidence=confidence,
        evidence_quality=_clamp(authority, 0.0, 1.0),
        risk_penalty=0.0 if signal_strength >= 0.0 else abs(signal_strength) * 0.3,
        tags=("policy", str(_value(row, "theme") or "")),
    )


def hot_money_specialist_signal(row: Mapping[str, Any] | pd.Series, horizon_days: int = 3) -> AgentSignal | None:
    """Convert A-share hot-money and flow fields into a short-horizon signal."""

    symbol = _symbol(row)
    if not symbol:
        return None
    amount = _num(row, "turnover_amount", "amount", default=None)
    main_net = _num(row, "main_net_amount", "main_net", "large_order_net", default=None)
    dragon_net = _num(row, "dragon_tiger_net_buy", "billboard_net_buy", default=None)
    volume_ratio = _num(row, "volume_ratio", "vol_ratio", default=None)
    theme_hot_count = _num(row, "theme_hot_count", "hot_theme_count", default=0.0)
    is_hot_stock = bool(_value(row, "is_hot_stock") or _value(row, "hot_stock"))

    components: list[float] = []
    evidence_points = 0
    if main_net is not None:
        evidence_points += 1
        scale = max(abs(amount or 0.0), 1e8)
        components.append(float(np.tanh((main_net / scale) * 8.0)))
    if dragon_net is not None:
        evidence_points += 1
        scale = max(abs(amount or 0.0), 5e7)
        components.append(float(np.tanh((dragon_net / scale) * 4.0)))
    if volume_ratio is not None:
        evidence_points += 1
        volume_component = float(np.tanh((volume_ratio - 1.0) / 2.0))
        components.append(volume_component)
    if theme_hot_count or is_hot_stock:
        evidence_points += 1
        components.append(_clamp(0.15 + 0.10 * float(theme_hot_count or 0.0), 0.0, 0.45))
    if not components:
        return None

    strength = _clamp(sum(components) / len(components), -1.0, 1.0)
    confidence = _clamp(0.25 + 0.18 * evidence_points + abs(strength) * 0.25, 0.0, 1.0)
    return AgentSignal(
        agent_name="hot_money_specialist",
        symbol=symbol,
        horizon_days=horizon_days,
        signal_strength=strength,
        confidence=confidence,
        evidence_quality=_clamp(0.45 + 0.12 * evidence_points, 0.0, 1.0),
        risk_penalty=0.0 if strength >= 0.0 else abs(strength) * 0.4,
        tags=("hot_money", "fund_flow", "theme"),
    )


def lockup_specialist_signal(row: Mapping[str, Any] | pd.Series, horizon_days: int = 90) -> AgentSignal | None:
    """Convert lockup and announced-reduction fields into supply-pressure risk."""

    symbol = _symbol(row)
    if not symbol:
        return None
    lockup_ratio = _num(row, "lockup_ratio_float", "free_ratio", "unlock_ratio", default=None)
    reduction_ratio = _num(row, "announced_reduction_ratio", "reduction_ratio", default=0.0)
    days_to_lockup = _num(row, "days_to_lockup", "days_until_unlock", default=None)
    if lockup_ratio is None and not reduction_ratio:
        return None

    ratio = max(lockup_ratio or 0.0, 0.0)
    if ratio > 1.5:
        ratio = ratio / 100.0
    reduction = max(reduction_ratio or 0.0, 0.0)
    if reduction > 1.5:
        reduction = reduction / 100.0
    if days_to_lockup is None:
        time_weight = 0.55
    elif days_to_lockup <= 0:
        time_weight = 1.0
    elif days_to_lockup <= horizon_days:
        time_weight = _clamp(1.0 - days_to_lockup / (horizon_days * 1.25), 0.25, 1.0)
    else:
        time_weight = 0.15

    pressure = _clamp(ratio / 0.20 + reduction / 0.05, 0.0, 1.25) * time_weight
    strength = -_clamp(pressure, 0.0, 1.0)
    confidence = _clamp(0.35 + min(0.35, ratio / 0.20 * 0.35) + min(0.20, reduction / 0.05 * 0.20), 0.0, 1.0)
    return AgentSignal(
        agent_name="lockup_specialist",
        symbol=symbol,
        horizon_days=horizon_days,
        signal_strength=strength,
        confidence=confidence,
        evidence_quality=0.70,
        risk_penalty=abs(strength),
        tags=("lockup", "supply_pressure"),
    )


def build_ashare_specialist_signals(frame: pd.DataFrame) -> list[AgentSignal]:
    """Build policy, hot-money, and lockup signals from a feature frame."""

    if frame is None or frame.empty:
        return []
    signals: list[AgentSignal] = []
    for _, row in frame.iterrows():
        for builder in (policy_specialist_signal, hot_money_specialist_signal, lockup_specialist_signal):
            signal = builder(row)
            if signal is not None:
                signals.append(signal)
    return signals


def specialist_evidence_records(signals: list[AgentSignal], timestamp: str) -> list[EvidenceRecord]:
    """Convert specialist signals into the AgentRouter evidence contract."""

    records: list[EvidenceRecord] = []
    for signal in signals:
        records.append(
            EvidenceRecord(
                source=signal.agent_name,
                timestamp=timestamp,
                symbol=signal.symbol,
                event_type="ashare_specialist",
                horizon_days=signal.horizon_days,
                direction=float(np.sign(signal.signal_strength)),
                magnitude=float(abs(signal.signal_strength)),
                confidence=signal.confidence,
                decay_half_life=max(1.0, signal.horizon_days / 2.0),
                rationale="A-share specialist signal",
                raw_reference={"tags": signal.tags, "risk_penalty": signal.risk_penalty},
            )
        )
    return records


def _symbol(row: Mapping[str, Any] | pd.Series) -> str:
    value = _value(row, "symbol") or _value(row, "ticker") or _value(row, "code")
    return "" if value is None else str(value)


def _num(row: Mapping[str, Any] | pd.Series, *keys: str, default: float | None = 0.0) -> float | None:
    for key in keys:
        value = _value(row, key)
        if value is None or _is_missing(value):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return default


def _value(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(row, pd.Series):
        return row[key] if key in row.index else None
    return row.get(key)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))
