from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.ashare.fund_flow import build_flow_feature_frame, flow_signals_from_features
from quantagent.domain.schemas import AgentSignal


def northbound_flow_signals(
    flow_frame: pd.DataFrame,
    horizon_days: int = 5,
    z_threshold: float = 1.5,
    window: int = 20,
) -> list[AgentSignal]:
    """A-share Stock Connect per-name flow z-score signal."""
    required = {"symbol", "trade_date", "holding_value"}
    missing = required.difference(flow_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = flow_frame.copy().sort_values(["symbol", "trade_date"])
    data["delta"] = data.groupby("symbol")["holding_value"].diff()
    data["z"] = data.groupby("symbol")["delta"].transform(
        lambda s: (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-12)
    )
    latest = data.groupby("symbol").tail(1)
    signals: list[AgentSignal] = []
    for _, row in latest.iterrows():
        z = float(row["z"]) if not pd.isna(row["z"]) else 0.0
        if abs(z) < z_threshold:
            continue
        strength = float(np.tanh(z / 3.0))
        signals.append(
            AgentSignal(
                agent_name="northbound_flow_agent",
                symbol=str(row["symbol"]),
                horizon_days=horizon_days,
                signal_strength=strength,
                confidence=min(1.0, abs(z) / 5.0),
                evidence_quality=0.7,
                risk_penalty=0.0,
                tags=("northbound", "flow"),
            )
        )
    return signals


def dragon_tiger_signals(
    list_frame: pd.DataFrame,
    horizon_days: int = 3,
) -> list[AgentSignal]:
    """A-share dragon-tiger institutional versus retail seat imbalance."""
    required = {"symbol", "inst_buy", "inst_sell", "retail_buy", "retail_sell"}
    missing = required.difference(list_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = list_frame.copy()
    data["inst_net"] = data["inst_buy"] - data["inst_sell"]
    data["retail_net"] = data["retail_buy"] - data["retail_sell"]
    data["imbalance"] = data["inst_net"] - data["retail_net"]
    data["scale"] = data[["inst_buy", "inst_sell", "retail_buy", "retail_sell"]].sum(axis=1)
    signals: list[AgentSignal] = []
    for _, row in data.iterrows():
        scale = float(row["scale"])
        if scale <= 0:
            continue
        ratio = float(row["imbalance"] / scale)
        signals.append(
            AgentSignal(
                agent_name="dragon_tiger_agent",
                symbol=str(row["symbol"]),
                horizon_days=horizon_days,
                signal_strength=float(np.tanh(ratio * 3.0)),
                confidence=min(1.0, abs(ratio) * 2.0),
                evidence_quality=0.55,
                risk_penalty=0.0,
                tags=("dragon_tiger",),
            )
        )
    return signals


def multi_source_flow_signals(
    source_frames: dict[str, pd.DataFrame],
    z_threshold: float = 1.0,
    window: int = 20,
) -> list[AgentSignal]:
    """Combine normalized A-share capital-flow sources into agent signals."""
    feature_frame = build_flow_feature_frame(source_frames, window=window)
    flow_signals = flow_signals_from_features(feature_frame.frame, z_threshold=z_threshold)
    signals: list[AgentSignal] = []
    for signal in flow_signals:
        signals.append(
            AgentSignal(
                agent_name=f"{signal.source}_flow_agent",
                symbol=signal.symbol,
                horizon_days=signal.horizon_days,
                signal_strength=signal.score,
                confidence=signal.confidence,
                evidence_quality=signal.evidence_quality,
                risk_penalty=0.0,
                tags=("flow", signal.source),
            )
        )
    return signals


def combined_flow_signal_by_symbol(signals: list[AgentSignal]) -> pd.DataFrame:
    """Aggregate flow signals while preserving source-level evidence quality."""
    if not signals:
        return pd.DataFrame(columns=["symbol", "signal_strength", "confidence", "evidence_quality", "horizon_days"])
    rows = [signal.__dict__ for signal in signals]
    data = pd.DataFrame(rows)
    data["weight"] = data["confidence"] * data["evidence_quality"]
    grouped = data.groupby("symbol", sort=False)
    return grouped.apply(_weighted_signal_row, include_groups=False).reset_index()


def _weighted_signal_row(group: pd.DataFrame) -> pd.Series:
    weight = group["weight"].clip(lower=0.0)
    denom = weight.sum()
    strength = group["signal_strength"].mean() if denom <= 0 else float((group["signal_strength"] * weight).sum() / denom)
    return pd.Series(
        {
            "signal_strength": float(strength),
            "confidence": float(group["confidence"].mean()),
            "evidence_quality": float(group["evidence_quality"].mean()),
            "horizon_days": int(round(group["horizon_days"].median())),
        }
    )
