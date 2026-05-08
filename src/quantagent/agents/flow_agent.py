from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal


def northbound_flow_signals(
    flow_frame: pd.DataFrame,
    horizon_days: int = 5,
    z_threshold: float = 1.5,
    window: int = 20,
) -> list[AgentSignal]:
    """A-share specific: HK Stock Connect (北向资金) per-name flow z-score signal.

    Expected columns: symbol, trade_date, holding_value, holding_value_prev.
    """
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
    """A-share 龙虎榜 institutional vs retail seat imbalance.

    Expected columns: symbol, trade_date, inst_buy, inst_sell, retail_buy, retail_sell.
    """
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
