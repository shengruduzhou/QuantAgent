from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HedgeLeg:
    symbol: str
    hedge_type: str = "etf"
    beta: float = 1.0
    notional: float = 0.0


def placeholder_hedge_leg(symbol: str = "510300.SH", notional: float = 0.0) -> HedgeLeg:
    return HedgeLeg(symbol=symbol, hedge_type="etf", beta=1.0, notional=notional)
