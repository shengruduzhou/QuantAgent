from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.quant_math.ashare import AshareRuleEngine, TPlusOnePosition


@dataclass(frozen=True)
class TPlusOneSimulationResult:
    fills: pd.DataFrame
    rejects: pd.DataFrame
    positions: dict[str, int]


class TPlusOneExecutionSimulator:
    """Explicit A-share retail execution simulator for order-intent level tests."""

    def __init__(self, rule_engine: AshareRuleEngine | None = None) -> None:
        self.rule_engine = rule_engine or AshareRuleEngine()

    def run(self, intents: pd.DataFrame) -> TPlusOneSimulationResult:
        if intents.empty:
            return TPlusOneSimulationResult(pd.DataFrame(), pd.DataFrame(), {})
        data = intents.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data["_sequence"] = range(len(data))
        data = data.sort_values(["trade_date", "_sequence"]).reset_index(drop=True)
        positions: dict[str, TPlusOnePosition] = {}
        current_date: pd.Timestamp | None = None
        fills: list[dict] = []
        rejects: list[dict] = []
        for _, row in data.iterrows():
            date = row["trade_date"]
            if current_date is None or date > current_date:
                for position in positions.values():
                    position.settle_overnight()
                current_date = date
            symbol = str(row["symbol"])
            positions.setdefault(symbol, TPlusOnePosition())
            position = positions[symbol]
            side = str(row["side"]).lower()
            quantity = self.rule_engine.round_order_quantity(symbol, side, float(row.get("quantity", 0)), date)
            state = {
                "symbol": symbol,
                "trade_date": date,
                "volume": row.get("volume", 1.0),
                "is_suspended": bool(row.get("is_suspended", False)),
                "is_limit_up": bool(row.get("is_limit_up", False)),
                "is_limit_down": bool(row.get("is_limit_down", False)),
                "available_shares": position.available_shares,
            }
            valid, reason = self.rule_engine.validate_order_intent({"symbol": symbol, "side": side, "quantity": quantity}, state)
            if not valid:
                rejects.append({"trade_date": date, "symbol": symbol, "side": side, "quantity": quantity, "reason": reason})
                continue
            price = float(row.get("price", 0.0))
            if side == "buy":
                position.buy(quantity)
            else:
                quantity = position.sell(quantity)
            if quantity <= 0:
                rejects.append({"trade_date": date, "symbol": symbol, "side": side, "quantity": 0, "reason": "zero_fill"})
                continue
            fills.append({"trade_date": date, "symbol": symbol, "side": side, "quantity": quantity, "price": price})
        final_positions = {symbol: pos.total_shares() for symbol, pos in positions.items()}
        return TPlusOneSimulationResult(pd.DataFrame(fills), pd.DataFrame(rejects), final_positions)
