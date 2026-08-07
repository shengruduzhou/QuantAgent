"""Cross-engine economic reconciliation.

An engine that agrees with itself proves nothing. These modules take the
canonical event log two engines produced for the *same* economic scenario and
compare every figure that can move money: order state, quantities, fills,
rejections, fees, taxes, slippage, lots, settled and sellable inventory, cash,
reserved cash, realised and unrealised PnL and NAV.

The output is a difference table, not a boolean. Every difference must carry a
classification and the rule that explains it; anything left over is counted as
`unexplained_economic_differences`, and that count is the gate.
"""

from quantagent.reconciliation.differences import (
    BOUNDED_FLOAT,
    DOCUMENTED_ENGINE_DIFFERENCE,
    Difference,
    DifferenceTable,
    ExplanationRule,
    UNEXPLAINED,
    compare_snapshots,
)
from quantagent.reconciliation.snapshot import EconomicSnapshot, OrderFacts

__all__ = [
    "BOUNDED_FLOAT",
    "DOCUMENTED_ENGINE_DIFFERENCE",
    "Difference",
    "DifferenceTable",
    "EconomicSnapshot",
    "ExplanationRule",
    "OrderFacts",
    "UNEXPLAINED",
    "compare_snapshots",
]
