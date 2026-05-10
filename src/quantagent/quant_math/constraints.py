from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import AshareRuleEngine


def round_to_lot_shares(shares: pd.Series, lot_size: int = 100) -> pd.Series:
    rounded = np.floor(shares / lot_size) * lot_size
    return pd.Series(rounded.astype(int), index=shares.index)


def weights_to_lot_shares(
    weights: pd.Series,
    nav: float,
    prices: pd.Series,
    lot_size: int = 100,
) -> pd.Series:
    raw_shares = weights.reindex(prices.index).fillna(0.0) * nav / prices.replace(0, np.nan)
    return round_to_lot_shares(raw_shares.fillna(0.0), lot_size)


def weights_to_board_lot_shares(
    weights: pd.Series,
    nav: float,
    prices: pd.Series,
    side: str | pd.Series = "buy",
    rule_engine: AshareRuleEngine | None = None,
) -> pd.Series:
    engine = rule_engine or AshareRuleEngine()
    raw = weights.reindex(prices.index).fillna(0.0) * nav / prices.replace(0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows: dict[str, int] = {}
    for symbol, quantity in raw.items():
        side_value = str(side.reindex(prices.index).get(symbol, "buy")) if isinstance(side, pd.Series) else str(side)
        rows[str(symbol)] = engine.round_order_quantity(str(symbol), side_value, float(abs(quantity)))
    return pd.Series(rows, dtype=int).reindex(prices.index).fillna(0).astype(int)


def lot_shares_to_weights(shares: pd.Series, nav: float, prices: pd.Series) -> pd.Series:
    if nav <= 0:
        raise ValueError("NAV must be positive")
    return shares.reindex(prices.index).fillna(0) * prices / nav


def liquidity_weight_limit(nav: float, adv: pd.Series, adv_ratio: float = 0.05) -> pd.Series:
    if nav <= 0:
        raise ValueError("NAV must be positive")
    return adv * adv_ratio / nav
