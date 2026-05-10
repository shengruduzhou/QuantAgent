from __future__ import annotations

from enum import Enum


class RejectReason(str, Enum):
    MISSING_PRICE = "missing_price"
    SUSPENDED = "suspended"
    LIMIT_UP_NO_BUY = "limit_up_no_buy"
    LIMIT_DOWN_NO_SELL = "limit_down_no_sell"
    INVALID_LOT_QUANTITY = "invalid_lot_quantity"
    T_PLUS_ONE_INSUFFICIENT_AVAILABLE = "t_plus_one_insufficient_available_shares"
    INSUFFICIENT_CASH = "insufficient_cash"
    ZERO_VOLUME = "zero_volume"
