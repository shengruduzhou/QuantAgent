"""Deterministic A-share opening-auction and continuous-impact estimates.

The model is intentionally conservative and produces expected fills for
research/paper simulation.  It does not submit or recommend orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MarketImpactConfig:
    round_lot: int = 100
    max_participation: float = 0.10
    spread_bps: float = 6.0
    temporary_impact_eta: float = 0.15
    permanent_impact_gamma: float = 0.03
    auction_queue_penalty: float = 1.5
    cancel_risk_penalty: float = 1.0
    min_fill_probability: float = 0.01


@dataclass(frozen=True)
class AuctionSnapshot:
    indicative_price: float
    previous_close: float
    matched_quantity: float
    unmatched_buy_quantity: float
    unmatched_sell_quantity: float
    indicative_volume: float
    cancel_ratio: float = 0.0
    limit_up: float | None = None
    limit_down: float | None = None
    volatility_20d: float = 0.02
    adv20_quantity: float = 0.0


@dataclass(frozen=True)
class AuctionFillEstimate:
    side: str
    requested_quantity: int
    expected_filled_quantity: int
    fill_probability: float
    expected_fill_price: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    queue_ahead_quantity: float
    status: str
    reason: str


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _round_lot(quantity: float, lot: int) -> int:
    lot = max(1, int(lot))
    return max(0, int(quantity) // lot * lot)


def _impact_bps(
    *,
    quantity: int,
    adv_quantity: float,
    volatility: float,
    config: MarketImpactConfig,
) -> tuple[float, float]:
    participation = quantity / max(float(adv_quantity), 1.0)
    temporary = config.temporary_impact_eta * max(volatility, 0.0) * sqrt(max(participation, 0.0))
    permanent = config.permanent_impact_gamma * max(volatility, 0.0) * max(participation, 0.0)
    return temporary * 10_000.0, permanent * 10_000.0


def estimate_opening_auction_fill(
    snapshot: AuctionSnapshot,
    *,
    side: Literal["buy", "sell"],
    quantity: int,
    limit_price: float | None = None,
    config: MarketImpactConfig | None = None,
) -> AuctionFillEstimate:
    cfg = config or MarketImpactConfig()
    requested = _round_lot(quantity, cfg.round_lot)
    if requested <= 0:
        return AuctionFillEstimate(side, int(quantity), 0, 0.0, 0.0, 0.0, 0.0, 0.0, "rejected", "invalid_quantity")
    price = float(snapshot.indicative_price)
    if price <= 0:
        return AuctionFillEstimate(side, requested, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "rejected", "invalid_indicative_price")

    near_limit = 1e-6
    if side == "buy" and snapshot.limit_up is not None and price >= snapshot.limit_up * (1.0 - near_limit):
        return AuctionFillEstimate(side, requested, 0, 0.0, price, 0.0, 0.0, snapshot.unmatched_buy_quantity, "rejected", "sealed_limit_up")
    if side == "sell" and snapshot.limit_down is not None and price <= snapshot.limit_down * (1.0 + near_limit):
        return AuctionFillEstimate(side, requested, 0, 0.0, price, 0.0, 0.0, snapshot.unmatched_sell_quantity, "rejected", "sealed_limit_down")
    if limit_price is not None:
        if side == "buy" and price > float(limit_price):
            return AuctionFillEstimate(side, requested, 0, 0.0, price, 0.0, 0.0, 0.0, "rejected", "limit_below_auction_price")
        if side == "sell" and price < float(limit_price):
            return AuctionFillEstimate(side, requested, 0, 0.0, price, 0.0, 0.0, 0.0, "rejected", "limit_above_auction_price")

    queue_ahead = float(snapshot.unmatched_buy_quantity if side == "buy" else snapshot.unmatched_sell_quantity)
    matched = max(float(snapshot.matched_quantity), float(snapshot.indicative_volume), 1.0)
    order_participation = requested / matched
    queue_ratio = queue_ahead / matched
    imbalance = (
        float(snapshot.unmatched_buy_quantity) - float(snapshot.unmatched_sell_quantity)
    ) / max(
        float(snapshot.unmatched_buy_quantity) + float(snapshot.unmatched_sell_quantity),
        1.0,
    )
    favorable_imbalance = imbalance if side == "sell" else -imbalance
    score = (
        2.0
        - 3.0 * order_participation
        - cfg.auction_queue_penalty * queue_ratio
        - cfg.cancel_risk_penalty * float(np.clip(snapshot.cancel_ratio, 0.0, 1.0))
        + 0.75 * favorable_imbalance
    )
    probability = float(np.clip(_sigmoid(score), cfg.min_fill_probability, 1.0))
    participation_cap = min(cfg.max_participation * matched, requested)
    expected_qty = _round_lot(min(requested * probability, participation_cap), cfg.round_lot)
    temporary_bps, permanent_bps = _impact_bps(
        quantity=max(expected_qty, cfg.round_lot),
        adv_quantity=max(snapshot.adv20_quantity, matched),
        volatility=snapshot.volatility_20d,
        config=cfg,
    )
    direction = 1.0 if side == "buy" else -1.0
    expected_price = price * (
        1.0 + direction * (cfg.spread_bps / 2.0 + temporary_bps + permanent_bps) / 10_000.0
    )
    status = "filled" if expected_qty >= requested else ("partial" if expected_qty > 0 else "unfilled")
    reason = "expected_full_fill" if status == "filled" else (
        "auction_queue_partial" if status == "partial" else "auction_queue_no_fill"
    )
    return AuctionFillEstimate(
        side=side,
        requested_quantity=requested,
        expected_filled_quantity=expected_qty,
        fill_probability=probability,
        expected_fill_price=float(expected_price),
        temporary_impact_bps=float(temporary_bps),
        permanent_impact_bps=float(permanent_bps),
        queue_ahead_quantity=queue_ahead,
        status=status,
        reason=reason,
    )


def estimate_continuous_market_impact_bps(
    *,
    quantity: int,
    adv20_quantity: float,
    volatility_20d: float,
    participation_rate: float,
    config: MarketImpactConfig | None = None,
) -> dict[str, float]:
    cfg = config or MarketImpactConfig()
    if quantity <= 0 or adv20_quantity <= 0:
        return {"temporary_bps": 0.0, "permanent_bps": 0.0, "total_bps": 0.0}
    effective_quantity = min(quantity, int(max(0.0, participation_rate) * adv20_quantity))
    temporary, permanent = _impact_bps(
        quantity=effective_quantity,
        adv_quantity=adv20_quantity,
        volatility=volatility_20d,
        config=cfg,
    )
    total = cfg.spread_bps / 2.0 + temporary + permanent
    return {
        "temporary_bps": float(temporary),
        "permanent_bps": float(permanent),
        "spread_half_bps": float(cfg.spread_bps / 2.0),
        "total_bps": float(total),
    }
