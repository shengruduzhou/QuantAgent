"""A-share differentiated dividend tax by holding period (R10-F05).

财税〔2015〕101号: cash dividends on listed A-shares are paid gross and the
individual income tax is settled when the shares are sold, by how long that
lot was held:

* held 1 month or less (含1个月) -> 20% of the dividend received on it;
* more than 1 month up to 1 year (含1年) -> 10%;
* more than 1 year -> exempt.

Lots are first-in-first-out per symbol. Bonus / transfer shares (送转股)
inherit the acquisition date of the lot they were issued on, which keeps the
dividend accrued per lot unchanged in total. The tax on the stock-dividend part
of 送股 (taxed at its 1 CNY par value) is not modelled -- stated, not guessed.
Shares still held at the end carry a *latent* tax that a mark-to-market NAV
does not deduct; :meth:`latent_tax` reports it so a reader can bound it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

RATE_WITHIN_ONE_MONTH = 0.20
RATE_WITHIN_ONE_YEAR = 0.10
RATE_OVER_ONE_YEAR = 0.0
POLICY = (
    "财税〔2015〕101号 differentiated dividend tax: 20% (held <=1 month), 10% "
    "(<=1 year), 0% (>1 year), FIFO lots, settled at sale; 送股 par-value tax "
    "not modelled; latent tax on open lots disclosed, not deducted from NAV"
)


def holding_period_rate(acquired: pd.Timestamp, sold: pd.Timestamp) -> float:
    acquired = pd.Timestamp(acquired).normalize()
    sold = pd.Timestamp(sold).normalize()
    if sold <= acquired + pd.DateOffset(months=1):
        return RATE_WITHIN_ONE_MONTH
    if sold <= acquired + pd.DateOffset(years=1):
        return RATE_WITHIN_ONE_YEAR
    return RATE_OVER_ONE_YEAR


@dataclass
class _Lot:
    acquired: pd.Timestamp
    shares: float
    dividend_per_share: float = 0.0


@dataclass
class DividendTaxLots:
    """FIFO lot book that turns gross dividend credits into tax due at sale."""

    lots: dict[str, deque] = field(default_factory=dict)
    tax_paid: float = 0.0
    dividends_gross: float = 0.0

    def buy(self, symbol: str, when, shares: float) -> None:
        if shares > 0:
            self.lots.setdefault(symbol, deque()).append(_Lot(pd.Timestamp(when), float(shares)))

    def dividend(self, symbol: str, cash_per_share: float) -> None:
        for lot in self.lots.get(symbol, ()):
            lot.dividend_per_share += float(cash_per_share)
            self.dividends_gross += lot.shares * float(cash_per_share)

    def bonus(self, symbol: str, share_ratio: float) -> None:
        if share_ratio <= 0:
            return
        for lot in self.lots.get(symbol, ()):
            lot.shares *= 1.0 + share_ratio
            lot.dividend_per_share /= 1.0 + share_ratio

    def sell(self, symbol: str, when, shares: float) -> float:
        """Consume FIFO lots; return the dividend tax due on them."""
        remaining = float(shares)
        tax = 0.0
        queue = self.lots.get(symbol, deque())
        while remaining > 1e-9 and queue:
            lot = queue[0]
            take = min(lot.shares, remaining)
            tax += take * lot.dividend_per_share * holding_period_rate(lot.acquired, when)
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-9:
                queue.popleft()
        self.tax_paid += tax
        return tax

    def write_off(self, symbol: str) -> None:
        self.lots.pop(symbol, None)

    def latent_tax(self, when) -> float:
        return sum(
            lot.shares * lot.dividend_per_share * holding_period_rate(lot.acquired, when)
            for queue in self.lots.values() for lot in queue
        )


__all__ = [
    "DividendTaxLots", "POLICY", "RATE_OVER_ONE_YEAR", "RATE_WITHIN_ONE_MONTH",
    "RATE_WITHIN_ONE_YEAR", "holding_period_rate",
]
