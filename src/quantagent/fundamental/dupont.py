from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WACCProvider(Protocol):
    def estimate_wacc(self, symbol: str) -> float:
        ...


@dataclass(frozen=True)
class DuPontResult:
    net_margin: float
    asset_turnover: float
    equity_multiplier: float
    roe: float
    roic: float
    invested_capital: float
    nopat: float


def dupont_decomposition(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
    operating_profit: float,
    cash: float = 0.0,
    tax_rate: float = 0.25,
) -> DuPontResult:
    net_margin_value = net_margin(net_income, revenue)
    turnover = asset_turnover(revenue, total_assets)
    multiplier = equity_multiplier(total_assets, total_equity)
    invested = invested_capital(total_assets, cash)
    nopat_value = nopat(operating_profit, tax_rate)
    return DuPontResult(
        net_margin=net_margin_value,
        asset_turnover=turnover,
        equity_multiplier=multiplier,
        roe=net_margin_value * turnover * multiplier,
        roic=roic(nopat_value, invested),
        invested_capital=invested,
        nopat=nopat_value,
    )


def net_margin(net_income: float, revenue: float) -> float:
    return net_income / revenue if revenue else float("nan")


def asset_turnover(revenue: float, total_assets: float) -> float:
    return revenue / total_assets if total_assets else float("nan")


def equity_multiplier(total_assets: float, total_equity: float) -> float:
    return total_assets / total_equity if total_equity else float("nan")


def invested_capital(total_assets: float, cash: float = 0.0) -> float:
    return max(total_assets - cash, 0.0)


def nopat(operating_profit: float, tax_rate: float = 0.25) -> float:
    return operating_profit * (1.0 - tax_rate)


def roic(nopat_value: float, invested_capital_value: float) -> float:
    return nopat_value / invested_capital_value if invested_capital_value else float("nan")

