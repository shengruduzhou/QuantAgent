from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log, pi, sqrt
from statistics import NormalDist
from typing import Literal

OptionType = Literal["call", "put"]
_N = NormalDist()


@dataclass(frozen=True)
class BlackScholesResult:
    option_type: OptionType
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _validate(spot: float, strike: float, maturity: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")


def _pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _expiry_result(spot: float, strike: float, option_type: OptionType) -> BlackScholesResult:
    call = max(spot - strike, 0.0)
    put = max(strike - spot, 0.0)
    if spot > strike:
        call_delta = 1.0
    elif spot < strike:
        call_delta = 0.0
    else:
        call_delta = 0.5
    delta = call_delta if option_type == "call" else call_delta - 1.0
    return BlackScholesResult(
        option_type=option_type,
        price=float(call if option_type == "call" else put),
        delta=float(delta),
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        rho=0.0,
    )


def _zero_vol_result(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> BlackScholesResult:
    """Deterministic-forward limit of Black-Scholes for positive maturity.

    The exercise boundary is based on discounted forward value, not ``spot > strike``.
    That distinction matters whenever rates or dividends are non-zero.
    """
    disc_r = exp(-rate * maturity)
    disc_q = exp(-dividend_yield * maturity)
    discounted_spot = spot * disc_q
    discounted_strike = strike * disc_r
    call_itm = discounted_spot > discounted_strike
    put_itm = discounted_strike > discounted_spot

    if option_type == "call":
        price = max(discounted_spot - discounted_strike, 0.0)
        delta = disc_q if call_itm else 0.0
        theta = (
            dividend_yield * discounted_spot - rate * discounted_strike
            if call_itm
            else 0.0
        )
        rho = strike * maturity * disc_r if call_itm else 0.0
    else:
        price = max(discounted_strike - discounted_spot, 0.0)
        delta = -disc_q if put_itm else 0.0
        theta = (
            rate * discounted_strike - dividend_yield * discounted_spot
            if put_itm
            else 0.0
        )
        rho = -strike * maturity * disc_r if put_itm else 0.0

    return BlackScholesResult(
        option_type=option_type,
        price=float(price),
        delta=float(delta),
        gamma=0.0,
        vega=0.0,
        theta=float(theta),
        rho=float(rho),
    )


def black_scholes(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """European Black-Scholes price and Greeks.

    ``maturity`` is in years, rates/volatility are decimal annualised values,
    and theta is the conventional calendar-time theta (value decay per year).
    This primitive is a derivatives/risk utility; it is not an equity-alpha factor.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    _validate(spot, strike, maturity, volatility)
    if maturity == 0:
        return _expiry_result(spot, strike, option_type)
    if volatility == 0:
        return _zero_vol_result(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )

    disc_r = exp(-rate * maturity)
    disc_q = exp(-dividend_yield * maturity)
    root_t = sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    nd1 = _N.cdf(d1)
    nd2 = _N.cdf(d2)
    pdf_d1 = _pdf(d1)

    if option_type == "call":
        price = spot * disc_q * nd1 - strike * disc_r * nd2
        delta = disc_q * nd1
        theta = (
            -(spot * disc_q * pdf_d1 * volatility) / (2.0 * root_t)
            - rate * strike * disc_r * nd2
            + dividend_yield * spot * disc_q * nd1
        )
        rho = strike * maturity * disc_r * nd2
    else:
        price = strike * disc_r * _N.cdf(-d2) - spot * disc_q * _N.cdf(-d1)
        delta = disc_q * (nd1 - 1.0)
        theta = (
            -(spot * disc_q * pdf_d1 * volatility) / (2.0 * root_t)
            + rate * strike * disc_r * _N.cdf(-d2)
            - dividend_yield * spot * disc_q * _N.cdf(-d1)
        )
        rho = -strike * maturity * disc_r * _N.cdf(-d2)

    gamma = disc_q * pdf_d1 / (spot * volatility * root_t)
    vega = spot * disc_q * pdf_d1 * root_t
    return BlackScholesResult(
        option_type=option_type,
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def _arbitrage_bounds(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> tuple[float, float]:
    disc_spot = spot * exp(-dividend_yield * maturity)
    disc_strike = strike * exp(-rate * maturity)
    if option_type == "call":
        return max(disc_spot - disc_strike, 0.0), disc_spot
    return max(disc_strike - disc_spot, 0.0), disc_strike


def implied_volatility(
    market_price: float,
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    lower: float = 1e-8,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Recover Black-Scholes implied volatility with fail-closed arbitrage checks."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if market_price < 0:
        raise ValueError("market_price must be non-negative")
    _validate(spot, strike, maturity, lower)
    if maturity <= 0:
        raise ValueError("implied volatility requires positive maturity")
    if lower <= 0 or upper <= lower:
        raise ValueError("require 0 < lower < upper")

    lower_bound, upper_bound = _arbitrage_bounds(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )
    if market_price < lower_bound - tolerance or market_price > upper_bound + tolerance:
        raise ValueError(
            "market price violates European option no-arbitrage bounds: "
            f"price={market_price}, bounds=({lower_bound}, {upper_bound})"
        )
    if abs(market_price - lower_bound) <= tolerance:
        return 0.0

    def price(vol: float) -> float:
        return black_scholes(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol,
            option_type=option_type,
            dividend_yield=dividend_yield,
        ).price

    lo = lower
    hi = upper
    hi_price = price(hi)
    while hi_price < market_price - tolerance and hi < 20.0:
        hi = min(20.0, hi * 2.0)
        hi_price = price(hi)
    if hi_price < market_price - tolerance:
        raise ValueError("could not bracket implied volatility below 2000% annualised vol")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_price = price(mid)
        if abs(mid_price - market_price) <= tolerance:
            return float(mid)
        if mid_price < market_price:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


__all__ = ["BlackScholesResult", "black_scholes", "implied_volatility"]
