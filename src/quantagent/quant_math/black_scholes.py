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

def black_scholes(*, spot: float, strike: float, maturity: float, rate: float, volatility: float, option_type: OptionType = "call", dividend_yield: float = 0.0) -> BlackScholesResult:
    """European Black-Scholes price and Greeks for derivative risk analysis."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    _validate(spot, strike, maturity, volatility)
    disc_r = exp(-rate * maturity)
    disc_q = exp(-dividend_yield * maturity)
    if maturity == 0 or volatility == 0:
        forward_spot = spot * disc_q
        discounted_strike = strike * disc_r
        call = max(forward_spot - discounted_strike, 0.0)
        put = max(discounted_strike - forward_spot, 0.0)
        if maturity == 0:
            call = max(spot - strike, 0.0)
            put = max(strike - spot, 0.0)
        price = call if option_type == "call" else put
        intrinsic_delta = 1.0 if spot > strike else 0.0
        delta = intrinsic_delta if option_type == "call" else intrinsic_delta - 1.0
        return BlackScholesResult(option_type, float(price), float(delta), 0.0, 0.0, 0.0, 0.0)
    root_t = sqrt(maturity)
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    nd1, nd2 = _N.cdf(d1), _N.cdf(d2)
    pdf_d1 = _pdf(d1)
    if option_type == "call":
        price = spot * disc_q * nd1 - strike * disc_r * nd2
        delta = disc_q * nd1
        theta = -(spot * disc_q * pdf_d1 * volatility) / (2.0 * root_t) - rate * strike * disc_r * nd2 + dividend_yield * spot * disc_q * nd1
        rho = strike * maturity * disc_r * nd2
    else:
        price = strike * disc_r * _N.cdf(-d2) - spot * disc_q * _N.cdf(-d1)
        delta = disc_q * (nd1 - 1.0)
        theta = -(spot * disc_q * pdf_d1 * volatility) / (2.0 * root_t) + rate * strike * disc_r * _N.cdf(-d2) - dividend_yield * spot * disc_q * _N.cdf(-d1)
        rho = -strike * maturity * disc_r * _N.cdf(-d2)
    gamma = disc_q * pdf_d1 / (spot * volatility * root_t)
    vega = spot * disc_q * pdf_d1 * root_t
    return BlackScholesResult(option_type, float(price), float(delta), float(gamma), float(vega), float(theta), float(rho))

def implied_volatility(market_price: float, *, spot: float, strike: float, maturity: float, rate: float, option_type: OptionType = "call", dividend_yield: float = 0.0, lower: float = 1e-6, upper: float = 5.0, tolerance: float = 1e-8, max_iter: int = 200) -> float:
    if market_price < 0:
        raise ValueError("market_price must be non-negative")
    _validate(spot, strike, maturity, lower)
    if maturity <= 0:
        raise ValueError("implied volatility requires positive maturity")
    if lower <= 0 or upper <= lower:
        raise ValueError("require 0 < lower < upper")
    def price(vol: float) -> float:
        return black_scholes(spot=spot, strike=strike, maturity=maturity, rate=rate, volatility=vol, option_type=option_type, dividend_yield=dividend_yield).price
    low_price, high_price = price(lower), price(upper)
    if market_price < low_price - tolerance or market_price > high_price + tolerance:
        raise ValueError("market price is outside the Black-Scholes price range for the volatility bracket")
    lo, hi = lower, upper
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
