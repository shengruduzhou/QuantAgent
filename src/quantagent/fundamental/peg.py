from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PegConfig:
    """Configuration for A-share PEG valuation overlays."""

    target_pe: float = 30.0
    min_analyst_count: int = 3
    high_digestion_years: float = 4.0
    strong_digestion_years: float = 2.0


@dataclass(frozen=True)
class PegInputs:
    """Single-symbol inputs for PEG and PE-digestion calculations."""

    symbol: str = ""
    price: float | None = None
    pe_ttm: float | None = None
    eps_forward: float | None = None
    eps_current_year: float | None = None
    eps_next_year: float | None = None
    growth_rate: float | None = None
    net_income_start: float | None = None
    net_income_end: float | None = None
    growth_years: float = 3.0
    analyst_count: int | None = None


@dataclass(frozen=True)
class PegValuation:
    """Deterministic valuation overlay; not a trading recommendation."""

    symbol: str
    pe_used: float | None
    forward_pe: float | None
    growth_rate: float | None
    growth_source: str
    peg: float | None
    pe_digestion_years: float | None
    rating: str
    score: float
    confidence: float
    risk_flags: tuple[str, ...]


def forward_pe(price: float | None, eps_forecast: float | None) -> float | None:
    """Return forward PE from price and EPS forecast."""

    price_value = _to_float(price)
    eps_value = _to_float(eps_forecast)
    if price_value is None or eps_value is None or price_value <= 0.0 or eps_value <= 0.0:
        return None
    return price_value / eps_value


def earnings_cagr(start: float | None, end: float | None, years: float = 3.0) -> float | None:
    """Return compound earnings growth as a decimal rate."""

    start_value = _to_float(start)
    end_value = _to_float(end)
    years_value = _to_float(years)
    if (
        start_value is None
        or end_value is None
        or years_value is None
        or start_value <= 0.0
        or end_value <= 0.0
        or years_value <= 0.0
    ):
        return None
    return (end_value / start_value) ** (1.0 / years_value) - 1.0


def peg_ratio(pe: float | None, growth_rate: float | None) -> float | None:
    """Return PEG = PE / growth percentage."""

    pe_value = _to_float(pe)
    growth = normalize_growth_rate(growth_rate)
    if pe_value is None or growth is None or pe_value <= 0.0 or growth <= 0.0:
        return None
    return pe_value / (growth * 100.0)


def pe_digestion_years(
    current_pe: float | None,
    growth_rate: float | None,
    target_pe: float = 30.0,
) -> float | None:
    """Years required for PE to decay to target PE through earnings growth."""

    pe_value = _to_float(current_pe)
    growth = normalize_growth_rate(growth_rate)
    target = _to_float(target_pe)
    if pe_value is None or target is None or pe_value <= 0.0 or target <= 0.0:
        return None
    if pe_value <= target:
        return 0.0
    if growth is None or growth <= 0.0:
        return None
    return math.log(pe_value / target) / math.log1p(growth)


def peg_rating(value: float | None) -> str:
    """Map PEG into a stable valuation bucket."""

    peg_value = _to_float(value)
    if peg_value is None:
        return "not_applicable"
    if peg_value < 0.5:
        return "deep_undervalued"
    if peg_value < 1.0:
        return "undervalued"
    if peg_value < 1.5:
        return "fair"
    if peg_value < 2.0:
        return "expensive"
    return "overvalued"


def normalize_growth_rate(value: float | None) -> float | None:
    """Normalize growth inputs so both 0.30 and 30 mean 30%."""

    result = _to_float(value)
    if result is None:
        return None
    if abs(result) > 1.5:
        result = result / 100.0
    return result


def estimate_peg(inputs: PegInputs, config: PegConfig | None = None) -> PegValuation:
    """Estimate PEG valuation from available A-share data fields."""

    cfg = config or PegConfig()
    fpe = forward_pe(inputs.price, inputs.eps_forward)
    risk_flags: list[str] = []
    if fpe is None and inputs.eps_current_year is not None:
        fpe = forward_pe(inputs.price, inputs.eps_current_year)
    if fpe is None:
        fpe = _positive(inputs.pe_ttm)
        if fpe is None:
            risk_flags.append("missing_valid_pe")
        else:
            risk_flags.append("used_ttm_pe_fallback")
    else:
        if inputs.eps_forward is None and inputs.eps_current_year is not None:
            risk_flags.append("used_current_year_eps_as_forward")

    growth, growth_source = _select_growth(inputs)
    if growth is None:
        risk_flags.append("missing_growth")
    elif growth <= 0.0:
        risk_flags.append("non_positive_growth")
    elif growth < 0.05:
        risk_flags.append("weak_growth_support")

    peg = peg_ratio(fpe, growth)
    digest = pe_digestion_years(fpe, growth, cfg.target_pe)
    rating = peg_rating(peg)

    if inputs.analyst_count is None:
        risk_flags.append("missing_analyst_count")
    elif inputs.analyst_count < cfg.min_analyst_count:
        risk_flags.append("low_analyst_coverage")
    if digest is None and fpe is not None and fpe > cfg.target_pe:
        risk_flags.append("pe_not_digestible")
    elif digest is not None and digest > cfg.high_digestion_years:
        risk_flags.append("long_pe_digestion")
    if peg is not None and peg >= 2.0:
        risk_flags.append("high_peg")

    score = peg_score(peg, digest, cfg)
    confidence = peg_confidence(inputs, fpe, growth, growth_source, cfg)
    return PegValuation(
        symbol=inputs.symbol,
        pe_used=fpe,
        forward_pe=fpe if "used_ttm_pe_fallback" not in risk_flags else None,
        growth_rate=growth,
        growth_source=growth_source,
        peg=peg,
        pe_digestion_years=digest,
        rating=rating,
        score=score,
        confidence=confidence,
        risk_flags=tuple(dict.fromkeys(risk_flags)),
    )


def peg_score(value: float | None, digestion_years: float | None, config: PegConfig | None = None) -> float:
    """Convert PEG and digestion time into a 0-100 valuation score."""

    cfg = config or PegConfig()
    peg_value = _to_float(value)
    if peg_value is None:
        base = 35.0
    elif peg_value < 0.5:
        base = 92.0 + min(8.0, (0.5 - peg_value) / 0.5 * 8.0)
    elif peg_value < 1.0:
        base = 78.0 + (1.0 - peg_value) / 0.5 * 14.0
    elif peg_value < 1.5:
        base = 60.0 + (1.5 - peg_value) / 0.5 * 18.0
    elif peg_value < 2.0:
        base = 42.0 + (2.0 - peg_value) / 0.5 * 18.0
    elif peg_value < 3.0:
        base = 20.0 + (3.0 - peg_value) * 22.0
    else:
        base = max(5.0, 20.0 - (peg_value - 3.0) * 5.0)

    digest = _to_float(digestion_years)
    if digest is None:
        base -= 12.0
    elif digest <= cfg.strong_digestion_years:
        base += 6.0
    elif digest > cfg.high_digestion_years:
        base -= min(25.0, (digest - cfg.high_digestion_years) * 5.0)
    return _clamp(base, 0.0, 100.0)


def peg_confidence(
    inputs: PegInputs,
    pe_used: float | None,
    growth_rate: float | None,
    growth_source: str,
    config: PegConfig | None = None,
) -> float:
    """Estimate data confidence for the PEG overlay."""

    cfg = config or PegConfig()
    confidence = 0.20
    if pe_used is not None:
        confidence += 0.18
    if inputs.eps_forward is not None or inputs.eps_current_year is not None:
        confidence += 0.18
    if growth_rate is not None:
        confidence += 0.18
    if growth_source in {"eps_forecast_growth", "explicit_growth"}:
        confidence += 0.10
    if inputs.analyst_count is not None:
        confidence += min(0.16, 0.16 * inputs.analyst_count / max(cfg.min_analyst_count, 1))
    return _clamp(confidence, 0.0, 1.0)


def enrich_peg_valuation(
    frame: pd.DataFrame,
    config: PegConfig | None = None,
    *,
    prefix: str = "peg_",
) -> pd.DataFrame:
    """Add deterministic PEG overlay columns to a PIT valuation frame."""

    if frame.empty:
        return frame.copy()
    cfg = config or PegConfig()
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        result = estimate_peg(_inputs_from_row(row), cfg)
        rows.append(
            {
                f"{prefix}pe_used": result.pe_used,
                f"{prefix}forward_pe": result.forward_pe,
                f"{prefix}growth_rate": result.growth_rate,
                f"{prefix}growth_source": result.growth_source,
                f"{prefix}ratio": result.peg,
                f"{prefix}digestion_years": result.pe_digestion_years,
                f"{prefix}rating": result.rating,
                f"{prefix}score": result.score,
                f"{prefix}confidence": result.confidence,
                f"{prefix}risk_flags": "|".join(result.risk_flags),
            }
        )
    overlay = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame.copy(), overlay], axis=1)


def _select_growth(inputs: PegInputs) -> tuple[float | None, str]:
    explicit = normalize_growth_rate(inputs.growth_rate)
    if explicit is not None:
        return explicit, "explicit_growth"

    eps_current = _positive(inputs.eps_current_year)
    eps_next = _positive(inputs.eps_next_year)
    if eps_current is not None and eps_next is not None:
        return eps_next / eps_current - 1.0, "eps_forecast_growth"

    income_growth = earnings_cagr(inputs.net_income_start, inputs.net_income_end, inputs.growth_years)
    if income_growth is not None:
        return income_growth, "net_income_cagr"
    return None, "missing"


def _inputs_from_row(row: pd.Series) -> PegInputs:
    return PegInputs(
        symbol=str(_first(row, ("symbol", "ticker", "code")) or ""),
        price=_first(row, ("price", "close", "last_price")),
        pe_ttm=_first(row, ("pe_ttm", "pe", "pe_static")),
        eps_forward=_first(row, ("eps_forward", "consensus_eps", "eps_consensus", "predict_this_year_eps")),
        eps_current_year=_first(row, ("eps_current_year", "eps_cur", "current_year_eps")),
        eps_next_year=_first(row, ("eps_next_year", "eps_next", "next_year_eps", "predict_next_year_eps")),
        growth_rate=_first(row, ("net_income_cagr", "profit_cagr", "profit_growth", "net_income_growth", "net_income_yoy")),
        net_income_start=_first(row, ("net_income_start", "net_income_3y_ago", "profit_start")),
        net_income_end=_first(row, ("net_income_end", "latest_net_income", "profit_end", "net_income")),
        growth_years=float(_first(row, ("growth_years", "cagr_years")) or 3.0),
        analyst_count=_to_int(_first(row, ("analyst_count", "forecast_analyst_count", "coverage_count"))),
    )


def _first(row: pd.Series, names: Iterable[str]) -> object | None:
    for name in names:
        if name in row.index:
            value = row[name]
            if not _is_missing(value):
                return value
    return None


def _positive(value: float | None) -> float | None:
    result = _to_float(value)
    if result is None or result <= 0.0:
        return None
    return result


def _to_float(value: object | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _to_int(value: object | None) -> int | None:
    number = _to_float(value)
    return None if number is None else int(number)


def _is_missing(value: object | None) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))
