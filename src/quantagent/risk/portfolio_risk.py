"""Deterministic portfolio-level risk evidence contract.

This module deliberately separates nonlinear alpha generation from portfolio risk
control.  A target-weight vector can only be described as production-risk-ready
when the systematic-risk evidence attached to it is explicit about coverage,
point-in-time safety, freshness and benchmark alignment.  Missing evidence is
represented as missing; it is never converted to zero exposure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def _normalise_weights(target_weights: pd.Series) -> pd.Series:
    weights = pd.to_numeric(target_weights, errors="coerce").astype(float)
    weights.index = weights.index.map(str)
    weights = weights.groupby(level=0).sum().sort_index()
    return weights


def portfolio_fingerprint(target_weights: pd.Series) -> str:
    """Stable hash binding risk evidence to one exact target-weight vector."""
    weights = _normalise_weights(target_weights)
    payload = [
        [str(symbol), None if not np.isfinite(value) else format(float(value), ".17g")]
        for symbol, value in weights.items()
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _weighted_coverage(weights: pd.Series, valid: pd.Series) -> float:
    gross = float(weights.abs().sum())
    if gross <= 1e-15:
        return 1.0
    mask = valid.reindex(weights.index).fillna(False).astype(bool)
    return float(weights[mask].abs().sum() / gross)


def compute_realized_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    min_overlap: int = 60,
) -> tuple[float | None, int]:
    """Annualised tracking error from frequency-aligned overlapping returns.

    The function performs an inner timestamp join and never forward-fills either
    return stream.  Insufficient overlap is returned as ``None`` rather than a
    fabricated zero.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")
    p = pd.to_numeric(portfolio_returns, errors="coerce").rename("portfolio")
    b = pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark")
    aligned = pd.concat([p, b], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    overlap = int(len(aligned))
    if overlap < int(min_overlap):
        return None, overlap
    active = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=1) * math.sqrt(float(periods_per_year)))
    if not math.isfinite(tracking_error):
        return None, overlap
    return tracking_error, overlap


@dataclass(frozen=True)
class PortfolioRiskSnapshot:
    target_fingerprint: str
    gross_exposure: float
    net_exposure: float

    beta_exposure: float | None = None
    beta_coverage: float = 0.0
    beta_pit_safe: bool | None = None
    beta_freshness_days: float | None = None
    beta_source: str | None = None

    sector_exposures: dict[str, float] = field(default_factory=dict)
    sector_coverage: float = 0.0
    sector_pit_safe: bool | None = None
    sector_freshness_days: float | None = None
    sector_source: str | None = None

    style_exposures: dict[str, float] = field(default_factory=dict)
    style_coverage: dict[str, float] = field(default_factory=dict)
    style_pit_safe: bool | None = None
    style_freshness_days: float | None = None
    style_source: str | None = None

    forecast_volatility: float | None = None
    covariance_coverage: float | None = None
    tracking_error: float | None = None
    tracking_overlap: int = 0
    tracking_frequency: str | None = None
    benchmark_symbol: str | None = None
    as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_portfolio_risk_snapshot(
    target_weights: pd.Series,
    *,
    beta: pd.Series | None = None,
    sector: pd.Series | None = None,
    style_loadings: pd.DataFrame | None = None,
    beta_pit_safe: bool | None = None,
    sector_pit_safe: bool | None = None,
    style_pit_safe: bool | None = None,
    beta_freshness_days: float | None = None,
    sector_freshness_days: float | None = None,
    style_freshness_days: float | None = None,
    beta_source: str | None = None,
    sector_source: str | None = None,
    style_source: str | None = None,
    portfolio_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: float = 252.0,
    min_tracking_overlap: int = 60,
    tracking_frequency: str | None = "daily",
    benchmark_symbol: str | None = None,
    forecast_volatility: float | None = None,
    covariance_coverage: float | None = None,
    as_of: str | None = None,
) -> PortfolioRiskSnapshot:
    weights = _normalise_weights(target_weights)
    finite_weights = weights.replace([np.inf, -np.inf], np.nan)
    gross = float(finite_weights.abs().sum(skipna=True))
    net = float(finite_weights.sum(skipna=True))

    beta_exposure: float | None = None
    beta_coverage = 0.0
    if beta is not None:
        beta_values = pd.to_numeric(beta, errors="coerce")
        beta_values.index = beta_values.index.map(str)
        beta_values = beta_values.groupby(level=0).last().reindex(weights.index)
        valid_beta = beta_values.notna() & np.isfinite(beta_values)
        beta_coverage = _weighted_coverage(finite_weights.fillna(0.0), valid_beta)
        if bool(valid_beta.any()) or gross <= 1e-15:
            beta_exposure = float((finite_weights.fillna(0.0)[valid_beta] * beta_values[valid_beta]).sum())

    sector_exposures: dict[str, float] = {}
    sector_coverage = 0.0
    if sector is not None:
        sector_values = sector.copy()
        sector_values.index = sector_values.index.map(str)
        sector_values = sector_values.groupby(level=0).last().reindex(weights.index)
        text = sector_values.astype("string")
        valid_sector = sector_values.notna() & text.str.strip().ne("")
        sector_coverage = _weighted_coverage(finite_weights.fillna(0.0), valid_sector)
        if bool(valid_sector.any()):
            grouped = finite_weights.fillna(0.0)[valid_sector].groupby(text[valid_sector].astype(str)).sum()
            sector_exposures = {str(name): float(value) for name, value in grouped.items()}

    style_exposures: dict[str, float] = {}
    style_coverage: dict[str, float] = {}
    if style_loadings is not None and not style_loadings.empty:
        styles = style_loadings.copy()
        styles.index = styles.index.map(str)
        styles = styles.groupby(level=0).last().reindex(weights.index)
        for column in styles.columns:
            values = pd.to_numeric(styles[column], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            style_coverage[str(column)] = _weighted_coverage(finite_weights.fillna(0.0), valid)
            if bool(valid.any()) or gross <= 1e-15:
                style_exposures[str(column)] = float(
                    (finite_weights.fillna(0.0)[valid] * values[valid]).sum()
                )

    tracking_error: float | None = None
    tracking_overlap = 0
    if portfolio_returns is not None and benchmark_returns is not None:
        tracking_error, tracking_overlap = compute_realized_tracking_error(
            portfolio_returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
            min_overlap=min_tracking_overlap,
        )

    return PortfolioRiskSnapshot(
        target_fingerprint=portfolio_fingerprint(weights),
        gross_exposure=gross,
        net_exposure=net,
        beta_exposure=beta_exposure,
        beta_coverage=float(beta_coverage),
        beta_pit_safe=beta_pit_safe,
        beta_freshness_days=beta_freshness_days,
        beta_source=beta_source,
        sector_exposures=sector_exposures,
        sector_coverage=float(sector_coverage),
        sector_pit_safe=sector_pit_safe,
        sector_freshness_days=sector_freshness_days,
        sector_source=sector_source,
        style_exposures=style_exposures,
        style_coverage=style_coverage,
        style_pit_safe=style_pit_safe,
        style_freshness_days=style_freshness_days,
        style_source=style_source,
        forecast_volatility=forecast_volatility,
        covariance_coverage=covariance_coverage,
        tracking_error=tracking_error,
        tracking_overlap=int(tracking_overlap),
        tracking_frequency=tracking_frequency,
        benchmark_symbol=benchmark_symbol,
        as_of=as_of,
    )


__all__ = [
    "PortfolioRiskSnapshot",
    "build_portfolio_risk_snapshot",
    "compute_realized_tracking_error",
    "portfolio_fingerprint",
]
