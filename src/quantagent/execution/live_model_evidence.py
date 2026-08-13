"""Derive live-model trust facts from bound evidence files.

Summary JSON is not authoritative for facts that can be recomputed from the
underlying data. This module validates FRESH prediction coverage, derives the
frozen strict-backtest return outcome, and re-runs QuantAgent's PBO/DSR/SPA
governance from the complete early-OOS candidate return matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.research.selection_governance import (
    FrozenSelectionGateReport,
    NestedSelectionConfig,
    evaluate_frozen_candidate,
)


@dataclass(frozen=True)
class FreshPredictionEvidence:
    trading_days: int
    start_date: str
    end_date: str
    rows: int
    symbols: int
    session_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrictReturnEvidence:
    trading_days: int
    start_date: str
    end_date: str
    portfolio_total_return: float
    benchmark_total_return: float
    benchmark_excess_positive: bool
    session_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatisticalEvidence:
    report: FrozenSelectionGateReport
    rows: int
    start_date: str
    end_date: str
    session_dates: tuple[str, ...] = ()


def validate_fresh_predictions(path: str | Path) -> FreshPredictionEvidence:
    """Recompute FRESH date coverage and basic prediction integrity."""
    frame = _read_table(Path(path))
    required = {"trade_date", "symbol", "prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fresh_predictions_missing_columns:{','.join(missing)}")
    if frame.empty:
        raise ValueError("fresh_predictions_empty")

    dates = _daily_dates(frame["trade_date"], "fresh_predictions")
    symbols = frame["symbol"].astype(str).str.strip()
    if (symbols == "").any():
        raise ValueError("fresh_predictions_symbol_blank")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce")
    if prediction.isna().any() or not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise ValueError("fresh_predictions_prediction_not_finite")

    keys = pd.DataFrame({"trade_date": dates, "symbol": symbols})
    if keys.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("fresh_predictions_duplicate_symbol_date")
    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    sessions = tuple(value.date().isoformat() for value in unique_dates)
    return FreshPredictionEvidence(
        trading_days=int(len(unique_dates)),
        start_date=sessions[0],
        end_date=sessions[-1],
        rows=int(len(frame)),
        symbols=int(symbols.nunique()),
        session_dates=sessions,
    )


def validate_strict_backtest_returns(path: str | Path) -> StrictReturnEvidence:
    """Derive the frozen strict-backtest outcome from its own return artifact."""
    frame = _read_table(Path(path))
    required = {"trade_date", "portfolio_return", "benchmark_return"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"strict_returns_missing_columns:{','.join(missing)}")
    unexpected = sorted(set(frame.columns).difference(required))
    if unexpected:
        raise ValueError(f"strict_returns_unexpected_columns:{','.join(unexpected)}")
    if frame.empty:
        raise ValueError("strict_returns_empty")

    dates = _daily_dates(frame["trade_date"], "strict_returns")
    if dates.duplicated().any():
        raise ValueError("strict_returns_duplicate_trade_date")
    if not dates.is_monotonic_increasing:
        raise ValueError("strict_returns_not_monotonic")

    numeric = frame[["portfolio_return", "benchmark_return"]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("strict_returns_not_finite")
    if (values < -1.0).any():
        raise ValueError("strict_returns_below_minus_one")

    portfolio_total = float(np.prod(1.0 + numeric["portfolio_return"].to_numpy(dtype=float)) - 1.0)
    benchmark_total = float(np.prod(1.0 + numeric["benchmark_return"].to_numpy(dtype=float)) - 1.0)
    sessions = tuple(value.date().isoformat() for value in dates)
    return StrictReturnEvidence(
        trading_days=int(len(frame)),
        start_date=sessions[0],
        end_date=sessions[-1],
        portfolio_total_return=portfolio_total,
        benchmark_total_return=benchmark_total,
        benchmark_excess_positive=portfolio_total > benchmark_total,
        session_dates=sessions,
    )


def recompute_statistical_evidence(
    path: str | Path,
    *,
    candidate_family: list[str],
    selected_candidate: str,
    cumulative_trials: int,
    max_pbo: float,
    min_dsr_probability: float,
    max_spa_p_value: float,
    minimum_observed_days: int = 80,
) -> StatisticalEvidence:
    """Re-run frozen-candidate PBO/DSR/SPA from the bound early-OOS matrix."""
    frame = _read_table(Path(path))
    required = {"trade_date", "benchmark", *candidate_family}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"statistical_returns_missing_columns:{','.join(missing)}")
    allowed = {"trade_date", "benchmark", *candidate_family}
    unexpected = sorted(set(frame.columns).difference(allowed))
    if unexpected:
        raise ValueError(f"statistical_returns_unexpected_columns:{','.join(unexpected)}")
    if frame.empty:
        raise ValueError("statistical_returns_empty")

    dates = _daily_dates(frame["trade_date"], "statistical_returns")
    if dates.duplicated().any():
        raise ValueError("statistical_returns_duplicate_trade_date")
    if not dates.is_monotonic_increasing:
        raise ValueError("statistical_returns_not_monotonic")

    numeric = frame[["benchmark", *candidate_family]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("statistical_returns_not_finite")
    if selected_candidate not in candidate_family:
        raise ValueError("statistical_returns_selected_candidate_not_in_family")

    index = pd.DatetimeIndex(dates)
    candidates = numeric[candidate_family].copy()
    candidates.index = index
    benchmark = numeric["benchmark"].copy()
    benchmark.index = index
    config = NestedSelectionConfig(
        periods_per_year=252,
        pbo_partitions=8,
        max_pbo=float(max_pbo),
        min_dsr_probability=float(min_dsr_probability),
        max_spa_pvalue=float(max_spa_p_value),
    )
    report = evaluate_frozen_candidate(
        candidates,
        selected_candidate=selected_candidate,
        benchmark_returns=benchmark,
        config=config,
        cumulative_trials=int(cumulative_trials),
        minimum_observed_days=int(minimum_observed_days),
    )
    sessions = tuple(value.date().isoformat() for value in index)
    return StatisticalEvidence(
        report=report,
        rows=int(len(frame)),
        start_date=sessions[0],
        end_date=sessions[-1],
        session_dates=sessions,
    )


def _daily_dates(series: pd.Series, prefix: str) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{prefix}_trade_date_invalid")
    if not dates.dt.normalize().equals(dates):
        raise ValueError(f"{prefix}_trade_date_not_daily")
    return dates


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported_evidence_table_format:{suffix or 'none'}")


__all__ = [
    "FreshPredictionEvidence",
    "StatisticalEvidence",
    "StrictReturnEvidence",
    "recompute_statistical_evidence",
    "validate_fresh_predictions",
    "validate_strict_backtest_returns",
]
