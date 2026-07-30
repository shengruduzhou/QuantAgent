"""Auditable factor-library evaluation and pre-training screening."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.factors.evaluation import factor_summary_table


@dataclass(frozen=True)
class FactorScreeningConfig:
    min_finite_ratio: float = 0.30
    min_abs_rank_ic: float = 0.005
    min_abs_rank_icir: float = 0.10
    min_abs_monotonicity: float = 0.15
    max_pairwise_correlation: float = 0.85
    quantiles: int = 5
    correlation_max_dates: int = 60
    correlation_max_symbols_per_date: int = 1_000


@dataclass(frozen=True)
class FactorScreeningResult:
    summary: pd.DataFrame
    correlation: pd.DataFrame
    selected_factors: tuple[str, ...]
    rejected_factors: tuple[str, ...]
    output_dir: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.selected_factors else "no_factor_passed",
            "evaluated": int(len(self.summary)),
            "selected": int(len(self.selected_factors)),
            "rejected": int(len(self.rejected_factors)),
            "selected_factors": list(self.selected_factors),
            "rejected_factors": list(self.rejected_factors),
            "summary_path": str(self.output_dir / "factor_summary.csv"),
            "correlation_path": str(self.output_dir / "factor_correlation.csv"),
            "selection_path": str(self.output_dir / "factor_selection.json"),
        }


def chronological_calibration_slice(
    frame: pd.DataFrame,
    *,
    calibration_days: int,
    holdout_days: int,
    date_column: str = "trade_date",
    minimum_calibration_days: int = 20,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze factor selection on early dates and leave later dates unseen."""

    if date_column not in frame.columns:
        raise ValueError(f"factor calibration requires {date_column}")
    if calibration_days < minimum_calibration_days:
        raise ValueError(f"calibration_days must be at least {minimum_calibration_days}")
    if holdout_days < 1:
        raise ValueError("holdout_days must be positive")
    dates = (
        pd.to_datetime(frame[date_column], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    resolved_days = min(int(calibration_days), len(dates) - int(holdout_days))
    if resolved_days < minimum_calibration_days:
        raise ValueError(
            "insufficient chronological history for factor calibration and holdout: "
            f"dates={len(dates)}, calibration={resolved_days}, holdout={holdout_days}"
        )
    cutoff = pd.Timestamp(dates[resolved_days - 1])
    normalized_dates = pd.to_datetime(frame[date_column], errors="coerce")
    calibration = frame.loc[normalized_dates <= cutoff].copy()
    return calibration, {
        "policy": "earliest fixed calibration window; later dates remain unseen",
        "dateColumn": date_column,
        "startDate": str(pd.Timestamp(dates[0]).date()),
        "cutoffDate": str(cutoff.date()),
        "calibrationDates": int(resolved_days),
        "holdoutDates": int(len(dates) - resolved_days),
        "calibrationRows": int(len(calibration)),
        "fullRows": int(len(frame)),
    }


def factor_columns_from_report(report: dict[str, object] | None) -> list[str]:
    if not isinstance(report, dict):
        return []
    columns: list[str] = []
    raw = report.get("added_columns")
    if isinstance(raw, list):
        columns.extend(str(item) for item in raw if item)
    members = report.get("members")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                columns.extend(factor_columns_from_report(member))
    return list(dict.fromkeys(columns))


def evaluate_factor_library(
    frame: pd.DataFrame,
    factor_columns: Iterable[str],
    return_column: str,
    output_dir: str | Path,
    *,
    config: FactorScreeningConfig | None = None,
) -> FactorScreeningResult:
    """Evaluate every materialised factor, then greedily remove correlations."""

    cfg = config or FactorScreeningConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        str(column)
        for column in dict.fromkeys(factor_columns)
        if str(column) in frame.columns
    ]
    if return_column not in frame.columns:
        raise ValueError(f"factor evaluation label is missing: {return_column}")
    if not columns:
        raise ValueError("factor evaluation found no materialised factor columns")
    if "amount" not in frame.columns:
        raise ValueError("factor evaluation requires amount for capacity evidence")

    summary = factor_summary_table(
        frame,
        columns,
        return_column,
        quantiles=cfg.quantiles,
        horizon_days=_horizon_from_label(return_column),
    )
    finite_ratios = {
        column: float(
            pd.to_numeric(frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .mean()
        )
        for column in columns
    }
    summary["finite_ratio"] = summary["factor_name"].map(finite_ratios).fillna(0.0)
    summary["abs_rank_ic"] = pd.to_numeric(summary["rank_ic"], errors="coerce").abs()
    summary["abs_rank_icir"] = pd.to_numeric(summary["rank_icir"], errors="coerce").abs()
    summary["abs_monotonicity"] = pd.to_numeric(summary["monotonicity"], errors="coerce").abs()
    summary["metric_gate"] = (
        (summary["finite_ratio"] >= cfg.min_finite_ratio)
        & (summary["abs_rank_ic"] >= cfg.min_abs_rank_ic)
        & (summary["abs_rank_icir"] >= cfg.min_abs_rank_icir)
        & (summary["abs_monotonicity"] >= cfg.min_abs_monotonicity)
    )

    correlation, evidence = _bounded_cross_sectional_spearman(
        frame,
        columns,
        max_dates=cfg.correlation_max_dates,
        max_symbols_per_date=cfg.correlation_max_symbols_per_date,
    )
    ordered = (
        summary.sort_values(
            ["metric_gate", "abs_rank_icir", "abs_rank_ic", "finite_ratio"],
            ascending=[False, False, False, False],
        )["factor_name"]
        .astype(str)
        .tolist()
    )
    selected: list[str] = []
    rejection_reason: dict[str, str] = {}
    metric_gate = summary.set_index("factor_name")["metric_gate"].to_dict()
    for factor in ordered:
        if not bool(metric_gate.get(factor)):
            rejection_reason[factor] = "metric_gate"
            continue
        duplicate = next(
            (
                existing
                for existing in selected
                if factor in correlation.index
                and existing in correlation.columns
                and np.isfinite(correlation.loc[factor, existing])
                and abs(float(correlation.loc[factor, existing])) > cfg.max_pairwise_correlation
            ),
            None,
        )
        if duplicate:
            rejection_reason[factor] = f"correlated_with:{duplicate}"
        else:
            selected.append(factor)

    summary["selected"] = summary["factor_name"].astype(str).isin(selected)
    summary["rejection_reason"] = summary["factor_name"].astype(str).map(rejection_reason).fillna("")
    summary = summary.sort_values(
        ["selected", "abs_rank_icir", "abs_rank_ic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    rejected = tuple(factor for factor in ordered if factor not in selected)
    result = FactorScreeningResult(
        summary=summary,
        correlation=correlation,
        selected_factors=tuple(selected),
        rejected_factors=rejected,
        output_dir=output,
    )
    summary.to_csv(output / "factor_summary.csv", index=False)
    correlation.to_csv(output / "factor_correlation.csv")
    selection_payload = {
        **result.to_dict(),
        "config": cfg.__dict__,
        "correlationEvidence": evidence,
    }
    (output / "factor_selection.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _bounded_cross_sectional_spearman(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_dates: int,
    max_symbols_per_date: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Bound rows deterministically while retaining every factor column."""

    data = frame[["trade_date", "symbol", *columns]].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    available_dates = data["trade_date"].dropna().drop_duplicates().sort_values()
    if len(available_dates) > max_dates:
        indexes = np.linspace(0, len(available_dates) - 1, max_dates, dtype=int)
        chosen_dates = set(available_dates.iloc[indexes])
        data = data[data["trade_date"].isin(chosen_dates)]
    data = (
        data.sort_values(["trade_date", "symbol"])
        .groupby("trade_date", sort=True, group_keys=False)
        .head(max_symbols_per_date)
    )
    ranked = data.groupby("trade_date", sort=False)[columns].rank(pct=True)
    matrix = ranked.corr(method="pearson")
    return matrix, {
        "method": "within-date percentile ranks followed by Pearson correlation",
        "equivalent": "bounded cross-sectional Spearman",
        "factorCount": len(columns),
        "sampledRows": int(len(data)),
        "sampledDates": int(data["trade_date"].nunique()),
        "maxDates": int(max_dates),
        "maxSymbolsPerDate": int(max_symbols_per_date),
    }


def _horizon_from_label(value: str) -> int:
    match = re.search(r"(\d+)d", value)
    return int(match.group(1)) if match else 1
