"""Build PIT-aware financial features that feed V7 alpha / risk layers.

This module takes the raw statement frames produced by the TuShare /
AkShare financial providers (or the local Parquet cache) and emits a
single wide ``financial_features_daily`` frame keyed by ``symbol`` and
``available_at``. Each output row is the most recent statement visible
at ``available_at`` — never a future report.

The features intentionally stay close to standard balance sheet /
income statement ratios. Composite scores (Beneish, Piotroski, Altman,
overall fraud score) are computed downstream by the existing fraud-risk
and financial-statement agents so the feature layer remains a pure data
transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


_GROWTH_FIELDS = ("revenue", "net_income", "operating_cash_flow", "gross_margin")


@dataclass(frozen=True)
class FinancialFeatureConfig:
    growth_lookback_periods: int = 1
    cap_extreme_quantile: float = 0.99
    min_required_columns: tuple[str, ...] = ("symbol", "report_period", "available_at")


def build_financial_features(
    income: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    financial_indicator: pd.DataFrame | None = None,
    valuation: pd.DataFrame | None = None,
    config: FinancialFeatureConfig | None = None,
) -> pd.DataFrame:
    """Merge statement frames and emit the ``financial_features_daily`` panel."""

    config = config or FinancialFeatureConfig()
    merged = _merge_statements(income, balance_sheet, cashflow, financial_indicator)
    if merged.empty:
        return pd.DataFrame()
    _require_columns(merged, config.min_required_columns)
    merged = merged.sort_values(["symbol", "report_period"]).reset_index(drop=True)
    merged = _compute_ratios(merged)
    merged = _compute_growth(merged, config.growth_lookback_periods)
    if valuation is not None and not valuation.empty:
        merged = _merge_valuation(merged, valuation)
    return _winsorize_numeric(merged, config.cap_extreme_quantile)


def apply_point_in_time_filter(
    frame: pd.DataFrame,
    trade_date: str,
    date_column: str = "available_at",
) -> pd.DataFrame:
    """Keep only the latest visible report for every symbol at ``trade_date``."""

    if frame is None or frame.empty or date_column not in frame.columns:
        return pd.DataFrame()
    parsed = pd.to_datetime(frame[date_column], errors="coerce")
    visible = frame.loc[parsed.notna() & (parsed <= pd.Timestamp(trade_date))].copy()
    if visible.empty:
        return visible
    visible = visible.sort_values(["symbol", "report_period", date_column])
    return visible.groupby("symbol", as_index=False, sort=False).tail(1).reset_index(drop=True)


def derive_v7_financial_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Project the merged feature frame into the V7 EvidenceRecord column names.

    Returns a frame keyed by ``symbol`` with the columns that
    :func:`score_financial_statements`, :func:`score_fraud_risk`,
    :func:`compute_long_horizon_factors` and the deep-alpha model expect.
    Missing fields are simply omitted so downstream consumers fall back to
    their own defaults — we never invent numbers.
    """

    if features is None or features.empty:
        return pd.DataFrame()
    keep_candidates = [
        "symbol",
        "report_period",
        "ann_date",
        "available_at",
        "revenue",
        "net_income",
        "operating_cash_flow",
        "cogs",
        "receivables",
        "inventory",
        "total_assets",
        "total_liabilities",
        "equity",
        "goodwill",
        "debt_to_asset",
        "current_ratio",
        "quick_ratio",
        "gross_margin",
        "net_margin",
        "roe",
        "roa",
        "asset_turnover",
        "inventory_turnover",
        "receivables_turnover",
        "rd_expense",
        "sell_expense",
        "admin_expense",
        "financial_expense",
        "fixed_assets",
        "construction_in_progress",
        "capex",
        "free_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "revenue_growth",
        "profit_growth",
        "ocf_growth",
        "gross_margin_change",
        "ocf_to_profit",
        "fcf_yield",
        "receivables_to_revenue",
        "inventory_to_revenue",
        "goodwill_ratio",
        "rd_intensity",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "ev_ebitda",
        "peg",
        "market_cap",
        "free_float_market_cap",
    ]
    keep = [column for column in keep_candidates if column in features.columns]
    return features[keep].copy()


_PIT_KEYS: tuple[str, ...] = ("symbol", "report_period", "ann_date", "available_at")
_PRESERVED_COLUMNS: frozenset[str] = frozenset(_PIT_KEYS)


def normalize_statement_frame(
    frame: pd.DataFrame | None,
    statement_type: str,
    *,
    prefix_collisions: bool = True,
) -> pd.DataFrame:
    """Return a deterministic copy of ``frame`` for ``statement_type``.

    * Required keys (``symbol``, ``report_period``, ``ann_date``,
      ``available_at``) are kept as-is.
    * All other columns are optionally prefixed with ``<statement>_`` so a
      later wide-merge cannot silently collapse columns that share a name
      across income / balance / cashflow / indicator statements.
    * Duplicate ``(symbol, report_period, available_at)`` rows are
      collapsed deterministically (last write wins after sorting).
    """

    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if prefix_collisions:
        rename: dict[str, str] = {}
        prefix = f"{statement_type}_"
        for column in data.columns:
            if column in _PRESERVED_COLUMNS:
                continue
            if column.startswith(prefix):
                continue
            rename[column] = f"{prefix}{column}"
        if rename:
            data = data.rename(columns=rename)
    sort_columns = [column for column in _PIT_KEYS if column in data.columns]
    if sort_columns:
        data = data.sort_values(sort_columns)
    dedup_keys = [column for column in ("symbol", "report_period", "available_at") if column in data.columns]
    if dedup_keys:
        data = data.drop_duplicates(subset=dedup_keys, keep="last")
    return data.reset_index(drop=True)


def pit_wide_merge_statements(
    statements: dict[str, pd.DataFrame],
    *,
    prefix_collisions: bool = True,
) -> pd.DataFrame:
    """Wide-merge per-statement PIT frames on the canonical PIT keys.

    ``statements`` maps statement type → raw frame. Each frame is run
    through :func:`normalize_statement_frame`, then outer-merged on
    ``(symbol, report_period, ann_date, available_at)``. Result is sorted
    and guaranteed unique per ``(symbol, report_period, available_at)``.
    Raises ``ValueError`` if any duplicate slips through (so a schema
    drift can never poison the downstream feature panel silently).
    """

    normalised = {
        name: normalize_statement_frame(frame, name, prefix_collisions=prefix_collisions)
        for name, frame in statements.items()
    }
    non_empty = [(name, frame) for name, frame in normalised.items() if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    _, merged = non_empty[0]
    merged = merged.copy()
    for _, frame in non_empty[1:]:
        on = [column for column in _PIT_KEYS if column in merged.columns and column in frame.columns]
        if not on:
            continue
        merged = merged.merge(frame, on=on, how="outer", suffixes=("", "_dup"))
        for column in list(merged.columns):
            if column.endswith("_dup"):
                base = column[: -len("_dup")]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[column])
                merged = merged.drop(columns=[column])
    sort_columns = [column for column in _PIT_KEYS if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns)
    dedup_keys = [column for column in ("symbol", "report_period", "available_at") if column in merged.columns]
    if dedup_keys:
        duplicate_count = int(merged.duplicated(subset=dedup_keys).sum())
        if duplicate_count:
            raise ValueError(
                f"PIT wide-merge produced {duplicate_count} duplicate (symbol, report_period, available_at) rows"
            )
    return merged.reset_index(drop=True)


def _merge_statements(
    income: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    indicator: pd.DataFrame | None,
) -> pd.DataFrame:
    # Back-compat path: keep behaviour stable for tests that pass already-merged
    # statement columns (revenue, total_assets, operating_cash_flow, ...).
    # New callers should prefer :func:`pit_wide_merge_statements` which
    # adds explicit statement-prefixing and duplicate detection.
    frames = [frame for frame in (income, balance_sheet, cashflow) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = _outer_merge(merged, frame)
    if indicator is not None and not indicator.empty:
        merged = _outer_merge(merged, indicator)
    return merged


def _outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    on = [column for column in _PIT_KEYS if column in left.columns and column in right.columns]
    if not on:
        return left
    merged = left.merge(right, on=on, how="outer", suffixes=("", "_dup"))
    for column in list(merged.columns):
        if column.endswith("_dup"):
            base = column[: -len("_dup")]
            merged[base] = merged[base].combine_first(merged[column])
            merged = merged.drop(columns=[column])
    return merged


def _compute_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    revenue = data.get("revenue")
    net_income = data.get("net_income")
    ocf = data.get("operating_cash_flow")
    if revenue is not None and net_income is not None and "net_margin" not in data.columns:
        data["net_margin"] = _safe_div(net_income, revenue)
    if revenue is not None and data.get("cogs") is not None and "gross_margin" not in data.columns:
        data["gross_margin"] = _safe_div(revenue - data["cogs"], revenue)
    if revenue is not None and data.get("receivables") is not None and "receivables_to_revenue" not in data.columns:
        data["receivables_to_revenue"] = _safe_div(data["receivables"], revenue)
    if revenue is not None and data.get("inventory") is not None and "inventory_to_revenue" not in data.columns:
        data["inventory_to_revenue"] = _safe_div(data["inventory"], revenue)
    if data.get("total_assets") is not None and data.get("goodwill") is not None and "goodwill_ratio" not in data.columns:
        data["goodwill_ratio"] = _safe_div(data["goodwill"], data["total_assets"])
    if data.get("total_assets") is not None and data.get("total_liabilities") is not None and "debt_to_asset" not in data.columns:
        data["debt_to_asset"] = _safe_div(data["total_liabilities"], data["total_assets"])
    if ocf is not None and net_income is not None and "ocf_to_profit" not in data.columns:
        data["ocf_to_profit"] = _safe_div(ocf, net_income)
    if revenue is not None and data.get("rd_expense") is not None and "rd_intensity" not in data.columns:
        data["rd_intensity"] = _safe_div(data["rd_expense"], revenue)
    if "fcf_yield" not in data.columns and data.get("free_cash_flow") is not None and data.get("market_cap") is not None:
        data["fcf_yield"] = _safe_div(data["free_cash_flow"], data["market_cap"])
    return data


def _compute_growth(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    data = frame.copy()
    for field in _GROWTH_FIELDS:
        if field in data.columns:
            prior = data.groupby("symbol")[field].shift(lookback)
            growth_column = {
                "revenue": "revenue_growth",
                "net_income": "profit_growth",
                "operating_cash_flow": "ocf_growth",
                "gross_margin": "gross_margin_change",
            }[field]
            if growth_column not in data.columns:
                data[growth_column] = _safe_div(data[field] - prior, prior.abs())
    return data


def _merge_valuation(frame: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    on = [column for column in ("symbol", "available_at") if column in valuation.columns and column in frame.columns]
    if not on:
        on = [column for column in ("symbol",) if column in valuation.columns and column in frame.columns]
    if not on:
        return frame
    return frame.merge(valuation, on=on, how="left", suffixes=("", "_val"))


def _winsorize_numeric(frame: pd.DataFrame, cap_quantile: float) -> pd.DataFrame:
    if cap_quantile <= 0.0 or cap_quantile >= 1.0:
        return frame
    data = frame.copy()
    for column in data.select_dtypes("number").columns:
        if column in {"report_period", "ann_date", "available_at"}:
            continue
        series = data[column].replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() < 4:
            data[column] = series
            continue
        upper = float(series.quantile(cap_quantile))
        lower = float(series.quantile(1.0 - cap_quantile))
        data[column] = series.clip(lower=lower, upper=upper)
    return data


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Financial feature input missing required columns: {missing}")


def _safe_div(numerator, denominator) -> pd.Series:
    num = pd.Series(numerator).astype(float)
    den = pd.Series(denominator).astype(float)
    out = num / den.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)
