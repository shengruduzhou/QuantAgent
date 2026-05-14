"""Long-horizon factor library targeting 3-6 month holding windows.

Existing factor sets (alpha101, cicc_high_freq) are price/volume-derived and decay
within ~20 trading days. For positions held 60-126 trading days the alpha must
come from slow-moving fundamentals (capex, ROE persistence, margin trend,
order backlog), structural exposure (theme, policy support, domestic substitution),
and valuation mean-reversion. This module computes those factors and exposes them
through a single pipeline-friendly entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongHorizonFactorConfig:
    revenue_growth_window_quarters: int = 4
    profit_growth_window_quarters: int = 4
    margin_trend_window_quarters: int = 4
    roe_persistence_window_quarters: int = 8
    valuation_history_window_years: int = 3
    valuation_industry_min_peers: int = 5
    momentum_long_window_days: int = 120
    momentum_medium_window_days: int = 60
    capex_efficiency_window_quarters: int = 4
    fraud_haircut_threshold: float = 60.0
    fraud_haircut_strength: float = 0.50
    policy_decay_half_life_days: float = 90.0
    half_life_floor_days: float = 14.0


LONG_HORIZON_FACTORS: tuple[str, ...] = (
    "quality_roe_persistence_120d",
    "quality_roic_trend_120d",
    "quality_gross_margin_trend_120d",
    "quality_fcf_yield_120d",
    "growth_revenue_yoy_120d",
    "growth_profit_yoy_120d",
    "growth_order_visibility_120d",
    "growth_capacity_release_120d",
    "valuation_history_zscore_120d",
    "valuation_industry_zscore_120d",
    "valuation_peg_120d",
    "valuation_margin_of_safety_120d",
    "valuation_bubble_risk_inverse_120d",
    "policy_support_decay_120d",
    "policy_chain_centrality_120d",
    "macro_industry_phase_120d",
    "macro_credit_cycle_120d",
    "macro_monetary_tailwind_120d",
    "structural_domestic_substitution_120d",
    "structural_bottleneck_120d",
    "flow_sector_rotation_60d",
    "flow_attention_persistence_60d",
    "risk_fraud_haircut_120d",
    "risk_management_quality_120d",
)


def compute_long_horizon_factors(
    fundamentals: pd.DataFrame,
    market_state: pd.DataFrame,
    price_panel: pd.DataFrame,
    chain_features: pd.DataFrame | None = None,
    economics_features: pd.DataFrame | None = None,
    config: LongHorizonFactorConfig | None = None,
) -> pd.DataFrame:
    """Return a per-symbol frame keyed by `symbol` with `LONG_HORIZON_FACTORS` columns."""
    config = config or LongHorizonFactorConfig()
    if fundamentals is None or fundamentals.empty:
        fundamentals = pd.DataFrame()
    if market_state is None or market_state.empty:
        market_state = pd.DataFrame()
    if price_panel is None or price_panel.empty:
        price_panel = pd.DataFrame()
    chain_features = chain_features if chain_features is not None else pd.DataFrame()
    economics_features = economics_features if economics_features is not None else pd.DataFrame()

    symbols = _collect_symbols(fundamentals, market_state, price_panel, chain_features, economics_features)
    if not symbols:
        return pd.DataFrame(columns=("symbol",) + LONG_HORIZON_FACTORS)

    fund_latest = _latest_per_symbol(fundamentals, date_column="report_date")
    fund_history = _sort_history(fundamentals, date_column="report_date")
    market_latest = _latest_per_symbol(market_state, date_column="trade_date")
    price_history = _sort_history(price_panel, date_column="trade_date")
    chain_latest = _latest_per_symbol(chain_features, date_column="as_of_date")
    economics_latest = _latest_per_symbol(economics_features, date_column="as_of_date")

    rows = []
    for symbol in symbols:
        row = {"symbol": symbol}
        fund_rows = fund_history[fund_history["symbol"] == symbol] if not fund_history.empty else pd.DataFrame()
        price_rows = price_history[price_history["symbol"] == symbol] if not price_history.empty else pd.DataFrame()
        fund_row = _row_or_empty(fund_latest, symbol)
        market_row = _row_or_empty(market_latest, symbol)
        chain_row = _row_or_empty(chain_latest, symbol)
        econ_row = _row_or_empty(economics_latest, symbol)

        row["quality_roe_persistence_120d"] = _roe_persistence(fund_rows, config)
        row["quality_roic_trend_120d"] = _roic_trend(fund_rows)
        row["quality_gross_margin_trend_120d"] = _gross_margin_trend(fund_rows, config)
        row["quality_fcf_yield_120d"] = _fcf_yield(fund_row, market_row)

        row["growth_revenue_yoy_120d"] = _yoy_growth(fund_rows, "revenue", config.revenue_growth_window_quarters)
        row["growth_profit_yoy_120d"] = _yoy_growth(fund_rows, "net_income", config.profit_growth_window_quarters)
        row["growth_order_visibility_120d"] = _percent_to_unit(fund_row.get("order_visibility_score"))
        row["growth_capacity_release_120d"] = _percent_to_unit(fund_row.get("capacity_release_score"))

        row["valuation_history_zscore_120d"] = _history_valuation_zscore(fund_row, fund_rows)
        row["valuation_industry_zscore_120d"] = _industry_valuation_zscore(fund_row, fund_latest)
        row["valuation_peg_120d"] = _peg_score(fund_row)
        row["valuation_margin_of_safety_120d"] = _safe_float(fund_row.get("margin_of_safety"))
        row["valuation_bubble_risk_inverse_120d"] = 1.0 - _percent_to_unit(fund_row.get("valuation_bubble_score"))

        row["policy_support_decay_120d"] = _percent_to_unit(chain_row.get("policy_support_decay") or fund_row.get("policy_strength"))
        row["policy_chain_centrality_120d"] = _percent_to_unit(chain_row.get("chain_centrality"))
        row["macro_industry_phase_120d"] = _industry_phase_to_unit(econ_row.get("industry_cycle_stage"))
        row["macro_credit_cycle_120d"] = _signed_unit(econ_row.get("credit_impulse_alignment"))
        row["macro_monetary_tailwind_120d"] = _signed_unit(econ_row.get("monetary_tailwind"))

        row["structural_domestic_substitution_120d"] = _percent_to_unit(chain_row.get("domestic_substitution_score"))
        row["structural_bottleneck_120d"] = _percent_to_unit(chain_row.get("bottleneck_score"))

        row["flow_sector_rotation_60d"] = _percent_to_unit(market_row.get("sector_rotation_score"))
        row["flow_attention_persistence_60d"] = _attention_persistence(market_row, price_rows)

        fraud_raw = _safe_float(fund_row.get("fraud_risk_score", 50.0))
        haircut = max(0.0, fraud_raw - config.fraud_haircut_threshold) / max(1.0, 100.0 - config.fraud_haircut_threshold)
        row["risk_fraud_haircut_120d"] = 1.0 - config.fraud_haircut_strength * haircut
        row["risk_management_quality_120d"] = 1.0 - _percent_to_unit(fund_row.get("management_risk_score"))
        rows.append(row)
    return pd.DataFrame(rows)


def long_horizon_alpha_score(
    factors_frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Linear blend of the long-horizon factor frame into a single score per symbol."""
    if factors_frame is None or factors_frame.empty:
        return pd.DataFrame(columns=("symbol", "long_horizon_alpha_120d"))
    weights = weights or _default_long_horizon_weights()
    rows = []
    for _, row in factors_frame.iterrows():
        total_weight = 0.0
        weighted_value = 0.0
        for factor, weight in weights.items():
            value = row.get(factor)
            if value is None or pd.isna(value):
                continue
            weighted_value += float(value) * float(weight)
            total_weight += abs(float(weight))
        score = weighted_value / total_weight if total_weight > 0 else 0.0
        fraud_haircut = float(row.get("risk_fraud_haircut_120d", 1.0))
        rows.append(
            {
                "symbol": row["symbol"],
                "long_horizon_alpha_120d": float(np.clip(score, -1.0, 1.0)) * fraud_haircut,
                "long_horizon_confidence": float(np.clip(0.40 + 0.60 * fraud_haircut, 0.10, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _default_long_horizon_weights() -> dict[str, float]:
    return {
        "quality_roe_persistence_120d": 0.08,
        "quality_roic_trend_120d": 0.06,
        "quality_gross_margin_trend_120d": 0.06,
        "quality_fcf_yield_120d": 0.06,
        "growth_revenue_yoy_120d": 0.08,
        "growth_profit_yoy_120d": 0.08,
        "growth_order_visibility_120d": 0.06,
        "growth_capacity_release_120d": 0.04,
        "valuation_history_zscore_120d": 0.06,
        "valuation_industry_zscore_120d": 0.06,
        "valuation_peg_120d": 0.04,
        "valuation_margin_of_safety_120d": 0.06,
        "valuation_bubble_risk_inverse_120d": 0.04,
        "policy_support_decay_120d": 0.06,
        "policy_chain_centrality_120d": 0.04,
        "macro_industry_phase_120d": 0.03,
        "macro_credit_cycle_120d": 0.02,
        "macro_monetary_tailwind_120d": 0.02,
        "structural_domestic_substitution_120d": 0.03,
        "structural_bottleneck_120d": 0.03,
        "flow_sector_rotation_60d": 0.02,
        "flow_attention_persistence_60d": 0.02,
    }


# ---------------------------------------------------------------------------
# Factor builders
# ---------------------------------------------------------------------------


def _roe_persistence(fund_rows: pd.DataFrame, config: LongHorizonFactorConfig) -> float:
    if fund_rows is None or fund_rows.empty or "roe" not in fund_rows.columns:
        return 0.0
    series = pd.to_numeric(fund_rows["roe"], errors="coerce").dropna().tail(config.roe_persistence_window_quarters)
    if series.empty:
        return 0.0
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    persistence = mean / (abs(mean) + std + 1e-6)
    return float(np.clip(persistence, -1.0, 1.0))


def _roic_trend(fund_rows: pd.DataFrame) -> float:
    if fund_rows is None or fund_rows.empty:
        return 0.0
    if "roic" in fund_rows.columns:
        series = pd.to_numeric(fund_rows["roic"], errors="coerce").dropna()
    else:
        if not {"net_income", "total_assets", "debt_to_asset"}.issubset(fund_rows.columns):
            return 0.0
        invested = fund_rows["total_assets"] * (1.0 - fund_rows["debt_to_asset"].clip(lower=0.0, upper=0.99))
        invested = invested.replace(0.0, np.nan)
        series = (fund_rows["net_income"] / invested).dropna()
    if len(series) < 2:
        return 0.0
    diff = float(series.iloc[-1] - series.iloc[0])
    base = max(abs(float(series.iloc[0])), 0.01)
    return float(np.clip(diff / base, -1.0, 1.0))


def _gross_margin_trend(fund_rows: pd.DataFrame, config: LongHorizonFactorConfig) -> float:
    if fund_rows is None or fund_rows.empty or "gross_margin" not in fund_rows.columns:
        return 0.0
    series = pd.to_numeric(fund_rows["gross_margin"], errors="coerce").dropna().tail(config.margin_trend_window_quarters)
    if len(series) < 2:
        return 0.0
    delta = float(series.iloc[-1] - series.iloc[0])
    return float(np.clip(delta / max(abs(float(series.iloc[0])), 0.05), -1.0, 1.0))


def _fcf_yield(fund_row: dict[str, object], market_row: dict[str, object]) -> float:
    fcf = _safe_float(fund_row.get("free_cash_flow", fund_row.get("operating_cash_flow", 0.0))) + _safe_float(fund_row.get("capex", 0.0))
    market_cap = _safe_float(fund_row.get("market_cap")) or _safe_float(market_row.get("market_cap"))
    if market_cap <= 0:
        return 0.0
    return float(np.clip(fcf / market_cap, -0.20, 0.30) / 0.30)


def _yoy_growth(fund_rows: pd.DataFrame, column: str, window: int) -> float:
    if fund_rows is None or fund_rows.empty or column not in fund_rows.columns:
        return 0.0
    series = pd.to_numeric(fund_rows[column], errors="coerce").dropna()
    if len(series) <= window:
        if len(series) < 2:
            return 0.0
        latest = float(series.iloc[-1])
        prior = float(series.iloc[0])
    else:
        latest = float(series.iloc[-1])
        prior = float(series.iloc[-1 - window])
    if prior == 0:
        return 0.0
    growth = (latest - prior) / abs(prior)
    return float(np.clip(growth, -1.0, 1.0))


def _percent_to_unit(value: object) -> float:
    numeric = _safe_float(value)
    if numeric > 1.0:
        numeric /= 100.0
    return float(np.clip(numeric, 0.0, 1.0))


def _signed_unit(value: object) -> float:
    return float(np.clip(_safe_float(value), -1.0, 1.0))


def _industry_phase_to_unit(value: object) -> float:
    if value is None:
        return 0.5
    mapping = {
        "downturn": 0.10,
        "trough": 0.25,
        "recovery": 0.65,
        "early_cycle": 0.80,
        "mid_cycle": 0.55,
        "late_cycle": 0.35,
    }
    return mapping.get(str(value), 0.50)


def _history_valuation_zscore(fund_row: dict[str, object], fund_rows: pd.DataFrame) -> float:
    if fund_rows is None or fund_rows.empty or "pe_ttm" not in fund_rows.columns:
        return 0.0
    series = pd.to_numeric(fund_rows["pe_ttm"], errors="coerce").dropna()
    if len(series) < 4:
        return 0.0
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return 0.0
    current = _safe_float(fund_row.get("pe_ttm")) or float(series.iloc[-1])
    z = (current - mean) / std
    return float(np.clip(-z / 2.0, -1.0, 1.0))


def _industry_valuation_zscore(fund_row: dict[str, object], fund_latest: pd.DataFrame) -> float:
    if fund_latest is None or fund_latest.empty:
        return 0.0
    industry = str(fund_row.get("industry") or fund_row.get("sector") or "")
    if not industry or "industry" not in fund_latest.columns:
        return 0.0
    peers = fund_latest[fund_latest["industry"].astype(str) == industry]
    if len(peers) < 3 or "pe_ttm" not in peers.columns:
        return 0.0
    series = pd.to_numeric(peers["pe_ttm"], errors="coerce").dropna()
    if len(series) < 3:
        return 0.0
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std <= 0:
        return 0.0
    current = _safe_float(fund_row.get("pe_ttm"))
    if current <= 0:
        return 0.0
    z = (current - mean) / std
    return float(np.clip(-z / 2.0, -1.0, 1.0))


def _peg_score(fund_row: dict[str, object]) -> float:
    peg = _safe_float(fund_row.get("peg"))
    if peg <= 0:
        return 0.0
    return float(np.clip(1.0 - peg, -1.0, 1.0))


def _attention_persistence(market_row: dict[str, object], price_rows: pd.DataFrame) -> float:
    if price_rows is None or price_rows.empty or "amount" not in price_rows.columns:
        return 0.0
    amount = pd.to_numeric(price_rows["amount"], errors="coerce").dropna()
    if len(amount) < 20:
        return 0.0
    recent = amount.tail(20).mean()
    longer = amount.tail(60).mean() if len(amount) >= 60 else amount.mean()
    if longer <= 0:
        return 0.0
    ratio = float(recent / longer)
    return float(np.clip((ratio - 1.0) * 2.0, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _collect_symbols(*frames: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "symbol" in frame.columns:
            symbols.update(str(value) for value in frame["symbol"].dropna().unique())
    return sorted(symbols)


def _latest_per_symbol(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values(["symbol", date_column]).groupby("symbol", sort=False).tail(1)
    else:
        data = data.drop_duplicates("symbol", keep="last")
    return data.set_index("symbol", drop=False)


def _sort_history(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values(["symbol", date_column])
    return data


def _row_or_empty(frame: pd.DataFrame, symbol: str) -> dict[str, object]:
    if frame is None or frame.empty or symbol not in frame.index:
        return {}
    row = frame.loc[symbol]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(numeric) or np.isinf(numeric):
        return 0.0
    return numeric
