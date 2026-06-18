"""Factor-combination training for TickFlow intraday Do-T research."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.execution.broker_base import OrderSide
from quantagent.execution.intraday_fill import CostConfig
from quantagent.execution.selective_dot import (
    SelectiveDotParams,
    build_day_contexts,
    prepare_day_arrays,
    simulate_prepared,
)
from quantagent.factors.intraday_volume_price import compute_intraday_factors


DEFAULT_FSM_GRID = {
    "mode": ["dip_buy", "spike_sell"],
    "dip_atr_mult": [0.10, 0.15, 0.20, 0.30, 0.45, 0.60],
    "target_atr_mult": [0.15, 0.25, 0.40, 0.60],
    "stop_atr_mult": [0.25, 0.40, 0.60, 0.80],
    "morning_deadline": ["10:00:00", "10:30:00"],
    "tail_exit_time": ["13:55:00", "14:10:00", "14:25:00", "14:40:00"],
}

DEFAULT_TIME_GRID = {
    "mode": ["time_buy", "time_sell"],
    "dip_atr_mult": [0.0],
    "target_atr_mult": [0.15, 0.25, 0.40],
    "stop_atr_mult": [0.25, 0.40, 0.60],
    "morning_deadline": ["10:00:00", "10:30:00", "13:30:00"],
    "tail_exit_time": ["14:10:00", "14:25:00", "14:40:00"],
}

DEFAULT_FACTOR_COLUMNS = [
    "first30_return",
    "last30_return",
    "vwap_deviation",
    "intraday_range_pos",
    "net_buy_pressure",
    "volume_concentration",
    "spike_minutes",
    "am_pm_volume_ratio",
    "minute_ret_skew",
    "liq_amihud_1min",
    "liq_amihud_1min_m20",
    "corr_prv",
    "corr_prv_m20",
    "open30_volume_share",
    "close30_volume_share",
    "close3_volume_share",
]

ENTRY_CAUSAL_FEATURE_COLUMNS = [
    "entry_minute_idx",
    "entry_price_vs_vwap_prev",
    "entry_return_from_open",
    "entry_range_pos_sofar",
    "entry_distance_to_hod_sofar",
    "entry_distance_to_lod_sofar",
    "entry_rolling_return_3m",
    "entry_rolling_return_5m",
    "entry_rolling_return_10m",
    "entry_rolling_volatility_5m",
    "entry_rolling_volatility_10m",
    "entry_volume_zscore_5m",
    "entry_volume_zscore_20m",
    "entry_volume_share_sofar",
    "entry_cum_volume",
    "entry_amount_zscore_5m",
    "entry_amount_zscore_20m",
    "entry_order_flow_imbalance_sofar",
    "entry_order_flow_imbalance_5m",
    "entry_order_flow_imbalance_20m",
    "entry_buy_pressure_5m",
    "entry_sell_pressure_5m",
    "entry_price_impact_bps_5m",
    "entry_price_impact_bps_20m",
    "entry_vwap_slope_5m",
    "entry_close_location_5m",
    "entry_large_volume_share_sofar",
    "entry_uptrend_persistence_10m",
    "entry_downtrend_persistence_10m",
    "entry_failed_breakout_proxy",
    "entry_failed_breakdown_proxy",
    "entry_near_hod_volume_pressure",
    "entry_near_lod_volume_pressure",
    "entry_momentum_persistence",
    "entry_mean_reversion_quality",
    "entry_mode_adverse_risk",
]

ENTRY_RELATIVE_FEATURE_COLUMNS = [
    "entry_market_return_sofar",
    "entry_industry_return_sofar",
    "entry_stock_return_minus_market",
    "entry_stock_return_minus_industry",
    "entry_stock_vwap_dev_minus_market",
    "entry_stock_vwap_dev_minus_industry",
    "entry_relative_volume_vs_market",
    "entry_relative_volume_vs_industry",
    "entry_market_breadth_sofar",
    "entry_industry_breadth_sofar",
    "entry_up_down_ratio_sofar",
    "entry_industry_market_return_spread",
]


@dataclass(frozen=True)
class TickFlowValidationSummary:
    minute_dir: str
    symbols_checked: int
    symbols_ok: int
    total_rows: int
    total_days: int
    start: str
    end: str
    median_rows_per_symbol: float
    bad_day_rate: float
    duplicate_bar_count: int
    invalid_price_rows: int
    missing_required_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "minute_dir": self.minute_dir,
            "symbols_checked": self.symbols_checked,
            "symbols_ok": self.symbols_ok,
            "total_rows": self.total_rows,
            "total_days": self.total_days,
            "start": self.start,
            "end": self.end,
            "median_rows_per_symbol": self.median_rows_per_symbol,
            "bad_day_rate": self.bad_day_rate,
            "duplicate_bar_count": self.duplicate_bar_count,
            "invalid_price_rows": self.invalid_price_rows,
            "missing_required_columns": self.missing_required_columns,
        }


@dataclass(frozen=True)
class FactorComboConfig:
    start: str = "2025-09-01"
    end: str = "2026-06-11"
    split: str = "2026-02-27"
    validation_split: str = "2026-04-15"
    dot_fraction: float = 0.30
    commission_bps: float = 2.5
    stamp_bps: float = 5.0
    slippage_bps: float = 8.0
    spread_bps: float = 6.0
    transfer_bps: float = 0.1
    order_notional_yuan: float = 30_000.0
    max_minute_participation: float = 0.05
    min_fill_ratio: float = 1.0
    top_fracs: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
    eod_restore_prob_caps: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0)
    min_train_legs: int = 100
    min_validation_legs: int = 100
    min_oos_legs: int = 300
    min_pred_net_bps: float = 0.0
    eod_restore_penalty_bps: float = 15.0
    max_validation_eod_restore_rate: float = 0.35
    max_validation_stop_rate: float = 0.35
    require_book_for_enable: bool = True
    selection_book_only: bool = True
    min_validation_book_legs: int = 30
    min_oos_book_legs: int = 100
    policy_eod_penalty_bps: float = 80.0
    policy_stop_penalty_bps: float = 60.0
    selection_eod_prob_penalty_bps: float = 0.0
    selection_stop_prob_penalty_bps: float = 0.0
    selection_entry_adverse_penalty_bps: float = 0.0
    tail_exit_deadline: str = "14:50:00"
    outcome_workers: int = 1
    outcome_cache_dir: str = ""
    force_rebuild_outcome_cache: bool = False
    relative_strength_cache_dir: str = ""
    force_rebuild_feature_cache: bool = False
    sector_map_path: str = "runtime/data/v7/silver/sector_map/sector_map.parquet"
    entry_adverse_risk_caps: tuple[float, ...] = (0.60, 0.75, 0.90, 1.0)
    min_reversion_qualities: tuple[float, ...] = (-1.0, 0.0, 0.10)
    stop_prob_caps: tuple[float, ...] = (0.60, 1.0)
    random_seed: int = 42

    @property
    def round_trip_cost(self) -> float:
        return (2.0 * self.commission_bps + self.stamp_bps + 2.0 * self.slippage_bps) / 10_000.0

    @property
    def fill_cost_config(self) -> CostConfig:
        return CostConfig(
            commission_rate=self.commission_bps / 10_000.0,
            min_commission=0.0,
            stamp_tax_sell=self.stamp_bps / 10_000.0,
            transfer_fee=self.transfer_bps / 10_000.0,
            slippage_bps=self.slippage_bps,
            spread_bps=self.spread_bps,
        )


def read_parquet_checked(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet with an actionable error for broken pyarrow environments."""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "IpcReadOptions size changed" in msg or "pyarrow" in msg.lower():
            raise RuntimeError(
                "Parquet read failed because the active Python environment has a broken pyarrow/pandas ABI. "
                "Use /home/shanhefu/QuantAgent/AI_quant_venv/bin/python for TickFlow minute training."
            ) from exc
        raise


def validate_tickflow_minute_cache(
    minute_dir: str | Path,
    *,
    symbols: Iterable[str] | None = None,
    max_symbols: int = 0,
) -> tuple[TickFlowValidationSummary, pd.DataFrame]:
    minute_path = Path(minute_dir)
    files = sorted(minute_path.glob("*.parquet"))
    if symbols is not None:
        wanted = {str(s) for s in symbols}
        files = [p for p in files if p.stem in wanted]
    if max_symbols:
        files = files[: int(max_symbols)]
    required = {"symbol", "trade_time", "open", "high", "low", "close", "volume", "amount"}
    rows = []
    missing_all: set[str] = set()
    for p in files:
        df = read_parquet_checked(p)
        missing = sorted(required.difference(df.columns))
        if missing:
            missing_all.update(missing)
            rows.append({"symbol": p.stem, "ok": False, "rows": 0, "days": 0, "bad_day_rate": 1.0})
            continue
        tt = pd.to_datetime(df["trade_time"], errors="coerce")
        d = tt.dt.normalize()
        day_counts = d.value_counts()
        bad_days = int(((day_counts < 180) | (day_counts > 260)).sum())
        prices = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        invalid_price = int(((prices <= 0).any(axis=1) | (prices["high"] < prices["low"])).sum())
        invalid_price_rate = float(invalid_price / max(1, len(df)))
        dupes = int(df.duplicated(["trade_time"]).sum())
        bad_day_rate = float(bad_days / max(1, len(day_counts)))
        rows.append({
            "symbol": p.stem,
            "ok": missing == [] and invalid_price_rate <= 0.01 and dupes == 0 and bad_day_rate <= 0.05,
            "rows": int(len(df)),
            "days": int(d.nunique()),
            "start": str(tt.min()),
            "end": str(tt.max()),
            "bad_days": bad_days,
            "bad_day_rate": bad_day_rate,
            "duplicate_bar_count": dupes,
            "invalid_price_rows": invalid_price,
            "invalid_price_rate": invalid_price_rate,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = TickFlowValidationSummary(str(minute_path), 0, 0, 0, 0, "", "", 0.0, 1.0, 0, 0, sorted(missing_all))
        return summary, detail
    summary = TickFlowValidationSummary(
        minute_dir=str(minute_path),
        symbols_checked=int(len(detail)),
        symbols_ok=int(detail["ok"].sum()),
        total_rows=int(detail["rows"].sum()),
        total_days=int(detail["days"].sum()),
        start=str(pd.to_datetime(detail["start"], errors="coerce").min()),
        end=str(pd.to_datetime(detail["end"], errors="coerce").max()),
        median_rows_per_symbol=float(detail["rows"].median()),
        bad_day_rate=float(detail["bad_days"].sum() / max(1, detail["days"].sum())),
        duplicate_bar_count=int(detail["duplicate_bar_count"].sum()),
        invalid_price_rows=int(detail["invalid_price_rows"].sum()),
        missing_required_columns=sorted(missing_all),
    )
    return summary, detail


def build_factor_combo_dataset(
    *,
    minute_dir: str | Path,
    market_panel_path: str | Path,
    intraday_factors_path: str | Path,
    holdings_csv: str | Path | None,
    config: FactorComboConfig,
    max_symbols: int = 0,
    reuse_outcomes_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    symbols = sorted(p.stem for p in Path(minute_dir).glob("*.parquet"))
    if max_symbols:
        symbols = symbols[: int(max_symbols)]
    holdings = _load_holdings_frame(holdings_csv)
    panel = read_parquet_checked(
        market_panel_path,
        columns=["symbol", "trade_date", "open", "high", "low", "close"],
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[
        (panel["trade_date"] >= start - pd.Timedelta(days=120))
        & (panel["trade_date"] <= end)
        & panel["symbol"].isin(symbols)
    ]
    contexts = build_day_contexts(panel)
    contexts = contexts[(contexts["trade_date"] >= start) & (contexts["trade_date"] <= end)]
    if bool(config.selection_book_only) and not holdings.empty:
        book_keys = holdings[
            (holdings["trade_date"] >= start)
            & (holdings["trade_date"] <= end)
            & holdings["symbol"].isin(symbols)
            & (holdings["weight"] > 0)
        ][["symbol", "trade_date"]].drop_duplicates()
        if not book_keys.empty:
            contexts = contexts.merge(book_keys, on=["symbol", "trade_date"], how="inner")
            symbols = sorted(contexts["symbol"].dropna().astype(str).unique().tolist())
        else:
            contexts = contexts.iloc[0:0].copy()
            symbols = []

    outcomes_path = Path(reuse_outcomes_path) if reuse_outcomes_path else None
    if outcomes_path is not None and outcomes_path.exists():
        outcomes = read_parquet_checked(outcomes_path)
        required_outcome_cols = {
            "eod_restore",
            "entry_fill_status",
            "entry_price_vs_vwap_prev",
            "fee_cost_bps",
            "tail_exit_time",
            "time_exit",
            "entry_order_flow_imbalance_5m",
            "entry_mode_adverse_risk",
        }
        if not required_outcome_cols.issubset(outcomes.columns):
            outcomes = pd.DataFrame()
    else:
        outcomes = pd.DataFrame()
    if outcomes.empty:
        outcomes = build_dot_outcomes(
            minute_dir=minute_dir,
            symbols=symbols,
            contexts=contexts,
            start=start,
            end=end,
            config=config,
        )
        if outcomes_path is not None:
            outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            outcomes.to_parquet(outcomes_path, index=False)

    relative = build_entry_relative_strength_features(
        outcomes,
        minute_dir=minute_dir,
        sector_map_path=config.sector_map_path,
        cache_dir=config.relative_strength_cache_dir,
        force_rebuild=config.force_rebuild_feature_cache,
    )
    if not relative.empty:
        outcomes = outcomes.merge(
            relative,
            on=["symbol", "trade_date", "entry_idx"],
            how="left",
        )

    factors = load_or_build_intraday_factors(
        intraday_factors_path=intraday_factors_path,
        minute_dir=minute_dir,
        symbols=symbols,
        start=start - pd.Timedelta(days=10),
        end=end,
    )
    factor_cols = [c for c in DEFAULT_FACTOR_COLUMNS if c in factors.columns]
    factors = factors.sort_values(["symbol", "trade_date"])
    lagged = factors[["symbol", "trade_date", *factor_cols]].copy()
    lagged[factor_cols] = lagged.groupby("symbol", sort=False)[factor_cols].shift(1)
    lagged = lagged.rename(columns={c: f"prev_{c}" for c in factor_cols})

    data = outcomes.merge(contexts, on=["symbol", "trade_date"], how="left")
    data = data.merge(lagged, on=["symbol", "trade_date"], how="left")
    if not holdings.empty:
        data = data.merge(holdings[["trade_date", "symbol", "weight"]], on=["trade_date", "symbol"], how="left")
    if "weight" not in data.columns:
        data["weight"] = np.nan
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    data["book_only_context"] = bool(config.selection_book_only and not holdings.empty)
    data["target_net_ret_bps"] = data["net_ret"] * 10_000.0
    data["mode_dip_buy"] = data["mode"].map(_is_buy_mode).astype(float)
    data["mode_spike_sell"] = (data["mode"] == "spike_sell").astype(float)
    data["mode_time_entry"] = data["mode"].astype(str).str.startswith("time_").astype(float)
    data["regime_bull"] = (data["regime"] == "bull").astype(float)
    data["regime_sideways"] = (data["regime"] == "sideways").astype(float)
    return data, contexts


def _load_holdings_frame(holdings_csv: str | Path | None) -> pd.DataFrame:
    if not holdings_csv or not Path(holdings_csv).exists():
        return pd.DataFrame(columns=["trade_date", "symbol", "weight"])
    h = pd.read_csv(holdings_csv)
    required = {"trade_date", "symbol", "weight"}
    if not required.issubset(h.columns):
        return pd.DataFrame(columns=["trade_date", "symbol", "weight"])
    h = h[["trade_date", "symbol", "weight"]].copy()
    h["trade_date"] = pd.to_datetime(h["trade_date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str)
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    return h.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)


def load_or_build_intraday_factors(
    *,
    intraday_factors_path: str | Path,
    minute_dir: str | Path,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load day-level intraday factors; rebuild from TickFlow cache if stale."""
    factors = pd.DataFrame()
    path = Path(intraday_factors_path)
    if path.exists():
        factors = read_parquet_checked(path)
        if not factors.empty:
            factors["trade_date"] = pd.to_datetime(factors["trade_date"], errors="coerce").dt.normalize()
            factors["symbol"] = factors["symbol"].astype(str)
            overlap = factors[
                (factors["trade_date"] >= start)
                & (factors["trade_date"] <= end)
                & factors["symbol"].isin(symbols)
            ]
            if not overlap.empty:
                return factors
    return build_intraday_factors_from_minute_cache(
        minute_dir=minute_dir,
        symbols=symbols,
        start=start,
        end=end,
    )


def build_intraday_factors_from_minute_cache(
    *,
    minute_dir: str | Path,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    parts = []
    for sym in symbols:
        p = Path(minute_dir) / f"{sym}.parquet"
        if not p.exists():
            continue
        bars = read_parquet_checked(p, columns=["symbol", "trade_date", "trade_time", "open", "high", "low", "close", "volume", "amount"])
        bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
        bars = bars[(bars["trade_time"] >= start) & (bars["trade_time"] <= end + pd.Timedelta(days=1))]
        if bars.empty:
            continue
        minute = bars.rename(columns={"trade_time": "datetime"}).copy()
        minute["trade_date"] = pd.to_datetime(minute["trade_date"], errors="coerce").dt.normalize()
        parts.append(compute_intraday_factors(minute))
    if not parts:
        return pd.DataFrame(columns=["symbol", "trade_date", *DEFAULT_FACTOR_COLUMNS])
    out = pd.concat(parts, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    return out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def build_entry_relative_strength_features(
    outcomes: pd.DataFrame,
    *,
    minute_dir: str | Path,
    sector_map_path: str | Path | None,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build causal entry-time market/industry relative-strength features.

    TickFlow currently provides minute OHLCV/amount but no Level-2 fields or
    index minute bars here, so the market benchmark is an equal-weight proxy
    from the same minute cache. All values are sampled at ``entry_idx``.
    """
    key_cols = ["symbol", "trade_date", "entry_idx"]
    empty = pd.DataFrame(columns=[*key_cols, *ENTRY_RELATIVE_FEATURE_COLUMNS])
    if outcomes is None or outcomes.empty or not set(key_cols).issubset(outcomes.columns):
        return empty
    keys = outcomes[key_cols].dropna().drop_duplicates().copy()
    if keys.empty:
        return empty
    keys["symbol"] = keys["symbol"].astype(str)
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce").dt.normalize()
    keys["entry_idx"] = pd.to_numeric(keys["entry_idx"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["trade_date", "entry_idx"])
    if keys.empty:
        return empty
    cache_path = _relative_feature_cache_path(
        keys,
        minute_dir=minute_dir,
        sector_map_path=sector_map_path,
        cache_dir=cache_dir,
    )
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        cached = _read_cached_relative_features(cache_path)
        if cached is not None:
            return cached
    sector_map = _load_sector_map(sector_map_path)
    needed = {
        sym: {
            pd.Timestamp(day): set(pd.to_numeric(group["entry_idx"], errors="coerce").dropna().astype(int).tolist())
            for day, group in sym_frame.groupby("trade_date", sort=False)
        }
        for sym, sym_frame in keys.groupby("symbol", sort=False)
    }
    rows = []
    minute_path = Path(minute_dir)
    for sym, day_map in needed.items():
        p = minute_path / f"{sym}.parquet"
        if not p.exists():
            continue
        try:
            bars = read_parquet_checked(p, columns=["symbol", "trade_time", "open", "close", "volume", "amount"])
        except Exception:  # noqa: BLE001
            continue
        bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
        bars = bars.dropna(subset=["trade_time"])
        if bars.empty:
            continue
        for day, g in bars.groupby(bars["trade_time"].dt.normalize(), sort=False):
            idxs = day_map.get(pd.Timestamp(day))
            if not idxs:
                continue
            g = g.sort_values("trade_time").reset_index(drop=True)
            close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype="float64")
            openp = pd.to_numeric(g["open"], errors="coerce").to_numpy(dtype="float64")
            volume = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            amount = pd.to_numeric(g.get("amount", g["close"] * g["volume"]), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            if len(close) == 0:
                continue
            day_open = float(openp[0]) if np.isfinite(openp[0]) and openp[0] > 0 else float(close[0])
            cum_v = np.cumsum(volume)
            cum_pv = np.cumsum(close * volume)
            vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), close)
            sector = sector_map.get(sym, "UNKNOWN")
            for raw_idx in idxs:
                idx = int(raw_idx)
                if idx < 0 or idx >= len(close):
                    continue
                px = float(close[idx])
                vw = float(vwap[idx])
                rows.append({
                    "symbol": sym,
                    "trade_date": pd.Timestamp(day),
                    "entry_idx": idx,
                    "_sector": sector,
                    "_stock_return_sofar": float(px / max(day_open, 1e-12) - 1.0),
                    "_stock_vwap_dev_sofar": float(px / max(vw, 1e-12) - 1.0) if vw > 0 else 0.0,
                    "_minute_volume": float(volume[idx]),
                    "_minute_amount": float(amount[idx]),
                })
    if not rows:
        return empty
    base = pd.DataFrame(rows)
    market = base.groupby(["trade_date", "entry_idx"], sort=False).agg(
        entry_market_return_sofar=("_stock_return_sofar", "mean"),
        _market_vwap_dev=("_stock_vwap_dev_sofar", "mean"),
        _market_volume_median=("_minute_volume", "median"),
        entry_market_breadth_sofar=("_stock_return_sofar", lambda s: float((s > 0).mean())),
        _up_count=("_stock_return_sofar", lambda s: float((s > 0).sum())),
        _down_count=("_stock_return_sofar", lambda s: float((s < 0).sum())),
    ).reset_index()
    market["entry_up_down_ratio_sofar"] = market["_up_count"] / market["_down_count"].clip(lower=1.0)
    industry = base.groupby(["trade_date", "entry_idx", "_sector"], sort=False).agg(
        entry_industry_return_sofar=("_stock_return_sofar", "mean"),
        _industry_vwap_dev=("_stock_vwap_dev_sofar", "mean"),
        _industry_volume_median=("_minute_volume", "median"),
        entry_industry_breadth_sofar=("_stock_return_sofar", lambda s: float((s > 0).mean())),
    ).reset_index()
    out = base.merge(market, on=["trade_date", "entry_idx"], how="left")
    out = out.merge(industry, on=["trade_date", "entry_idx", "_sector"], how="left")
    out["entry_stock_return_minus_market"] = out["_stock_return_sofar"] - out["entry_market_return_sofar"]
    out["entry_stock_return_minus_industry"] = out["_stock_return_sofar"] - out["entry_industry_return_sofar"]
    out["entry_stock_vwap_dev_minus_market"] = out["_stock_vwap_dev_sofar"] - out["_market_vwap_dev"]
    out["entry_stock_vwap_dev_minus_industry"] = out["_stock_vwap_dev_sofar"] - out["_industry_vwap_dev"]
    out["entry_relative_volume_vs_market"] = out["_minute_volume"] / out["_market_volume_median"].clip(lower=1.0)
    out["entry_relative_volume_vs_industry"] = out["_minute_volume"] / out["_industry_volume_median"].clip(lower=1.0)
    out["entry_industry_market_return_spread"] = out["entry_industry_return_sofar"] - out["entry_market_return_sofar"]
    result = out[[*key_cols, *ENTRY_RELATIVE_FEATURE_COLUMNS]].replace([np.inf, -np.inf], np.nan)
    result = result.drop_duplicates(key_cols, keep="first").reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def _load_sector_map(sector_map_path: str | Path | None) -> dict[str, str]:
    if not sector_map_path:
        return {}
    p = Path(sector_map_path)
    if not p.exists():
        return {}
    try:
        sector = read_parquet_checked(p)
    except Exception:  # noqa: BLE001
        return {}
    if "symbol" not in sector.columns:
        return {}
    if "sector_level_1" in sector.columns:
        col = "sector_level_1"
    elif "sector_level_2" in sector.columns:
        col = "sector_level_2"
    else:
        return {}
    sector = sector.dropna(subset=["symbol"]).copy()
    sector["symbol"] = sector["symbol"].astype(str)
    sector[col] = sector[col].fillna("UNKNOWN").astype(str)
    return sector.drop_duplicates("symbol").set_index("symbol")[col].to_dict()


def _relative_feature_cache_path(
    keys: pd.DataFrame,
    *,
    minute_dir: str | Path,
    sector_map_path: str | Path | None,
    cache_dir: str | Path | None,
) -> Path | None:
    if not cache_dir:
        return None
    sector_stat = Path(sector_map_path).stat() if sector_map_path and Path(sector_map_path).exists() else None
    payload = {
        "version": "entry_relative_strength_v1",
        "minute_dir": str(minute_dir),
        "sector_size": int(sector_stat.st_size) if sector_stat else 0,
        "sector_mtime_ns": int(sector_stat.st_mtime_ns) if sector_stat else 0,
        "keys_hash": _stable_hash(keys.sort_values(["symbol", "trade_date", "entry_idx"]).to_dict(orient="records")),
        "key_count": int(len(keys)),
    }
    return Path(cache_dir) / f"entry_relative_{_stable_hash(payload)[:20]}.parquet"


def _read_cached_relative_features(path: Path) -> pd.DataFrame | None:
    try:
        cached = read_parquet_checked(path)
    except Exception:  # noqa: BLE001
        return None
    required = {"symbol", "trade_date", "entry_idx", *ENTRY_RELATIVE_FEATURE_COLUMNS}
    if required.issubset(cached.columns):
        return cached
    return None


def build_dot_outcomes(
    *,
    minute_dir: str | Path,
    symbols: list[str],
    contexts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: FactorComboConfig,
) -> pd.DataFrame:
    context_cols = ["symbol", "trade_date", "atr_pct", "prev_close"]
    context_cols.extend(c for c in ("mom_5d", "gap_open", "regime") if c in contexts.columns)
    context_frame = contexts[[c for c in context_cols if c in contexts.columns]].dropna(subset=["atr_pct"]).copy()
    grouped_contexts = {
        str(sym): group.to_dict(orient="records")
        for sym, group in context_frame.groupby("symbol", sort=False)
    }
    jobs = [
        {
            "minute_dir": str(minute_dir),
            "symbol": str(sym),
            "context_records": grouped_contexts.get(str(sym), []),
            "start": str(pd.Timestamp(start).date()),
            "end": str(pd.Timestamp(end).date()),
            "config": config,
        }
        for sym in symbols
    ]
    if not jobs:
        return pd.DataFrame(columns=_outcome_columns())
    workers = max(1, int(config.outcome_workers or 1))
    if workers == 1:
        frames = [_build_symbol_dot_outcomes(job) for job in jobs]
    else:
        max_workers = min(workers, len(jobs), max(1, os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            frames = list(pool.map(_build_symbol_dot_outcomes, jobs, chunksize=1))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=_outcome_columns())
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_symbol_dot_outcomes(job: dict) -> pd.DataFrame:
    minute_dir = Path(str(job["minute_dir"]))
    sym = str(job["symbol"])
    context_records = list(job.get("context_records") or [])
    start = pd.Timestamp(job["start"])
    end = pd.Timestamp(job["end"])
    config: FactorComboConfig = job["config"]
    cache_path = _symbol_outcome_cache_path(
        minute_dir=minute_dir,
        symbol=sym,
        context_records=context_records,
        start=start,
        end=end,
        config=config,
    )
    if cache_path is not None and cache_path.exists() and not config.force_rebuild_outcome_cache:
        cached = _read_cached_outcome(cache_path)
        if cached is not None:
            return cached
    result = _compute_symbol_dot_outcomes(
        minute_dir=minute_dir,
        symbol=sym,
        context_records=context_records,
        start=start,
        end=end,
        config=config,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def _compute_symbol_dot_outcomes(
    *,
    minute_dir: Path,
    symbol: str,
    context_records: list[dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: FactorComboConfig,
) -> pd.DataFrame:
    p = minute_dir / f"{symbol}.parquet"
    if not p.exists() or not context_records:
        return pd.DataFrame(columns=_outcome_columns())
    contexts = pd.DataFrame(context_records)
    contexts["trade_date"] = pd.to_datetime(contexts["trade_date"], errors="coerce").dt.normalize()
    ctx_map = {pd.Timestamp(r.trade_date): r for r in contexts.itertuples()}
    combos = _fsm_combos()
    rows = []
    bars = read_parquet_checked(
        p,
        columns=["symbol", "trade_date", "trade_time", "open", "high", "low", "close", "volume", "amount"],
    )
    bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
    bars = bars[(bars["trade_time"] >= start) & (bars["trade_time"] <= end + pd.Timedelta(days=1))]
    if bars.empty:
        return pd.DataFrame(columns=_outcome_columns())
    for day, g in bars.groupby(bars["trade_time"].dt.normalize(), sort=False):
        ctx = ctx_map.get(pd.Timestamp(day))
        if ctx is None:
            continue
        atr = float(ctx.atr_pct)
        if not np.isfinite(atr) or atr <= 0:
            continue
        prev_close = float(getattr(ctx, "prev_close", np.nan))
        limit_up = prev_close * 1.10 if np.isfinite(prev_close) and prev_close > 0 else np.nan
        limit_down = prev_close * 0.90 if np.isfinite(prev_close) and prev_close > 0 else np.nan
        prepared = prepare_day_arrays(g.copy())
        if prepared is None:
            continue
        for combo_id, combo in combos.iterrows():
            mode = str(combo["mode"])
            params = SelectiveDotParams(
                mode=mode,
                dip_atr_mult=float(combo["dip_atr_mult"]),
                target_atr_mult=float(combo["target_atr_mult"]),
                stop_atr_mult=float(combo["stop_atr_mult"]),
                morning_deadline=str(combo["morning_deadline"]),
                eod_close=str(combo["tail_exit_time"]),
            )
            if mode.startswith("time_"):
                state, entry_px, exit_px, ret, entry_idx, exit_idx = _simulate_fixed_time_prepared(
                    prepared,
                    atr,
                    params,
                    mode,
                )
            else:
                state, entry_px, exit_px, ret, entry_idx, exit_idx = simulate_prepared(prepared, atr, params, mode)
            if ret is None or state == "waiting_no_entry":
                continue
            requested_qty = _quantity_for_notional(config.order_notional_yuan, float(entry_px))
            if requested_qty <= 0:
                continue
            entry_side = OrderSide.BUY if _is_buy_mode(mode) else OrderSide.SELL
            exit_side = OrderSide.SELL if _is_buy_mode(mode) else OrderSide.BUY
            entry_fill = _fast_conservative_fill(
                prepared,
                signal_index=int(entry_idx),
                side=entry_side,
                quantity=requested_qty,
                config=config,
                limit_up=limit_up,
                limit_down=limit_down,
            )
            if not _is_full_enough(entry_fill["filled_qty"], requested_qty, config.min_fill_ratio):
                continue
            planned_time_exit = state == "closed_eod"
            exit_fill = _fast_conservative_window_fill(
                prepared,
                signal_index=int(exit_idx),
                side=exit_side,
                quantity=entry_fill["filled_qty"],
                config=config,
                limit_up=limit_up,
                limit_down=limit_down,
                latest_time=config.tail_exit_deadline,
            )
            if _is_full_enough(exit_fill["filled_qty"], entry_fill["filled_qty"], config.min_fill_ratio):
                actual_exit_px = float(exit_fill["fill_price"])
                actual_exit_time = exit_fill["fill_time"]
                actual_exit_qty = int(exit_fill["filled_qty"])
                exit_fill_status = str(exit_fill["status"])
                exit_fill_reason = str(exit_fill["reason"])
                eod_restore = False
                state_out = "closed_time_exit" if planned_time_exit else state
                restore_source = "none"
            else:
                eod_restore = True
                actual_exit_px = _forced_eod_price(prepared, exit_side, config)
                actual_exit_time = str(prepared["time"][-1])
                actual_exit_qty = int(entry_fill["filled_qty"])
                exit_fill_status = "forced_eod_restore"
                exit_fill_reason = str(exit_fill["reason"])
                state_out = "closed_eod"
                restore_source = "tail_exit_fill_failed" if planned_time_exit else "event_exit_fill_failed"
            actual_entry_px = float(entry_fill["fill_price"])
            gross_ret = _gross_round_trip_return(mode, actual_entry_px, actual_exit_px)
            fee_ret = _round_trip_fee_return(
                entry_side,
                exit_side,
                actual_entry_px,
                actual_exit_px,
                actual_exit_qty,
                config,
            )
            restore_penalty = config.eod_restore_penalty_bps / 10_000.0 if eod_restore else 0.0
            net_ret = gross_ret - fee_ret - restore_penalty
            entry_features = _causal_entry_features(prepared, int(entry_idx), mode=mode)
            rows.append({
                "symbol": symbol,
                "trade_date": pd.Timestamp(day),
                "combo": int(combo_id),
                "mode": mode,
                "dip_atr_mult": combo["dip_atr_mult"],
                "target_atr_mult": combo["target_atr_mult"],
                "stop_atr_mult": combo["stop_atr_mult"],
                "morning_deadline": combo["morning_deadline"],
                "tail_exit_time": combo["tail_exit_time"],
                "tail_exit_minute": _time_to_minute(str(combo["tail_exit_time"])),
                "state": state_out,
                "time_exit": int(state_out == "closed_time_exit"),
                "eod_restore": int(eod_restore),
                "eod_restore_source": restore_source,
                "gross_ret": float(gross_ret),
                "net_ret": float(net_ret),
                "fee_cost_bps": float(fee_ret * 10_000.0),
                "restore_penalty_bps": float(restore_penalty * 10_000.0),
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_signal_px": entry_px,
                "exit_signal_px": exit_px,
                "entry_px": actual_entry_px,
                "exit_px": actual_exit_px,
                "requested_qty": int(requested_qty),
                "filled_qty": int(actual_exit_qty),
                "entry_fill_time": entry_fill["fill_time"],
                "exit_fill_time": actual_exit_time,
                "entry_fill_status": entry_fill["status"],
                "exit_fill_status": exit_fill_status,
                "entry_fill_reason": entry_fill["reason"],
                "exit_fill_reason": exit_fill_reason,
                "entry_participation_rate": float(entry_fill["participation_rate"]),
                "exit_participation_rate": float(exit_fill["participation_rate"]),
                "entry_volume_capacity_ratio": float(entry_fill["volume_capacity_ratio"]),
                "exit_volume_capacity_ratio": float(exit_fill["volume_capacity_ratio"]),
                **entry_features,
            })
    if not rows:
        return pd.DataFrame(columns=_outcome_columns())
    return pd.DataFrame(rows)


def _simulate_fixed_time_prepared(
    day: dict,
    atr_pct: float,
    params: SelectiveDotParams,
    mode: str,
) -> tuple[str, float | None, float | None, float | None, int | None, int | None]:
    o, h, l, c, t = day["open"], day["high"], day["low"], day["close"], day["time"]
    n = len(c)
    if n <= 2:
        return "waiting_no_entry", None, None, None, None, None
    e = int(np.searchsorted(t, params.morning_deadline, side="left"))
    e = max(int(params.min_bars_before_entry), e)
    if e < 0 or e >= n - 1:
        return "waiting_no_entry", None, None, None, None, None
    entry_px = float(c[e])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return "waiting_no_entry", None, None, None, None, None
    atr = float(atr_pct)
    if _is_buy_mode(mode):
        target = entry_px * (1.0 + float(params.target_atr_mult) * atr)
        stop = entry_px * (1.0 - float(params.stop_atr_mult) * atr)
    else:
        target = entry_px * (1.0 - float(params.target_atr_mult) * atr)
        stop = entry_px * (1.0 + float(params.stop_atr_mult) * atr)
    j0 = e + 1
    if _is_buy_mode(mode):
        stop_hits = l[j0:] <= stop
        target_hits = h[j0:] >= target
    else:
        stop_hits = h[j0:] >= stop
        target_hits = l[j0:] <= target
    js = int(np.argmax(stop_hits)) if stop_hits.any() else n
    jt = int(np.argmax(target_hits)) if target_hits.any() else n
    je = int(np.searchsorted(t[j0:], params.eod_close, side="left"))
    je = min(je, n - 1 - j0)
    first = min(js, jt, je)
    bar = j0 + first
    if first == js and js <= jt:
        px = min(stop, float(o[bar])) if _is_buy_mode(mode) else max(stop, float(o[bar]))
        state = "closed_stop"
    elif first == jt:
        px = max(target, float(o[bar])) if _is_buy_mode(mode) else min(target, float(o[bar]))
        state = "closed_profit"
    else:
        px = float(c[bar])
        state = "closed_eod"
    return state, entry_px, float(px), _gross_round_trip_return(mode, entry_px, px), e, bar


def _read_cached_outcome(path: Path) -> pd.DataFrame | None:
    try:
        cached = read_parquet_checked(path)
    except Exception:  # noqa: BLE001
        return None
    required = {
        "symbol",
        "trade_date",
        "net_ret",
        "entry_idx",
        "tail_exit_time",
        "entry_order_flow_imbalance_5m",
        "entry_mode_adverse_risk",
    }
    if required.issubset(cached.columns):
        return cached
    return None


def _symbol_outcome_cache_path(
    *,
    minute_dir: Path,
    symbol: str,
    context_records: list[dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: FactorComboConfig,
) -> Path | None:
    if not config.outcome_cache_dir:
        return None
    p = minute_dir / f"{symbol}.parquet"
    stat = p.stat() if p.exists() else None
    payload = {
        "version": "intraday_dot_outcome_v7",
        "symbol": symbol,
        "start": str(start.date()),
        "end": str(end.date()),
        "source_size": int(stat.st_size) if stat else 0,
        "source_mtime_ns": int(stat.st_mtime_ns) if stat else 0,
        "context_hash": _stable_hash(context_records),
        "fsm_grid": DEFAULT_FSM_GRID,
        "time_grid": DEFAULT_TIME_GRID,
        "order_notional_yuan": config.order_notional_yuan,
        "commission_bps": config.commission_bps,
        "stamp_bps": config.stamp_bps,
        "slippage_bps": config.slippage_bps,
        "spread_bps": config.spread_bps,
        "transfer_bps": config.transfer_bps,
        "max_minute_participation": config.max_minute_participation,
        "min_fill_ratio": config.min_fill_ratio,
        "eod_restore_penalty_bps": config.eod_restore_penalty_bps,
        "tail_exit_deadline": config.tail_exit_deadline,
    }
    digest = _stable_hash(payload)[:20]
    safe = symbol.replace("/", "_").replace("\\", "_")
    return Path(config.outcome_cache_dir) / f"{safe}_{digest}.parquet"


def _stable_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _outcome_columns() -> list[str]:
    return [
        "symbol",
        "trade_date",
        "combo",
        "mode",
        "dip_atr_mult",
        "target_atr_mult",
        "stop_atr_mult",
        "morning_deadline",
        "tail_exit_time",
        "tail_exit_minute",
        "state",
        "time_exit",
        "eod_restore",
        "eod_restore_source",
        "gross_ret",
        "net_ret",
        "fee_cost_bps",
        "restore_penalty_bps",
        "entry_idx",
        "exit_idx",
        "entry_signal_px",
        "exit_signal_px",
        "entry_px",
        "exit_px",
        "requested_qty",
        "filled_qty",
        "entry_fill_time",
        "exit_fill_time",
        "entry_fill_status",
        "exit_fill_status",
        "entry_fill_reason",
        "exit_fill_reason",
        "entry_participation_rate",
        "exit_participation_rate",
        "entry_volume_capacity_ratio",
        "exit_volume_capacity_ratio",
        "entry_minute_idx",
        "entry_price_vs_vwap_prev",
        "entry_return_from_open",
        "entry_range_pos_sofar",
        "entry_distance_to_hod_sofar",
        "entry_distance_to_lod_sofar",
        "entry_rolling_return_3m",
        "entry_rolling_return_5m",
        "entry_rolling_return_10m",
        "entry_rolling_volatility_5m",
        "entry_rolling_volatility_10m",
        "entry_volume_zscore_5m",
        "entry_volume_zscore_20m",
        "entry_volume_share_sofar",
        "entry_cum_volume",
        *ENTRY_CAUSAL_FEATURE_COLUMNS[15:],
    ]


def _quantity_for_notional(notional_yuan: float, price: float, lot_size: int = 100) -> int:
    if not np.isfinite(price) or price <= 0 or notional_yuan <= 0:
        return 0
    raw = int(float(notional_yuan) / float(price))
    return int(raw // lot_size * lot_size)


def _time_to_minute(value: str) -> float:
    try:
        hh, mm, *_ = str(value).split(":")
        return float(int(hh) * 60 + int(mm))
    except Exception:  # noqa: BLE001
        return 0.0


def _is_full_enough(filled_qty: int, requested_qty: int, min_fill_ratio: float) -> bool:
    if requested_qty <= 0:
        return False
    return int(filled_qty) >= int(requested_qty) * float(min_fill_ratio)


def _fast_conservative_fill(
    day: dict,
    *,
    signal_index: int,
    side: OrderSide,
    quantity: int,
    config: FactorComboConfig,
    limit_up: float,
    limit_down: float,
) -> dict:
    idx = int(signal_index) + 1
    n = len(day["close"])
    if idx < 0 or idx >= n:
        return _empty_fast_fill(quantity, "no_next_bar")
    open_px = float(day["open"][idx])
    close_px = float(day["close"][idx])
    if side == OrderSide.BUY and np.isfinite(limit_up):
        if max(open_px, close_px) >= float(limit_up) * 0.998:
            return _empty_fast_fill(quantity, "near_price_limit", fill_time=str(day["time"][idx]))
    if side == OrderSide.SELL and np.isfinite(limit_down):
        if min(open_px, close_px) <= float(limit_down) * 1.002:
            return _empty_fast_fill(quantity, "near_price_limit", fill_time=str(day["time"][idx]))
    penalty = (config.slippage_bps + config.spread_bps) / 10_000.0
    fill_price = open_px * (1.0 + penalty) if side == OrderSide.BUY else open_px * (1.0 - penalty)
    volume = max(0.0, float(day["volume"][idx]))
    capacity = int(volume * float(config.max_minute_participation)) // 100 * 100
    filled_qty = min(int(quantity), int(capacity)) // 100 * 100
    if filled_qty <= 0:
        return _empty_fast_fill(quantity, "no_liquidity", fill_time=str(day["time"][idx]), capacity=capacity)
    status = "filled" if filled_qty == int(quantity) else "partial"
    return {
        "filled_qty": int(filled_qty),
        "fill_price": float(fill_price),
        "fill_time": str(day["time"][idx]),
        "status": status,
        "reason": "filled" if status == "filled" else "capacity_partial",
        "participation_rate": float(filled_qty / volume) if volume > 0 else 0.0,
        "volume_capacity_ratio": float(int(quantity) / max(volume * float(config.max_minute_participation), 1.0))
        if volume > 0 else 0.0,
    }


def _fast_conservative_window_fill(
    day: dict,
    *,
    signal_index: int,
    side: OrderSide,
    quantity: int,
    config: FactorComboConfig,
    limit_up: float,
    limit_down: float,
    latest_time: str,
) -> dict:
    start_idx = int(signal_index) + 1
    n = len(day["close"])
    if start_idx < 0 or start_idx >= n:
        return _empty_fast_fill(quantity, "no_next_bar")
    latest = str(latest_time)
    remaining = int(quantity)
    filled = 0
    notional = 0.0
    volume_seen = 0.0
    first_time = ""
    last_time = ""
    for idx in range(start_idx, n):
        t = str(day["time"][idx])
        if t > latest:
            break
        open_px = float(day["open"][idx])
        close_px = float(day["close"][idx])
        if side == OrderSide.BUY and np.isfinite(limit_up):
            if max(open_px, close_px) >= float(limit_up) * 0.998:
                continue
        if side == OrderSide.SELL and np.isfinite(limit_down):
            if min(open_px, close_px) <= float(limit_down) * 1.002:
                continue
        volume = max(0.0, float(day["volume"][idx]))
        volume_seen += volume
        capacity = int(volume * float(config.max_minute_participation)) // 100 * 100
        qty = min(remaining, int(capacity)) // 100 * 100
        if qty <= 0:
            continue
        penalty = (config.slippage_bps + config.spread_bps) / 10_000.0
        px = open_px * (1.0 + penalty) if side == OrderSide.BUY else open_px * (1.0 - penalty)
        notional += float(qty) * float(px)
        filled += int(qty)
        remaining -= int(qty)
        first_time = first_time or t
        last_time = t
        if remaining <= 0:
            break
    if filled <= 0:
        return _empty_fast_fill(quantity, "no_liquidity_window", fill_time="", capacity=0)
    status = "filled" if filled >= int(quantity) else "partial"
    return {
        "filled_qty": int(filled),
        "fill_price": float(notional / max(filled, 1)),
        "fill_time": last_time or first_time,
        "status": status,
        "reason": "filled_window" if status == "filled" else "capacity_partial_window",
        "participation_rate": float(filled / volume_seen) if volume_seen > 0 else 0.0,
        "volume_capacity_ratio": float(int(quantity) / max(volume_seen * float(config.max_minute_participation), 1.0))
        if volume_seen > 0 else 0.0,
    }


def _empty_fast_fill(
    quantity: int,
    reason: str,
    *,
    fill_time: str = "",
    capacity: int = 0,
) -> dict:
    return {
        "filled_qty": 0,
        "fill_price": 0.0,
        "fill_time": fill_time,
        "status": "rejected",
        "reason": reason,
        "participation_rate": 0.0,
        "volume_capacity_ratio": float(int(quantity) / max(int(capacity), 1)),
    }


def _gross_round_trip_return(mode: str, entry_px: float, exit_px: float) -> float:
    if entry_px <= 0 or exit_px <= 0:
        return 0.0
    if _is_buy_mode(mode):
        return float(exit_px) / float(entry_px) - 1.0
    return float(entry_px) / float(exit_px) - 1.0


def _is_buy_mode(mode: str) -> bool:
    return str(mode) in {"dip_buy", "time_buy"}


def _is_sell_mode(mode: str) -> bool:
    return str(mode) in {"spike_sell", "time_sell"}


def _round_trip_fee_return(
    entry_side: OrderSide,
    exit_side: OrderSide,
    entry_px: float,
    exit_px: float,
    quantity: int,
    config: FactorComboConfig,
) -> float:
    base_notional = max(float(entry_px) * int(quantity), 1e-12)
    fees = (
        _leg_fee(entry_side, int(quantity), float(entry_px), config)
        + _leg_fee(exit_side, int(quantity), float(exit_px), config)
    )
    return float(fees / base_notional)


def _leg_fee(side: OrderSide, quantity: int, price: float, config: FactorComboConfig) -> float:
    notional = max(0.0, int(quantity) * float(price))
    if notional <= 0:
        return 0.0
    commission = notional * config.commission_bps / 10_000.0
    stamp = notional * config.stamp_bps / 10_000.0 if side == OrderSide.SELL else 0.0
    transfer = notional * config.transfer_bps / 10_000.0
    return float(commission + stamp + transfer)


def _forced_eod_price(day: dict, side: OrderSide, config: FactorComboConfig) -> float:
    close_px = float(day["close"][-1])
    penalty = (config.slippage_bps + config.spread_bps) / 10_000.0
    if side == OrderSide.BUY:
        return close_px * (1.0 + penalty)
    return close_px * (1.0 - penalty)


def _causal_entry_features(day: dict, entry_idx: int, *, mode: str = "") -> dict[str, float]:
    idx = max(0, min(int(entry_idx), len(day["close"]) - 1))
    o = day["open"][: idx + 1]
    h = day["high"][: idx + 1]
    l = day["low"][: idx + 1]
    c = day["close"][: idx + 1]
    v = day["volume"][: idx + 1]
    amount = day.get("amount")
    a = amount[: idx + 1] if amount is not None else c * v
    vwap_prev = day["vwap_prev"][idx] if "vwap_prev" in day else np.nan
    px = float(c[-1])
    day_open = float(o[0]) if len(o) else np.nan
    high_so_far = float(np.nanmax(h)) if len(h) else np.nan
    low_so_far = float(np.nanmin(l)) if len(l) else np.nan
    rng = max(high_so_far - low_so_far, 1e-12)

    def rolling_return(window: int) -> float:
        if len(c) <= window:
            return float(px / max(float(c[0]), 1e-12) - 1.0)
        return float(px / max(float(c[-window - 1]), 1e-12) - 1.0)

    def rolling_vol(window: int) -> float:
        if len(c) <= 2:
            return 0.0
        x = c[-min(window + 1, len(c)) :]
        r = pd.Series(x).pct_change(fill_method=None).dropna()
        return float(r.std(ddof=0)) if len(r) else 0.0

    def volume_z(window: int) -> float:
        if len(v) <= 2:
            return 0.0
        hist = v[-min(window + 1, len(v)) : -1]
        if len(hist) == 0:
            return 0.0
        sd = float(np.nanstd(hist))
        if sd <= 1e-12:
            return 0.0
        return float((v[-1] - np.nanmean(hist)) / sd)

    def amount_z(window: int) -> float:
        if len(a) <= 2:
            return 0.0
        hist = a[-min(window + 1, len(a)) : -1]
        if len(hist) == 0:
            return 0.0
        sd = float(np.nanstd(hist))
        if sd <= 1e-12:
            return 0.0
        return float((a[-1] - np.nanmean(hist)) / sd)

    def window_slice(window: int) -> slice:
        return slice(max(0, len(c) - int(window)), len(c))

    def signed_volume_imbalance(window: int | None = None) -> float:
        if len(c) <= 1:
            return 0.0
        sl = window_slice(window) if window else slice(0, len(c))
        cc = c[sl]
        vv = v[sl]
        if len(cc) <= 1:
            return 0.0
        rets = np.diff(cc)
        vols = vv[1:]
        up_vol = float(np.nansum(vols[rets > 0]))
        down_vol = float(np.nansum(vols[rets < 0]))
        total = up_vol + down_vol
        return float((up_vol - down_vol) / total) if total > 0 else 0.0

    def buy_pressure(window: int) -> float:
        if len(c) <= 1:
            return 0.0
        sl = window_slice(window)
        cc = c[sl]
        vv = v[sl]
        if len(cc) <= 1:
            return 0.0
        rets = np.diff(cc)
        vols = vv[1:]
        total = float(np.nansum(vols))
        if total <= 0:
            return 0.0
        return float(np.nansum(vols[rets > 0]) / total)

    def price_impact_bps(window: int) -> float:
        if len(c) <= 1:
            return 0.0
        sl = window_slice(window + 1)
        cc = c[sl]
        aa = a[sl]
        if len(cc) <= 1:
            return 0.0
        ret_bps = abs(float(cc[-1] / max(float(cc[0]), 1e-12) - 1.0)) * 10_000.0
        amount_m = float(np.nansum(aa)) / 1_000_000.0
        return float(ret_bps / max(amount_m, 1e-6))

    def vwap_slope(window: int) -> float:
        if len(c) <= 1:
            return 0.0
        sl = window_slice(window + 1)
        cc = c[sl]
        vv = v[sl]
        if len(cc) <= 1:
            return 0.0
        cum_v = np.cumsum(vv)
        cum_pv = np.cumsum(cc * vv)
        vw = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), cc)
        return float(vw[-1] / max(float(vw[0]), 1e-12) - 1.0)

    def close_location(window: int) -> float:
        sl = window_slice(window)
        hh = float(np.nanmax(h[sl])) if len(h[sl]) else high_so_far
        ll = float(np.nanmin(l[sl])) if len(l[sl]) else low_so_far
        return float((px - ll) / max(hh - ll, 1e-12))

    def trend_persistence(window: int, direction: str) -> float:
        if len(c) <= 1:
            return 0.0
        sl = window_slice(window + 1)
        rr = np.diff(c[sl])
        if len(rr) == 0:
            return 0.0
        if direction == "up":
            return float(np.nanmean(rr > 0))
        return float(np.nanmean(rr < 0))

    flow_sofar = signed_volume_imbalance()
    flow_5m = signed_volume_imbalance(5)
    flow_20m = signed_volume_imbalance(20)
    buy_5m = buy_pressure(5)
    sell_5m = 1.0 - buy_5m if len(c) > 1 else 0.0
    loc_5m = close_location(5)
    up_persist = trend_persistence(10, "up")
    down_persist = trend_persistence(10, "down")
    vol_total = float(np.nansum(v))
    if len(v) and vol_total > 0:
        top_n = min(5, len(v))
        large_share = float(np.sort(v)[-top_n:].sum() / vol_total)
    else:
        large_share = 0.0
    current_vol_share = float(v[-1] / max(vol_total, 1e-12)) if len(v) else 0.0
    near_hod_pressure = max(0.0, loc_5m - 0.75) * max(0.0, flow_5m) * (1.0 + current_vol_share)
    near_lod_pressure = max(0.0, 0.25 - loc_5m) * max(0.0, -flow_5m) * (1.0 + current_vol_share)
    failed_breakout = max(0.0, 0.85 - loc_5m) * max(0.0, flow_5m) * (1.0 + up_persist)
    failed_breakdown = max(0.0, loc_5m - 0.15) * max(0.0, -flow_5m) * (1.0 + down_persist)
    momentum_persistence = up_persist if _is_sell_mode(mode) else down_persist
    mean_reversion_quality = (
        (1.0 - abs(float((px - low_so_far) / rng) - 0.5))
        + max(0.0, -flow_5m if _is_buy_mode(mode) else flow_5m)
        - 0.5 * momentum_persistence
    )
    if _is_buy_mode(mode):
        adverse_risk = 0.45 * down_persist + 0.35 * max(0.0, -flow_5m) + 0.20 * near_lod_pressure
    elif _is_sell_mode(mode):
        adverse_risk = 0.45 * up_persist + 0.35 * max(0.0, flow_5m) + 0.20 * near_hod_pressure
    else:
        adverse_risk = 0.5 * max(up_persist, down_persist) + 0.5 * abs(flow_5m)

    minute_of_day = float(idx)
    return {
        "entry_minute_idx": minute_of_day,
        "entry_price_vs_vwap_prev": float(px / vwap_prev - 1.0) if np.isfinite(vwap_prev) and vwap_prev > 0 else 0.0,
        "entry_return_from_open": float(px / day_open - 1.0) if np.isfinite(day_open) and day_open > 0 else 0.0,
        "entry_range_pos_sofar": float((px - low_so_far) / rng),
        "entry_distance_to_hod_sofar": float(px / max(high_so_far, 1e-12) - 1.0),
        "entry_distance_to_lod_sofar": float(px / max(low_so_far, 1e-12) - 1.0),
        "entry_rolling_return_3m": rolling_return(3),
        "entry_rolling_return_5m": rolling_return(5),
        "entry_rolling_return_10m": rolling_return(10),
        "entry_rolling_volatility_5m": rolling_vol(5),
        "entry_rolling_volatility_10m": rolling_vol(10),
        "entry_volume_zscore_5m": volume_z(5),
        "entry_volume_zscore_20m": volume_z(20),
        "entry_volume_share_sofar": float(v[-1] / max(np.nansum(v), 1e-12)) if len(v) else 0.0,
        "entry_cum_volume": float(np.nansum(v)),
        "entry_amount_zscore_5m": amount_z(5),
        "entry_amount_zscore_20m": amount_z(20),
        "entry_order_flow_imbalance_sofar": flow_sofar,
        "entry_order_flow_imbalance_5m": flow_5m,
        "entry_order_flow_imbalance_20m": flow_20m,
        "entry_buy_pressure_5m": buy_5m,
        "entry_sell_pressure_5m": sell_5m,
        "entry_price_impact_bps_5m": price_impact_bps(5),
        "entry_price_impact_bps_20m": price_impact_bps(20),
        "entry_vwap_slope_5m": vwap_slope(5),
        "entry_close_location_5m": loc_5m,
        "entry_large_volume_share_sofar": large_share,
        "entry_uptrend_persistence_10m": up_persist,
        "entry_downtrend_persistence_10m": down_persist,
        "entry_failed_breakout_proxy": failed_breakout,
        "entry_failed_breakdown_proxy": failed_breakdown,
        "entry_near_hod_volume_pressure": near_hod_pressure,
        "entry_near_lod_volume_pressure": near_lod_pressure,
        "entry_momentum_persistence": float(momentum_persistence),
        "entry_mean_reversion_quality": float(mean_reversion_quality),
        "entry_mode_adverse_risk": float(np.clip(adverse_risk, 0.0, 1.0)),
    }


def train_factor_combo_model(
    dataset: pd.DataFrame,
    *,
    config: FactorComboConfig,
    backend: str = "lightgbm",
) -> tuple[object, list[str], pd.DataFrame, dict]:
    train_end = pd.Timestamp(config.split)
    validation_end = pd.Timestamp(config.validation_split)
    if validation_end <= train_end:
        raise ValueError("validation_split must be later than split/train_end")
    data = dataset.dropna(subset=["target_net_ret_bps"]).copy()
    state = data["state"] if "state" in data.columns else pd.Series("", index=data.index)
    if "eod_restore" not in data.columns:
        data["eod_restore"] = state.eq("closed_eod").astype(int)
    data["stop_exit"] = state.eq("closed_stop").astype(int)
    feature_cols = [c for c in _feature_columns(data) if data[c].notna().any()]
    train = data[data["trade_date"] <= train_end].copy()
    validation = data[(data["trade_date"] > train_end) & (data["trade_date"] <= validation_end)].copy()
    test = data[data["trade_date"] > validation_end].copy()
    if len(train) < config.min_train_legs:
        raise ValueError(f"not enough train rows: {len(train)} < {config.min_train_legs}")
    if validation.empty:
        raise ValueError("validation split has no rows; move validation_split later or expand the date range")
    model = _fit_regressor(train, feature_cols, backend=backend, random_seed=config.random_seed)
    risk_model = _fit_eod_classifier(train, feature_cols, backend=backend, random_seed=config.random_seed)
    stop_model = _fit_stop_classifier(train, feature_cols, backend=backend, random_seed=config.random_seed + 11)
    for frame in (train, validation, test):
        if frame.empty:
            frame["pred_net_ret_bps"] = pd.Series(dtype=float)
            frame["pred_eod_restore_prob"] = pd.Series(dtype=float)
            frame["pred_stop_prob"] = pd.Series(dtype=float)
            continue
        frame["pred_net_ret_bps"] = model.predict(frame[feature_cols])
        frame["pred_eod_restore_prob"] = _predict_eod_prob(risk_model, frame[feature_cols])
        frame["pred_stop_prob"] = _predict_stop_prob(stop_model, frame[feature_cols])
        frame["pred_policy_score_bps"] = _policy_score(frame, config)
    policy_grid = []
    for frac in config.top_fracs:
        for cap in config.eod_restore_prob_caps:
            for stop_cap in config.stop_prob_caps:
                for risk_cap in config.entry_adverse_risk_caps:
                    for min_quality in config.min_reversion_qualities:
                        selection = _evaluate_selection(
                            validation,
                            frac=frac,
                            config=config,
                            name=(
                                f"validation_top_{frac:g}_eod_{cap:g}_stop_{stop_cap:g}"
                                f"_risk_{risk_cap:g}_mr_{min_quality:g}"
                            ),
                            max_eod_restore_prob=cap,
                            max_stop_prob=stop_cap,
                            max_entry_adverse_risk=risk_cap,
                            min_entry_mean_reversion_quality=min_quality,
                        )
                        baselines = _evaluate_baseline_set(
                            validation,
                            frac=frac,
                            config=config,
                            max_eod_restore_prob=cap,
                            max_stop_prob=stop_cap,
                            max_entry_adverse_risk=risk_cap,
                            min_entry_mean_reversion_quality=min_quality,
                        )
                        policy_grid.append(_attach_excess_metrics(selection, baselines))
    policy_frame = pd.DataFrame(policy_grid)
    policy = _choose_validation_policy(policy_frame, config)
    chosen_frac = float(policy["top_frac"])
    chosen_eod_cap = float(policy["max_eod_restore_prob"])
    chosen_stop_cap = float(policy.get("max_stop_prob", 1.0))
    chosen_risk_cap = float(policy.get("max_entry_adverse_risk", 1.0))
    chosen_min_quality = float(policy.get("min_entry_mean_reversion_quality", -1.0))
    train_eval = _evaluate_selection(
        train,
        frac=chosen_frac,
        config=config,
        name="train_chosen",
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    )
    train_baselines = _evaluate_baseline_set(
        train,
        frac=chosen_frac,
        config=config,
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    )
    train_eval = _attach_excess_metrics(train_eval, train_baselines)
    validation_eval = _evaluate_selection(
        validation,
        frac=chosen_frac,
        config=config,
        name="validation_chosen",
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    )
    validation_baselines = _evaluate_baseline_set(
        validation,
        frac=chosen_frac,
        config=config,
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    )
    validation_eval = _attach_excess_metrics(validation_eval, validation_baselines)
    test_eval = _evaluate_selection(
        test,
        frac=chosen_frac,
        config=config,
        name="test_chosen",
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    ) if not test.empty else {"n_legs": 0}
    test_baselines = _evaluate_baseline_set(
        test,
        frac=chosen_frac,
        config=config,
        max_eod_restore_prob=chosen_eod_cap,
        max_stop_prob=chosen_stop_cap,
        max_entry_adverse_risk=chosen_risk_cap,
        min_entry_mean_reversion_quality=chosen_min_quality,
    ) if not test.empty else {}
    test_eval = _attach_excess_metrics(test_eval, test_baselines)
    random_eval = test_baselines.get("random_time_same_count_baseline", {"n_legs": 0})
    shuffled_eval = test_baselines.get("shuffled_signal_baseline", {"n_legs": 0})
    vwap_eval = test_baselines.get("vwap_only_baseline", {"n_legs": 0})
    metrics = {
        "chosen_top_frac": chosen_frac,
        "chosen_max_eod_restore_prob": chosen_eod_cap,
        "chosen_max_stop_prob": chosen_stop_cap,
        "chosen_max_entry_adverse_risk": chosen_risk_cap,
        "chosen_min_entry_mean_reversion_quality": chosen_min_quality,
        "policy_selected_on": "validation",
        "min_validation_legs": int(config.min_validation_legs),
        "min_oos_legs": int(config.min_oos_legs),
        "require_book_for_enable": bool(config.require_book_for_enable),
        "selection_book_only": bool(config.selection_book_only),
        "min_validation_book_legs": int(config.min_validation_book_legs),
        "min_oos_book_legs": int(config.min_oos_book_legs),
        "max_validation_eod_restore_rate": float(config.max_validation_eod_restore_rate),
        "max_validation_stop_rate": float(config.max_validation_stop_rate),
        "book_candidate_coverage": {
            "train": _book_candidate_coverage(train, config),
            "validation": _book_candidate_coverage(validation, config),
            "test": _book_candidate_coverage(test, config),
        },
        "policy_grid": policy_frame.to_dict(orient="records"),
        "train": train_eval,
        "validation": validation_eval,
        "test": test_eval,
        "train_baselines": train_baselines,
        "validation_baselines": validation_baselines,
        "test_baselines": test_baselines,
        "random_time_same_count_baseline": random_eval,
        "shuffled_signal_baseline": shuffled_eval,
        "vwap_only_baseline": vwap_eval,
        "feature_columns": feature_cols,
    }
    scored = pd.concat([train, validation, test], ignore_index=True)
    return model, feature_cols, scored, metrics


def feature_importance_frame(model: object, feature_cols: list[str]) -> pd.DataFrame:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        values = np.zeros(len(feature_cols), dtype=float)
    if len(values) < len(feature_cols):
        values = np.pad(values, (0, len(feature_cols) - len(values)), constant_values=0.0)
    elif len(values) > len(feature_cols):
        values = values[: len(feature_cols)]
    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False)


def verdict_from_metrics(metrics: dict) -> tuple[str, str]:
    test = metrics.get("test", {})
    validation = metrics.get("validation", {})
    rnd = metrics.get("random_time_same_count_baseline", {})
    shuf = metrics.get("shuffled_signal_baseline", {})
    vwap = metrics.get("vwap_only_baseline", {})
    min_validation_legs = int(metrics.get("min_validation_legs", 100) or 100)
    min_oos_legs = int(metrics.get("min_oos_legs", 300) or 300)
    require_book = bool(metrics.get("require_book_for_enable", True))
    min_validation_book_legs = int(metrics.get("min_validation_book_legs", 30) or 0)
    min_oos_book_legs = int(metrics.get("min_oos_book_legs", 100) or 0)
    max_validation_eod = float(metrics.get("max_validation_eod_restore_rate", 0.35) or 0.35)
    max_validation_stop = float(metrics.get("max_validation_stop_rate", 0.35) or 0.35)
    n = int(test.get("n_legs", 0) or 0)
    validation_n = int(validation.get("n_legs", 0) or 0)
    validation_book_n = int(validation.get("book_n_legs", 0) or 0)
    test_book_n = int(test.get("book_n_legs", 0) or 0)
    validation_book_daily = float(validation.get("book_daily_uplift_bps", 0.0) or 0.0)
    test_book_daily = float(test.get("book_daily_uplift_bps", 0.0) or 0.0)
    if n <= 0:
        if bool(metrics.get("selection_book_only", False)):
            coverage = metrics.get("book_candidate_coverage", {}).get("test", {})
            book_candidates = int(coverage.get("book_symbol_days", 0) or 0)
            positive_preds = int(coverage.get("book_positive_pred_symbol_days", 0) or 0)
            return (
                "DO_NOT_ENABLE",
                "book-only OOS 没有可执行 legs "
                f"(book candidates {book_candidates}, positive-pred candidates {positive_preds})",
            )
        return "DO_NOT_ENABLE", "OOS 没有可执行 legs"
    if validation_n < min_validation_legs:
        return "DO_NOT_ENABLE", "validation selected legs below minimum"
    if (
        float(validation.get("mean_net_bps", 0.0)) <= 0
        or float(validation.get("daily_uplift_bps_excess", 0.0)) <= 0
        or float(validation.get("eod_restore_rate", 1.0)) > max_validation_eod
        or float(validation.get("stop_rate", 1.0)) > max_validation_stop
    ):
        return "DO_NOT_ENABLE", "validation net/excess/risk gates failed"
    if n < min_oos_legs:
        return "PAPER_ONLY", f"信号未证伪，但 OOS completed/selected legs 少于 {min_oos_legs}，不能部署"
    oos_passed = (
        float(test.get("mean_net_bps", 0.0)) > 0
        and float(test.get("daily_uplift_bps", 0.0)) > 0
        and float(test.get("daily_uplift_bps_excess", 0.0)) > 0
        and float(test.get("daily_uplift_bps", 0.0)) > float(rnd.get("daily_uplift_bps", 0.0))
        and float(test.get("daily_uplift_bps", 0.0)) > float(shuf.get("daily_uplift_bps", 0.0))
        and float(test.get("daily_uplift_bps", 0.0)) > float(vwap.get("daily_uplift_bps", 0.0))
        and float(test.get("eod_restore_rate", 1.0)) <= 0.20
        and float(test.get("stop_rate", 1.0)) <= 0.35
    )
    if not oos_passed:
        return "DO_NOT_ENABLE", "OOS net/excess/baseline gates failed"
    if require_book and (
        validation_book_n < min_validation_book_legs
        or test_book_n < min_oos_book_legs
    ):
        return (
            "PAPER_ONLY",
            "book executable legs below minimum "
            f"(validation {validation_book_n}/{min_validation_book_legs}, OOS {test_book_n}/{min_oos_book_legs})",
        )
    if require_book and (validation_book_daily <= 0.0 or test_book_daily <= 0.0):
        return "DO_NOT_ENABLE", "book-level validation/OOS uplift gate failed"
    return "ENABLE", "conservative OOS excess and book gates passed"


def _fsm_combos() -> pd.DataFrame:
    import itertools

    rows = [dict(zip(DEFAULT_FSM_GRID, vals)) for vals in itertools.product(*DEFAULT_FSM_GRID.values())]
    rows.extend(dict(zip(DEFAULT_TIME_GRID, vals)) for vals in itertools.product(*DEFAULT_TIME_GRID.values()))
    return pd.DataFrame(rows)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "atr_pct",
        "mom_5d",
        "gap_open",
        "mode_dip_buy",
        "mode_spike_sell",
        "mode_time_entry",
        "dip_atr_mult",
        "target_atr_mult",
        "stop_atr_mult",
        "tail_exit_minute",
        "regime_bull",
        "regime_sideways",
    ]
    cols.extend(c for c in df.columns if c.startswith("prev_"))
    cols.extend(c for c in ENTRY_CAUSAL_FEATURE_COLUMNS if c in df.columns)
    cols.extend(c for c in ENTRY_RELATIVE_FEATURE_COLUMNS if c in df.columns)
    return [c for c in cols if c in df.columns]


def _fit_regressor(train: pd.DataFrame, feature_cols: list[str], *, backend: str, random_seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    backend = backend.lower()
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_seed,
            verbosity=-1,
        )
    elif backend == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_seed,
            objective="reg:squarederror",
            verbosity=0,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        estimator = HistGradientBoostingRegressor(random_state=random_seed)
    return Pipeline([
        ("imputer", _named_imputer()),
        ("model", estimator),
    ]).fit(train[feature_cols], train["target_net_ret_bps"])


def _fit_eod_classifier(train: pd.DataFrame, feature_cols: list[str], *, backend: str, random_seed: int):
    return _fit_binary_classifier(
        train,
        feature_cols,
        target=pd.to_numeric(train["eod_restore"], errors="coerce").fillna(0).astype(int),
        backend=backend,
        random_seed=random_seed,
    )


def _fit_stop_classifier(train: pd.DataFrame, feature_cols: list[str], *, backend: str, random_seed: int):
    return _fit_binary_classifier(
        train,
        feature_cols,
        target=pd.to_numeric(train["stop_exit"], errors="coerce").fillna(0).astype(int),
        backend=backend,
        random_seed=random_seed,
    )


def _fit_binary_classifier(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: pd.Series,
    backend: str,
    random_seed: int,
):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    min_class = int(y.value_counts().min()) if len(y) else 0
    if y.nunique() < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(y.iloc[0]) if len(y) else 0)
    elif min_class < 2:
        estimator = DummyClassifier(strategy="prior")
    else:
        backend = backend.lower()
        if backend == "lightgbm":
            from lightgbm import LGBMClassifier

            base = LGBMClassifier(
                objective="binary",
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_seed,
                verbosity=-1,
            )
        elif backend == "xgboost":
            from xgboost import XGBClassifier

            base = XGBClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_seed,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            base = HistGradientBoostingClassifier(random_state=random_seed)
        cv = max(2, min(3, min_class))
        if len(train) <= 120_000:
            estimator = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        else:
            estimator = base
    return Pipeline([
        ("imputer", _named_imputer()),
        ("model", estimator),
    ]).fit(train[feature_cols], y)


def _named_imputer():
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    if hasattr(imputer, "set_output"):
        imputer.set_output(transform="pandas")
    return imputer


def _predict_eod_prob(model: object, features: pd.DataFrame) -> np.ndarray:
    return _predict_binary_positive_prob(model, features)


def _predict_stop_prob(model: object, features: pd.DataFrame) -> np.ndarray:
    return _predict_binary_positive_prob(model, features)


def _predict_binary_positive_prob(model: object, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is not None and 1 in list(classes):
        idx = list(classes).index(1)
    elif proba.shape[1] == 2:
        idx = 1
    else:
        return np.zeros(len(features), dtype=float)
    return np.asarray(proba[:, idx], dtype=float)


def _choose_validation_policy(policy_frame: pd.DataFrame, config: FactorComboConfig) -> dict:
    if policy_frame.empty:
        raise ValueError("empty validation policy grid")
    frame = policy_frame.copy()
    frame["n_legs"] = _numeric_frame_column(frame, "n_legs", default=0.0, fill=0.0)
    frame["daily_uplift_bps_excess"] = _numeric_frame_column(
        frame, "daily_uplift_bps_excess", default=-1e9, fill=-1e9
    )
    frame["mean_net_bps"] = _numeric_frame_column(frame, "mean_net_bps", default=-1e9, fill=-1e9)
    frame["eod_restore_rate"] = _numeric_frame_column(frame, "eod_restore_rate", default=1.0, fill=1.0)
    frame["stop_rate"] = _numeric_frame_column(frame, "stop_rate", default=1.0, fill=1.0)
    frame["avg_entry_adverse_risk"] = _numeric_frame_column(frame, "avg_entry_adverse_risk", default=1.0, fill=1.0)
    frame["avg_stop_prob"] = _numeric_frame_column(frame, "avg_stop_prob", default=1.0, fill=1.0)
    frame["book_n_legs"] = _numeric_frame_column(frame, "book_n_legs", default=0.0, fill=0.0)
    frame["book_daily_uplift_bps"] = _numeric_frame_column(frame, "book_daily_uplift_bps", default=0.0, fill=0.0)
    eligible = frame[
        (frame["n_legs"] >= int(config.min_validation_legs))
        & (frame["mean_net_bps"] > 0)
        & (frame["daily_uplift_bps_excess"] > 0)
        & (frame["eod_restore_rate"] <= float(config.max_validation_eod_restore_rate))
        & (frame["stop_rate"] <= float(config.max_validation_stop_rate))
    ].copy()
    if bool(config.require_book_for_enable) and not eligible.empty:
        book_eligible = eligible[
            (eligible["book_n_legs"] >= int(config.min_validation_book_legs))
            & (eligible["book_daily_uplift_bps"] > 0)
        ].copy()
        if not book_eligible.empty:
            eligible = book_eligible
    if eligible.empty:
        eligible = frame[
            (frame["n_legs"] >= int(config.min_validation_legs))
            & (frame["mean_net_bps"] > 0)
            & (frame["daily_uplift_bps_excess"] > 0)
        ].copy()
    if eligible.empty:
        eligible = frame[frame["n_legs"] >= int(config.min_validation_legs)].copy()
    if eligible.empty:
        eligible = frame
    eligible["robust_score"] = (
        eligible["daily_uplift_bps_excess"]
        - float(config.policy_eod_penalty_bps) * eligible["eod_restore_rate"]
        - float(config.policy_stop_penalty_bps) * eligible["stop_rate"]
        - 20.0 * eligible["avg_stop_prob"]
        - 25.0 * eligible["avg_entry_adverse_risk"]
        + 0.002 * np.minimum(eligible["n_legs"], int(config.min_oos_legs))
        + 0.25 * eligible["book_daily_uplift_bps"]
        + 0.001 * np.minimum(eligible["book_n_legs"], int(config.min_oos_book_legs))
    )
    ordered = eligible.sort_values(
        [
            "robust_score",
            "daily_uplift_bps_excess",
            "book_daily_uplift_bps",
            "n_legs",
            "book_n_legs",
            "eod_restore_rate",
            "stop_rate",
        ],
        ascending=[False, False, False, False, False, True, True],
    )
    return ordered.iloc[0].to_dict()


def _numeric_frame_column(frame: pd.DataFrame, column: str, *, default: float, fill: float) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(fill)


def _policy_score(df: pd.DataFrame, config: FactorComboConfig) -> pd.Series:
    pred = pd.to_numeric(df.get("pred_net_ret_bps", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    eod = pd.to_numeric(df.get("pred_eod_restore_prob", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    stop = pd.to_numeric(df.get("pred_stop_prob", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    adverse = pd.to_numeric(df.get("entry_mode_adverse_risk", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    return (
        pred
        - float(config.selection_eod_prob_penalty_bps) * eod
        - float(config.selection_stop_prob_penalty_bps) * stop
        - float(config.selection_entry_adverse_penalty_bps) * adverse
    )


def _score_column(df: pd.DataFrame) -> str:
    return "pred_policy_score_bps" if "pred_policy_score_bps" in df.columns else "pred_net_ret_bps"


def _best_pred_per_symbol_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    score_col = _score_column(df)
    order = df.sort_values(["trade_date", "symbol", score_col], ascending=[True, True, False])
    return order.drop_duplicates(["trade_date", "symbol"], keep="first").reset_index(drop=True)


def _selection_candidate_frame(df: pd.DataFrame, config: FactorComboConfig) -> pd.DataFrame:
    candidates = _best_pred_per_symbol_day(df)
    if candidates.empty or not bool(config.selection_book_only):
        return candidates
    if "weight" not in candidates.columns:
        return candidates.iloc[0:0].copy()
    weight = pd.to_numeric(candidates["weight"], errors="coerce")
    return candidates[weight > 0].copy()


def _book_candidate_coverage(df: pd.DataFrame, config: FactorComboConfig) -> dict:
    candidates = _best_pred_per_symbol_day(df)
    if candidates.empty:
        return {
            "candidate_symbol_days": 0,
            "book_symbol_days": 0,
            "book_symbols": 0,
            "book_days": 0,
            "book_coverage_rate": 0.0,
            "book_positive_pred_symbol_days": 0,
            "book_positive_pred_rate": 0.0,
        }
    if "weight" not in candidates.columns:
        book = candidates.iloc[0:0].copy()
    else:
        weight = pd.to_numeric(candidates["weight"], errors="coerce")
        book = candidates[weight > 0].copy()
    if book.empty:
        positive = book
    else:
        score_col = _score_column(book)
        score = pd.to_numeric(book[score_col], errors="coerce").fillna(-np.inf)
        positive = book[score >= float(config.min_pred_net_bps)]
    candidate_n = int(len(candidates))
    book_n = int(len(book))
    return {
        "candidate_symbol_days": candidate_n,
        "book_symbol_days": book_n,
        "book_symbols": int(book["symbol"].nunique()) if book_n else 0,
        "book_days": int(book["trade_date"].nunique()) if book_n else 0,
        "book_coverage_rate": float(book_n / candidate_n) if candidate_n else 0.0,
        "book_positive_pred_symbol_days": int(len(positive)),
        "book_positive_pred_rate": float(len(positive) / book_n) if book_n else 0.0,
    }


def _evaluate_selection(
    df: pd.DataFrame,
    *,
    frac: float,
    config: FactorComboConfig,
    name: str,
    max_eod_restore_prob: float = 1.0,
    max_stop_prob: float = 1.0,
    max_entry_adverse_risk: float = 1.0,
    min_entry_mean_reversion_quality: float = -1.0,
) -> dict:
    best = _selection_candidate_frame(df, config)
    if best.empty:
        return _selection_metrics(
            best,
            config=config,
            name=name,
            top_frac=frac,
            threshold=0.0,
            max_eod_restore_prob=max_eod_restore_prob,
            max_stop_prob=max_stop_prob,
            max_entry_adverse_risk=max_entry_adverse_risk,
            min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
        )
    score_col = _score_column(best)
    threshold = float(best[score_col].quantile(max(0.0, min(1.0, 1.0 - frac))))
    threshold = max(threshold, float(config.min_pred_net_bps))
    adverse = pd.to_numeric(best.get("entry_mode_adverse_risk", pd.Series(0.0, index=best.index)), errors="coerce").fillna(0.0)
    quality = pd.to_numeric(best.get("entry_mean_reversion_quality", pd.Series(0.0, index=best.index)), errors="coerce").fillna(0.0)
    selected = best[
        (best[score_col] >= threshold)
        & (best.get("pred_eod_restore_prob", pd.Series(0.0, index=best.index)) <= float(max_eod_restore_prob))
        & (best.get("pred_stop_prob", pd.Series(0.0, index=best.index)) <= float(max_stop_prob))
        & (adverse <= float(max_entry_adverse_risk))
        & (quality >= float(min_entry_mean_reversion_quality))
    ].copy()
    return _selection_metrics(
        selected,
        config=config,
        name=name,
        top_frac=frac,
        threshold=threshold,
        max_eod_restore_prob=max_eod_restore_prob,
        max_stop_prob=max_stop_prob,
        max_entry_adverse_risk=max_entry_adverse_risk,
        min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
    )


def _evaluate_random_baseline(
    df: pd.DataFrame,
    *,
    frac: float,
    config: FactorComboConfig,
    max_eod_restore_prob: float = 1.0,
    max_stop_prob: float = 1.0,
    max_entry_adverse_risk: float = 1.0,
    min_entry_mean_reversion_quality: float = -1.0,
) -> dict:
    best = _selection_candidate_frame(df, config)
    if best.empty:
        return {"name": "random_time_same_count_baseline", "top_frac": frac, "n_legs": 0}
    rng = np.random.default_rng(config.random_seed)
    best = best.copy()
    best["pred_net_ret_bps"] = rng.random(len(best))
    best["pred_policy_score_bps"] = best["pred_net_ret_bps"]
    return _evaluate_selection(
        best,
        frac=frac,
        config=config,
        name="random_time_same_count_baseline",
        max_eod_restore_prob=max_eod_restore_prob,
        max_stop_prob=max_stop_prob,
        max_entry_adverse_risk=max_entry_adverse_risk,
        min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
    )


def _evaluate_shuffled_baseline(
    df: pd.DataFrame,
    *,
    frac: float,
    config: FactorComboConfig,
    max_eod_restore_prob: float = 1.0,
    max_stop_prob: float = 1.0,
    max_entry_adverse_risk: float = 1.0,
    min_entry_mean_reversion_quality: float = -1.0,
) -> dict:
    best = _selection_candidate_frame(df, config)
    if best.empty:
        return {"name": "shuffled_signal_baseline", "top_frac": frac, "n_legs": 0}
    rng = np.random.default_rng(config.random_seed + 7)
    best = best.copy()
    score_col = _score_column(best)
    shuffled = rng.permutation(best[score_col].to_numpy())
    best["pred_net_ret_bps"] = shuffled
    best["pred_policy_score_bps"] = shuffled
    return _evaluate_selection(
        best,
        frac=frac,
        config=config,
        name="shuffled_signal_baseline",
        max_eod_restore_prob=max_eod_restore_prob,
        max_stop_prob=max_stop_prob,
        max_entry_adverse_risk=max_entry_adverse_risk,
        min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
    )


def _evaluate_vwap_baseline(
    df: pd.DataFrame,
    *,
    frac: float,
    config: FactorComboConfig,
    max_eod_restore_prob: float = 1.0,
    max_stop_prob: float = 1.0,
    max_entry_adverse_risk: float = 1.0,
    min_entry_mean_reversion_quality: float = -1.0,
) -> dict:
    best = _selection_candidate_frame(df, config)
    if best.empty:
        return {"name": "vwap_only_baseline", "top_frac": frac, "n_legs": 0}
    best = best.copy()
    score = best.get("prev_vwap_deviation", pd.Series(0.0, index=best.index)).abs()
    best["pred_net_ret_bps"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    best["pred_policy_score_bps"] = best["pred_net_ret_bps"]
    return _evaluate_selection(
        best,
        frac=frac,
        config=config,
        name="vwap_only_baseline",
        max_eod_restore_prob=max_eod_restore_prob,
        max_stop_prob=max_stop_prob,
        max_entry_adverse_risk=max_entry_adverse_risk,
        min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
    )


def _evaluate_baseline_set(
    df: pd.DataFrame,
    *,
    frac: float,
    config: FactorComboConfig,
    max_eod_restore_prob: float = 1.0,
    max_stop_prob: float = 1.0,
    max_entry_adverse_risk: float = 1.0,
    min_entry_mean_reversion_quality: float = -1.0,
) -> dict[str, dict]:
    return {
        "random_time_same_count_baseline": _evaluate_random_baseline(
            df,
            frac=frac,
            config=config,
            max_eod_restore_prob=max_eod_restore_prob,
            max_stop_prob=max_stop_prob,
            max_entry_adverse_risk=max_entry_adverse_risk,
            min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
        ),
        "shuffled_signal_baseline": _evaluate_shuffled_baseline(
            df,
            frac=frac,
            config=config,
            max_eod_restore_prob=max_eod_restore_prob,
            max_stop_prob=max_stop_prob,
            max_entry_adverse_risk=max_entry_adverse_risk,
            min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
        ),
        "vwap_only_baseline": _evaluate_vwap_baseline(
            df,
            frac=frac,
            config=config,
            max_eod_restore_prob=max_eod_restore_prob,
            max_stop_prob=max_stop_prob,
            max_entry_adverse_risk=max_entry_adverse_risk,
            min_entry_mean_reversion_quality=min_entry_mean_reversion_quality,
        ),
    }


def _attach_excess_metrics(metrics: dict, baselines: dict[str, dict]) -> dict:
    out = dict(metrics)
    daily_uplift = float(out.get("daily_uplift_bps", 0.0) or 0.0)
    baseline_daily: dict[str, float] = {"no_trade": 0.0}
    for name, baseline in baselines.items():
        value = float(baseline.get("daily_uplift_bps", 0.0) or 0.0)
        baseline_daily[name] = value
        out[f"daily_uplift_bps_excess_vs_{_metric_safe_name(name)}"] = daily_uplift - value
    best_name, best_value = max(baseline_daily.items(), key=lambda item: item[1])
    out["baseline_daily_uplift_bps"] = float(best_value)
    out["excess_baseline_name"] = best_name
    out["daily_uplift_bps_excess"] = daily_uplift - float(best_value)
    return out


def _metric_safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _selection_metrics(
    selected: pd.DataFrame,
    *,
    config: FactorComboConfig,
    name: str,
    top_frac: float,
    threshold: float,
    max_eod_restore_prob: float,
    max_stop_prob: float,
    max_entry_adverse_risk: float,
    min_entry_mean_reversion_quality: float,
) -> dict:
    if selected.empty:
        return {
            "name": name,
            "top_frac": top_frac,
            "max_eod_restore_prob": max_eod_restore_prob,
            "max_stop_prob": max_stop_prob,
            "max_entry_adverse_risk": max_entry_adverse_risk,
            "min_entry_mean_reversion_quality": min_entry_mean_reversion_quality,
            "pred_threshold_bps": threshold,
            "n_legs": 0,
            "days": 0,
            "hit_rate": 0.0,
            "mean_gross_bps": 0.0,
            "mean_net_bps": 0.0,
            "daily_uplift_bps": 0.0,
            "annualized_uplift": 0.0,
            "eod_restore_rate": 0.0,
            "time_exit_rate": 0.0,
            "stop_rate": 0.0,
            "profit_rate": 0.0,
            "avg_pred_bps": 0.0,
            "avg_policy_score_bps": 0.0,
            "avg_eod_restore_prob": 0.0,
            "avg_stop_prob": 0.0,
            "avg_entry_adverse_risk": 0.0,
            "book_n_legs": 0,
            "book_days": 0,
            "book_daily_uplift_bps": 0.0,
            "baseline_daily_uplift_bps": 0.0,
            "excess_baseline_name": "not_computed",
            "daily_uplift_bps_excess": 0.0,
        }
    selected = selected.copy()
    if "weight" not in selected.columns:
        selected["weight"] = np.nan
    selected["_equal_weight"] = 1.0 / selected.groupby("trade_date")["symbol"].transform("count").clip(lower=1)
    selected["weighted_uplift"] = selected["_equal_weight"] * config.dot_fraction * selected["net_ret"]
    daily = selected.groupby("trade_date")["weighted_uplift"].sum()
    book = selected[selected["weight"].notna() & (selected["weight"] > 0)].copy()
    if not book.empty:
        book["book_uplift"] = book["weight"] * config.dot_fraction * book["net_ret"]
        book_daily = book.groupby("trade_date")["book_uplift"].sum()
    else:
        book_daily = pd.Series(dtype=float)
    metrics = {
        "name": name,
        "top_frac": float(top_frac),
        "max_eod_restore_prob": float(max_eod_restore_prob),
        "max_stop_prob": float(max_stop_prob),
        "max_entry_adverse_risk": float(max_entry_adverse_risk),
        "min_entry_mean_reversion_quality": float(min_entry_mean_reversion_quality),
        "pred_threshold_bps": float(threshold),
        "n_legs": int(len(selected)),
        "days": int(selected["trade_date"].nunique()),
        "hit_rate": float((selected["net_ret"] > 0).mean()),
        "mean_gross_bps": float(selected["gross_ret"].mean() * 10_000.0),
        "mean_net_bps": float(selected["net_ret"].mean() * 10_000.0),
        "daily_uplift_bps": float(daily.mean() * 10_000.0) if len(daily) else 0.0,
        "annualized_uplift": float((1.0 + daily.mean()) ** 244 - 1.0) if len(daily) else 0.0,
        "eod_restore_rate": float((selected["state"] == "closed_eod").mean()),
        "time_exit_rate": float((selected["state"] == "closed_time_exit").mean()),
        "stop_rate": float((selected["state"] == "closed_stop").mean()),
        "profit_rate": float((selected["state"] == "closed_profit").mean()),
        "avg_pred_bps": float(selected["pred_net_ret_bps"].mean()),
        "avg_policy_score_bps": _finite_mean(selected.get("pred_policy_score_bps", selected["pred_net_ret_bps"])),
        "avg_eod_restore_prob": _finite_mean(selected.get("pred_eod_restore_prob", pd.Series(0.0, index=selected.index))),
        "avg_stop_prob": _finite_mean(selected.get("pred_stop_prob", pd.Series(0.0, index=selected.index))),
        "avg_entry_adverse_risk": _finite_mean(selected.get("entry_mode_adverse_risk", pd.Series(0.0, index=selected.index))),
        "avg_entry_mean_reversion_quality": _finite_mean(selected.get("entry_mean_reversion_quality", pd.Series(0.0, index=selected.index))),
        "capacity_usage": _finite_mean(selected.get("entry_participation_rate", pd.Series(0.0, index=selected.index))),
        "avg_entry_capacity_ratio": _finite_mean(selected.get("entry_volume_capacity_ratio", pd.Series(0.0, index=selected.index))),
        "avg_exit_capacity_ratio": _finite_mean(selected.get("exit_volume_capacity_ratio", pd.Series(0.0, index=selected.index))),
    }
    metrics["book_n_legs"] = int(len(book))
    metrics["book_days"] = int(book["trade_date"].nunique()) if not book.empty else 0
    metrics["book_daily_uplift_bps"] = float(book_daily.mean() * 10_000.0) if len(book_daily) else 0.0
    metrics["baseline_daily_uplift_bps"] = 0.0
    metrics["excess_baseline_name"] = "not_computed"
    metrics["daily_uplift_bps_excess"] = 0.0
    return metrics


def _finite_mean(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else 0.0


__all__ = [
    "DEFAULT_FACTOR_COLUMNS",
    "DEFAULT_FSM_GRID",
    "FactorComboConfig",
    "TickFlowValidationSummary",
    "build_entry_relative_strength_features",
    "build_factor_combo_dataset",
    "build_dot_outcomes",
    "build_intraday_factors_from_minute_cache",
    "feature_importance_frame",
    "load_or_build_intraday_factors",
    "read_parquet_checked",
    "train_factor_combo_model",
    "validate_tickflow_minute_cache",
    "verdict_from_metrics",
]
