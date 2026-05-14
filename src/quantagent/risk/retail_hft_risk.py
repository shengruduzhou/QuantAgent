"""Structural risk module for retail (T+1) vs HFT (T+0 / co-located) asymmetry.

A retail account on a Chinese A-share suffers three structural disadvantages
against a co-located HFT desk:

1. Latency — institutions update quotes in microseconds; retail orders queue
   behind theirs and pay the wider effective spread.
2. T+1 — retail cannot exit on the same day, so any institutional dump after
   our buy crystallises into an overnight gap.
3. Information & quote-stuffing — institutions can detect retail flow imbalance
   and front-run / fake-out the move.

This module produces a per-symbol penalty (0-1) that downstream code applies as
extra slippage_bps and a haircut to the alpha confidence. Inputs are widely
available: daily amount, turnover ratio, top-5 buyer/seller concentration,
limit-up/down history, sudden volume z-scores, and short-term reversal patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetailHFTRiskConfig:
    institutional_volume_zscore_warning: float = 2.0
    turnover_anomaly_threshold: float = 0.15
    short_reversal_window_days: int = 5
    short_reversal_penalty_strength: float = 0.35
    limit_volatility_penalty_strength: float = 0.30
    amount_low_floor: float = 1.0e8
    amount_high_ceiling: float = 5.0e10
    base_penalty: float = 0.20
    t_plus_one_overnight_gap_penalty: float = 0.10
    block_trade_share_warning: float = 0.30
    open_close_volatility_penalty: float = 0.10
    max_penalty: float = 0.85


@dataclass(frozen=True)
class RetailHFTRiskReport:
    symbol: str
    penalty_score: float
    extra_slippage_bps: float
    confidence_haircut: float
    institutional_dump_risk: float
    quote_stuffing_risk: float
    overnight_gap_risk: float
    limit_volatility_risk: float
    short_reversal_risk: float
    block_trade_risk: float
    notes: tuple[str, ...]


def score_retail_hft_risk(
    market_panel: pd.DataFrame,
    market_state: pd.DataFrame,
    config: RetailHFTRiskConfig | None = None,
) -> list[RetailHFTRiskReport]:
    """Score retail/HFT structural risk per symbol using market panel + state."""
    config = config or RetailHFTRiskConfig()
    if market_panel is None or market_panel.empty:
        return _from_state_only(market_state, config)
    panel = market_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel.get("trade_date"), errors="coerce")
    panel = panel.sort_values(["symbol", "trade_date"])
    state_by_symbol = _state_lookup(market_state)
    reports: list[RetailHFTRiskReport] = []
    for symbol, group in panel.groupby("symbol"):
        state_row = state_by_symbol.get(str(symbol), {})
        report = _score_one(str(symbol), group, state_row, config)
        reports.append(report)
    return reports


def apply_retail_hft_penalty(
    alphas: dict,
    reports: list[RetailHFTRiskReport],
) -> dict:
    """Mutate (return new dict of) MultiHorizonAlpha-like objects with retail haircut."""
    if not reports or not alphas:
        return alphas
    haircut_by_symbol = {report.symbol: report.confidence_haircut for report in reports}
    output = {}
    for symbol, alpha in alphas.items():
        haircut = haircut_by_symbol.get(symbol)
        if haircut is None:
            output[symbol] = alpha
            continue
        if hasattr(alpha, "confidence") and hasattr(alpha, "downside_risk"):
            output[symbol] = alpha.__class__(
                **{
                    field: getattr(alpha, field)
                    for field in alpha.__dataclass_fields__
                    if field not in {"confidence", "downside_risk", "risk_penalty"}
                },
                confidence=float(np.clip(alpha.confidence * (1.0 - haircut), 0.05, 0.95)),
                downside_risk=float(np.clip(alpha.downside_risk + haircut * 0.10, 0.0, 1.0)),
                risk_penalty=float(np.clip(alpha.risk_penalty + haircut * 0.25, 0.0, 1.0)),
            )
        else:
            output[symbol] = alpha
    return output


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------


def _score_one(
    symbol: str,
    group: pd.DataFrame,
    state_row: dict,
    config: RetailHFTRiskConfig,
) -> RetailHFTRiskReport:
    notes: list[str] = []
    institutional_dump = _institutional_dump_risk(group, config, notes)
    quote_stuffing = _quote_stuffing_risk(group, state_row, config, notes)
    overnight_gap = _overnight_gap_risk(group, config, notes)
    limit_volatility = _limit_volatility_risk(group, state_row, config, notes)
    short_reversal = _short_reversal_risk(group, config, notes)
    block_trade = _block_trade_risk(state_row, config, notes)

    raw_penalty = (
        config.base_penalty
        + 0.25 * institutional_dump
        + 0.20 * quote_stuffing
        + config.t_plus_one_overnight_gap_penalty * overnight_gap
        + config.limit_volatility_penalty_strength * limit_volatility
        + config.short_reversal_penalty_strength * short_reversal
        + 0.10 * block_trade
        + config.open_close_volatility_penalty * _open_close_volatility(group)
    )
    penalty = float(np.clip(raw_penalty, 0.0, config.max_penalty))
    extra_slippage_bps = float(np.clip(5.0 + 60.0 * penalty, 5.0, 80.0))
    confidence_haircut = float(np.clip(penalty * 0.50, 0.0, 0.40))
    return RetailHFTRiskReport(
        symbol=symbol,
        penalty_score=penalty,
        extra_slippage_bps=extra_slippage_bps,
        confidence_haircut=confidence_haircut,
        institutional_dump_risk=institutional_dump,
        quote_stuffing_risk=quote_stuffing,
        overnight_gap_risk=overnight_gap,
        limit_volatility_risk=limit_volatility,
        short_reversal_risk=short_reversal,
        block_trade_risk=block_trade,
        notes=tuple(notes),
    )


def _institutional_dump_risk(group: pd.DataFrame, config: RetailHFTRiskConfig, notes: list) -> float:
    if "amount" not in group.columns or len(group) < 20:
        return 0.20
    amount = pd.to_numeric(group["amount"], errors="coerce").dropna()
    if len(amount) < 20:
        return 0.20
    mean = amount.tail(60).mean() if len(amount) >= 60 else amount.mean()
    std = amount.tail(60).std(ddof=0) if len(amount) >= 60 else amount.std(ddof=0)
    if std <= 0:
        return 0.20
    z = (amount.iloc[-1] - mean) / std
    if z > config.institutional_volume_zscore_warning:
        notes.append(f"institutional_volume_z={z:.2f}")
        return float(np.clip(z / 4.0, 0.0, 1.0))
    return float(np.clip(max(0.0, z) / 4.0, 0.0, 1.0))


def _quote_stuffing_risk(group: pd.DataFrame, state_row: dict, config: RetailHFTRiskConfig, notes: list) -> float:
    turnover = pd.to_numeric(group.get("turnover_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    if turnover.empty and "turnover_ratio" in state_row:
        turnover = pd.Series([float(state_row.get("turnover_ratio", 0.0))])
    if turnover.empty:
        return 0.30
    ratio = float(turnover.iloc[-1])
    if ratio > config.turnover_anomaly_threshold:
        notes.append(f"high_turnover_ratio={ratio:.2f}")
        return float(np.clip(ratio * 2.5, 0.0, 1.0))
    return float(np.clip(ratio * 2.0, 0.0, 1.0))


def _overnight_gap_risk(group: pd.DataFrame, config: RetailHFTRiskConfig, notes: list) -> float:
    if not {"open", "close"}.issubset(group.columns) or len(group) < 5:
        return 0.50
    closes = pd.to_numeric(group["close"], errors="coerce")
    opens = pd.to_numeric(group["open"], errors="coerce")
    gaps = (opens / closes.shift(1) - 1.0).dropna()
    if gaps.empty:
        return 0.40
    abs_mean = float(gaps.abs().tail(20).mean())
    if abs_mean > 0.012:
        notes.append(f"overnight_gap_avg={abs_mean:.3f}")
    return float(np.clip(abs_mean * 25.0, 0.0, 1.0))


def _limit_volatility_risk(group: pd.DataFrame, state_row: dict, config: RetailHFTRiskConfig, notes: list) -> float:
    limit_hits = 0
    if "is_limit_up" in group.columns:
        limit_hits += int(group["is_limit_up"].astype(bool).tail(40).sum())
    if "is_limit_down" in group.columns:
        limit_hits += int(group["is_limit_down"].astype(bool).tail(40).sum())
    if limit_hits > 0:
        notes.append(f"limit_hits_40d={limit_hits}")
    state_penalty = 0.0
    if bool(state_row.get("is_limit_up", False)) or bool(state_row.get("is_limit_down", False)):
        state_penalty = 0.50
    return float(np.clip(limit_hits / 8.0 + state_penalty, 0.0, 1.0))


def _short_reversal_risk(group: pd.DataFrame, config: RetailHFTRiskConfig, notes: list) -> float:
    if "close" not in group.columns or len(group) < config.short_reversal_window_days + 5:
        return 0.30
    closes = pd.to_numeric(group["close"], errors="coerce").dropna()
    if len(closes) < config.short_reversal_window_days + 5:
        return 0.30
    short_return = float(closes.iloc[-1] / closes.iloc[-config.short_reversal_window_days - 1] - 1.0)
    prior_return = float(closes.iloc[-config.short_reversal_window_days - 1] / closes.iloc[-config.short_reversal_window_days - 5] - 1.0)
    if short_return * prior_return < 0 and abs(short_return) > 0.04:
        notes.append(f"reversal_pattern={short_return:+.2f}_vs_{prior_return:+.2f}")
        return float(np.clip(abs(short_return) * 5.0, 0.0, 1.0))
    return float(np.clip(abs(short_return) * 2.0, 0.0, 1.0))


def _block_trade_risk(state_row: dict, config: RetailHFTRiskConfig, notes: list) -> float:
    share = state_row.get("block_trade_share") if state_row else None
    if share is None:
        return 0.20
    try:
        share = float(share)
    except (TypeError, ValueError):
        return 0.20
    if share > config.block_trade_share_warning:
        notes.append(f"block_trade_share={share:.2f}")
        return float(np.clip(share * 2.0, 0.0, 1.0))
    return float(np.clip(share, 0.0, 1.0))


def _open_close_volatility(group: pd.DataFrame) -> float:
    if not {"open", "close", "high", "low"}.issubset(group.columns) or len(group) < 5:
        return 0.30
    last = group.tail(20)
    intraday = (last["high"] - last["low"]) / last["close"].abs().replace(0.0, np.nan)
    return float(np.clip(intraday.mean(), 0.0, 1.0))


def _state_lookup(market_state: pd.DataFrame) -> dict[str, dict]:
    if market_state is None or market_state.empty or "symbol" not in market_state.columns:
        return {}
    return {str(row["symbol"]): row.to_dict() for _, row in market_state.iterrows()}


def _from_state_only(market_state: pd.DataFrame, config: RetailHFTRiskConfig) -> list[RetailHFTRiskReport]:
    if market_state is None or market_state.empty or "symbol" not in market_state.columns:
        return []
    reports = []
    for _, row in market_state.iterrows():
        notes = []
        penalty = config.base_penalty
        if bool(row.get("is_limit_up", False)) or bool(row.get("is_limit_down", False)):
            penalty += 0.30
            notes.append("limit_state_active")
        block = row.get("block_trade_share")
        if block is not None:
            try:
                share = float(block)
                if share > config.block_trade_share_warning:
                    penalty += 0.10 * share
                    notes.append(f"block_trade_share={share:.2f}")
            except (TypeError, ValueError):
                pass
        penalty = float(np.clip(penalty, 0.0, config.max_penalty))
        reports.append(
            RetailHFTRiskReport(
                symbol=str(row["symbol"]),
                penalty_score=penalty,
                extra_slippage_bps=float(np.clip(5.0 + 60.0 * penalty, 5.0, 80.0)),
                confidence_haircut=float(np.clip(penalty * 0.50, 0.0, 0.40)),
                institutional_dump_risk=0.30,
                quote_stuffing_risk=0.30,
                overnight_gap_risk=0.40,
                limit_volatility_risk=0.30 if "limit_state_active" in notes else 0.10,
                short_reversal_risk=0.20,
                block_trade_risk=0.20,
                notes=tuple(notes),
            )
        )
    return reports
