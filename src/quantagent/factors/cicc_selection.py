"""CICC-style stock and sector selection scores.

This layer turns many CICC-style daily/high-frequency factor columns into a
small set of model-ready stock/sector selection features. It is not a live
order generator; it produces PIT same-date cross-sectional scores that can be
fed into deep models, decision-chain pool ranking, or index-enhancement RL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


AGGRESSIVE_PATTERNS = ("mmt", "momentum", "breakout", "close_strength", "money_flow")
DEFENSIVE_PATTERNS = ("doc", "shape", "range_position")
LIQUIDITY_PATTERNS = ("liq", "amihud", "amount_mean", "volume_z")
RISK_PATTERNS = ("vol_", "crowd", "tail", "downside")


def compute_cicc_selection_scores(
    factor_frame: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute stock/sector selection scores from CICC-style factor columns.

    Accepts either a wide frame with ``cicc_*`` columns or long-form
    ``factor_name/factor_value`` rows. Scores are per-date cross-sectional
    percentiles centred at 0, so they remain comparable through time.
    """
    wide = _to_wide(factor_frame)
    if wide.empty:
        return pd.DataFrame(columns=[
            "trade_date", "symbol", "cicc_stock_selection_score",
            "cicc_aggressive_momentum_score", "cicc_defensive_quality_score",
            "cicc_liquidity_defense_score", "cicc_sector_selection_score",
        ])
    factor_cols = [c for c in wide.columns if c.startswith("cicc_")]
    if not factor_cols:
        raise ValueError("factor_frame has no cicc_* factor columns")
    out = wide[["trade_date", "symbol"]].copy()
    aggressive = _bucket_score(wide, factor_cols, AGGRESSIVE_PATTERNS, direction=1)
    defensive = _bucket_score(wide, factor_cols, DEFENSIVE_PATTERNS, direction=1)
    liquidity = _bucket_score(wide, factor_cols, LIQUIDITY_PATTERNS, direction=-1)
    risk = _bucket_score(wide, factor_cols, RISK_PATTERNS, direction=-1)
    out["cicc_aggressive_momentum_score"] = aggressive
    out["cicc_defensive_quality_score"] = defensive
    out["cicc_liquidity_defense_score"] = liquidity
    out["cicc_stock_selection_score"] = (
        0.40 * aggressive + 0.25 * defensive + 0.25 * liquidity + 0.10 * risk
    )
    out["cicc_sector_selection_score"] = 0.0
    if sector_map is not None and not sector_map.empty:
        out = _attach_sector_score(out, sector_map)
    return out.replace([np.inf, -np.inf], np.nan)


def _to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str)
    if {"factor_name", "factor_value"}.issubset(data.columns):
        wide = data.pivot_table(
            index=["trade_date", "symbol"],
            columns="factor_name",
            values="factor_value",
            aggfunc="last",
        ).reset_index()
        wide.columns = [str(c) for c in wide.columns]
        return wide
    return data


def _bucket_score(
    frame: pd.DataFrame,
    factor_cols: list[str],
    patterns: tuple[str, ...],
    *,
    direction: int,
) -> pd.Series:
    cols = [c for c in factor_cols if any(p in c for p in patterns)]
    if not cols:
        return pd.Series(0.0, index=frame.index)
    ranked = []
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        rank = values.groupby(frame["trade_date"]).rank(pct=True) - 0.5
        ranked.append(rank * float(direction))
    return pd.concat(ranked, axis=1).mean(axis=1).fillna(0.0)


def _attach_sector_score(scores: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    sm = sector_map.copy()
    if "sector_level_1" not in sm.columns:
        return scores
    sm["symbol"] = sm["symbol"].astype(str)
    merged = scores.merge(
        sm[["symbol", "sector_level_1"]].drop_duplicates("symbol"),
        on="symbol",
        how="left",
    )
    sector_mean = (
        merged.groupby(["trade_date", "sector_level_1"])["cicc_stock_selection_score"]
        .mean()
        .rename("_sector_mean")
        .reset_index()
    )
    merged = merged.merge(sector_mean, on=["trade_date", "sector_level_1"], how="left")
    merged["cicc_sector_selection_score"] = (
        merged["_sector_mean"].groupby(merged["trade_date"]).rank(pct=True) - 0.5
    ).fillna(0.0)
    return merged.drop(columns=["sector_level_1", "_sector_mean"])


__all__ = ["compute_cicc_selection_scores"]
