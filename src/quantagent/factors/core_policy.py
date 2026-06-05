"""Core factor policy for A-share regime experts.

This module compresses a wide alpha/CICC/intraday panel into a small,
auditable feature set.  It keeps the model input below 30 columns and exposes
explicit score columns for policy/news, fundamentals, sector resonance,
dip-buying flow, and old-dealer risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "core_policy_score",
    "core_sentiment_score",
    "fundamental_quality_score",
    "cicc_stock_selection_score",
    "cicc_sector_selection_score",
    "cicc_aggressive_momentum_score",
    "cicc_defensive_quality_score",
    "cicc_liquidity_defense_score",
    "sector_resonance_score",
    "dip_buy_flow_score",
    "old_dealer_risk_score",
    "trend_strength_score",
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "amount_mean_20d",
    "volume_mean_20d",
    "intraday_return",
    "first30_return",
    "last30_return",
    "vwap_deviation",
    "intraday_range_pos",
    "net_buy_pressure",
    "volume_concentration",
    "spike_minutes",
    "close30_volume_share",
    "flow_north_total",
    "agent_stock_score",
    "agent_conviction_score",
)


CORE_FACTOR_PRIOR_WEIGHTS: dict[str, float] = {
    "core_policy_score": 0.16,
    "core_sentiment_score": 0.12,
    "fundamental_quality_score": 0.10,
    "cicc_stock_selection_score": 0.10,
    "cicc_sector_selection_score": 0.08,
    "sector_resonance_score": 0.08,
    "dip_buy_flow_score": 0.08,
    "trend_strength_score": 0.08,
    "cicc_aggressive_momentum_score": 0.06,
    "cicc_defensive_quality_score": 0.05,
    "cicc_liquidity_defense_score": 0.05,
    "agent_stock_score": 0.10,
    "agent_conviction_score": 0.06,
    "old_dealer_risk_score": -0.12,
}


@dataclass(frozen=True)
class CoreFactorSummary:
    feature_columns: tuple[str, ...]
    prior_weights: dict[str, float]
    rows: int
    old_dealer_block_rate: float
    evidence_policy_available: bool
    evidence_sentiment_available: bool
    fundamentals_available: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "prior_weights": dict(self.prior_weights),
            "rows": int(self.rows),
            "old_dealer_block_rate": float(self.old_dealer_block_rate),
            "evidence_policy_available": bool(self.evidence_policy_available),
            "evidence_sentiment_available": bool(self.evidence_sentiment_available),
            "fundamentals_available": bool(self.fundamentals_available),
        }


def build_core_factor_frame(
    dataset: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    evidence_scores: pd.DataFrame | None = None,
    agent_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, CoreFactorSummary]:
    """Return keys, labels, and the core <=30 feature set.

    ``dataset`` must already be point-in-time safe.  Optional fundamentals and
    evidence are joined by ``available_at`` semantics before scoring.
    """
    if dataset is None or dataset.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *CORE_FEATURE_COLUMNS]), CoreFactorSummary(
            feature_columns=CORE_FEATURE_COLUMNS,
            prior_weights=CORE_FACTOR_PRIOR_WEIGHTS,
            rows=0,
            old_dealer_block_rate=0.0,
            evidence_policy_available=False,
            evidence_sentiment_available=False,
            fundamentals_available=False,
        )
    data = dataset.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str)
    data = _merge_fundamentals(data, fundamentals)
    data = _merge_evidence_scores(data, evidence_scores)
    data = _merge_agent_scores(data, agent_scores)
    data = _attach_sector(data, sector_map)

    out = data[["symbol", "trade_date"]].copy()
    for col in _label_columns(data.columns):
        out[col] = data[col]

    out["trend_strength_score"] = _mean_existing(
        data,
        ("return_1d", "momentum_5d", "momentum_20d", "intraday_return", "vwap_deviation", "last30_return"),
        rank=True,
    )
    out["sector_resonance_score"] = _sector_resonance(data, out["trend_strength_score"])
    out["fundamental_quality_score"] = _fundamental_quality(data)
    out["core_policy_score"] = _policy_score(data)
    out["core_sentiment_score"] = _sentiment_score(data)
    out["dip_buy_flow_score"] = _dip_buy_flow_score(data)
    out["old_dealer_risk_score"] = _old_dealer_risk_score(data, out)
    out["old_dealer_block"] = (
        (out["old_dealer_risk_score"] >= 0.70)
        | ((out["trend_strength_score"] <= -0.20) & (out["sector_resonance_score"] <= -0.15))
    ).astype("int8")

    for col in CORE_FEATURE_COLUMNS:
        if col in out.columns:
            continue
        if col in data.columns:
            out[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            out[col] = 0.0

    for col in CORE_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    keep = ["symbol", "trade_date", *_label_columns(out.columns), "old_dealer_block", *CORE_FEATURE_COLUMNS]
    out = out[[c for c in keep if c in out.columns]].reset_index(drop=True)
    summary = CoreFactorSummary(
        feature_columns=CORE_FEATURE_COLUMNS,
        prior_weights=CORE_FACTOR_PRIOR_WEIGHTS,
        rows=len(out),
        old_dealer_block_rate=float(out["old_dealer_block"].mean()) if len(out) else 0.0,
        evidence_policy_available="evidence_policy_score" in data.columns and data["evidence_policy_score"].notna().any(),
        evidence_sentiment_available="evidence_sentiment_score" in data.columns and data["evidence_sentiment_score"].notna().any(),
        fundamentals_available=any(c in data.columns for c in _FUNDAMENTAL_COLUMNS),
    )
    return out, summary


def aggregate_evidence_scores(evidence: pd.DataFrame | None) -> pd.DataFrame:
    """Aggregate PIT evidence into date/symbol score rows.

    If evidence lacks explicit policy/sentiment scores, the function returns
    neutral scores instead of inventing values.
    """
    if evidence is None or evidence.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "evidence_policy_score", "evidence_sentiment_score"])
    ev = evidence.copy()
    date_col = "available_at" if "available_at" in ev.columns else "published_at"
    ev["trade_date"] = pd.to_datetime(ev[date_col], errors="coerce").dt.normalize()
    symbols = _extract_evidence_symbols(ev)
    if symbols.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "evidence_policy_score", "evidence_sentiment_score"])
    ev["symbol"] = symbols
    policy_col = next((c for c in ("policy_direction_score", "policy_score") if c in ev.columns), None)
    sentiment_col = next((c for c in ("sentiment_score", "news_score") if c in ev.columns), None)
    ev["evidence_policy_score"] = pd.to_numeric(ev[policy_col], errors="coerce") if policy_col else 0.0
    ev["evidence_sentiment_score"] = pd.to_numeric(ev[sentiment_col], errors="coerce") if sentiment_col else 0.0
    conf = pd.to_numeric(ev["confidence"], errors="coerce").clip(0.0, 1.0) if "confidence" in ev.columns else 1.0
    ev["evidence_policy_score"] = ev["evidence_policy_score"].fillna(0.0) * conf
    ev["evidence_sentiment_score"] = ev["evidence_sentiment_score"].fillna(0.0) * conf
    out = (
        ev.dropna(subset=["trade_date", "symbol"])
        .groupby(["trade_date", "symbol"], as_index=False)[["evidence_policy_score", "evidence_sentiment_score"]]
        .mean()
    )
    out["symbol"] = out["symbol"].astype(str)
    return out


def core_feature_columns(available_columns: Iterable[str]) -> list[str]:
    """Return available core feature columns, capped at 30."""
    available = set(available_columns)
    return [c for c in CORE_FEATURE_COLUMNS if c in available][:30]


_FUNDAMENTAL_COLUMNS = (
    "roe", "revenue_yoy", "net_income_yoy", "gross_margin", "operating_cash_to_revenue",
    "debt_to_asset_ratio",
)


def _label_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c.startswith("forward_return_") or c.startswith("label_end_")]


def _merge_fundamentals(data: pd.DataFrame, fundamentals: pd.DataFrame | None) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return data
    keep = ["symbol", "available_at", *[c for c in _FUNDAMENTAL_COLUMNS if c in fundamentals.columns]]
    if len(keep) <= 2:
        return data
    f = fundamentals[keep].copy()
    f["symbol"] = f["symbol"].astype(str)
    f["available_at"] = pd.to_datetime(f["available_at"], errors="coerce")
    f = f.dropna(subset=["available_at"]).sort_values(["symbol", "available_at"])
    fund_by_symbol = {sym: g.sort_values("available_at") for sym, g in f.groupby("symbol", sort=False)}
    left = data.copy()
    left["_qa_original_order"] = np.arange(len(left))
    pieces = []
    for sym, g in left.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False):
        fg = fund_by_symbol.get(sym)
        if fg is None or fg.empty:
            piece = g.copy()
            for col in keep:
                if col not in {"symbol", "available_at"} and col not in piece.columns:
                    piece[col] = np.nan
            pieces.append(piece)
            continue
        piece = pd.merge_asof(
            g.sort_values("trade_date"),
            fg.drop(columns=["symbol"]).sort_values("available_at"),
            left_on="trade_date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        if "available_at_y" in piece.columns:
            piece = piece.drop(columns=["available_at_y"])
        if "available_at_x" in piece.columns:
            piece = piece.rename(columns={"available_at_x": "available_at"})
        pieces.append(piece)
    merged = pd.concat(pieces, ignore_index=True) if pieces else left
    merged = merged.sort_values("_qa_original_order").drop(columns=["_qa_original_order"]).reset_index(drop=True)
    return merged


def _merge_evidence_scores(data: pd.DataFrame, evidence_scores: pd.DataFrame | None) -> pd.DataFrame:
    if evidence_scores is None or evidence_scores.empty:
        return data
    ev = evidence_scores.copy()
    ev["trade_date"] = pd.to_datetime(ev["trade_date"], errors="coerce")
    ev["symbol"] = ev["symbol"].astype(str)
    return data.merge(ev, on=["trade_date", "symbol"], how="left")


def _merge_agent_scores(data: pd.DataFrame, agent_scores: pd.DataFrame | None) -> pd.DataFrame:
    if agent_scores is None or agent_scores.empty:
        return data
    keep = ["trade_date", "symbol"] + [
        c for c in agent_scores.columns
        if c.startswith("agent_") or c.endswith("_agent_score")
    ]
    if len(keep) <= 2:
        return data
    agent = agent_scores[keep].copy()
    agent["trade_date"] = pd.to_datetime(agent["trade_date"], errors="coerce")
    agent["symbol"] = agent["symbol"].astype(str)
    return data.merge(agent, on=["trade_date", "symbol"], how="left")


def _attach_sector(data: pd.DataFrame, sector_map: pd.DataFrame | None) -> pd.DataFrame:
    if sector_map is None or sector_map.empty or "sector_level_1" not in sector_map.columns:
        data["_sector_level_1"] = "unknown"
        return data
    sm = sector_map[["symbol", "sector_level_1"]].drop_duplicates("symbol").copy()
    sm["symbol"] = sm["symbol"].astype(str)
    out = data.merge(sm, on="symbol", how="left")
    out["_sector_level_1"] = out["sector_level_1"].fillna("unknown").astype(str)
    return out


def _rank_centered(data: pd.DataFrame, col: str, *, ascending: bool = True) -> pd.Series:
    if col not in data.columns:
        return pd.Series(0.0, index=data.index)
    values = pd.to_numeric(data[col], errors="coerce")
    rank = values.groupby(data["trade_date"]).rank(pct=True, ascending=ascending) - 0.5
    return rank.fillna(0.0)


def _mean_existing(data: pd.DataFrame, cols: tuple[str, ...], *, rank: bool) -> pd.Series:
    parts = []
    for col in cols:
        if col not in data.columns:
            continue
        parts.append(_rank_centered(data, col) if rank else pd.to_numeric(data[col], errors="coerce"))
    if not parts:
        return pd.Series(0.0, index=data.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _sector_resonance(data: pd.DataFrame, trend_score: pd.Series) -> pd.Series:
    base = trend_score.copy()
    if "cicc_sector_selection_score" in data.columns:
        base = 0.60 * base + 0.40 * pd.to_numeric(data["cicc_sector_selection_score"], errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({
        "trade_date": data["trade_date"],
        "sector": data.get("_sector_level_1", "unknown"),
        "score": base,
    })
    sector_mean = tmp.groupby(["trade_date", "sector"])["score"].transform("mean")
    return sector_mean.groupby(data["trade_date"]).rank(pct=True).sub(0.5).fillna(0.0)


def _fundamental_quality(data: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("roe", "revenue_yoy", "net_income_yoy", "gross_margin", "operating_cash_to_revenue"):
        if col in data.columns:
            parts.append(_rank_centered(data, col))
    if "debt_to_asset_ratio" in data.columns:
        parts.append(_rank_centered(data, "debt_to_asset_ratio", ascending=False))
    if not parts:
        return pd.Series(0.0, index=data.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _policy_score(data: pd.DataFrame) -> pd.Series:
    parts = []
    if "evidence_policy_score" in data.columns:
        parts.append(pd.to_numeric(data["evidence_policy_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0))
    for col in ("flow_north_total", "flow_margin_sh", "idx_csi300_ret5"):
        if col in data.columns:
            parts.append(_date_level_rolling_score(data, col))
    if not parts:
        return pd.Series(0.0, index=data.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _sentiment_score(data: pd.DataFrame) -> pd.Series:
    if "evidence_sentiment_score" in data.columns:
        return pd.to_numeric(data["evidence_sentiment_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    return pd.Series(0.0, index=data.index)


def _dip_buy_flow_score(data: pd.DataFrame) -> pd.Series:
    parts = []
    if "net_buy_pressure" in data.columns:
        parts.append(_rank_centered(data, "net_buy_pressure"))
    if "vwap_deviation" in data.columns:
        # Buying near/below VWAP is a low-absorption setup; far above VWAP is chasing.
        parts.append(_rank_centered(data.assign(_neg_vwap=-pd.to_numeric(data["vwap_deviation"], errors="coerce")), "_neg_vwap"))
    if "intraday_range_pos" in data.columns:
        near_low = 1.0 - pd.to_numeric(data["intraday_range_pos"], errors="coerce")
        parts.append((near_low.groupby(data["trade_date"]).rank(pct=True) - 0.5).fillna(0.0))
    if "last30_return" in data.columns:
        parts.append(_rank_centered(data, "last30_return"))
    if not parts:
        return pd.Series(0.0, index=data.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _old_dealer_risk_score(data: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    trend_bad = (0.5 - scores["trend_strength_score"].clip(-0.5, 0.5)).clip(0.0, 1.0)
    sector_bad = (0.5 - scores["sector_resonance_score"].clip(-0.5, 0.5)).clip(0.0, 1.0)
    liq_bad = (0.5 - _rank_centered(data, "amount_mean_20d").clip(-0.5, 0.5)).clip(0.0, 1.0)
    vol_abnormal = _rank_centered(data, "volume_concentration").add(0.5).clip(0.0, 1.0)
    spike_abnormal = _rank_centered(data, "spike_minutes").add(0.5).clip(0.0, 1.0)
    return (0.35 * trend_bad + 0.35 * sector_bad + 0.15 * liq_bad + 0.10 * vol_abnormal + 0.05 * spike_abnormal).fillna(0.0)


def _extract_evidence_symbols(ev: pd.DataFrame) -> pd.Series:
    values = []
    for _, row in ev.iterrows():
        symbol = row.get("symbol")
        if pd.notna(symbol) and str(symbol).strip():
            values.append(str(symbol).strip())
            continue
        affected = row.get("affected_symbols")
        if isinstance(affected, str) and affected.strip():
            values.append(affected.split(",")[0].strip())
        elif isinstance(affected, (list, tuple)) and affected:
            values.append(str(affected[0]))
        else:
            values.append(np.nan)
    return pd.Series(values, index=ev.index, name="symbol")


def _date_level_rolling_score(data: pd.DataFrame, col: str, window: int = 120) -> pd.Series:
    """PIT date-level percentile score for market-wide policy/flow proxies."""
    values = pd.to_numeric(data[col], errors="coerce")
    date_value = values.groupby(data["trade_date"]).mean().sort_index()
    if date_value.empty:
        return pd.Series(0.0, index=data.index)

    def _last_percentile(x: np.ndarray) -> float:
        clean = x[np.isfinite(x)]
        if clean.size == 0:
            return 0.0
        return float((clean <= clean[-1]).mean() - 0.5)

    scored = date_value.rolling(window, min_periods=20).apply(_last_percentile, raw=True).fillna(0.0)
    return data["trade_date"].map(scored).fillna(0.0)


__all__ = [
    "CORE_FACTOR_PRIOR_WEIGHTS",
    "CORE_FEATURE_COLUMNS",
    "CoreFactorSummary",
    "aggregate_evidence_scores",
    "build_core_factor_frame",
    "core_feature_columns",
]
