"""Probabilistic aggregation for public state-team evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal, Sequence

import numpy as np
import pandas as pd


STATE_TEAM_HOLDER_KEYWORDS: tuple[str, ...] = (
    "中央汇金", "汇金", "中国证券金融", "证金", "国新投资", "国新",
    "全国社保", "社保基金", "中国投资有限责任", "中投", "CIC",
)

EVIDENCE_RELIABILITY: dict[str, float] = {
    "official_holder_filing": 0.90,
    "top10_holder_appearance": 0.85,
    "etf_share_creation": 0.75,
    "etf_concentrated_inflow": 0.65,
    "post_crash_index_buying": 0.60,
    "block_trade_match": 0.55,
    "index_futures_basis": 0.45,
    "other": 0.30,
}


@dataclass(frozen=True)
class StateTeamPosteriorConfig:
    prior_probability: float = 0.05
    min_independent_evidence_types: int = 2
    max_event_age_days: int = 90
    half_life_days: float = 20.0
    etf_flow_unit: Literal["cny_bn", "cny_mn", "cny", "auto"] = "auto"
    concentrated_etf_inflow_threshold_cny_bn: float = 5.0  # 50亿元


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, probability))
    return log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def normalise_etf_flows(
    frame: pd.DataFrame,
    *,
    unit: Literal["cny_bn", "cny_mn", "cny", "auto"] = "auto",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    excluded = {"trade_date", "available_at", "source"}
    numeric_cols = [column for column in out.columns if column not in excluded]
    values = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    sample = values.stack().dropna().abs()
    resolved = unit
    if unit == "auto":
        median = float(sample.median()) if not sample.empty else 0.0
        resolved = "cny" if median >= 1e8 else ("cny_mn" if median >= 1e3 else "cny_bn")
    if resolved == "cny":
        values /= 1e9
    elif resolved == "cny_mn":
        values /= 1e3
    elif resolved != "cny_bn":
        raise ValueError(f"unsupported ETF flow unit: {resolved}")
    out[numeric_cols] = values
    out.attrs.update(flow_unit="cny_bn", input_flow_unit=resolved)
    return out


def holder_filings_to_events(
    filings: pd.DataFrame,
    *,
    allow_estimated_availability: bool = False,
    holder_keywords: Sequence[str] = STATE_TEAM_HOLDER_KEYWORDS,
) -> pd.DataFrame:
    """Convert recognized public-institution holder filings into evidence.

    Ordinary top-ten holders are discarded.  A quarter-end plus 45-business-day
    timestamp is disabled unless explicitly requested and remains labelled as
    estimated rather than reported.
    """
    if filings is None or filings.empty:
        return pd.DataFrame()
    required = {"symbol", "holder_name", "share_pct"}
    missing = required - set(filings.columns)
    if missing:
        raise ValueError(f"holder filings missing columns: {sorted(missing)}")
    out = filings.copy()
    names = out["holder_name"].fillna("").astype(str)
    pattern = tuple(str(keyword) for keyword in holder_keywords if str(keyword))
    out = out[names.map(lambda name: any(keyword.lower() in name.lower() for keyword in pattern))].copy()
    if out.empty:
        return pd.DataFrame()

    if "announcement_date" in out.columns:
        available = pd.to_datetime(out["announcement_date"], errors="coerce")
        quality = "reported"
    elif "available_at" in out.columns:
        available = pd.to_datetime(out["available_at"], errors="coerce")
        quality = "reported"
    elif allow_estimated_availability and "report_period" in out.columns:
        available = pd.to_datetime(out["report_period"], errors="coerce") + pd.tseries.offsets.BDay(45)
        quality = "estimated_45bd"
    else:
        raise ValueError(
            "holder filings require announcement_date/available_at; "
            "quarter-end plus 45 business days is disabled by default"
        )

    out["available_at"] = available
    out = out[out["available_at"].notna()].copy()
    if out.empty:
        return pd.DataFrame()
    share = pd.to_numeric(out["share_pct"], errors="coerce").fillna(0.0).clip(0, 5)
    out["trade_date"] = out["available_at"].dt.normalize()
    out["evidence_type"] = "official_holder_filing"
    out["evidence_label"] = "inferred"
    out["evidence_strength"] = (0.45 + 0.10 * share).clip(0, 0.95)
    out["scope"] = "symbol"
    out["scope_value"] = out["symbol"].astype(str)
    out["source_independence_key"] = "holder_filing"
    out["availability_quality"] = quality
    out["description"] = out["holder_name"].astype(str) + " " + share.astype(str) + "%"
    return out[
        ["trade_date", "available_at", "evidence_type", "evidence_label",
         "evidence_strength", "scope", "scope_value", "source_independence_key",
         "availability_quality", "description"]
    ]


def compute_state_team_posterior(
    events: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    config: StateTeamPosteriorConfig | None = None,
) -> pd.DataFrame:
    cfg = config or StateTeamPosteriorConfig()
    empty_columns = [
        "as_of", "scope", "scope_value", "posterior_probability",
        "independent_evidence_types", "feature_usable", "evidence_label",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=empty_columns)
    work = events.copy()
    required = {"available_at", "evidence_type", "evidence_strength", "scope", "scope_value"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"state-team events missing columns: {sorted(missing)}")
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    work = work[work["available_at"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=empty_columns)
    cutoff = pd.Timestamp(as_of) if as_of is not None else work["available_at"].max()
    work = work[work["available_at"] <= cutoff].copy()
    age_days = (cutoff - work["available_at"]).dt.total_seconds() / 86400.0
    work = work[(age_days >= 0) & (age_days <= cfg.max_event_age_days)].copy()
    if work.empty:
        return pd.DataFrame(columns=empty_columns)
    age_days = (cutoff - work["available_at"]).dt.total_seconds() / 86400.0
    work["decay"] = np.exp(-np.log(2.0) * age_days / max(cfg.half_life_days, 1e-6))
    work["reliability"] = work["evidence_type"].map(EVIDENCE_RELIABILITY).fillna(
        EVIDENCE_RELIABILITY["other"]
    )
    if "source_independence_key" not in work.columns:
        work["source_independence_key"] = work["evidence_type"].astype(str)
    else:
        work["source_independence_key"] = work["source_independence_key"].fillna(
            work["evidence_type"]
        ).astype(str)
    work["contribution"] = (
        pd.to_numeric(work["evidence_strength"], errors="coerce").fillna(0.0).clip(0, 1)
        * work["reliability"] * work["decay"]
    )
    work = work.sort_values("contribution", ascending=False).drop_duplicates(
        ["scope", "scope_value", "source_independence_key"], keep="first"
    )

    rows: list[dict[str, object]] = []
    for (scope, scope_value), group in work.groupby(["scope", "scope_value"], dropna=False):
        log_odds = _logit(cfg.prior_probability) + float((2.5 * group["contribution"]).sum())
        independent = int(group["source_independence_key"].nunique())
        rows.append(
            {
                "as_of": cutoff,
                "scope": str(scope),
                "scope_value": str(scope_value),
                "posterior_probability": float(_sigmoid(log_odds)),
                "independent_evidence_types": independent,
                "feature_usable": independent >= cfg.min_independent_evidence_types,
                "evidence_label": "inferred",
                "evidence_types": sorted(group["evidence_type"].astype(str).unique()),
                "latest_available_at": group["available_at"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["posterior_probability", "independent_evidence_types"], ascending=False
    ).reset_index(drop=True)
