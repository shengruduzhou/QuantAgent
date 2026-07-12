"""Probabilistic aggregation for public state-team evidence.

The output is an inference, never a confirmed ownership or trading claim.  A
posterior is produced only from evidence available at the evaluation time and
requires multiple independent evidence families before it can be marked
feature-usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal

import numpy as np
import pandas as pd


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
    # 5 CNY bn = 50 亿元.  The previous code comment incorrectly called this
    # 5 亿元, which created a tenfold ambiguity.
    concentrated_etf_inflow_threshold_cny_bn: float = 5.0


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
    """Return an ETF-flow frame in CNY billions with explicit unit metadata."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    numeric_cols = [c for c in out.columns if c not in {"trade_date", "available_at", "source"}]
    values = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    sample = values.stack().dropna().abs()
    resolved = unit
    if unit == "auto":
        median = float(sample.median()) if not sample.empty else 0.0
        if median >= 1e8:
            resolved = "cny"
        elif median >= 1e3:
            resolved = "cny_mn"
        else:
            resolved = "cny_bn"
    if resolved == "cny":
        values = values / 1e9
    elif resolved == "cny_mn":
        values = values / 1e3
    elif resolved != "cny_bn":
        raise ValueError(f"unsupported ETF flow unit: {resolved}")
    out[numeric_cols] = values
    out.attrs["flow_unit"] = "cny_bn"
    out.attrs["input_flow_unit"] = resolved
    return out


def holder_filings_to_events(
    filings: pd.DataFrame,
    *,
    allow_estimated_availability: bool = False,
) -> pd.DataFrame:
    """Convert holder filings to evidence using actual announcement time.

    Required columns are ``symbol``, ``holder_name`` and ``share_pct`` plus
    ``announcement_date`` or ``available_at``.  Quarter-end dates are not valid
    feature timestamps.  A fallback estimate is allowed only when explicitly
    requested and is labelled as estimated.
    """
    if filings is None or filings.empty:
        return pd.DataFrame()
    required = {"symbol", "holder_name", "share_pct"}
    missing = required - set(filings.columns)
    if missing:
        raise ValueError(f"holder filings missing columns: {sorted(missing)}")
    out = filings.copy()
    if "announcement_date" in out.columns:
        available = pd.to_datetime(out["announcement_date"], errors="coerce")
        availability_quality = "reported"
    elif "available_at" in out.columns:
        available = pd.to_datetime(out["available_at"], errors="coerce")
        availability_quality = "reported"
    elif allow_estimated_availability and "report_period" in out.columns:
        available = pd.to_datetime(out["report_period"], errors="coerce") + pd.tseries.offsets.BDay(45)
        availability_quality = "estimated_45bd"
    else:
        raise ValueError(
            "holder filings require announcement_date/available_at; "
            "quarter-end plus 45 business days is disabled by default"
        )
    out["available_at"] = available
    out = out[out["available_at"].notna()].copy()
    out["trade_date"] = out["available_at"].dt.normalize()
    out["evidence_type"] = "official_holder_filing"
    out["evidence_label"] = "inferred"
    out["evidence_strength"] = (
        0.45 + 0.10 * pd.to_numeric(out["share_pct"], errors="coerce").fillna(0.0).clip(0, 5)
    ).clip(0, 0.95)
    out["scope"] = "symbol"
    out["scope_value"] = out["symbol"].astype(str)
    out["source_independence_key"] = "holder_filing"
    out["availability_quality"] = availability_quality
    out["description"] = (
        out["holder_name"].astype(str) + " " + out["share_pct"].astype(str) + "%"
    )
    return out[
        [
            "trade_date",
            "available_at",
            "evidence_type",
            "evidence_label",
            "evidence_strength",
            "scope",
            "scope_value",
            "source_independence_key",
            "availability_quality",
            "description",
        ]
    ]


def compute_state_team_posterior(
    events: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    config: StateTeamPosteriorConfig | None = None,
) -> pd.DataFrame:
    """Aggregate evidence into a posterior by date and scope.

    Correlated rows from the same evidence family are capped by taking the
    strongest row per ``source_independence_key``.  This prevents several ETF
    products tracking the same index from being treated as independent votes.
    """
    cfg = config or StateTeamPosteriorConfig()
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "as_of",
                "scope",
                "scope_value",
                "posterior_probability",
                "independent_evidence_types",
                "feature_usable",
                "evidence_label",
            ]
        )
    work = events.copy()
    required = {"available_at", "evidence_type", "evidence_strength", "scope", "scope_value"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"state-team events missing columns: {sorted(missing)}")
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    work = work[work["available_at"].notna()].copy()
    cutoff = pd.Timestamp(as_of) if as_of is not None else work["available_at"].max()
    work = work[work["available_at"] <= cutoff].copy()
    if work.empty:
        return pd.DataFrame()
    age_days = (cutoff - work["available_at"]).dt.total_seconds() / 86400.0
    work = work[age_days <= cfg.max_event_age_days].copy()
    age_days = (cutoff - work["available_at"]).dt.total_seconds() / 86400.0
    work["decay"] = np.exp(-np.log(2.0) * age_days / max(cfg.half_life_days, 1e-6))
    work["reliability"] = work["evidence_type"].map(EVIDENCE_RELIABILITY).fillna(
        EVIDENCE_RELIABILITY["other"]
    )
    work["source_independence_key"] = work.get(
        "source_independence_key", work["evidence_type"]
    ).astype(str)
    work["contribution"] = (
        pd.to_numeric(work["evidence_strength"], errors="coerce").fillna(0.0).clip(0, 1)
        * work["reliability"]
        * work["decay"]
    )
    work = work.sort_values("contribution", ascending=False).drop_duplicates(
        subset=["scope", "scope_value", "source_independence_key"], keep="first"
    )

    rows: list[dict[str, object]] = []
    for (scope, scope_value), group in work.groupby(["scope", "scope_value"], dropna=False):
        # Contribution is converted to a bounded log-likelihood increment.
        log_odds = _logit(cfg.prior_probability) + float((2.5 * group["contribution"]).sum())
        probability = _sigmoid(log_odds)
        independent = int(group["source_independence_key"].nunique())
        rows.append(
            {
                "as_of": cutoff,
                "scope": str(scope),
                "scope_value": str(scope_value),
                "posterior_probability": float(probability),
                "independent_evidence_types": independent,
                "feature_usable": independent >= cfg.min_independent_evidence_types,
                "evidence_label": "inferred",
                "evidence_types": sorted(group["evidence_type"].astype(str).unique().tolist()),
                "latest_available_at": group["available_at"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["posterior_probability", "independent_evidence_types"], ascending=[False, False]
    ).reset_index(drop=True)
