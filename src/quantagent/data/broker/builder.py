"""BrokerReportBuilder — silver/broker_reports.parquet."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BROKER_REPORT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "symbol",
    "broker",
    "broker_tier",
    "announced_at",
    "available_at",
    "rating",
    "rating_change",
    "target_price",
    "prev_target_price",
    "target_price_pct_change",
    "summary",
    "broker_credibility",
    "source",
    "source_version",
    "fetched_at",
)


# Tier table — credibility weight for top-tier Chinese brokers
BROKER_TIER_TABLE: dict[str, tuple[str, float]] = {
    # tier_1 — major bulge-bracket / sell-side with the best historical accuracy
    "中信证券": ("tier_1", 0.85),
    "华泰证券": ("tier_1", 0.85),
    "中金公司": ("tier_1", 0.85),
    "中信建投": ("tier_1", 0.82),
    "招商证券": ("tier_1", 0.82),
    "国泰君安": ("tier_1", 0.82),
    "海通证券": ("tier_1", 0.80),
    "申万宏源": ("tier_1", 0.80),
    "广发证券": ("tier_1", 0.80),
    "兴业证券": ("tier_1", 0.78),
    # tier_2 — mid-market sell-side
    "光大证券": ("tier_2", 0.70),
    "国信证券": ("tier_2", 0.70),
    "民生证券": ("tier_2", 0.65),
    "东方证券": ("tier_2", 0.65),
    "天风证券": ("tier_2", 0.65),
    "西部证券": ("tier_2", 0.60),
    # any not in this list defaults to tier_3 (0.50)
}


VALID_RATINGS: tuple[str, ...] = (
    "buy",
    "overweight",
    "hold",
    "underweight",
    "sell",
    "n/a",
)


VALID_RATING_CHANGES: tuple[str, ...] = (
    "initiate",
    "upgrade",
    "maintain",
    "downgrade",
    "drop",
    "n/a",
)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrokerReportConfig:
    source: str = "manual_local_import"
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    available_at_lag_days: int = 1
    # Gate
    min_events: int = 10
    min_unique_brokers: int = 3
    min_mean_credibility: float = 0.50
    # Default credibility for unknown brokers
    default_tier: str = "tier_3"
    default_credibility: float = 0.50


@dataclass
class BrokerReportResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _event_id(broker: str, symbol: str, announced_at: pd.Timestamp) -> str:
    raw = f"{broker}||{symbol}||{announced_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _resolve_broker_tier(broker: str, config: BrokerReportConfig) -> tuple[str, float]:
    """Return (tier, credibility) for a broker name."""
    if broker in BROKER_TIER_TABLE:
        return BROKER_TIER_TABLE[broker]
    # Partial match (e.g. "中信证券股份有限公司" → "中信证券")
    for known, (tier, cred) in BROKER_TIER_TABLE.items():
        if known in broker:
            return tier, cred
    return config.default_tier, config.default_credibility


def _normalise_rating(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    s = str(value).strip().lower()
    if s in VALID_RATINGS:
        return s
    # Chinese mappings
    cn_map = {
        "买入": "buy", "增持": "overweight", "持有": "hold",
        "中性": "hold", "推荐": "buy", "谨慎推荐": "overweight",
        "减持": "underweight", "卖出": "sell",
    }
    return cn_map.get(str(value).strip(), "n/a")


def _normalise_rating_change(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "maintain"
    s = str(value).strip().lower()
    return s if s in VALID_RATING_CHANGES else "maintain"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_broker_reports(
    raw: pd.DataFrame,
    *,
    config: BrokerReportConfig | None = None,
) -> BrokerReportResult:
    cfg = config or BrokerReportConfig()
    if raw is None or raw.empty:
        return _empty_result(cfg)

    required_in = {"broker", "symbol", "announced_at"}
    missing = required_in - set(raw.columns)
    if missing:
        raise ValueError(
            f"raw broker frame missing required columns: {sorted(missing)}"
        )

    work = raw.copy()
    work["broker"] = work["broker"].fillna("").astype(str)
    work["symbol"] = work["symbol"].fillna("").astype(str)
    work["announced_at"] = work["announced_at"].map(_coerce_ts)

    # Drop unparseable date rows
    before = len(work)
    work = work[work["announced_at"].notna()].reset_index(drop=True)
    rejected_no_date = before - len(work)

    # Apply tier + credibility lookup
    tier_pairs = work["broker"].apply(lambda b: _resolve_broker_tier(b, cfg))
    work["broker_tier"] = [p[0] for p in tier_pairs]
    work["broker_credibility"] = [p[1] for p in tier_pairs]

    # Optional caller override
    if "broker_credibility_override" in raw.columns:
        ovr = pd.to_numeric(raw["broker_credibility_override"], errors="coerce")
        mask = ovr.notna()
        work.loc[mask, "broker_credibility"] = ovr[mask].clip(0.0, 1.0)

    # Numeric prices
    work["target_price"] = pd.to_numeric(work.get("target_price"), errors="coerce")
    work["prev_target_price"] = pd.to_numeric(
        work.get("prev_target_price"), errors="coerce"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = work["target_price"] / work["prev_target_price"] - 1.0
    work["target_price_pct_change"] = pct.where(pct.notna(), other=np.nan)

    # Ratings
    if "rating" not in work.columns:
        work["rating"] = "n/a"
    work["rating"] = work["rating"].apply(_normalise_rating)
    if "rating_change" not in work.columns:
        work["rating_change"] = "maintain"
    work["rating_change"] = work["rating_change"].apply(_normalise_rating_change)

    if "summary" not in work.columns:
        work["summary"] = ""
    work["summary"] = work["summary"].fillna("").astype(str)

    if "source" not in work.columns:
        work["source"] = cfg.source
    work["source"] = work["source"].fillna(cfg.source).astype(str)

    if "source_version" not in work.columns:
        work["source_version"] = cfg.source_version
    work["source_version"] = work["source_version"].fillna(cfg.source_version).astype(str)

    fetched_default = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    if "fetched_at" not in work.columns:
        work["fetched_at"] = fetched_default
    work["fetched_at"] = work["fetched_at"].map(_coerce_ts).fillna(fetched_default)
    # available_at = announced_at + N business days (research disclosures
    # are typically published next morning)
    fallback_available = work["announced_at"] + pd.tseries.offsets.BDay(
        cfg.available_at_lag_days
    )
    if "available_at" in work.columns:
        work["available_at"] = (
            work["available_at"].map(_coerce_ts).fillna(fallback_available)
        )
    else:
        work["available_at"] = fallback_available

    # event_id
    work["event_id"] = work.apply(
        lambda r: _event_id(r["broker"], r["symbol"], r["announced_at"]), axis=1
    )

    # De-dup by event_id
    before_dedup = len(work)
    work = work.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)
    dup_removed = before_dedup - len(work)

    out = work[list(BROKER_REPORT_REQUIRED_COLUMNS)].copy()

    n = int(len(out))
    unique_brokers = int(out["broker"].nunique())
    mean_cred = float(out["broker_credibility"].mean()) if n else 0.0
    rating_counts = out["rating"].value_counts().to_dict()
    rating_change_counts = out["rating_change"].value_counts().to_dict()
    tier_counts = out["broker_tier"].value_counts().to_dict()

    gate_open = (
        n >= cfg.min_events
        and unique_brokers >= cfg.min_unique_brokers
        and mean_cred >= cfg.min_mean_credibility
    )
    reason = "passed" if gate_open else _gate_reason(
        n, unique_brokers, mean_cred, cfg
    )

    coverage = {
        "n_events": n,
        "unique_brokers": unique_brokers,
        "mean_credibility": mean_cred,
        "rejected_no_date": int(rejected_no_date),
        "duplicates_removed": int(dup_removed),
        "rating_counts": rating_counts,
        "rating_change_counts": rating_change_counts,
        "tier_counts": tier_counts,
        "gate": {
            "broker_reports_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {"status": "passed", "n": n, "errors": []}
    return BrokerReportResult(frame=out, coverage=coverage, validation=validation)


def _empty_result(cfg: BrokerReportConfig) -> BrokerReportResult:
    return BrokerReportResult(
        frame=pd.DataFrame(columns=list(BROKER_REPORT_REQUIRED_COLUMNS)),
        coverage={
            "n_events": 0,
            "unique_brokers": 0,
            "mean_credibility": 0.0,
            "gate": {"broker_reports_usable_for_features": False, "reason": "no_events"},
        },
        validation={"status": "passed", "n": 0, "errors": []},
    )


def _gate_reason(
    n: int,
    unique_brokers: int,
    mean_cred: float,
    cfg: BrokerReportConfig,
) -> str:
    if n < cfg.min_events:
        return f"too_few_events_{n}_lt_{cfg.min_events}"
    if unique_brokers < cfg.min_unique_brokers:
        return f"too_few_brokers_{unique_brokers}_lt_{cfg.min_unique_brokers}"
    if mean_cred < cfg.min_mean_credibility:
        return f"mean_credibility_{mean_cred:.3f}_below_{cfg.min_mean_credibility:.3f}"
    return "unknown"


# ---------------------------------------------------------------------------
# Builder with writer
# ---------------------------------------------------------------------------

class BrokerReportBuilder:
    def __init__(self, config: BrokerReportConfig | None = None) -> None:
        self.config = config or BrokerReportConfig()

    def build(self, raw: pd.DataFrame) -> BrokerReportResult:
        return build_broker_reports(raw, config=self.config)

    def write(self, result: BrokerReportResult) -> BrokerReportResult:
        root = Path(self.config.output_root)
        silver_dir = root / "silver" / "broker_reports"
        silver_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = silver_dir / "broker_reports.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        (silver_dir / "coverage_report.json").write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        (silver_dir / "validation_report.json").write_text(
            json.dumps(result.validation, indent=2, default=str), encoding="utf-8"
        )
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "broker_reports.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "broker_reports",
                    "rows": int(len(result.frame)),
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "broker_reports": str(parquet_path),
            "coverage_report": str(silver_dir / "coverage_report.json"),
            "validation_report": str(silver_dir / "validation_report.json"),
            "manifest": str(manifest_path),
        }
        return result


# ---------------------------------------------------------------------------
# Feature attach
# ---------------------------------------------------------------------------

_RATING_SCORE: dict[str, float] = {
    "buy": 1.0,
    "overweight": 0.5,
    "hold": 0.0,
    "underweight": -0.5,
    "sell": -1.0,
    "n/a": 0.0,
}


def apply_broker_report_features(
    panel: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    decay_days: int = 60,
) -> pd.DataFrame:
    """Attach broker-consensus features to a training panel.

    Adds two columns per symbol:
      * ``broker_consensus_score`` — weighted average rating score
        across all reports with ``available_at <= trade_date``, decayed
        with half-life ``decay_days``.
      * ``broker_target_premium`` — weighted average of
        ``target_price / spot_close - 1`` from reports with
        non-null target_price (spot proxied by the most-recent
        ``target_price`` instead of a live spot to keep the function
        self-contained).
    """
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()
    if reports is None or reports.empty:
        out = panel.copy()
        out["broker_consensus_score"] = 0.0
        out["broker_target_premium"] = 0.0
        return out

    panel_out = panel.copy()
    panel_out["trade_date"] = pd.to_datetime(panel_out["trade_date"])
    panel_out["symbol"] = panel_out["symbol"].astype(str)

    rep = reports.copy()
    rep["available_at"] = pd.to_datetime(rep["available_at"])
    rep["rating_score"] = rep["rating"].map(_RATING_SCORE).fillna(0.0)
    rep["weight"] = rep["broker_credibility"].astype(float)
    rep = rep.sort_values(["symbol", "available_at"])

    consensus = pd.Series(0.0, index=panel_out.index, dtype=float)
    premium = pd.Series(0.0, index=panel_out.index, dtype=float)

    for sym, sub in rep.groupby("symbol"):
        mask = panel_out["symbol"] == sym
        if not mask.any():
            continue
        sub = sub.sort_values("available_at")
        left = panel_out.loc[mask, ["trade_date"]].copy()
        left["__orig_index"] = left.index
        left = left.sort_values("trade_date").reset_index(drop=True)
        # For each panel row, compute weighted-sum / weight-sum across all
        # reports with available_at <= trade_date, with exponential decay.
        # We do this row-by-row because pandas doesn't have a clean rolling
        # weighted-mean primitive for our (asof, weighted, decayed) case.
        for _, panel_row in left.iterrows():
            t = panel_row["trade_date"]
            visible = sub[sub["available_at"] <= t]
            if visible.empty:
                continue
            age_days = (t - visible["available_at"]).dt.days.clip(lower=0)
            decay = np.exp(-age_days / max(1, decay_days))
            w = visible["weight"].values * decay.values
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                continue
            cons = float(np.sum(visible["rating_score"].values * w) / w_sum)
            consensus.loc[panel_row["__orig_index"]] = cons
            tp = visible["target_price"].astype(float).values
            if np.any(np.isfinite(tp)):
                spot_proxy = float(np.nanmedian(tp))
                ratio = (tp / spot_proxy) - 1.0
                ratio_mean = float(
                    np.nansum(ratio * w) / max(1e-12, np.nansum(w[np.isfinite(ratio)]))
                )
                premium.loc[panel_row["__orig_index"]] = ratio_mean

    panel_out["broker_consensus_score"] = consensus
    panel_out["broker_target_premium"] = premium
    return panel_out


# ---------------------------------------------------------------------------
# Manifest-gated helper
# ---------------------------------------------------------------------------

def broker_reports_for_features(
    reports: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    if reports is None or len(reports) == 0:
        return None
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate = (
        (payload.get("extra") or {})
        .get("coverage_report", {})
        .get("gate", {})
    )
    if not gate.get("broker_reports_usable_for_features"):
        return None
    return reports
