"""Daily evidence job: orchestrates all ingestors and emits a unified table.

The job runs once a day (typically before market open) and merges:

* Policy documents from official government sources.
* Exchange and listed-company disclosures (公告 / 问询函 / 处罚).
* News articles (regulated > tier1 > tier2 > self media).
* Financial statements (TuShare / AkShare PIT cache).
* Order / contract evidence from announcements.
* Regulatory penalty / audit-opinion evidence.

Every ingestor returns a DataFrame with the columns in :data:`EVIDENCE_COLUMNS`,
which lets the downstream V7DataHub treat them as a single table even when
some sources are missing.

The job has two execution modes:

* ``dry_run=True`` (default): only re-reads local CSV/Parquet caches.
* ``dry_run=False``: each individual ingestor is allowed to issue network
  calls if ``allow_network=True`` is set on it. This is opt-in per source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry, SourceProfile


EVIDENCE_COLUMNS: tuple[str, ...] = (
    "evidence_id",
    "source_type",
    "source_name",
    "source_authority",
    "source_reliability",
    "is_primary_source",
    "is_official",
    "url",
    "title",
    "body",
    "published_at",
    "available_at",
    "ingested_at",
    "symbol",
    "company_name",
    "theme_candidates",
    "chain_node_candidates",
    "event_type",
    "confidence",
    "cross_validation_count",
    "contradiction_count",
    "horizon_days",
    "decay_half_life",
    "rumor_risk_flag",
    "affected_symbols",
    "raw_hash",
    "point_in_time_valid",
)


_EVENT_TYPE_HORIZON_TABLE: dict[str, tuple[int, float]] = {
    "policy_support": (120, 90.0),
    "subsidy": (90, 60.0),
    "industrial_plan": (180, 120.0),
    "demand_growth": (60, 45.0),
    "supply_shortage": (60, 30.0),
    "order_confirmed": (60, 45.0),
    "earnings_growth": (60, 60.0),
    "margin_expansion": (90, 60.0),
    "valuation_repair": (40, 30.0),
    "capital_inflow": (20, 15.0),
    "sentiment_positive": (5, 3.0),
    "sentiment_negative": (5, 3.0),
    "regulatory_penalty": (60, 30.0),
    "fraud_risk": (120, 60.0),
    "accounting_anomaly": (90, 45.0),
    "audit_opinion": (180, 90.0),
    "liquidity_risk": (5, 2.0),
    "theme_rotation": (20, 10.0),
    "bubble_warning": (20, 10.0),
    "no_trade": (1, 1.0),
    "hedge_signal": (5, 2.0),
    "financial_statement": (90, 60.0),
    "inquiry_letter": (60, 30.0),
    "restatement": (120, 60.0),
    "shareholder_change": (20, 10.0),
    "goodwill_impairment": (60, 30.0),
    "pledge": (20, 10.0),
}


@dataclass(frozen=True)
class DailyEvidenceJobConfig:
    """Configuration for the daily evidence job."""

    as_of_date: str
    dry_run: bool = True
    cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence"))
    enabled_sources: tuple[str, ...] = (
        "policy",
        "disclosure",
        "news",
        "financial",
        "order_contract",
        "regulatory_penalty",
    )
    available_lag_days: int = 1
    write_to_store: bool = True
    store_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "store"))


@dataclass(frozen=True)
class DailyEvidenceJobResult:
    frame: pd.DataFrame
    per_ingestor_rows: dict[str, int]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


class DailyEvidenceJob:
    """Aggregates evidence ingestors into a single per-day frame.

    Each ingestor is injected — that keeps the job 100% deterministic in
    tests (we feed it pre-built frames) while in production it pulls from
    the real provider stack. The job never silently invents evidence: if an
    ingestor returns an empty frame the warning is propagated.
    """

    def __init__(
        self,
        registry: SourceCredibilityRegistry | None = None,
        ingestors: dict[str, "EvidenceIngestor"] | None = None,
    ) -> None:
        self.registry = registry or SourceCredibilityRegistry()
        self.ingestors = ingestors or {}

    def run(self, config: DailyEvidenceJobConfig) -> DailyEvidenceJobResult:
        frames: list[pd.DataFrame] = []
        per_ingestor: dict[str, int] = {}
        warnings: list[str] = []
        for name in config.enabled_sources:
            ingestor = self.ingestors.get(name)
            if ingestor is None:
                warnings.append(f"missing_ingestor:{name}")
                per_ingestor[name] = 0
                continue
            try:
                frame = ingestor.fetch(config, self.registry)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"ingestor_failed:{name}:{exc}")
                per_ingestor[name] = 0
                continue
            if frame is None or frame.empty:
                warnings.append(f"ingestor_empty:{name}")
                per_ingestor[name] = 0
                continue
            normalised = normalise_evidence_frame(frame, name, config)
            per_ingestor[name] = int(len(normalised))
            frames.append(normalised)
        unified = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVIDENCE_COLUMNS)
        unified = enforce_pit(unified, config.as_of_date)
        cached_path = _maybe_cache(unified, config)
        store_paths = _maybe_write_store(unified, config)
        return DailyEvidenceJobResult(
            frame=unified,
            per_ingestor_rows=per_ingestor,
            warnings=tuple(warnings),
            metadata={
                "as_of_date": config.as_of_date,
                "rows": str(len(unified)),
                "cache_path": str(cached_path) if cached_path else "",
                "store_partitions": ",".join(str(path.parent) for path in store_paths) if store_paths else "",
            },
        )


class EvidenceIngestor:
    """Protocol used by the daily evidence job.

    Subclasses must implement :meth:`fetch` and return a DataFrame whose
    columns are a subset of :data:`EVIDENCE_COLUMNS`.
    """

    name: str = "evidence"
    source_type: str = "news"

    def fetch(
        self,
        config: DailyEvidenceJobConfig,
        registry: SourceCredibilityRegistry,
    ) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError


def normalise_evidence_frame(
    frame: pd.DataFrame,
    ingestor_name: str,
    config: DailyEvidenceJobConfig,
) -> pd.DataFrame:
    data = frame.copy()
    for column in EVIDENCE_COLUMNS:
        if column not in data.columns:
            data[column] = None
    if "published_at" in data.columns:
        data["published_at"] = pd.to_datetime(data["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "available_at" not in data.columns or data["available_at"].isna().all():
        data["available_at"] = data["published_at"]
    data["available_at"] = pd.to_datetime(data["available_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["ingested_at"] = config.as_of_date
    data["point_in_time_valid"] = pd.to_datetime(data["available_at"], errors="coerce") <= pd.Timestamp(config.as_of_date)
    if "evidence_id" not in data.columns or data["evidence_id"].isna().any():
        data["evidence_id"] = data.apply(lambda row: _stable_evidence_id(row, ingestor_name), axis=1)
    if "raw_hash" not in data.columns or data["raw_hash"].isna().any():
        data["raw_hash"] = data.apply(lambda row: _stable_raw_hash(row), axis=1)
    _fill_decay_columns(data)
    if data["affected_symbols"].isna().any():
        data["affected_symbols"] = data["affected_symbols"].fillna(data["symbol"].fillna(""))
    if data["rumor_risk_flag"].isna().any():
        data["rumor_risk_flag"] = data["rumor_risk_flag"].fillna(False)
    if data["cross_validation_count"].isna().any():
        data["cross_validation_count"] = data["cross_validation_count"].fillna(0).astype(int)
    if data["contradiction_count"].isna().any():
        data["contradiction_count"] = data["contradiction_count"].fillna(0).astype(int)
    return data[list(EVIDENCE_COLUMNS)]


def _fill_decay_columns(data: pd.DataFrame) -> None:
    event_types = data["event_type"].fillna("no_trade").astype(str)
    horizons = data["horizon_days"].copy()
    decays = data["decay_half_life"].copy()
    for index, event in event_types.items():
        horizon, decay = _EVENT_TYPE_HORIZON_TABLE.get(event, (20, 14.0))
        if pd.isna(horizons.at[index]):
            horizons.at[index] = horizon
        if pd.isna(decays.at[index]):
            decays.at[index] = decay
    data["horizon_days"] = horizons.astype(int)
    data["decay_half_life"] = decays.astype(float)


def enforce_pit(frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    parsed = pd.to_datetime(frame["available_at"], errors="coerce")
    visible = parsed.notna() & (parsed <= pd.Timestamp(as_of_date))
    return frame.loc[visible].reset_index(drop=True)


def _stable_evidence_id(row: pd.Series, ingestor_name: str) -> str:
    base = f"{ingestor_name}|{row.get('source_name','')}|{row.get('url','')}|{row.get('title','')}|{row.get('published_at','')}"
    return sha256(base.encode("utf-8")).hexdigest()[:24]


def _stable_raw_hash(row: pd.Series) -> str:
    base = f"{row.get('url','')}|{row.get('title','')}|{row.get('body','')}"
    return sha256(base.encode("utf-8")).hexdigest()


def _maybe_cache(frame: pd.DataFrame, config: DailyEvidenceJobConfig) -> Path | None:
    if frame is None or frame.empty:
        return None
    root = Path(config.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"evidence_{config.as_of_date}.csv"
    frame.to_csv(path, index=False)
    return path


def _maybe_write_store(frame: pd.DataFrame, config: DailyEvidenceJobConfig) -> list[Path]:
    if frame is None or frame.empty or not config.write_to_store:
        return []
    # Defer the import so EvidenceStore stays optional in fully-offline tests.
    from quantagent.data.ingestion.evidence_store import EvidenceStore, EvidenceStoreConfig

    store = EvidenceStore(EvidenceStoreConfig(root=config.store_root))
    return store.write(frame)


def attach_source_profile(
    frame: pd.DataFrame,
    registry: SourceCredibilityRegistry,
    default_tier_fallback: SourceProfile | None = None,
) -> pd.DataFrame:
    """Resolve ``source_name`` -> profile and fill credibility / authority columns."""

    if frame is None or frame.empty:
        return frame
    data = frame.copy()
    profiles = [registry.resolve(str(name) if name else "") for name in data.get("source_name", [])]
    data["source_authority"] = [profile.authority_score() for profile in profiles]
    data["source_reliability"] = [profile.reliability for profile in profiles]
    data["is_primary_source"] = [bool(profile.is_primary) for profile in profiles]
    data["is_official"] = [bool(profile.is_official) for profile in profiles]
    if "source_type" not in data.columns or data["source_type"].isna().any():
        data["source_type"] = [profile.source_type for profile in profiles]
    return data


def _iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
