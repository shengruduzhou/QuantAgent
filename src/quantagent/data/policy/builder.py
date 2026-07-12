"""PIT-safe policy event builder with separate public and ingestion clocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quantagent.data.policy.theme_tagger import tag_policy_event


POLICY_EVENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id", "source", "source_authority", "document_type", "url",
    "title", "body_summary", "announced_at", "published_at",
    "effective_at", "public_available_at", "ingested_at", "available_at",
    "revised_at", "superseded_at", "jurisdiction", "funding_amount_cny",
    "themes", "sectors_hint", "policy_strength", "raw_hash", "fetched_at",
    "source_version",
)

VALID_SOURCES: tuple[str, ...] = (
    "csrc", "pboc", "mof", "ndrc", "state_council", "sse", "szse",
    "bse", "stats", "local_government", "manual_local_import",
)

SOURCE_AUTHORITY: dict[str, float] = {
    "state_council": 1.00, "pboc": 0.95, "mof": 0.95, "ndrc": 0.92,
    "csrc": 0.92, "sse": 0.85, "szse": 0.85, "bse": 0.85,
    "stats": 0.82, "local_government": 0.75, "manual_local_import": 0.50,
}

DOCUMENT_TYPE_WEIGHT: dict[str, float] = {
    "law": 1.00, "administrative_regulation": 0.95, "measure": 0.90,
    "rule": 0.88, "opinion": 0.82, "notice": 0.78, "plan": 0.76,
    "meeting": 0.65, "speech": 0.55, "other": 0.50,
}


@dataclass(frozen=True)
class PolicyEventConfig:
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    availability_mode: Literal["public", "ingested"] = "public"
    min_theme_coverage: float = 0.50
    min_strength_median: float = 0.30
    min_events: int = 5
    require_official_url: bool = False


@dataclass
class PolicyEventResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


def _series(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(default, index=frame.index)


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts)


def _normalise_source(value: Any) -> str:
    source = str(value or "").strip().lower()
    return source if source in VALID_SOURCES else "manual_local_import"


def _normalise_document_type(value: Any) -> str:
    value = str(value or "other").strip().lower()
    aliases = {
        "法律": "law", "行政法规": "administrative_regulation", "办法": "measure",
        "规定": "rule", "意见": "opinion", "通知": "notice", "规划": "plan",
        "会议": "meeting", "讲话": "speech",
    }
    value = aliases.get(value, value)
    return value if value in DOCUMENT_TYPE_WEIGHT else "other"


def _ensure_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _numeric_or_nan(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else float("nan")


def _content_hash(title: str, body: str, url: str) -> str:
    return hashlib.sha256(f"{title.strip()}\n{body.strip()}\n{url.strip()}".encode()).hexdigest()


def _event_id(source: str, url: str, published_at: pd.Timestamp, title: str) -> str:
    raw = f"{source}||{url}||{published_at.isoformat()}||{title.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _derive_strength(source: str, document_type: str, tagged: float, funding: float) -> float:
    funding_bonus = 0.0
    if pd.notna(funding) and funding > 0:
        funding_bonus = min(0.12, 0.02 * max(0, len(str(int(funding))) - 8))
    score = (
        0.45 * float(tagged)
        + 0.35 * SOURCE_AUTHORITY.get(source, 0.50)
        + 0.20 * DOCUMENT_TYPE_WEIGHT.get(document_type, 0.50)
        + funding_bonus
    )
    return float(max(0.0, min(1.0, score)))


def build_policy_events(
    raw: pd.DataFrame,
    *,
    config: PolicyEventConfig | None = None,
) -> PolicyEventResult:
    cfg = config or PolicyEventConfig()
    if raw is None or raw.empty:
        return _empty_result(cfg)
    missing = {"source", "title"} - set(raw.columns)
    if missing:
        raise ValueError(f"raw policy frame missing required columns: {sorted(missing)}")
    if "published_at" not in raw.columns and "announced_at" not in raw.columns:
        raise ValueError("raw policy frame requires published_at or announced_at")

    frame = raw.copy().reset_index(drop=False).rename(columns={"index": "_source_row_id"})
    frame["source"] = frame["source"].map(_normalise_source)
    frame["source_authority"] = frame["source"].map(SOURCE_AUTHORITY).fillna(0.50)
    frame["document_type"] = _series(frame, "document_type", "other").map(
        _normalise_document_type
    )
    frame["url"] = _series(frame, "url", "").fillna("").astype(str)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["body_summary"] = _series(frame, "body_summary", "").fillna("").astype(str)

    published = frame["published_at"] if "published_at" in frame else frame["announced_at"]
    frame["published_at"] = published.map(_coerce_timestamp)
    frame["announced_at"] = frame["published_at"]
    frame["effective_at"] = _series(frame, "effective_at", pd.NaT).map(_coerce_timestamp)
    frame["effective_at"] = frame["effective_at"].fillna(frame["published_at"])
    frame["public_available_at"] = _series(
        frame, "public_available_at", pd.NaT
    ).map(_coerce_timestamp).fillna(frame["published_at"])

    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    ingest_source = frame["ingested_at"] if "ingested_at" in frame else _series(
        frame, "fetched_at", now
    )
    frame["ingested_at"] = ingest_source.map(_coerce_timestamp).fillna(now)
    frame["fetched_at"] = frame["ingested_at"]
    frame["revised_at"] = _series(frame, "revised_at", pd.NaT).map(_coerce_timestamp)
    frame["superseded_at"] = _series(frame, "superseded_at", pd.NaT).map(_coerce_timestamp)
    frame["jurisdiction"] = _series(frame, "jurisdiction", "national").fillna("national").astype(str)
    frame["funding_amount_cny"] = _series(frame, "funding_amount_cny", float("nan")).map(
        _numeric_or_nan
    )

    if cfg.availability_mode == "public":
        frame["available_at"] = frame["public_available_at"]
    elif cfg.availability_mode == "ingested":
        frame["available_at"] = frame[["public_available_at", "ingested_at"]].max(axis=1)
    else:
        raise ValueError(f"unsupported availability_mode: {cfg.availability_mode}")

    before = len(frame)
    frame = frame[frame["published_at"].notna()].copy()
    rejected_no_date = before - len(frame)
    rejected_no_url = 0
    if cfg.require_official_url:
        before = len(frame)
        frame = frame[frame["url"].str.startswith(("http://", "https://"))].copy()
        rejected_no_url = before - len(frame)

    frame["event_id"] = frame.apply(
        lambda row: _event_id(row["source"], row["url"], row["published_at"], row["title"]),
        axis=1,
    )
    frame["raw_hash"] = frame.apply(
        lambda row: _content_hash(row["title"], row["body_summary"], row["url"]), axis=1
    )
    tags = frame.apply(lambda row: tag_policy_event(row["title"], row["body_summary"]), axis=1)
    frame["themes"] = [tag["themes"] for tag in tags]
    frame["sectors_hint"] = [tag["sectors_hint"] for tag in tags]
    tag_strength = pd.Series([float(tag["policy_strength"]) for tag in tags], index=frame.index)

    for column, target in (("themes_override", "themes"), ("sectors_hint_override", "sectors_hint")):
        if column in frame.columns:
            for index, value in frame[column].items():
                override = _ensure_list(value)
                if override:
                    frame.at[index, target] = override

    frame["policy_strength"] = [
        _derive_strength(source, doc_type, tagged, funding)
        for source, doc_type, tagged, funding in zip(
            frame["source"], frame["document_type"], tag_strength, frame["funding_amount_cny"]
        )
    ]
    frame["source_version"] = cfg.source_version
    before = len(frame)
    frame = frame.sort_values(["published_at", "ingested_at", "event_id"]).drop_duplicates(
        "event_id", keep="first"
    )
    duplicates_removed = before - len(frame)
    out = frame[list(POLICY_EVENT_REQUIRED_COLUMNS)].reset_index(drop=True)

    n = len(out)
    theme_coverage = float(out["themes"].map(bool).mean()) if n else 0.0
    median_strength = float(out["policy_strength"].median()) if n else 0.0
    invalid_effective = int((out["effective_at"] < out["published_at"]).sum())
    gate_open = (
        n >= cfg.min_events
        and theme_coverage >= cfg.min_theme_coverage
        and median_strength >= cfg.min_strength_median
        and invalid_effective == 0
    )
    if n < cfg.min_events:
        reason = f"too_few_events_{n}_lt_{cfg.min_events}"
    elif theme_coverage < cfg.min_theme_coverage:
        reason = f"theme_coverage_{theme_coverage:.3f}_below_{cfg.min_theme_coverage:.3f}"
    elif median_strength < cfg.min_strength_median:
        reason = f"median_strength_{median_strength:.3f}_below_{cfg.min_strength_median:.3f}"
    elif invalid_effective:
        reason = f"effective_before_publish_{invalid_effective}"
    else:
        reason = "passed"

    coverage = {
        "n_events": int(n),
        "availability_mode": cfg.availability_mode,
        "theme_coverage": theme_coverage,
        "median_policy_strength": median_strength,
        "public_before_ingest_rate": float(
            (out["public_available_at"] <= out["ingested_at"]).mean()
        ) if n else 0.0,
        "rejected_no_date": int(rejected_no_date),
        "rejected_no_url": int(rejected_no_url),
        "duplicates_removed": int(duplicates_removed),
        "source_counts": out["source"].value_counts().to_dict(),
        "document_type_counts": out["document_type"].value_counts().to_dict(),
        "theme_counts": out["themes"].explode().dropna().value_counts().to_dict(),
        "sector_hint_counts": out["sectors_hint"].explode().dropna().value_counts().to_dict(),
        "gate": {"policy_events_usable_for_features": gate_open, "reason": reason},
    }
    validation = {
        "status": "passed" if gate_open else "failed",
        "n": int(n),
        "errors": [] if not invalid_effective else [
            f"{invalid_effective} rows have effective_at before published_at"
        ],
    }
    return PolicyEventResult(out, coverage, validation)


def _empty_result(cfg: PolicyEventConfig) -> PolicyEventResult:
    return PolicyEventResult(
        pd.DataFrame(columns=POLICY_EVENT_REQUIRED_COLUMNS),
        {
            "n_events": 0,
            "availability_mode": cfg.availability_mode,
            "theme_coverage": 0.0,
            "median_policy_strength": 0.0,
            "gate": {"policy_events_usable_for_features": False, "reason": "no_events"},
        },
        {"status": "failed", "n": 0, "errors": ["no_rows"]},
    )


class PolicyEventBuilder:
    def __init__(self, config: PolicyEventConfig | None = None) -> None:
        self.config = config or PolicyEventConfig()

    def build(self, raw: pd.DataFrame) -> PolicyEventResult:
        return build_policy_events(raw, config=self.config)

    def write(self, result: PolicyEventResult) -> PolicyEventResult:
        root = Path(self.config.output_root)
        silver = root / "silver" / "policy_events"
        silver.mkdir(parents=True, exist_ok=True)
        parquet = silver / "policy_events.parquet"
        coverage = silver / "coverage_report.json"
        validation = silver / "validation_report.json"
        result.frame.to_parquet(parquet, index=False)
        coverage.write_text(json.dumps(result.coverage, indent=2, default=str), encoding="utf-8")
        validation.write_text(json.dumps(result.validation, indent=2, default=str), encoding="utf-8")
        manifests = root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        manifest = manifests / "policy_events.json"
        manifest.write_text(
            json.dumps(
                {
                    "name": "policy_events", "rows": len(result.frame), "schema_version": 2,
                    "availability_mode": self.config.availability_mode,
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "policy_events": str(parquet), "coverage_report": str(coverage),
            "validation_report": str(validation), "manifest": str(manifest),
        }
        return result


def policy_events_for_features(
    events: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    if events is None or events.empty or manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate = ((payload.get("extra") or {}).get("coverage_report") or {}).get("gate") or {}
    required = {"event_id", "published_at", "available_at", "themes", "policy_strength"}
    if not gate.get("policy_events_usable_for_features") or not required.issubset(events.columns):
        return None
    return events.copy()
