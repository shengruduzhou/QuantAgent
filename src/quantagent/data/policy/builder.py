"""Canonical policy-event builder with explicit publication and ingestion time.

The original implementation overloaded ``available_at`` with the later of the
announcement timestamp and the crawler fetch timestamp.  That is safe for a
live replay, but it destroys historical point-in-time semantics when an old
policy document is backfilled years later.  This module keeps the clocks
separate:

``published_at``
    Timestamp at which the document became public.
``public_available_at``
    First timestamp at which a market participant could have observed it.
``ingested_at``
    Timestamp at which QuantAgent fetched the document.
``available_at``
    Feature-availability timestamp selected by ``availability_mode``.

Historical research should normally use ``availability_mode='public'``.  A
strict live replay can use ``availability_mode='ingested'`` so a document is
not consumed before the local pipeline actually received it.
"""

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
    "event_id",
    "source",
    "source_authority",
    "document_type",
    "url",
    "title",
    "body_summary",
    "announced_at",  # backward-compatible alias of published_at
    "published_at",
    "effective_at",
    "public_available_at",
    "ingested_at",
    "available_at",
    "revised_at",
    "superseded_at",
    "jurisdiction",
    "funding_amount_cny",
    "themes",
    "sectors_hint",
    "policy_strength",
    "raw_hash",
    "fetched_at",  # backward-compatible alias of ingested_at
    "source_version",
)

VALID_SOURCES: tuple[str, ...] = (
    "csrc",
    "pboc",
    "mof",
    "ndrc",
    "state_council",
    "sse",
    "szse",
    "bse",
    "stats",
    "local_government",
    "manual_local_import",
)

SOURCE_AUTHORITY: dict[str, float] = {
    "state_council": 1.00,
    "pboc": 0.95,
    "mof": 0.95,
    "ndrc": 0.92,
    "csrc": 0.92,
    "sse": 0.85,
    "szse": 0.85,
    "bse": 0.85,
    "stats": 0.82,
    "local_government": 0.75,
    "manual_local_import": 0.50,
}

DOCUMENT_TYPE_WEIGHT: dict[str, float] = {
    "law": 1.00,
    "administrative_regulation": 0.95,
    "measure": 0.90,
    "rule": 0.88,
    "opinion": 0.82,
    "notice": 0.78,
    "plan": 0.76,
    "meeting": 0.65,
    "speech": 0.55,
    "other": 0.50,
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
        "法律": "law",
        "行政法规": "administrative_regulation",
        "办法": "measure",
        "规定": "rule",
        "意见": "opinion",
        "通知": "notice",
        "规划": "plan",
        "会议": "meeting",
        "讲话": "speech",
    }
    value = aliases.get(value, value)
    return value if value in DOCUMENT_TYPE_WEIGHT else "other"


def _ensure_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


def _content_hash(title: str, body: str, url: str) -> str:
    payload = f"{title.strip()}\n{body.strip()}\n{url.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_id(source: str, url: str, published_at: pd.Timestamp, title: str) -> str:
    raw = f"{source}||{url}||{published_at.isoformat()}||{title.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _numeric_or_nan(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else float("nan")


def _derive_strength(
    *,
    source: str,
    document_type: str,
    tag_strength: float,
    funding_amount_cny: float,
) -> float:
    authority = SOURCE_AUTHORITY.get(source, 0.50)
    document = DOCUMENT_TYPE_WEIGHT.get(document_type, 0.50)
    funding_bonus = 0.0
    if pd.notna(funding_amount_cny) and funding_amount_cny > 0:
        # Log-scaled bonus: explicit funding matters, but cannot dominate.
        funding_bonus = min(0.12, 0.02 * max(0.0, len(str(int(funding_amount_cny))) - 8))
    score = 0.45 * float(tag_strength) + 0.35 * authority + 0.20 * document + funding_bonus
    return float(min(1.0, max(0.0, score)))


def build_policy_events(
    raw: pd.DataFrame,
    *,
    config: PolicyEventConfig | None = None,
) -> PolicyEventResult:
    """Normalise raw policy rows into an auditable PIT-safe silver product."""
    cfg = config or PolicyEventConfig()
    if raw is None or raw.empty:
        return _empty_result(cfg)

    required = {"source", "title"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw policy frame missing required columns: {sorted(missing)}")
    if "published_at" not in raw.columns and "announced_at" not in raw.columns:
        raise ValueError("raw policy frame requires published_at or announced_at")

    frame = raw.copy().reset_index(drop=False).rename(columns={"index": "_source_row_id"})
    frame["source"] = frame["source"].map(_normalise_source)
    frame["source_authority"] = frame["source"].map(SOURCE_AUTHORITY).fillna(0.50)
    frame["document_type"] = frame.get("document_type", "other").map(_normalise_document_type)
    frame["url"] = frame.get("url", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["body_summary"] = frame.get(
        "body_summary", pd.Series("", index=frame.index)
    ).fillna("").astype(str)

    published_source = frame["published_at"] if "published_at" in frame.columns else frame["announced_at"]
    frame["published_at"] = published_source.map(_coerce_timestamp)
    frame["announced_at"] = frame["published_at"]
    frame["effective_at"] = frame.get(
        "effective_at", frame["published_at"]
    ).map(_coerce_timestamp)
    frame["effective_at"] = frame["effective_at"].fillna(frame["published_at"])

    public_source = frame.get("public_available_at", frame["published_at"])
    frame["public_available_at"] = public_source.map(_coerce_timestamp)
    frame["public_available_at"] = frame["public_available_at"].fillna(frame["published_at"])

    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    ingest_source = frame.get(
        "ingested_at",
        frame.get("fetched_at", pd.Series(now, index=frame.index)),
    )
    frame["ingested_at"] = ingest_source.map(_coerce_timestamp).fillna(now)
    frame["fetched_at"] = frame["ingested_at"]

    frame["revised_at"] = frame.get(
        "revised_at", pd.Series(pd.NaT, index=frame.index)
    ).map(_coerce_timestamp)
    frame["superseded_at"] = frame.get(
        "superseded_at", pd.Series(pd.NaT, index=frame.index)
    ).map(_coerce_timestamp)
    frame["jurisdiction"] = frame.get(
        "jurisdiction", pd.Series("national", index=frame.index)
    ).fillna("national").astype(str)
    frame["funding_amount_cny"] = frame.get(
        "funding_amount_cny", pd.Series(float("nan"), index=frame.index)
    ).map(_numeric_or_nan)

    if cfg.availability_mode == "public":
        frame["available_at"] = frame["public_available_at"]
    elif cfg.availability_mode == "ingested":
        frame["available_at"] = frame[["public_available_at", "ingested_at"]].max(axis=1)
    else:  # pragma: no cover - dataclass typing is not runtime enforcement
        raise ValueError(f"unsupported availability_mode: {cfg.availability_mode}")

    before = len(frame)
    frame = frame[frame["published_at"].notna()].copy()
    rejected_no_date = before - len(frame)
    if cfg.require_official_url:
        before_url = len(frame)
        frame = frame[frame["url"].str.startswith(("http://", "https://"))].copy()
        rejected_no_url = before_url - len(frame)
    else:
        rejected_no_url = 0

    frame["event_id"] = frame.apply(
        lambda row: _event_id(row["source"], row["url"], row["published_at"], row["title"]),
        axis=1,
    )
    frame["raw_hash"] = frame.apply(
        lambda row: _content_hash(row["title"], row["body_summary"], row["url"]),
        axis=1,
    )

    tags = frame.apply(lambda row: tag_policy_event(row["title"], row["body_summary"]), axis=1)
    frame["themes"] = [tag["themes"] for tag in tags]
    frame["sectors_hint"] = [tag["sectors_hint"] for tag in tags]
    tag_strengths = [float(tag["policy_strength"]) for tag in tags]

    # Overrides are read from the filtered frame itself.  Keeping _source_row_id
    # avoids the old reset-index/raw.at mismatch after invalid rows are removed.
    if "themes_override" in frame.columns:
        for idx, value in frame["themes_override"].items():
            override = _ensure_list(value)
            if override:
                frame.at[idx, "themes"] = override
    if "sectors_hint_override" in frame.columns:
        for idx, value in frame["sectors_hint_override"].items():
            override = _ensure_list(value)
            if override:
                frame.at[idx, "sectors_hint"] = override

    frame["policy_strength"] = [
        _derive_strength(
            source=source,
            document_type=document_type,
            tag_strength=tag_strength,
            funding_amount_cny=funding,
        )
        for source, document_type, tag_strength, funding in zip(
            frame["source"], frame["document_type"], tag_strengths, frame["funding_amount_cny"]
        )
    ]
    frame["source_version"] = cfg.source_version

    before_dedup = len(frame)
    frame = frame.sort_values(["published_at", "ingested_at", "event_id"])
    frame = frame.drop_duplicates(subset=["event_id"], keep="first")
    duplicates_removed = before_dedup - len(frame)

    out = frame[list(POLICY_EVENT_REQUIRED_COLUMNS)].reset_index(drop=True)
    n = int(len(out))
    theme_coverage = float(out["themes"].map(bool).mean()) if n else 0.0
    median_strength = float(out["policy_strength"].median()) if n else 0.0
    public_before_ingest_rate = float(
        (out["public_available_at"] <= out["ingested_at"]).mean()
    ) if n else 0.0
    invalid_effective_before_publish = int(
        (out["effective_at"] < out["published_at"]).sum()
    )

    gate_open = (
        n >= cfg.min_events
        and theme_coverage >= cfg.min_theme_coverage
        and median_strength >= cfg.min_strength_median
        and invalid_effective_before_publish == 0
    )
    reason = "passed" if gate_open else _gate_reason(
        n=n,
        theme_coverage=theme_coverage,
        median_strength=median_strength,
        invalid_effective_before_publish=invalid_effective_before_publish,
        cfg=cfg,
    )

    coverage = {
        "n_events": n,
        "availability_mode": cfg.availability_mode,
        "theme_coverage": theme_coverage,
        "median_policy_strength": median_strength,
        "public_before_ingest_rate": public_before_ingest_rate,
        "rejected_no_date": int(rejected_no_date),
        "rejected_no_url": int(rejected_no_url),
        "duplicates_removed": int(duplicates_removed),
        "source_counts": out["source"].value_counts().to_dict(),
        "document_type_counts": out["document_type"].value_counts().to_dict(),
        "theme_counts": out["themes"].explode().dropna().value_counts().to_dict(),
        "sector_hint_counts": out["sectors_hint"].explode().dropna().value_counts().to_dict(),
        "gate": {
            "policy_events_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {
        "status": "passed" if gate_open else "failed",
        "n": n,
        "errors": [] if invalid_effective_before_publish == 0 else [
            f"{invalid_effective_before_publish} rows have effective_at before published_at"
        ],
    }
    return PolicyEventResult(frame=out, coverage=coverage, validation=validation)


def _gate_reason(
    *,
    n: int,
    theme_coverage: float,
    median_strength: float,
    invalid_effective_before_publish: int,
    cfg: PolicyEventConfig,
) -> str:
    if n < cfg.min_events:
        return f"too_few_events_{n}_lt_{cfg.min_events}"
    if theme_coverage < cfg.min_theme_coverage:
        return f"theme_coverage_{theme_coverage:.3f}_below_{cfg.min_theme_coverage:.3f}"
    if median_strength < cfg.min_strength_median:
        return f"median_strength_{median_strength:.3f}_below_{cfg.min_strength_median:.3f}"
    if invalid_effective_before_publish:
        return f"effective_before_publish_{invalid_effective_before_publish}"
    return "unknown"


def _empty_result(cfg: PolicyEventConfig) -> PolicyEventResult:
    return PolicyEventResult(
        frame=pd.DataFrame(columns=list(POLICY_EVENT_REQUIRED_COLUMNS)),
        coverage={
            "n_events": 0,
            "availability_mode": cfg.availability_mode,
            "theme_coverage": 0.0,
            "median_policy_strength": 0.0,
            "gate": {
                "policy_events_usable_for_features": False,
                "reason": "no_events",
            },
        },
        validation={"status": "failed", "n": 0, "errors": ["no_rows"]},
    )


class PolicyEventBuilder:
    def __init__(self, config: PolicyEventConfig | None = None) -> None:
        self.config = config or PolicyEventConfig()

    def build(self, raw: pd.DataFrame) -> PolicyEventResult:
        return build_policy_events(raw, config=self.config)

    def write(self, result: PolicyEventResult) -> PolicyEventResult:
        root = Path(self.config.output_root)
        silver_dir = root / "silver" / "policy_events"
        silver_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = silver_dir / "policy_events.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        coverage_path = silver_dir / "coverage_report.json"
        validation_path = silver_dir / "validation_report.json"
        coverage_path.write_text(json.dumps(result.coverage, indent=2, default=str), encoding="utf-8")
        validation_path.write_text(json.dumps(result.validation, indent=2, default=str), encoding="utf-8")

        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "policy_events.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "policy_events",
                    "rows": int(len(result.frame)),
                    "schema_version": 2,
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
            "policy_events": str(parquet_path),
            "coverage_report": str(coverage_path),
            "validation_report": str(validation_path),
            "manifest": str(manifest_path),
        }
        return result


def policy_events_for_features(
    events: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    """Return policy events only when the manifest gate and schema are valid."""
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
    if not gate.get("policy_events_usable_for_features"):
        return None
    required = {"event_id", "published_at", "available_at", "themes", "policy_strength"}
    if not required.issubset(events.columns):
        return None
    return events.copy()
