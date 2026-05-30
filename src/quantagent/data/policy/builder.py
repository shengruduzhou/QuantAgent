"""PolicyEventBuilder — produces silver/policy_events.parquet + manifest.

Contract:

* Input: a frame of *raw* policy rows with at least ``source``,
  ``announced_at``, ``title``, and a URL.  Optional: ``effective_at``,
  ``body_summary``.  Any rows missing one of the required columns are
  rejected.
* Output: a normalised parquet with the canonical columns listed in
  :data:`POLICY_EVENT_REQUIRED_COLUMNS`, plus a JSON coverage_report
  and a manifest gate (`policy_events_usable_for_features`).

The builder does not crawl. Fetching live data is the job of a
separate ingestion script (``scripts/fetch_policy_events.py`` in
spec); the builder *only* normalises + validates + writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.policy.theme_tagger import (
    POLICY_THEMES,
    SECTOR_KEYWORDS,
    tag_policy_event,
)


POLICY_EVENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "source",
    "url",
    "announced_at",
    "effective_at",
    "available_at",
    "title",
    "body_summary",
    "themes",
    "sectors_hint",
    "policy_strength",
    "fetched_at",
    "source_version",
)


VALID_SOURCES: tuple[str, ...] = (
    "csrc", "pboc", "mof", "ndrc", "state_council", "manual_local_import",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyEventConfig:
    source_version: str = "unknown"
    output_root: str | Path = "runtime/data/v7"
    # Coverage gate
    min_theme_coverage: float = 0.50  # ≥50% rows must have at least one theme
    min_strength_median: float = 0.30  # median policy_strength to "open" the gate
    min_events: int = 5  # below this, the gate stays closed regardless


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PolicyEventResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_event_id(source: str, url: str, announced_at: str) -> str:
    raw = f"{source}||{url}||{announced_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    try:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    except (AttributeError, TypeError):
        return ts


def _normalise_source(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in VALID_SOURCES:
        return s
    return "manual_local_import"


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        # JSON-encoded list or comma-separated
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_policy_events(
    raw: pd.DataFrame,
    *,
    config: PolicyEventConfig | None = None,
) -> PolicyEventResult:
    """Normalise a raw policy frame into the canonical silver schema."""
    cfg = config or PolicyEventConfig()
    if raw is None or raw.empty:
        return _empty_result(cfg)

    required_in = {"source", "announced_at", "title"}
    missing = required_in - set(raw.columns)
    if missing:
        raise ValueError(
            f"raw policy frame missing required columns: {sorted(missing)}"
        )

    frame = raw.copy()
    frame["source"] = frame["source"].map(_normalise_source)
    frame["url"] = frame.get("url", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["body_summary"] = (
        frame.get("body_summary", pd.Series([""] * len(frame))).fillna("").astype(str)
    )

    frame["announced_at"] = frame["announced_at"].map(_coerce_timestamp)
    if "effective_at" in frame.columns:
        frame["effective_at"] = frame["effective_at"].map(_coerce_timestamp)
    else:
        frame["effective_at"] = frame["announced_at"]
    frame["effective_at"] = frame["effective_at"].fillna(frame["announced_at"])

    # available_at = max(announced_at, fetched_at) — we never see a row
    # before the fetcher actually pulled it. If fetched_at missing, fall
    # back to announced_at.
    fetched_default = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    if "fetched_at" in frame.columns:
        frame["fetched_at"] = frame["fetched_at"].map(_coerce_timestamp).fillna(fetched_default)
    else:
        frame["fetched_at"] = fetched_default
    frame["available_at"] = frame[["announced_at", "fetched_at"]].max(axis=1)

    # Drop rows with unparseable announced_at
    before = len(frame)
    frame = frame[frame["announced_at"].notna()].reset_index(drop=True)
    rejected_no_date = before - len(frame)

    # event_id: stable hash for deduplication
    frame["event_id"] = frame.apply(
        lambda r: _compute_event_id(r["source"], r["url"], r["announced_at"].isoformat()),
        axis=1,
    )

    # Theme + sector + strength tagging
    tags = frame.apply(
        lambda r: tag_policy_event(r["title"], r.get("body_summary", "")),
        axis=1,
    )
    frame["themes"] = [t["themes"] for t in tags]
    frame["sectors_hint"] = [t["sectors_hint"] for t in tags]
    frame["policy_strength"] = [t["policy_strength"] for t in tags]

    # Allow caller to override themes/sectors_hint if supplied. We assign
    # cell-by-cell because pandas .loc unpacks lists when bulk-assigning
    # object columns.
    if "themes_override" in raw.columns:
        for idx in frame.index:
            override = _ensure_list(raw.at[idx, "themes_override"])
            if override:
                frame.at[idx, "themes"] = override
    if "sectors_hint_override" in raw.columns:
        for idx in frame.index:
            override = _ensure_list(raw.at[idx, "sectors_hint_override"])
            if override:
                frame.at[idx, "sectors_hint"] = override

    # Dedup by event_id (first-write-wins)
    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(frame)

    frame["source_version"] = cfg.source_version

    # Canonical column ordering
    out = frame[list(POLICY_EVENT_REQUIRED_COLUMNS)].copy()

    # Coverage report
    n = int(len(out))
    n_tagged = int(out["themes"].map(lambda l: bool(l)).sum())
    median_strength = float(out["policy_strength"].median()) if n else 0.0
    theme_coverage = float(n_tagged / n) if n else 0.0

    sector_counts = (
        out["sectors_hint"]
        .explode()
        .dropna()
        .value_counts()
        .to_dict()
    )
    theme_counts = (
        out["themes"].explode().dropna().value_counts().to_dict()
    )

    gate_open = (
        n >= cfg.min_events
        and theme_coverage >= cfg.min_theme_coverage
        and median_strength >= cfg.min_strength_median
    )
    reason = "passed" if gate_open else _gate_reason(
        n, theme_coverage, median_strength, cfg
    )

    coverage = {
        "n_events": n,
        "n_tagged_at_least_one_theme": n_tagged,
        "theme_coverage": theme_coverage,
        "median_policy_strength": median_strength,
        "rejected_no_date": int(rejected_no_date),
        "duplicates_removed": int(duplicates_removed),
        "source_counts": out["source"].value_counts().to_dict(),
        "theme_counts": theme_counts,
        "sector_hint_counts": sector_counts,
        "gate": {
            "policy_events_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {"status": "passed", "n": n, "errors": []}

    return PolicyEventResult(frame=out, coverage=coverage, validation=validation)


def _empty_result(cfg: PolicyEventConfig) -> PolicyEventResult:
    empty_frame = pd.DataFrame(columns=list(POLICY_EVENT_REQUIRED_COLUMNS))
    coverage = {
        "n_events": 0,
        "n_tagged_at_least_one_theme": 0,
        "theme_coverage": 0.0,
        "median_policy_strength": 0.0,
        "gate": {
            "policy_events_usable_for_features": False,
            "reason": "no_events",
        },
    }
    return PolicyEventResult(frame=empty_frame, coverage=coverage, validation={"status": "passed", "n": 0, "errors": []})


def _gate_reason(
    n: int,
    theme_coverage: float,
    median_strength: float,
    cfg: PolicyEventConfig,
) -> str:
    if n < cfg.min_events:
        return f"too_few_events_{n}_lt_{cfg.min_events}"
    if theme_coverage < cfg.min_theme_coverage:
        return f"theme_coverage_{theme_coverage:.3f}_below_{cfg.min_theme_coverage:.3f}"
    if median_strength < cfg.min_strength_median:
        return f"median_strength_{median_strength:.3f}_below_{cfg.min_strength_median:.3f}"
    return "unknown"


# ---------------------------------------------------------------------------
# Builder with writer
# ---------------------------------------------------------------------------

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
        (silver_dir / "coverage_report.json").write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        (silver_dir / "validation_report.json").write_text(
            json.dumps(result.validation, indent=2, default=str), encoding="utf-8"
        )
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "policy_events.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "policy_events",
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
            "policy_events": str(parquet_path),
            "coverage_report": str(silver_dir / "coverage_report.json"),
            "validation_report": str(silver_dir / "validation_report.json"),
            "manifest": str(manifest_path),
        }
        return result


# ---------------------------------------------------------------------------
# Overlay helper for downstream consumers
# ---------------------------------------------------------------------------

def policy_events_for_features(
    events: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    """Return ``events`` only if the manifest gate is open.

    Downstream feature engineering (Stage 4.2 time-lag model and the
    training feature panel) must call this — never read the parquet
    directly — so the gate cannot be bypassed.
    """
    if events is None or len(events) == 0:
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
    if not gate.get("policy_events_usable_for_features"):
        return None
    return events
