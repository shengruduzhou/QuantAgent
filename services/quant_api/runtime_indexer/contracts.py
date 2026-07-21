from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from services.quant_api.config import ApiSettings, project_relative


MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_INLINE_HASH_BYTES = 64 * 1024 * 1024


class TrustClass(str, Enum):
    PRODUCTION_READY = "production_ready"
    PAPER_ONLY = "paper_only"
    RESEARCH_ONLY = "research_only"
    CONTAMINATED = "contaminated"
    UNCLASSIFIED = "unclassified"


class ValidationStatus(str, Enum):
    VERIFIED = "verified"
    DECLARED = "declared"
    UNVERIFIED = "unverified"
    INVALID = "invalid"


class FreshnessStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ArtifactCapability(str, Enum):
    METADATA = "metadata"
    PREVIEW = "preview"
    RESEARCH_DISPLAY = "research_display"
    PRODUCTION_DISPLAY = "production_display"
    PAPER_EXECUTION = "paper_execution"
    AUDIT_REPLAY = "audit_replay"


@dataclass(frozen=True)
class ArtifactContract:
    schemaVersion: str | None = None
    trustClass: str = TrustClass.UNCLASSIFIED.value
    validationStatus: str = ValidationStatus.UNVERIFIED.value
    freshnessStatus: str = FreshnessStatus.UNKNOWN.value
    staleReason: str | None = None
    sourceTime: str | None = None
    manifestPath: str | None = None
    contentHash: str | None = None
    declaredKind: str | None = None
    runId: str | None = None
    horizon: str | None = None
    producer: str | None = None
    qualityStatus: str | None = None
    dataAsOf: str | None = None
    rows: int | None = None
    dateStart: str | None = None
    dateEnd: str | None = None
    upstreamPaths: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=lambda: [ArtifactCapability.METADATA.value])
    contractIssues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_artifact_contract(
    path: Path,
    settings: ApiSettings,
    *,
    previewable: bool,
) -> ArtifactContract:
    """Resolve schema, trust and safe capabilities for one persisted artifact.

    Historical artifacts remain visible for research, but they are fail-closed:
    without a readable manifest and a verified content hash they never receive
    ``production_display`` or ``paper_execution`` capabilities.
    """

    capabilities = [ArtifactCapability.METADATA.value]
    if previewable:
        capabilities.append(ArtifactCapability.PREVIEW.value)

    manifest_path = _find_manifest(path, settings.runtime_root)
    if manifest_path is None:
        capabilities.append(ArtifactCapability.RESEARCH_DISPLAY.value)
        return ArtifactContract(capabilities=capabilities)

    logical_manifest_path = project_relative(settings, manifest_path)
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds the 5 MiB control-plane limit")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be a JSON object")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return ArtifactContract(
            validationStatus=ValidationStatus.INVALID.value,
            manifestPath=logical_manifest_path,
            capabilities=capabilities,
            contractIssues=[{
                "code": "manifest_invalid",
                "message": str(exc),
                "path": logical_manifest_path,
                "recoverable": True,
            }],
        )

    schema_version = _string_value(payload, "schema_version", "schemaVersion")
    source_time = _string_value(payload, "created_at", "created", "fetch_time", "generated_at")
    declared_kind = _string_value(payload, "artifact_type", "artifactType", "kind")
    run_id = _string_value(payload, "run_id", "runId")
    horizon = _string_value(payload, "horizon", "horizon_class", "horizonClass")
    producer = _producer(payload)
    quality_status = _quality_status(payload)
    data_as_of = _string_value(payload, "data_as_of", "dataAsOf", "as_of", "asOf")
    rows = _integer_value(payload, "row_count", "rowCount", "rows")
    date_start, date_end = _date_range(payload)
    upstream_paths = _upstream_paths(payload)
    trust_class = _trust_class(payload)
    expected_hash = _expected_hash(payload, path, manifest_path)
    issues: list[dict[str, Any]] = []
    content_hash: str | None = None

    if expected_hash and path.stat().st_size <= MAX_INLINE_HASH_BYTES:
        content_hash = _sha256_file(path)
        if content_hash.lower() != expected_hash.lower():
            issues.append({
                "code": "content_hash_mismatch",
                "message": "artifact content does not match its manifest hash",
                "path": project_relative(settings, path),
                "recoverable": False,
            })
            validation_status = ValidationStatus.INVALID
        else:
            validation_status = ValidationStatus.VERIFIED
    elif expected_hash:
        validation_status = ValidationStatus.DECLARED
        issues.append({
            "code": "hash_verification_deferred",
            "message": "artifact exceeds the inline hash verification limit",
            "path": project_relative(settings, path),
            "recoverable": True,
        })
    elif schema_version or trust_class is not TrustClass.UNCLASSIFIED:
        validation_status = ValidationStatus.DECLARED
    else:
        validation_status = ValidationStatus.UNVERIFIED

    freshness_status, stale_reason = _freshness(payload)
    if validation_status is not ValidationStatus.INVALID:
        capabilities.append(ArtifactCapability.RESEARCH_DISPLAY.value)
        capabilities.append(ArtifactCapability.AUDIT_REPLAY.value)

    if trust_class is TrustClass.PRODUCTION_READY:
        if validation_status is ValidationStatus.VERIFIED:
            capabilities.extend([
                ArtifactCapability.PRODUCTION_DISPLAY.value,
                ArtifactCapability.PAPER_EXECUTION.value,
            ])
        else:
            issues.append({
                "code": "production_trust_unverified",
                "message": "production trust requires a verified artifact hash",
                "path": logical_manifest_path,
                "recoverable": True,
            })
    elif trust_class is TrustClass.PAPER_ONLY and validation_status is ValidationStatus.VERIFIED:
        capabilities.append(ArtifactCapability.PAPER_EXECUTION.value)

    return ArtifactContract(
        schemaVersion=schema_version,
        trustClass=trust_class.value,
        validationStatus=validation_status.value,
        freshnessStatus=freshness_status.value,
        staleReason=stale_reason,
        sourceTime=source_time,
        manifestPath=logical_manifest_path,
        contentHash=content_hash,
        declaredKind=declared_kind,
        runId=run_id,
        horizon=horizon,
        producer=producer,
        qualityStatus=quality_status,
        dataAsOf=data_as_of,
        rows=rows,
        dateStart=date_start,
        dateEnd=date_end,
        upstreamPaths=upstream_paths,
        capabilities=list(dict.fromkeys(capabilities)),
        contractIssues=issues,
    )


def _find_manifest(path: Path, runtime_root: Path) -> Path | None:
    candidates = [
        path.with_name(f"{path.name}.manifest.json"),
        path.with_name(f"{path.stem}.manifest.json"),
    ]
    current = path.parent
    runtime = runtime_root.resolve()
    while current == runtime or runtime in current.resolve().parents:
        candidates.extend([
            current / "artifact_manifest.json",
            current / "run_manifest.json",
            current / "manifest.json",
        ])
        if current.resolve() == runtime:
            break
        current = current.parent
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file() and candidate.resolve() != path.resolve():
            return candidate
    return None


def _trust_class(payload: dict[str, Any]) -> TrustClass:
    values: list[Any] = [payload.get("trust_class"), payload.get("trustClass")]
    config_echo = payload.get("config_echo")
    if isinstance(config_echo, dict):
        trust = config_echo.get("trust")
        if isinstance(trust, dict):
            values.append(trust.get("class"))
    trust = payload.get("trust")
    if isinstance(trust, dict):
        values.append(trust.get("class"))
    if payload.get("production_ready") is True:
        values.append("production_ready")
    elif payload.get("production_ready") is False:
        values.append("research_only")
    values.extend([payload.get("status"), payload.get("verdict")])

    normalized = " ".join(str(value).strip().lower() for value in values if value not in (None, ""))
    if any(token in normalized for token in ("contaminated", "forensic")):
        return TrustClass.CONTAMINATED
    if "production_ready" in normalized or "production-ready" in normalized:
        return TrustClass.PRODUCTION_READY
    if "paper_only" in normalized or "paper-only" in normalized:
        return TrustClass.PAPER_ONLY
    if any(token in normalized for token in (
        "research_only", "research-only", "validation_only", "likely_overfit",
        "do_not_enable", "rejected", "failed",
    )):
        return TrustClass.RESEARCH_ONLY
    return TrustClass.UNCLASSIFIED


def _expected_hash(payload: dict[str, Any], path: Path, manifest_path: Path) -> str | None:
    direct = _string_value(payload, "content_sha256", "sha256")
    sidecar = manifest_path.name == f"{path.name}.manifest.json"
    output = payload.get("output")
    if sidecar or (isinstance(output, str) and Path(output).name == path.name):
        direct = _string_value(payload, "output_sha256") or direct
    if direct:
        return direct

    hashes = payload.get("content_hashes")
    if isinstance(hashes, dict):
        for key, value in hashes.items():
            if not isinstance(value, str):
                continue
            key_path = Path(str(key))
            if key_path.name == path.name:
                return value
    return None


def _freshness(payload: dict[str, Any]) -> tuple[FreshnessStatus, str | None]:
    expires = _string_value(payload, "expires_at", "valid_until")
    if not expires:
        return FreshnessStatus.UNKNOWN, None
    parsed = _parse_datetime(expires)
    if parsed is None:
        return FreshnessStatus.UNKNOWN, "manifest freshness timestamp is invalid"
    if parsed < datetime.now(timezone.utc):
        return FreshnessStatus.STALE, f"manifest expired at {expires}"
    return FreshnessStatus.CURRENT, None


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _string_value(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _integer_value(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value in (None, ""):
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        return result if result >= 0 else None
    return None


def _producer(payload: dict[str, Any]) -> str | None:
    direct = _string_value(payload, "producer", "generated_by", "generatedBy", "command_id", "commandId")
    if direct:
        return direct
    job = payload.get("job")
    if isinstance(job, dict):
        return _string_value(job, "command_id", "commandId", "type", "name")
    return None


def _quality_status(payload: dict[str, Any]) -> str | None:
    direct = _string_value(payload, "quality_status", "qualityStatus")
    if direct:
        return direct
    quality = payload.get("quality")
    if isinstance(quality, dict):
        return _string_value(quality, "status", "verdict")
    return None


def _date_range(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    start = _string_value(payload, "date_start", "dateStart", "start_date", "startDate")
    end = _string_value(payload, "date_end", "dateEnd", "end_date", "endDate")
    value = payload.get("date_range", payload.get("dateRange"))
    if isinstance(value, dict):
        start = start or _string_value(value, "start", "from", "min")
        end = end or _string_value(value, "end", "to", "max")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start = start or (str(value[0]) if value[0] not in (None, "") else None)
        end = end or (str(value[1]) if value[1] not in (None, "") else None)
    return start, end


def _upstream_paths(payload: dict[str, Any]) -> list[str]:
    """Return only safe, explicit repository-relative lineage references."""

    candidates: list[Any] = []
    for key in ("upstream_paths", "upstreamPaths", "source_paths", "sourcePaths", "inputs", "upstream"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
        elif value not in (None, ""):
            candidates.append(value)

    resolved: list[str] = []
    for value in candidates:
        if isinstance(value, dict):
            value = next((value.get(key) for key in ("path", "artifact", "output", "source") if value.get(key)), None)
        if not isinstance(value, str):
            continue
        normalized = value.strip().replace("\\", "/")
        if not normalized or normalized.startswith(("/", "~")) or "../" in normalized:
            continue
        if len(normalized) >= 2 and normalized[1] == ":":
            continue
        resolved.append(normalized.lstrip("./"))
    return list(dict.fromkeys(resolved))


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ArtifactCapability",
    "ArtifactContract",
    "FreshnessStatus",
    "TrustClass",
    "ValidationStatus",
    "resolve_artifact_contract",
]
