from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
import time
from services.quant_api.config import ApiSettings, project_relative, stable_id
from services.quant_api.runtime_indexer.contracts import resolve_artifact_contract
from services.quant_api.runtime_indexer.parsers import parser_for


CACHE_SCHEMA_VERSION = 3


@dataclass
class IndexedArtifact:
    id: str
    kind: str
    name: str
    path: str
    extension: str
    sizeBytes: int
    modifiedAt: str
    status: str = "ready"
    parser: str | None = None
    runId: str | None = None
    horizon: str | None = None
    rows: int | None = None
    dateStart: str | None = None
    dateEnd: str | None = None
    tags: list[str] = field(default_factory=list)
    schemaVersion: str | None = None
    trustClass: str = "unclassified"
    validationStatus: str = "unverified"
    freshnessStatus: str = "unknown"
    staleReason: str | None = None
    sourceTime: str | None = None
    manifestPath: str | None = None
    contentHash: str | None = None
    declaredKind: str | None = None
    kindSource: str = "path_heuristic"
    runIdSource: str | None = None
    producer: str | None = None
    qualityStatus: str | None = None
    dataAsOf: str | None = None
    upstreamPaths: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=lambda: ["metadata"])
    issues: list[dict] = field(default_factory=list)


class RuntimeIndexer:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.cache_path = settings.cache_root / "runtime_index.json"
        self._lock = RLock()
        self._artifacts: list[dict] = []
        self._indexed_at = 0.0

    def scan(self, *, force: bool = False) -> list[dict]:
        with self._lock:
            if not force and self._artifacts and time.time() - self._indexed_at < self.settings.index_ttl_seconds:
                return list(self._artifacts)
            if not force and self._load_cache_if_fresh():
                return list(self._artifacts)
            artifacts: list[dict] = []
            for root, dirs, files in os.walk(self.settings.runtime_root):
                root_path = Path(root)
                dirs[:] = [
                    name
                    for name in dirs
                    if name not in {"node_modules", "__pycache__"}
                    and not self._is_internal_path(root_path / name)
                    and not self._is_bulk_storage_path(root_path / name)
                ]
                for name in files:
                    path = root_path / name
                    if self._is_internal_path(path):
                        continue
                    try:
                        stat = path.stat()
                        relative = project_relative(self.settings, path)
                        parser_name, _ = parser_for(path)
                        contract = resolve_artifact_contract(
                            path,
                            self.settings,
                            previewable=True,
                        )
                        contract_payload = contract.to_dict()
                        fallback_kind, tags = _classify(relative, path.name)
                        declared_kind = _canonical_declared_kind(contract_payload["declaredKind"])
                        kind = declared_kind or fallback_kind
                        declared_run_id = contract_payload["runId"]
                        fallback_run_id = _run_id(relative)
                        artifact = IndexedArtifact(
                            id=stable_id("artifact", relative),
                            kind=kind,
                            name=path.name,
                            path=relative,
                            extension=path.suffix.lower(),
                            sizeBytes=int(stat.st_size),
                            modifiedAt=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                            status=(
                                "error" if contract.validationStatus == "invalid"
                                else "partial" if contract.freshnessStatus == "stale"
                                else "ready"
                            ),
                            parser=parser_name,
                            runId=declared_run_id or fallback_run_id,
                            horizon=contract_payload["horizon"] or _horizon(relative),
                            rows=contract_payload["rows"],
                            dateStart=contract_payload["dateStart"],
                            dateEnd=contract_payload["dateEnd"],
                            tags=tags,
                            schemaVersion=contract_payload["schemaVersion"],
                            trustClass=contract_payload["trustClass"],
                            validationStatus=contract_payload["validationStatus"],
                            freshnessStatus=contract_payload["freshnessStatus"],
                            staleReason=contract_payload["staleReason"],
                            sourceTime=contract_payload["sourceTime"],
                            manifestPath=contract_payload["manifestPath"],
                            contentHash=contract_payload["contentHash"],
                            declaredKind=contract_payload["declaredKind"],
                            kindSource="manifest" if declared_kind else "path_heuristic",
                            runIdSource="manifest" if declared_run_id else "path_heuristic" if fallback_run_id else None,
                            producer=contract_payload["producer"],
                            qualityStatus=contract_payload["qualityStatus"],
                            dataAsOf=contract_payload["dataAsOf"],
                            upstreamPaths=contract_payload["upstreamPaths"],
                            capabilities=contract_payload["capabilities"],
                            issues=contract_payload["contractIssues"],
                        )
                        artifacts.append(asdict(artifact))
                    except (OSError, ValueError) as exc:
                        relative = path.as_posix()
                        artifacts.append(asdict(IndexedArtifact(
                            id=stable_id("artifact", relative),
                            kind="unknown",
                            name=path.name,
                            path=relative,
                            extension=path.suffix.lower(),
                            sizeBytes=0,
                            modifiedAt="",
                            status="error",
                            issues=[{"code": "index_error", "message": str(exc), "recoverable": True}],
                        )))
            artifacts.sort(key=lambda item: item["modifiedAt"], reverse=True)
            self._artifacts = artifacts
            self._indexed_at = time.time()
            self._write_cache()
            return list(self._artifacts)

    def filter(
        self,
        *,
        kind: str | None = None,
        query: str | None = None,
        extension: str | None = None,
        run_id: str | None = None,
        horizon: str | None = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        strategy: str | None = None,
        model: str | None = None,
        symbol: str | None = None,
        trust_class: str | None = None,
        validation_status: str | None = None,
        freshness_status: str | None = None,
        capability: str | None = None,
        sort_by: str = "modifiedAt",
        sort_direction: str = "desc",
    ) -> list[dict]:
        items = self.scan()
        if kind:
            items = [item for item in items if item["kind"] == kind]
        if extension:
            normalized = extension if extension.startswith(".") else f".{extension}"
            items = [item for item in items if item["extension"] == normalized]
        if query:
            needle = query.lower()
            items = [item for item in items if needle in item["path"].lower()]
        if run_id:
            items = [item for item in items if str(item.get("runId") or "") == run_id]
        if horizon:
            items = [item for item in items if str(item.get("horizon") or "") == horizon]
        if trust_class:
            items = [item for item in items if str(item.get("trustClass") or "") == trust_class]
        if validation_status:
            items = [item for item in items if str(item.get("validationStatus") or "") == validation_status]
        if freshness_status:
            items = [item for item in items if str(item.get("freshnessStatus") or "") == freshness_status]
        if capability:
            items = [item for item in items if capability in item.get("capabilities", [])]
        if modified_after:
            items = [item for item in items if item.get("modifiedAt", "") >= modified_after]
        if modified_before:
            items = [item for item in items if item.get("modifiedAt", "") <= modified_before]
        for value in (strategy, model, symbol):
            if value:
                needle = value.lower()
                items = [
                    item for item in items
                    if needle in " ".join([
                        str(item.get("path") or ""),
                        str(item.get("runId") or ""),
                        str(item.get("horizon") or ""),
                        " ".join(str(tag) for tag in item.get("tags", [])),
                    ]).lower()
                ]
        sort_fields = {"modifiedAt", "sizeBytes", "name", "kind", "runId", "trustClass", "validationStatus"}
        resolved_sort = sort_by if sort_by in sort_fields else "modifiedAt"
        reverse = sort_direction.lower() != "asc"
        return sorted(
            items,
            key=lambda item: (item.get(resolved_sort) is not None, item.get(resolved_sort) or ""),
            reverse=reverse,
        )

    def get(self, artifact_id: str) -> dict | None:
        return next((item for item in self.scan() if item["id"] == artifact_id), None)

    def invalidate(self) -> None:
        with self._lock:
            self._artifacts = []
            self._indexed_at = 0.0
            if self.cache_path.exists():
                self.cache_path.unlink()

    def stats(self) -> dict:
        items = self.scan()
        by_kind: dict[str, int] = {}
        by_trust: dict[str, int] = {}
        by_validation: dict[str, int] = {}
        by_freshness: dict[str, int] = {}
        by_capability: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total_size = 0
        for item in items:
            by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
            trust_class = str(item.get("trustClass") or "unclassified")
            validation_status = str(item.get("validationStatus") or "unverified")
            by_trust[trust_class] = by_trust.get(trust_class, 0) + 1
            by_validation[validation_status] = by_validation.get(validation_status, 0) + 1
            freshness_status = str(item.get("freshnessStatus") or "unknown")
            by_freshness[freshness_status] = by_freshness.get(freshness_status, 0) + 1
            status = str(item.get("status") or "unavailable")
            by_status[status] = by_status.get(status, 0) + 1
            for capability in item.get("capabilities", []):
                by_capability[capability] = by_capability.get(capability, 0) + 1
            total_size += int(item["sizeBytes"])
        return {
            "artifactCount": len(items),
            "totalSizeBytes": total_size,
            "byKind": by_kind,
            "byTrust": by_trust,
            "byValidation": by_validation,
            "byFreshness": by_freshness,
            "byCapability": by_capability,
            "byStatus": by_status,
            "runCount": len(self.runs(items=items)),
            "manifestCoverage": (
                sum(bool(item.get("manifestPath")) for item in items) / len(items)
                if items else 0.0
            ),
            "indexedAt": datetime.fromtimestamp(self._indexed_at, tz=timezone.utc).isoformat(),
        }

    def runs(self, *, items: list[dict] | None = None) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for item in items if items is not None else self.scan():
            run_id = item.get("runId")
            if run_id:
                grouped.setdefault(str(run_id), []).append(item)

        runs: list[dict] = []
        for run_id, artifacts in grouped.items():
            runs.append({
                "id": run_id,
                "artifactCount": len(artifacts),
                "totalSizeBytes": sum(int(item.get("sizeBytes") or 0) for item in artifacts),
                "kinds": sorted({str(item.get("kind") or "unknown") for item in artifacts}),
                "trustClasses": sorted({str(item.get("trustClass") or "unclassified") for item in artifacts}),
                "validationStatuses": sorted({str(item.get("validationStatus") or "unverified") for item in artifacts}),
                "capabilities": sorted({cap for item in artifacts for cap in item.get("capabilities", [])}),
                "issueCount": sum(len(item.get("issues", [])) for item in artifacts),
                "latestModifiedAt": max(str(item.get("modifiedAt") or "") for item in artifacts),
                "dateStart": min((str(item["dateStart"]) for item in artifacts if item.get("dateStart")), default=None),
                "dateEnd": max((str(item["dateEnd"]) for item in artifacts if item.get("dateEnd")), default=None),
            })
        return sorted(runs, key=lambda item: item["latestModifiedAt"], reverse=True)

    def catalog(self) -> dict:
        items = self.scan()
        return {
            "summary": self.stats(),
            "runs": self.runs(items=items),
            "roots": [project_relative(self.settings, self.settings.runtime_root)],
        }

    def lineage(self, artifact_id: str) -> dict | None:
        items = self.scan()
        artifact = next((item for item in items if item["id"] == artifact_id), None)
        if artifact is None:
            return None

        by_path = {str(item["path"]): item for item in items}
        upstream = []
        for reference in artifact.get("upstreamPaths", []):
            match = by_path.get(reference)
            if match is None:
                candidates = [item for path, item in by_path.items() if path.endswith(f"/{reference}")]
                match = candidates[0] if len(candidates) == 1 else None
            upstream.append({"reference": reference, "artifact": match})

        downstream = [
            item for item in items
            if artifact["path"] in item.get("upstreamPaths", [])
            or any(artifact["path"].endswith(f"/{reference}") for reference in item.get("upstreamPaths", []))
        ]
        unresolved = [edge["reference"] for edge in upstream if edge["artifact"] is None]
        return {
            "artifact": artifact,
            "upstream": upstream,
            "downstream": downstream,
            "status": "complete" if upstream and not unresolved else "partial" if upstream else "undeclared",
            "issues": ([{
                "code": "lineage_reference_unresolved",
                "message": f"{len(unresolved)} declared upstream reference(s) are not indexed",
                "recoverable": True,
            }] if unresolved else []),
        }

    def _load_cache_if_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                return False
            indexed_at = float(payload.get("indexed_at_epoch", 0.0))
            if time.time() - indexed_at >= self.settings.index_ttl_seconds:
                return False
            self._artifacts = list(payload.get("artifacts", []))
            self._indexed_at = indexed_at
            return True
        except (OSError, ValueError, TypeError):
            return False

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "indexed_at_epoch": self._indexed_at,
            "artifacts": self._artifacts,
        }
        temp = self.cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.cache_path)

    def _is_internal_path(self, path: Path) -> bool:
        resolved = path.resolve()
        for internal_root in (self.settings.cache_root.resolve(), self.settings.jobs_root.resolve()):
            if resolved == internal_root or internal_root in resolved.parents:
                return True
        return False

    def _is_bulk_storage_path(self, path: Path) -> bool:
        relative = path.resolve().relative_to(self.settings.runtime_root.resolve()).as_posix()
        if path.name in {"feature_cache", "outcome_cache"}:
            return True
        return relative in {
            "data/raw/qlib/cn_data",
            "data/raw/qlib/cn_data_1min",
        }


def _classify(relative: str, name: str) -> tuple[str, list[str]]:
    lower = relative.lower()
    tags: list[str] = []
    if "manifest" in name.lower():
        return "manifest", tags
    if "/backtest/" in lower or name in {"nav.csv", "pnl.csv", "realized_trades.csv", "failed_orders.csv"}:
        return "backtest", tags
    if name.endswith((".pt", ".pth", ".joblib", ".zip")) or "/models/" in lower:
        return "model", tags
    if "prediction" in name:
        return "prediction", tags
    if "target_weight" in name or name.startswith("targets_"):
        return "target_weights", tags
    if "factor" in lower or "ic" in name.lower():
        return "factor", tags
    if "stock_pool" in lower or "selection" in lower:
        return "selection", tags
    if "risk" in lower or "failed_order" in name:
        return "risk", tags
    if "do_t" in lower or "dot_" in lower or "intraday_dot" in lower:
        return "do_t", tags
    if "/logs/" in lower or path_suffix(name) in {".log", ".jsonl"}:
        return "log", tags
    if "/data/" in lower:
        return "dataset", tags
    if "/reports/" in lower or path_suffix(name) in {".md", ".html"}:
        return "report", tags
    return "unknown", tags


def path_suffix(name: str) -> str:
    return Path(name).suffix.lower()


def _canonical_declared_kind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    direct = {
        "backtest", "model", "prediction", "target_weights", "factor", "selection",
        "risk", "do_t", "log", "dataset", "report", "manifest",
    }
    if normalized in direct:
        return normalized
    prefixes = {
        "backtest": "backtest",
        "model": "model",
        "prediction": "prediction",
        "factor": "factor",
        "selection": "selection",
        "risk": "risk",
        "dataset": "dataset",
        "report": "report",
        "target_weight": "target_weights",
    }
    return next((kind for prefix, kind in prefixes.items() if normalized.startswith(prefix)), None)


def _run_id(relative: str) -> str | None:
    parts = Path(relative).parts
    for marker in ("deep", "pipeline", "v89_closed_loop"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    if "backtest" in parts:
        index = parts.index("backtest")
        return parts[index - 1] if index > 0 else None
    return None


def _horizon(relative: str) -> str | None:
    for value in ("short_5d", "mid_5d_30d", "long_30d_120d"):
        if value in relative:
            return value
    return None
