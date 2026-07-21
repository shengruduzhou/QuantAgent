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


CACHE_SCHEMA_VERSION = 2


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
                        kind, tags = _classify(relative, path.name)
                        contract = resolve_artifact_contract(
                            path,
                            self.settings,
                            previewable=True,
                        )
                        contract_payload = contract.to_dict()
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
                            runId=_run_id(relative),
                            horizon=_horizon(relative),
                            tags=tags,
                            schemaVersion=contract_payload["schemaVersion"],
                            trustClass=contract_payload["trustClass"],
                            validationStatus=contract_payload["validationStatus"],
                            freshnessStatus=contract_payload["freshnessStatus"],
                            staleReason=contract_payload["staleReason"],
                            sourceTime=contract_payload["sourceTime"],
                            manifestPath=contract_payload["manifestPath"],
                            contentHash=contract_payload["contentHash"],
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
        return items

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
        total_size = 0
        for item in items:
            by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
            trust_class = str(item.get("trustClass") or "unclassified")
            validation_status = str(item.get("validationStatus") or "unverified")
            by_trust[trust_class] = by_trust.get(trust_class, 0) + 1
            by_validation[validation_status] = by_validation.get(validation_status, 0) + 1
            total_size += int(item["sizeBytes"])
        return {
            "artifactCount": len(items),
            "totalSizeBytes": total_size,
            "byKind": by_kind,
            "byTrust": by_trust,
            "byValidation": by_validation,
            "indexedAt": datetime.fromtimestamp(self._indexed_at, tz=timezone.utc).isoformat(),
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
