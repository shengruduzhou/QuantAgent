from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from services.quant_api.config import ApiSettings, project_relative, stable_id


class RuntimeCleanupService:
    """Build auditable cleanup candidates without touching canonical market data."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings

    def analyze(self) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []

        invalid_registry = self.settings.runtime_root / "models" / "v7_alpha" / "registry"
        if self._is_test_registry(invalid_registry):
            candidates.append(self._candidate(
                invalid_registry,
                category="invalid_test_registry",
                label="测试生成的 V7 model registry",
                reason="所有 model output_dir 均指向 pytest/Claude 临时目录，不能用于真实推理。",
                safe_default=True,
            ))

        temp_paths = [
            self.settings.runtime_root / "tmp" / "test_dot_models.joblib",
            self.settings.runtime_root / "tmp" / "demo_book_oos.csv",
            self.settings.runtime_root / "tmp" / "ff_test_syms.txt",
            self.settings.runtime_root / "tmp" / "dot_smoke",
            self.settings.runtime_root / "tmp" / "board_smoke",
        ]
        existing_temp = [path for path in temp_paths if path.exists()]
        if existing_temp:
            candidates.append(self._multi_candidate(
                existing_temp,
                category="test_temp",
                label="明确的 test/smoke 临时产物",
                reason="文件名和目录均明确标记为 test、demo 或 smoke，且不属于 canonical data。",
                safe_default=True,
            ))

        smoke_dirs = [
            path
            for path in self.settings.runtime_root.joinpath("reports").iterdir()
            if path.is_dir() and "smoke" in path.name.lower()
        ] if self.settings.runtime_root.joinpath("reports").exists() else []
        nested_smoke = self.settings.runtime_root / "reports" / "v8" / "llm_stock_selection_smoke"
        if nested_smoke.exists():
            smoke_dirs.append(nested_smoke)
        if smoke_dirs:
            candidates.append(self._multi_candidate(
                sorted(set(smoke_dirs)),
                category="smoke_reports",
                label="历史 smoke report",
                reason="只用于快速验证的可再生产物，不作为模型、回测或风控基线。",
                safe_default=True,
            ))

        qa_dir = self.settings.runtime_root / "reports" / "quant_ui" / "qa"
        keep_qa = {
            "dashboard-final-1440x1024.png",
            "stock-replay-final-1440x1024.png",
            "t-plus-one-final-1440x1024.png",
            "stock-replay-side-by-side-1800x800.png",
            "stock-replay-comparison.html",
        }
        cleanup_audits = list((self.settings.runtime_root / "reports" / "quant_ui" / "cleanup").glob("cleanup_*.json"))
        latest_cleanup_time = max((path.stat().st_mtime for path in cleanup_audits), default=None)
        stale_qa = [
            path for path in qa_dir.iterdir()
            if (
                path.is_file()
                and path.name not in keep_qa
                and (latest_cleanup_time is None or path.stat().st_mtime < latest_cleanup_time)
            )
        ] if qa_dir.exists() else []
        if stale_qa:
            candidates.append(self._multi_candidate(
                stale_qa,
                category="stale_ui_captures",
                label="旧版 Quant UI 截图",
                reason="已由 final capture 和 side-by-side QA 取代。",
                safe_default=True,
            ))

        cache_root = self.settings.cache_root
        if cache_root.exists():
            candidates.append(self._candidate(
                cache_root,
                category="rebuildable_cache",
                label="Quant UI runtime index cache",
                reason="可安全重建；删除后首次加载会重新索引 runtime。",
                safe_default=False,
            ))

        training_root = self.settings.runtime_root / "data" / "v7" / "gold" / "training_dataset"
        keep_datasets = {
            "training_dataset_alpha181_exec_v89_plus8.parquet",
            "training_dataset_alpha181_full_nosynth.parquet",
        }
        superseded = [
            path
            for path in training_root.glob("training_dataset_alpha181_exec_*.parquet")
            if path.name not in keep_datasets
        ] if training_root.exists() else []
        for path in sorted(superseded, key=lambda item: item.stat().st_mtime):
            candidates.append(self._candidate(
                path,
                category="superseded_large_dataset",
                label=f"待复核大型训练集 · {path.name}",
                reason="存在更新的 v89 plus8 dataset；删除可显著释放空间，但可能影响历史复现实验。",
                safe_default=False,
                requires_explicit=True,
            ))

        candidates = [item for item in candidates if item["sizeBytes"] > 0 or item["itemCount"] > 0]
        return {
            "runtimeSizeBytes": _path_size(self.settings.runtime_root),
            "candidateSizeBytes": sum(item["sizeBytes"] for item in candidates),
            "safeDefaultSizeBytes": sum(item["sizeBytes"] for item in candidates if item["safeDefault"]),
            "candidates": candidates,
            "protected": [
                "runtime/data/raw",
                "runtime/data/v7/silver",
                "runtime/data/v7/manifests",
                "runtime/models/registry",
                "runtime/reports/v8/deep/v89_rankfix_20260613_1044",
                "runtime/reports/v8/deep/v86_governed_full_20260610_1858",
            ],
        }

    def execute(self, candidate_ids: list[str], confirmation: str) -> dict[str, Any]:
        if confirmation != "DELETE":
            raise ValueError("confirmation must equal DELETE")
        analysis = self.analyze()
        mapping = {item["id"]: item for item in analysis["candidates"]}
        unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in mapping]
        if unknown:
            raise ValueError(f"unknown cleanup candidates: {unknown}")
        selected = [mapping[candidate_id] for candidate_id in candidate_ids]
        deleted = []
        errors = []
        for candidate in selected:
            candidate_deleted = []
            for relative in candidate["paths"]:
                path = self.settings.project_root / relative
                try:
                    self._assert_deletable(path)
                    size = _path_size(path)
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                    candidate_deleted.append({"path": relative, "sizeBytes": size})
                except (OSError, ValueError) as exc:
                    errors.append({"path": relative, "message": str(exc)})
            if candidate_deleted:
                deleted.append({
                    "id": candidate["id"],
                    "label": candidate["label"],
                    "items": candidate_deleted,
                    "sizeBytes": sum(item["sizeBytes"] for item in candidate_deleted),
                })
        audit = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "confirmation": confirmation,
            "deleted": deleted,
            "errors": errors,
            "freedBytes": sum(item["sizeBytes"] for item in deleted),
        }
        audit_dir = self.settings.runtime_root / "reports" / "quant_ui" / "cleanup"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"cleanup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        audit["auditPath"] = project_relative(self.settings, audit_path)
        return audit

    def _candidate(
        self,
        path: Path,
        *,
        category: str,
        label: str,
        reason: str,
        safe_default: bool,
        requires_explicit: bool = False,
    ) -> dict[str, Any]:
        return self._multi_candidate(
            [path],
            category=category,
            label=label,
            reason=reason,
            safe_default=safe_default,
            requires_explicit=requires_explicit,
        )

    def _multi_candidate(
        self,
        paths: list[Path],
        *,
        category: str,
        label: str,
        reason: str,
        safe_default: bool,
        requires_explicit: bool = False,
    ) -> dict[str, Any]:
        existing = [path.resolve() for path in paths if path.exists()]
        relative_paths = [project_relative(self.settings, path) for path in existing]
        latest = max((path.stat().st_mtime for path in existing), default=0.0)
        identity = "|".join(relative_paths)
        return {
            "id": stable_id("cleanup", f"{category}:{identity}"),
            "category": category,
            "label": label,
            "reason": reason,
            "paths": relative_paths,
            "sizeBytes": sum(_path_size(path) for path in existing),
            "itemCount": sum(_path_count(path) for path in existing),
            "modifiedAt": datetime.fromtimestamp(latest, tz=timezone.utc).isoformat() if latest else None,
            "safeDefault": safe_default,
            "requiresExplicit": requires_explicit,
        }

    def _is_test_registry(self, directory: Path) -> bool:
        if not directory.exists():
            return False
        records = []
        for path in directory.glob("*.json"):
            payload = _read_json(path)
            if payload:
                records.append(payload)
        if not records:
            return False
        outputs = [
            str((payload.get("metadata") or {}).get("output_dir") or "")
            for payload in records
        ]
        project_root = self.settings.project_root.resolve()
        return bool(outputs) and all(
            value
            and any(marker in value.lower() for marker in ("pytest", "claude", "test"))
            and project_root not in Path(value).expanduser().resolve().parents
            and Path(value).expanduser().resolve() != project_root
            for value in outputs
        )

    def _assert_deletable(self, path: Path) -> None:
        runtime = self.settings.runtime_root.resolve()
        resolved = path.resolve()
        if resolved == runtime or runtime not in resolved.parents:
            raise ValueError("cleanup path is outside runtime or points to runtime root")
        protected_roots = [
            runtime / "data" / "raw",
            runtime / "data" / "v7" / "silver",
            runtime / "data" / "v7" / "manifests",
            runtime / "models" / "registry",
        ]
        if any(resolved == root or root in resolved.parents for root in protected_roots):
            raise ValueError("cleanup path is protected")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _path_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())
