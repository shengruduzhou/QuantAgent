"""Read-side adapter for factor-fusion search runs.

A run is any directory holding a ``manifest.json`` whose ``artifact`` field is
``factor_fusion_search``. The adapter never computes metrics of its own: every
number it returns comes from a file the search wrote, so a value shown in the
workstation can always be traced back to an artifact on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.quant_api.adapters.utils import read_csv_rows, read_json
from services.quant_api.config import (
    ApiSettings,
    project_relative,
    safe_project_path,
    stable_id,
)

MANIFEST_ARTIFACT = "factor_fusion_search"
SEARCH_ROOTS = (
    "runtime/reports/fusion",
    "runtime/reports/v7/fusion",
    "runtime/reports/quant_ui_jobs",
)
MAX_SCAN_DEPTH = 4


class FusionAdapter:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._runs: dict[str, Path] = {}

    # ----------------------------------------------------------------- scan #

    def _manifest_directories(self) -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = set()
        for root in SEARCH_ROOTS:
            base = self.settings.project_root / root
            if not base.is_dir():
                continue
            for manifest in base.rglob("manifest.json"):
                try:
                    depth = len(manifest.relative_to(base).parts)
                except ValueError:
                    continue
                if depth > MAX_SCAN_DEPTH:
                    continue
                payload = read_json(manifest, {}) or {}
                if payload.get("artifact") != MANIFEST_ARTIFACT:
                    continue
                directory = manifest.parent.resolve()
                if directory in seen:
                    continue
                seen.add(directory)
                found.append(directory)
        return found

    def invalidate(self) -> None:
        self._runs = {}

    # ----------------------------------------------------------------- list #

    def list(self) -> list[dict[str, Any]]:
        self._runs = {}
        summaries: list[dict[str, Any]] = []
        for directory in self._manifest_directories():
            manifest = read_json(directory / "manifest.json", {}) or {}
            summary = read_json(directory / "fusion_summary.json", {}) or {}
            relative = project_relative(self.settings, directory)
            run_id = stable_id("fusion", relative)
            self._runs[run_id] = directory
            summaries.append(
                {
                    "id": run_id,
                    "name": directory.name,
                    "path": relative,
                    "generatedAt": manifest.get("generatedAt") or summary.get("generatedAt"),
                    "contentHash": manifest.get("contentHash"),
                    "nTrials": summary.get("nTrials") or manifest.get("nTrials"),
                    "pbo": summary.get("pbo"),
                    "benchmarkMode": summary.get("benchmarkMode") or manifest.get("benchmarkMode"),
                    "horizonDays": summary.get("horizonDays") or manifest.get("horizonDays"),
                    "topK": summary.get("topK") or manifest.get("topK"),
                    "transactionCostBps": summary.get("transactionCostBps"),
                    "factorNames": summary.get("factorNames") or manifest.get("factorNames") or [],
                    "frontierSize": len(summary.get("frontier") or []),
                    "candidateCount": summary.get("candidateCount"),
                    "evaluatedCandidateCount": summary.get("evaluatedCandidateCount"),
                    "foldCount": len(summary.get("foldWindows") or manifest.get("folds") or []),
                }
            )
        summaries.sort(key=lambda row: str(row.get("generatedAt") or ""), reverse=True)
        return summaries

    # ------------------------------------------------------------------ get #

    def _directory(self, run_id: str) -> Path:
        if run_id not in self._runs:
            self.list()
        directory = self._runs.get(run_id)
        if directory is None:
            raise KeyError(run_id)
        return directory

    def detail(self, run_id: str) -> dict[str, Any]:
        directory = self._directory(run_id)
        summary = read_json(directory / "fusion_summary.json", {}) or {}
        manifest = read_json(directory / "manifest.json", {}) or {}
        ranking = read_json(directory / "fusion_ranking.json", []) or []
        candidates = read_json(directory / "fusion_candidates.json", []) or []
        frontier = set(summary.get("frontier") or [])
        rank_index = {
            str(row.get("id")): index for index, row in enumerate(ranking)
        }
        enriched = [
            {
                **candidate,
                "onFrontier": str(candidate.get("id")) in frontier,
                "preferenceRank": rank_index.get(str(candidate.get("id"))),
                "preferenceScore": next(
                    (
                        row.get("preferenceScore")
                        for row in ranking
                        if str(row.get("id")) == str(candidate.get("id"))
                    ),
                    None,
                ),
            }
            for candidate in candidates
        ]
        return {
            "id": run_id,
            "path": project_relative(self.settings, directory),
            "summary": summary,
            "manifest": manifest,
            "ranking": ranking,
            "candidates": enriched,
        }

    def navs(self, run_id: str, limit: int = 4_000) -> list[dict[str, Any]]:
        directory = self._directory(run_id)
        return read_csv_rows(directory / "fusion_nav.csv", limit=limit)

    def candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        detail = self.detail(run_id)
        for candidate in detail["candidates"]:
            if str(candidate.get("id")) == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    def compare(self, run_id: str, candidate_ids: list[str]) -> dict[str, Any]:
        """Side-by-side view of at most four candidates from one run."""
        if not candidate_ids:
            raise ValueError("at least one candidate id is required")
        if len(candidate_ids) > 4:
            raise ValueError("at most four candidates can be compared at once")
        detail = self.detail(run_id)
        by_id = {str(item.get("id")): item for item in detail["candidates"]}
        missing = [item for item in candidate_ids if item not in by_id]
        if missing:
            raise KeyError(", ".join(missing))
        selected = [by_id[item] for item in candidate_ids]
        return {
            "runId": run_id,
            "summary": detail["summary"],
            "candidates": selected,
            "factorNames": detail["summary"].get("factorNames") or [],
        }

    def resolve_path(self, run_id: str) -> Path:
        return safe_project_path(self.settings, project_relative(self.settings, self._directory(run_id)))


__all__ = ["FusionAdapter", "MANIFEST_ARTIFACT", "SEARCH_ROOTS"]
