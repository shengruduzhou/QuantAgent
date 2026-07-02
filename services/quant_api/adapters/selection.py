from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from services.quant_api.adapters.utils import read_json, require_relative_path
from services.quant_api.config import ApiSettings, stable_id


class SelectionAdapter:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._runs: dict[str, Path] = {}

    def list(self) -> list[dict[str, Any]]:
        runs = []
        self._runs = {}
        for summary_path in self.settings.runtime_root.glob("reports/v8/**/summary.json"):
            directory = summary_path.parent
            pool_path = directory / "hybrid_stock_pool.parquet"
            if not pool_path.exists():
                continue
            payload = read_json(summary_path, {}) or {}
            relative = require_relative_path(self.settings, directory)
            run_id = stable_id("selection", relative)
            self._runs[run_id] = directory
            runs.append({
                "id": run_id,
                "asOfDate": payload.get("as_of_date"),
                "candidateCount": payload.get("candidate_rows"),
                "finalCount": payload.get("final_stock_rows"),
                "usedFallback": payload.get("used_fallback"),
                "noOrdersGenerated": (payload.get("position_hint") or {}).get("no_orders_generated"),
                "path": relative,
                "status": "ready",
                "modifiedAt": summary_path.stat().st_mtime,
            })
        return sorted(runs, key=lambda item: item["modifiedAt"], reverse=True)

    def get(self, run_id: str) -> dict[str, Any] | None:
        directory = self._resolve(run_id)
        if directory is None:
            return None
        return {
            "id": run_id,
            "summary": read_json(directory / "summary.json", {}) or {},
            "analysis": read_json(directory / "capital_flow_stock_pool_analysis.json", {}) or {},
            "path": require_relative_path(self.settings, directory),
        }

    def ranking(self, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        directory = self._require(run_id)
        path = directory / "hybrid_stock_pool.parquet"
        if not path.exists():
            return []
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
        sort_column = next((name for name in ("hybrid_rank", "model_rank", "llm_rank") if name in schema), None)
        if sort_column is not None:
            lazy = lazy.sort(sort_column)
        frame = lazy.head(limit).collect()
        rows = []
        for row in frame.to_dicts():
            contributions = {
                key: float(row[key])
                for key in (
                    "core_policy_score", "core_sentiment_score", "fundamental_quality_score",
                    "cicc_stock_selection_score", "cicc_sector_selection_score",
                    "cicc_aggressive_momentum_score", "cicc_defensive_quality_score",
                    "cicc_liquidity_defense_score", "sector_resonance_score",
                    "dip_buy_flow_score", "trend_strength_score",
                )
                if row.get(key) is not None
            }
            rows.append({
                "symbol": str(row.get("symbol") or ""),
                "name": None,
                "sector": row.get("sector_level_1"),
                "modelRank": _int(row.get("model_rank")),
                "modelScore": _float(row.get("prediction")),
                "factorScore": _float(row.get("factor_prior_score") or row.get("factor_rank_score")),
                "llmScore": _float(row.get("llm_stock_score")),
                "confidence": _float(row.get("llm_confidence")),
                "riskScore": _float(row.get("old_dealer_risk_score")),
                "doTSuitability": _float(row.get("do_t_suitability_score")),
                "finalScore": _float(row.get("hybrid_score")),
                "finalRank": _int(row.get("hybrid_rank")),
                "actionBucket": row.get("action_bucket"),
                "included": True,
                "exclusionReason": None,
                "factorContributions": contributions,
                "researchWeightHint": _float(row.get("research_weight_hint")),
                "noOrdersGenerated": bool(row.get("no_orders_generated", True)),
            })
        return rows

    def funnel(self, run_id: str) -> list[dict[str, Any]]:
        detail = self.get(run_id) or {}
        summary = detail.get("summary", {})
        ranking = self.ranking(run_id)
        positive = sum((item.get("finalScore") or 0) > 0 for item in ranking)
        actionable = sum(str(item.get("actionBucket") or "").lower() not in {"avoid", "exclude", "watch"} for item in ranking)
        return [
            {"stage": "Initial candidates", "count": summary.get("candidate_rows") or len(ranking), "reason": None},
            {"stage": "Deterministic factor/risk scoring", "count": positive, "reason": "positive hybrid inputs"},
            {"stage": "LLM research overlay", "count": summary.get("final_stock_rows") or len(ranking), "reason": "research only"},
            {"stage": "Actionable research bucket", "count": actionable, "reason": "no orders generated"},
        ]

    def decision_chain(self, run_id: str, symbol: str) -> dict[str, Any] | None:
        row = next((item for item in self.ranking(run_id) if item["symbol"] == symbol), None)
        if row is None:
            return None
        stages = [
            ("initial_pool", True, "present_in_hybrid_pool", {}),
            ("model_rank", row.get("modelRank") is not None, None, {"rank": row.get("modelRank"), "score": row.get("modelScore")}),
            ("factor_score", row.get("factorScore") is not None, None, row.get("factorContributions", {})),
            ("risk_overlay", True, None, {"old_dealer_risk": row.get("riskScore")}),
            ("do_t_suitability", row.get("doTSuitability") is not None, None, {"score": row.get("doTSuitability")}),
            ("llm_research_overlay", row.get("llmScore") is not None, None, {"score": row.get("llmScore"), "confidence": row.get("confidence")}),
            ("final_rank", True, row.get("actionBucket"), {"score": row.get("finalScore"), "rank": row.get("finalRank")}),
        ]
        return {
            "runId": run_id,
            "symbol": symbol,
            "datetime": (self.get(run_id) or {}).get("summary", {}).get("as_of_date"),
            "finalDecision": row.get("actionBucket"),
            "failedGate": None,
            "traceType": "score_pipeline",
            "gates": [
                {"order": index + 1, "name": name, "passed": passed, "reason": reason, "detail": detail}
                for index, (name, passed, reason, detail) in enumerate(stages)
            ],
            "issues": [{
                "code": "no_persisted_gate_trace",
                "message": "该 run 没有 persisted decision_traces；当前展示 score pipeline。",
                "recoverable": True,
            }],
        }

    def _resolve(self, run_id: str) -> Path | None:
        if run_id not in self._runs:
            self.list()
        return self._runs.get(run_id)

    def _require(self, run_id: str) -> Path:
        path = self._resolve(run_id)
        if path is None:
            raise KeyError(run_id)
        return path


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
