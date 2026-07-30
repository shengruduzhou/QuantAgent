from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from services.quant_api.config import ApiSettings, project_relative, safe_project_path
from services.quant_api.schemas.strategy import StrategyDraft


DECISION_COUNCIL = (
    ("data_quality", "Data Quality", "PIT、覆盖率、重复键与隔离区"),
    ("factor_research", "Factor Research", "因子评审、相关性和失效条件"),
    ("model_validation", "Model Validation", "滚动切分、embargo 与 OOS 证据"),
    ("portfolio", "Portfolio", "目标权重、集中度与换手约束"),
    ("backtest", "Backtest", "A股撮合、成本、涨跌停与 T+1"),
    ("risk", "Risk", "回撤、流动性、kill switch 与否决"),
    ("challenger", "Challenger", "反证、敏感性与替代解释"),
    ("human_gate", "Human Gate", "最终研究启动授权"),
)


class StrategyService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.root = settings.runtime_root / "strategies"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            items.append(self._public(payload, path))
        return items

    def defaults(self) -> dict[str, Any]:
        groups = {
            "marketPanelPath": (
                "runtime/data/gold/full_universe/adjusted_market_panel.parquet",
                "runtime/data/v7/silver/market_panel/market_panel.parquet",
                "runtime/data/v7/full_universe/full_universe_market_panel.parquet",
                "runtime/data/u0/panel/daily_bars_raw.parquet",
            ),
            "labelsPath": (
                "runtime/data/gold/full_universe/labels.parquet",
                "runtime/data/v7/labels.parquet",
            ),
            "fundamentalsRoot": (
                "runtime/data/v7/silver/fundamentals",
                "runtime/data/u0/pit/fundamentals",
            ),
            "valuationPath": (
                "runtime/data/v7/silver/valuation/valuation.parquet",
                "runtime/data/u0/pit/valuation.parquet",
            ),
            "disclosuresPath": (
                "runtime/data/v7/silver/disclosures/disclosures.parquet",
                "runtime/data/u0/pit/disclosures.parquet",
            ),
            "trainingDatasetPath": (
                "runtime/data/gold/full_universe/training_dataset.parquet",
                "runtime/data/v7/gold/training_dataset/training_dataset.parquet",
            ),
            "sectorMapPath": (
                "runtime/data/v7/silver/sector_map.parquet",
                "runtime/data/u0/pit/sector_map.parquet",
            ),
        }
        selected: dict[str, str | None] = {}
        evidence: list[dict[str, Any]] = []
        for field, candidates in groups.items():
            found: list[tuple[float, Path, str]] = []
            for candidate in candidates:
                path = safe_project_path(self.settings, candidate)
                is_expected_input = path.is_dir() if field == "fundamentalsRoot" else path.is_file()
                if not is_expected_input:
                    continue
                stat = path.stat()
                found.append((stat.st_mtime, path, candidate))
                evidence.append(
                    {
                        "field": field,
                        "path": candidate,
                        "sizeBytes": stat.st_size,
                        "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    }
                )
            found.sort(reverse=True, key=lambda item: item[0])
            selected[field] = found[0][2] if found else None
        return {
            "selected": selected,
            "evidence": sorted(evidence, key=lambda item: item["modifiedAt"], reverse=True),
            "selectionRule": "newest existing canonical path; no recursive scan of large Runtime",
        }

    def validate(self, draft: StrategyDraft) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        inputs = {
            "marketPanelPath": draft.market_panel_path,
            "labelsPath": draft.labels_path,
            "sectorMapPath": draft.sector_map_path,
            "trainingDatasetPath": draft.training_dataset_path,
            "synthesizedFactorsPath": draft.synthesized_factors_path,
            "fundamentalsRoot": draft.fundamentals_root,
            "valuationPath": draft.valuation_path,
            "disclosuresPath": draft.disclosures_path,
            "minutePanelPath": draft.minute_panel_path,
        }
        resolved_inputs: dict[str, str] = {}
        for key, value in inputs.items():
            if not value:
                continue
            try:
                path = safe_project_path(self.settings, value)
                resolved_inputs[key] = project_relative(self.settings, path)
                if not path.exists():
                    if key in {"marketPanelPath", "labelsPath"} or (
                        key == "minutePanelPath" and draft.do_t_mode in {"intraday", "both"}
                    ):
                        errors.append(f"{key}: input path does not exist")
                    else:
                        warnings.append(
                            f"{key}: optional input path does not exist; dependent candidate will fail closed."
                        )
            except ValueError as exc:
                errors.append(f"{key}: {exc}")
        try:
            output = safe_project_path(self.settings, draft.output_dir)
            runtime = self.settings.runtime_root.resolve()
            if output != runtime and runtime not in output.parents:
                errors.append("outputDir must remain inside runtime")
        except ValueError as exc:
            errors.append(f"outputDir: {exc}")
        if not draft.human_approved:
            warnings.append("Human Gate 尚未授权；可保存草稿和验证，但不能启动。")
        if draft.model == "ft_transformer" and not draft.require_gpu:
            warnings.append("FT-Transformer 未强制 GPU；运行时可能降级或失败，取决于训练实现。")
        if "fundamental" in draft.stock_selection_modes and not (
            draft.fundamentals_root or draft.training_dataset_path
        ):
            warnings.append(
                "基本面选股候选缺少 PIT 财务输入；该候选将 fail closed，无筛选基线仍可比较。"
            )
        if len(set(draft.top_k_candidates)) > 1:
            warnings.append(
                "Top-K 将逐候选执行同一成本后回测，再从 Pareto 前沿按研究偏好选冠军。"
            )
        if not draft.benchmark_symbol:
            warnings.append("未指定 benchmarkSymbol；无法验证最大超额目标。")
        if draft.do_t_mode in {"daily_swing", "both"}:
            warnings.append("日线波段使用 ATR timing gate 与持有期软锁，不冒充盘中成交能力。")
        warnings.append("优化目标是研究偏好，不是收益承诺；验收只读取真实 OOS/回测产物。")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "resolvedInputs": resolved_inputs,
            "launch": {
                "jobType": "strategy-pipeline",
                "commandId": "run-full-real-training-v7",
                "parameters": self.launch_parameters(draft),
                "armed": draft.human_approved and not errors,
            },
            "decisionCouncil": [
                {
                    "id": role_id,
                    "label": label,
                    "responsibility": responsibility,
                    "status": "ready" if not errors and role_id != "human_gate" else "approved" if role_id == "human_gate" and draft.human_approved else "blocked" if errors else "waiting",
                    "veto": role_id in {"data_quality", "model_validation", "risk", "challenger", "human_gate"},
                }
                for role_id, label, responsibility in DECISION_COUNCIL
            ],
        }

    def save(self, draft: StrategyDraft) -> dict[str, Any]:
        validation = self.validate(draft)
        strategy_id = draft.id or self._slug(draft.name)
        strategy_dir = self.root / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        version = now.strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "schemaVersion": "quantagent.strategy.v1",
            "id": strategy_id,
            "version": version,
            "createdAt": now.isoformat(timespec="seconds"),
            "trustClass": "research_only",
            "draft": draft.model_dump(by_alias=True),
            "validation": validation,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        payload["contentHash"] = sha256(content.encode("utf-8")).hexdigest()
        path = strategy_dir / f"{version}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._public(payload, path)

    @staticmethod
    def launch_parameters(draft: StrategyDraft) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "market_panel_path": draft.market_panel_path,
            "labels_path": draft.labels_path,
            "output_dir": draft.output_dir,
            "factor_library": draft.factor_library,
            "model": draft.model,
            "horizons": draft.horizons,
            "primary_horizon": draft.primary_horizon,
            "split_mode": draft.split_mode,
            "n_splits": draft.n_splits,
            "require_gpu": draft.require_gpu,
            "top_k": draft.top_k,
            "top_k_candidates": draft.top_k_candidates,
            "stock_selection_modes": draft.stock_selection_modes,
            "fundamental_selection_threshold": draft.fundamental_selection_threshold,
            "factor_screening_mode": draft.factor_screening_mode,
            "do_t_mode": draft.do_t_mode,
            "max_weight_per_name": draft.max_weight_per_name,
            "max_sector_weight": draft.max_sector_weight,
            "max_turnover": draft.max_turnover,
            "objective": draft.objective,
            "weighting": draft.weighting,
            "initial_cash": draft.initial_cash,
            "objective_excess_weight": draft.objective_weights.excess_return,
            "objective_annual_weight": draft.objective_weights.annual_return,
            "objective_drawdown_weight": draft.objective_weights.drawdown_control,
            "acceptance_max_drawdown": draft.risk_limits.max_drawdown,
            "acceptance_min_sharpe": draft.risk_limits.min_sharpe,
        }
        optional = {
            "sector_map_path": draft.sector_map_path,
            "training_dataset_path": draft.training_dataset_path,
            "synthesized_factors_path": draft.synthesized_factors_path,
            "benchmark_symbol": draft.benchmark_symbol,
            "fundamentals_root": draft.fundamentals_root,
            "valuation_path": draft.valuation_path,
            "disclosures_path": draft.disclosures_path,
            "minute_panel_path": draft.minute_panel_path,
        }
        parameters.update({key: value for key, value in optional.items() if value})
        return parameters

    def _public(self, payload: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "id": payload["id"],
            "version": payload["version"],
            "name": payload["draft"]["name"],
            "createdAt": payload["createdAt"],
            "trustClass": payload["trustClass"],
            "contentHash": payload.get("contentHash"),
            "path": project_relative(self.settings, path),
            "valid": bool(payload.get("validation", {}).get("valid")),
            "humanApproved": bool(payload["draft"].get("humanApproved")),
            "draft": payload["draft"],
        }

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
        return (slug or "strategy")[:48]
