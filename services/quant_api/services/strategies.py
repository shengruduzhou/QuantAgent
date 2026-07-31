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

HORIZON_COLUMN = re.compile(r"^forward_return_(\d+)d$")


def _parquet_horizons(path: Path) -> list[int]:
    import pyarrow.dataset as parquet_dataset

    columns = parquet_dataset.dataset(path, format="parquet").schema.names
    return sorted(
        {
            int(match.group(1))
            for column in columns
            if (match := HORIZON_COLUMN.fullmatch(column))
        }
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
        options: dict[str, list[dict[str, Any]]] = {}
        for field, candidates in groups.items():
            found: list[tuple[int, float, str]] = []
            field_options: list[dict[str, Any]] = []
            for candidate in candidates:
                path = safe_project_path(self.settings, candidate)
                is_expected_input = path.is_dir() if field == "fundamentalsRoot" else path.is_file()
                available_horizons: list[int] = []
                if is_expected_input and field == "labelsPath":
                    try:
                        available_horizons = _parquet_horizons(path)
                    except Exception:
                        available_horizons = []
                if is_expected_input:
                    stat = path.stat()
                    modified_at = datetime.fromtimestamp(
                        stat.st_mtime,
                        timezone.utc,
                    ).isoformat(timespec="seconds")
                    option = {
                        "field": field,
                        "path": candidate,
                        "exists": True,
                        "isDirectory": path.is_dir(),
                        "sizeBytes": stat.st_size,
                        "modifiedAt": modified_at,
                        "availableHorizons": available_horizons,
                    }
                    evidence.append(option)
                    coverage_score = len(available_horizons) if field == "labelsPath" else 0
                    found.append((coverage_score, stat.st_mtime, candidate))
                else:
                    option = {
                        "field": field,
                        "path": candidate,
                        "exists": False,
                        "isDirectory": field == "fundamentalsRoot",
                        "sizeBytes": 0,
                        "modifiedAt": None,
                        "availableHorizons": [],
                    }
                field_options.append(option)
            found.sort(reverse=True, key=lambda item: (item[0], item[1]))
            selected[field] = found[0][2] if found else None
            options[field] = field_options
        return {
            "selected": selected,
            "options": options,
            "evidence": sorted(
                evidence,
                key=lambda item: item["modifiedAt"] or "",
                reverse=True,
            ),
            "selectionRule": (
                "labels prefer the widest available horizon schema, then newest; "
                "other inputs prefer newest existing canonical path"
            ),
        }

    def validate(self, draft: StrategyDraft) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []

        def add_issue(
            code: str,
            severity: str,
            title: str,
            detail: str,
            *,
            field: str | None = None,
            action: dict[str, str] | None = None,
            evidence: dict[str, Any] | None = None,
        ) -> None:
            issue: dict[str, Any] = {
                "code": code,
                "severity": severity,
                "title": title,
                "detail": detail,
            }
            if field:
                issue["field"] = field
            if action:
                issue["action"] = action
            if evidence:
                issue["evidence"] = evidence
            issues.append(issue)

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
        resolved_paths: dict[str, Path] = {}
        required_paths = {"marketPanelPath", "labelsPath"}
        if draft.do_t_mode in {"intraday", "both"}:
            required_paths.add("minutePanelPath")
        for key, value in inputs.items():
            if not value:
                continue
            try:
                path = safe_project_path(self.settings, value)
                resolved_inputs[key] = project_relative(self.settings, path)
                resolved_paths[key] = path
                if not path.exists():
                    required = key in required_paths
                    fundamental_input = key in {
                        "fundamentalsRoot",
                        "trainingDatasetPath",
                    } and "fundamental" in draft.stock_selection_modes
                    if required or not fundamental_input:
                        add_issue(
                            f"{key}_missing",
                            "blocking" if required else "warning",
                            "必需输入不存在" if required else "可选输入不可用",
                            (
                                f"{key}: input path does not exist"
                                if required
                                else (
                                    f"{key}: optional input path does not exist; "
                                    "dependent candidate will fail closed."
                                )
                            ),
                            field=key,
                            action={
                                "kind": "choose_input",
                                "label": "选择可用输入",
                            },
                        )
            except ValueError as exc:
                add_issue(
                    f"{key}_unsafe",
                    "blocking",
                    "输入路径不安全",
                    f"{key}: {exc}",
                    field=key,
                )

        requested_horizons = sorted(
            {int(value) for value in draft.horizons.split(",")}
        )
        available_horizons: list[int] = []
        labels_path = resolved_paths.get("labelsPath")
        if labels_path is not None and labels_path.exists():
            try:
                available_horizons = _parquet_horizons(labels_path)
            except Exception as exc:
                add_issue(
                    "labels_unreadable",
                    "blocking",
                    "Labels 无法读取",
                    (
                        "labelsPath: unreadable Parquet schema; rebuild the label artifact "
                        f"({type(exc).__name__})"
                    ),
                    field="labelsPath",
                    action={"kind": "rebuild_labels", "label": "重建 Labels"},
                )
            else:
                missing_horizons = sorted(
                    set(requested_horizons) - set(available_horizons)
                )
                if missing_horizons:
                    missing_columns = [
                        f"forward_return_{horizon}d"
                        for horizon in missing_horizons
                    ]
                    add_issue(
                        "missing_horizon_columns",
                        "blocking",
                        "Labels 缺少研究周期",
                        (
                            "labelsPath: missing requested horizon columns "
                            f"{', '.join(missing_columns)}; rebuild labels or remove "
                            "those Horizons"
                        ),
                        field="horizons",
                        action={
                            "kind": "resolve_horizons",
                            "label": "选择修复方式",
                        },
                        evidence={
                            "requestedHorizons": requested_horizons,
                            "availableHorizons": available_horizons,
                            "missingHorizons": missing_horizons,
                        },
                    )
        try:
            output = safe_project_path(self.settings, draft.output_dir)
            runtime = self.settings.runtime_root.resolve()
            if output != runtime and runtime not in output.parents:
                add_issue(
                    "output_outside_runtime",
                    "blocking",
                    "输出目录越界",
                    "outputDir must remain inside runtime",
                    field="outputDir",
                )
        except ValueError as exc:
            add_issue(
                "output_path_unsafe",
                "blocking",
                "输出目录不安全",
                f"outputDir: {exc}",
                field="outputDir",
            )

        if not draft.human_approved:
            add_issue(
                "human_gate_pending",
                "warning",
                "等待 Human Gate",
                "Human Gate 尚未授权；可保存草稿和验证，但不能启动。",
                field="humanApproved",
                action={"kind": "arm_human_gate", "label": "人工确认后授权"},
            )
        if draft.model == "ft_transformer" and not draft.require_gpu:
            add_issue(
                "gpu_not_enforced",
                "warning",
                "GPU 契约未锁定",
                "FT-Transformer 未强制 GPU；运行时将 fail closed。",
                field="requireGpu",
            )

        fundamentals_path = resolved_paths.get("fundamentalsRoot")
        training_path = resolved_paths.get("trainingDatasetPath")
        fundamentals_available = bool(
            (fundamentals_path and fundamentals_path.is_dir())
            or (training_path and training_path.is_file())
        )
        if (
            "fundamental" in draft.stock_selection_modes
            and not fundamentals_available
        ):
            has_baseline = "none" in draft.stock_selection_modes
            add_issue(
                "fundamentals_missing",
                "warning" if has_baseline else "blocking",
                "基本面候选缺少 PIT 输入",
                (
                    "基本面选股候选缺少 PIT 财务输入；该候选将 fail closed，"
                    + (
                        "无筛选基线仍可比较。"
                        if has_baseline
                        else "且没有无筛选基线可回退。"
                    )
                ),
                field="fundamentalsRoot",
                action={
                    "kind": "resolve_fundamentals",
                    "label": "选择目录或关闭基本面候选",
                },
            )

        if len(set(draft.top_k_candidates)) > 1:
            add_issue(
                "pareto_top_k_protocol",
                "info",
                "Top-K 使用同成本候选赛",
                "Top-K 将逐候选执行同一成本后回测，再从 Pareto 前沿按研究偏好选冠军。",
            )
        if (
            draft.fundamental_selection_mode == "auto"
            and "fundamental" in draft.stock_selection_modes
        ):
            candidate_count = len(set(draft.top_k_candidates)) * (
                (1 if "none" in draft.stock_selection_modes else 0)
                + len(set(draft.fundamental_threshold_candidates))
                * len(set(draft.fundamental_blend_candidates))
            )
            add_issue(
                "bounded_early_oos_search",
                "info",
                "参数只在早期 OOS 学习",
                (
                    "基本面权重、入围分位和 Top-K 将只在早期滚动 OOS 上搜索 "
                    f"{candidate_count} 组；参数冻结后，最终 holdout 只验收一次。"
                ),
                evidence={
                    "candidateCount": candidate_count,
                    "candidateLimit": draft.selection_max_candidates,
                },
            )

        blend_details = {
            "adaptive_oos": (
                "以早期 OOS 横截面 RankIC/稳定性学习非负权重；"
                "按各周期清除 forward-label 重叠区，并在最终 holdout 前冻结。"
            ),
            "balanced": "以 20D/60D 中周期为主的固定审计权重。",
            "short_tactical": "以 1D/5D 短周期为主，强调战术响应。",
            "long_fundamental": "以 20D/60D/120D 中长周期为主，强调基本面兑现。",
            "primary_only": "仅使用主周期，作为最易解释的消融基线。",
        }
        add_issue(
            "horizon_blend_policy",
            "info",
            "周期融合已声明",
            blend_details[draft.horizon_blend_method],
            evidence={"method": draft.horizon_blend_method},
        )

        add_issue(
            "overfit_gates",
            "info",
            "过拟合闸门",
            (
                f"PBO ≤ {draft.max_pbo:.2f}、"
                f"DSR probability ≥ {draft.min_dsr_probability:.2f}、"
                f"SPA p-value ≤ {draft.max_spa_pvalue:.2f}；任一失败即阻止晋级。"
            ),
        )
        if not draft.benchmark_symbol:
            add_issue(
                "benchmark_missing",
                "blocking" if draft.objective_weights.excess_return > 0 else "warning",
                "缺少超额收益基准",
                "未指定 benchmarkSymbol；无法验证最大超额目标。",
                field="benchmarkSymbol",
                action={"kind": "set_benchmark", "label": "使用沪深 300"},
            )
        if draft.do_t_mode in {"daily_swing", "both"}:
            add_issue(
                "daily_swing_contract",
                "info",
                "T+1 日线能力边界",
                "日线波段使用 ATR timing gate 与持有期软锁，不冒充盘中成交能力。",
            )
        add_issue(
            "research_only",
            "info",
            "研究目标不是收益承诺",
            "优化权重只表达研究偏好；验收只读取真实 OOS 与成本后回测产物。",
        )
        add_issue(
            "public_principles_only",
            "info",
            "原创且可审计的系统边界",
            (
                "架构吸收公开可验证的机构工程原则，并由本项目独立实现；"
                "不声称访问或复制任何机构未公开的私有系统。"
            ),
        )

        errors = [
            issue["detail"]
            for issue in issues
            if issue["severity"] == "blocking"
        ]
        warnings = [
            issue["detail"]
            for issue in issues
            if issue["severity"] == "warning"
        ]
        role_issue_codes = {
            "data_quality": {
                "marketPanelPath_missing",
                "labelsPath_missing",
                "labels_unreadable",
                "missing_horizon_columns",
                "fundamentals_missing",
            },
            "factor_research": {
                "fundamentals_missing",
                "horizon_blend_policy",
            },
            "model_validation": {
                "missing_horizon_columns",
                "gpu_not_enforced",
                "overfit_gates",
            },
            "portfolio": {
                "pareto_top_k_protocol",
                "bounded_early_oos_search",
                "benchmark_missing",
            },
            "backtest": {
                "minutePanelPath_missing",
                "daily_swing_contract",
                "benchmark_missing",
            },
            "risk": {
                "overfit_gates",
                "benchmark_missing",
                "research_only",
            },
            "challenger": {
                "bounded_early_oos_search",
                "overfit_gates",
                "public_principles_only",
            },
            "human_gate": {"human_gate_pending"},
        }
        next_actions = {
            "data_quality": "核对 PIT、覆盖率与 Labels schema",
            "factor_research": "检查融合权重和因子冗余证据",
            "model_validation": "复核滚动切分、embargo 与 holdout 隔离",
            "portfolio": "复核 Pareto 候选、集中度和换手",
            "backtest": "复核 A 股撮合、成本、涨跌停与 T+1",
            "risk": "复核回撤、压力场景与 kill switch",
            "challenger": "提交反证、敏感性和替代解释",
            "human_gate": "人工核对研究范围后决定是否授权",
        }
        decision_council: list[dict[str, Any]] = []
        for role_id, label, responsibility in DECISION_COUNCIL:
            relevant = [
                issue
                for issue in issues
                if issue["code"] in role_issue_codes[role_id]
            ]
            blocking = [
                issue for issue in relevant if issue["severity"] == "blocking"
            ]
            if role_id == "human_gate":
                status = "approved" if draft.human_approved else "waiting"
            else:
                status = "blocked" if blocking else "ready"
            primary_issue = blocking[0] if blocking else next(
                (
                    issue
                    for issue in relevant
                    if issue["severity"] == "warning"
                ),
                None,
            )
            decision_council.append(
                {
                    "id": role_id,
                    "label": label,
                    "responsibility": responsibility,
                    "status": status,
                    "veto": role_id
                    in {
                        "data_quality",
                        "model_validation",
                        "risk",
                        "challenger",
                        "human_gate",
                    },
                    "finding": (
                        primary_issue["title"]
                        if primary_issue
                        else "当前契约无角色级阻塞"
                    ),
                    "nextAction": (
                        primary_issue.get("action", {}).get("label")
                        if primary_issue
                        else next_actions[role_id]
                    ) or next_actions[role_id],
                    "issueCount": len(relevant),
                    "issueCodes": [issue["code"] for issue in relevant],
                }
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "issues": issues,
            "requestedHorizons": requested_horizons,
            "availableHorizons": available_horizons,
            "resolvedInputs": resolved_inputs,
            "launch": {
                "jobType": "strategy-pipeline",
                "commandId": "run-full-real-training-v7",
                "parameters": self.launch_parameters(draft),
                "armed": draft.human_approved and not errors,
            },
            "decisionCouncil": decision_council,
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
            "horizon_blend_method": draft.horizon_blend_method,
            "split_mode": draft.split_mode,
            "n_splits": draft.n_splits,
            "require_gpu": draft.require_gpu,
            "top_k": draft.top_k,
            "top_k_candidates": draft.top_k_candidates,
            "stock_selection_modes": draft.stock_selection_modes,
            "fundamental_selection_mode": draft.fundamental_selection_mode,
            "fundamental_selection_threshold": draft.fundamental_selection_threshold,
            "fundamental_blend_weight": draft.fundamental_blend_weight,
            "fundamental_threshold_candidates": draft.fundamental_threshold_candidates,
            "fundamental_blend_candidates": draft.fundamental_blend_candidates,
            "selection_max_candidates": draft.selection_max_candidates,
            "selection_min_oos_days": draft.selection_min_oos_days,
            "selection_min_holdout_days": draft.selection_min_holdout_days,
            "max_pbo": draft.max_pbo,
            "min_dsr_probability": draft.min_dsr_probability,
            "max_spa_pvalue": draft.max_spa_pvalue,
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
