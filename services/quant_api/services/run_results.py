"""Read a finished run's own artifacts and state what it concluded.

A completed pipeline leaves ~100 files across eight directories. Before this,
answering "did my strategy work, and why or why not" meant knowing which of
those files to open. This module reads the artifacts the run actually wrote and
returns one structure: the verdict, the gates behind it, the measured numbers,
and what is missing.

Nothing here computes performance. Every number is copied from an artifact and
carries the path it came from, so a claim in the UI can always be traced back to
the file that supports it. When an artifact is absent the field is absent — a
missing gate is reported as missing, never as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from services.quant_api.config import ApiSettings, project_relative, safe_project_path


# Where each fact lives inside a run directory, relative to its root.
PIPELINE_REPORT = "reports/full_pipeline_report.json"
ACCEPTANCE_REPORT = "reports/acceptance_report.json"
BACKTEST_REPORT = "reports/walk_forward_backtest.json"
PAPER_REPORT = "reports/paper_report/paper_report.json"
GOVERNANCE_REPORT = "portfolio_search/selection_governance.json"
TRAINING_METRICS = "training/metrics.json"
RESEARCH_VERDICT = "research_verdict.json"
TARGET_WEIGHTS = "portfolio/target_weights.parquet"
PREDICTIONS = "predictions/predictions.parquet"

# The stages a strategy-pipeline run passes through, with the artifact that
# proves each one actually happened.
STAGE_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    ("dataset", "PIT 数据集", "dataset/training_dataset.parquet"),
    ("training", "滚动样本外训练", TRAINING_METRICS),
    ("prediction", "样本外预测", PREDICTIONS),
    ("portfolio", "目标权重", TARGET_WEIGHTS),
    ("backtest", "A 股回测", BACKTEST_REPORT),
    ("risk", "验收闸门", ACCEPTANCE_REPORT),
    ("evidence", "证据归档", PIPELINE_REPORT),
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


@dataclass
class RunResultResolver:
    settings: ApiSettings

    # ------------------------------------------------------------------
    def resolve(self, output_dir: str) -> dict[str, Any]:
        try:
            root = safe_project_path(self.settings, output_dir)
        except ValueError as exc:
            return {
                "status": "unavailable",
                "outputDir": output_dir,
                "issues": [{"code": "unsafe_path", "message": str(exc)}],
            }
        if not root.exists():
            return {
                "status": "absent",
                "outputDir": output_dir,
                "issues": [{
                    "code": "output_dir_missing",
                    "message": "运行目录不存在：任务可能尚未开始写入产物，或产物已被清理。",
                }],
                "stages": [
                    {"id": stage, "label": label, "present": False, "path": None}
                    for stage, label, _ in STAGE_EVIDENCE
                ],
            }

        verdict = _load_json(root / RESEARCH_VERDICT)
        pipeline = _load_json(root / PIPELINE_REPORT)
        acceptance = self._acceptance(root)
        governance = self._governance(root)
        training = self._training(root)
        backtest = self._backtest(root)
        paper = self._paper(root)
        candidates = self._candidates(root)
        stages = self._stages(root)
        artifacts = self._artifacts(root)

        produced = pipeline is not None
        if verdict:
            status = "rejected"
        elif produced:
            status = "complete"
        elif any(stage["present"] for stage in stages):
            status = "partial"
        else:
            status = "empty"

        conclusion = self._conclusion(
            status=status,
            verdict=verdict,
            acceptance=acceptance,
            governance=governance,
            pipeline=pipeline,
        )
        return {
            "status": status,
            "outputDir": project_relative(self.settings, root),
            "conclusion": conclusion,
            "verdict": verdict,
            "acceptance": acceptance,
            "governance": governance,
            "training": training,
            "backtest": backtest,
            "paper": paper,
            "candidates": candidates,
            "stages": stages,
            "artifacts": artifacts,
            "pipelineReport": pipeline,
        }

    # ------------------------------------------------------------------
    def _conclusion(
        self,
        *,
        status: str,
        verdict: dict[str, Any] | None,
        acceptance: dict[str, Any] | None,
        governance: dict[str, Any] | None,
        pipeline: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """One sentence an operator can act on, plus the reasons behind it."""
        if status == "rejected" and verdict:
            return {
                "outcome": "rejected",
                "headline": verdict.get("title") or "研究闸门否决了该候选",
                "reasons": list(verdict.get("reasons") or []),
                "remediation": verdict.get("remediation") or "",
                "promotable": False,
            }
        if status in {"empty", "absent"}:
            return {
                "outcome": "no_evidence",
                "headline": "该运行尚未产出可评估的证据",
                "reasons": [],
                "remediation": "任务仍在运行，或在产出任何产物前就中止了。",
                "promotable": False,
            }
        if status == "partial":
            return {
                "outcome": "incomplete",
                "headline": "运行未走完全部阶段，结论不完整",
                "reasons": ["缺少 full_pipeline_report.json"],
                "remediation": "查看任务失败诊断后决定是重试还是修改研究设计。",
                "promotable": False,
            }

        # "Accepted" is a claim about evidence, so it requires the evidence to
        # exist. A run whose acceptance report is missing has not passed those
        # gates — it simply has not been judged by them.
        missing_evidence = [
            name
            for name, present in (
                ("acceptance_report.json", acceptance is not None),
                ("selection_governance.json", governance is not None),
            )
            if not present
        ]
        if missing_evidence:
            return {
                "outcome": "incomplete",
                "headline": "流程走完，但缺少判定所需的证据",
                "reasons": [f"缺少 {name}" for name in missing_evidence],
                "remediation": (
                    "没有这些产物就无法判断闸门是否通过；缺失不等于通过。"
                    "检查运行日志确认该阶段是否被跳过或写入失败。"
                ),
                "promotable": False,
            }

        failures = list(acceptance.get("failures") or [])
        accepted = bool(governance.get("accepted", False))
        gate_status = (pipeline or {}).get("QUANT_ACCEPTANCE_STATUS")
        if failures or gate_status == "failed" or not accepted:
            reasons = [
                f"{gate['name']}: 实测 {gate['actual']}，阈值 {gate['threshold']}"
                for gate in acceptance.get("gates", [])
                if gate.get("passed") is False
            ]
            reasons.extend(str(item) for item in governance.get("rejectionReasons") or [])
            return {
                "outcome": "not_accepted",
                "headline": "流程完整跑通，但验收闸门未通过",
                "reasons": reasons or failures,
                "remediation": (
                    "这是研究结论而不是故障：产物齐全，可据此调整假设、成本假定或"
                    "组合约束后重新研究。闸门不允许事后放宽。"
                ),
                "promotable": False,
            }
        return {
            "outcome": "accepted",
            "headline": "流程完整跑通且通过全部已声明闸门",
            "reasons": [],
            "remediation": "仍为 research/paper 结论，晋级需要人工复核与独立验证。",
            "promotable": True,
        }

    # ------------------------------------------------------------------
    def _acceptance(self, root: Path) -> dict[str, Any] | None:
        payload = _load_json(root / ACCEPTANCE_REPORT)
        if not isinstance(payload, dict):
            return None
        gates = [
            {
                "name": gate.get("name"),
                "passed": bool(gate.get("passed")),
                "actual": gate.get("actual"),
                "threshold": gate.get("threshold"),
                "reason": gate.get("reason"),
            }
            for gate in payload.get("gates", [])
            if isinstance(gate, dict)
        ]
        return {
            "failures": list(payload.get("failures") or []),
            "gates": gates,
            "passedCount": sum(1 for gate in gates if gate["passed"]),
            "totalCount": len(gates),
            "sourcePath": project_relative(self.settings, root / ACCEPTANCE_REPORT),
        }

    def _governance(self, root: Path) -> dict[str, Any] | None:
        payload = _load_json(root / GOVERNANCE_REPORT)
        if not isinstance(payload, dict):
            return None
        return {
            "accepted": bool(payload.get("accepted")),
            "pbo": _finite(payload.get("pbo")),
            "dsrProbability": _finite(payload.get("dsr_probability")),
            "spaPValue": _finite(payload.get("spa_pvalue")),
            "losingFoldRate": _finite(payload.get("losing_fold_rate")),
            "observedDays": payload.get("observed_days"),
            "cumulativeTrials": payload.get("cumulative_trials"),
            "selectedCandidate": payload.get("selected_candidate"),
            "rejectionReasons": list(payload.get("rejection_reasons") or []),
            "sourcePath": project_relative(self.settings, root / GOVERNANCE_REPORT),
        }

    def _training(self, root: Path) -> dict[str, Any] | None:
        payload = _load_json(root / TRAINING_METRICS)
        if not isinstance(payload, dict):
            return None
        executable = payload.get("executable_backtest")
        executable = executable if isinstance(executable, dict) else {}
        data_range = payload.get("data_range")
        data_range = data_range if isinstance(data_range, dict) else {}
        evaluated_days = payload.get("evaluated_days")
        return {
            "backend": payload.get("backend"),
            "rankIcMean": _finite(payload.get("rank_ic_mean")),
            "icir": _finite(payload.get("ICIR")),
            "hitRate": _finite(payload.get("hit_rate")),
            "foldCount": payload.get("fold_count"),
            "featureCount": payload.get("feature_count"),
            "evaluatedDays": evaluated_days,
            "dataRange": {"start": data_range.get("start"), "end": data_range.get("end")},
            "annualisedReturn": _finite(payload.get("annualised_return")),
            "annualisedSharpe": _finite(payload.get("annualised_sharpe")),
            "excessReturnAfterCosts": _finite(payload.get("excess_return_after_costs")),
            "adverseRegimePassed": payload.get("adverse_regime_passed"),
            "executable": {
                "annualisedReturnPct": _finite(executable.get("annualised_return_pct")),
                "annualisedVolPct": _finite(executable.get("annualised_vol_pct")),
                "excessAnnualisedPct": _finite(executable.get("excess_annualised_pct")),
                "maxDrawdownPct": _finite(executable.get("max_drawdown_pct")),
                "averageTurnover": _finite(executable.get("average_turnover")),
                "costBps": _finite(executable.get("executable_cost_bps")),
                "benchmarkLabel": executable.get("benchmark_label"),
                "benchmarkAnnualisedPct": _finite(executable.get("benchmark_annualised_pct")),
                "status": executable.get("executable_backtest_status"),
            } if executable else None,
            # Annualising a short window produces impressive-looking ratios.
            # The window travels with the numbers so nobody reads a 48-day
            # Sharpe as a durable one.
            "annualisationWarning": (
                f"这些年化数字由 {evaluated_days} 个评估日推算，样本越短越不稳定。"
                if isinstance(evaluated_days, (int, float)) and evaluated_days < 250
                else None
            ),
            "sourcePath": project_relative(self.settings, root / TRAINING_METRICS),
        }

    def _backtest(self, root: Path) -> dict[str, Any] | None:
        payload = _load_json(root / BACKTEST_REPORT)
        if not isinstance(payload, dict):
            return None
        # The simulator writes nav as a {date: value} mapping; older/other
        # producers write a list of records. Both are accepted, and anything
        # else yields an empty curve rather than a fabricated one.
        nav = payload.get("nav")
        curve: list[dict[str, Any]] = []
        if isinstance(nav, dict):
            for date, value in nav.items():
                number = _finite(value)
                if number is not None:
                    curve.append({"date": str(date)[:10], "nav": number})
            curve.sort(key=lambda item: item["date"])
        elif isinstance(nav, list):
            for point in nav:
                if not isinstance(point, dict):
                    continue
                value = _finite(point.get("nav") or point.get("value") or point.get("equity"))
                date = point.get("trade_date") or point.get("date") or point.get("datetime")
                if value is not None and date is not None:
                    curve.append({"date": str(date)[:10], "nav": value})
        total_return = None
        max_drawdown = None
        if curve:
            start, end = curve[0]["nav"], curve[-1]["nav"]
            if start:
                total_return = round(end / start - 1.0, 6)
            peak = curve[0]["nav"]
            worst = 0.0
            for point in curve:
                peak = max(peak, point["nav"])
                if peak:
                    worst = min(worst, point["nav"] / peak - 1.0)
            max_drawdown = round(worst, 6)
        orders = payload.get("orders")
        skipped = payload.get("skipped_orders")
        failed = payload.get("failed_orders")
        return {
            "navPoints": len(curve),
            "nav": curve[-2_000:],
            "totalReturn": total_return,
            "maxDrawdown": max_drawdown,
            "orderCount": len(orders) if isinstance(orders, list) else orders,
            "skippedOrderCount": len(skipped) if isinstance(skipped, list) else skipped,
            "failedOrderCount": len(failed) if isinstance(failed, list) else failed,
            "sourcePath": project_relative(self.settings, root / BACKTEST_REPORT),
        }

    def _candidates(self, root: Path) -> list[dict[str, Any]]:
        """Every portfolio candidate the run evaluated, with its own numbers.

        The search evaluates a bounded grid of candidates on the same cost model
        and keeps one. Showing only the winner makes the choice look arbitrary;
        the losing candidates are what make it legible — and they are what the
        overfitting gate counts as trials.
        """
        search_root = root / "portfolio_search"
        if not search_root.is_dir():
            return []
        governance = _load_json(root / GOVERNANCE_REPORT)
        selected = (governance or {}).get("selected_candidate") if isinstance(governance, dict) else None

        candidates: list[dict[str, Any]] = []
        for directory in sorted(item for item in search_root.iterdir() if item.is_dir()):
            report = _load_json(directory / "paper_report.json")
            summary = report.get("summary") if isinstance(report, dict) else None
            summary = summary if isinstance(summary, dict) else {}
            candidates.append({
                "id": directory.name,
                "selected": directory.name == selected,
                "annualisedReturn": _finite(summary.get("annualized_return")),
                "netReturnAfterCosts": _finite(summary.get("net_return_after_estimated_costs")),
                "excessReturnAfterCosts": _finite(summary.get("excess_return_after_costs")),
                "maxDrawdown": _finite(summary.get("max_drawdown")),
                "sharpe": _finite(summary.get("sharpe")),
                "informationRatio": _finite(summary.get("information_ratio")),
                "tradeCount": summary.get("trade_count"),
                "skippedOrderCount": summary.get("skipped_order_count"),
                "estimatedCosts": _finite(
                    (summary.get("total_estimated_fees") or 0)
                    + (summary.get("total_estimated_slippage") or 0)
                ) or None,
                "acceptanceStatus": report.get("quant_acceptance_status") if isinstance(report, dict) else None,
                "sourcePath": project_relative(self.settings, directory / "paper_report.json"),
            })
        return candidates

    def _paper(self, root: Path) -> dict[str, Any] | None:
        payload = _load_json(root / PAPER_REPORT)
        if not isinstance(payload, dict):
            return None
        return {
            "status": payload.get("status"),
            "acceptanceStatus": payload.get("quant_acceptance_status"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else None,
            "warnings": list(payload.get("warnings") or []),
            "sourcePath": project_relative(self.settings, root / PAPER_REPORT),
        }

    def _stages(self, root: Path) -> list[dict[str, Any]]:
        stages: list[dict[str, Any]] = []
        for stage_id, label, relative in STAGE_EVIDENCE:
            path = root / relative
            present = path.exists()
            stages.append({
                "id": stage_id,
                "label": label,
                "present": present,
                "path": project_relative(self.settings, path) if present else relative,
                "sizeBytes": path.stat().st_size if present else None,
            })
        return stages

    def _artifacts(self, root: Path, limit: int = 400) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append({
                "name": path.name,
                "path": project_relative(self.settings, path),
                "relative": path.relative_to(root).as_posix(),
                "sizeBytes": stat.st_size,
                "extension": path.suffix.lower(),
            })
            if len(items) >= limit:
                break
        return items


__all__ = ["RunResultResolver", "STAGE_EVIDENCE"]
