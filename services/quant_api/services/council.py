"""The ATLAS decision council: role-scoped structural review of research runs.

The council is not a chat layer and it does not generate prose. Each agent owns
one domain, reads only structured evidence (artifact fields, counts, hashes),
and returns a verdict plus the evidence it used. An agent may only block inside
its own veto scope, which keeps a data-quality objection from silently vetoing a
portfolio decision.

Three rules make the council trustworthy rather than decorative:

1. **A verdict always names its evidence.** Every finding carries the fields it
   was computed from, so an operator can check the reasoning rather than trust
   the badge.
2. **Absence of evidence is not a pass.** A check whose inputs are missing
   returns ``unknown``, never ``pass``. ``unknown`` does not block promotion but
   it is never counted as clearance either.
3. **Overrides are recorded, not hidden.** A human can overrule any agent, but
   the override is appended to a durable log with author, timestamp, and the
   verdict it replaced. Nothing in this module can delete that log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Literal

Verdict = Literal["pass", "warn", "blocked", "unknown"]

# Ordered: the council is read top to bottom, data first and governance last.
COUNCIL_ROLES: tuple[dict[str, Any], ...] = (
    {
        "id": "data_quality",
        "label": "数据质量",
        "domain": "PIT 完整性、provenance、复权口径、基准口径",
        "vetoScope": "输入数据不可信时阻塞整条链",
        "veto": True,
    },
    {
        "id": "factor_integrity",
        "label": "因子完整性",
        "domain": "因子冗余、单因子支配、融合是否带来增量",
        "vetoScope": "因子入池",
        "veto": True,
    },
    {
        "id": "model_validation",
        "label": "模型验证",
        "domain": "折切分、embargo、训练/测试隔离",
        "vetoScope": "模型与权重晋级",
        "veto": True,
    },
    {
        "id": "fusion_search",
        "label": "搜索统计",
        "domain": "试验计数、PBO、收缩后显著性、前沿合法性",
        "vetoScope": "融合候选晋级",
        "veto": True,
    },
    {
        "id": "portfolio_risk",
        "label": "组合风险",
        "domain": "回撤、换手、集中度、容量",
        "vetoScope": "目标权重发布",
        "veto": True,
    },
    {
        "id": "execution_realism",
        "label": "执行可实现性",
        "domain": "成本、T+1、涨跌停、可卖库存",
        "vetoScope": "回测可实现性主张",
        "veto": True,
    },
    {
        "id": "governance",
        "label": "治理",
        "domain": "readiness tier、人工 Gate、审计链",
        "vetoScope": "任何 live 意图",
        "veto": True,
    },
)

ROLE_IDS = tuple(role["id"] for role in COUNCIL_ROLES)


@dataclass(frozen=True)
class CouncilThresholds:
    """Promotion bars. Operator-visible, and every one of them is checked."""

    max_pbo: float = 0.50
    min_deflated_sharpe: float = 0.50
    min_observations: int = 60
    min_folds: int = 3
    max_drawdown: float = 0.25
    max_turnover: float = 1.00
    min_transaction_cost_bps: float = 1.0
    min_factor_count: int = 2

    def as_dict(self) -> dict[str, float | int]:
        return {
            "maxPbo": self.max_pbo,
            "minDeflatedSharpe": self.min_deflated_sharpe,
            "minObservations": self.min_observations,
            "minFolds": self.min_folds,
            "maxDrawdown": self.max_drawdown,
            "maxTurnover": self.max_turnover,
            "minTransactionCostBps": self.min_transaction_cost_bps,
            "minFactorCount": self.min_factor_count,
        }


def _finding(
    role_id: str,
    verdict: Verdict,
    headline: str,
    detail: str,
    evidence: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    return {
        "roleId": role_id,
        "verdict": verdict,
        "headline": headline,
        "detail": detail,
        "evidence": evidence,
        "nextAction": next_action,
    }


class CouncilService:
    """Assembles council reviews and owns the override audit log."""

    def __init__(self, settings, fusion_adapter, *, thresholds: CouncilThresholds | None = None) -> None:
        self.settings = settings
        self.fusion = fusion_adapter
        self.thresholds = thresholds or CouncilThresholds()
        self._log_path = Path(settings.jobs_root) / "council_overrides.jsonl"

    # ------------------------------------------------------------- roster --

    def roster(self) -> dict[str, Any]:
        return {
            "roles": [dict(role) for role in COUNCIL_ROLES],
            "thresholds": self.thresholds.as_dict(),
            "protocol": (
                "每个角色只在自身职责域内否决；证据缺失记为 unknown，不记为通过；"
                "人工推翻会写入不可删除的审计日志。"
            ),
        }

    # ------------------------------------------------------------- review --

    def review_fusion_run(self, run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        """Review one fusion search run, optionally focused on one candidate."""
        detail = self.fusion.detail(run_id)
        summary = detail.get("summary") or {}
        candidates = detail.get("candidates") or []
        frontier = [item for item in candidates if item.get("onFrontier")]
        subject = None
        if candidate_id:
            subject = next(
                (item for item in candidates if str(item.get("id")) == candidate_id), None
            )
            if subject is None:
                raise KeyError(candidate_id)
        else:
            subject = frontier[0] if frontier else (candidates[0] if candidates else None)

        findings = [
            check(summary, subject, candidates, self.thresholds)
            for check in (
                _review_data_quality,
                _review_factor_integrity,
                _review_model_validation,
                _review_fusion_search,
                _review_portfolio_risk,
                _review_execution_realism,
                _review_governance,
            )
        ]
        overrides = self.overrides(subject_type="fusion_run", subject_id=run_id)
        latest_override = {
            item["roleId"]: item
            for item in sorted(overrides, key=lambda row: str(row.get("recordedAt") or ""))
        }
        for finding in findings:
            override = latest_override.get(finding["roleId"])
            if override:
                finding["override"] = {
                    "verdict": override["verdict"],
                    "reason": override["reason"],
                    "author": override["author"],
                    "recordedAt": override["recordedAt"],
                    "replacedVerdict": finding["verdict"],
                }

        return {
            "subject": {
                "type": "fusion_run",
                "id": run_id,
                "path": detail.get("path"),
                "candidateId": subject.get("id") if subject else None,
                "candidateLabel": subject.get("label") if subject else None,
            },
            "roles": [dict(role) for role in COUNCIL_ROLES],
            "thresholds": self.thresholds.as_dict(),
            "findings": findings,
            "decision": _aggregate_decision(findings),
            "overrides": overrides,
        }

    # ---------------------------------------------------------- overrides --

    def overrides(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if subject_type and record.get("subjectType") != subject_type:
                continue
            if subject_id and record.get("subjectId") != subject_id:
                continue
            records.append(record)
        return records

    def record_override(
        self,
        *,
        subject_type: str,
        subject_id: str,
        role_id: str,
        verdict: str,
        reason: str,
        author: str,
    ) -> dict[str, Any]:
        """Append a human override. The log is append-only by construction."""
        if role_id not in ROLE_IDS:
            raise ValueError(f"unknown council role: {role_id}")
        if verdict not in {"pass", "warn", "blocked"}:
            raise ValueError("override verdict must be pass, warn or blocked")
        reason = reason.strip()
        if len(reason) < 8:
            raise ValueError("override reason must explain the decision (>= 8 characters)")
        author = author.strip()
        if not author:
            raise ValueError("override author is required")

        record = {
            "subjectType": subject_type,
            "subjectId": subject_id,
            "roleId": role_id,
            "verdict": verdict,
            "reason": reason,
            "author": author,
            "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


# --------------------------------------------------------------------------- #
# Role checks. Each reads only what it declares in `evidence`.                 #
# --------------------------------------------------------------------------- #

CheckFn = Callable[
    [dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], CouncilThresholds],
    dict[str, Any],
]


def _metric(candidate: dict[str, Any] | None, key: str) -> float | None:
    if not candidate:
        return None
    value = (candidate.get("metrics") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _review_data_quality(summary, subject, candidates, thresholds) -> dict[str, Any]:
    benchmark = summary.get("benchmarkMode")
    observations = _metric(subject, "observations")
    evidence = {"benchmarkMode": benchmark, "observations": observations}
    if benchmark is None:
        return _finding(
            "data_quality", "unknown", "基准口径未记录",
            "产物没有写入 benchmarkMode，无法判断超额收益相对什么计算。",
            evidence, "重新运行搜索以写入完整 manifest",
        )
    if observations is None or observations < thresholds.min_observations:
        return _finding(
            "data_quality", "blocked", "样本外观测不足",
            f"候选只有 {observations if observations is not None else 0} 个样本外观测，"
            f"低于 {thresholds.min_observations} 的最低要求。",
            evidence, "延长面板区间或降低最少测试日要求",
        )
    if str(benchmark).startswith("universe_equal_weight"):
        return _finding(
            "data_quality", "warn", "基准为宇宙等权而非指数",
            "等权宇宙基准会包含不可交易标的（停牌、ST、一字涨跌停），"
            "超额收益会被系统性高估。晋级前应改用可交易指数基准。",
            evidence, "提供指数基准序列后重跑",
        )
    return _finding(
        "data_quality", "pass", "输入口径可追溯",
        f"基准 {benchmark}，样本外观测 {int(observations)}。",
        evidence, "无",
    )


def _review_factor_integrity(summary, subject, candidates, thresholds) -> dict[str, Any]:
    factors = summary.get("factorNames") or []
    subject_id = str((subject or {}).get("id") or "")
    # The subject is excluded from its own baseline set: a single-factor
    # candidate cannot be faulted for failing to beat itself.
    singles = [
        item for item in candidates
        if str(item.get("scheme")) == "single_factor"
        and str(item.get("id")) != subject_id
        and int((item.get("metrics") or {}).get("observations") or 0) > 0
    ]
    if str((subject or {}).get("scheme")) == "single_factor":
        best_other = max(
            (_metric(item, "excessReturn") or float("-inf") for item in singles),
            default=float("-inf"),
        )
        evidence = {
            "factorCount": len(factors),
            "subjectIsSingleFactor": True,
            "candidateExcessReturn": _metric(subject, "excessReturn"),
            "bestOtherSingleFactorExcessReturn":
                None if best_other == float("-inf") else best_other,
        }
        return _finding(
            "factor_integrity", "warn", "候选本身是单因子，不构成融合",
            "该候选没有融合任何因子。它可以作为基线结论保留，但不应作为融合策略晋级。",
            evidence, "改用融合候选，或接受单因子这一结论",
        )
    subject_excess = _metric(subject, "excessReturn")
    best_single = max(
        (_metric(item, "excessReturn") or float("-inf") for item in singles),
        default=None,
    )
    evidence = {
        "factorCount": len(factors),
        "singleFactorBaselines": len(singles),
        "candidateExcessReturn": subject_excess,
        "bestSingleFactorExcessReturn": None if best_single in (None, float("-inf")) else best_single,
    }
    if len(factors) < thresholds.min_factor_count:
        return _finding(
            "factor_integrity", "blocked", "参与融合的因子过少",
            f"只有 {len(factors)} 个因子，无法构成融合；这实际上是单因子策略。",
            evidence, "增加已审核因子后重跑",
        )
    if not singles:
        return _finding(
            "factor_integrity", "unknown", "缺少单因子基线",
            "本次搜索没有生成单因子基线，无法判断融合是否带来增量。",
            evidence, "把单因子基线数设为大于 0 后重跑",
        )
    if subject_excess is None or best_single in (None, float("-inf")):
        return _finding(
            "factor_integrity", "unknown", "无法比较融合与单因子",
            "候选或单因子基线缺少可用的超额收益。",
            evidence, "检查标签覆盖率",
        )
    if subject_excess <= best_single:
        return _finding(
            "factor_integrity", "blocked", "融合没有跑赢最好的单因子",
            f"候选超额 {subject_excess:.4f} 未超过最好单因子基线 {best_single:.4f}；"
            "融合在此配置下没有增量，复杂度不成立。",
            evidence, "改用单因子，或换一组更互补的因子",
        )
    return _finding(
        "factor_integrity", "pass", "融合优于最好单因子",
        f"候选超额 {subject_excess:.4f} 高于最好单因子 {best_single:.4f}。",
        evidence, "无",
    )


def _review_model_validation(summary, subject, candidates, thresholds) -> dict[str, Any]:
    folds = summary.get("foldWindows") or []
    horizon = summary.get("horizonDays")
    evidence = {"foldCount": len(folds), "horizonDays": horizon}
    if not folds:
        return _finding(
            "model_validation", "unknown", "折窗口未记录",
            "产物没有 foldWindows，无法验证训练与测试是否隔离。",
            evidence, "重新运行搜索以写入折窗口",
        )
    if len(folds) < thresholds.min_folds:
        return _finding(
            "model_validation", "warn", "折数偏少",
            f"只有 {len(folds)} 折，低于建议的 {thresholds.min_folds} 折；"
            "折间一致性的统计意义有限。",
            evidence, "提高折数后重跑",
        )
    overlaps = [
        {"foldIndex": fold.get("foldIndex"), "trainEnd": fold.get("trainEnd"), "testStart": fold.get("testStart")}
        for fold in folds
        if str(fold.get("testStart") or "") <= str(fold.get("trainEnd") or "")
    ]
    evidence["overlappingFolds"] = overlaps
    if overlaps:
        return _finding(
            "model_validation", "blocked", "训练段与测试段重叠",
            f"{len(overlaps)} 折的测试起点不晚于训练终点，样本外结论无效。",
            evidence, "检查 embargo 与折切分实现",
        )
    return _finding(
        "model_validation", "pass", "折切分与隔离成立",
        f"{len(folds)} 折，测试段均严格晚于训练段。",
        evidence, "无",
    )


def _review_fusion_search(summary, subject, candidates, thresholds) -> dict[str, Any]:
    trials = summary.get("nTrials")
    pbo = summary.get("pbo")
    breakdown = (subject or {}).get("robustnessBreakdown") or {}
    dsr = breakdown.get("deflatedSharpeProbability")
    evidence = {
        "nTrials": trials,
        "pbo": pbo,
        "deflatedSharpeProbability": dsr,
        "evaluatedCandidates": summary.get("evaluatedCandidateCount"),
    }
    if trials is None:
        return _finding(
            "fusion_search", "unknown", "试验次数未记录",
            "没有 nTrials 就无法收缩 Sharpe，任何显著性主张都不可信。",
            evidence, "重新运行搜索",
        )
    if pbo is None:
        return _finding(
            "fusion_search", "warn", "PBO 无估计",
            "样本外时间切片不足以做组合对称交叉验证；抗过拟合项按无证据计入，"
            "不能当作通过。",
            evidence, "延长样本外区间后重跑",
        )
    if float(pbo) > thresholds.max_pbo:
        return _finding(
            "fusion_search", "blocked", "过拟合概率过高",
            f"PBO {float(pbo):.3f} 超过上限 {thresholds.max_pbo:.2f}；"
            "样本内冠军很可能只是选择偏差的产物。",
            evidence, "缩小搜索空间或延长样本外区间",
        )
    if isinstance(dsr, (int, float)) and float(dsr) < thresholds.min_deflated_sharpe:
        return _finding(
            "fusion_search", "warn", "收缩后显著性偏低",
            f"按 {trials} 次试验收缩后的 Sharpe 显著性只有 {float(dsr):.3f}，"
            f"低于 {thresholds.min_deflated_sharpe:.2f}。",
            evidence, "减少试验或提高单候选质量",
        )
    return _finding(
        "fusion_search", "pass", "统计口径成立",
        f"{trials} 次试验，PBO {float(pbo):.3f}。",
        evidence, "无",
    )


def _review_portfolio_risk(summary, subject, candidates, thresholds) -> dict[str, Any]:
    drawdown = _metric(subject, "maxDrawdown")
    turnover = _metric(subject, "averageTurnover")
    top_k = summary.get("topK")
    evidence = {"maxDrawdown": drawdown, "averageTurnover": turnover, "topK": top_k}
    if drawdown is None:
        return _finding(
            "portfolio_risk", "unknown", "无回撤证据",
            "候选没有可用的回撤指标。", evidence, "检查候选是否产生了样本外观测",
        )
    if drawdown > thresholds.max_drawdown:
        return _finding(
            "portfolio_risk", "blocked", "最大回撤超过限额",
            f"回撤 {drawdown:.2%} 超过 {thresholds.max_drawdown:.0%} 上限。"
            "注意该回撤按调仓频率净值计算，日频标记只会更深。",
            evidence, "提高回撤偏好权重或收紧约束后重跑",
        )
    if turnover is not None and turnover > thresholds.max_turnover:
        return _finding(
            "portfolio_risk", "warn", "换手偏高",
            f"平均换手 {turnover:.2f} 超过 {thresholds.max_turnover:.2f}；"
            "成本与冲击对结论的敏感度显著上升。",
            evidence, "提高成本假设复核稳健性",
        )
    return _finding(
        "portfolio_risk", "pass", "风险指标在限额内",
        f"回撤 {drawdown:.2%}"
        + (f"，换手 {turnover:.2f}" if turnover is not None else ""),
        evidence, "无",
    )


def _review_execution_realism(summary, subject, candidates, thresholds) -> dict[str, Any]:
    cost = summary.get("transactionCostBps")
    horizon = summary.get("horizonDays")
    cost_drag = _metric(subject, "costDrag")
    evidence = {"transactionCostBps": cost, "horizonDays": horizon, "costDrag": cost_drag}
    if cost is None:
        return _finding(
            "execution_realism", "unknown", "成本假设未记录",
            "无法判断结论是否已扣除交易成本。", evidence, "重新运行搜索",
        )
    if float(cost) < thresholds.min_transaction_cost_bps:
        return _finding(
            "execution_realism", "blocked", "成本假设不成立",
            f"成本 {float(cost)} bps 低于最低 {thresholds.min_transaction_cost_bps} bps；"
            "A 股实际佣金、印花税与冲击不可能低于此。",
            evidence, "使用真实成本重跑",
        )
    if horizon is not None and int(horizon) < 2:
        return _finding(
            "execution_realism", "warn", "持有周期与 T+1 边界接近",
            "1 日持有周期在 T+1 制度下没有日内退出空间，"
            "实际可执行性需要单独用可卖库存约束验证。",
            evidence, "在 T+1 实验室验证可卖库存约束",
        )
    return _finding(
        "execution_realism", "pass", "成本与周期假设成立",
        f"成本 {float(cost)} bps，持有周期 {horizon} 日。",
        evidence, "无",
    )


def _review_governance(summary, subject, candidates, thresholds) -> dict[str, Any]:
    generated = summary.get("generatedAt")
    evidence = {
        "generatedAt": generated,
        "mode": "RESEARCH",
        "liveIntent": False,
        "candidateIsControl": bool((subject or {}).get("isControl")),
    }
    if (subject or {}).get("isControl"):
        return _finding(
            "governance", "warn", "首选候选是对照组",
            "当前首选是一个不读取训练段的对照方案。这是一个有效的研究结论"
            "（拟合方案没有赢过基线），但它不构成可晋级的策略。",
            evidence, "接受该负面结论，或更换因子集合重跑",
        )
    if not generated:
        return _finding(
            "governance", "unknown", "产物缺少生成时间",
            "无法把该结论固定到审计链上。", evidence, "重新运行搜索",
        )
    return _finding(
        "governance", "pass", "研究态、无实盘意图",
        "本产物处于 RESEARCH 模式，未产生任何订单意图，可进入人工 Gate。",
        evidence, "人工复核后决定是否晋级",
    )


def _aggregate_decision(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Council-level outcome. An override replaces the verdict it names."""
    effective = [
        (item["override"]["verdict"] if item.get("override") else item["verdict"])
        for item in findings
    ]
    blocked = [
        item["roleId"]
        for item, verdict in zip(findings, effective)
        if verdict == "blocked"
    ]
    unknown = [
        item["roleId"]
        for item, verdict in zip(findings, effective)
        if verdict == "unknown"
    ]
    warned = [
        item["roleId"]
        for item, verdict in zip(findings, effective)
        if verdict == "warn"
    ]
    if blocked:
        state = "BLOCKED"
        summary = f"{len(blocked)} 个角色否决：{', '.join(blocked)}"
    elif unknown:
        state = "INSUFFICIENT_EVIDENCE"
        summary = f"{len(unknown)} 个角色证据不足：{', '.join(unknown)}"
    elif warned:
        state = "PROMOTABLE_WITH_WARNINGS"
        summary = f"{len(warned)} 个角色提出保留意见：{', '.join(warned)}"
    else:
        state = "PROMOTABLE"
        summary = "全部角色通过；仍需人工 Gate 才能进入下一层。"
    return {
        "state": state,
        "summary": summary,
        "blockedRoles": blocked,
        "unknownRoles": unknown,
        "warnedRoles": warned,
        "overriddenRoles": [
            item["roleId"] for item in findings if item.get("override")
        ],
    }


__all__ = ["COUNCIL_ROLES", "ROLE_IDS", "CouncilService", "CouncilThresholds"]
