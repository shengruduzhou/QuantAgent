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
   returns ``unknown``, never ``pass``. Research may continue, but ``unknown``
   blocks promotion clearance until the evidence exists.
3. **Overrides are recorded, not hidden.** A human can overrule any agent, but
   the override is appended to a durable log with author, timestamp, and the
   verdict it replaced. Nothing in this module can delete that log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

Verdict = Literal["pass", "warn", "blocked", "unknown"]

COUNCIL_PROTOCOL_VERSION = 2
COUNCIL_PROTOCOL_EFFECTIVE_AT = "2026-08-27"

# Ordered: the council is read top to bottom, data first and governance last.
COUNCIL_ROLES: tuple[dict[str, Any], ...] = (
    {
        "id": "data_acquisition",
        "label": "数据采购",
        "domain": "供应商、抓取批次、时间戳与输入产物完整性",
        "vetoScope": "数据来源与采集证据",
        "veto": True,
        "phase": "data_admission",
        "introducedInProtocol": 2,
    },
    {
        "id": "data_quality",
        "label": "数据质量",
        "domain": "PIT 完整性、provenance、复权口径、基准口径",
        "vetoScope": "输入数据不可信时阻塞整条链",
        "veto": True,
        "phase": "data_admission",
        "introducedInProtocol": 1,
    },
    {
        "id": "microstructure",
        "label": "市场微观结构",
        "domain": "频率、时钟、撮合粒度与日内假设适用边界",
        "vetoScope": "日内或微观结构相关主张",
        "veto": True,
        "phase": "data_admission",
        "introducedInProtocol": 2,
    },
    {
        "id": "factor_integrity",
        "label": "因子完整性",
        "domain": "因子冗余、单因子支配、融合是否带来增量",
        "vetoScope": "因子入池",
        "veto": True,
        "phase": "research_validation",
        "introducedInProtocol": 1,
    },
    {
        "id": "model_validation",
        "label": "模型验证",
        "domain": "折切分、embargo、训练/测试隔离",
        "vetoScope": "模型与权重晋级",
        "veto": True,
        "phase": "research_validation",
        "introducedInProtocol": 1,
    },
    {
        "id": "fusion_search",
        "label": "搜索统计",
        "domain": "试验计数、PBO、收缩后显著性、前沿合法性",
        "vetoScope": "融合候选晋级",
        "veto": True,
        "phase": "research_validation",
        "introducedInProtocol": 1,
    },
    {
        "id": "portfolio_risk",
        "label": "组合风险",
        "domain": "回撤、换手、集中度、容量",
        "vetoScope": "目标权重发布",
        "veto": True,
        "phase": "portfolio_delivery",
        "introducedInProtocol": 1,
    },
    {
        "id": "execution_realism",
        "label": "执行可实现性",
        "domain": "成本、T+1、涨跌停、可卖库存",
        "vetoScope": "回测可实现性主张",
        "veto": True,
        "phase": "portfolio_delivery",
        "introducedInProtocol": 1,
    },
    {
        "id": "challenger",
        "label": "独立挑战者",
        "domain": "基线、随机对照、负面结论与替代解释",
        "vetoScope": "未经过对照挑战的候选晋级",
        "veto": True,
        "phase": "independent_decision",
        "introducedInProtocol": 2,
    },
    {
        "id": "compliance",
        "label": "合规与模型风险",
        "domain": "研究/生产边界、授权范围、模型风险披露",
        "vetoScope": "越权或 production/live 声明",
        "veto": True,
        "phase": "independent_decision",
        "introducedInProtocol": 2,
    },
    {
        "id": "governance",
        "label": "CIO / 决策主席",
        "domain": "汇总各部门裁决、readiness tier、人工 Gate 与审计链",
        "vetoScope": "公司级晋级结论与任何 live 意图",
        "veto": True,
        "phase": "independent_decision",
        "introducedInProtocol": 1,
    },
)

ROLE_IDS = tuple(role["id"] for role in COUNCIL_ROLES)


@dataclass(frozen=True)
class CouncilThresholds:
    """Promotion bars. Operator-visible, and every one of them is checked."""

    max_pbo: float = 0.25
    min_deflated_sharpe: float = 0.95
    max_spa_pvalue: float = 0.05
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
            "maxSpaPValue": self.max_spa_pvalue,
            "minObservations": self.min_observations,
            "minFolds": self.min_folds,
            "maxDrawdown": self.max_drawdown,
            "maxTurnover": self.max_turnover,
            "minTransactionCostBps": self.min_transaction_cost_bps,
            "minFactorCount": self.min_factor_count,
        }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_fingerprint(thresholds: CouncilThresholds) -> str:
    return _canonical_hash({
        "protocolVersion": COUNCIL_PROTOCOL_VERSION,
        "roles": COUNCIL_ROLES,
        "thresholds": thresholds.as_dict(),
    })


COUNCIL_POLICY_FINGERPRINT = _policy_fingerprint(CouncilThresholds())


def _finding(
    role_id: str,
    verdict: Verdict,
    headline: str,
    detail: str,
    evidence: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    finding = {
        "roleId": role_id,
        "verdict": verdict,
        "headline": headline,
        "detail": detail,
        "evidence": evidence,
        "nextAction": next_action,
    }
    finding["findingHash"] = _canonical_hash(finding)
    return finding


class CouncilService:
    """Assembles council reviews and owns the override audit log."""

    def __init__(self, settings, fusion_adapter, *, thresholds: CouncilThresholds | None = None) -> None:
        self.settings = settings
        self.fusion = fusion_adapter
        self.thresholds = thresholds or CouncilThresholds()
        self._log_path = Path(settings.jobs_root) / "council_overrides.jsonl"

    @property
    def policy_fingerprint(self) -> str:
        return _policy_fingerprint(self.thresholds)

    def _protocol_envelope(self) -> dict[str, Any]:
        return {
            "protocolVersion": COUNCIL_PROTOCOL_VERSION,
            "policyFingerprint": self.policy_fingerprint,
            "protocolEffectiveAt": COUNCIL_PROTOCOL_EFFECTIVE_AT,
        }

    def _apply_overrides(
        self,
        *,
        findings: list[dict[str, Any]],
        records: list[dict[str, Any]],
        subject_content_hash: str,
        candidate_id: str | None,
    ) -> list[dict[str, Any]]:
        """Classify every audit row and apply only an exact v2 scope match.

        Historical rows remain visible, but a change to the candidate, artifact,
        role finding, thresholds, or protocol makes them stale. This prevents an
        override written for candidate A (or yesterday's bytes) from clearing a
        different decision merely because the run id is unchanged.
        """
        by_role = {item["roleId"]: item for item in findings}
        annotated: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            status = "effective"
            if record.get("protocolVersion") != COUNCIL_PROTOCOL_VERSION:
                status = "historical_protocol"
            elif record.get("policyFingerprint") != self.policy_fingerprint:
                status = "stale_policy"
            elif record.get("subjectContentHash") != subject_content_hash:
                status = "stale_subject"
            elif record.get("candidateId") != candidate_id:
                status = "stale_candidate"
            else:
                finding = by_role.get(str(record.get("roleId")))
                if finding is None:
                    status = "stale_role"
                elif record.get("findingHash") != finding.get("findingHash"):
                    status = "stale_finding"
                elif record.get("originalVerdict") != finding.get("verdict"):
                    status = "stale_original_verdict"
                elif finding.get("verdict") == "unknown" and record.get("verdict") == "pass":
                    status = "inadmissible_missing_evidence"
            record["scopeStatus"] = status
            record["effective"] = status == "effective"
            annotated.append(record)

        latest: dict[str, dict[str, Any]] = {}
        for record in sorted(annotated, key=lambda row: str(row.get("recordedAt") or "")):
            if record["effective"]:
                latest[str(record["roleId"])] = record
        for finding in findings:
            override = latest.get(finding["roleId"])
            if override:
                finding["override"] = {
                    "verdict": override["verdict"],
                    "reason": override["reason"],
                    "author": override["author"],
                    "recordedAt": override["recordedAt"],
                    "replacedVerdict": override["originalVerdict"],
                    "findingHash": override["findingHash"],
                }
        return annotated

    # ------------------------------------------------------------- roster --

    def roster(self) -> dict[str, Any]:
        return {
            **self._protocol_envelope(),
            "roles": [dict(role) for role in COUNCIL_ROLES],
            "thresholds": self.thresholds.as_dict(),
            "protocol": (
                "Council v2 固定 11 岗与四阶段；证据缺失记为 unknown 并阻止晋级；"
                "人工推翻只对同一候选、同一产物 hash、同一 finding 与同一 policy 生效。"
            ),
            "migration": (
                "v1 七岗裁决与无版本 override 仅保留为历史记录，不参与 v2 effective decision。"
            ),
        }

    # ------------------------------------------------------------- review --

    def review_fusion_run(self, run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        """Review one fusion search run, optionally focused on one candidate."""
        detail = self.fusion.detail(run_id)
        summary = detail.get("summary") or {}
        # promotion_gate.json is the canonical PBO/DSR/SPA and research/live
        # boundary evidence. Keep it out of persisted summary files but make it
        # available to the role checks in this review invocation.
        summary = {
            **summary,
            "_promotionGate": detail.get("promotionGate"),
            "_manifest": detail.get("manifest"),
        }
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
                _review_data_acquisition,
                _review_data_quality,
                _review_microstructure,
                _review_factor_integrity,
                _review_model_validation,
                _review_fusion_search,
                _review_portfolio_risk,
                _review_execution_realism,
                _review_challenger,
                _review_compliance,
            )
        ]
        findings.append(
            _review_governance(summary, subject, candidates, self.thresholds, findings)
        )
        subject_content_hash = _canonical_hash({
            "manifest": detail.get("manifest"),
            "summary": detail.get("summary"),
            "promotionGate": detail.get("promotionGate"),
            "ranking": detail.get("ranking"),
            "candidates": detail.get("candidates"),
        })
        candidate_scope = str(subject.get("id")) if subject and subject.get("id") is not None else None
        overrides = self._apply_overrides(
            findings=findings,
            records=self.overrides(subject_type="fusion_run", subject_id=run_id),
            subject_content_hash=subject_content_hash,
            candidate_id=candidate_scope,
        )

        return {
            **self._protocol_envelope(),
            "subject": {
                "type": "fusion_run",
                "id": run_id,
                "path": detail.get("path"),
                "contentHash": subject_content_hash,
                "candidateId": candidate_scope,
                "candidateLabel": subject.get("label") if subject else None,
            },
            "roles": [dict(role) for role in COUNCIL_ROLES],
            "thresholds": self.thresholds.as_dict(),
            "findings": findings,
            "decision": _aggregate_decision(findings),
            "overrides": overrides,
        }

    def review_strategy_run(self, run_id: str, resolver, runs_service) -> dict[str, Any]:
        """Review one completed strategy-pipeline run from its own artifacts.

        The council previously only adjudicated fusion searches, leaving the
        main research loop — the run that actually produces a strategy — with no
        role-scoped review at all. Each agent reads the run's persisted evidence
        and returns a verdict in its own scope; missing evidence yields
        ``unknown``, which never counts as clearance.
        """
        record = runs_service.run(run_id)
        if record is None:
            raise KeyError(run_id)
        result = resolver.resolve(record["outputDir"])

        findings = [
            check(result, self.thresholds)
            for check in (
                _run_data_acquisition,
                _run_data_quality,
                _run_microstructure,
                _run_factor_integrity,
                _run_model_validation,
                _run_search_statistics,
                _run_portfolio_risk,
                _run_execution_realism,
                _run_challenger,
                _run_compliance,
            )
        ]
        findings.append(_run_governance(result, self.thresholds, findings))
        subject_content_hash = _canonical_hash({"record": record, "result": result})
        overrides = self._apply_overrides(
            findings=findings,
            records=self.overrides(subject_type="strategy_run", subject_id=run_id),
            subject_content_hash=subject_content_hash,
            candidate_id=None,
        )

        return {
            **self._protocol_envelope(),
            "subject": {
                "type": "strategy_run",
                "id": run_id,
                "path": record["outputDir"],
                "contentHash": subject_content_hash,
                "candidateId": None,
                "candidateLabel": None,
                "strategyId": record.get("strategyId"),
                "strategyName": record.get("strategyName"),
                "outcome": (result.get("conclusion") or {}).get("outcome"),
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
        with self._log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
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
        subject_content_hash: str,
        candidate_id: str | None,
        role_id: str,
        finding_hash: str,
        original_verdict: str,
        verdict: str,
        reason: str,
        author: str,
        protocol_version: int,
        policy_fingerprint: str,
    ) -> dict[str, Any]:
        """Append a human override. The log is append-only by construction."""
        if role_id not in ROLE_IDS:
            raise ValueError(f"unknown council role: {role_id}")
        if protocol_version != COUNCIL_PROTOCOL_VERSION:
            raise ValueError("override protocolVersion does not match Council v2")
        if policy_fingerprint != self.policy_fingerprint:
            raise ValueError("override policyFingerprint is stale")
        if len(subject_content_hash) != 64 or len(finding_hash) != 64:
            raise ValueError("override hashes must be complete SHA-256 values")
        if original_verdict not in {"pass", "warn", "blocked", "unknown"}:
            raise ValueError("originalVerdict is invalid")
        if verdict not in {"pass", "warn", "blocked"}:
            raise ValueError("override verdict must be pass, warn or blocked")
        if original_verdict == "unknown" and verdict == "pass":
            raise ValueError("missing evidence cannot be overridden into clearance")
        reason = reason.strip()
        if len(reason) < 8:
            raise ValueError("override reason must explain the decision (>= 8 characters)")
        author = author.strip()
        if not author:
            raise ValueError("override author is required")

        record = {
            "subjectType": subject_type,
            "subjectId": subject_id,
            "subjectContentHash": subject_content_hash,
            "candidateId": candidate_id,
            "roleId": role_id,
            "findingHash": finding_hash,
            "originalVerdict": original_verdict,
            "verdict": verdict,
            "reason": reason,
            "author": author,
            "protocolVersion": protocol_version,
            "policyFingerprint": policy_fingerprint,
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


def _promotion_gate(summary: dict[str, Any]) -> dict[str, Any] | None:
    value = summary.get("_promotionGate")
    return value if isinstance(value, dict) else None


def _review_data_acquisition(summary, subject, candidates, thresholds) -> dict[str, Any]:
    manifest = summary.get("_manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    generated = summary.get("generatedAt")
    factors = summary.get("factorNames")
    declared = summary.get("candidateCount")
    evaluated = summary.get("evaluatedCandidateCount")
    evidence = {
        "generatedAt": generated,
        "factorNames": factors,
        "declaredCandidateCount": declared,
        "persistedCandidateCount": len(candidates),
        "evaluatedCandidateCount": evaluated,
        "manifestContentHash": manifest.get("contentHash"),
    }
    if (
        not generated
        or not isinstance(factors, list)
        or not factors
        or not isinstance(manifest.get("contentHash"), str)
        or not manifest["contentHash"].strip()
    ):
        return _finding(
            "data_acquisition", "unknown", "采集批次证据不完整",
            "缺少生成时间或输入因子清单，无法把本次搜索绑定到一个可审计的数据批次。",
            evidence, "补齐 manifest 的时间戳与输入清单后重跑",
        )
    if not isinstance(declared, int) or not isinstance(evaluated, int):
        return _finding(
            "data_acquisition", "unknown", "候选落盘计数未声明",
            "缺少声明候选数或已评估候选数，无法证明搜索输出完整落盘。",
            evidence, "重新生成带计数的搜索产物",
        )
    if declared != len(candidates) or evaluated <= 0 or evaluated > declared:
        return _finding(
            "data_acquisition", "blocked", "搜索产物计数不一致",
            "声明数量、实际落盘数量与已评估数量不一致，产物可能被截断或混入其他批次。",
            evidence, "清理该运行目录并从同一数据快照完整重跑",
        )
    return _finding(
        "data_acquisition", "pass", "输入批次与产物计数可追溯",
        f"{len(factors)} 个输入因子，{evaluated}/{declared} 个候选完成评估并落盘。",
        evidence, "无",
    )


def _review_data_quality(summary, subject, candidates, thresholds) -> dict[str, Any]:
    benchmark = summary.get("benchmarkMode")
    observations = _metric(subject, "observations")
    promotion = _promotion_gate(summary)
    checks = {
        str(item.get("name")): item
        for item in ((promotion or {}).get("checks") or [])
        if isinstance(item, dict)
    }
    pit = checks.get("point_in_time")
    benchmark_check = checks.get("explicit_benchmark")
    evidence = {
        "benchmarkMode": benchmark,
        "observations": observations,
        "pointInTime": pit.get("observed") if pit else None,
        "explicitBenchmark": benchmark_check.get("observed") if benchmark_check else None,
    }
    if benchmark is None or pit is None or benchmark_check is None:
        return _finding(
            "data_quality", "unknown", "PIT 或基准证据未记录",
            "必须同时记录 benchmarkMode、point_in_time 与 explicit_benchmark 检查。",
            evidence, "用完整 research promotion gate 重新运行搜索",
        )
    if observations is None or pit.get("observed") is None or benchmark_check.get("observed") is None:
        return _finding(
            "data_quality", "unknown", "数据质量实测值缺失",
            "passed 标记不能替代 PIT、基准与样本外观测的实际证据。",
            evidence, "持久化每项 gate 的 observed 值后重跑",
        )
    if pit.get("passed") is not True or benchmark_check.get("passed") is not True:
        return _finding(
            "data_quality", "blocked", "PIT 或显式基准检查失败",
            "输入可得时点或超额收益基准不满足研究晋级契约。",
            evidence, "修复数据时序与基准输入后重跑",
        )
    if observations < thresholds.min_observations:
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


def _review_microstructure(summary, subject, candidates, thresholds) -> dict[str, Any]:
    horizon = summary.get("horizonDays")
    frequency = summary.get("barFrequency")
    intraday_claim = summary.get("intradayClaim")
    evidence = {
        "horizonDays": horizon,
        "barFrequency": frequency,
        "intradayClaim": intraday_claim,
    }
    if not isinstance(horizon, int) or not isinstance(frequency, str) or not isinstance(intraday_claim, bool):
        return _finding(
            "microstructure", "unknown", "研究时钟未声明",
            "没有持有期或频率字段，无法判断需要日频还是日内微观结构证据。",
            evidence, "写入 horizonDays 与 barFrequency",
        )
    if horizon < 1:
        return _finding(
            "microstructure", "blocked", "持有期不符合日频协议",
            "持有期小于一个交易日，却没有盘口、成交队列或事件时钟证据。",
            evidence, "使用日内专用数据和撮合协议重新研究",
        )
    if intraday_claim or frequency.lower() not in {"1d", "daily", "day"}:
        return _finding(
            "microstructure", "blocked", "日内能力没有对应证据",
            "当前 Council 只认证日频边界，不能凭日线产物提出日内成交或排队主张。",
            evidence, "提交盘口、事件时钟与延迟/冲击证据后走独立协议",
        )
    return _finding(
        "microstructure", "pass", "日频研究时钟已明确",
        "barFrequency 与 intradayClaim 明确限定为日频，不包含日内能力声明。",
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
            "factor_integrity", "blocked", "候选本身是单因子，不构成融合",
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
    required_fold_fields = ("trainStart", "trainEnd", "testStart", "testEnd")
    incomplete: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, dict):
            incomplete.append({"foldIndex": None, "missing": list(required_fold_fields)})
            continue
        missing = [key for key in required_fold_fields if not fold.get(key)]
        if missing:
            incomplete.append({"foldIndex": fold.get("foldIndex"), "missing": missing})
    evidence["incompleteFolds"] = incomplete
    if not isinstance(horizon, int) or incomplete:
        return _finding(
            "model_validation", "unknown", "折窗口字段不完整",
            "每折必须记录 train/test 起止日期，且研究必须声明 horizonDays。",
            evidence, "重新持久化完整折窗口与持有期",
        )
    if len(folds) < thresholds.min_folds:
        return _finding(
            "model_validation", "blocked", "折数不足",
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
    promotion = _promotion_gate(summary) or {}
    statistical = promotion.get("statisticalEvidence")
    statistical = statistical if isinstance(statistical, dict) else {}
    dsr = statistical.get("dsrProbability", breakdown.get("deflatedSharpeProbability"))
    spa = statistical.get("spaPValue")
    evidence = {
        "nTrials": trials,
        "pbo": pbo,
        "deflatedSharpeProbability": dsr,
        "spaPValue": spa,
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
            "fusion_search", "unknown", "PBO 无估计",
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
    if not isinstance(dsr, (int, float)) or not isinstance(spa, (int, float)):
        return _finding(
            "fusion_search", "unknown", "DSR / SPA 证据不完整",
            "PBO 不能替代多重试验收缩后的 DSR 与 SPA；缺失任一项都不能记为通过。",
            evidence, "使用 promotion_gate.json 持久化完整统计证据",
        )
    if float(dsr) < thresholds.min_deflated_sharpe:
        return _finding(
            "fusion_search", "blocked", "收缩后显著性未达标",
            f"按 {trials} 次试验收缩后的 Sharpe 显著性只有 {float(dsr):.3f}，"
            f"低于 {thresholds.min_deflated_sharpe:.2f}。",
            evidence, "减少试验或提高单候选质量",
        )
    if float(spa) > thresholds.max_spa_pvalue:
        return _finding(
            "fusion_search", "blocked", "SPA 未排除数据挖掘偏差",
            f"SPA p-value {float(spa):.3f} 高于 {thresholds.max_spa_pvalue:.2f}。",
            evidence, "缩小候选集合并在更长的样本外窗口复核",
        )
    return _finding(
        "fusion_search", "pass", "统计口径成立",
        f"{trials} 次试验，PBO {float(pbo):.3f}，DSR {float(dsr):.3f}，SPA {float(spa):.3f}。",
        evidence, "无",
    )


def _review_portfolio_risk(summary, subject, candidates, thresholds) -> dict[str, Any]:
    drawdown = _metric(subject, "maxDrawdown")
    turnover = _metric(subject, "averageTurnover")
    top_k = summary.get("topK")
    evidence = {"maxDrawdown": drawdown, "averageTurnover": turnover, "topK": top_k}
    if drawdown is None or turnover is None or not isinstance(top_k, int):
        return _finding(
            "portfolio_risk", "unknown", "风险证据不完整",
            "候选必须同时记录回撤、换手与 Top-K 集中度口径。", evidence, "检查候选是否产生了完整样本外指标",
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
    if cost is None or not isinstance(horizon, int) or cost_drag is None:
        return _finding(
            "execution_realism", "unknown", "执行成本证据不完整",
            "必须记录成本假设、持有周期与候选实际 cost drag。", evidence, "重新运行搜索",
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


def _review_challenger(summary, subject, candidates, thresholds) -> dict[str, Any]:
    controls = [item for item in candidates if item.get("isControl") is True]
    random_controls = [
        item for item in controls if str(item.get("scheme") or "").startswith("random")
    ]
    single_controls = [
        item for item in controls if item.get("scheme") == "single_factor"
    ]
    evidence = {
        "controlCount": len(controls),
        "randomControlCount": len(random_controls),
        "singleFactorControlCount": len(single_controls),
        "selectedIsControl": (subject or {}).get("isControl"),
    }
    if not candidates:
        return _finding(
            "challenger", "unknown", "没有可挑战的候选集合",
            "候选列表为空，挑战者无法比较冠军、基线与负面对照。",
            evidence, "恢复完整候选产物",
        )
    if (subject or {}).get("isControl") is True:
        return _finding(
            "challenger", "blocked", "对照组赢得本轮搜索",
            "这是有效的负面研究结论，但对照组本身不应被包装成可晋级策略。",
            evidence, "保留负面结论并停止本候选晋级",
        )
    if not controls:
        return _finding(
            "challenger", "blocked", "没有独立对照组",
            "搜索只比较了拟合候选，没有随机或单因子基线，无法排除复杂度幻觉。",
            evidence, "加入预注册随机对照与单因子基线后重跑",
        )
    if not random_controls or not single_controls:
        return _finding(
            "challenger", "blocked", "挑战集合不完整",
            "已有对照，但随机对照与单因子基线没有同时覆盖。",
            evidence, "补齐缺失的一类对照后复核",
        )
    controls_have_metrics = all(
        _metric(item, "observations") is not None and _metric(item, "excessReturn") is not None
        for item in controls
    )
    if not controls_have_metrics:
        return _finding(
            "challenger", "unknown", "对照组指标不完整",
            "对照方案存在，但缺少样本外观测或成本后超额收益，无法完成独立挑战。",
            evidence, "保留每个对照的完整样本外指标",
        )
    return _finding(
        "challenger", "pass", "候选通过双重对照挑战",
        f"已比较 {len(random_controls)} 个随机对照与 {len(single_controls)} 个单因子基线。",
        evidence, "无",
    )


def _review_compliance(summary, subject, candidates, thresholds) -> dict[str, Any]:
    promotion = _promotion_gate(summary)
    evidence = {
        "promotionGatePresent": promotion is not None,
        "researchOnly": promotion.get("researchOnly") if promotion else None,
        "productionEligible": promotion.get("productionEligible") if promotion else None,
        "stage4Governed": promotion.get("stage4Governed") if promotion else None,
        "productionBlockers": promotion.get("productionBlockers") if promotion else None,
    }
    if promotion is None:
        return _finding(
            "compliance", "unknown", "研究/生产边界证据缺失",
            "没有 promotion_gate.json，无法确认该产物是否明确禁止 production/live 使用。",
            evidence, "生成研究晋级门并保留 production blockers",
        )
    if (
        not isinstance(promotion.get("researchOnly"), bool)
        or not isinstance(promotion.get("productionEligible"), bool)
        or not isinstance(promotion.get("stage4Governed"), bool)
        or not isinstance(promotion.get("productionBlockers"), list)
        or not promotion.get("productionBlockers")
    ):
        return _finding(
            "compliance", "unknown", "权限边界字段不完整",
            "researchOnly、productionEligible、stage4Governed 与 productionBlockers 必须同时落盘。",
            evidence, "重新生成完整 promotion_gate.json",
        )
    if promotion.get("researchOnly") is not True or promotion.get("productionEligible") is not False:
        return _finding(
            "compliance", "blocked", "研究产物出现越权声明",
            "本层产物必须显式 researchOnly=true 且 productionEligible=false。",
            evidence, "撤销越权状态并走完整 Stage-4 独立认证",
        )
    return _finding(
        "compliance", "pass", "研究与生产权限已隔离",
        "该运行明确是 research-only，并保留了进入生产前仍需解决的阻塞项。",
        evidence, "不得把本裁决解释为 live 授权",
    )


def _review_governance(
    summary,
    subject,
    candidates,
    thresholds,
    prior_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    generated = summary.get("generatedAt")
    promotion = _promotion_gate(summary)
    evidence = {
        "generatedAt": generated,
        "mode": "RESEARCH",
        "liveIntent": False,
        "candidateIsControl": bool((subject or {}).get("isControl")),
        "researchPromotionEligible": (
            promotion.get("researchPromotionEligible") if promotion else None
        ),
        "upstreamVerdicts": {
            item["roleId"]: item["verdict"] for item in prior_findings
        },
    }
    if (subject or {}).get("isControl"):
        return _finding(
            "governance", "blocked", "首选候选是对照组",
            "当前首选是一个不读取训练段的对照方案。这是一个有效的研究结论"
            "（拟合方案没有赢过基线），但它不构成可晋级的策略。",
            evidence, "接受该负面结论，或更换因子集合重跑",
        )
    if not generated or promotion is None:
        return _finding(
            "governance", "unknown", "主席缺少完整审计证据",
            "生成时间或研究晋级门缺失，无法形成公司级决议。", evidence, "重新运行搜索",
        )
    if promotion.get("researchPromotionEligible") is not True:
        return _finding(
            "governance", "blocked", "研究晋级门未通过",
            "统计或 PIT/基准/holdout 门未全部通过，CIO 不得将候选提交到人工晋级 Gate。",
            evidence, "按 promotion_gate.json 的 blockers 修复后重跑",
        )
    blocked = [item["roleId"] for item in prior_findings if item["verdict"] == "blocked"]
    unknown = [item["roleId"] for item in prior_findings if item["verdict"] == "unknown"]
    warned = [item["roleId"] for item in prior_findings if item["verdict"] == "warn"]
    if blocked:
        return _finding(
            "governance", "blocked", "前置部门存在否决",
            f"CIO 不得绕过前置部门否决：{', '.join(blocked)}。",
            evidence, "先解决对应部门 blocker 后重新提交",
        )
    if unknown:
        return _finding(
            "governance", "unknown", "前置部门证据不足",
            f"以下部门尚不能形成裁决：{', '.join(unknown)}。",
            evidence, "补齐对应结构化证据后重新提交",
        )
    if warned:
        return _finding(
            "governance", "warn", "主席附保留意见提交人工 Gate",
            f"前置部门保留意见：{', '.join(warned)}；人工 Gate 必须逐项复核。",
            evidence, "人工 Gate 逐项确认保留意见",
        )
    return _finding(
        "governance", "pass", "主席同意提交人工 Gate",
        "各角色的结构化裁决可供人工复核；该决议仍处于 RESEARCH，未产生任何订单意图。",
        evidence, "人工复核后决定是否晋级",
    )


def _aggregate_decision(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Council-level outcome. An override replaces the verdict it names."""
    effective = []
    for item in findings:
        override = item.get("override")
        # A reason can change a judgment, but it cannot manufacture evidence.
        # Keep this defense even though v2 write validation also rejects it.
        if item["verdict"] == "unknown" and override and override["verdict"] == "pass":
            effective.append("unknown")
        else:
            effective.append(override["verdict"] if override else item["verdict"])
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
        "researchMayContinue": True,
        "eligibleForHumanGate": state in {"PROMOTABLE", "PROMOTABLE_WITH_WARNINGS"},
        "liveEligible": False,
        "blockedRoles": blocked,
        "unknownRoles": unknown,
        "warnedRoles": warned,
        "overriddenRoles": [
            item["roleId"] for item in findings if item.get("override")
        ],
    }


__all__ = [
    "COUNCIL_POLICY_FINGERPRINT",
    "COUNCIL_PROTOCOL_EFFECTIVE_AT",
    "COUNCIL_PROTOCOL_VERSION",
    "COUNCIL_ROLES",
    "ROLE_IDS",
    "CouncilService",
    "CouncilThresholds",
]


# ---------------------------------------------------------------------------
# Strategy-run role checks.
#
# Each reads the resolved run result and reports inside its own scope. They
# share one discipline: a check whose input artifact is absent returns
# ``unknown`` with the field it looked for, never ``pass``.
# ---------------------------------------------------------------------------


def _gate(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for gate in ((result.get("acceptance") or {}).get("gates") or []):
        if gate.get("name") == name:
            return gate
    return None


def _run_data_acquisition(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    stages = result.get("stages") or []
    dataset = next(
        (stage for stage in stages if isinstance(stage, dict) and stage.get("id") == "dataset"),
        None,
    )
    evidence = {
        "datasetPresent": dataset.get("present") if dataset else None,
        "datasetPath": dataset.get("path") if dataset else None,
        "datasetSizeBytes": dataset.get("sizeBytes") if dataset else None,
        "artifactCount": len(result.get("artifacts") or []),
    }
    if dataset is None or dataset.get("present") is not True:
        return _finding(
            "data_acquisition", "unknown", "训练数据集产物缺失",
            "运行目录没有 dataset/training_dataset.parquet，无法把后续结果绑定到输入快照。",
            evidence, "恢复数据集产物与其 manifest 后重新提交评审",
        )
    if not dataset.get("path"):
        return _finding(
            "data_acquisition", "unknown", "训练数据集路径缺失",
            "存在标记不能替代可追溯的 dataset path。",
            evidence, "重新生成包含 sourcePath 的阶段清单",
        )
    if not isinstance(dataset.get("sizeBytes"), int) or dataset["sizeBytes"] <= 0:
        return _finding(
            "data_acquisition", "blocked", "训练数据集为空",
            "文件存在但没有有效字节，不能作为采集完成的证据。",
            evidence, "检查采集任务与落盘权限后重跑",
        )
    return _finding(
        "data_acquisition", "pass", "训练数据快照已落盘",
        "下游运行可追溯到非空的训练数据集产物。",
        evidence, "无",
    )


def _run_data_quality(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    pit = _gate(result, "no_pit_violations")
    mock = _gate(result, "no_mock_or_synthetic")
    symbols = _gate(result, "training_symbols")
    evidence = {
        "noPitViolations": pit.get("actual") if pit else None,
        "noMockOrSynthetic": mock.get("actual") if mock else None,
        "trainingSymbols": symbols.get("actual") if symbols else None,
    }
    if pit is None or mock is None or symbols is None:
        return _finding(
            "data_quality", "unknown", "缺少 PIT / 合成数据判定",
            "验收报告没有写入 no_pit_violations 或 no_mock_or_synthetic，无法确认输入可信。",
            evidence, "确认运行是否走完验收阶段",
        )
    if pit.get("actual") is None or mock.get("actual") is None or symbols.get("actual") is None:
        return _finding(
            "data_quality", "unknown", "PIT / 数据来源实测值缺失",
            "passed=true 不能替代 no_pit_violations、no_mock_or_synthetic 与 training_symbols 的 actual。",
            evidence, "重新生成包含 actual 的验收报告",
        )
    if not pit.get("passed") or not mock.get("passed") or not symbols.get("passed"):
        return _finding(
            "data_quality", "blocked", "输入数据不可信",
            "存在 PIT 违规或使用了 mock/synthetic 数据，整条链的结论都不成立。",
            evidence, "修复数据来源后重跑，不得带着该状态晋级",
        )
    return _finding(
        "data_quality", "pass", "PIT 与真实数据校验通过",
        f"零 PIT 违规，非合成数据，训练覆盖 {evidence['trainingSymbols']} 个标的。",
        evidence, "无",
    )


def _run_microstructure(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    backtest = result.get("backtest") or {}
    orders = backtest.get("orderCount")
    skipped = backtest.get("skippedOrderCount")
    evidence = {
        "orderCount": orders,
        "skippedOrderCount": skipped,
        "microstructureArtifact": backtest.get("microstructureSourcePath"),
    }
    if orders is None or skipped is None or not backtest.get("sourcePath"):
        return _finding(
            "microstructure", "unknown", "缺少市场时钟与撮合证据",
            "没有回测委托记录，无法确认研究是否至少遵循日频交易时钟。",
            evidence, "完成严格 A 股回测；日内策略另需盘口证据",
        )
    return _finding(
        "microstructure", "warn", "日频约束已覆盖，日内能力未认证",
        "当前证据只支持日频订单约束；盘口队列、延迟与冲击曲线仍不在本次认证范围。",
        evidence, "保持日频声明；提出日内能力前补做微观结构验证",
    )


def _run_factor_integrity(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    dominance = _gate(result, "single_factor_dominance")
    features = (result.get("training") or {}).get("featureCount")
    evidence = {
        "singleFactorDominance": dominance.get("actual") if dominance else None,
        "threshold": dominance.get("threshold") if dominance else None,
        "featureCount": features,
    }
    if dominance is None or dominance.get("actual") is None or dominance.get("threshold") is None:
        return _finding(
            "factor_integrity", "unknown", "未记录单因子支配度",
            "验收报告缺少 single_factor_dominance，无法判断结论是否由单一因子驱动。",
            evidence, "确认验收阶段是否完整执行",
        )
    if not dominance.get("passed"):
        return _finding(
            "factor_integrity", "blocked", "单因子支配度过高",
            f"实测 {dominance.get('actual')}，超过 {dominance.get('threshold')}；"
            "组合表现主要来自单一因子，稳健性不足。",
            evidence, "扩大因子集合或降低该因子权重后重跑",
        )
    return _finding(
        "factor_integrity", "pass", "无单因子支配",
        f"支配度 {dominance.get('actual')} 在阈值内，特征数 {features}。",
        evidence, "无",
    )


def _run_model_validation(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    training = result.get("training") or {}
    folds = training.get("foldCount")
    days = training.get("evaluatedDays")
    adverse = training.get("adverseRegimePassed")
    evidence = {"foldCount": folds, "evaluatedDays": days, "adverseRegimePassed": adverse}
    if folds is None or days is None or adverse is None or not training.get("sourcePath"):
        return _finding(
            "model_validation", "unknown", "缺少训练评估证据",
            "training/metrics.json 缺失或不完整，无法核对折数与评估窗口。",
            evidence, "确认训练阶段产物是否写入",
        )
    if folds < thresholds.min_folds:
        return _finding(
            "model_validation", "blocked", "折数不足",
            f"仅 {folds} 折，低于 {thresholds.min_folds} 折的最低要求，样本外结论不稳定。",
            evidence, "提高 nSplits 后重跑",
        )
    if adverse is not True:
        return _finding(
            "model_validation", "blocked", "逆境区间未通过",
            "模型在不利行情下没有通过预注册检验。",
            evidence, "调整研究设计并重跑 adverse regime 检验",
        )
    return _finding(
        "model_validation", "pass", "滚动验证与逆境检验通过",
        f"{folds} 折、{days} 个评估交易日，逆境区间通过。",
        evidence, "无",
    )


def _run_search_statistics(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    governance = result.get("governance")
    if governance is None:
        return _finding(
            "fusion_search", "unknown", "缺少过拟合治理记录",
            "selection_governance.json 缺失，无法核对 PBO / DSR / SPA 与试验次数。",
            {"selectionGovernance": None}, "确认组合选择阶段是否执行",
        )
    evidence = {
        "pbo": governance.get("pbo"),
        "dsrProbability": governance.get("dsrProbability"),
        "spaPValue": governance.get("spaPValue"),
        "cumulativeTrials": governance.get("cumulativeTrials"),
        "observedDays": governance.get("observedDays"),
    }
    if not isinstance(governance.get("accepted"), bool):
        return _finding(
            "fusion_search", "unknown", "统计闸门终态缺失",
            "selection_governance.json 必须显式记录 accepted=true/false。",
            evidence, "重新生成完整 selection governance",
        )
    if not governance.get("accepted"):
        return _finding(
            "fusion_search", "blocked", "过拟合治理否决",
            "; ".join(governance.get("rejectionReasons") or ["候选未通过统计闸门"]),
            evidence, "减少候选数量或延长观测窗口后重新预注册",
        )
    pbo = governance.get("pbo")
    dsr = governance.get("dsrProbability")
    spa = governance.get("spaPValue")
    if (
        not all(isinstance(value, (int, float)) for value in (pbo, dsr, spa))
        or not isinstance(governance.get("cumulativeTrials"), int)
        or governance.get("cumulativeTrials", 0) <= 0
        or not governance.get("sourcePath")
    ):
        return _finding(
            "fusion_search", "unknown", "统计闸门数值不完整",
            "accepted 标记不能替代 PBO、DSR 与 SPA 三项实测值。",
            evidence, "重新生成 selection_governance.json",
        )
    if pbo > thresholds.max_pbo:
        return _finding(
            "fusion_search", "blocked", "PBO 高于公司阈值",
            f"PBO {pbo:.4f} 高于 {thresholds.max_pbo}。",
            evidence, "减少试验次数或延长样本外窗口",
        )
    if dsr < thresholds.min_deflated_sharpe:
        return _finding(
            "fusion_search", "blocked", "DSR 低于公司阈值",
            f"DSR {dsr:.4f} 低于 {thresholds.min_deflated_sharpe}。",
            evidence, "提高样本外稳定性后重新预注册",
        )
    if spa > thresholds.max_spa_pvalue:
        return _finding(
            "fusion_search", "blocked", "SPA 未通过公司阈值",
            f"SPA p-value {spa:.4f} 高于 {thresholds.max_spa_pvalue}。",
            evidence, "降低数据挖掘自由度后重跑",
        )
    return _finding(
        "fusion_search", "pass", "统计闸门通过",
        f"PBO {pbo}，DSR {dsr}，SPA {spa}，"
        f"计入 {governance.get('cumulativeTrials')} 次试验。",
        evidence, "无",
    )


def _run_portfolio_risk(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    drawdown_gate = _gate(result, "max_drawdown")
    backtest = result.get("backtest") or {}
    measured = backtest.get("maxDrawdown")
    evidence = {
        "gateDrawdown": drawdown_gate.get("actual") if drawdown_gate else None,
        "backtestMaxDrawdown": measured,
        "councilLimit": thresholds.max_drawdown,
    }
    gate_actual = drawdown_gate.get("actual") if drawdown_gate else None
    if gate_actual is None and measured is None:
        return _finding(
            "portfolio_risk", "unknown", "缺少回撤证据",
            "既没有验收闸门中的 max_drawdown，也没有回测净值序列。",
            evidence, "确认组合与回测阶段是否产出",
        )
    if drawdown_gate is not None and gate_actual is not None and not drawdown_gate.get("passed"):
        return _finding(
            "portfolio_risk", "blocked", "回撤超过声明上限",
            f"实测 {drawdown_gate.get('actual')}，阈值 {drawdown_gate.get('threshold')}。",
            evidence, "收紧集中度或换手预算后重跑",
        )
    if isinstance(measured, (int, float)) and abs(measured) > thresholds.max_drawdown:
        return _finding(
            "portfolio_risk", "warn", "回撤高于议事会内部上限",
            f"回测最大回撤 {measured:.4f} 超过议事会 {thresholds.max_drawdown} 的内部上限。",
            evidence, "在结论中说明风险预算",
        )
    return _finding(
        "portfolio_risk", "pass", "回撤在声明范围内",
        f"回测最大回撤 {measured if measured is not None else drawdown_gate.get('actual')}。",
        evidence, "无",
    )


def _run_execution_realism(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    backtest = result.get("backtest") or {}
    orders = backtest.get("orderCount")
    skipped = backtest.get("skippedOrderCount")
    failed = backtest.get("failedOrderCount")
    evidence = {
        "orderCount": orders,
        "skippedOrderCount": skipped,
        "failedOrderCount": failed,
        "sourcePath": backtest.get("sourcePath"),
    }
    if orders is None or skipped is None or failed is None or not backtest.get("sourcePath"):
        return _finding(
            "execution_realism", "unknown", "缺少撮合记录",
            "回测报告没有委托计数，无法判断 A 股约束下的可实现性。",
            evidence, "确认回测阶段是否产出",
        )
    total = (orders or 0) + (skipped or 0)
    if total and (skipped or 0) / total > 0.5:
        return _finding(
            "execution_realism", "warn", "过半委托被交易约束拒绝",
            f"{skipped} / {total} 笔委托因 T+1、涨跌停、停牌、整手或成交量上限被跳过；"
            "纸面权重与可执行组合差距很大。",
            evidence, "降低换手或调整标的池，使目标权重可实现",
        )
    return _finding(
        "execution_realism", "pass", "撮合约束下可执行",
        f"{orders} 笔成交，{skipped} 笔被约束跳过。",
        evidence, "无",
    )


def _run_challenger(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    candidates = [item for item in (result.get("candidates") or []) if isinstance(item, dict)]
    selected = [item for item in candidates if item.get("selected") is True]
    trials = (result.get("governance") or {}).get("cumulativeTrials")
    evidence = {
        "candidateCount": len(candidates),
        "selectedCount": len(selected),
        "cumulativeTrials": trials,
    }
    if not candidates:
        return _finding(
            "challenger", "unknown", "候选与反事实未落盘",
            "没有逐候选 paper report，无法独立复核为什么选择冠军。",
            evidence, "保留每个候选的成本后报告",
        )
    if trials is None:
        return _finding(
            "challenger", "unknown", "挑战试验计数缺失",
            "候选集合存在，但 cumulativeTrials 未落盘，无法证明所有替代方案均被计数。",
            evidence, "保留候选计数与 selection governance",
        )
    candidate_evidence_complete = all(
        item.get("sourcePath")
        and isinstance(item.get("netReturnAfterCosts"), (int, float))
        for item in candidates
    )
    if not candidate_evidence_complete:
        return _finding(
            "challenger", "unknown", "候选成本后证据不完整",
            "每个候选都必须带 sourcePath 与 netReturnAfterCosts。",
            evidence, "重新生成完整逐候选 paper report",
        )
    if len(candidates) < 2 or len(selected) != 1:
        return _finding(
            "challenger", "blocked", "候选挑战协议不成立",
            "至少需要一个替代候选，且必须恰好有一个被选候选。",
            evidence, "补齐对照候选并固定唯一选择结果",
        )
    return _finding(
        "challenger", "pass", "冠军可与替代候选逐项复核",
        f"保留 {len(candidates)} 个候选，唯一冠军及其成本后结果均可追溯。",
        evidence, "无",
    )


def _run_compliance(result: dict[str, Any], thresholds: CouncilThresholds) -> dict[str, Any]:
    conclusion = result.get("conclusion") or {}
    pipeline = result.get("pipelineReport") or {}
    live_claims = {
        key: value
        for key, value in pipeline.items()
        if any(token in str(key).upper() for token in ("LIVE", "PRODUCTION"))
        and value not in (None, False, "", "disabled", "research_only")
    }
    evidence = {
        "outcome": conclusion.get("outcome"),
        "promotable": conclusion.get("promotable"),
        "liveOrProductionClaims": live_claims,
        "acceptanceStatus": pipeline.get("QUANT_ACCEPTANCE_STATUS"),
    }
    if live_claims:
        return _finding(
            "compliance", "blocked", "研究报告包含 production/live 声明",
            "策略流水线只能产出 research/paper 证据，不能在本层授予生产权限。",
            evidence, "删除越权声明并提交独立生产认证",
        )
    if (
        conclusion.get("outcome") is None
        or not isinstance(conclusion.get("promotable"), bool)
        or pipeline.get("QUANT_ACCEPTANCE_STATUS") not in {"passed", "failed"}
    ):
        return _finding(
            "compliance", "unknown", "合规边界证据不完整",
            "运行尚未形成完整研究结论，无法完成模型风险披露。",
            evidence, "补齐完整产物后复核",
        )
    return _finding(
        "compliance", "warn", "仅允许 research/paper 使用",
        "未发现 production/live 授权，但流水线也不具备授予该权限的能力；本裁决只确认研究边界。",
        evidence, "保持禁用 live；需要生产能力时走独立审批",
    )


def _run_governance(
    result: dict[str, Any],
    thresholds: CouncilThresholds,
    prior_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    conclusion = result.get("conclusion") or {}
    outcome = conclusion.get("outcome")
    promotable = conclusion.get("promotable")
    evidence = {
        "outcome": outcome,
        "promotable": promotable,
        "upstreamVerdicts": {item["roleId"]: item["verdict"] for item in prior_findings},
    }
    if outcome is None or not isinstance(promotable, bool):
        return _finding(
            "governance", "unknown", "结论证据不完整",
            "运行没有产出足以判定的完整证据链。",
            evidence, "先补齐运行产物再提交评审",
        )
    if outcome in {"no_evidence", "incomplete"}:
        return _finding(
            "governance", "unknown", "结论证据不完整",
            "运行没有产出足以判定的完整证据链。",
            evidence, "先补齐运行产物再提交评审",
        )
    if outcome in {"rejected", "not_accepted"} or promotable is not True:
        return _finding(
            "governance", "blocked", "未达到晋级条件",
            "运行已完成但闸门未通过；这是有效的研究结论，不能作为晋级依据。",
            evidence, "记录该结论并调整研究设计",
        )
    blocked = [item["roleId"] for item in prior_findings if item["verdict"] == "blocked"]
    unknown = [item["roleId"] for item in prior_findings if item["verdict"] == "unknown"]
    warned = [item["roleId"] for item in prior_findings if item["verdict"] == "warn"]
    if blocked:
        return _finding(
            "governance", "blocked", "前置部门存在否决",
            f"CIO 不得绕过前置部门否决：{', '.join(blocked)}。",
            evidence, "先解决对应部门 blocker 后重新提交",
        )
    if unknown:
        return _finding(
            "governance", "unknown", "前置部门证据不足",
            f"以下部门尚不能形成裁决：{', '.join(unknown)}。",
            evidence, "补齐对应证据后重新提交",
        )
    if warned:
        return _finding(
            "governance", "warn", "主席附保留意见提交人工 Gate",
            f"前置部门保留意见：{', '.join(warned)}。",
            evidence, "人工 Gate 逐项确认保留意见",
        )
    return _finding(
        "governance", "pass", "满足 research/paper 记录要求",
        "闸门通过且证据齐全；晋级仍需人工复核与独立验证，本裁决不构成 live 授权。",
        evidence, "人工复核后决定是否进入 paper 跟踪",
    )
