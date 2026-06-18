from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


REPORT_FAILURE_MARKERS: tuple[str, ...] = (
    "无法获取",
    "工具调用失败",
    "I cannot retrieve",
    "I don't have access",
    "unable to fetch",
    "network_error",
)

GRADE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "F")
GRADE_TO_CONFIDENCE: dict[str, float] = {"A": 0.95, "B": 0.78, "C": 0.55, "D": 0.25, "F": 0.0}


@dataclass(frozen=True)
class ReportRequirement:
    """A report requirement is satisfied when any marker appears in text."""

    label: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class ReportQualityResult:
    agent_name: str
    grade: str
    confidence: float
    length: int
    has_table: bool
    missing_requirements: tuple[str, ...]
    failure_markers: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class BundleQualityResult:
    overall_grade: str
    data_confidence: float
    results: tuple[ReportQualityResult, ...]
    blocking_agents: tuple[str, ...]


ASHARE_ANALYST_REQUIREMENTS: dict[str, tuple[ReportRequirement, ...]] = {
    "market": (
        ReportRequirement("price_volume", ("成交量", "volume", "量价", "turnover")),
        ReportRequirement("trend", ("趋势", "trend", "均线", "moving average")),
        ReportRequirement("liquidity_or_limit", ("涨跌停", "limit", "liquidity", "流动性")),
    ),
    "social": (
        ReportRequirement("retail_sentiment", ("散户", "retail", "sentiment", "情绪")),
        ReportRequirement("attention", ("热度", "attention", "volume z", "讨论")),
        ReportRequirement("pump_risk", ("pumping", "炒作", "coordinated", "风险")),
    ),
    "news": (
        ReportRequirement("source", ("source", "来源", "媒体")),
        ReportRequirement("published_at", ("published", "发布时间", "发布日期")),
        ReportRequirement("cross_validation", ("交叉验证", "cross", "validation", "独立")),
    ),
    "fundamentals": (
        ReportRequirement("valuation", ("PE", "PB", "估值", "市值", "market cap")),
        ReportRequirement("growth", ("营收", "利润", "growth", "同比")),
        ReportRequirement("cashflow_quality", ("现金流", "cash flow", "ROE", "quality")),
    ),
    "policy": (
        ReportRequirement("policy_event", ("政策", "policy", "国务院", "部委")),
        ReportRequirement("impact_direction", ("利好", "利空", "direction", "影响方向")),
        ReportRequirement("time_window", ("时间窗口", "horizon", "window", "持续")),
    ),
    "hot_money": (
        ReportRequirement("northbound", ("北向", "northbound", "沪股通", "深股通")),
        ReportRequirement("fund_flow", ("主力", "资金流", "fund flow", "大单")),
        ReportRequirement("theme_reason", ("题材", "reason", "concept", "板块")),
    ),
    "lockup": (
        ReportRequirement("lockup_schedule", ("解禁", "lockup", "限售")),
        ReportRequirement("reduction_pressure", ("减持", "pressure", "供给", "sell-down")),
        ReportRequirement("three_month_risk", ("90", "三个月", "3 months", "未来")),
    ),
}


class AgentReportQualityGate:
    """Deterministic quality gate for analyst reports and agent summaries."""

    def __init__(
        self,
        requirements: Mapping[str, Sequence[ReportRequirement]] | None = None,
        *,
        min_report_length: int = 200,
    ) -> None:
        self.requirements = {
            str(agent): tuple(reqs)
            for agent, reqs in (requirements or ASHARE_ANALYST_REQUIREMENTS).items()
        }
        self.min_report_length = int(min_report_length)

    def evaluate_report(self, agent_name: str, report: str | None) -> ReportQualityResult:
        text = (report or "").strip()
        length = len(text)
        has_table = "|" in text and "---" in text
        failures = tuple(marker for marker in REPORT_FAILURE_MARKERS if marker.lower() in text.lower())
        missing = self._missing_requirements(agent_name, text)

        if not text:
            grade, detail = "F", "report_empty"
        elif length < self.min_report_length:
            grade, detail = "D", f"report_too_short:{length}"
        elif failures and _remaining_text_length(text, failures) < self.min_report_length:
            grade, detail = "D", f"failure_dominated:{len(failures)}"
        elif len(missing) >= 3:
            grade, detail = "C", f"missing_requirements:{len(missing)}"
        elif missing or not has_table:
            grade, detail = "B", "minor_quality_gaps"
        else:
            grade, detail = "A", "complete"

        return ReportQualityResult(
            agent_name=agent_name,
            grade=grade,
            confidence=GRADE_TO_CONFIDENCE[grade],
            length=length,
            has_table=has_table,
            missing_requirements=missing,
            failure_markers=failures,
            detail=detail,
        )

    def evaluate_bundle(self, reports: Mapping[str, str | None]) -> BundleQualityResult:
        results = tuple(self.evaluate_report(agent_name, report) for agent_name, report in reports.items())
        if not results:
            return BundleQualityResult("F", 0.0, (), ())
        worst_grade = _worst_grade(result.grade for result in results)
        confidence = sum(result.confidence for result in results) / len(results)
        blocking = tuple(result.agent_name for result in results if result.grade in {"D", "F"})
        return BundleQualityResult(
            overall_grade=worst_grade,
            data_confidence=float(confidence),
            results=results,
            blocking_agents=blocking,
        )

    def _missing_requirements(self, agent_name: str, text: str) -> tuple[str, ...]:
        reqs = self.requirements.get(agent_name, ())
        lowered = text.lower()
        missing: list[str] = []
        for req in reqs:
            if not any(marker.lower() in lowered for marker in req.markers):
                missing.append(req.label)
        return tuple(missing)


def _worst_grade(grades: Sequence[str] | object) -> str:
    rank = {grade: idx for idx, grade in enumerate(GRADE_ORDER)}
    worst = "A"
    for grade in grades:
        grade_text = str(grade).upper()
        if rank.get(grade_text, len(GRADE_ORDER)) > rank[worst]:
            worst = grade_text
    return worst


def _remaining_text_length(text: str, markers: Sequence[str]) -> int:
    stripped = text
    for marker in markers:
        stripped = stripped.replace(marker, "")
    return len(stripped.strip())
