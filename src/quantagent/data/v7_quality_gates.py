"""Hard quality and acceptance gates for V7 real-data training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.ashare.gold_bridge import MASK_FALSE, MASK_TRUE, MASK_UNKNOWN


@dataclass(frozen=True)
class V7DataQualityGateConfig:
    min_rows: int = 1_000
    min_symbols: int = 50
    min_dates: int = 120
    require_real_data: bool = True
    max_single_factor_dominance: float = 0.60


@dataclass(frozen=True)
class V7ModelAcceptanceGateConfig:
    min_rank_ic_mean: float = 0.0
    min_rank_ic_stability: float = 0.0
    min_turnover_adjusted_return: float = 0.0
    max_drawdown: float = 0.10
    min_sharpe: float | None = None
    max_single_factor_dominance: float = 0.60
    require_adverse_regime: bool = True
    require_paper_report: bool = True
    require_benchmark: bool = True
    min_excess_return_after_costs: float = 0.0
    min_selection_pressure: float = 3.0
    min_training_symbols: int = 50
    min_prediction_symbols: int = 50
    min_effective_universe_by_date: int = 50
    no_mock_or_synthetic: bool = True
    no_pit_violations: bool = True
    #: Survivorship is checkable only where the master dates its delistings. Left
    #: on by default: a research pipeline that does not check is not thereby free
    #: of the bias, and this repository has measured pre-2020 survivorship bias.
    require_survivorship_check: bool = True
    #: Whether every feature row was knowable before the return it is scored on
    #: began accruing. Distinct from `no_pit_violations`, which compares a row
    #: against an as-of date: this compares it against *its own label window*, and
    #: that is the comparison nothing in the pipeline had ever made (DEF-026).
    require_label_alignment_check: bool = True
    adverse_regime_min_rank_ic: float = -0.02
    adverse_regime_max_drawdown: float = 0.40


#: A gate that was evaluated against a real measurement and cleared it.
GATE_PASS = "pass"
#: A gate that was evaluated against a real measurement and did not clear it.
GATE_FAIL = "fail"
#: The measurement a gate needs does not exist. This is *not* a failure — there
#: is nothing to fail — and it is emphatically not a pass. Reporting it as a
#: number (historically ``0.0``) invented evidence and then judged the run on it:
#: a run with no benchmark reported ``excess_return_after_costs = 0.0`` and was
#: recorded as having measured zero alpha, when in truth nothing was measured.
GATE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class V7GateReport:
    passed: bool
    failures: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: tuple[dict[str, Any], ...] = ()

    @property
    def unknowns(self) -> tuple[dict[str, Any], ...]:
        """Gates that could not be evaluated for want of a measurement."""
        return tuple(gate for gate in self.gates if gate.get("status") == GATE_UNKNOWN)

    @property
    def has_unknowns(self) -> bool:
        return bool(self.unknowns)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failures"] = list(self.failures)
        data["gates"] = list(self.gates)
        data["unknowns"] = [gate["name"] for gate in self.unknowns]
        return data


def evaluate_data_quality_gates(frame: pd.DataFrame, config: V7DataQualityGateConfig | None = None) -> V7GateReport:
    config = config or V7DataQualityGateConfig()
    failures: list[str] = []
    pit = audit_pit(frame)
    provenance = audit_provenance(frame)
    metrics = {
        "row_count": int(0 if frame is None else len(frame)),
        "symbol_count": int(frame["symbol"].nunique()) if frame is not None and "symbol" in frame.columns else 0,
        "date_count": int(pd.to_datetime(frame["trade_date"], errors="coerce").nunique()) if frame is not None and "trade_date" in frame.columns else 0,
        # `None` when nothing could be audited. Never 0 — see `audit_pit`.
        "pit_violation_count": pit.violations,
        "uses_mock_or_synthetic": provenance.uses_mock_or_synthetic,
        PIT_AUDIT_STATUS_KEY: pit.status,
        PIT_AUDIT_DETAIL_KEY: pit.detail,
        PIT_AUDIT_REFERENCE_KEY: pit.reference_column,
        PROVENANCE_AUDIT_STATUS_KEY: provenance.status,
        PROVENANCE_AUDIT_DETAIL_KEY: provenance.detail,
    }
    if (metrics["pit_violation_count"] or 0) > 0:
        failures.append("pit_violations_present")
    if metrics["row_count"] < config.min_rows:
        failures.append("insufficient_training_rows")
    if metrics["symbol_count"] < config.min_symbols:
        failures.append("insufficient_symbol_coverage")
    if metrics["date_count"] < config.min_dates:
        failures.append("insufficient_date_coverage")
    if config.require_real_data and metrics["uses_mock_or_synthetic"] is True:
        failures.append("mock_or_synthetic_data_not_production_ready")
    # `passed` keeps its old meaning — this report gates a *dataset build*, and a
    # panel with no provenance spine is not thereby a bad panel. What changed is
    # that the report no longer claims those audits came back clean: the tri-state
    # gates below travel with it, and the acceptance gate (which decides whether a
    # model may be called production-ready) is where an unaudited claim blocks.
    gates: list[dict[str, Any]] = []
    if pit.status == AUDIT_NOT_AUDITABLE:
        gates.append(
            {
                "name": "pit_audit", "passed": False, "status": GATE_UNKNOWN,
                "actual": None, "threshold": "0", "reason": "pit_audit_not_auditable",
                "detail": pit.detail,
            }
        )
    else:
        gates.append(
            {
                "name": "pit_audit", "passed": pit.violations == 0,
                "status": GATE_PASS if pit.violations == 0 else GATE_FAIL,
                "actual": pit.violations, "threshold": "0",
                "reason": "passed" if pit.violations == 0 else "pit_violations_present",
                "detail": pit.detail,
            }
        )
    if provenance.status == AUDIT_NOT_AUDITABLE:
        gates.append(
            {
                "name": "provenance_audit", "passed": False, "status": GATE_UNKNOWN,
                "actual": None, "threshold": "False",
                "reason": "provenance_audit_not_auditable", "detail": provenance.detail,
            }
        )
    else:
        clean = provenance.uses_mock_or_synthetic is False
        gates.append(
            {
                "name": "provenance_audit", "passed": clean,
                "status": GATE_PASS if clean else GATE_FAIL,
                "actual": provenance.uses_mock_or_synthetic, "threshold": "False",
                "reason": "passed" if clean else "mock_or_synthetic_data_not_production_ready",
                "detail": provenance.detail,
            }
        )
    return V7GateReport(not failures, tuple(failures), metrics, tuple(gates))


def evaluate_model_acceptance_gates(
    metrics: dict[str, Any],
    config: V7ModelAcceptanceGateConfig | None = None,
    paper_report_path: str | Path | None = None,
) -> V7GateReport:
    config = config or V7ModelAcceptanceGateConfig()
    gates: list[dict[str, Any]] = []

    def add_gate(
        name: str, passed: bool, actual: Any, threshold: Any, reason: str, **context: Any
    ) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "status": GATE_PASS if passed else GATE_FAIL,
                "actual": actual,
                "threshold": threshold,
                "reason": "passed" if passed else reason,
                **context,
            }
        )

    def add_waived_gate(name: str, actual: Any, requirement: str) -> None:
        """A gate the configuration switched off.

        Kept distinct from a measured pass. Recording `reason: "passed"` for a
        requirement nobody asked to check reads as "this cleared its threshold",
        which is a different statement from "this was not required" — and the
        difference matters when a reviewer is deciding what the run actually
        demonstrates.
        """
        gates.append(
            {
                "name": name,
                "passed": True,
                "status": GATE_PASS,
                "actual": actual,
                "threshold": f"not required ({requirement}=False)",
                "reason": "waived_by_configuration",
                "required": False,
            }
        )

    def add_unknown_gate(
        name: str, threshold: Any, reason: str, detail: str, **context: Any
    ) -> None:
        """Record a gate whose measurement is absent.

        It never passes, and it is kept distinct from a measured failure so an
        operator can tell "this candidate underperformed" from "nobody measured
        whether it did".

        `reason` is a machine-readable code and stays stable — anything matching on
        it would break if the code mutated to carry a new distinction. Extra
        `context` goes into structured fields, which is where a *cause* belongs.
        """
        gates.append(
            {
                "name": name,
                "passed": False,
                "status": GATE_UNKNOWN,
                "actual": None,
                "threshold": threshold,
                "reason": reason,
                "detail": detail,
                **context,
            }
        )

    def _optional_float(key_chain: tuple[str, ...]) -> float | None:
        for key in key_chain:
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _optional_int(key_chain: tuple[str, ...]) -> int | None:
        for key in key_chain:
            if key in metrics and metrics[key] is not None:
                try:
                    return int(metrics[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _optional_bool(key: str) -> bool | None:
        value = metrics.get(key)
        return None if value is None else bool(value)

    def add_measured_gate(
        name: str,
        keys: tuple[str, ...],
        holds,
        threshold: str,
        fail_reason: str,
        unknown_detail: str,
        *,
        as_int: bool = False,
    ) -> None:
        """A gate that needs a measurement, and says so when it does not have one.

        Every gate below used `metrics.get(key, 0.0)`. Measured consequence
        (DEF-023): with nothing measured at all, `max_drawdown`,
        `single_factor_dominance`, `no_mock_or_synthetic` and `no_pit_violations`
        all **passed** — a run that was never audited for leakage or synthetic data
        was recorded as clean on both, and one with no drawdown measurement was
        recorded as having an acceptable drawdown. The gates that did fail reported
        a fabricated `0.0` as the measured value, so an operator read
        "rank_ic_mean_not_positive" and concluded the model had no edge when in
        truth no IC had been computed.
        """
        value = _optional_int(keys) if as_int else _optional_float(keys)
        if value is None:
            add_unknown_gate(name, threshold, f"{name}_unknown", unknown_detail)
            return
        add_gate(name, holds(value), value, threshold, fail_reason)

    add_measured_gate(
        "rank_ic_mean",
        ("rank_ic_mean",),
        lambda v: v > config.min_rank_ic_mean,
        f"> {config.min_rank_ic_mean}",
        "rank_ic_mean_not_positive",
        "没有计算过 rank IC。这不是「IC 为 0」，而是「没有测量」——"
        "跑完 OOS 评估后重试。",
    )
    add_measured_gate(
        "rank_ic_stability",
        ("rank_ic_stability", "ICIR"),
        lambda v: v > config.min_rank_ic_stability,
        f"> {config.min_rank_ic_stability}",
        "rank_ic_stability_not_positive",
        "没有计算过 IC 稳定性 / ICIR。没有测量不等于稳定性为 0。",
    )
    add_measured_gate(
        "turnover_adjusted_net_return",
        ("turnover_adjusted_net_return",),
        lambda v: v > config.min_turnover_adjusted_return,
        f"> {config.min_turnover_adjusted_return}",
        "turnover_adjusted_net_return_failed",
        "没有换手调整后收益。没有测量不等于收益为 0。",
    )
    # This one *passed* on a missing measurement: abs(0.0) <= any limit. A run with
    # no drawdown measurement was recorded as having an acceptable drawdown.
    add_measured_gate(
        "max_drawdown",
        ("max_drawdown",),
        lambda v: abs(v) <= config.max_drawdown,
        f"abs(drawdown) <= {config.max_drawdown}",
        "max_drawdown_exceeded",
        "没有回撤测量。缺失的回撤**不是 0 回撤**——先前的默认值让未测量的运行通过了回撤闸门。",
    )
    if config.min_sharpe is not None:
        add_measured_gate(
            "sharpe",
            ("sharpe",),
            lambda v: v >= config.min_sharpe,
            f">= {config.min_sharpe}",
            "sharpe_below_minimum",
            "没有 Sharpe。没有测量不等于 Sharpe 为 0。",
        )
    # Also passed on a missing measurement: 0.0 <= any limit. A run with no
    # concentration measurement was recorded as perfectly diversified.
    add_measured_gate(
        "single_factor_dominance",
        ("single_factor_dominance",),
        lambda v: v <= config.max_single_factor_dominance,
        f"<= {config.max_single_factor_dominance}",
        "single_factor_dominance_too_high",
        "没有单因子集中度测量。缺失的集中度**不是 0 集中度**。",
    )
    adverse_actual = _optional_bool("adverse_regime_passed")
    if not config.require_adverse_regime:
        add_waived_gate("adverse_regime", adverse_actual, "require_adverse_regime")
    elif adverse_actual is None:
        add_unknown_gate(
            "adverse_regime",
            f"min_rank_ic={config.adverse_regime_min_rank_ic}",
            "adverse_regime_unknown",
            "没有在逆境 regime 上验证过。没有验证不等于验证未通过，也不等于通过。",
        )
    else:
        add_gate(
            "adverse_regime", adverse_actual, adverse_actual,
            f"min_rank_ic={config.adverse_regime_min_rank_ic}",
            "adverse_regime_not_validated",
        )
    report_exists = bool(paper_report_path and Path(paper_report_path).exists())
    if not config.require_paper_report:
        add_waived_gate(
            "paper_report",
            str(paper_report_path) if paper_report_path else None,
            "require_paper_report",
        )
    else:
        add_gate(
            "paper_report", report_exists,
            str(paper_report_path) if paper_report_path else None,
            "the report file exists", "paper_trading_report_missing",
        )
    has_benchmark = (
        bool(metrics.get("benchmark_symbol")) or metrics.get("benchmark_return") is not None
    )
    benchmark_actual = metrics.get("benchmark_symbol") or metrics.get("benchmark_return")
    if not config.require_benchmark:
        add_waived_gate("benchmark", benchmark_actual, "require_benchmark")
    else:
        add_gate(
            "benchmark", has_benchmark, benchmark_actual, "a benchmark is configured",
            "benchmark_missing_quant_alpha_not_validated",
            benchmark_status=metrics.get("benchmark_status"),
        )
    # Excess return is only defined against a benchmark. With no benchmark the
    # honest answer is "not measured" — coercing it to 0.0 and failing the gate
    # reported a run as having *measured* zero alpha.
    excess_after_costs = _optional_float(("excess_return_after_costs", "excess_return"))
    if excess_after_costs is None:
        # Two different causes, two different fixes. "No benchmark was requested"
        # and "the benchmark has holes in it" both produce an unmeasurable excess
        # return, and an operator told only the first would go looking for a
        # configuration problem that does not exist. The coverage counts come from
        # paper_report's benchmark_status (DEF-022).
        status = str(metrics.get("benchmark_status") or "absent")
        covered = metrics.get("benchmark_sessions_covered")
        missing = metrics.get("benchmark_sessions_missing")
        if status == "incomplete":
            remediation = (
                f"基准数据不完整：{covered} 个交易日有数据、{missing} 个缺失。"
                "缺口不是「当日零收益」——那样会把超额收益虚增基准在缺口期间的涨幅。"
                "补齐基准行情后重跑。"
            )
        else:
            remediation = (
                "没有基准，超额收益无从计算。这不是「超额为 0」，而是「没有测量」。"
                "指定 benchmarkSymbol 后重跑，或明确接受该候选不做超额验证。"
            )
        add_unknown_gate(
            "excess_return_after_costs",
            f"> {config.min_excess_return_after_costs}",
            "excess_return_after_costs_unknown",
            remediation,
            benchmark_status=status,
            benchmark_sessions_covered=covered,
            benchmark_sessions_missing=missing,
        )
    else:
        add_gate(
            "excess_return_after_costs",
            excess_after_costs > config.min_excess_return_after_costs,
            excess_after_costs,
            f"> {config.min_excess_return_after_costs}",
            "excess_return_after_costs_failed",
        )
    add_measured_gate(
        "selection_pressure",
        ("selection_pressure_min", "selection_pressure"),
        lambda v: v >= config.min_selection_pressure,
        f">= {config.min_selection_pressure}",
        "selection_pressure_too_low",
        "没有选股压力测量（候选数 / 入选数）。没有测量不等于压力为 0。",
    )
    add_measured_gate(
        "training_symbols",
        ("training_dataset_symbol_count", "training_symbol_count", "symbol_count"),
        lambda v: v >= config.min_training_symbols,
        f">= {config.min_training_symbols}",
        "insufficient_training_symbols",
        "训练标的数未记录。没有记录不等于 0 个标的。",
        as_int=True,
    )
    add_measured_gate(
        "prediction_symbols",
        ("prediction_symbol_count",),
        lambda v: v >= config.min_prediction_symbols,
        f">= {config.min_prediction_symbols}",
        "insufficient_prediction_symbols",
        "预测标的数未记录。没有记录不等于 0 个标的。",
        as_int=True,
    )
    add_measured_gate(
        "effective_universe_by_date",
        ("effective_universe_min", "eligible_symbol_count_min"),
        lambda v: v >= config.min_effective_universe_by_date,
        f">= {config.min_effective_universe_by_date}",
        "insufficient_effective_universe_by_date",
        "逐日有效宇宙规模未记录。没有记录不等于宇宙为空。",
        as_int=True,
    )
    # The most dangerous of the four: absent -> False -> passed. A run that was
    # never audited for synthetic data was recorded as having none.
    uses_mock = _optional_bool("uses_mock_or_synthetic")
    if not config.no_mock_or_synthetic:
        add_gate("no_mock_or_synthetic", True, uses_mock, "not required", "not_required")
    elif uses_mock is None:
        add_unknown_gate(
            "no_mock_or_synthetic", "False", "no_mock_or_synthetic_unknown",
            "没有做过合成/mock 数据审计。**没有审计不等于没有合成数据**——"
            "先前的默认值让未审计的运行通过了这道闸门。",
        )
    else:
        add_gate(
            "no_mock_or_synthetic", not uses_mock, uses_mock, "False",
            "mock_data_model_not_production_ready",
        )
    survivorship = metrics.get(SURVIVORSHIP_STATUS_KEY)
    if not config.require_survivorship_check:
        add_waived_gate("survivorship", survivorship, "require_survivorship_check")
    elif survivorship is None:
        add_unknown_gate(
            "survivorship", "pass", "survivorship_unknown",
            "没有做过幸存者偏差检查。**没有检查不等于没有偏差** —— "
            "先经 gold_bridge.build_masks 掩码面板，再跑 evaluate_survivorship。",
        )
    elif str(survivorship) == GATE_UNKNOWN:
        add_unknown_gate(
            "survivorship", "pass", "survivorship_unknown",
            "幸存者偏差查不了：见 survivorship_unknown_sessions / "
            "survivorship_delisted_symbols。数据无法支撑「无幸存者偏差」的断言。",
            survivorship_unknown_sessions=metrics.get(SURVIVORSHIP_UNKNOWN_SESSIONS_KEY),
            survivorship_delisted_symbols=metrics.get(SURVIVORSHIP_DELISTED_SYMBOLS_KEY),
        )
    else:
        add_gate(
            "survivorship", str(survivorship) == GATE_PASS, survivorship, "pass",
            "survivorship_bias_present",
            survivorship_delisted_symbols=metrics.get(SURVIVORSHIP_DELISTED_SYMBOLS_KEY),
        )
    alignment = metrics.get(LABEL_ALIGNMENT_STATUS_KEY)
    if not config.require_label_alignment_check:
        add_waived_gate("label_alignment", alignment, "require_label_alignment_check")
    elif alignment is None:
        add_unknown_gate(
            "label_alignment", "pass", "label_alignment_unknown",
            "没有比对过特征可用时点与其标签窗口。这与 no_pit_violations 不是同一件事："
            "后者只问「相对某个 as-of 日期是否可用」，而这里问的是"
            "「标签开始计收益时，这一行的特征拿得到吗」。"
            "先跑 evaluate_label_alignment(dataset)。",
        )
    elif str(alignment) == GATE_UNKNOWN:
        add_unknown_gate(
            "label_alignment", "pass", "label_alignment_unknown",
            "标签对齐查不了：数据集缺 available_at 或缺 forward_return_*d 标签。"
            "数据无法支撑「标签没有提前于特征」的断言。",
            label_alignment_violation_rows=metrics.get(LABEL_ALIGNMENT_VIOLATION_ROWS_KEY),
        )
    else:
        add_gate(
            "label_alignment", str(alignment) == GATE_PASS, alignment, "pass",
            "label_window_opens_before_features_are_available",
            label_alignment_violation_rows=metrics.get(LABEL_ALIGNMENT_VIOLATION_ROWS_KEY),
            label_alignment_worst_horizon=metrics.get(LABEL_ALIGNMENT_WORST_HORIZON_KEY),
        )
    # And the other: absent -> 0 -> passed. A run with no PIT audit was recorded as
    # having no look-ahead, which is the single claim this programme most needs to
    # be earned rather than defaulted.
    pit_violations = _optional_int(("pit_violation_count",))
    if not config.no_pit_violations:
        add_gate("no_pit_violations", True, pit_violations, "not required", "not_required")
    elif pit_violations is None:
        add_unknown_gate(
            "no_pit_violations", "0", "no_pit_violations_unknown",
            "没有做过 PIT 审计。**没有审计不等于没有前视**——"
            "先前的默认值让未审计的运行被记录为「无 PIT 违规」。",
        )
    else:
        add_gate(
            "no_pit_violations", pit_violations == 0, pit_violations, "0",
            "pit_violations_present",
        )
    failures = [str(gate["reason"]) for gate in gates if not gate["passed"]]
    return V7GateReport(not failures, tuple(failures), dict(metrics), tuple(gates))


# ---------------------------------------------------------------------------
# Survivorship (M5-02)
# ---------------------------------------------------------------------------
#: Metric keys `evaluate_survivorship` produces, so the acceptance gate can read
#: them out of a metrics dict like any other measurement.
SURVIVORSHIP_STATUS_KEY = "survivorship_status"
SURVIVORSHIP_UNKNOWN_SESSIONS_KEY = "survivorship_unknown_sessions"
SURVIVORSHIP_DELISTED_SYMBOLS_KEY = "survivorship_delisted_symbols"
SURVIVORSHIP_UNDATED_SYMBOLS_KEY = "survivorship_undated_delisted_symbols"


@dataclass(frozen=True)
class SurvivorshipReport:
    """Whether a panel can support a survivorship-free claim.

    Three outcomes, and the middle one is the point. `pass` means dead names are
    present *and* dated, so the panel contains the losers. `fail` means the panel
    demonstrably includes sessions after a name stopped trading. `unknown` means the
    data cannot answer — and per the programme's rule that is emphatically not a
    pass, because "we could not check for survivorship bias" and "there is no
    survivorship bias" are the two statements this repository has already confused
    once.
    """

    status: str
    total_sessions: int
    verified_eligible_sessions: int
    unknown_sessions: int
    delisted_symbols: int
    #: Names the master knows are dead but cannot date. Counted separately because
    #: it is the *actionable* number: `delisted_symbols` can only count names whose
    #: delisting is dated, so a register with no dates at all reports 0 there and
    #: says nothing about how much of the panel is affected.
    undated_delisted_symbols: int = 0
    unknown_by_mask: dict[str, int] = field(default_factory=dict)
    unknown_symbols: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total_sessions": self.total_sessions,
            "verified_eligible_sessions": self.verified_eligible_sessions,
            "unknown_sessions": self.unknown_sessions,
            "delisted_symbols": self.delisted_symbols,
            "undated_delisted_symbols": self.undated_delisted_symbols,
            "unknown_by_mask": dict(self.unknown_by_mask),
            "unknown_symbols": list(self.unknown_symbols),
            "detail": self.detail,
        }

    def as_metrics(self) -> dict[str, Any]:
        return {
            SURVIVORSHIP_STATUS_KEY: self.status,
            SURVIVORSHIP_UNKNOWN_SESSIONS_KEY: self.unknown_sessions,
            SURVIVORSHIP_DELISTED_SYMBOLS_KEY: self.delisted_symbols,
            SURVIVORSHIP_UNDATED_SYMBOLS_KEY: self.undated_delisted_symbols,
        }


def evaluate_survivorship(
    masked_panel: pd.DataFrame,
    *,
    master: pd.DataFrame | None = None,
    max_unknown_symbols: int = 12,
) -> SurvivorshipReport:
    """Does the panel contain the names that died?

    That is the survivorship question, and it is **not** the question
    `mask_post_delisting` answers. The mask says "is this row after the security
    died", which is FALSE for a live name *and* FALSE for a dead name's rows before
    it died. A panel that correctly includes dead names and correctly stops each one
    at its delisting date therefore has **zero** TRUE rows — by construction.

    Counting delisted names as "symbols with a TRUE row" (DEF-028) inverted the
    verdict on exactly the panels that are right. Measured on the shipped
    full-universe gold: 261 delisted names present, 424,662 sessions, not one bar
    past any delisting date — the healthiest shape available — and the audit
    reported "the panel contains no delisted names at all, which is the signature
    of survivorship bias". An operator acting on that would have gone looking for
    dead names that were already there.

    So presence is read from the security's *status*, and the mask is used for what
    it does answer: whether any row survived past its own delisting, which is a
    real defect when it happens. `master` is accepted for panels built before
    `listing_status` was published as a column.
    """
    required = {"symbol", "mask_post_delisting"}
    missing_columns = sorted(required - set(masked_panel.columns))
    if missing_columns:
        return SurvivorshipReport(
            status=GATE_UNKNOWN, total_sessions=len(masked_panel),
            verified_eligible_sessions=0, unknown_sessions=0, delisted_symbols=0,
            detail=(
                f"面板缺少 {missing_columns}，无法判断幸存者偏差。"
                "先经 gold_bridge.build_masks 生成三态掩码。"
            ),
        )

    total = len(masked_panel)
    post = masked_panel["mask_post_delisting"].astype("object")
    unknown_rows = post == MASK_UNKNOWN
    unknown_sessions = int(unknown_rows.sum())
    # Rows that outlived their own delisting date. Should be zero; when it is not,
    # the panel is trading a security that no longer exists.
    post_delisting_rows = int((post == MASK_TRUE).sum())
    verified = (
        int((masked_panel["eligibility_status"] == MASK_TRUE).sum())
        if "eligibility_status" in masked_panel.columns
        else 0
    )
    unknown_by_mask: dict[str, int] = {}
    if "unknown_masks" in masked_panel.columns:
        # Counted column-wise. The Python loop this replaces walked every row, which
        # was tolerable while survivorship ran once at gold-build time and is not now
        # that it runs on every training dataset — the same cost `build_masks` was
        # just vectorised out of.
        exploded = (
            masked_panel["unknown_masks"].astype(str).str.split(",").explode()
        )
        counts = exploded[exploded.str.len() > 0].value_counts()
        unknown_by_mask = {str(name): int(count) for name, count in counts.items()}
    undated = sorted(masked_panel.loc[unknown_rows, "symbol"].unique())
    unknown_symbols = tuple(undated[:max_unknown_symbols])

    listing_status = _resolve_panel_listing_status(masked_panel, master)
    if listing_status is None:
        return SurvivorshipReport(
            status=GATE_UNKNOWN, total_sessions=total,
            verified_eligible_sessions=verified, unknown_sessions=unknown_sessions,
            delisted_symbols=0, undated_delisted_symbols=len(undated),
            unknown_by_mask=unknown_by_mask, unknown_symbols=unknown_symbols,
            detail=(
                "面板没有 listing_status 列，也没有传入 master —— 无法知道其中哪些标的"
                "后来退了市。`mask_post_delisting` 回答不了这个问题：一个正确构建的面板"
                "（包含死掉的名字、且在退市日就停止）**本来就**一个 TRUE 行都没有。"
                "重建面板以带上 listing_status，或把 master 传给 evaluate_survivorship。"
            ),
        )
    delisted_symbols = int(
        masked_panel.loc[listing_status == GOLD_LISTING_DELISTED, "symbol"].nunique()
    )

    mismatch = _master_disagrees_with_mask(masked_panel, master, listing_status, post)
    if mismatch:
        return SurvivorshipReport(
            status=GATE_UNKNOWN, total_sessions=total,
            verified_eligible_sessions=verified, unknown_sessions=unknown_sessions,
            delisted_symbols=delisted_symbols, undated_delisted_symbols=len(undated),
            unknown_by_mask=unknown_by_mask, unknown_symbols=unknown_symbols,
            detail=mismatch,
        )

    if post_delisting_rows:
        symbols = masked_panel.loc[post == MASK_TRUE, "symbol"].nunique()
        detail = (
            f"{post_delisting_rows} 行发生在其标的退市日之后（{symbols} 个标的）。"
            "面板在证券已经不存在之后还在给它记录行情 —— 这是真正的错误，"
            "不是覆盖不足。"
        )
        status = GATE_FAIL
    elif unknown_sessions:
        detail = (
            f"{unknown_sessions} / {total} 个 (标的, 交易日) 的退市状态未知，"
            f"涉及 {len(undated)} 个标的（例如 {', '.join(unknown_symbols[:3])}）。"
            "两种成因：master 知道它们已退市但没有退市日期，或者该标的根本不在 master 里 —— "
            "无论哪种，都无法确认面板从哪一天起不该再包含它们。"
            "这不是「没有幸存者偏差」，而是「查不了」；"
            f"缺的是这 {len(undated)} 个标的的退市日期/身份，不是整个宇宙。"
        )
        status = GATE_UNKNOWN
    elif delisted_symbols == 0:
        detail = (
            "面板里一个退市标的都没有。对全宇宙面板而言这本身就是幸存者偏差的特征 —— "
            "只有活到今天的名字被包含进来了。若这是一个刻意的小样本宇宙，请显式声明。"
        )
        status = GATE_UNKNOWN
    else:
        detail = (
            f"面板包含 {delisted_symbols} 个后来退市的标的，且没有任何一行超出它自己的"
            "退市日；也没有任何 (标的, 交易日) 的退市状态未知。"
            "宇宙里既有活下来的，也有死掉的。"
        )
        status = GATE_PASS
    return SurvivorshipReport(
        status=status, total_sessions=total, verified_eligible_sessions=verified,
        unknown_sessions=unknown_sessions, delisted_symbols=delisted_symbols,
        undated_delisted_symbols=len(undated),
        unknown_by_mask=unknown_by_mask, unknown_symbols=unknown_symbols, detail=detail,
    )


#: Mirrors `gold_bridge.LISTING_STATUS_DELISTED` without importing it at module
#: scope; `gold_bridge` is already imported here for the mask constants.
GOLD_LISTING_DELISTED = "delisted"


def _master_disagrees_with_mask(
    panel: pd.DataFrame,
    master: pd.DataFrame | None,
    listing_status: pd.Series,
    post: pd.Series,
) -> str:
    """Detect a mask that was not built from the master it is being judged against.

    This is the hazard that produced a wrong verdict about the shipped gold panel:
    U0 keeps two masters, the artifact was built from one and audited against the
    other, and the audit answered confidently instead of noticing. A persisted mask
    carries no record of which master produced it, so the disagreement has to be
    inferred — and it is inferrable, because the two masters cannot both be right
    about the same security.

    The tell used here: if the master calls a security delisted but has no date for
    it, the mask cannot honestly say FALSE for that security's rows. FALSE is a
    confident "this row is not after the delisting", and nothing in *this* master
    supports that confidence. So a FALSE there came from somewhere else.
    """
    if master is None or "symbol" not in panel.columns:
        return ""
    if "delisting_date" not in master.columns:
        dated = pd.Series(dtype="datetime64[ns]")
    else:
        dated = pd.to_datetime(
            master.set_index(master["symbol"].astype(str))["delisting_date"],
            errors="coerce",
        )
    symbols = panel["symbol"].astype(str)
    has_date = symbols.isin(set(dated[dated.notna()].index)).to_numpy()
    undatable_dead = (listing_status == GOLD_LISTING_DELISTED).to_numpy() & ~has_date
    confident = undatable_dead & (post == MASK_FALSE).to_numpy()
    if not confident.any():
        return ""
    affected = sorted(symbols[confident].unique())
    return (
        f"掩码与传入的 master 不一致：{len(affected)} 个标的（例如 "
        f"{', '.join(affected[:3])}）在这个 master 里是「已退市且无退市日期」，"
        f"但面板的 mask_post_delisting 对它们给出了确定的 FALSE —— "
        "这个 master 支撑不了那个确定性，所以掩码不是由它构建的。"
        "用构建该产物时真正使用的 master（见 lineage.json 的 inputs.security_master）"
        "重新审计；拿另一个 master 得到的结论衡量的是两者的差异，不是产物本身。"
    )


def _resolve_panel_listing_status(
    panel: pd.DataFrame, master: pd.DataFrame | None
) -> pd.Series | None:
    """Per-row listing status, from the panel's own column or from a master."""
    if "listing_status" in panel.columns:
        return panel["listing_status"].astype("object")
    if master is None or "symbol" not in panel.columns:
        return None
    from quantagent.data.ashare.gold_bridge import resolve_listing_status

    resolved, _ = resolve_listing_status(master)
    return panel["symbol"].astype(str).map(resolved)


# ---------------------------------------------------------------------------
# Label alignment / leakage (M5-02, DEF-026)
# ---------------------------------------------------------------------------
LABEL_ALIGNMENT_STATUS_KEY = "label_alignment_status"
LABEL_ALIGNMENT_VIOLATION_ROWS_KEY = "label_alignment_violation_rows"
LABEL_ALIGNMENT_WORST_HORIZON_KEY = "label_alignment_worst_horizon"


@dataclass(frozen=True)
class LabelAlignmentReport:
    """Was every feature row knowable before the return it is scored on began?

    `no_pit_violations` asks whether a row was knowable by some as-of date.
    This asks the question that decides whether a backtest number means anything:
    the row carries a stamp saying when a decision maker could first have used it,
    and a label measuring a return over a window. If the stamp falls inside the
    window, the label credits the model with a move it could not have traded.
    """

    status: str
    total_rows: int
    violation_rows: int
    entry_basis: str
    by_horizon: dict[str, dict[str, Any]] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total_rows": self.total_rows,
            "violation_rows": self.violation_rows,
            "entry_basis": self.entry_basis,
            "by_horizon": {key: dict(value) for key, value in self.by_horizon.items()},
            "detail": self.detail,
        }

    def as_metrics(self) -> dict[str, Any]:
        worst = max(
            self.by_horizon.items(),
            key=lambda kv: kv[1].get("violation_rows", 0),
            default=(None, {}),
        )[0]
        return {
            LABEL_ALIGNMENT_STATUS_KEY: self.status,
            LABEL_ALIGNMENT_VIOLATION_ROWS_KEY: self.violation_rows,
            LABEL_ALIGNMENT_WORST_HORIZON_KEY: worst,
        }


def evaluate_label_alignment(
    dataset: pd.DataFrame,
    *,
    availability_column: str = "available_at",
    entry_column: str = "label_entry_at",
) -> LabelAlignmentReport:
    """Compare each row's availability stamp against its own label windows.

    The label window opens where the position is entered. A builder that enters on
    a delayed session (`gold_bridge.LABEL_CONVENTION`: ``close(t+1+h)/close(t+1)``)
    should publish that instant as `label_entry_at`; absent that column the entry
    is taken to be `trade_date`, which is what `v7_label_builder` implements
    (``close(t+h)/close(t) - 1``). The assumption is recorded in `entry_basis`
    rather than left implicit, because the two conventions coexist in this
    repository and the audit's verdict depends on which one is in force.
    """
    if dataset is None or len(dataset) == 0:
        return LabelAlignmentReport(
            GATE_UNKNOWN, 0, 0, "none", detail="空数据集，无法判断标签对齐。"
        )
    columns = set(dataset.columns)
    if availability_column not in columns:
        return LabelAlignmentReport(
            GATE_UNKNOWN, len(dataset), 0, "none",
            detail=(
                f"数据集没有 {availability_column} 列 —— 没有任何一行声明自己何时可用，"
                "因此无法判断标签窗口是否早于特征可用时点。"
                "这不是「没有泄漏」，而是「查不了」。"
            ),
        )
    label_columns = sorted(
        column for column in columns
        if column.startswith("forward_return_") and column.endswith("d")
    )
    if not label_columns:
        return LabelAlignmentReport(
            GATE_UNKNOWN, len(dataset), 0, "none",
            detail="数据集没有 forward_return_*d 标签列，没有标签窗口可比。",
        )

    if entry_column in columns:
        entry = pd.to_datetime(dataset[entry_column], errors="coerce")
        entry_basis = entry_column
    elif "trade_date" in columns:
        entry = pd.to_datetime(dataset["trade_date"], errors="coerce")
        entry_basis = "trade_date (v7_label_builder: close(t+h)/close(t)-1)"
    else:
        return LabelAlignmentReport(
            GATE_UNKNOWN, len(dataset), 0, "none",
            detail=f"既没有 {entry_column} 也没有 trade_date，无法定位标签窗口起点。",
        )
    available = pd.to_datetime(dataset[availability_column], errors="coerce")

    by_horizon: dict[str, dict[str, Any]] = {}
    violating = pd.Series(False, index=dataset.index)
    for label in label_columns:
        horizon = label[len("forward_return_"):-1]
        labelled = pd.to_numeric(dataset[label], errors="coerce").notna()
        end_column = f"label_end_{horizon}d"
        window_end = (
            pd.to_datetime(dataset[end_column], errors="coerce")
            if end_column in columns else None
        )
        late = labelled & available.notna() & entry.notna() & (available > entry)
        inside = (
            int((late & window_end.notna() & (available <= window_end)).sum())
            if window_end is not None else None
        )
        by_horizon[label] = {
            "labelled_rows": int(labelled.sum()),
            "violation_rows": int(late.sum()),
            "violation_rows_inside_window": inside,
            "window_end_column": end_column if window_end is not None else None,
        }
        violating |= late

    violation_rows = int(violating.sum())
    labelled_total = int(sum(stats["labelled_rows"] for stats in by_horizon.values()))
    if labelled_total == 0:
        return LabelAlignmentReport(
            GATE_UNKNOWN, len(dataset), 0, entry_basis, by_horizon,
            detail="所有标签列都是空的，没有一行带标签可以判断对齐。",
        )
    if violation_rows == 0:
        return LabelAlignmentReport(
            GATE_PASS, len(dataset), 0, entry_basis, by_horizon,
            detail=(
                f"{labelled_total} 个带标签的 (标的, 交易日) 中没有任何一行的 "
                f"{availability_column} 晚于其标签窗口起点（起点取自 {entry_basis}）。"
            ),
        )
    worst = max(by_horizon.items(), key=lambda kv: kv[1]["violation_rows"])
    return LabelAlignmentReport(
        GATE_FAIL, len(dataset), violation_rows, entry_basis, by_horizon,
        detail=(
            f"{violation_rows} 行的 {availability_column} 晚于其标签窗口起点"
            f"（起点取自 {entry_basis}），最严重的是 {worst[0]}："
            f"{worst[1]['violation_rows']} / {worst[1]['labelled_rows']} 行。"
            "标签在特征可用之前就开始计收益 —— 模型被记功于它当时拿不到的信息。"
            "要么把入场推迟到 available_at（并发布 label_entry_at），"
            "要么把 available_at 改成该行真正可知的时点；两者不能都不动。"
        ),
    )


# ---------------------------------------------------------------------------
# Audits that decline to answer (DEF-025)
# ---------------------------------------------------------------------------
#: The audit ran against columns that can support an answer.
AUDIT_MEASURED = "measured"
#: The frame carries nothing the audit could read. Reported as `None`, never as a
#: clean number.
AUDIT_NOT_AUDITABLE = "not_auditable"

PIT_AUDIT_STATUS_KEY = "pit_audit_status"
PIT_AUDIT_DETAIL_KEY = "pit_audit_detail"
PIT_AUDIT_REFERENCE_KEY = "pit_audit_reference_column"
PROVENANCE_AUDIT_STATUS_KEY = "provenance_audit_status"
PROVENANCE_AUDIT_DETAIL_KEY = "provenance_audit_detail"


@dataclass(frozen=True)
class PitAuditReport:
    status: str
    violations: int | None
    reference_column: str | None
    detail: str

    def as_metrics(self) -> dict[str, Any]:
        return {
            "pit_violation_count": self.violations,
            PIT_AUDIT_STATUS_KEY: self.status,
            PIT_AUDIT_DETAIL_KEY: self.detail,
            PIT_AUDIT_REFERENCE_KEY: self.reference_column,
        }


@dataclass(frozen=True)
class ProvenanceAuditReport:
    status: str
    uses_mock_or_synthetic: bool | None
    source_column: str | None
    detail: str

    def as_metrics(self) -> dict[str, Any]:
        return {
            "uses_mock_or_synthetic": self.uses_mock_or_synthetic,
            PROVENANCE_AUDIT_STATUS_KEY: self.status,
            PROVENANCE_AUDIT_DETAIL_KEY: self.detail,
        }


def audit_pit(frame: pd.DataFrame | None) -> PitAuditReport:
    """Count look-ahead violations, or say that none could be counted.

    This used to return `0` for a frame it could not read at all — no
    `available_at`, or an `available_at` with nothing to compare it against — and
    `0` is indistinguishable from a clean audit. Measured (DEF-025): the frame the
    v7 dataset builder produces has `available_at` but no as-of reference and no
    `point_in_time_valid`, so it took the fall-through branch and reported **0
    violations**. The acceptance gate then recorded `no_pit_violations` as a
    *measured pass* — for a dataset on which no PIT comparison had ever run.

    The DEF-023 hardening one layer up was defeated exactly here: making a gate say
    `unknown` on a missing measurement achieves nothing while the producer supplies
    a fabricated one.
    """
    if frame is None or len(frame) == 0:
        return PitAuditReport(
            AUDIT_NOT_AUDITABLE, None, None,
            "空表，没有任何行可供 PIT 审计。",
        )
    if "available_at" not in frame.columns:
        return PitAuditReport(
            AUDIT_NOT_AUDITABLE, None, None,
            "没有 available_at 列，无法判断任何一行在何时可用。"
            "这不是「没有前视」，而是「查不了」。",
        )
    reference = (
        "as_of_date" if "as_of_date" in frame.columns
        else "inference_date" if "inference_date" in frame.columns
        else ""
    )
    has_validity_flag = "point_in_time_valid" in frame.columns
    if not reference and not has_validity_flag:
        return PitAuditReport(
            AUDIT_NOT_AUDITABLE, None, None,
            "有 available_at，但没有可比的参照（as_of_date / inference_date），"
            "也没有 point_in_time_valid 标志 —— 没有任何比较可做。"
            "若要审计特征可用时点与其标签窗口的关系，用 evaluate_label_alignment。",
        )
    invalid = (
        int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum())
        if has_validity_flag else 0
    )
    if not reference:
        return PitAuditReport(
            AUDIT_MEASURED, invalid, "point_in_time_valid",
            f"没有 as-of 参照列；按 point_in_time_valid 标志计得 {invalid} 行无效。",
        )
    date_violations = int(
        (
            pd.to_datetime(frame["available_at"], errors="coerce")
            > pd.to_datetime(frame[reference], errors="coerce")
        ).sum()
    )
    total = date_violations + invalid
    return PitAuditReport(
        AUDIT_MEASURED, total, reference,
        f"对照 {reference}：{date_violations} 行 available_at 晚于参照时点"
        f"，另有 {invalid} 行 point_in_time_valid 为假。",
    )


def audit_provenance(frame: pd.DataFrame | None) -> ProvenanceAuditReport:
    """Say whether the rows came from mock or synthetic sources — or that no
    column records where they came from.

    Same defect as `audit_pit` (DEF-025): a frame with no provenance column at all
    returned `False`, which reads as "audited, and clean". Not having asked is not
    an answer.
    """
    if frame is None or len(frame) == 0:
        return ProvenanceAuditReport(
            AUDIT_NOT_AUDITABLE, None, None, "空表，没有任何行可供来源审计。"
        )
    for column in ("source", "source_name", "data_source"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            hits = int(values.str.contains("mock|synthetic|demo").sum())
            return ProvenanceAuditReport(
                AUDIT_MEASURED, hits > 0, column,
                f"按 {column} 列审计：{hits} / {len(frame)} 行来自 mock/synthetic/demo。",
            )
    return ProvenanceAuditReport(
        AUDIT_NOT_AUDITABLE, None, None,
        "没有任何来源列（source / source_name / data_source），无法判断数据是真是合成。"
        "**没有审计不等于没有合成数据。**",
    )


def evaluate_adverse_regime(
    predictions: pd.DataFrame | None,
    market_panel: pd.DataFrame | None = None,
    label_column: str = "forward_return_1d",
    config: V7ModelAcceptanceGateConfig | None = None,
) -> dict[str, Any]:
    """Score the model in adverse regimes.

    Adverse regime is defined as trading days where the cross-sectional
    market return is in the bottom-quartile of the prediction window.
    We compute the rank-IC inside that subset and compare with the
    ``adverse_regime_*`` thresholds in ``V7ModelAcceptanceGateConfig``.

    Falls back to ``passed=False`` (not ``True``) when there is not
    enough data to evaluate — silent passes are no longer allowed.
    """
    config = config or V7ModelAcceptanceGateConfig()
    report: dict[str, Any] = {
        "passed": False,
        "reason": "insufficient_data",
        "adverse_dates_count": 0,
        "adverse_rank_ic_mean": 0.0,
    }
    if predictions is None or predictions.empty:
        return report
    if label_column not in predictions.columns or "prediction" not in predictions.columns:
        report["reason"] = "missing_prediction_or_label_columns"
        return report
    data = predictions.copy()
    data["trade_date"] = pd.to_datetime(data.get("trade_date"), errors="coerce")
    data = data.dropna(subset=["trade_date", "prediction", label_column])
    if data.empty:
        return report
    daily_return = data.groupby("trade_date")[label_column].mean()
    if daily_return.empty:
        return report
    threshold = daily_return.quantile(0.25)
    adverse_dates = daily_return[daily_return <= threshold].index
    if len(adverse_dates) == 0:
        report["reason"] = "no_adverse_dates"
        return report
    subset = data[data["trade_date"].isin(adverse_dates)]
    by_date_ic = subset.groupby("trade_date").apply(
        lambda f: float(f["prediction"].rank().corr(f[label_column].rank()))
        if len(f) >= 2 and f["prediction"].nunique() >= 2 and f[label_column].nunique() >= 2
        else float("nan")
    ).dropna()
    rank_ic_mean = float(by_date_ic.mean()) if not by_date_ic.empty else 0.0
    report["adverse_dates_count"] = int(len(adverse_dates))
    report["adverse_rank_ic_mean"] = rank_ic_mean
    report["passed"] = bool(rank_ic_mean >= config.adverse_regime_min_rank_ic)
    report["reason"] = "evaluated"
    return report
