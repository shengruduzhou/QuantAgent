"""Turn a failed job's own output into an actionable diagnosis.

A job that reports nothing but ``exit code 1`` forces every operator to open a
log file and read a Python traceback before they can decide what to do. The
rules below read the log the job actually wrote and name the failure, the
evidence line it was derived from, and the next action. When no rule matches we
say so explicitly rather than inventing a cause: an unclassified failure still
carries its log tail, which is strictly more than the exit code alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class FailureRule:
    code: str
    title: str
    remediation: str
    patterns: tuple[re.Pattern[str], ...]
    detail_template: str | None = None
    retryable: bool = False


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Ordered most specific first; the first matching rule wins.
FAILURE_RULES: tuple[FailureRule, ...] = (
    FailureRule(
        code="insufficient_oos_dates",
        title="样本外交易日不足以支撑嵌套选择与最终 holdout",
        remediation=(
            "提高 nSplits（每折默认 20 个 OOS 交易日），或降低 selectionMinOosDays / "
            "selectionMinHoldoutDays。所需 OOS 天数 = 选择段 + holdout 段。"
        ),
        patterns=_compile(r"insufficient OOS dates for nested portfolio selection"),
    ),
    FailureRule(
        code="benchmark_absent",
        title="基准标的不在行情面板内",
        remediation=(
            "改用面板内存在的基准，或清空 benchmarkSymbol 并把超额收益偏好权重设为 0；"
            "全宇宙个股面板通常不含指数标的。"
        ),
        patterns=_compile(r"benchmark .* is absent from the market panel"),
    ),
    FailureRule(
        code="universe_too_small_for_top_k",
        title="可交易宇宙相对 Top-K 太小",
        remediation=(
            "选股压力 = 可交易宇宙 / Top-K。扩大研究范围（试点宇宙改为更多标的或改用全宇宙）、"
            "降低 Top-K，或放宽因子筛选，使每期有足够候选可选。"
        ),
        patterns=_compile(
            r"selection_pressure=[\d.]+ is below",
            r"top_k selection covers the eligible universe",
            r"every portfolio candidate failed",
        ),
    ),
    FailureRule(
        code="gpu_unavailable",
        title="要求 GPU 但训练进程拿不到 CUDA",
        remediation=(
            "确认 nvidia-smi 可见且显存未被其他任务占满；或改用 ridge 基线，"
            "或在策略中关闭 requireGpu（FT-Transformer 在 CPU 上会被拒绝而不是降级）。"
        ),
        patterns=_compile(
            r"require[_ -]?gpu",
            r"CUDA .*(not available|unavailable)",
            r"no CUDA-capable device",
        ),
    ),
    FailureRule(
        code="out_of_memory",
        title="进程内存或显存耗尽",
        remediation=(
            "缩小研究范围（universeScope / 更少 symbol）、降低 batch size，"
            "或等待占用显存的其他任务结束后重试。"
        ),
        patterns=_compile(
            r"CUDA out of memory",
            r"MemoryError",
            r"Cannot allocate memory",
            r"torch\.cuda\.OutOfMemoryError",
        ),
        retryable=True,
    ),
    FailureRule(
        code="killed_by_oom_killer",
        title="进程被操作系统终止（很可能是 OOM killer）",
        remediation=(
            "内核在内存压力下杀掉了进程。缩小研究范围或减少并发任务后重试；"
            "dmesg 会记录 Out of memory 事件。"
        ),
        patterns=_compile(r"\bKilled\b", r"signal 9", r"SIGKILL"),
        retryable=True,
    ),
    FailureRule(
        code="input_missing",
        title="输入产物缺失或不可读",
        remediation="在数据实验室确认该产物已生成；缺失的输入不会被静默替换。",
        patterns=_compile(
            r"FileNotFoundError",
            r"No such file or directory",
            r"input path does not exist",
            r"does not exist: ",
        ),
    ),
    FailureRule(
        code="schema_mismatch",
        title="输入 schema 与请求的研究周期不一致",
        remediation="重建 labels 或调整 horizons，使 forward_return_<h>d 列齐备。",
        patterns=_compile(
            r"missing .*(column|horizon)",
            r"KeyError: '(forward_return|label_end)",
            r"columns are missing",
        ),
    ),
    FailureRule(
        code="dependency_missing",
        title="缺少可选依赖",
        remediation="在服务端安装缺失的包后重试；生产路径不允许静默降级。",
        patterns=_compile(
            r"ModuleNotFoundError",
            r"ImportError",
            r"No module named",
        ),
    ),
    FailureRule(
        code="parameter_rejected",
        title="参数被命令契约拒绝",
        remediation="按提示修改参数后重新提交；该拒绝发生在任何计算之前。",
        patterns=_compile(
            r"BadParameter",
            r"Invalid value for",
            r"must be one of",
            r"must remain",
        ),
    ),
    FailureRule(
        code="research_gate_failed",
        title="研究闸门未通过",
        remediation=(
            "这不是工程故障：数据/OOS 证据没有达到预先声明的门槛。"
            "查看结论页的失败原因，再决定是否修改研究设计。"
        ),
        patterns=_compile(
            r"NOT_READY",
            r"BLOCKED_BY_DATA",
            r"acceptance gate .*(failed|rejected)",
        ),
    ),
)

UNCLASSIFIED = FailureRule(
    code="unclassified",
    title="未归类失败",
    remediation="查看下方日志尾部；若可复现，请把日志与参数一并保留后再重试。",
    patterns=(),
)

_TRACEBACK_FINAL = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):\s*(?P<message>.*)$")
_RICH_FINAL = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception)):\s*(?P<message>.+)$")


def extract_exception(lines: list[str]) -> str | None:
    """Return the exception line a Python process ended on, if there is one.

    Both plain tracebacks and Rich-rendered ones end with the exception type and
    message on their own line, sometimes wrapped across two lines. We scan
    backwards so the outermost (last raised) exception wins.
    """
    collected: list[str] = []
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip().strip("│").strip()
        if not stripped:
            continue
        match = _TRACEBACK_FINAL.match(stripped) or _RICH_FINAL.match(stripped)
        if match:
            message = match.group("message").strip()
            # Rich wraps long messages onto following lines; re-join them.
            for follow in lines[index + 1:index + 4]:
                text = follow.strip().strip("│").strip()
                if not text or text.startswith(("╰", "╭", "│ ", "$")):
                    break
                if _TRACEBACK_FINAL.match(text):
                    break
                message = f"{message} {text}"
            collected.append(f"{match.group('type')}: {message}".strip())
            break
    return collected[0] if collected else None


@dataclass
class JobFailure:
    code: str
    title: str
    detail: str
    remediation: str
    retryable: bool
    evidence: str | None = None
    log_tail: list[str] = field(default_factory=list)
    exit_code: int | None = None
    signal: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "remediation": self.remediation,
            "retryable": self.retryable,
            "evidence": self.evidence,
            "logTail": self.log_tail,
            "exitCode": self.exit_code,
            "signal": self.signal,
        }


def diagnose(
    lines: list[str],
    *,
    exit_code: int | None,
    cancelled: bool = False,
    tail_size: int = 40,
) -> JobFailure:
    """Classify a terminated job from its log tail and exit status."""

    tail = [line.rstrip() for line in lines[-tail_size:]]
    signal_number: int | None = None
    if exit_code is not None and exit_code < 0:
        signal_number = -exit_code
    haystack = "\n".join(tail)
    exception = extract_exception(tail)

    if cancelled:
        return JobFailure(
            code="cancelled",
            title="任务被操作员取消",
            detail="进程收到终止信号并退出。",
            remediation="需要时可用相同参数重试。",
            retryable=True,
            log_tail=tail,
            exit_code=exit_code,
            signal=signal_number,
        )

    for rule in FAILURE_RULES:
        for pattern in rule.patterns:
            match = pattern.search(haystack)
            if not match:
                continue
            evidence = next(
                (line for line in reversed(tail) if pattern.search(line)),
                None,
            )
            return JobFailure(
                code=rule.code,
                title=rule.title,
                detail=exception or (evidence or "").strip() or rule.title,
                remediation=rule.remediation,
                retryable=rule.retryable,
                evidence=evidence,
                log_tail=tail,
                exit_code=exit_code,
                signal=signal_number,
            )

    if signal_number is not None:
        return JobFailure(
            code="killed_by_signal",
            title=f"进程被信号 {signal_number} 终止",
            detail=f"process terminated by signal {signal_number}",
            remediation="确认没有外部 kill、OOM 或超时；然后重试。",
            retryable=True,
            log_tail=tail,
            exit_code=exit_code,
            signal=signal_number,
        )

    return JobFailure(
        code=UNCLASSIFIED.code,
        title=UNCLASSIFIED.title,
        detail=exception or f"command exited with code {exit_code}",
        remediation=UNCLASSIFIED.remediation,
        retryable=False,
        evidence=exception,
        log_tail=tail,
        exit_code=exit_code,
        signal=signal_number,
    )


__all__ = ["JobFailure", "diagnose", "extract_exception", "FAILURE_RULES"]
