"""Contracts for forward-looking A-share research reports.

The research layer must not become a prettier post-mortem.  Every report
declares the PIT cutoff and the future window it is trying to forecast, so
weekly/monthly runs remain comparable in later OOS review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd


Cadence = Literal["daily", "weekly", "monthly"]


@dataclass(frozen=True)
class PredictionWindow:
    """Future window covered by a research report."""

    cadence: Cadence
    as_of: str
    prediction_start: str
    prediction_end: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ForwardResearchContract:
    """Audit contract shared by daily, weekly and monthly reports."""

    window: PredictionWindow
    pit_cutoff_at: str
    report_type: str
    objective: str = "forecast future A-share sector and stock-pool opportunity, not summarize the past"
    min_events: int = 6
    min_themes: int = 4
    min_candidate_stocks: int = 20
    required_sections: tuple[str, ...] = (
        "market_outlook",
        "event_calendar",
        "themes",
        "stock_pool",
        "risk_controls",
        "oos_validation_plan",
    )
    external_design_refs: tuple[str, ...] = (
        "ZhuLinsen/daily_stock_analysis: multi-source data, news, dashboard, scheduled delivery",
        "TauricResearch/TradingAgents: analyst/researcher/trader/risk roles, memory, grounded sentiment",
    )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["window"] = self.window.as_dict()
        return data

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


@dataclass(frozen=True)
class ForwardResearchValidation:
    passed: bool
    warnings: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "warnings": list(self.warnings), "counts": self.counts}


def build_prediction_window(as_of: str | date | pd.Timestamp, cadence: Cadence) -> PredictionWindow:
    """Build a future prediction window from an as-of date."""

    ts = pd.Timestamp(as_of).normalize()
    if cadence == "daily":
        start = ts + pd.tseries.offsets.BDay(1)
        end = start
        label = f"{start.strftime('%Y年%m月%d日')} 当日预测"
    elif cadence == "weekly":
        start = ts + pd.tseries.offsets.BDay(1)
        end = start + pd.Timedelta(days=6)
        label = f"{start.strftime('%Y年%m月%d日')} 至 {end.strftime('%Y年%m月%d日')} 下周预测"
    elif cadence == "monthly":
        start = ts + pd.offsets.MonthBegin(1)
        end = start + pd.offsets.MonthEnd(0)
        label = f"{start.strftime('%Y年%m月')} 下月预测"
    else:
        raise ValueError(f"unsupported cadence: {cadence}")
    return PredictionWindow(
        cadence=cadence,
        as_of=ts.strftime("%Y-%m-%d"),
        prediction_start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        prediction_end=pd.Timestamp(end).strftime("%Y-%m-%d"),
        label=label,
    )


def build_forward_research_contract(
    as_of: str | date | pd.Timestamp,
    *,
    cadence: Cadence,
    report_type: str | None = None,
    min_events: int | None = None,
    min_themes: int | None = None,
    min_candidate_stocks: int | None = None,
) -> ForwardResearchContract:
    window = build_prediction_window(as_of, cadence)
    defaults = {"daily": (3, 2, 10), "weekly": (8, 4, 25), "monthly": (10, 5, 40)}
    ev, th, stocks = defaults[cadence]
    return ForwardResearchContract(
        window=window,
        pit_cutoff_at=f"{window.as_of} 23:59:59",
        report_type=report_type or f"{cadence}_forward_research",
        min_events=ev if min_events is None else int(min_events),
        min_themes=th if min_themes is None else int(min_themes),
        min_candidate_stocks=stocks if min_candidate_stocks is None else int(min_candidate_stocks),
    )


def validate_forward_research_payload(
    payload: dict[str, Any],
    contract: ForwardResearchContract,
    *,
    stock_count: int = 0,
) -> ForwardResearchValidation:
    """Lint a generated report payload for forward-looking breadth."""

    warnings: list[str] = []
    event_count = len(payload.get("event_calendar") or payload.get("events") or [])
    theme_count = len(payload.get("themes") or [])
    if event_count < contract.min_events:
        warnings.append(f"event_calendar too thin: {event_count} < {contract.min_events}")
    if theme_count < contract.min_themes:
        warnings.append(f"themes too thin: {theme_count} < {contract.min_themes}")
    if stock_count < contract.min_candidate_stocks:
        warnings.append(f"candidate stock pool too small: {stock_count} < {contract.min_candidate_stocks}")
    for section in contract.required_sections:
        if section == "stock_pool":
            continue
        if section in {"risk_controls", "oos_validation_plan"}:
            continue
        if section not in payload:
            warnings.append(f"missing section: {section}")
    return ForwardResearchValidation(
        passed=not warnings,
        warnings=tuple(warnings),
        counts={"events": event_count, "themes": theme_count, "candidate_stocks": int(stock_count)},
    )


def render_forward_research_header(contract: ForwardResearchContract) -> str:
    window = contract.window
    return "\n".join(
        [
            f"*预测窗口：{window.label} ({window.prediction_start} → {window.prediction_end})*",
            f"*PIT cutoff：{contract.pit_cutoff_at}；只允许使用 cutoff 之前可获得的信息。*",
            "*定位：forward-looking research evidence，不是订单，不是收益承诺。*",
        ]
    )
