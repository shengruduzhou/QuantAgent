"""Walk-forward reporting utilities for the intraday Do-T EV engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASELINE_NAMES = [
    "no_t_baseline",
    "old_parametric_t_strategy",
    "new_engine_production_gates",
    "new_dynamic_ev_engine",
    "forced_trade_diagnostic_engine",
    "random_time_same_count_baseline",
    "shuffled_signal_baseline",
    "vwap_only_baseline",
]


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardReport:
    verdict: str
    reason: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    baseline_comparison: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    confidence_buckets: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_buckets: pd.DataFrame = field(default_factory=pd.DataFrame)


def make_walk_forward_splits(
    dates: Iterable[object],
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardSplit]:
    unique = pd.Series(pd.to_datetime(list(dates), errors="coerce")).dropna().dt.normalize().drop_duplicates().sort_values()
    ds = list(unique)
    if not ds:
        return []
    step = int(step_days or test_days)
    splits: list[WalkForwardSplit] = []
    total = train_days + validation_days + test_days
    for start in range(0, max(0, len(ds) - total + 1), max(1, step)):
        tr = ds[start:start + train_days]
        va = ds[start + train_days:start + train_days + validation_days]
        te = ds[start + train_days + validation_days:start + total]
        if len(tr) == train_days and len(va) == validation_days and len(te) == test_days:
            splits.append(WalkForwardSplit(tr[0], tr[-1], va[0], va[-1], te[0], te[-1]))
    return splits


def evaluate_walk_forward_results(
    trades: pd.DataFrame,
    *,
    baselines: dict[str, pd.DataFrame] | None = None,
    min_round_trips: int = 300,
) -> WalkForwardReport:
    """Evaluate strict forward-test outputs and return ENABLE/PAPER_ONLY/DO_NOT_ENABLE."""
    if trades is None or trades.empty:
        return WalkForwardReport(
            verdict="DO_NOT_ENABLE",
            reason="no executed forward-validation trades; engine has no deployable evidence",
            metrics=_empty_metrics(),
        )
    t = trades.copy()
    if "trade_date" in t.columns:
        t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce").dt.normalize()
    for col in (
        "net_pnl_bps",
        "gross_pnl_bps",
        "daily_uplift_bps",
        "adverse_excursion_after_sell",
        "adverse_excursion_after_buy",
        "capacity_usage",
        "turnover",
    ):
        if col not in t.columns:
            t[col] = 0.0
        t[col] = pd.to_numeric(t[col], errors="coerce").fillna(0.0)
    for col in ("completed_round_trip", "eod_restore", "sell_high_fail_new_high", "buy_low_fail_breakdown"):
        if col not in t.columns:
            t[col] = 0
        t[col] = pd.to_numeric(t[col], errors="coerce").fillna(0).astype(int)
    if "action" not in t.columns:
        t["action"] = ""

    completed = int(t["completed_round_trip"].sum())
    executed_legs = int(len(t[t["action"].astype(str) != "NO_TRADE"]))
    eod_restore_count = int(t["eod_restore"].sum())
    net_positive = t.loc[t["completed_round_trip"] == 1, "net_pnl_bps"] > 0
    daily = t.groupby("trade_date")["daily_uplift_bps"].sum() if "trade_date" in t.columns else pd.Series(dtype=float)
    avg_net = float(t.loc[t["completed_round_trip"] == 1, "net_pnl_bps"].mean()) if completed else 0.0
    avg_gross = float(t.loc[t["completed_round_trip"] == 1, "gross_pnl_bps"].mean()) if completed else 0.0
    cost_to_gross = float((avg_gross - avg_net) / avg_gross) if avg_gross > 0 else 0.0
    metrics = {
        "executed_legs": executed_legs,
        "completed_round_trips": completed,
        "EOD_restore_count": eod_restore_count,
        "restore_ratio": float(eod_restore_count / max(executed_legs, 1)),
        "hit_rate": float(net_positive.mean()) if completed else 0.0,
        "avg_gross_bps": avg_gross,
        "avg_net_bps": avg_net,
        "daily_uplift_bps": float(daily.mean()) if len(daily) else 0.0,
        "annualized_uplift": float((1.0 + (daily.mean() / 10_000.0 if len(daily) else 0.0)) ** 244 - 1.0),
        "cost_to_gross_ratio": cost_to_gross,
        "max_intraday_drawdown": float(t.get("intraday_drawdown_bps", pd.Series(0.0, index=t.index)).min()),
        "adverse_excursion_after_sell": float(t["adverse_excursion_after_sell"].mean()),
        "adverse_excursion_after_buy": float(t["adverse_excursion_after_buy"].mean()),
        "sell_high_fail_new_high_rate": float(t["sell_high_fail_new_high"].mean()),
        "buy_low_fail_breakdown_rate": float(t["buy_low_fail_breakdown"].mean()),
        "capacity_usage": float(t["capacity_usage"].mean()),
        "turnover": float(t["turnover"].sum()),
    }
    conf = confidence_bucket_performance(t)
    regime = regime_bucket_performance(t)
    baseline_cmp = compare_baselines(t, baselines or {})
    verdict, reason = _verdict(metrics, baseline_cmp, min_round_trips)
    return WalkForwardReport(verdict, reason, metrics, baseline_cmp, conf, regime)


def confidence_bucket_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if "confidence" not in trades.columns or trades.empty:
        return pd.DataFrame(columns=["confidence_bucket", "count", "avg_net_bps", "hit_rate"])
    t = trades.copy()
    t["confidence"] = pd.to_numeric(t["confidence"], errors="coerce")
    t = t.dropna(subset=["confidence"])
    if t.empty:
        return pd.DataFrame(columns=["confidence_bucket", "count", "avg_net_bps", "hit_rate"])
    t["confidence_bucket"] = pd.cut(t["confidence"], bins=[-np.inf, 0.55, 0.65, 0.75, 0.85, np.inf])
    return (
        t.groupby("confidence_bucket", observed=False)
        .agg(count=("net_pnl_bps", "size"), avg_net_bps=("net_pnl_bps", "mean"), hit_rate=("net_pnl_bps", lambda x: float((x > 0).mean())))
        .reset_index()
    )


def regime_bucket_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if "regime" not in trades.columns or trades.empty:
        return pd.DataFrame(columns=["regime", "count", "avg_net_bps", "restore_ratio"])
    return (
        trades.groupby("regime", dropna=False)
        .agg(
            count=("net_pnl_bps", "size"),
            avg_net_bps=("net_pnl_bps", "mean"),
            restore_ratio=("eod_restore", "mean"),
        )
        .reset_index()
    )


def compare_baselines(
    trades: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int | str]]:
    out: dict[str, dict[str, float | int | str]] = {}
    engine_daily = _daily_uplift(trades)
    for name in BASELINE_NAMES:
        b = baselines.get(name)
        if b is None or b.empty:
            out[name] = {"available": 0, "daily_uplift_bps": 0.0, "delta_vs_engine_bps": 0.0}
            continue
        bd = _daily_uplift(b)
        out[name] = {
            "available": 1,
            "daily_uplift_bps": float(bd.mean()) if len(bd) else 0.0,
            "delta_vs_engine_bps": float(engine_daily.mean() - bd.mean()) if len(engine_daily) and len(bd) else 0.0,
        }
    return out


def render_markdown_report(report: WalkForwardReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A股日内做T模型重构报告",
        "",
        "## 1. 当前模型问题",
        "旧模型偏分时高低点打分，固定阈值和乐观成交容易导致过度交易或无交易。",
        "## 2. A股交易规则与T+1约束",
        "新 ledger 严格区分 carried_shares、today_sold、today_bought；今日买入不允许今日卖出。",
        "## 3. 做T label 重构",
        "逐分钟标签改为完整 round-trip success、net edge、adverse excursion、EOD restore。",
        "## 4. 分时特征工程",
        "特征为 causal rolling/expanding；Level-2 字段缺失时不伪造。",
        "## 5. AI量化模型结构",
        "表格模型输出 calibrated_probability、expected_net_edge_bps、risk_score。",
        "## 6. EV决策逻辑",
        "动作由动态成本、概率、流动性、趋势和库存约束共同决定，默认 NO_TRADE。",
        "## 7. 成交模拟",
        "主报告使用 conservative next-bar fill、5% minute volume cap、slippage/spread/limit risk。",
        "## 8. 库存与资金会计",
        "未闭合 pair 进入 EOD restore 风险事件，不计为成功 round-trip。",
        "## 9. 风控规则",
        "限制单票次数、卖出比例、超配比例、尾盘新开、涨跌停和单边趋势逆向交易。",
        "## 10. Walk-forward 回测协议",
        "train/validation/test 顺序切分；test 不参与调参。",
        "## 11. Baseline 对比",
        _format_baselines(report.baseline_comparison),
        "## 12. 结果表",
        _format_metrics(report.metrics),
        "## 13. 失败归因",
        report.reason,
        "## 14. 是否可部署",
        report.verdict,
        "## 15. 下一步修改建议",
        "接入真实分钟级训练集后跑完整 walk-forward；若 completed_round_trips < 300，只能继续 paper research。",
    ]
    path.write_text("\n\n".join(lines), encoding="utf-8")
    return path


def _daily_uplift(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "daily_uplift_bps" not in df.columns:
        return pd.Series(dtype=float)
    if "trade_date" in df.columns:
        t = df.copy()
        t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce").dt.normalize()
        return t.groupby("trade_date")["daily_uplift_bps"].sum()
    return pd.Series(pd.to_numeric(df["daily_uplift_bps"], errors="coerce").dropna())


def _verdict(
    metrics: dict[str, float | int | str],
    baseline_cmp: dict[str, dict[str, float | int | str]],
    min_round_trips: int,
) -> tuple[str, str]:
    completed = int(metrics.get("completed_round_trips", 0))
    if completed <= 0:
        return "DO_NOT_ENABLE", "no completed round-trip under conservative validation"
    if completed < min_round_trips:
        return "PAPER_ONLY", "信号未证伪，但证据不足，不能部署。"
    random_delta = float(baseline_cmp.get("random_time_same_count_baseline", {}).get("delta_vs_engine_bps", 0.0))
    shuffled_delta = float(baseline_cmp.get("shuffled_signal_baseline", {}).get("delta_vs_engine_bps", 0.0))
    gates = [
        float(metrics.get("avg_net_bps", 0.0)) > 0,
        float(metrics.get("daily_uplift_bps", 0.0)) > 0,
        random_delta > 0,
        shuffled_delta > 0,
        float(metrics.get("restore_ratio", 1.0)) <= 0.20,
        float(metrics.get("sell_high_fail_new_high_rate", 1.0)) <= 0.35,
        float(metrics.get("buy_low_fail_breakdown_rate", 1.0)) <= 0.35,
    ]
    if all(gates):
        return "ENABLE", "all conservative validation gates passed"
    return "DO_NOT_ENABLE", "one or more conservative validation gates failed"


def _empty_metrics() -> dict[str, float | int | str]:
    return {
        "executed_legs": 0,
        "completed_round_trips": 0,
        "EOD_restore_count": 0,
        "restore_ratio": 0.0,
        "hit_rate": 0.0,
        "avg_gross_bps": 0.0,
        "avg_net_bps": 0.0,
        "daily_uplift_bps": 0.0,
        "annualized_uplift": 0.0,
        "cost_to_gross_ratio": 0.0,
        "max_intraday_drawdown": 0.0,
        "adverse_excursion_after_sell": 0.0,
        "adverse_excursion_after_buy": 0.0,
        "sell_high_fail_new_high_rate": 0.0,
        "buy_low_fail_breakdown_rate": 0.0,
        "capacity_usage": 0.0,
        "turnover": 0.0,
    }


def _format_metrics(metrics: dict[str, float | int | str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in metrics.items())


def _format_baselines(baselines: dict[str, dict[str, float | int | str]]) -> str:
    if not baselines:
        return "- baseline artifacts not supplied"
    return "\n".join(
        f"- {name}: available={vals.get('available', 0)}, daily_uplift_bps={vals.get('daily_uplift_bps', 0.0)}, "
        f"delta_vs_engine_bps={vals.get('delta_vs_engine_bps', 0.0)}"
        for name, vals in baselines.items()
    )


__all__ = [
    "BASELINE_NAMES",
    "WalkForwardReport",
    "WalkForwardSplit",
    "compare_baselines",
    "confidence_bucket_performance",
    "evaluate_walk_forward_results",
    "make_walk_forward_splits",
    "regime_bucket_performance",
    "render_markdown_report",
]
