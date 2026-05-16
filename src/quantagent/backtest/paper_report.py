"""User-facing paper trading report writer for V7 A-share simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationResult
from quantagent.execution.broker_base import OrderSide
from quantagent.execution.cost_model import AShareCostModel


@dataclass(frozen=True)
class PaperReportConfig:
    initial_cash: float = 1_000_000.0
    benchmark_symbol: str | None = None
    slippage_bps: float = 8.0
    output_dir: str | Path = "paper_report"
    title: str = "QuantAgent V7 Paper Trading Report"


@dataclass(frozen=True)
class PaperReportResult:
    status: str
    output_dir: str
    files: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


def write_paper_report(
    result: AShareExecutionSimulationResult,
    *,
    market_panel: pd.DataFrame | None = None,
    config: PaperReportConfig | None = None,
) -> PaperReportResult:
    """Write JSON, CSV, Markdown and HTML report files for a paper run."""
    config = config or PaperReportConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = _trade_frame(result.order_audit, config)
    failed = result.failed_order_audit.copy()
    holdings = result.position_history.copy()
    pnl = _pnl_frame(result.nav, config.initial_cash, market_panel, config.benchmark_symbol)
    selected = _selected_stocks(trades, holdings)
    summary = _summary(pnl, trades, failed, config.initial_cash)

    files = {
        "selected_stocks": str(_write_csv(selected, output_dir / "selected_stocks.csv")),
        "trades": str(_write_csv(trades, output_dir / "trades.csv")),
        "failed_orders": str(_write_csv(failed, output_dir / "failed_orders.csv")),
        "holdings": str(_write_csv(holdings, output_dir / "holdings.csv")),
        "pnl": str(_write_csv(pnl, output_dir / "pnl.csv")),
    }
    payload = {
        "status": "passed",
        "summary": summary,
        "config": asdict(config),
        "files": files,
        "warnings": [
            "paper_report_is_not_financial_advice",
            "paper_report_uses_out_of_sample_target_weights_only_if_upstream_pipeline_enforced_it",
        ],
    }
    json_path = output_dir / "paper_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path = output_dir / "paper_report.md"
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    html_path = output_dir / "paper_report.html"
    html_path.write_text(_html_report(payload, pnl, trades, failed), encoding="utf-8")
    files |= {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}
    return PaperReportResult("passed", str(output_dir), files, summary)


def _trade_frame(orders: pd.DataFrame, config: PaperReportConfig) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "symbol",
                "side",
                "quantity",
                "reference_price",
                "avg_price",
                "gross_amount",
                "estimated_fee",
                "estimated_slippage",
                "status",
                "last_message",
            ]
        )
    frame = orders.copy()
    for column in ("filled_quantity", "avg_price", "reference_price"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    cost_model = AShareCostModel()
    fees: list[float] = []
    slippage: list[float] = []
    gross: list[float] = []
    for row in frame.to_dict("records"):
        side = OrderSide.BUY if str(row.get("side", "buy")).lower() == "buy" else OrderSide.SELL
        quantity = int(float(row.get("filled_quantity", 0) or 0))
        avg_price = float(row.get("avg_price", 0.0) or 0.0)
        ref_price = float(row.get("reference_price", avg_price) or avg_price)
        gross_amount = quantity * avg_price
        gross.append(gross_amount)
        fees.append(float(cost_model.calculate(side, quantity, avg_price)["total"]))
        if side == OrderSide.BUY:
            slippage.append(max(0.0, avg_price - ref_price) * quantity)
        else:
            slippage.append(max(0.0, ref_price - avg_price) * quantity)
    frame["gross_amount"] = gross
    frame["estimated_fee"] = fees
    frame["estimated_slippage"] = slippage
    return frame


def _pnl_frame(
    nav: pd.Series,
    initial_cash: float,
    market_panel: pd.DataFrame | None,
    benchmark_symbol: str | None,
) -> pd.DataFrame:
    if nav is None or nav.empty:
        return pd.DataFrame(columns=["trade_date", "nav", "daily_pnl", "daily_return", "drawdown"])
    frame = nav.rename("nav").reset_index()
    frame.columns = ["trade_date", "nav"]
    frame["daily_pnl"] = frame["nav"].diff().fillna(frame["nav"] - float(initial_cash))
    frame["daily_return"] = frame["nav"].pct_change().fillna(frame["nav"] / float(initial_cash) - 1.0)
    frame["drawdown"] = frame["nav"] / frame["nav"].cummax() - 1.0
    if market_panel is not None and benchmark_symbol:
        bench = market_panel.copy()
        bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
        bench = bench[bench["symbol"].astype(str) == str(benchmark_symbol)].sort_values("trade_date")
        if not bench.empty and "close" in bench.columns:
            bench["benchmark_return"] = pd.to_numeric(bench["close"], errors="coerce").pct_change().fillna(0.0)
            frame = frame.merge(bench[["trade_date", "benchmark_return"]], on="trade_date", how="left")
    return frame


def _selected_stocks(trades: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(
        set(trades.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(holdings.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    )
    return pd.DataFrame({"symbol": symbols})


def _summary(pnl: pd.DataFrame, trades: pd.DataFrame, failed: pd.DataFrame, initial_cash: float) -> dict[str, object]:
    if pnl.empty:
        final_nav = float(initial_cash)
        gross_return = 0.0
        max_drawdown = 0.0
    else:
        final_nav = float(pnl["nav"].iloc[-1])
        gross_return = final_nav / float(initial_cash) - 1.0
        max_drawdown = float(pnl["drawdown"].min())
    realized_pnl = final_nav - float(initial_cash)
    fees = float(trades.get("estimated_fee", pd.Series(dtype=float)).sum())
    slippage = float(trades.get("estimated_slippage", pd.Series(dtype=float)).sum())
    completed = trades[trades.get("status", pd.Series(dtype=str)).astype(str).isin(["filled", "partial"])] if not trades.empty else trades
    return {
        "initial_cash": float(initial_cash),
        "final_nav": final_nav,
        "realized_money_earned_lost": realized_pnl,
        "gross_return": gross_return,
        "net_return_after_estimated_costs": (final_nav - fees - slippage) / float(initial_cash) - 1.0,
        "total_estimated_fees": fees,
        "total_estimated_slippage": slippage,
        "max_drawdown": max_drawdown,
        "trade_count": int(len(completed)),
        "failed_order_count": int(0 if failed is None else len(failed)),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _markdown_report(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    lines = ["# QuantAgent V7 Paper Trading Report", "", "本报告只描述 paper/backtest 结果，不构成 financial advice。", ""]
    lines.append("## Summary")
    for key, value in summary.items():  # type: ignore[union-attr]
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Files")
    for key, value in payload["files"].items():  # type: ignore[union-attr]
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _html_report(payload: dict[str, object], pnl: pd.DataFrame, trades: pd.DataFrame, failed: pd.DataFrame) -> str:
    summary_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in payload["summary"].items())  # type: ignore[union-attr]
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>QuantAgent V7 Paper Trading Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5}}table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f6f8fa}}</style></head>
<body>
<h1>QuantAgent V7 Paper Trading Report</h1>
<p>本报告只描述 paper/backtest 结果，不构成 financial advice；LLM/Agent 不生成订单。</p>
<h2>Summary</h2><table>{summary_rows}</table>
<h2>Recent PnL</h2>{pnl.tail(20).to_html(index=False)}
<h2>Trades</h2>{trades.tail(50).to_html(index=False)}
<h2>Failed Orders</h2>{failed.tail(50).to_html(index=False) if failed is not None and not failed.empty else '<p>None</p>'}
</body></html>"""
