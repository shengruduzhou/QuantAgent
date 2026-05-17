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
    target_weights_path: str | None = None
    acceptance_report_path: str | Path | None = None
    metrics_path: str | Path | None = None
    full_pipeline_report_path: str | Path | None = None


@dataclass(frozen=True)
class PaperReportResult:
    status: str
    output_dir: str
    files: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    quant_acceptance_status: str = "not_evaluated"


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
    skipped = result.skipped_order_audit.copy()
    holdings = result.position_history.copy()
    pnl = _pnl_frame(result.nav, config.initial_cash, market_panel, config.benchmark_symbol)
    selected = _selected_stocks(trades, holdings)
    summary = _summary(pnl, trades, failed, skipped, config.initial_cash)
    acceptance = _load_quant_acceptance(config, output_dir)

    files = {
        "selected_stocks": str(_write_csv(selected, output_dir / "selected_stocks.csv")),
        "trades": str(_write_csv(trades, output_dir / "trades.csv")),
        "failed_orders": str(_write_csv(failed, output_dir / "failed_orders.csv")),
        "skipped_orders": str(_write_csv(skipped, output_dir / "skipped_orders.csv")),
        "holdings": str(_write_csv(holdings, output_dir / "holdings.csv")),
        "pnl": str(_write_csv(pnl, output_dir / "pnl.csv")),
    }
    if config.target_weights_path:
        files["target_weights"] = str(config.target_weights_path)

    warnings = [
        "paper_report_is_not_financial_advice",
        "paper_report_uses_out_of_sample_target_weights_only_if_upstream_pipeline_enforced_it",
    ]
    if not config.benchmark_symbol:
        warnings.append("benchmark_missing_quant_alpha_not_validated")

    payload = {
        "status": "report_generation_passed",
        "report_generation_status": "passed",
        "quant_acceptance_status": acceptance["status"],
        "summary": summary,
        "acceptance": acceptance["payload"],
        "config": asdict(config),
        "files": files,
        "warnings": warnings,
    }
    json_path = output_dir / "paper_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path = output_dir / "paper_report.md"
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    html_path = output_dir / "paper_report.html"
    html_path.write_text(_html_report(payload, pnl, trades, failed, skipped), encoding="utf-8")
    files |= {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}
    return PaperReportResult("passed", str(output_dir), files, summary, str(acceptance["status"]))


def _trade_frame(orders: pd.DataFrame, config: PaperReportConfig) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "symbol",
                "side",
                "quantity",
                "filled_quantity",
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
    if "quantity" not in frame.columns and "filled_quantity" in frame.columns:
        frame["quantity"] = frame["filled_quantity"]
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
    columns = [
        "symbol",
        "first_buy_date",
        "last_trade_date",
        "buy_count",
        "sell_count",
        "gross_buy_amount",
        "gross_sell_amount",
        "ending_market_value",
        "estimated_symbol_pnl",
    ]
    if not symbols:
        return pd.DataFrame(columns=columns)
    completed = trades.copy() if trades is not None else pd.DataFrame()
    if not completed.empty and "status" in completed.columns:
        completed = completed[completed["status"].astype(str).isin(["filled", "partial"])]
    latest_holdings = pd.DataFrame()
    if holdings is not None and not holdings.empty and {"trade_date", "symbol", "market_value"}.issubset(holdings.columns):
        latest_holdings = (
            holdings.sort_values("trade_date")
            .groupby("symbol", as_index=False)
            .tail(1)[["symbol", "market_value"]]
            .rename(columns={"market_value": "ending_market_value"})
        )
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_trades = completed[completed.get("symbol", pd.Series(dtype=str)).astype(str) == symbol] if not completed.empty else completed
        buys = symbol_trades[symbol_trades.get("side", pd.Series(dtype=str)).astype(str).str.lower() == "buy"] if not symbol_trades.empty else symbol_trades
        sells = symbol_trades[symbol_trades.get("side", pd.Series(dtype=str)).astype(str).str.lower() == "sell"] if not symbol_trades.empty else symbol_trades
        gross_buy = float(buys.get("gross_amount", pd.Series(dtype=float)).sum()) if not buys.empty else 0.0
        gross_sell = float(sells.get("gross_amount", pd.Series(dtype=float)).sum()) if not sells.empty else 0.0
        fees = float(symbol_trades.get("estimated_fee", pd.Series(dtype=float)).sum()) if not symbol_trades.empty else 0.0
        slippage = float(symbol_trades.get("estimated_slippage", pd.Series(dtype=float)).sum()) if not symbol_trades.empty else 0.0
        ending_value = 0.0
        if not latest_holdings.empty:
            matched = latest_holdings[latest_holdings["symbol"].astype(str) == symbol]
            if not matched.empty:
                ending_value = float(matched["ending_market_value"].iloc[-1])
        first_buy_date = None if buys.empty or "trade_date" not in buys.columns else str(pd.to_datetime(buys["trade_date"]).min().date())
        last_trade_date = None if symbol_trades.empty or "trade_date" not in symbol_trades.columns else str(pd.to_datetime(symbol_trades["trade_date"]).max().date())
        rows.append(
            {
                "symbol": symbol,
                "first_buy_date": first_buy_date,
                "last_trade_date": last_trade_date,
                "buy_count": int(len(buys)),
                "sell_count": int(len(sells)),
                "gross_buy_amount": gross_buy,
                "gross_sell_amount": gross_sell,
                "ending_market_value": ending_value,
                "estimated_symbol_pnl": gross_sell - gross_buy + ending_value - fees - slippage,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summary(
    pnl: pd.DataFrame,
    trades: pd.DataFrame,
    failed: pd.DataFrame,
    skipped: pd.DataFrame,
    initial_cash: float,
) -> dict[str, object]:
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
    net_return = (final_nav - fees - slippage) / float(initial_cash) - 1.0
    daily_returns = pd.to_numeric(pnl.get("daily_return", pd.Series(dtype=float)), errors="coerce").dropna() if not pnl.empty else pd.Series(dtype=float)
    annualized_return = float((1.0 + gross_return) ** (252.0 / max(len(daily_returns), 1)) - 1.0) if len(daily_returns) else 0.0
    daily_std = float(daily_returns.std(ddof=0)) if len(daily_returns) else 0.0
    annualized_volatility = float(daily_std * (252.0**0.5))
    sharpe = float(daily_returns.mean() / daily_std * (252.0**0.5)) if daily_std > 0 else None
    benchmark_return = None
    benchmark_max_drawdown = None
    information_ratio = None
    if not pnl.empty and "benchmark_return" in pnl.columns:
        bench_returns = pd.to_numeric(pnl["benchmark_return"], errors="coerce").fillna(0.0)
        benchmark_curve = (1.0 + bench_returns).cumprod()
        benchmark_return = float(benchmark_curve.iloc[-1] - 1.0) if not benchmark_curve.empty else None
        benchmark_max_drawdown = float((benchmark_curve / benchmark_curve.cummax() - 1.0).min()) if not benchmark_curve.empty else None
        excess_daily = daily_returns.reset_index(drop=True).reindex(bench_returns.reset_index(drop=True).index).fillna(0.0) - bench_returns.reset_index(drop=True)
        tracking_error = float(excess_daily.std(ddof=0))
        information_ratio = float(excess_daily.mean() / tracking_error * (252.0**0.5)) if tracking_error > 0 else None
    excess_return = None if benchmark_return is None else float(net_return - benchmark_return)
    return {
        "initial_cash": float(initial_cash),
        "final_nav": final_nav,
        "realized_money_earned_lost": realized_pnl,
        "gross_return": gross_return,
        "net_return_after_estimated_costs": net_return,
        "turnover_adjusted_net_return": net_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "excess_return_after_costs": excess_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "information_ratio": information_ratio,
        "total_estimated_fees": fees,
        "total_estimated_slippage": slippage,
        "trade_count": int(len(completed)),
        "failed_order_count": int(0 if failed is None else len(failed)),
        "skipped_order_count": int(0 if skipped is None else len(skipped)),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _markdown_report(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    acceptance = payload.get("acceptance", {})
    metrics = acceptance.get("metrics", {}) if isinstance(acceptance, dict) else {}
    lines = [
        "# QuantAgent V7 Paper Trading Report",
        "",
        "本报告只描述 paper/backtest 结果，不构成 financial advice；LLM/Agent 不生成订单。",
        "",
        "## Quant Acceptance",
        f"- `status`: {payload.get('quant_acceptance_status')}",
    ]
    if isinstance(acceptance, dict):
        for failure in acceptance.get("failures", []):
            lines.append(f"- `failure`: {failure}")
    for key in (
        "rank_ic_mean",
        "rank_ic_stability",
        "ICIR",
        "turnover_adjusted_net_return",
        "max_drawdown",
        "adverse_regime_passed",
        "single_factor_dominance",
        "benchmark_excess_return",
        "excess_return",
        "selection_pressure_min",
    ):
        if key in metrics:
            lines.append(f"- `{key}`: {metrics[key]}")
    lines.extend(["", "## Summary"])
    for key, value in summary.items():  # type: ignore[union-attr]
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Files")
    for key, value in payload["files"].items():  # type: ignore[union-attr]
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _html_report(
    payload: dict[str, object],
    pnl: pd.DataFrame,
    trades: pd.DataFrame,
    failed: pd.DataFrame,
    skipped: pd.DataFrame,
) -> str:
    summary_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in payload["summary"].items())  # type: ignore[union-attr]
    acceptance = payload.get("acceptance", {})
    acceptance_metrics = acceptance.get("metrics", {}) if isinstance(acceptance, dict) else {}
    acceptance_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in {
            "status": payload.get("quant_acceptance_status"),
            "failures": acceptance.get("failures", []) if isinstance(acceptance, dict) else [],
            "rank_ic_mean": acceptance_metrics.get("rank_ic_mean"),
            "rank_ic_stability": acceptance_metrics.get("rank_ic_stability", acceptance_metrics.get("ICIR")),
            "turnover_adjusted_net_return": acceptance_metrics.get("turnover_adjusted_net_return"),
            "max_drawdown": acceptance_metrics.get("max_drawdown"),
            "adverse_regime_passed": acceptance_metrics.get("adverse_regime_passed"),
            "single_factor_dominance": acceptance_metrics.get("single_factor_dominance"),
            "benchmark_excess_return": acceptance_metrics.get("benchmark_excess_return", acceptance_metrics.get("excess_return")),
            "selection_pressure": acceptance_metrics.get("selection_pressure_min"),
        }.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>QuantAgent V7 Paper Trading Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5}}table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f6f8fa}}</style></head>
<body>
<h1>QuantAgent V7 Paper Trading Report</h1>
<p>本报告只描述 paper/backtest 结果，不构成 financial advice；LLM/Agent 不生成订单。</p>
<h2>Quant Acceptance</h2><table>{acceptance_rows}</table>
<h2>Summary</h2><table>{summary_rows}</table>
<h2>Recent PnL</h2>{pnl.tail(20).to_html(index=False)}
<h2>Trades</h2>{trades.tail(50).to_html(index=False)}
<h2>Failed Orders</h2>{failed.tail(50).to_html(index=False) if failed is not None and not failed.empty else '<p>None</p>'}
<h2>Skipped Orders</h2>{skipped.tail(50).to_html(index=False) if skipped is not None and not skipped.empty else '<p>None</p>'}
</body></html>"""


def _load_quant_acceptance(config: PaperReportConfig, output_dir: Path) -> dict[str, object]:
    candidates = [
        config.acceptance_report_path,
        output_dir / "acceptance_report.json",
        output_dir.parent / "acceptance_report.json",
        config.full_pipeline_report_path,
        config.metrics_path,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "acceptance_report" in payload and isinstance(payload["acceptance_report"], dict):
            payload = payload["acceptance_report"]
        if "passed" in payload or "failures" in payload or "metrics" in payload:
            status = "passed" if bool(payload.get("passed", False)) else "failed"
            return {"status": status, "payload": payload, "path": str(path)}
    return {
        "status": "not_evaluated",
        "payload": {
            "passed": False,
            "failures": [],
            "metrics": {},
            "reason": "metrics_or_acceptance_report_missing",
        },
    }
