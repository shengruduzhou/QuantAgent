"""Unified v8 result tables.

The v8 experiment directories contain a mix of ``headline_report.json`` and
strict backtest ``metrics.json`` files. This module normalises them into one
table so total return, annualised return, volatility, drawdown and excess
return are never mixed by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPORT_COLUMNS = (
    "market_env",
    "strategy",
    "horizon",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "turnover",
    "cost_after_return",
    "excess_equal_weight_return",
    "benchmark_equal_weight_ann",
    "return_first_score",
    "path",
)


@dataclass(frozen=True)
class ResultScoreConfig:
    """Return-first score used for ranking candidate model configurations."""

    max_drawdown_soft_cap: float = 0.25
    drawdown_penalty: float = 0.50
    excess_weight: float = 1.00


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _infer_market_env(path: Path, headline: dict) -> str:
    parts = {p.lower() for p in path.parts}
    if "bull" in parts:
        return "bull"
    if "bear" in parts:
        return "bear"
    start = str(headline.get("oos_start", ""))
    end = str(headline.get("oos_end", ""))
    if "2022" in start or "2023" in end:
        return "bear_like"
    if "2024" in start or "2026" in end:
        return "bull_like"
    return "unknown"


def _metrics_path_for_headline(headline_path: Path) -> Path | None:
    candidate = headline_path.parent / "backtest" / "metrics.json"
    if candidate.exists():
        return candidate
    candidates = sorted(headline_path.parent.rglob("metrics.json"))
    return candidates[0] if candidates else None


def _strategy_name(path: Path, root: Path) -> str:
    try:
        rel = path.parent.relative_to(root)
        return str(rel)
    except ValueError:
        return path.parent.name


def _score(row: dict, cfg: ResultScoreConfig) -> float:
    ann = float(row.get("cost_after_return") or 0.0)
    excess = float(row.get("excess_equal_weight_return") or 0.0)
    max_dd = float(row.get("max_drawdown") or 0.0)
    dd_over = max(0.0, max_dd - cfg.max_drawdown_soft_cap)
    return ann + cfg.excess_weight * excess - cfg.drawdown_penalty * dd_over


def collect_v8_result_rows(
    roots: Iterable[str | Path],
    *,
    score_config: ResultScoreConfig | None = None,
) -> pd.DataFrame:
    """Collect v8 headline/metric files into one normalised table."""
    cfg = score_config or ResultScoreConfig()
    rows: list[dict] = []
    for root_in in roots:
        root = Path(root_in)
        if not root.exists():
            continue
        for headline_path in sorted(root.rglob("headline_report.json")):
            headline = _read_json(headline_path)
            metrics_path = _metrics_path_for_headline(headline_path)
            metrics = _read_json(metrics_path) if metrics_path is not None else {}
            ann = metrics.get("annualized_return", headline.get("strategy_ann"))
            bench = headline.get("benchmark_equal_weight_ann")
            excess = headline.get("excess_return_ann")
            if excess is None and ann is not None and bench is not None:
                excess = float(ann) - float(bench)
            row = {
                "market_env": _infer_market_env(headline_path, headline),
                "strategy": _strategy_name(headline_path, root),
                "horizon": headline.get("horizon", "unknown"),
                "total_return": metrics.get("total_return"),
                "annualized_return": ann,
                "annualized_volatility": metrics.get("volatility"),
                "sharpe": metrics.get("sharpe", headline.get("strategy_sharpe")),
                "max_drawdown": metrics.get("max_drawdown", headline.get("strategy_max_dd")),
                "calmar": metrics.get("calmar"),
                "turnover": metrics.get("turnover", headline.get("strategy_turnover")),
                # Strict backtest NAV is already net of commission/stamp/slippage.
                "cost_after_return": ann,
                "excess_equal_weight_return": excess,
                "benchmark_equal_weight_ann": bench,
                "path": str(headline_path.parent),
            }
            row["return_first_score"] = _score(row, cfg)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    out = pd.DataFrame(rows)
    for col in REPORT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(REPORT_COLUMNS)].sort_values(
        ["market_env", "return_first_score"], ascending=[True, False],
    ).reset_index(drop=True)


def write_v8_result_table(
    table: pd.DataFrame,
    *,
    output_csv: str | Path,
    output_md: str | Path | None = None,
) -> dict[str, Path]:
    """Write CSV plus an optional compact Markdown table."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    paths = {"csv": csv_path}
    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        display = table.copy()
        numeric = [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe", "max_drawdown", "calmar", "turnover",
            "cost_after_return", "excess_equal_weight_return",
            "benchmark_equal_weight_ann", "return_first_score",
        ]
        for col in numeric:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce").round(4)
        md_path.write_text(display.to_markdown(index=False), encoding="utf-8")
        paths["md"] = md_path
    return paths


__all__ = [
    "REPORT_COLUMNS",
    "ResultScoreConfig",
    "collect_v8_result_rows",
    "write_v8_result_table",
]
