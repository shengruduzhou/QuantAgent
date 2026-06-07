#!/usr/bin/env python3
"""Validate PIT LLM stock-selection outputs with strict A-share backtests.

This script is intentionally offline: it never calls an LLM.  It validates
LLM outputs that already exist on disk, then compares them with factor-only
baselines over a future window.  A typical use is:

  as_of=2026-03-31  -> use only reports/pools generated at or before as_of
  test=2026-04-01..2026-04-30 -> strict T+1/cost/slippage backtest

When no LLM chain pool is provided, the script still writes a factor baseline
report and marks LLM validation as blocked rather than fabricating a signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _score_col(frame: pd.DataFrame) -> str:
    for col in ("prediction", "alpha_score", "composite_score", "score"):
        if col in frame.columns:
            return col
    raise ValueError("prediction frame must include prediction/alpha_score/composite_score/score")


def _code6(symbol: str) -> str:
    return str(symbol).split(".")[0].zfill(6)


def _window_panel(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out[(out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))].copy()


def _available_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.Timestamp(d) for d in pd.to_datetime(panel["trade_date"], errors="coerce").dropna().unique())


def _benchmark_return(panel: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> tuple[pd.Series, float]:
    idx = list(dates)
    px = panel[panel["trade_date"].isin(idx)].pivot_table(index="trade_date", columns="symbol", values="close")
    daily = px.pct_change(fill_method=None).mean(axis=1).dropna()
    total = float((1.0 + daily).prod() - 1.0) if not daily.empty else 0.0
    return daily, total


def _normalise_weights(rows: pd.DataFrame, method: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    if method == "equal":
        return pd.Series(1.0 / len(rows), index=rows["symbol"].astype(str))
    score = pd.to_numeric(rows.get("mix_score", rows.get("score", 1.0)), errors="coerce").fillna(0.0)
    score = score.clip(lower=0.0)
    if float(score.sum()) <= 1e-12:
        score = pd.Series(1.0, index=rows.index)
    return pd.Series((score / score.sum()).to_numpy(), index=rows["symbol"].astype(str))


def _target_weights_for_window(symbol_weights: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
    if symbol_weights.empty or not dates:
        return pd.DataFrame(index=pd.DatetimeIndex(dates))
    wide = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=symbol_weights.index, dtype=float)
    for sym, weight in symbol_weights.items():
        wide[sym] = float(weight)
    return wide.fillna(0.0)


def _factor_rows(pred: pd.DataFrame, as_of: str, n: int) -> pd.DataFrame:
    date = pd.Timestamp(as_of)
    score = _score_col(pred)
    data = pred.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    day = data[data["trade_date"] == date].dropna(subset=[score]).copy()
    if day.empty:
        raise ValueError(f"no predictions for as_of={as_of}")
    out = day.sort_values(score, ascending=False).head(int(n))[["symbol", score]].copy()
    out = out.rename(columns={score: "score"})
    out["source"] = "factor"
    out["mix_score"] = np.linspace(1.0, 0.5, len(out)) if len(out) else []
    return out


def _chain_rows(chain: pd.DataFrame, n: int) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame(columns=["symbol", "score", "source", "mix_score"])
    data = chain.copy()
    if "source" in data.columns:
        llm = data[data["source"].astype(str).str.contains("LLM|产业链|chain", case=False, regex=True)].copy()
        if not llm.empty:
            data = llm
    score_col = "chain_conviction" if "chain_conviction" in data.columns else (
        "mix_score" if "mix_score" in data.columns else None
    )
    if score_col is None:
        data["score"] = 1.0
    else:
        data["score"] = pd.to_numeric(data[score_col], errors="coerce").fillna(0.0)
    out = data.sort_values("score", ascending=False).head(int(n))[["symbol", "score"]].copy()
    out["source"] = "llm_chain"
    out["mix_score"] = out["score"].clip(lower=0.0)
    return out


def _build_pool(
    pred: pd.DataFrame,
    chain: pd.DataFrame,
    *,
    as_of: str,
    n_factor: int,
    n_chain: int,
    pool_type: str,
) -> pd.DataFrame:
    fac = _factor_rows(pred, as_of, n_factor)
    ch = _chain_rows(chain, n_chain)
    if pool_type == "factor":
        return fac
    if pool_type == "chain":
        return ch
    if pool_type == "union":
        return pd.concat([fac, ch], ignore_index=True).drop_duplicates("symbol", keep="first")
    raise ValueError(f"unknown pool_type: {pool_type}")


def _run_one(
    name: str,
    pool: pd.DataFrame,
    *,
    dates: list[pd.Timestamp],
    panel: pd.DataFrame,
    sector_map: pd.DataFrame,
    weighting: str,
    slippage_bps: float,
    initial_cash: float,
    benchmark_total: float,
) -> dict[str, object]:
    weights = _normalise_weights(pool, weighting)
    tw = _target_weights_for_window(weights, dates)
    result = run_strict_backtest_v8(
        tw,
        panel,
        sector_map=sector_map if not sector_map.empty else None,
        config=AShareExecutionSimulationConfig(
            initial_cash=initial_cash,
            slippage_bps=slippage_bps,
        ),
    )
    metrics = result.metrics.to_dict()
    total_return = float(metrics["total_return"])
    return {
        "name": name,
        "pool_size": int(len(pool)),
        "symbols": pool["symbol"].astype(str).head(50).tolist() if not pool.empty else [],
        "weighting": weighting,
        "slippage_bps": float(slippage_bps),
        "total_return": round(total_return, 6),
        "benchmark_total_return": round(float(benchmark_total), 6),
        "excess_total_return": round(total_return - float(benchmark_total), 6),
        "annualized_return": round(float(metrics["annualized_return"]), 6),
        "max_drawdown": round(float(metrics["max_drawdown"]), 6),
        "sharpe": round(float(metrics["sharpe"]), 6),
        "turnover": round(float(metrics["turnover"]), 6),
        "n_trades": int(metrics["n_trades"]),
        "n_fills": int(metrics["n_fills"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--predictions", default="runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet")
    ap.add_argument("--chain-pool", type=Path, default=None)
    ap.add_argument("--market-panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--sector-map", default="runtime/data/v7/silver/sector_map/sector_map.parquet")
    ap.add_argument("--n-factor", nargs="+", type=int, default=[12, 20, 30])
    ap.add_argument("--n-chain", nargs="+", type=int, default=[6, 8, 12])
    ap.add_argument("--weighting", nargs="+", default=["equal", "score"])
    ap.add_argument("--slippage-bps", nargs="+", type=float, default=[8.0, 16.0])
    ap.add_argument("--initial-cash", type=float, default=1_000_000.0)
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/llm_validation"))
    args = ap.parse_args()

    if pd.Timestamp(args.as_of) >= pd.Timestamp(args.start_date):
        raise SystemExit("--as-of must be before --start-date for a forward validation")
    pred = _read_table(Path(args.predictions))
    chain = _read_table(args.chain_pool)
    panel_all = _read_table(Path(args.market_panel))
    sector_map = _read_table(Path(args.sector_map))
    panel = _window_panel(panel_all, args.start_date, args.end_date)
    dates = _available_dates(panel)
    if not dates:
        raise SystemExit("no market panel dates in validation window")
    _, benchmark_total = _benchmark_return(panel, dates)

    configs: list[tuple[str, pd.DataFrame, str]] = []
    for nf in args.n_factor:
        configs.append((f"factor_top{nf}", _build_pool(pred, chain, as_of=args.as_of, n_factor=nf, n_chain=0, pool_type="factor"), "equal"))
    if not chain.empty:
        for nc in args.n_chain:
            configs.append((f"chain_top{nc}", _build_pool(pred, chain, as_of=args.as_of, n_factor=0, n_chain=nc, pool_type="chain"), "equal"))
        for nf in args.n_factor:
            for nc in args.n_chain:
                pool = _build_pool(pred, chain, as_of=args.as_of, n_factor=nf, n_chain=nc, pool_type="union")
                for weighting in args.weighting:
                    configs.append((f"union_f{nf}_c{nc}_{weighting}", pool, weighting))

    rows = []
    for slip in args.slippage_bps:
        for name, pool, weighting in configs:
            if pool.empty:
                continue
            rows.append(
                _run_one(
                    name,
                    pool,
                    dates=dates,
                    panel=panel,
                    sector_map=sector_map,
                    weighting=weighting,
                    slippage_bps=slip,
                    initial_cash=args.initial_cash,
                    benchmark_total=benchmark_total,
                )
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["slippage_bps", "excess_total_return", "max_drawdown"], ascending=[True, False, True])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"llm_pit_validation_{args.as_of}_{args.start_date}_{args.end_date}".replace("-", "")
    csv_path = args.out_dir / f"{stem}.csv"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    result.to_csv(csv_path, index=False)
    summary = {
        "as_of": args.as_of,
        "validation_window": {"start": args.start_date, "end": args.end_date, "dates": len(dates)},
        "prediction_path": str(args.predictions),
        "chain_pool_path": str(args.chain_pool) if args.chain_pool else None,
        "llm_validation_status": "ok" if not chain.empty else "blocked_no_chain_pool_on_disk",
        "benchmark_total_return": round(float(benchmark_total), 6),
        "best_by_base_slippage": result[result["slippage_bps"] == min(args.slippage_bps)].head(5).to_dict("records") if not result.empty else [],
        "data_rules": {
            "as_of_before_start": True,
            "future_prices_used_only_for_scoring": True,
            "no_llm_called_by_this_script": True,
            "strict_backtest": "T+1, cost, slippage, ST/suspension/limit flags via strict_v8",
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [
        f"# LLM PIT Selection Validation - {args.as_of}",
        "",
        f"- validation window: {args.start_date} -> {args.end_date}",
        f"- benchmark total return: {benchmark_total:.4%}",
        f"- LLM validation status: {summary['llm_validation_status']}",
        f"- chain pool: {summary['chain_pool_path'] or '-'}",
        "",
        "## Best Configs",
        "",
    ]
    if result.empty:
        md.append("No valid portfolio configs produced results.")
    else:
        top = result[result["slippage_bps"] == min(args.slippage_bps)].head(12)
        md += ["| config | pool | slip | total | excess | maxDD | sharpe |", "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in top.iterrows():
            md.append(
                f"| {r['name']} | {int(r['pool_size'])} | {float(r['slippage_bps']):.0f} | "
                f"{float(r['total_return']):.2%} | {float(r['excess_total_return']):.2%} | "
                f"{float(r['max_drawdown']):.2%} | {float(r['sharpe']):.2f} |"
            )
    md += [
        "",
        "## Interpretation Guardrails",
        "",
        "- If `blocked_no_chain_pool_on_disk`, this run validates the factor baseline and harness only; it does not prove LLM edge.",
        "- To validate LLM, first generate a PIT chain pool using only information available at the as-of date, then rerun with `--chain-pool`.",
        "- Compare base and stress slippage; discard configs that only win at low friction.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "md": str(md_path), **summary}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
