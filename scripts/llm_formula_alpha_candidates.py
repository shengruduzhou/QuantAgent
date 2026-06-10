#!/usr/bin/env python3
"""Generate LLM-proposed formula alpha candidates and validate them.

The output format matches ``factor_synthesis.save_definitions`` so the
survivors can be passed to ``materialize-alpha181-v7 --synthesized-factors``.
LLM output is treated as untrusted text: every expression is parsed through the
restricted factor DSL namespace, evaluated on PIT data, and filtered by
chronological OOS RankIC plus correlation diversity before it is saved.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.factors import expr as E
from quantagent.factors.factor_synthesis import (
    _evaluate_ic,
    _node_count,
    parse_expression,
    save_definitions,
)


def _sample_panel(
    frame: pd.DataFrame,
    *,
    label_column: str,
    sample_dates: int,
    sample_symbols: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = frame.dropna(subset=[label_column]).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"])
    dates = np.array(sorted(data["trade_date"].unique()))
    if sample_dates and len(dates) > sample_dates:
        chosen = set(rng.choice(dates, size=sample_dates, replace=False))
        data = data[data["trade_date"].isin(chosen)]
    symbols = np.array(sorted(data["symbol"].astype(str).unique()))
    if sample_symbols and len(symbols) > sample_symbols:
        chosen_symbols = set(rng.choice(symbols, size=sample_symbols, replace=False))
        data = data[data["symbol"].astype(str).isin(chosen_symbols)]
    return data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _chronological_split(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique())
    if len(dates) < 4:
        raise ValueError("need at least four dates for train/OOS validation")
    split_at = max(1, int(len(dates) * (1.0 - validation_fraction)))
    split_date = dates[min(split_at, len(dates) - 1)]
    train = frame[frame["trade_date"] < split_date].reset_index(drop=True)
    valid = frame[frame["trade_date"] >= split_date].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError("empty train or validation split")
    return train, valid


def _prompt(columns: list[str], args: argparse.Namespace) -> tuple[str, str]:
    system = (
        "You generate quantitative A-share formula-alpha candidates. "
        "Return exactly one JSON object. Never emit orders or financial advice. "
        "Only use the safe Python repr-style DSL shown by the user."
    )
    user = {
        "goal": "Find formula alpha candidates likely to improve strict OOS excess return after full-universe retraining.",
        "required_output_schema": {
            "candidates": [
                {
                    "name": "short_snake_case_name",
                    "expression": "Rank(Returns(Column('close'), 5))",
                    "hypothesis": "one sentence",
                    "horizon": "short_5d|mid_5d_30d|long_30d_120d",
                }
            ]
        },
        "allowed_expression_nodes": [
            "Column('open'|'high'|'low'|'close'|'volume'|'amount')",
            "OptionalColumn('turnover_rate'|'pe_ttm'|'pb'|'roe'|'gross_margin')",
            "Constant(float)",
            "Add(left,right)", "Sub(left,right)", "Mul(left,right)", "Div(numerator,denominator)",
            "Abs(expr)", "Sign(expr)", "Log(expr)", "Rank(expr)", "Returns(expr, periods)",
            "Delay(expr, periods)", "Delta(expr, periods)", "TsRank(expr, window)",
            "_RollingReduction(expr, window, 'mean'|'std'|'sum'|'max'|'min')",
        ],
        "constraints": [
            "Use only information available at or before trade_date.",
            "Prefer volume-price, liquidity, reversal, momentum decay, and crowding/risk-control formulas.",
            "Do not use future returns, labels, raw_hash, source text, or any order/trade instruction.",
            f"Return {args.n_candidates} candidates.",
        ],
        "available_columns": columns[:200],
    }
    return system, json.dumps(user, ensure_ascii=False)


def _candidate_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("candidates", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market-panel", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output-dir", default="runtime/reports/v8/llm_formula_alpha")
    ap.add_argument("--label-column", default="forward_return_5d")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sample-dates", type=int, default=500, help="0 disables date subsampling")
    ap.add_argument("--sample-symbols", type=int, default=0, help="0 means full universe")
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--min-validation-rank-ic", type=float, default=0.0)
    ap.add_argument("--max-correlation", type=float, default=0.85)
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    market = pd.read_parquet(args.market_panel)
    labels = pd.read_parquet(args.labels, columns=["symbol", "trade_date", args.label_column])
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
    merged = market.merge(labels, on=["symbol", "trade_date"], how="inner")
    sample = _sample_panel(
        merged,
        label_column=args.label_column,
        sample_dates=args.sample_dates,
        sample_symbols=args.sample_symbols,
        seed=args.seed,
    )
    train, valid = _chronological_split(sample, args.validation_fraction)

    cfg = LLMSkillConfig.from_env()
    cfg = replace(cfg, allow_network=bool(args.allow_network), timeout_seconds=max(cfg.timeout_seconds, 180.0))
    system, user_text = _prompt(list(sample.columns), args)
    result = LLMSkillClient(cfg).invoke(
        "formula_alpha_designer",
        system_prompt=system,
        user_text=user_text,
        fallback={"candidates": []},
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_raw_result.json").write_text(
        json.dumps(
            {
                "used_fallback": result.used_fallback,
                "fallback_reason": result.fallback_reason,
                "output": result.output,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.used_fallback and not args.allow_fallback:
        raise SystemExit(f"LLM call failed: {result.fallback_reason}")

    rows: list[dict[str, Any]] = []
    definitions: list[E.FactorDefinition] = []
    chosen_values: list[pd.Series] = []
    for item in _candidate_items(result.output):
        raw_name = str(item.get("name") or f"llm_formula_{len(rows)+1:03d}")
        name = "llm_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name.lower()).strip("_")
        expr_text = str(item.get("expression") or "")
        try:
            expr = parse_expression(expr_text)
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "expression": expr_text, "status": "parse_failed", "error": str(exc)})
            continue
        train_ic, train_finite = _evaluate_ic(expr, train, train[args.label_column], train["trade_date"])
        oriented = expr if train_ic >= 0 else E.Mul(E.Constant(-1.0), expr)
        valid_ic, valid_finite = _evaluate_ic(oriented, valid, valid[args.label_column], valid["trade_date"])
        status = "rejected"
        corr_max = 0.0
        values = pd.Series(dtype=float)
        if valid_finite >= 0.30 and valid_ic >= args.min_validation_rank_ic:
            values = pd.to_numeric(oriented.evaluate(valid), errors="coerce").replace([np.inf, -np.inf], np.nan)
            corr_values = [
                abs(values.corr(prev, method="spearman"))
                for prev in chosen_values
                if prev is not None and not prev.empty
            ]
            corr_max = float(max(corr_values)) if corr_values else 0.0
            if corr_max <= args.max_correlation:
                status = "selected"
                definitions.append(
                    E.FactorDefinition(
                        name=name,
                        expr=oriented,
                        description=str(item.get("hypothesis") or "LLM-proposed formula alpha"),
                    )
                )
                chosen_values.append(values)
        rows.append(
            {
                "name": name,
                "expression": repr(oriented),
                "raw_expression": expr_text,
                "status": status,
                "train_rank_ic": float(train_ic),
                "validation_rank_ic": float(valid_ic),
                "train_finite_ratio": float(train_finite),
                "validation_finite_ratio": float(valid_finite),
                "max_selected_corr": corr_max,
                "complexity": int(_node_count(oriented)),
                "hypothesis": str(item.get("hypothesis") or ""),
                "horizon": str(item.get("horizon") or ""),
            }
        )
        if len(definitions) >= args.top_k:
            break

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_parquet(out_dir / "llm_formula_leaderboard.parquet", index=False)
    definitions_path = save_definitions(definitions, out_dir / "synthesized_definitions.json")
    summary = {
        "status": "passed",
        "used_fallback": result.used_fallback,
        "fallback_reason": result.fallback_reason,
        "selected": len(definitions),
        "leaderboard": str(out_dir / "llm_formula_leaderboard.parquet"),
        "definitions": str(definitions_path),
        "sample_rows": int(len(sample)),
        "sample_symbols": int(sample["symbol"].nunique()),
        "sample_dates": int(sample["trade_date"].nunique()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
