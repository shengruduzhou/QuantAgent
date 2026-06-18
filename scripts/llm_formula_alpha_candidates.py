#!/usr/bin/env python3
"""Generate LLM-proposed formula alpha candidates and validate them.

The LLM is a *researcher*, never a trader: it proposes formulas in the
restricted factor DSL; deterministic code parses, evaluates on PIT data,
and gates by chronological validation RankIC, finite ratio, correlation
diversity and correlation to the existing factor library. Output format
matches ``factor_synthesis.save_definitions`` so survivors flow into
``materialize-alpha181-v7 --synthesized-factors``.

Hard-won operational fixes baked in:
- Thinking models (gemma-4) need 300s+ timeouts; quota throttling causes
  read timeouts → retry with backoff and fall back through fast models.
- Prompt stays small: only the DSL-usable columns are advertised, not the
  whole 200-column training frame.
- A persistent rejected-formula memory is fed back into every round so
  the LLM stops regenerating known failure patterns.
- The sample never crosses --train-end, keeping later dates clean OOS.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.factors import expr as E
from quantagent.factors.factor_loop_memory import (  # single source of truth (shared with the rd-agent loop)
    ALLOWED_NODES,
    A_SHARE_STRUCTURES,
    FALLBACK_MODELS,
    append_memory as _append_memory,
    load_memory as _load_memory,
    memory_digest as _memory_digest,
)
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
        # Contiguous block: rolling operators over gapped dates are wrong.
        start = int(rng.integers(0, len(dates) - sample_dates + 1))
        chosen = set(dates[start : start + sample_dates])
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


def _prompt(args: argparse.Namespace, memory: list[dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are a quantitative researcher generating A-share formula-alpha candidates. "
        "Return exactly one JSON object. Never emit orders or financial advice. "
        "Only use the safe Python repr-style DSL shown by the user."
    )
    user = {
        "goal": (
            "Propose cross-sectional stock-ranking formulas for China A-shares with stable "
            "5-day forward rank-IC out of sample. Each formula must encode one clear "
            "economic hypothesis."
        ),
        "required_output_schema": {
            "candidates": [
                {
                    "name": "short_snake_case_name",
                    "expression": "Rank(Returns(Column('close'), 5))",
                    "hypothesis": "one sentence of economic logic",
                    "horizon": "short_5d|mid_5d_30d|long_30d_120d",
                    "expected_direction": "positive|negative",
                }
            ]
        },
        "allowed_expression_nodes": ALLOWED_NODES,
        "prefer_structures": A_SHARE_STRUCTURES,
        "constraints": [
            "Use only information available at or before trade_date (no future data, no labels).",
            "Windows must be <= 120 trading days; keep formulas parseable and under ~12 nodes.",
            "Every formula must have a clear economic meaning; no random operator soup.",
            "Avoid formulas equivalent to plain size/volatility/turnover ranks already in the library.",
            f"Return exactly {args.n_candidates} candidates.",
        ],
        "feedback_from_previous_rounds": _memory_digest(memory),
    }
    return system, json.dumps(user, ensure_ascii=False)


def _candidate_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("candidates", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _invoke_with_retry(args: argparse.Namespace, system: str, user_text: str, out_dir: Path, round_idx: int):
    """Call the LLM with backoff, cycling to fast models on repeated failure."""
    base_cfg = LLMSkillConfig.from_env()
    base_cfg = replace(
        base_cfg,
        allow_network=bool(args.allow_network),
        timeout_seconds=max(base_cfg.timeout_seconds, float(args.timeout_seconds)),
    )
    models = [base_cfg.model, *[m for m in FALLBACK_MODELS if m != base_cfg.model]]
    last = None
    for attempt, model in enumerate(models[: args.max_attempts]):
        cfg = replace(base_cfg, model=model)
        result = LLMSkillClient(cfg).invoke(
            "formula_alpha_designer",
            system_prompt=system,
            user_text=user_text,
            fallback={"candidates": []},
        )
        (out_dir / f"llm_raw_result_round{round_idx}_attempt{attempt}.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "used_fallback": result.used_fallback,
                    "fallback_reason": result.fallback_reason,
                    "output": result.output,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        last = result
        if not result.used_fallback and _candidate_items(result.output):
            return result
        time.sleep(args.retry_backoff_seconds * (attempt + 1))
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market-panel", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output-dir", default="runtime/reports/v8/llm_formula_alpha")
    ap.add_argument("--label-column", default="forward_return_5d")
    ap.add_argument("--train-end", default="2024-07-31", help="Candidates only ever see dates <= this.")
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2, help="Generation rounds with rejected-memory feedback in between.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sample-dates", type=int, default=350, help="0 disables date subsampling")
    ap.add_argument("--sample-symbols", type=int, default=600, help="0 means full universe")
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--min-validation-rank-ic", type=float, default=0.01)
    ap.add_argument("--max-correlation", type=float, default=0.7)
    ap.add_argument("--reference-columns", default="alpha016,alpha015,alpha050,alpha044,alpha040,alpha161,alpha163,alpha088,alpha145")
    ap.add_argument("--max-reference-correlation", type=float, default=0.6)
    ap.add_argument("--memory-path", default="runtime/reports/v8/llm_formula_alpha/memory.jsonl")
    ap.add_argument("--timeout-seconds", type=float, default=360.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    references = [c.strip() for c in args.reference_columns.split(",") if c.strip()]
    market = pd.read_parquet(
        args.market_panel,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"],
    )
    labels = pd.read_parquet(
        args.labels,
        columns=["symbol", "trade_date", args.label_column, *references],
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
    train_end = pd.Timestamp(args.train_end)
    market = market[market["trade_date"] <= train_end]
    labels = labels[labels["trade_date"] <= train_end]
    merged = market.merge(labels, on=["symbol", "trade_date"], how="inner")
    sample = _sample_panel(
        merged,
        label_column=args.label_column,
        sample_dates=args.sample_dates,
        sample_symbols=args.sample_symbols,
        seed=args.seed,
    )
    train, valid = _chronological_split(sample, args.validation_fraction)
    reference_values = {
        ref: pd.to_numeric(valid[ref], errors="coerce") for ref in references if ref in valid.columns
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = Path(args.memory_path)

    rows: list[dict[str, Any]] = []
    definitions: list[E.FactorDefinition] = []
    chosen_values: list[pd.Series] = []
    seen_exprs: set[str] = set()
    any_llm_success = False

    for round_idx in range(args.rounds):
        memory = _load_memory(memory_path) + rows
        system, user_text = _prompt(args, memory)
        result = _invoke_with_retry(args, system, user_text, out_dir, round_idx)
        if result is None or (result.used_fallback and not args.allow_fallback and not any_llm_success):
            if round_idx == 0 and (result is None or result.used_fallback) and not args.allow_fallback:
                raise SystemExit(f"LLM call failed: {getattr(result, 'fallback_reason', 'no result')}")
            break
        if result.used_fallback:
            break
        any_llm_success = True

        round_rows: list[dict[str, Any]] = []
        for item in _candidate_items(result.output):
            raw_name = str(item.get("name") or f"llm_formula_{len(rows) + len(round_rows) + 1:03d}")
            name = "llm_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name.lower()).strip("_")
            expr_text = str(item.get("expression") or "")
            row: dict[str, Any] = {
                "round": round_idx,
                "name": name,
                "raw_expression": expr_text,
                "hypothesis": str(item.get("hypothesis") or ""),
                "horizon": str(item.get("horizon") or ""),
            }
            try:
                expr = parse_expression(expr_text)
            except Exception as exc:  # noqa: BLE001
                round_rows.append({**row, "expression": expr_text, "status": "parse_failed", "error": str(exc)})
                continue
            if repr(expr) in seen_exprs:
                round_rows.append({**row, "expression": repr(expr), "status": "duplicate"})
                continue
            seen_exprs.add(repr(expr))
            train_ic, train_finite = _evaluate_ic(expr, train, train[args.label_column], train["trade_date"])
            oriented = expr if train_ic >= 0 else E.Mul(E.Constant(-1.0), expr)
            valid_ic, valid_finite = _evaluate_ic(oriented, valid, valid[args.label_column], valid["trade_date"])
            row.update(
                expression=repr(oriented),
                train_rank_ic=float(train_ic),
                validation_rank_ic=float(valid_ic),
                train_finite_ratio=float(train_finite),
                validation_finite_ratio=float(valid_finite),
                complexity=int(_node_count(oriented)),
            )
            if valid_finite < 0.30:
                round_rows.append({**row, "status": "low_finite_ratio"})
                continue
            if valid_ic < args.min_validation_rank_ic:
                round_rows.append({**row, "status": "low_validation_ic"})
                continue
            values = pd.to_numeric(oriented.evaluate(valid), errors="coerce").replace([np.inf, -np.inf], np.nan)
            corr_values = [
                abs(values.corr(prev, method="spearman"))
                for prev in chosen_values
                if prev is not None and not prev.empty
            ]
            corr_max = float(max(corr_values)) if corr_values else 0.0
            if corr_max > args.max_correlation:
                round_rows.append({**row, "status": "high_corr_to_selected", "max_selected_corr": corr_max})
                continue
            ref_corr = 0.0
            for ref_series in reference_values.values():
                corr = abs(values.corr(ref_series, method="spearman"))
                if np.isfinite(corr):
                    ref_corr = max(ref_corr, float(corr))
            if reference_values and ref_corr > args.max_reference_correlation:
                round_rows.append({**row, "status": "high_corr_to_library", "max_reference_corr": ref_corr})
                continue
            definitions.append(
                E.FactorDefinition(
                    name=name,
                    expr=oriented,
                    description=str(item.get("hypothesis") or "LLM-proposed formula alpha"),
                )
            )
            chosen_values.append(values)
            round_rows.append({**row, "status": "selected", "max_selected_corr": corr_max, "max_reference_corr": ref_corr})
            if len(definitions) >= args.top_k:
                break
        rows.extend(round_rows)
        _append_memory(memory_path, round_rows)
        if len(definitions) >= args.top_k:
            break

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_parquet(out_dir / "llm_formula_leaderboard.parquet", index=False)
    definitions_path = save_definitions(definitions, out_dir / "synthesized_definitions.json")
    summary = {
        "status": "passed" if any_llm_success else "llm_unavailable",
        "rounds_completed": int(leaderboard["round"].nunique()) if not leaderboard.empty else 0,
        "selected": len(definitions),
        "rejected": int((leaderboard["status"] != "selected").sum()) if not leaderboard.empty else 0,
        "leaderboard": str(out_dir / "llm_formula_leaderboard.parquet"),
        "definitions": str(definitions_path),
        "memory": str(memory_path),
        "train_end": args.train_end,
        "sample_rows": int(len(sample)),
        "sample_symbols": int(sample["symbol"].nunique()),
        "sample_dates": int(sample["trade_date"].nunique()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
