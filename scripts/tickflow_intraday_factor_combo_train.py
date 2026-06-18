#!/usr/bin/env python3
"""Validate TickFlow minute bars and train an intraday Do-T factor combo model."""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path

import pandas as pd

from quantagent.research.intraday_dot_factor_combo import (
    FactorComboConfig,
    build_factor_combo_dataset,
    feature_importance_frame,
    train_factor_combo_model,
    validate_tickflow_minute_cache,
    verdict_from_metrics,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minute-dir", default="runtime/data/v7/silver/minute_bars")
    ap.add_argument("--market-panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--intraday-factors", default="runtime/data/v7/silver/intraday_factors/intraday_cicc_1min.parquet")
    ap.add_argument("--holdings-csv", default="runtime/paper/replay_2026/holdings_daily.csv")
    ap.add_argument("--output-dir", default="runtime/reports/intraday_dot_factor_combo")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--split", default="2026-02-27", help="Train end date. Validation starts on the next date.")
    ap.add_argument("--validation-split", default="2026-04-15", help="Validation end date. Test starts after this date.")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--backend", choices=["lightgbm", "xgboost", "sklearn"], default="lightgbm")
    ap.add_argument("--reuse-outcomes", action="store_true")
    ap.add_argument("--reuse-dataset", action="store_true")
    ap.add_argument("--min-train-legs", type=int, default=100)
    ap.add_argument("--dot-fraction", type=float, default=0.30)
    ap.add_argument("--commission-bps", type=float, default=2.5)
    ap.add_argument("--stamp-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--spread-bps", type=float, default=6.0)
    ap.add_argument("--transfer-bps", type=float, default=0.1)
    ap.add_argument("--order-notional-yuan", type=float, default=30000.0)
    ap.add_argument("--max-minute-participation", type=float, default=0.05)
    ap.add_argument("--min-fill-ratio", type=float, default=1.0)
    ap.add_argument("--min-validation-legs", type=int, default=100)
    ap.add_argument("--min-oos-legs", type=int, default=300)
    ap.add_argument("--min-pred-net-bps", type=float, default=0.0)
    ap.add_argument("--eod-restore-penalty-bps", type=float, default=15.0)
    ap.add_argument("--max-validation-eod-restore-rate", type=float, default=0.35)
    ap.add_argument("--max-validation-stop-rate", type=float, default=0.35)
    ap.add_argument("--no-require-book-for-enable", dest="require_book_for_enable", action="store_false")
    ap.add_argument("--all-universe-selection", dest="selection_book_only", action="store_false")
    ap.add_argument("--min-validation-book-legs", type=int, default=30)
    ap.add_argument("--min-oos-book-legs", type=int, default=100)
    ap.add_argument("--policy-eod-penalty-bps", type=float, default=80.0)
    ap.add_argument("--policy-stop-penalty-bps", type=float, default=60.0)
    ap.add_argument("--selection-eod-prob-penalty-bps", type=float, default=0.0)
    ap.add_argument("--selection-stop-prob-penalty-bps", type=float, default=0.0)
    ap.add_argument("--selection-entry-adverse-penalty-bps", type=float, default=0.0)
    ap.add_argument("--stop-prob-caps", default="", help="Comma-separated stop probability caps; defaults to config")
    ap.add_argument("--tail-exit-deadline", default="14:50:00")
    ap.add_argument("--outcome-workers", type=int, default=1)
    ap.add_argument("--outcome-cache-dir", default="")
    ap.add_argument("--disable-outcome-cache", action="store_true")
    ap.add_argument("--force-rebuild-outcome-cache", action="store_true")
    ap.add_argument("--relative-strength-cache-dir", default="")
    ap.add_argument("--disable-relative-strength-cache", action="store_true")
    ap.add_argument("--force-rebuild-feature-cache", action="store_true")
    ap.add_argument("--sector-map", default="runtime/data/v7/silver/sector_map/sector_map.parquet")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("=== Phase 0: TickFlow minute cache validation ===", flush=True)
    validation, detail = validate_tickflow_minute_cache(args.minute_dir, max_symbols=args.max_symbols)
    detail.to_csv(out / "tickflow_minute_validation_detail.csv", index=False)
    (out / "tickflow_minute_validation_summary.json").write_text(
        json.dumps(validation.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2), flush=True)
    if validation.symbols_ok == 0:
        raise SystemExit("TickFlow minute validation found no usable symbols")

    cfg = FactorComboConfig(
        start=args.start,
        end=args.end,
        split=args.split,
        validation_split=args.validation_split,
        dot_fraction=args.dot_fraction,
        commission_bps=args.commission_bps,
        stamp_bps=args.stamp_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        transfer_bps=args.transfer_bps,
        order_notional_yuan=args.order_notional_yuan,
        max_minute_participation=args.max_minute_participation,
        min_fill_ratio=args.min_fill_ratio,
        min_train_legs=args.min_train_legs,
        min_validation_legs=args.min_validation_legs,
        min_oos_legs=args.min_oos_legs,
        min_pred_net_bps=args.min_pred_net_bps,
        eod_restore_penalty_bps=args.eod_restore_penalty_bps,
        max_validation_eod_restore_rate=args.max_validation_eod_restore_rate,
        max_validation_stop_rate=args.max_validation_stop_rate,
        require_book_for_enable=args.require_book_for_enable,
        selection_book_only=args.selection_book_only,
        min_validation_book_legs=args.min_validation_book_legs,
        min_oos_book_legs=args.min_oos_book_legs,
        policy_eod_penalty_bps=args.policy_eod_penalty_bps,
        policy_stop_penalty_bps=args.policy_stop_penalty_bps,
        selection_eod_prob_penalty_bps=args.selection_eod_prob_penalty_bps,
        selection_stop_prob_penalty_bps=args.selection_stop_prob_penalty_bps,
        selection_entry_adverse_penalty_bps=args.selection_entry_adverse_penalty_bps,
        stop_prob_caps=_float_tuple(args.stop_prob_caps) or FactorComboConfig.stop_prob_caps,
        tail_exit_deadline=args.tail_exit_deadline,
        outcome_workers=args.outcome_workers,
        outcome_cache_dir="" if args.disable_outcome_cache else (args.outcome_cache_dir or str(out / "outcome_cache")),
        force_rebuild_outcome_cache=args.force_rebuild_outcome_cache,
        relative_strength_cache_dir="" if args.disable_relative_strength_cache else (
            args.relative_strength_cache_dir or str(out / "feature_cache")
        ),
        force_rebuild_feature_cache=args.force_rebuild_feature_cache,
        sector_map_path=args.sector_map,
    )
    outcomes_path = out / "dot_outcomes.parquet" if args.reuse_outcomes else None

    dataset_path = out / "factor_combo_dataset.parquet"
    contexts_path = out / "day_contexts.parquet"
    if args.reuse_dataset and dataset_path.exists():
        print("=== Phase 1: reuse factor/outcome dataset ===", flush=True)
        dataset = pd.read_parquet(dataset_path)
        required_dataset_cols = {
            "eod_restore",
            "entry_fill_status",
            "entry_price_vs_vwap_prev",
            "fee_cost_bps",
            "tail_exit_time",
            "time_exit",
            "entry_order_flow_imbalance_5m",
            "entry_mode_adverse_risk",
            "weight",
            "book_only_context",
        }
        if not required_dataset_cols.issubset(dataset.columns):
            print("cached dataset is pre-strict-fill schema; rebuilding", flush=True)
            dataset, contexts = build_factor_combo_dataset(
                minute_dir=args.minute_dir,
                market_panel_path=args.market_panel,
                intraday_factors_path=args.intraday_factors,
                holdings_csv=args.holdings_csv,
                config=cfg,
                max_symbols=args.max_symbols,
                reuse_outcomes_path=outcomes_path,
            )
            dataset.to_parquet(dataset_path, index=False)
            contexts.to_parquet(contexts_path, index=False)
        elif cfg.selection_book_only and not bool(dataset["book_only_context"].fillna(False).all()):
            print("cached dataset is not book-only; rebuilding", flush=True)
            dataset, contexts = build_factor_combo_dataset(
                minute_dir=args.minute_dir,
                market_panel_path=args.market_panel,
                intraday_factors_path=args.intraday_factors,
                holdings_csv=args.holdings_csv,
                config=cfg,
                max_symbols=args.max_symbols,
                reuse_outcomes_path=outcomes_path,
            )
            dataset.to_parquet(dataset_path, index=False)
            contexts.to_parquet(contexts_path, index=False)
    else:
        print("=== Phase 1: build factor/outcome dataset ===", flush=True)
        dataset, contexts = build_factor_combo_dataset(
            minute_dir=args.minute_dir,
            market_panel_path=args.market_panel,
            intraday_factors_path=args.intraday_factors,
            holdings_csv=args.holdings_csv,
            config=cfg,
            max_symbols=args.max_symbols,
            reuse_outcomes_path=outcomes_path,
        )
        dataset.to_parquet(dataset_path, index=False)
        contexts.to_parquet(contexts_path, index=False)
    dataset_book_coverage = _dataset_book_coverage(dataset, args.holdings_csv)
    print(json.dumps({
        "dataset": str(dataset_path),
        "rows": int(len(dataset)),
        "symbols": int(dataset["symbol"].nunique()) if not dataset.empty else 0,
        "days": int(dataset["trade_date"].nunique()) if not dataset.empty else 0,
        "book_dataset_coverage": dataset_book_coverage,
        "cost_roundtrip_bps": cfg.round_trip_cost * 10_000.0,
        "strict_fill": f"conservative_next_bar_{cfg.max_minute_participation:.1%}_volume_cap",
        "outcome_workers": cfg.outcome_workers,
        "outcome_cache_dir": cfg.outcome_cache_dir,
        "relative_strength_cache_dir": cfg.relative_strength_cache_dir,
        "train_end": cfg.split,
        "validation_end": cfg.validation_split,
        "require_book_for_enable": cfg.require_book_for_enable,
        "selection_book_only": cfg.selection_book_only,
        "min_validation_book_legs": cfg.min_validation_book_legs,
        "min_oos_book_legs": cfg.min_oos_book_legs,
    }, ensure_ascii=False, indent=2), flush=True)

    print("=== Phase 2: train factor combination model ===", flush=True)
    model, feature_cols, scored, metrics = train_factor_combo_model(dataset, config=cfg, backend=args.backend)
    scored_path = out / "factor_combo_scored.parquet"
    scored.to_parquet(scored_path, index=False)
    importance = feature_importance_frame(model, feature_cols)
    importance.to_csv(out / "factor_importance.csv", index=False)
    verdict, reason = verdict_from_metrics(metrics)
    payload = {
        "verdict": verdict,
        "reason": reason,
        "metrics": metrics,
        "dataset_book_coverage": dataset_book_coverage,
        "scored": str(scored_path),
        "factor_importance": str(out / "factor_importance.csv"),
        "validation_summary": validation.as_dict(),
    }
    payload = _json_ready(payload)
    (out / "factor_combo_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(out / "factor_combo_report.md", payload, importance)
    print(json.dumps(_console_summary(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False), flush=True)
    return 0


def _console_summary(payload: dict) -> dict:
    metrics = payload.get("metrics", {})
    return {
        "verdict": payload.get("verdict"),
        "reason": payload.get("reason"),
        "chosen_top_frac": metrics.get("chosen_top_frac"),
        "chosen_max_eod_restore_prob": metrics.get("chosen_max_eod_restore_prob"),
        "chosen_max_stop_prob": metrics.get("chosen_max_stop_prob"),
        "chosen_max_entry_adverse_risk": metrics.get("chosen_max_entry_adverse_risk"),
        "chosen_min_entry_mean_reversion_quality": metrics.get("chosen_min_entry_mean_reversion_quality"),
        "policy_selected_on": metrics.get("policy_selected_on"),
        "train": metrics.get("train", {}),
        "validation": metrics.get("validation", {}),
        "test": metrics.get("test", {}),
        "random_time_same_count_baseline": metrics.get("random_time_same_count_baseline", {}),
        "shuffled_signal_baseline": metrics.get("shuffled_signal_baseline", {}),
        "vwap_only_baseline": metrics.get("vwap_only_baseline", {}),
        "scored": payload.get("scored"),
        "factor_importance": payload.get("factor_importance"),
        "dataset_book_coverage": payload.get("dataset_book_coverage", {}),
        "validation_summary": payload.get("validation_summary", {}),
    }


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def _float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _dataset_book_coverage(dataset: pd.DataFrame, holdings_csv: str | Path | None) -> dict:
    empty = {
        "holding_symbol_days": 0,
        "dataset_book_symbol_days": 0,
        "dataset_book_symbols": 0,
        "coverage_rate": 0.0,
    }
    if dataset.empty or not holdings_csv or not Path(holdings_csv).exists():
        return empty
    holdings = pd.read_csv(holdings_csv)
    required = {"trade_date", "symbol", "weight"}
    if not required.issubset(holdings.columns):
        return empty
    h = holdings[["trade_date", "symbol", "weight"]].copy()
    h["trade_date"] = pd.to_datetime(h["trade_date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str)
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h = h[h["weight"] > 0].dropna(subset=["trade_date", "symbol"])
    ds = dataset[["trade_date", "symbol", "weight"]].copy()
    ds["trade_date"] = pd.to_datetime(ds["trade_date"], errors="coerce").dt.normalize()
    ds["symbol"] = ds["symbol"].astype(str)
    ds["weight"] = pd.to_numeric(ds["weight"], errors="coerce")
    ds = ds[ds["weight"] > 0].drop_duplicates(["trade_date", "symbol"])
    h_keys = h.drop_duplicates(["trade_date", "symbol"])
    matches = h_keys.merge(ds[["trade_date", "symbol"]], on=["trade_date", "symbol"], how="inner")
    holding_n = int(len(h_keys))
    return {
        "holding_symbol_days": holding_n,
        "dataset_book_symbol_days": int(len(matches)),
        "dataset_book_symbols": int(matches["symbol"].nunique()) if not matches.empty else 0,
        "coverage_rate": float(len(matches) / holding_n) if holding_n else 0.0,
    }


def _write_markdown(path: Path, payload: dict, importance: pd.DataFrame) -> None:
    top = importance.head(20)
    lines = [
        "# TickFlow 分时做T因子组合训练报告",
        "",
        f"## 结论\n\n{payload['verdict']} - {payload['reason']}",
        "",
        "## TickFlow 分钟数据验证",
        "",
        "```json",
        json.dumps(payload["validation_summary"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Dataset Book Coverage",
        "",
        "```json",
        json.dumps(payload.get("dataset_book_coverage", {}), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## OOS 指标",
        "",
        "```json",
        json.dumps(payload["metrics"].get("test", {}), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Validation 指标",
        "",
        "```json",
        json.dumps(payload["metrics"].get("validation", {}), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Chosen Policy",
        "",
        "```json",
        json.dumps({
            "chosen_top_frac": payload["metrics"].get("chosen_top_frac"),
            "chosen_max_eod_restore_prob": payload["metrics"].get("chosen_max_eod_restore_prob"),
            "chosen_max_stop_prob": payload["metrics"].get("chosen_max_stop_prob"),
            "chosen_max_entry_adverse_risk": payload["metrics"].get("chosen_max_entry_adverse_risk"),
            "chosen_min_entry_mean_reversion_quality": payload["metrics"].get("chosen_min_entry_mean_reversion_quality"),
            "policy_selected_on": payload["metrics"].get("policy_selected_on"),
            "require_book_for_enable": payload["metrics"].get("require_book_for_enable"),
            "selection_book_only": payload["metrics"].get("selection_book_only"),
            "min_validation_book_legs": payload["metrics"].get("min_validation_book_legs"),
            "min_oos_book_legs": payload["metrics"].get("min_oos_book_legs"),
        }, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Baselines",
        "",
        "```json",
        json.dumps({
            "random_time_same_count_baseline": payload["metrics"].get("random_time_same_count_baseline", {}),
            "shuffled_signal_baseline": payload["metrics"].get("shuffled_signal_baseline", {}),
            "vwap_only_baseline": payload["metrics"].get("vwap_only_baseline", {}),
        }, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Top Factor Importance",
        "",
        top.to_markdown(index=False) if not top.empty else "No importance available.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
