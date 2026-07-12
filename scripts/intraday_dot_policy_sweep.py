#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): 做T on 1-min OHLCV: no realizable edge (stage3b/4 REJECT).
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Sweep intraday Do-T policy gates on an existing factor-combo report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.research.intraday_dot_factor_combo import (
    FactorComboConfig,
    _attach_excess_metrics,
    _evaluate_baseline_set,
    _evaluate_selection,
    _evaluate_vwap_baseline,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="Path to factor_combo_report.json")
    ap.add_argument("--output-dir", default="", help="Defaults to <report_dir>/policy_sweep")
    ap.add_argument("--signal", choices=["model", "vwap"], default="model")
    ap.add_argument("--split", default="2026-02-27")
    ap.add_argument("--validation-split", default="2026-04-15")
    ap.add_argument("--order-notional-yuan", type=float, default=2_000.0)
    ap.add_argument("--min-validation-legs", type=int, default=100)
    ap.add_argument("--min-oos-legs", type=int, default=300)
    ap.add_argument("--max-validation-eod-restore-rate", type=float, default=0.35)
    ap.add_argument("--max-validation-stop-rate", type=float, default=0.35)
    ap.add_argument("--max-oos-eod-restore-rate", type=float, default=0.20)
    ap.add_argument("--max-oos-stop-rate", type=float, default=0.35)
    ap.add_argument("--top-fracs", default="0.05,0.10,0.15,0.20,0.30,0.40,0.60,0.80,1.00")
    ap.add_argument("--eod-caps", default="0.20,0.25,0.30,0.35,0.40,0.50,0.70,1.00")
    ap.add_argument("--stop-caps", default="0.25,0.35,0.45,0.60,1.00")
    ap.add_argument("--risk-caps", default="0.60,0.75,0.90,1.00")
    ap.add_argument("--min-reversion-qualities", default="-1.0,0.0,0.10")
    ap.add_argument("--min-pred-net-bps", default="0.0")
    args = ap.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scored_path = Path(report["scored"])
    if not scored_path.is_absolute():
        scored_path = Path.cwd() / scored_path
    scored = pd.read_parquet(scored_path)
    scored["trade_date"] = pd.to_datetime(scored["trade_date"], errors="coerce").dt.normalize()

    train_end = pd.Timestamp(args.split)
    validation_end = pd.Timestamp(args.validation_split)
    validation = scored[(scored["trade_date"] > train_end) & (scored["trade_date"] <= validation_end)].copy()
    test = scored[scored["trade_date"] > validation_end].copy()
    if validation.empty or test.empty:
        raise SystemExit("validation or test split is empty")

    rows: list[dict] = []
    for min_pred in _float_list(args.min_pred_net_bps):
        cfg = FactorComboConfig(
            order_notional_yuan=args.order_notional_yuan,
            min_validation_legs=args.min_validation_legs,
            min_oos_legs=args.min_oos_legs,
            min_pred_net_bps=min_pred,
            max_validation_eod_restore_rate=args.max_validation_eod_restore_rate,
            max_validation_stop_rate=args.max_validation_stop_rate,
        )
        for frac in _float_list(args.top_fracs):
            for eod_cap in _float_list(args.eod_caps):
                for stop_cap in _float_list(args.stop_caps):
                    for risk_cap in _float_list(args.risk_caps):
                        for min_quality in _float_list(args.min_reversion_qualities):
                            val_metrics = _evaluate_split(
                                validation,
                                signal=args.signal,
                                frac=frac,
                                config=cfg,
                                eod_cap=eod_cap,
                                stop_cap=stop_cap,
                                risk_cap=risk_cap,
                                min_quality=min_quality,
                            )
                            test_metrics = _evaluate_split(
                                test,
                                signal=args.signal,
                                frac=frac,
                                config=cfg,
                                eod_cap=eod_cap,
                                stop_cap=stop_cap,
                                risk_cap=risk_cap,
                                min_quality=min_quality,
                            )
                            row = {
                                "signal": args.signal,
                                "min_pred_net_bps": min_pred,
                                "top_frac": frac,
                                "max_eod_restore_prob": eod_cap,
                                "max_stop_prob": stop_cap,
                                "max_entry_adverse_risk": risk_cap,
                                "min_entry_mean_reversion_quality": min_quality,
                            }
                            row.update(_prefix("validation", val_metrics))
                            row.update(_prefix("test", test_metrics))
                            rows.append(row)

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        raise SystemExit("policy sweep produced no rows")
    sweep["passes_validation"] = (
        (sweep["validation_n_legs"] >= args.min_validation_legs)
        & (sweep["validation_mean_net_bps"] > 0)
        & (sweep["validation_daily_uplift_bps"] > 0)
        & (sweep["validation_daily_uplift_bps_excess"] > 0)
        & (sweep["validation_eod_restore_rate"] <= args.max_validation_eod_restore_rate)
        & (sweep["validation_stop_rate"] <= args.max_validation_stop_rate)
    )
    sweep["passes_oos"] = (
        (sweep["test_n_legs"] >= args.min_oos_legs)
        & (sweep["test_mean_net_bps"] > 0)
        & (sweep["test_daily_uplift_bps"] > 0)
        & (sweep["test_daily_uplift_bps_excess"] > 0)
        & (sweep["test_eod_restore_rate"] <= args.max_oos_eod_restore_rate)
        & (sweep["test_stop_rate"] <= args.max_oos_stop_rate)
    )
    sweep["passes_both"] = sweep["passes_validation"] & sweep["passes_oos"]

    out_dir = Path(args.output_dir) if args.output_dir else report_path.parent / f"policy_sweep_{args.signal}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "policy_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    summary = {
        "signal": args.signal,
        "report": str(report_path),
        "scored": str(scored_path),
        "rows": int(len(sweep)),
        "passes_validation": int(sweep["passes_validation"].sum()),
        "passes_oos": int(sweep["passes_oos"].sum()),
        "passes_both": int(sweep["passes_both"].sum()),
        "best_validation_passing_by_test_excess": _records(
            sweep[sweep["passes_validation"]].sort_values(
                ["test_daily_uplift_bps_excess", "test_n_legs"],
                ascending=[False, False],
            )
        ),
        "best_oos_coverage_by_test_excess": _records(
            sweep[sweep["test_n_legs"] >= args.min_oos_legs].sort_values(
                ["test_daily_uplift_bps_excess", "validation_daily_uplift_bps_excess"],
                ascending=[False, False],
            )
        ),
        "best_both": _records(
            sweep[sweep["passes_both"]].sort_values(
                ["test_daily_uplift_bps_excess", "test_n_legs"],
                ascending=[False, False],
            )
        ),
    }
    summary_path = out_dir / "policy_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("signal", "rows", "passes_validation", "passes_oos", "passes_both")}, ensure_ascii=False, indent=2))
    print(json.dumps({"sweep": str(sweep_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def _evaluate_split(
    frame: pd.DataFrame,
    *,
    signal: str,
    frac: float,
    config: FactorComboConfig,
    eod_cap: float,
    stop_cap: float,
    risk_cap: float,
    min_quality: float,
) -> dict:
    if signal == "vwap":
        metrics = _evaluate_vwap_baseline(
            frame,
            frac=frac,
            config=config,
            max_eod_restore_prob=eod_cap,
            max_stop_prob=stop_cap,
            max_entry_adverse_risk=risk_cap,
            min_entry_mean_reversion_quality=min_quality,
        )
    else:
        metrics = _evaluate_selection(
            frame,
            frac=frac,
            config=config,
            name="model_policy",
            max_eod_restore_prob=eod_cap,
            max_stop_prob=stop_cap,
            max_entry_adverse_risk=risk_cap,
            min_entry_mean_reversion_quality=min_quality,
        )
    baselines = _evaluate_baseline_set(
        frame,
        frac=frac,
        config=config,
        max_eod_restore_prob=eod_cap,
        max_stop_prob=stop_cap,
        max_entry_adverse_risk=risk_cap,
        min_entry_mean_reversion_quality=min_quality,
    )
    return _attach_excess_metrics(metrics, baselines)


def _prefix(prefix: str, metrics: dict) -> dict:
    fields = (
        "n_legs",
        "days",
        "mean_net_bps",
        "daily_uplift_bps",
        "daily_uplift_bps_excess",
        "baseline_daily_uplift_bps",
        "excess_baseline_name",
        "eod_restore_rate",
        "stop_rate",
        "hit_rate",
        "avg_pred_bps",
        "avg_stop_prob",
        "pred_threshold_bps",
    )
    return {f"{prefix}_{field}": metrics.get(field, 0) for field in fields}


def _float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _records(frame: pd.DataFrame, n: int = 10) -> list[dict]:
    if frame.empty:
        return []
    return frame.head(n).replace({pd.NA: None}).to_dict(orient="records")


if __name__ == "__main__":
    raise SystemExit(main())
