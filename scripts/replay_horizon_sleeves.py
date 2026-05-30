"""Re-run the horizon-sleeve executable backtest on existing OOS predictions.

Use this when you have already trained the FT model in a probe (so
`runtime/models/.../walk_forward/fold_*/fold_*_oos_predictions.parquet`
exists) and want to evaluate a NEW portfolio config without retraining.

It loads all per-horizon predictions for each fold, runs
`_compute_horizon_sleeve_backtest` per fold (per-fold supervised window)
AND on the concatenated 4-fold panel (aggregate judgement). Prints a
table so you can compare per-fold vs aggregate excess / DD / IR.

Env vars:
  QA_PROBE_DIR — model dir containing walk_forward/ (default:
    runtime/models/v7_alpha_full_universe_nosynth_probe_small_cleanlabels)
  QA_REPLAY_OUT — output dir (default: runtime/reports/sleeve_replay/)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.training.v7_experiment import (
    V7TrainingConfig,
    _compute_horizon_sleeve_backtest,
)


PROBE_DIR = Path(os.environ.get(
    "QA_PROBE_DIR",
    "runtime/models/v7_alpha_full_universe_nosynth_probe_small_cleanlabels",
))
OUTPUT_DIR = Path(os.environ.get("QA_REPLAY_OUT", "runtime/reports/sleeve_replay"))


def load_fold_predictions(fold_dir: Path) -> pd.DataFrame:
    """Concatenate all per-horizon prediction parquets in a fold dir.

    Each file has `prediction`, `horizon`, `forward_return_1d`, and the
    horizon-specific forward_return_{H}d. The horizon-sleeve backtest only
    needs `forward_return_1d` for compounding daily NAV, so we union the
    rows and drop the H-specific label columns.
    """
    frames: list[pd.DataFrame] = []
    for parquet in sorted(fold_dir.glob("fold_*d_oos_predictions.parquet")):
        df = pd.read_parquet(parquet)
        keep = ["trade_date", "symbol", "horizon", "prediction", "forward_return_1d"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(metrics: dict) -> dict:
    """Reduce the verbose backtest dict to the columns we care about."""
    return {
        "status": metrics.get("executable_backtest_status"),
        "n_days": _coerce_int(metrics.get("monthly_total_months")) * 21 if metrics.get("monthly_total_months") else "-",
        "ann_ret_%": _round(metrics.get("annualised_return_pct")),
        "bench_ann_%": _round(metrics.get("benchmark_annualised_pct")),
        "excess_ann_%": _round(metrics.get("excess_annualised_pct")),
        "vol_%": _round(metrics.get("annualised_vol_pct")),
        "sharpe": _round(metrics.get("sharpe"), 2),
        "max_DD_%": _round(metrics.get("max_drawdown_pct")),
        "DD_pass": metrics.get("max_drawdown_target_passed"),
        "IR": _round(metrics.get("information_ratio"), 2),
        "hit_vs_bench_%": _round(metrics.get("hit_vs_benchmark_pct")),
        "monthly_win": f"{metrics.get('monthly_win_months')}/{metrics.get('monthly_total_months')}",
        "avg_gross": _round(metrics.get("average_gross_exposure"), 3),
        "avg_turnover": _round(metrics.get("average_turnover"), 3),
        "oos": f"{metrics.get('oos_start')} → {metrics.get('oos_end')}",
    }


def _round(value, places: int = 2):
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return "-"


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def print_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("  (no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(f"{c:<{widths[c]}}" for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):<{widths[c]}}" for c in cols))


def main() -> None:
    walk_forward = PROBE_DIR / "walk_forward"
    fold_dirs = sorted([p for p in walk_forward.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise SystemExit(f"no fold dirs under {walk_forward}")

    universe_filter_on = bool(os.environ.get("QA_UNIVERSE_FILTER", "0") in {"1", "true", "yes"})
    base_cfg = V7TrainingConfig(
        feature_columns=("placeholder",),
        output_dir=str(OUTPUT_DIR / "_unused"),
        benchmark_path="runtime/data/v7/raw/akshare/index/equity_index.parquet",
        universe_filter_enabled=universe_filter_on,
    )
    if universe_filter_on:
        print(f"  universe filter ENABLED: st_block_rate={base_cfg.universe_st_min_block_rate}  suspended_block={base_cfg.universe_suspended_block_new}  limit_up_block={base_cfg.universe_limit_up_block_new}")
    print(f"using NEW sleeves: {base_cfg.executable_sleeves}")
    print(
        f"  base_gross={base_cfg.executable_base_gross}  "
        f"per_name_cap={base_cfg.executable_max_weight_per_name}  "
        f"turnover_cap={base_cfg.executable_max_turnover}  "
        f"vol_tgt={base_cfg.executable_vol_target_annual}"
    )
    print(
        f"  DD gates: soft={base_cfg.drawdown_soft_limit} "
        f"hard={base_cfg.drawdown_hard_limit} kill={base_cfg.drawdown_kill_limit}"
    )

    rows: list[dict[str, object]] = []
    aggregate_frames: list[pd.DataFrame] = []
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        preds = load_fold_predictions(fold_dir)
        if preds.empty:
            print(f"[{fold_name}] no predictions, skipping")
            continue
        fold_out = OUTPUT_DIR / fold_name
        fold_out.mkdir(parents=True, exist_ok=True)
        fold_cfg = replace(base_cfg, output_dir=str(fold_out))
        metrics = _compute_horizon_sleeve_backtest(preds, fold_cfg, fold_out)
        (fold_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        row = {"fold": fold_name, **summarize(metrics)}
        rows.append(row)
        aggregate_frames.append(preds)
        print(f"[{fold_name}] excess={row.get('excess_ann_%')}%  DD={row.get('max_DD_%')}%  IR={row.get('IR')}")

    print("\n=== per-fold ===")
    print_table(rows)

    if aggregate_frames:
        agg_preds = pd.concat(aggregate_frames, ignore_index=True)
        agg_out = OUTPUT_DIR / "all_folds_concat"
        agg_out.mkdir(parents=True, exist_ok=True)
        agg_cfg = replace(base_cfg, output_dir=str(agg_out))
        agg_metrics = _compute_horizon_sleeve_backtest(agg_preds, agg_cfg, agg_out)
        (agg_out / "metrics.json").write_text(json.dumps(agg_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        agg_row = {"fold": "ALL_CONCAT", **summarize(agg_metrics)}
        print("\n=== aggregate (all folds concatenated) ===")
        print_table([agg_row])
        verdict_lines: list[str] = []
        excess = float(agg_metrics.get("excess_annualised_pct", 0.0) or 0.0)
        dd = abs(float(agg_metrics.get("max_drawdown_pct", 0.0) or 0.0))
        hit = float(agg_metrics.get("hit_vs_benchmark_pct", 0.0) or 0.0)
        verdict_lines.append(f"excess > 0: {'PASS' if excess > 0 else 'FAIL'} (got {excess:+.2f}%)")
        verdict_lines.append(f"max DD <= 10%: {'PASS' if dd <= 10.0 else 'FAIL'} (got {dd:.2f}%)")
        verdict_lines.append(f"hit_vs_bench > 50%: {'PASS' if hit > 50.0 else 'FAIL'} (got {hit:.2f}%)")
        print("\n=== verdict (aggregate) ===")
        for line in verdict_lines:
            print(f"  {line}")

    summary_path = OUTPUT_DIR / "replay_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "probe_dir": str(PROBE_DIR),
                "sleeves": list(base_cfg.executable_sleeves),
                "base_gross": base_cfg.executable_base_gross,
                "per_name_cap": base_cfg.executable_max_weight_per_name,
                "per_fold": rows,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
