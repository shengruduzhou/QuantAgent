#!/usr/bin/env python3
"""Two questions the research stack could not previously answer, answered together.

**Q1 — is the nonlinearity real?** Runs
:func:`quantagent.research.model_comparison.run_model_comparison`, which fits a
rank-weighted additive baseline, explicit ``x_i x_j`` and ``x_j ⊗ regime``
models, a LightGBM, and an OOF stack on *identical* purged walk-forward folds,
then asks whether any of them beats the baseline by a margin that is both
statistically significant and economically material. The trailing folds are
scored but excluded from the choice.

**Q2 — is the alpha new?** Builds A-share MKT/SMB/HML/UMD from the same panel
and runs the CAPM → FF3 → Carhart ladder on the champion's book, so a return
that is really a small-cap or value tilt is named as one.

The two belong in one script because they fail together. A model that beats its
baseline on IC can still be selling nothing but size exposure, and a model with
genuine Carhart alpha built on a broken baseline comparison has not been shown to
need its complexity. Neither result means much alone.

Both stages emit a verdict from the same five-way vocabulary the rest of the
research stack uses — ``production_accepted`` / ``hypothesis_rejected`` /
``pipeline_failed`` / ``data_invalid`` / ``model_invalid`` — so a refusal is
never confused with a crash.

Usage::

    python scripts/audit_nonlinearity_and_style_alpha.py \
        --dataset runtime/data/v7/gold/training_dataset/<panel>.parquet \
        --start 2021-01-01 --output runtime/reports/nonlinearity_audit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest.factor_model_attribution import (  # noqa: E402
    attribute_strategy_returns,
    build_ashare_style_factors,
)
from quantagent.research.model_comparison import (  # noqa: E402
    ComparisonConfig,
    run_model_comparison,
    save_comparison_report,
)
from quantagent.risk.regime_family import compute_regime_family  # noqa: E402

DEFAULT_DATASET = (
    "runtime/data/v7/gold/training_dataset/"
    "training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"
)
SECURITY_MASTER = "runtime/data/u0/security_master.parquet"

#: A fixed, pre-declared factor list spanning the families the repository
#: actually trades. It is written down here rather than inferred from the panel
#: so that re-running the audit cannot quietly widen the search.
DEFAULT_FACTORS = (
    "momentum_5d", "momentum_20d", "return_1d",
    "volatility_20d", "amount_mean_20d", "volume_mean_20d", "intraday_return",
    "pb", "pe_ttm", "earnings_yield", "book_yield", "valuation_percentile",
    "roe", "net_margin", "gross_margin", "debt_to_asset",
    "revenue_yoy", "net_income_yoy", "quality_composite", "growth_composite",
    "alpha102", "alpha103", "alpha105", "alpha110", "alpha115", "alpha120",
    "gtja001", "gtja006",
)

IDENTITY = ("trade_date", "symbol", "close", "return_1d")
FLAGS = ("is_st", "is_suspended", "is_limit_up", "is_limit_down")


def load_panel(path: Path, start: str, label: str, factors: tuple[str, ...]) -> pd.DataFrame:
    """Read only the columns and dates the audit needs, row group by row group."""
    handle = pq.ParquetFile(path)
    available = set(handle.schema_arrow.names)
    wanted = [
        column
        for column in (*IDENTITY, label, "book_yield", *FLAGS, *factors)
        if column in available
    ]
    wanted = list(dict.fromkeys(wanted))
    missing = [column for column in factors if column not in available]
    if missing:
        print(f"[warn] factor columns absent from the panel: {missing}")
    parts: list[pd.DataFrame] = []
    for index in range(handle.metadata.num_row_groups):
        frame = handle.read_row_group(index, columns=wanted).to_pandas()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame[frame["trade_date"] >= pd.Timestamp(start)]
        if len(frame):
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=wanted)
    return pd.concat(parts, ignore_index=True)


def load_share_counts(path: Path) -> tuple[dict[str, float], str]:
    """Share counts from the security master, with their PIT status declared.

    The master holds one row per symbol, so its ``float_shares`` is a *current*
    snapshot. Multiplying it by a 2021 close does not give the 2021 market cap,
    and the attribution layer is told so rather than left to assume otherwise.
    """
    if not path.exists():
        return {}, "absent"
    master = pq.read_table(path, columns=["symbol", "float_shares"]).to_pandas()
    counts = dict(
        zip(
            master["symbol"].astype(str),
            pd.to_numeric(master["float_shares"], errors="coerce"),
        )
    )
    return {k: v for k, v in counts.items() if np.isfinite(v) and v > 0}, "current_snapshot"


def champion_book_returns(
    panel: pd.DataFrame,
    report,
    config: ComparisonConfig,
) -> pd.Series:
    """Daily returns of the champion arm's long-only top-K book.

    Rebuilt from the arm's own out-of-sample selection-fold predictions so the
    attribution is run on the same book the comparison scored, not on a fresh
    backtest with different conventions.
    """
    arm = next((item for item in report.arms if item.name == report.champion), None)
    if arm is None or arm.daily_returns.empty:
        return pd.Series(dtype=float)
    # ``daily_returns`` are already daily-equivalent net top-K returns.
    return arm.daily_returns.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--security-master", default=SECURITY_MASTER)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--label", default="forward_return_5d")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--holdout-folds", type=int, default=2)
    parser.add_argument("--valid-size-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=400)
    parser.add_argument("--max-interaction-pairs", type=int, default=12)
    parser.add_argument(
        "--factors",
        default="",
        help="comma-separated override for the pre-declared factor list",
    )
    parser.add_argument("--output", default="runtime/reports/nonlinearity_audit")
    args = parser.parse_args()

    factors = (
        tuple(name.strip() for name in args.factors.split(",") if name.strip())
        or DEFAULT_FACTORS
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset} from {args.start}")
    panel = load_panel(Path(args.dataset), args.start, args.label, factors)
    if panel.empty:
        print("[data_invalid] panel is empty for the requested window")
        return 4
    print(
        f"[load] rows={len(panel):,} dates={panel['trade_date'].nunique()} "
        f"symbols={panel['symbol'].nunique()}"
    )

    regime = compute_regime_family(panel[["trade_date", "symbol", "close"]])
    print(f"[regime] {regime.value_counts().to_dict()}")

    config = ComparisonConfig(
        label_column=args.label,
        horizon_days=args.horizon_days,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        n_folds=args.n_folds,
        holdout_folds=args.holdout_folds,
        valid_size_days=args.valid_size_days,
        min_train_days=args.min_train_days,
        max_interaction_pairs=args.max_interaction_pairs,
    )
    usable = [name for name in factors if name in panel.columns]
    print(f"[compare] {len(usable)} factors, {config.n_folds} folds "
          f"({config.holdout_folds} held out)")
    report = run_model_comparison(
        panel,
        usable,
        config=config,
        regime_by_date=regime,
        progress=lambda stage, done, total: print(f"  [{done}/{total}] {stage}", flush=True),
    )
    save_comparison_report(report, output)

    print(f"\n=== Q1 nonlinearity: {report.verdict} | champion={report.champion} ===")
    header = f"{'arm':30s} {'class':26s} {'selIC':>8s} {'ICIR':>7s} {'net':>8s} {'holdIC':>8s}"
    print(header)
    for arm in report.arms:
        selection, holdout = arm.selection_metrics, arm.holdout_metrics
        print(
            f"{arm.name:30s} {arm.model_class:26s} "
            f"{selection.get('rank_ic_mean', float('nan')):+8.4f} "
            f"{selection.get('rank_ic_ir', float('nan')):+7.2f} "
            f"{selection.get('net_annual_return', float('nan')):+8.3f} "
            f"{holdout.get('rank_ic_mean', float('nan')):+8.4f}"
            + (f"  ERROR {arm.error[:60]}" if arm.error else "")
        )
    print()
    for test in report.incremental:
        print(
            f"  vs baseline · {test.arm:28s} dIC={test.ic_delta:+.4f} "
            f"t={test.ic_delta_t_stat:+6.2f} dNet={test.net_return_delta:+.4f} "
            f"pass={test.passes}"
        )
        for reason in test.reasons:
            print(f"      - {reason}")
    for reason in report.verdict_reasons:
        print(f"  [verdict] {reason}")

    # ---- Q2: is whatever survived actually new? -------------------------
    shares, share_status = load_share_counts(Path(args.security_master))
    style = build_ashare_style_factors(
        panel,
        return_column="return_1d",
        shares_outstanding=shares or None,
        share_count_status=share_status,
    )
    book = champion_book_returns(panel, report, config)
    attribution_payload: dict[str, object]
    if book.empty:
        attribution_payload = {
            "status": "unavailable",
            "reason": "champion produced no out-of-sample book returns",
        }
        print("\n=== Q2 style attribution: unavailable (no champion book) ===")
    else:
        attribution = attribute_strategy_returns(book, style)
        attribution_payload = attribution.as_dict()
        print(
            f"\n=== Q2 style attribution of '{report.champion}' "
            f"(strictest measured: {attribution.strictest_measured}) ==="
        )
        for name, level in attribution.levels.items():
            if level.status != "measured":
                print(f"  {name:10s} UNAVAILABLE (missing {', '.join(level.missing_factors)})")
                continue
            loadings = " ".join(f"{k}={v:+.2f}" for k, v in level.loadings.items())
            print(
                f"  {name:10s} alpha={level.alpha_annual:+7.2%} "
                f"t={level.alpha_t_stat:+5.2f} R2={level.r_squared:.3f}  {loadings}"
            )
        print(f"  survives style controls: {attribution.survives_style_controls}")
        for factor, status in style.status.items():
            if status != "constructed":
                print(f"  [{status}] {factor}: {style.notes.get(factor, '')}")

    (output / "style_attribution.json").write_text(
        json.dumps(attribution_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    style.returns.to_csv(output / "style_factors.csv", index_label="trade_date")
    print(f"\n[write] {output}")

    # Exit codes mirror quantagent.research.verdict so a job runner can tell a
    # refusal from a crash without parsing stdout.
    return {
        "production_accepted": 0,
        "hypothesis_rejected": 3,
        "data_invalid": 4,
        "model_invalid": 1,
        "pipeline_failed": 1,
    }.get(report.verdict, 1)


if __name__ == "__main__":
    raise SystemExit(main())
