"""Rank-IC of the certified panel's price factors, inside the tradable domain.

Computing a factor is not evidence that it is worth anything. This measures the
147 usable Alpha101 / GTJA-191 columns against the panel's own forward returns.

Three choices that decide whether the number means anything:

* **Tradable domain only.** Rows are restricted to `entry_feasible`, so a factor
  cannot earn IC on names that could not have been bought. Scoring on the full
  panel inflates every result by exactly the amount that is unreachable.
* **Rank IC, per cross-section, then aggregated.** Spearman within each date,
  never pooled across dates -- a pooled correlation is dominated by the
  time-series level of the factor rather than its cross-sectional ordering.
* **ICIR reported beside IC.** A mean IC of 0.02 with a standard deviation of
  0.30 is not the same finding as 0.02 with 0.05, and only the second is
  tradeable. Both are reported; neither is reported alone.

This is a screen, not a promotion gate. It does not test costs, decay,
correlation with existing factors, or capacity, and passing it grants nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.quarantine import clean_label_mask

GOLD = Path("runtime/data/gold/full_universe")
LABEL = "forward_return_5d"


def _rank_ic(frame: pd.DataFrame, factor: str, label: str) -> tuple[float, float, int]:
    """Mean and std of per-date Spearman correlation, plus the date count."""
    work = frame[["trade_date", factor, label]].dropna()
    if work.empty:
        return float("nan"), float("nan"), 0
    per_date = work.groupby("trade_date").apply(
        lambda g: g[factor].corr(g[label], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    )
    per_date = per_date.dropna()
    if per_date.empty:
        return float("nan"), float("nan"), 0
    return float(per_date.mean()), float(per_date.std(ddof=1)), int(len(per_date))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=["alpha101", "gtja191"])
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--start", default=None, help="restrict to trade_date >= this")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="labelled dataset (default: <gold>/dataset.parquet)")
    ap.add_argument("--factors-dir", type=Path, default=None,
                    help="directory holding factors_<family>.parquet (default: gold)")
    ap.add_argument("--summary-csv", type=Path, default=None,
                    help="also write a factor_summary.csv the Factor Lab adapter indexes")
    args = ap.parse_args()

    dataset_path = args.dataset or (GOLD / "dataset.parquet")
    artifact = (args.factors_dir or GOLD) / f"factors_{args.family}.parquet"
    if not artifact.exists():
        print(f"BLOCKED_BY_DATA: {artifact} not built", file=sys.stderr)
        return 2

    base = pd.read_parquet(
        dataset_path,
        columns=["symbol", "trade_date", args.label, "entry_feasible"],
    )
    if args.start:
        base = base[base["trade_date"] >= pd.Timestamp(args.start)]
    # A screen that ranks factors is a selection step: no label window may
    # touch a quarantined holdout (configs/quarantined_windows.json).
    horizon = re.fullmatch(r"forward_return_(\d+)d", args.label)
    if horizon is None:
        print(f"cannot infer the label horizon from {args.label!r}", file=sys.stderr)
        return 2
    clean = clean_label_mask(
        base["trade_date"], horizon_sessions=int(horizon.group(1)), symbols=base["symbol"]
    )
    quarantine_rows_dropped = int((~clean).sum())
    base = base[clean]
    tradable = base["entry_feasible"].astype(bool)
    print(
        f"[domain] rows={len(base):,} tradable={int(tradable.sum()):,} "
        f"({tradable.mean():.4f})",
        flush=True,
    )
    base = base[tradable].reset_index(drop=True)

    import pyarrow.parquet as pq

    names = [
        c for c in pq.ParquetFile(artifact).schema_arrow.names
        if c not in {"symbol", "trade_date"}
    ]
    results = []
    for i, factor in enumerate(names, 1):
        piece = pd.read_parquet(artifact, columns=["symbol", "trade_date", factor])
        merged = base.merge(piece, on=["symbol", "trade_date"], how="inner")
        del piece
        ic, sd, n = _rank_ic(merged, factor, args.label)
        del merged
        icir = ic / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        results.append(
            {"factor": factor, "rank_ic": ic, "ic_std": sd, "icir": icir, "n_dates": n}
        )
        if i % 10 == 0 or i == len(names):
            print(f"[{args.family}] {i}/{len(names)}", flush=True)

    frame = pd.DataFrame(results)
    frame["abs_ic"] = frame["rank_ic"].abs()
    frame = frame.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    out = args.output or (GOLD / f"factor_ic_{args.family}.json")
    payload = {
        "family": args.family,
        "label": args.label,
        "domain": "entry_feasible only",
        "start": args.start,
        "quarantine_rows_dropped": quarantine_rows_dropped,
        "factors_scored": int(frame["rank_ic"].notna().sum()),
        "factors_all_nan": int(frame["rank_ic"].isna().sum()),
        "median_abs_ic": round(float(frame["abs_ic"].median(skipna=True)), 6),
        "count_abs_ic_above_0.02": int((frame["abs_ic"] > 0.02).sum()),
        "count_abs_icir_above_0.30": int((frame["icir"].abs() > 0.30).sum()),
        "top": frame.head(15).round(6).to_dict(orient="records"),
    }
    payload["dataset"] = str(dataset_path)
    payload["factors"] = frame.round(6).to_dict(orient="records")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame({
            "factor_name": frame["factor"],
            "rank_ic": frame["rank_ic"],
            "rank_icir": frame["icir"],
            "ic_std": frame["ic_std"],
            "n_dates": frame["n_dates"],
            "label": args.label,
            "domain": "entry_feasible, quarantine-clean label windows",
            "dataset": str(dataset_path),
        })
        summary.to_csv(args.summary_csv, index=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "top"}, ensure_ascii=False))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
