#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): 做T on 1-min OHLCV: no realizable edge (stage3b/4 REJECT).
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Generate a deployment-gate report for intraday Do-T EV validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.research.intraday_dot_walkforward import (
    BASELINE_NAMES,
    evaluate_walk_forward_results,
    render_markdown_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", required=True, help="CSV/parquet of conservative fill validation trades")
    ap.add_argument("--output-dir", default="runtime/reports/intraday_dot_ev")
    ap.add_argument("--min-round-trips", type=int, default=300)
    for name in BASELINE_NAMES:
        ap.add_argument(f"--{name.replace('_', '-')}", default="")
    args = ap.parse_args()

    trades = _read_frame(args.trades)
    baselines = {}
    for name in BASELINE_NAMES:
        path = getattr(args, name)
        if path:
            baselines[name] = _read_frame(path)
    report = evaluate_walk_forward_results(trades, baselines=baselines, min_round_trips=args.min_round_trips)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown_report(report, out / "intraday_dot_ev_report.md")
    payload = {
        "verdict": report.verdict,
        "reason": report.reason,
        "metrics": report.metrics,
        "baseline_comparison": report.baseline_comparison,
    }
    (out / "intraday_dot_ev_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report.confidence_buckets.empty:
        report.confidence_buckets.to_csv(out / "confidence_bucket_performance.csv", index=False)
    if not report.regime_buckets.empty:
        report.regime_buckets.to_csv(out / "regime_bucket_performance.csv", index=False)
    print(json.dumps({"verdict": report.verdict, "reason": report.reason, "report": str(md)}, ensure_ascii=False, indent=2))
    return 0


def _read_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


if __name__ == "__main__":
    raise SystemExit(main())
