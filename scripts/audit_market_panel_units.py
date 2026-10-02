#!/usr/bin/env python3
"""Read-only unit / adjustment / PIT-claim audit of a daily market panel.

For any market panel parquet, reports per ``source`` and per board:

* implied-VWAP-in-range rate (``amount / volume`` inside ``[low, high]``),
  plus the rate the panel WOULD have at x100 / x0.01 (a lots/shares defect);
* volume / amount / close ratios against the certified U0 raw panel;
* the inferred price-adjustment basis per (symbol, source): ratio to U0 raw
  stepping at U0 ex-dates => forward-adjusted as of the fetch date;
* a panel-level verdict on its ``point_in_time_valid`` stamps: qfq-as-of-fetch
  prices are REFUTED as point-in-time regardless of what the flag says.

The audited panel is never modified. Outputs a JSON report and a short
markdown summary.

Usage:
  AI_quant_venv/bin/python3 scripts/audit_market_panel_units.py \\
      --panel runtime/data/v7/silver/market_panel/market_panel.parquet \\
      --start 2021-01-01 --end 2025-08-31 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.panel_unit_audit import audit_market_panel  # noqa: E402

DEFAULT_U0 = REPO / "runtime/data/u0/panel/daily_bars_raw.parquet"
DEFAULT_EX_DATES = REPO / "runtime/data/u0/pit/adjust_factors.parquet"
PANEL_COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                 "source", "point_in_time_valid")


def _load(path: Path, columns, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    dataset = ds.dataset(str(path))
    available = set(dataset.schema.names)
    wanted = [c for c in columns if c in available]
    expr = None
    if start is not None and "trade_date" in available:
        expr = ds.field("trade_date") >= start
    if end is not None and "trade_date" in available:
        cond = ds.field("trade_date") <= end
        expr = cond if expr is None else expr & cond
    return dataset.to_table(columns=wanted, filter=expr).to_pandas()


def _markdown(report: dict, panel: Path, elapsed: float) -> str:
    claim = report["point_in_time_claim"]
    lines = [
        "# Market panel unit / adjustment audit",
        "",
        f"- 面板 panel: `{panel}`",
        f"- 窗口 window: {report['date_range'][0]} .. {report['date_range'][1]}; "
        f"rows {report['rows']:,}; symbols {report['symbols']:,}; "
        f"rows with U0 raw reference {report['rows_with_u0_reference']:,}",
        f"- PIT claim verdict: **{claim['verdict']}** — stamped point_in_time_valid "
        f"{claim['rows_stamped_point_in_time_valid']:,}; stamped but non-raw prices "
        f"{claim['rows_stamped_but_non_raw_prices']:,}; qfq-as-of-fetch rows "
        f"{claim['rows_qfq_as_of_fetch']:,}",
        f"- runtime {elapsed:.1f}s",
        "",
        "| source | rows | VWAP in range x1 | if x100 | close match vs U0 | median vol ratio | "
        "vol≈0.01 share | median amt ratio | qfq-as-of-fetch rows | PIT stamps refuted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, row in report["by_source"].items():
        def f(key, digits=4):
            value = row.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {source} | {row['rows']:,} | {f('vwap_in_range_rate_x1')} | "
            f"{f('vwap_in_range_rate_if_x100')} | {f('close_match_rate_vs_u0')} | "
            f"{f('median_volume_ratio_vs_u0')} | {f('volume_ratio_share_near_0_01')} | "
            f"{f('median_amount_ratio_vs_u0')} | {row.get('rows_qfq_as_of_fetch', 'n/a')} | "
            f"{row.get('pit_stamped_rows_refuted', 'n/a')} |"
        )
    lines += [
        "",
        "价格口径 price basis: `raw` = ratio to U0 raw ≈ 1; `qfq_*` = ratio (or difference) "
        "constant between and stepping AT U0 ex-dates, i.e. forward-adjusted as of the fetch "
        "date; `constant_scaled` = non-1 ratio with no step in window. Any non-raw basis "
        "under a point_in_time_valid=True stamp refutes that stamp.",
        "",
        "Adjustment basis, (symbol, source) groups:",
        "",
    ]
    for source, counts in report["adjustment_basis_symbol_counts_by_source"].items():
        lines.append(f"- {source}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines += ["", "Per-board detail is in the JSON report (`by_source_board`)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--u0", type=Path, default=DEFAULT_U0,
                        help="certified RAW reference panel (shares / CNY)")
    parser.add_argument("--ex-dates", type=Path, default=DEFAULT_EX_DATES,
                        help="U0 adjust-factor table (symbol, effective_date)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="market_panel_unit_audit")
    args = parser.parse_args(argv)

    started = time.time()
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    panel = _load(args.panel, PANEL_COLUMNS, start, end)
    if panel.empty:
        print(f"no rows in {args.panel} for the requested window")
        return 2
    reference = None
    if args.u0 and args.u0.exists():
        reference = _load(args.u0, ("symbol", "trade_date", "close", "volume", "amount"), start, end)
    else:
        print(f"WARNING: U0 raw reference not found at {args.u0}; adjustment basis and the "
              "PIT claim will be reported UNVERIFIABLE (pass --u0)", file=sys.stderr)
    ex_dates = (pd.read_parquet(args.ex_dates, columns=["symbol", "effective_date"])
                if args.ex_dates and args.ex_dates.exists() else None)
    report = audit_market_panel(panel, reference, ex_dates)
    elapsed = time.time() - started
    report["inputs"] = {
        "panel": str(args.panel), "u0_reference": str(args.u0) if reference is not None else None,
        "ex_dates": str(args.ex_dates) if ex_dates is not None else None,
        "window": [args.start, args.end], "elapsed_s": round(elapsed, 1),
        "read_only": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.name}.json"
    md_path = args.out_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                         encoding="utf-8")
    md_path.write_text(_markdown(report, args.panel, elapsed), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    # Exit 3 when the panel's own PIT claim is refuted, so the audit can gate.
    return 3 if report["point_in_time_claim"]["verdict"] == "REFUTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
