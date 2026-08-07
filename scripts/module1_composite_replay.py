#!/usr/bin/env python3
"""Run Module One's composite replay and write the difference table.

The gate this produces evidence for: one economic scenario driven through the
fast backtest, the paper broker and the OMS-to-paper chain; every path rebuilt
from its canonical ledger file with nothing kept in memory; every figure that can
move money compared, both against the engine's own books and across engines.

`unexplainedEconomicDifferences` must be zero. A difference is only excused by an
`ExplanationRule` naming the code that causes it, so the count cannot be lowered
by widening a tolerance without that widening being visible in the report.

Usage:
    python scripts/module1_composite_replay.py           # human summary, exit 1 if dirty
    python scripts/module1_composite_replay.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantagent.reconciliation.composite import run_composite  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "docs" / "architecture" / "module1_composite_replay.json"
#: Ledgers are run artifacts, not documentation, so they land under runtime/.
WORK_DIR = PROJECT_ROOT / "runtime" / "module1_composite"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full report")
    parser.add_argument(
        "--work-dir", default=str(WORK_DIR), help="where the canonical ledgers are written"
    )
    parser.add_argument(
        "--report", default=str(REPORT_PATH), help="where the difference table is written"
    )
    args = parser.parse_args()

    # Each invocation is a fresh measurement, so it starts from fresh ledgers.
    # Appending a second run of the same scenario to the previous run's chain is
    # not a variation worth measuring: order ids are content-addressed over
    # lineage, so identical runs collide and the chain correctly refuses the
    # second RISK_APPROVED for an order it already recorded as filled.
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)

    report = run_composite(work)
    payload = report.to_dict()

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if report.clean else 1

    print(f"composite replay -> {destination.relative_to(PROJECT_ROOT)}")
    print(f"unexplained economic differences: {report.unexplained_economic_differences}")
    print()
    for path in report.paths:
        snapshot = path.snapshot
        print(f"  {path.label}")
        print(
            f"    cash {snapshot.cash:,.4f}  nav {snapshot.nav:,.4f}  "
            f"fees {snapshot.fees_total:,.4f}  stamp duty {snapshot.stamp_duty:,.4f}"
        )
        print(
            f"    orders {snapshot.counts['orders']}  fills {snapshot.counts['fills']}  "
            f"rejected {snapshot.counts['rejected']}  cancelled {snapshot.counts['cancelled']}"
        )
        print(
            f"    accounting identity residual {snapshot.identity_residual:+.12f}  "
            f"lineage gaps {len(snapshot.lineage_gaps)}"
        )
        print(
            f"    own books vs replay: "
            f"{path.native_table.unexplained_economic_differences} unexplained "
            f"of {len(path.native_table.differences)}"
        )
        for note in path.notes:
            print(f"    note: {note}")
    print()
    for table in report.cross_tables:
        print(
            f"  {table.left_label} vs {table.right_label}: "
            f"{len(table.differences)} differences, "
            f"{table.unexplained_economic_differences} unexplained"
        )
        for difference in table.unexplained:
            print(
                f"    BLOCKING {difference.dimension}: "
                f"{difference.left_value} vs {difference.right_value}"
            )
    if not report.clean:
        print("\nGATE BLOCKED: every difference must be explained by a named rule")
        return 1
    print("\nGATE OPEN: unexplained_economic_differences = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
