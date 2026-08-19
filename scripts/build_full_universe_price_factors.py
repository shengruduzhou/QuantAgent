"""Compute the Alpha101 + GTJA-191 price-volume families on the certified panel.

Why this exists
---------------
`runtime/data/gold/full_universe/dataset.parquet` is the only artifact holding
FULL_UNIVERSE_GOLD_READY, and it carries fifteen features -- returns, moving
averages, volatility, turnover, Amihud, gaps. Three of the fifteen are moving
averages. It has zero Alpha101, zero GTJA-191, zero fundamental, zero event and
zero macro columns, which is why the workstation reads as a technical-analysis
system: on that artifact it is one.

The richer 348-column panel is not a substitute -- it covers 3,638 symbols with
zero STAR and zero BSE, on a qfq basis rather than hfq. Neither artifact
dominates. This script closes the half of the gap that is not blocked: the
price-volume families need nothing but OHLCV, which the certified panel already
has, on the adjustment basis it already uses.

What is NOT closed by this script: fundamentals, events and macro. Fundamental
coverage is frozen at 3,658 symbols with STAR (613) and BSE (328) never
fetched, so those remain BLOCKED_BY_DATA and are recorded as such rather than
approximated.

Two things this script must not do, both of which are measured repository
defects rather than hypotheticals
--------------------------------------------------------------------------
1. **Never batch by symbol.** Alpha101 is full of cross-sectional `rank`
   operators. Computing it on a symbol subset makes every rank a rank within
   that subset. That is not a subtle degradation: it silently corrupted 22
   alpha columns across an entire sleeve when a `--batch-symbols 300` flag was
   introduced, and the columns looked completely normal afterwards.

2. **Never truncate per-symbol history.** The longest look-back among these
   families is 250 sessions. Computing on a window shorter than that produces
   different numbers with no error: measured at 145d against 390d of warmup,
   58 of 101 alphas differed by more than 1e-9 and alpha072 differed by 396.8.

Both constraints point the same way: one pass, whole panel, whole history. That
costs memory rather than correctness, and the memory is affordable.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.factors.alpha101 import compute_alpha101  # noqa: E402
from quantagent.factors.gtja191 import compute_gtja191_factors  # noqa: E402

BASE_COLUMNS = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
DEFAULT_PANEL = Path("runtime/data/gold/full_universe/dataset.parquet")


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _load_panel(path: Path, symbol_limit: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=BASE_COLUMNS)
    if symbol_limit is not None:
        # A SCALE probe, not a production build. Whole histories of a few names,
        # never a date slice: truncating history changes the numbers silently,
        # and the resulting cross-section is explicitly not comparable to the
        # full-panel run.
        keep = sorted(frame["symbol"].unique())[:symbol_limit]
        frame = frame[frame["symbol"].isin(keep)]
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return frame


def _finite_report(frame: pd.DataFrame, key_columns: int = 2) -> dict:
    factor_columns = [c for c in frame.columns if c not in {"trade_date", "symbol"}]
    finite = frame[factor_columns].notna().to_numpy().mean()
    per_column = frame[factor_columns].notna().mean().sort_values()
    all_nan = [c for c in factor_columns if per_column[c] == 0.0]
    return {
        "factor_count": len(factor_columns),
        "finite_rate": round(float(finite), 6),
        "all_nan_columns": all_nan,
        "weakest": {c: round(float(per_column[c]), 6) for c in per_column.index[:5]},
    }


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PANEL.parent)
    parser.add_argument(
        "--symbol-limit", type=int, default=None,
        help="SCALE PROBE ONLY. Whole histories of the first N symbols. Cross-sectional "
             "ranks are then ranks within that subset, so the output is not a "
             "production artifact and is written with a _probe suffix.",
    )
    parser.add_argument("--families", default="alpha101,gtja191")
    args = parser.parse_args()

    if not args.panel.exists():
        print(f"BLOCKED_BY_DATA: panel not found at {args.panel}", file=sys.stderr)
        return 2

    probe = args.symbol_limit is not None
    suffix = "_probe" if probe else ""
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    t0 = time.time()
    panel = _load_panel(args.panel, args.symbol_limit)
    load_s = time.time() - t0
    rows = len(panel)
    symbols = panel["symbol"].nunique()
    print(
        f"[load] rows={rows:,} symbols={symbols:,} "
        f"span={panel['trade_date'].min().date()}..{panel['trade_date'].max().date()} "
        f"{load_s:.1f}s peak_rss={_peak_rss_gb():.2f}GB",
        flush=True,
    )

    report: dict = {
        "panel": str(args.panel),
        "rows": int(rows),
        "symbols": int(symbols),
        "probe": probe,
        "families": {},
    }

    for family in families:
        t = time.time()
        if family == "alpha101":
            out = compute_alpha101(panel, wide=True)
        elif family == "gtja191":
            out = compute_gtja191_factors(panel, wide=True)
        else:
            print(f"unknown family {family!r}", file=sys.stderr)
            return 2
        wall = time.time() - t

        # Row alignment is an invariant, not a hope: both engines return one row
        # per input row in input order. A mismatch means the join below would
        # silently pair a factor with the wrong bar.
        if len(out) != rows:
            print(
                f"FAIL {family}: produced {len(out):,} rows for {rows:,} input rows",
                file=sys.stderr,
            )
            return 3
        if not out["symbol"].equals(panel["symbol"]) or not out["trade_date"].equals(
            panel["trade_date"]
        ):
            print(f"FAIL {family}: row order diverged from the input panel", file=sys.stderr)
            return 3

        stats = _finite_report(out)
        stats.update(
            {"wall_seconds": round(wall, 1),
             "microseconds_per_row": round(wall / max(rows, 1) * 1e6, 2),
             "peak_rss_gb": round(_peak_rss_gb(), 2)}
        )
        path = args.output_dir / f"factors_{family}{suffix}.parquet"
        _write(out, path)
        stats["output"] = str(path)
        stats["output_mb"] = round(path.stat().st_size / (1024 * 1024), 1)
        report["families"][family] = stats
        print(f"[{family}] {json.dumps(stats, ensure_ascii=False)}", flush=True)

        del out
        gc.collect()

    report["total_wall_seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = round(_peak_rss_gb(), 2)
    report_path = args.output_dir / f"price_factor_build_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {report_path} total={report['total_wall_seconds']}s "
          f"peak_rss={report['peak_rss_gb']}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
