#!/usr/bin/env python3
"""Resumable full-universe RAW daily-bar acquisition with provider fallback.

Why RAW: the previous full-universe panel silently mixed forward-adjusted (qfq)
prices for the frozen 3,872-symbol cohort with unadjusted prices for backfilled
names while declaring ``adjustment_method = "none"`` everywhere. Raw traded
prices are the only vendor-neutral base; adjustment is applied downstream from
the explicit factor table (``u0_pit_intervals.py``).

Provider chain (first non-empty answer wins the symbol, never a blend):

* ``tickflow`` — entitled, publishes turnover (``amount``); hard 10 requests/min.
* ``tencent``  — public, fast, every board, but publishes no turnover column.

Each symbol lands in its own parquet partition plus a ledger row recording the
provider, retry class, row count and date range, so an interrupted run resumes
without refetching and a permanent failure is not retried forever.

Usage:
  AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \\
      --providers tencent --max-minutes 180
  AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \\
      --providers tickflow --staging-name tickflow --max-minutes 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.acquire import (  # noqa: E402
    BarAcquisition,
    ProviderSpec,
    trading_day_cutoff,
    write_run_manifest,
)
from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.ashare.sources import SinaSource, TencentSource, TickFlowSource  # noqa: E402

BARS_ROOT = REPO / "runtime/data/u0/bars"
MASTER = REPO / "runtime/data/u0/security_master.parquet"
HISTORY_START = "1990-12-01"          # first A-share session
TICKFLOW_PACE_S = 6.5                 # measured hard limit: 10 requests / minute
TENCENT_PACE_S = 0.05                 # module-level host pacer adds the real spacing


def build_providers(names: list[str], end: pd.Timestamp) -> list[ProviderSpec]:
    specs: list[ProviderSpec] = []
    client = HttpClient(timeout=20, max_attempts=3)
    for name in names:
        if name == "tickflow":
            source = TickFlowSource()
            specs.append(ProviderSpec(
                "tickflow",
                lambda symbol, s=source: s.daily_bars(symbol, pd.Timestamp(HISTORY_START), end),
                TICKFLOW_PACE_S))
        elif name == "tencent":
            source = TencentSource(client)
            specs.append(ProviderSpec(
                "tencent",
                lambda symbol, s=source: s.daily_bars(symbol, HISTORY_START, str(end.date())),
                TENCENT_PACE_S))
        elif name == "sina_factors":
            source = SinaSource(client)
            specs.append(ProviderSpec("sina", lambda symbol, s=source: s.adjust_factors(symbol), 0.05))
        else:
            raise SystemExit(f"unknown provider {name!r}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="required: this command performs real vendor calls")
    parser.add_argument("--providers", default="tickflow,tencent",
                        help="comma-separated fallback chain, highest priority first")
    parser.add_argument("--staging-name", default=None,
                        help="staging sub-directory (defaults to the first provider name)")
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0, help="stop after N symbols (smoke runs)")
    parser.add_argument("--boards", default="", help="restrict to these boards")
    parser.add_argument("--symbols", default="", help="restrict to these symbols")
    parser.add_argument("--refetch", action="store_true",
                        help="ignore existing partitions instead of resuming past them")
    parser.add_argument("--shard", default="",
                        help="'i/n' — process only shard i of n disjoint symbol slices, so "
                             "several workers can share one staging directory safely")
    parser.add_argument("--order", choices=["forward", "reverse"], default="forward",
                        help="reverse lets a second worker converge from the other end of the "
                             "universe instead of duplicating the first worker's queue")
    parser.add_argument("--skip-if-in", default="",
                        help="comma-separated staging directory names under runtime/data/u0/bars "
                             "whose partitions already satisfy a symbol")
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to acquire: --allow-network was not confirmed")
        return 2
    load_repo_env()
    if not MASTER.exists():
        print(f"missing security master: {MASTER.relative_to(REPO)} — run u0_security_master.py first")
        return 3

    master = pd.read_parquet(MASTER)
    if args.boards:
        wanted = {b.strip() for b in args.boards.split(",") if b.strip()}
        master = master[master["board"].isin(wanted)]
    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",") if s.strip()}
        master = master[master["symbol"].isin(wanted)]
    symbols = sorted(master["symbol"].astype(str).unique())
    shard_label = "1/1"
    if args.shard:
        index_str, _, count_str = args.shard.partition("/")
        shard_index, shard_count = int(index_str), int(count_str)
        if not 1 <= shard_index <= shard_count:
            raise SystemExit(f"invalid --shard {args.shard!r}")
        symbols = [s for i, s in enumerate(symbols) if i % shard_count == shard_index - 1]
        shard_label = f"{shard_index}/{shard_count}"
    already: set[str] = set()
    for name in (n.strip() for n in args.skip_if_in.split(",") if n.strip()):
        directory = BARS_ROOT / name
        already |= {p.stem.replace("sym_", "").replace("_", ".")
                    for p in directory.glob("sym_*.parquet")} if directory.exists() else set()
    for legacy in (REPO / "runtime/data/v7/full_universe/_staging",):
        if "legacy" in args.skip_if_in and legacy.exists():
            already |= {p.stem.replace("sym_", "").replace("_", ".")
                        for p in legacy.glob("sym_*.parquet")}
    if already:
        before = len(symbols)
        symbols = [s for s in symbols if s not in already]
        print(f"skipping {before - len(symbols)} symbols already served by "
              f"{args.skip_if_in}", flush=True)
    if args.order == "reverse":
        symbols = list(reversed(symbols))
    if args.limit:
        symbols = symbols[:args.limit]
    boards = dict(zip(master["symbol"].astype(str), master["board"].astype(str)))

    provider_names = [p.strip() for p in args.providers.split(",") if p.strip()]
    staging = BARS_ROOT / (args.staging_name or provider_names[0])
    ledger_suffix = f"_shard{shard_label.replace('/', 'of')}" if args.shard else ""
    ledger = staging.parent / f"{staging.name}{ledger_suffix}_ledger.csv"
    cancel = staging.parent / f"{staging.name}.cancel"
    end = trading_day_cutoff()

    print(f"universe {len(symbols)} symbols (shard {shard_label}) · providers {provider_names} · "
          f"staging {staging.relative_to(REPO)} · through {end.date()}", flush=True)
    worker = BarAcquisition(staging, ledger, build_providers(provider_names, end),
                            cancel_file=cancel)
    report = worker.run(symbols, boards=boards, max_minutes=args.max_minutes,
                        skip_existing=not args.refetch)
    payload = {
        "command": "u0_acquire_bars", "providers": provider_names,
        "universe_size": len(symbols), "shard": shard_label,
        "staging": str(staging.relative_to(REPO)),
        "ledger": str(ledger.relative_to(REPO)), "history_start": HISTORY_START,
        "through": str(end.date()), "adjustment": "none (raw traded prices)",
        "volume_unit": "shares", "amount_unit": "CNY",
        **report.as_dict(),
    }
    write_run_manifest(staging.parent / f"{staging.name}{ledger_suffix}_run_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
