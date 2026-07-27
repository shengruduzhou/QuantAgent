#!/usr/bin/env python3
"""Build the full-universe gold training dataset from the raw U0 panel.

Bounded by default: ``--max-symbols`` keeps a smoke run cheap so the pipeline
can be proven before an expensive full build. The training gate is derived from
the U0 PIT certificate, so a successful build does NOT imply permission to
train -- and when PIT is blocked, the certificate says so and the script exits
non-zero.

    python scripts/build_full_universe_gold.py \
        --max-symbols 300 --start-date 2023-01-01 \
        --output runtime/data/gold/full_universe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantagent.data.ashare import contracts, gold_bridge  # noqa: E402
from quantagent.data.microstructure import contracts as mc  # noqa: E402
from quantagent.data.microstructure.store import RawEventStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime" / "data" / "u0"


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runtime/data/gold/full_universe")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="0 builds the whole universe")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--adjustment", default=contracts.ADJUST_HFQ,
                        choices=list(gold_bridge.ADJUSTMENT_METHODS))
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--seasoning-days", type=int,
                        default=gold_bridge.DEFAULT_SEASONING_DAYS)
    parser.add_argument("--journal", default="runtime/data/market_events")
    args = parser.parse_args()

    panel_path = U0 / "panel" / "daily_bars_raw.parquet"
    columns = ["symbol", "trade_date", "open", "high", "low", "close",
               "volume", "amount", "serving_provider"]
    panel = pd.read_parquet(panel_path, columns=columns)
    if args.start_date:
        panel = panel[panel["trade_date"] >= args.start_date]

    master = _read(U0 / "security_master.parquet")
    if args.max_symbols:
        # Deterministic subset spanning every board, so a smoke run is not
        # accidentally all-main-board.
        picks: list[str] = []
        merged = master.merge(panel[["symbol"]].drop_duplicates(), on="symbol")
        per_board = max(1, args.max_symbols // max(1, merged["board"].nunique()))
        for _, group in merged.groupby("board"):
            picks.extend(sorted(group["symbol"])[:per_board])
        panel = panel[panel["symbol"].isin(picks[:args.max_symbols])]

    factors = _read(U0 / "pit" / "adjust_factors.parquet")
    suspension = _read(U0 / "pit" / "suspension_intervals.parquet")
    st = _read(U0 / "pit" / "st_intervals.parquet")

    # The U0 PIT certificate is the authority on whether the ST register is a
    # complete dated source. It is not, so st_available stays False and the
    # masks record UNKNOWN rather than a confident FALSE.
    pit_certificate_path = U0 / "u0_strict_pit_certificate.json"
    pit_certificate = (
        json.loads(pit_certificate_path.read_text(encoding="utf-8"))
        if pit_certificate_path.exists() else None
    )
    st_field = (pit_certificate or {}).get("pit_field_availability", {}).get("st_intervals", "")
    st_available = st_field.startswith("AVAILABLE")

    observed: dict[str, pd.DataFrame] = {}
    store = RawEventStore(args.journal)
    inventory = store.inventory()
    if not inventory.empty:
        ticks = inventory[inventory.get("family") == mc.FAMILY_TRADE]
        if not ticks.empty:
            observed["tick_events"] = pd.DataFrame({
                "symbol": ticks["symbol"], "trade_date": ticks["trade_date"],
            })

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    rebuild = " ".join(["python", "scripts/build_full_universe_gold.py", *sys.argv[1:]])

    dataset, manifest = gold_bridge.build_gold_dataset(
        panel, master=master, factors=factors, suspension=suspension, st=st,
        st_available=st_available, observed_families=observed,
        adjustment_method=args.adjustment, horizons=horizons,
        seasoning_days=args.seasoning_days, source_commit=_source_commit(),
        rebuild_command=rebuild,
        inputs={
            "panel": str(panel_path.relative_to(REPO)),
            "adjust_factors": "runtime/data/u0/pit/adjust_factors.parquet",
            "suspension_intervals": "runtime/data/u0/pit/suspension_intervals.parquet",
            "st_intervals": "runtime/data/u0/pit/st_intervals.parquet",
            "security_master": "runtime/data/u0/security_master.parquet",
        },
    )
    certificate = gold_bridge.certify_training_slice(
        manifest, u0_pit_certificate=pit_certificate
    )

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(target / "full_universe_gold.parquet", index=False)
    (target / "build_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (target / "training_slice_certificate.json").write_text(
        json.dumps(certificate.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps({
        "rows": manifest.rows,
        "symbols": manifest.symbols,
        "date_range": manifest.date_range,
        "adjustment_method": manifest.adjustment_method,
        "adjustment_factor_version": manifest.adjustment_factor_version,
        "label_columns": manifest.label_columns,
        "mask_distribution": manifest.mask_distribution,
        "availability_columns": manifest.availability_columns,
        "rows_dropped": manifest.rows_dropped,
        "content_hash": manifest.content_hash,
        "warnings": manifest.warnings,
        "training_permitted": certificate.training_permitted,
        "decision": certificate.decision,
        "blockers": certificate.blockers,
        "output": str(target),
    }, ensure_ascii=False, indent=2))

    # A blocked training gate is a non-zero exit: callers chaining a training
    # run must stop here rather than proceed on an unusable dataset.
    return 0 if certificate.training_permitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
