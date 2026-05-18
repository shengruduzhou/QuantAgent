"""Reconcile qlib↔akshare adjustment scales in the merged market_panel.

Problem: qlib's free CN dump is normalised so each symbol's first trading day
has close=1.0. AkShare's daily fetch with adjust="qfq" uses today-anchored
forward-adjusted prices. Concatenating the two produces a 3–10× jump at the
boundary (qlib last date → akshare first date), which corrupts every
return/momentum/volatility signal computed downstream.

Fix strategy: for each symbol, compute
    ratio = first_akshare_close / last_qlib_close
and multiply all qlib OHLC rows for that symbol by ratio. Volume/amount are
left unscaled (they aren't price-adjusted).

We accept a single trading-day return as the residual error (typical |return|
< 5%). Symbols present only in qlib or only in akshare are left untouched.

Outputs:
    market_panel.parquet     — repaired panel (replaces stale file)
    market_panel_repaired.csv — sanity dump (optional, off by default)
    manifest market_panel.json — refreshed with vendor=qlib_rescaled+akshare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLS = ("open", "high", "low", "close")


def repair_panel(panel_path: Path, manifest_path: Path, dry_run: bool = False) -> dict[str, object]:
    print(f"[load] {panel_path}")
    df = pd.read_csv(panel_path) if panel_path.suffix == ".csv" else pd.read_parquet(panel_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "available_at" in df:
        df["available_at"] = pd.to_datetime(df["available_at"], errors="coerce")
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    print(f"[load] rows={len(df):,}, symbols={df['symbol'].nunique()}, dates={df['trade_date'].nunique()}")

    is_qlib = df["source"] == "qlib"
    is_akshare = df["source"].astype(str).str.startswith("akshare")
    print(f"[split] qlib rows={int(is_qlib.sum()):,}, akshare rows={int(is_akshare.sum()):,}")

    # Boundary metrics per symbol: last qlib row + first akshare row.
    qlib_last = (
        df[is_qlib]
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "trade_date", "close"]]
        .rename(columns={"trade_date": "qlib_last_date", "close": "qlib_last_close"})
    )
    ak_first = (
        df[is_akshare]
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol", as_index=False)
        .head(1)[["symbol", "trade_date", "close"]]
        .rename(columns={"trade_date": "ak_first_date", "close": "ak_first_close"})
    )
    bridge = qlib_last.merge(ak_first, on="symbol", how="inner")
    bridge = bridge[(bridge["qlib_last_close"] > 0) & (bridge["ak_first_close"] > 0)].copy()
    bridge["ratio"] = bridge["ak_first_close"] / bridge["qlib_last_close"]
    bridge["gap_days"] = (bridge["ak_first_date"] - bridge["qlib_last_date"]).dt.days
    print("[bridge] ratio percentiles:")
    print(bridge["ratio"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_string())

    # Sanity: discard outlier ratios (likely delisted/relisted or symbol mis-merge).
    sane = bridge[(bridge["ratio"].between(0.05, 50)) & (bridge["gap_days"].between(0, 30))]
    discarded = bridge[~bridge.index.isin(sane.index)]
    print(f"[bridge] kept {len(sane)} / {len(bridge)} symbols within sane ratio band; "
          f"{len(discarded)} discarded (extreme ratio or large date gap)")

    if dry_run:
        return {"status": "dry_run", "symbols_bridged": int(len(sane))}

    # Apply per-symbol rescaling to qlib rows only.
    ratio_lookup = sane.set_index("symbol")["ratio"]
    qlib_rows = df.loc[is_qlib].copy()
    qlib_rows["_ratio"] = qlib_rows["symbol"].map(ratio_lookup)
    has_ratio = qlib_rows["_ratio"].notna()
    n_rescaled_rows = int(has_ratio.sum())
    print(f"[rescale] rescaling {n_rescaled_rows:,} qlib rows across {ratio_lookup.size} symbols")
    for col in PRICE_COLS:
        if col not in qlib_rows:
            continue
        qlib_rows.loc[has_ratio, col] = qlib_rows.loc[has_ratio, col] * qlib_rows.loc[has_ratio, "_ratio"]
    qlib_rows.loc[has_ratio, "source"] = "qlib_rescaled"
    qlib_rows = qlib_rows.drop(columns=["_ratio"])

    rebuilt = pd.concat([qlib_rows, df.loc[is_akshare].copy()], ignore_index=True)
    rebuilt = rebuilt.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Continuity sanity: per symbol absolute log return at boundary should be < 0.10 for most stocks.
    pivot = rebuilt[rebuilt["symbol"].isin(sane["symbol"])].copy()
    pivot["log_close"] = np.log(pivot["close"].clip(lower=1e-6))
    pivot["dlog"] = pivot.groupby("symbol")["log_close"].diff()
    boundary_rows = pivot.merge(sane[["symbol", "qlib_last_date"]], on="symbol")
    boundary_returns = boundary_rows[
        (boundary_rows["trade_date"] > boundary_rows["qlib_last_date"])
        & (boundary_rows["trade_date"] <= boundary_rows["qlib_last_date"] + pd.Timedelta(days=10))
    ].dropna(subset=["dlog"])
    bnd_first = boundary_returns.sort_values(["symbol", "trade_date"]).groupby("symbol").head(1)
    print("[continuity] |Δlog close| at boundary first-akshare-day percentiles:")
    print(bnd_first["dlog"].abs().describe(percentiles=[0.5, 0.9, 0.99]).to_string())

    parquet_path = panel_path.with_suffix(".parquet")
    # Normalize dtypes for parquet safety.
    for col in PRICE_COLS + ("volume", "amount"):
        if col in rebuilt:
            rebuilt[col] = pd.to_numeric(rebuilt[col], errors="coerce").astype("float64")
    if "source_reliability" in rebuilt:
        rebuilt["source_reliability"] = pd.to_numeric(rebuilt["source_reliability"], errors="coerce").astype("float64")
    if "point_in_time_valid" in rebuilt:
        rebuilt["point_in_time_valid"] = rebuilt["point_in_time_valid"].fillna(True).astype("bool")
    for col in ("symbol", "source", "source_type"):
        if col in rebuilt:
            rebuilt[col] = rebuilt[col].astype("string")
    rebuilt["trade_date"] = pd.to_datetime(rebuilt["trade_date"])
    if "available_at" in rebuilt:
        rebuilt["available_at"] = pd.to_datetime(rebuilt["available_at"], errors="coerce")
    else:
        rebuilt["available_at"] = pd.NaT
    missing_available = rebuilt["available_at"].isna()
    rebuilt.loc[missing_available, "available_at"] = (
        pd.to_datetime(rebuilt.loc[missing_available, "trade_date"], errors="coerce") + pd.offsets.BDay(1)
    )

    print(f"[write] {parquet_path}")
    rebuilt.to_parquet(parquet_path, index=False)

    # Refresh manifest if present.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["vendor"] = "qlib_rescaled+akshare"
        manifest["row_count"] = int(len(rebuilt))
        manifest["start_date"] = str(rebuilt["trade_date"].min().date())
        manifest["end_date"] = str(rebuilt["trade_date"].max().date())
        manifest.setdefault("extra", {})["adjustment_repair"] = {
            "rescale_anchor": "akshare_qfq_first_close",
            "symbols_rescaled": int(ratio_lookup.size),
            "symbols_discarded": int(len(discarded)),
            "boundary_dlog_p99": float(bnd_first["dlog"].abs().quantile(0.99)) if not bnd_first.empty else None,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[manifest] refreshed {manifest_path}")

    return {
        "status": "repaired",
        "rows": int(len(rebuilt)),
        "symbols_rescaled": int(ratio_lookup.size),
        "symbols_discarded": int(len(discarded)),
        "output": str(parquet_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile qlib↔akshare adjustment scales.")
    parser.add_argument(
        "--panel",
        default="runtime/data/v7/silver/market_panel/market_panel.csv",
        help="Merged market panel file (csv or parquet)",
    )
    parser.add_argument(
        "--manifest",
        default="runtime/data/v7/manifests/market_panel.json",
        help="Manifest path to refresh",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = repair_panel(Path(args.panel), Path(args.manifest), dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
