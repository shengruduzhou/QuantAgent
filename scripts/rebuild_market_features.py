"""Rebuild runtime/data/v7/silver/market_panel/market_features.parquet.

Stage 2.1 fix for Task #19 (broken market_features pipeline). The old
parquet last updated 2020-09-25 with an all-NaN ``amount_mean_20d``;
the new build pulls fresh ``market_panel.parquet``, derives
``is_suspended`` / ``is_limit_up`` / ``is_limit_down`` from OHLCV
(matching the universe-filter contract), backfills ``is_st`` from any
prior parquet where available, and writes a coherent panel covering
the full date window.

Env vars:
  QA_MARKET_PANEL — input parquet (default:
    runtime/data/v7/silver/market_panel/market_panel.parquet)
  QA_PRIOR_FEATURES — optional prior market_features.parquet to
    backfill is_st from. Default: same dir / market_features.parquet
  QA_OUTPUT — output path (default: overwrite same dir)
  QA_BACKUP — when ``1`` (default), copy the old parquet to
    ``market_features.parquet.bak`` before overwriting.

The script does NOT use GPU and is safe to run while v10 training
is in progress.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.v7_dataset_builder import build_market_features


def main() -> None:
    mp_path = Path(os.environ.get(
        "QA_MARKET_PANEL",
        "runtime/data/v7/silver/market_panel/market_panel.parquet",
    ))
    prior_path = Path(os.environ.get(
        "QA_PRIOR_FEATURES",
        "runtime/data/v7/silver/market_panel/market_features.parquet",
    ))
    out_path = Path(os.environ.get("QA_OUTPUT", str(prior_path)))
    do_backup = os.environ.get("QA_BACKUP", "1") in {"1", "true", "yes"}

    if not mp_path.exists():
        raise SystemExit(f"market_panel not found at {mp_path}")
    print(f"input  market_panel : {mp_path}")
    print(f"prior  features     : {prior_path} ({'exists' if prior_path.exists() else 'missing'})")
    print(f"output features     : {out_path}")
    print(f"backup prior        : {do_backup}")

    print("loading market_panel ...", flush=True)
    mp = pd.read_parquet(mp_path)
    print(f"  rows={len(mp):,}  cols={list(mp.columns)}")
    print(f"  date range: {pd.to_datetime(mp['trade_date']).min()} → {pd.to_datetime(mp['trade_date']).max()}")
    print(f"  symbols   : {mp['symbol'].nunique():,}")

    # Pull ST flag from the prior parquet if present (covers 2020-02 → 2020-09 in v9 era).
    st_flags = None
    if prior_path.exists():
        try:
            prior = pd.read_parquet(prior_path, columns=["trade_date", "symbol", "is_st"])
            n_st_true = int(prior["is_st"].fillna(False).astype(bool).sum())
            print(f"  prior is_st coverage: {n_st_true:,} True rows in {len(prior):,} total")
            st_flags = prior
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  WARNING: could not read prior is_st column: {exc}")

    print("building features (derived flags) ...", flush=True)
    feats = build_market_features(mp, st_flags=st_flags)
    print(f"  output rows={len(feats):,}  cols={len(feats.columns)}")
    summary = {
        "is_suspended_true": int(feats["is_suspended"].sum()),
        "is_st_true": int(feats["is_st"].sum()),
        "is_limit_up_true": int(feats["is_limit_up"].sum()),
        "is_limit_down_true": int(feats["is_limit_down"].sum()),
        "amount_mean_20d_nonnull": int(feats["amount_mean_20d"].notna().sum()),
        "volatility_20d_nonnull": int(feats["volatility_20d"].notna().sum()),
    }
    print("  flag / numeric summary:")
    for k, v in summary.items():
        print(f"    {k}: {v:,}")

    if do_backup and prior_path.exists() and out_path.resolve() == prior_path.resolve():
        bak = prior_path.with_suffix(".parquet.bak")
        if not bak.exists():
            shutil.copy2(prior_path, bak)
            print(f"  backed up prior to {bak}")
        else:
            print(f"  backup {bak} already exists — not overwriting")

    print(f"writing {out_path} ...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print("done")


if __name__ == "__main__":
    main()
