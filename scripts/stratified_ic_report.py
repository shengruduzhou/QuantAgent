"""Run quantagent.diagnostics.stratified_ic on existing OOS predictions.

Usage:
  AI_quant_venv/bin/python scripts/stratified_ic_report.py

Env vars:
  QA_PROBE_DIR — model dir with walk_forward/fold_*/fold_*_oos_predictions.parquet
                 (default: runtime/models/v7_alpha_full_universe_nosynth_v9)
  QA_MARKET_FEATURES — path to market_features.parquet (default:
                       runtime/data/v7/silver/market_panel/market_features.parquet)
  QA_BENCHMARK — path to benchmark index parquet for regime labelling
                 (default: runtime/data/v7/raw/akshare/index/equity_index.parquet)
  QA_SECTOR_MAP — path to PIT sector map parquet. Current snapshots with
                  available_at after OOS dates will not backfill history.
                  (default: runtime/data/v7/silver/sector_map/sector_map.parquet)
  QA_OUTPUT_DIR — output dir for JSON + markdown + per-axis CSVs
                  (default: runtime/reports/stratified_ic)

The script does NOT train models or touch the GPU. It only reads parquet
files and emits report files. Safe to run while training is in progress.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.diagnostics.stratified_ic import (
    StratifiedICConfig,
    compute_stratified_ic,
    write_report,
)


def _load_predictions(probe_dir: Path) -> pd.DataFrame:
    """Concatenate every per-horizon per-fold OOS predictions parquet."""
    pattern = str(probe_dir / "walk_forward" / "fold_*" / "fold_*_oos_predictions.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no prediction parquets under {pattern}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        keep = ["trade_date", "symbol", "horizon", "prediction"]
        # carry whichever forward_return_*d columns exist
        keep += [c for c in df.columns if c.startswith("forward_return_")]
        frames.append(df[[c for c in keep if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def _load_market_features(path: Path) -> pd.DataFrame:
    """Load (or compute) market features needed for liquidity + vol buckets.

    Data quality gap (May 2026): ``market_features.parquet`` was last
    refreshed 2020-09-25 (so it covers 1999-11 → 2020-09 only), and the
    ``amount_mean_20d`` column is all-NaN even for that window. The raw
    ``market_panel.parquet`` has ``amount`` and ``close`` covering
    2020-09-28 onwards. The two files are therefore non-overlapping in
    time which breaks any join-based feature lookup.

    Workaround: compute both ``amount_mean_20d`` and ``volatility_20d``
    directly from ``market_panel.parquet`` (which covers the v9 OOS
    period 2020-02 → 2023-01 partially) plus fall back to the legacy
    ``market_features.parquet`` for the older window. Until the feature
    pipeline is rebuilt this is the only way to get a coherent panel.
    """
    raw_path = path.parent / "market_panel.parquet"
    if not raw_path.exists():
        print(f"warning: {raw_path} not found")
        return pd.DataFrame()
    raw = pd.read_parquet(raw_path, columns=["trade_date", "symbol", "close", "amount"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw["symbol"] = raw["symbol"].astype(str)
    raw = raw.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    # 20-day rolling mean of dollar amount (proxy for size)
    raw["amount_mean_20d"] = raw.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    # 20-day realised vol from close-to-close log returns
    raw["log_ret"] = raw.groupby("symbol")["close"].transform(lambda s: pd.Series(s).pct_change())
    raw["volatility_20d"] = raw.groupby("symbol")["log_ret"].transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    n_amount = int(raw["amount_mean_20d"].notna().sum())
    n_vol = int(raw["volatility_20d"].notna().sum())
    print(f"    computed from raw market_panel: amount_mean_20d non-null={n_amount:,}  volatility_20d non-null={n_vol:,}")
    # Backfill volatility_20d from legacy market_features.parquet where
    # raw computation has no data (i.e. pre-2020-09 window) — keeps the
    # report usable across the full v9 OOS window even before the
    # feature pipeline rebuild.
    if path.exists():
        legacy = pd.read_parquet(path, columns=["trade_date", "symbol", "volatility_20d"])
        legacy["trade_date"] = pd.to_datetime(legacy["trade_date"], errors="coerce")
        legacy["symbol"] = legacy["symbol"].astype(str)
        legacy = legacy.dropna(subset=["trade_date", "symbol", "volatility_20d"])
        legacy = legacy.rename(columns={"volatility_20d": "volatility_20d_legacy"})
        raw = raw.merge(legacy, on=["trade_date", "symbol"], how="left")
        raw["volatility_20d"] = raw["volatility_20d"].fillna(raw["volatility_20d_legacy"])
        raw = raw.drop(columns=["volatility_20d_legacy"], errors="ignore")
        backfilled = int(raw["volatility_20d"].notna().sum())
        print(f"    after legacy backfill: volatility_20d non-null={backfilled:,}")
    return raw[["trade_date", "symbol", "amount_mean_20d", "volatility_20d"]]


def _load_regime_frame(benchmark_path: Path) -> pd.DataFrame:
    """Use the same regime computation the training pipeline uses."""
    if not benchmark_path.exists():
        print(f"warning: {benchmark_path} not found — regime axis will be UNKNOWN")
        return pd.DataFrame()
    bench = pd.read_parquet(benchmark_path)
    # Schema uses observation_date (not trade_date) and symbol like "sh000300"
    date_col = "trade_date" if "trade_date" in bench.columns else "observation_date"
    bench = bench.rename(columns={date_col: "trade_date"})
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.dropna(subset=["trade_date"]).reset_index(drop=True)
    if "symbol" in bench.columns:
        bench = bench[bench["symbol"].astype(str).str.contains("000300", na=False)]
    elif "code" in bench.columns:
        bench = bench[bench["code"].astype(str).str.contains("000300|csi300", case=False, regex=True, na=False)]
    bench = bench.sort_values("trade_date").reset_index(drop=True)
    if bench.empty:
        return pd.DataFrame()
    # Minimal regime labelling: replicate the v7 _compute_regime_frame logic
    from quantagent.training.v7_experiment import _compute_regime_frame, V7TrainingConfig
    cfg = V7TrainingConfig(feature_columns=("placeholder",))
    return _compute_regime_frame(bench[["trade_date", "close"]], cfg)


def _load_sector_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"warning: {path} not found — sector axes will be UNKNOWN")
        return pd.DataFrame()
    cols = ["symbol", "sector_level_1", "sector_level_2", "source", "available_at", "coverage_status"]
    frame = pd.read_parquet(path)
    keep = [c for c in cols if c in frame.columns]
    frame = frame[keep].copy()
    if "available_at" in frame.columns:
        frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str)
    return frame


def main() -> None:
    probe_dir = Path(os.environ.get(
        "QA_PROBE_DIR",
        "runtime/models/v7_alpha_full_universe_nosynth_v9",
    ))
    mf_path = Path(os.environ.get(
        "QA_MARKET_FEATURES",
        "runtime/data/v7/silver/market_panel/market_features.parquet",
    ))
    bench_path = Path(os.environ.get(
        "QA_BENCHMARK",
        "runtime/data/v7/raw/akshare/index/equity_index.parquet",
    ))
    sector_path = Path(os.environ.get(
        "QA_SECTOR_MAP",
        "runtime/data/v7/silver/sector_map/sector_map.parquet",
    ))
    out_dir = Path(os.environ.get("QA_OUTPUT_DIR", "runtime/reports/stratified_ic"))

    print(f"probe_dir       : {probe_dir}")
    print(f"market_features : {mf_path}")
    print(f"benchmark       : {bench_path}")
    print(f"sector_map      : {sector_path}")
    print(f"output_dir      : {out_dir}")

    print("loading predictions ...", flush=True)
    preds = _load_predictions(probe_dir)
    print(f"  predictions rows: {len(preds):,}  symbols: {preds['symbol'].nunique():,}  horizons: {sorted(preds['horizon'].unique().tolist())}")

    print("loading market_features ...", flush=True)
    mf = _load_market_features(mf_path)
    if not mf.empty:
        print(f"  market_features rows: {len(mf):,}")

    print("computing regime frame ...", flush=True)
    regime = _load_regime_frame(bench_path)
    if not regime.empty:
        print(f"  regime rows: {len(regime):,}  states: {regime['regime_state'].value_counts().to_dict()}")

    print("loading sector_map ...", flush=True)
    sector = _load_sector_map(sector_path)
    if not sector.empty:
        status_counts = sector["coverage_status"].value_counts(dropna=False).to_dict() if "coverage_status" in sector.columns else {}
        print(f"  sector rows: {len(sector):,}  symbols: {sector['symbol'].nunique():,}  coverage_status: {status_counts}")

    print("computing stratified IC ...", flush=True)
    result = compute_stratified_ic(
        preds,
        market_features=mf if not mf.empty else None,
        regime_frame=regime if not regime.empty else None,
        sector_map=sector if not sector.empty else None,
        config=StratifiedICConfig(min_symbols_per_date=10, min_days_per_bucket=20),
    )
    print(f"  axes computed: {list(result.by_axis.keys())}")

    paths = write_report(result, out_dir)
    print("written:")
    for k, v in paths.items():
        print(f"  {k:20s} → {v}")


if __name__ == "__main__":
    main()
