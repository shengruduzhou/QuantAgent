#!/usr/bin/env python3
"""Stage 2.5: full per-minute causal intraday panel for the 675 names +
VECTORIZED cross-sectional features + causal relative-volume profile.

Per-stock causal features via build_causal_intraday_feature_frame (no future
high/low/volume/VWAP). Cross-sectional features computed with built-in groupby
transforms ONLY (no Python lambdas) so it runs in minutes, not hours.
relative_volume = current volume / expanding mean of PRIOR same-minute-of-day
volume (causal). Benchmark = held-universe mean, industry = sector groups
(675-name held-universe proxies; labeled as such).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "../src"))
from quantagent.execution.intraday_features import build_causal_intraday_feature_frame  # noqa: E402

MIN_DIR = "runtime/data/v7/silver/minute_bars"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
OUT = Path("runtime/data/v7/silver/intraday_2026"); OUT.mkdir(parents=True, exist_ok=True)
START, END = "2025-08-01", "2026-05-31"
PERSTOCK_CKPT = OUT / "intraday_panel_675_perstock.parquet"


def build_perstock() -> pd.DataFrame:
    if PERSTOCK_CKPT.exists():
        print(f"reusing per-stock checkpoint {PERSTOCK_CKPT}", flush=True)
        return pd.read_parquet(PERSTOCK_CKPT)
    files = sorted(glob.glob(MIN_DIR + "/*.parquet"))
    print(f"{len(files)} minute files; building per-stock causal features...", flush=True)
    feats = []
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["symbol", "trade_time", "trade_date", "open", "high", "low", "close", "volume", "amount"])
        except Exception:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df[(df["trade_date"] >= pd.Timestamp(START)) & (df["trade_date"] <= pd.Timestamp(END))]
        if df.empty:
            continue
        ff = build_causal_intraday_feature_frame(df, include_level2=False)
        if not ff.empty:
            feats.append(ff)
        if (i + 1) % 50 == 0:
            print(f"  per-stock {i+1}/{len(files)}", flush=True)
    panel = pd.concat(feats, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["trade_time"] = pd.to_datetime(panel["trade_time"])
    panel.to_parquet(PERSTOCK_CKPT, index=False)
    print(f"per-stock panel checkpointed: {len(panel):,} rows, {panel['symbol'].nunique()} symbols", flush=True)
    return panel


def main() -> int:
    panel = build_perstock()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]); panel["trade_time"] = pd.to_datetime(panel["trade_time"])
    sec = pd.read_parquet(SECTOR)
    scol = next((c for c in sec.columns if "sector" in c.lower() or "industry" in c.lower()), None)
    panel["sector"] = panel["symbol"].astype(str).map(dict(zip(sec["symbol"].astype(str), sec[scol].astype(str)))).fillna("UNK") if scol else "UNK"

    r = "intraday_return"
    print("vectorized cross-sectional pass...", flush=True)
    panel["_pos"] = (panel[r] > 0).astype("float32")
    panel["_neg"] = (panel[r] < 0).astype("float32")
    g = panel.groupby(["trade_date", "trade_time"])
    mkt = g[r].transform("mean")
    panel["cs_return_rank"] = g[r].rank(pct=True)
    if "volume_zscore_20m" in panel:
        panel["cs_volume_shock_rank"] = g["volume_zscore_20m"].rank(pct=True)
    panel["stock_return_minus_index"] = panel[r] - mkt
    panel["market_breadth_intraday"] = g["_pos"].transform("mean")
    panel["up_down_ratio_intraday"] = g["_pos"].transform("sum") / g["_neg"].transform("sum").replace(0, np.nan)
    gs = panel.groupby(["trade_date", "trade_time", "sector"])
    ind = gs[r].transform("mean")
    panel["stock_return_minus_industry"] = panel[r] - ind
    panel["industry_breadth_intraday"] = gs["_pos"].transform("mean")
    panel["sector_synchronized_move"] = ind
    panel["intraday_beta_proxy"] = (panel[r] / mkt.replace(0, np.nan)).clip(-5, 5)
    panel["intraday_residual_return"] = panel[r] - panel["intraday_beta_proxy"].fillna(1.0) * mkt

    # causal relative volume vs prior same-minute days (expanding mean of PRIOR obs)
    if "volume" in panel:
        panel = panel.sort_values(["symbol", "trade_time", "trade_date"])
        panel["mod"] = panel["trade_time"].dt.strftime("%H:%M")
        gb = panel.groupby(["symbol", "mod"])["volume"]
        prior_sum = gb.cumsum() - panel["volume"]
        prior_cnt = gb.cumcount()
        prof = prior_sum / prior_cnt.replace(0, np.nan)
        panel["relative_volume_vs_20d"] = (panel["volume"] / prof.replace(0, np.nan)).clip(0, 50)
        panel = panel.drop(columns=["mod"])
    panel = panel.drop(columns=["_pos", "_neg"])

    panel = panel[panel["trade_date"] >= pd.Timestamp("2025-09-01")].sort_values(
        ["symbol", "trade_date", "trade_time"]).reset_index(drop=True)
    cs = ["cs_return_rank", "cs_volume_shock_rank", "stock_return_minus_index", "market_breadth_intraday",
          "up_down_ratio_intraday", "stock_return_minus_industry", "industry_breadth_intraday",
          "sector_synchronized_move", "intraday_beta_proxy", "intraday_residual_return", "relative_volume_vs_20d"]
    print("cross-sectional null rates:", {c: round(float(panel[c].isna().mean()), 3) for c in cs if c in panel}, flush=True)
    panel.to_parquet(OUT / "intraday_panel_675.parquet", index=False)
    print(f"wrote {OUT/'intraday_panel_675.parquet'}: rows={len(panel):,} symbols={panel['symbol'].nunique()} dates={panel['trade_date'].nunique()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
