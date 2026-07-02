#!/usr/bin/env python3
"""Stage 9 step 1 — extend the SW1 sector panel with wave-state features.

Adds the features the wave state machine needs that the base panel lacks:
  breakout_60 / breakout_120  sector NAV makes a new 60/120d high (主线启动)
  breadth_expansion           20d change in %-above-MA60 (breadth widening)
  limitup_diffusion           fraction of sector members limit-up that day
  limitup_diffusion_5d        5d mean of the above (sustained 涨停扩散)
  beta_60 / alpha_60          rolling 60d beta/alpha vs equal-sector market
  rs_60_chg_20                20d change in rs_60 (relative-strength acceleration)
Writes sector_panel_wave.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
OUT = "runtime/stage8_sector_rotation/sector_panel_wave.parquet"
SEC = "sector_level_1"


def main() -> int:
    sp = pd.read_parquet(SECTOR_PANEL)
    sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[sp.trade_date >= "2018-01-01"].sort_values([SEC, "trade_date"]).reset_index(drop=True)

    # ---- sector NAV breakout (clean rolling cum-return level per sector) ----
    g = sp.groupby(SEC, sort=False)
    logret = np.log1p(sp["ret_eqw"].clip(lower=-0.99))
    sp["_lvl"] = g.apply(lambda x: pd.Series(np.nan, index=x.index)) if False else logret.groupby(sp[SEC]).cumsum()
    lvl = sp["_lvl"]
    for w in (60, 120):
        roll_max = lvl.groupby(sp[SEC]).transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).max())
        sp[f"breakout_{w}"] = (lvl >= roll_max - 1e-9).astype(float)
    sp["breadth_expansion"] = g["breadth_ma60"].transform(lambda s: s - s.shift(20))
    sp["rs_60_chg_20"] = g["rs_60"].transform(lambda s: s - s.shift(20))

    # ---- limit-up diffusion from raw panel ----
    sm = pd.read_parquet(SECTOR)[["symbol", SEC]].dropna().drop_duplicates("symbol")
    pf = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "is_limit_up", "is_suspended"])
    pf["trade_date"] = pd.to_datetime(pf["trade_date"])
    pf = pf[pf.trade_date >= "2018-01-01"].merge(sm, on="symbol", how="inner")
    pf = pf[~pf["is_suspended"].fillna(False).astype(bool)]
    lu = (pf.assign(lu=pf["is_limit_up"].fillna(False).astype(float))
            .groupby(["trade_date", SEC])["lu"].mean().rename("limitup_diffusion").reset_index())
    sp = sp.merge(lu, on=["trade_date", SEC], how="left")
    sp["limitup_diffusion"] = sp["limitup_diffusion"].fillna(0.0)
    sp["limitup_diffusion_5d"] = (sp.sort_values([SEC, "trade_date"])
                                  .groupby(SEC)["limitup_diffusion"]
                                  .transform(lambda s: s.rolling(5, min_periods=2).mean()))

    # ---- rolling 60d beta / alpha vs equal-sector market ----
    mkt = sp.groupby("trade_date")["ret_eqw"].transform("mean")
    sp["_mkt"] = mkt
    def _beta(x):
        cov = x["ret_eqw"].rolling(60, min_periods=30).cov(x["_mkt"])
        var = x["_mkt"].rolling(60, min_periods=30).var()
        return cov / (var + 1e-12)
    sp["beta_60"] = sp.groupby(SEC, group_keys=False).apply(_beta)
    sp["alpha_60"] = (g["ret_eqw"].transform(lambda s: s.rolling(60, min_periods=30).mean())
                      - sp["beta_60"] * sp.groupby(SEC)["_mkt"].transform(lambda s: s.rolling(60, min_periods=30).mean()))

    sp = sp.drop(columns=["_lvl", "_mkt"])
    sp.to_parquet(OUT, index=False)
    print(f"[write] {OUT}  rows={len(sp):,} cols={len(sp.columns)}")
    new = ["breakout_60", "breakout_120", "breadth_expansion", "rs_60_chg_20",
           "limitup_diffusion", "limitup_diffusion_5d", "beta_60", "alpha_60"]
    print("new feature coverage / sample (last day):")
    last = sp[sp.trade_date == sp.trade_date.max()].set_index(SEC)
    print(last[["mom_60", "rs_60", "breadth_ma60"] + new].round(3).head(8).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
