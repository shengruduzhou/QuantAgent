#!/usr/bin/env python3
"""Stage 8 step 1 — materialise the PIT-safe SW1 sector aggregate panel.

Reads the silver price panel + sector_map, builds daily SW1 sector returns /
breadth / momentum / RS, writes parquet + a sanity report so we can eyeball
that sector returns look like real A-share sectors before searching signals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.data.sector.sector_panel import (  # noqa: E402
    SECTOR_COL, add_sector_signals, build_sector_panel,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {PANEL}")
    panel = pd.read_parquet(PANEL)
    smap = pd.read_parquet(SECTOR)
    print(f"  panel rows={len(panel):,} symbols={panel.symbol.nunique()} "
          f"dates {panel.trade_date.min().date()}..{panel.trade_date.max().date()}")
    print(f"  sector_map symbols={smap.symbol.nunique()} sectors={smap[SECTOR_COL].nunique()}")

    print("[build] sector aggregates ...")
    sp = build_sector_panel(panel, smap, min_members=5)
    print(f"  sector-day rows={len(sp):,} sectors={sp[SECTOR_COL].nunique()} "
          f"dates {sp.trade_date.min().date()}..{sp.trade_date.max().date()}")
    print("[build] signals ...")
    sp = add_sector_signals(sp)

    out = OUT_DIR / "sector_panel.parquet"
    sp.to_parquet(out, index=False)
    print(f"[write] {out}")

    # sanity report: per-sector full-sample CAGR (eqw, naive close-to-close)
    rep = {}
    full = sp[sp.trade_date >= "2010-01-01"]
    cagr = {}
    for sec, grp in full.groupby(SECTOR_COL):
        r = grp.sort_values("trade_date")["ret_eqw"].dropna()
        if len(r) < 250:
            continue
        nav = (1 + r).prod()
        yrs = len(r) / 244.0
        cagr[sec] = round(nav ** (1 / yrs) - 1, 4)
    cagr = dict(sorted(cagr.items(), key=lambda kv: kv[1], reverse=True))
    rep["full_sample_eqw_cagr_2010plus"] = cagr
    rep["n_sectors"] = sp[SECTOR_COL].nunique()
    rep["date_range"] = [str(sp.trade_date.min().date()), str(sp.trade_date.max().date())]
    (OUT_DIR / "sector_panel_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2))
    print("\n[sanity] per-SW1 eqw CAGR (2010+), top/bottom 8:")
    items = list(cagr.items())
    for k, v in items[:8]:
        print(f"   {k:<8} {v:+.1%}")
    print("   ...")
    for k, v in items[-8:]:
        print(f"   {k:<8} {v:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
