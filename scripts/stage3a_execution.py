#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 3A: execution-timing alpha for the daily rebalance (vs close-fill base).

The strict base sim fills at CLOSE. This asks: would filling the daily rebalance
legs at OPEN / VWAP / TWAP instead beat close-fill, net? Causal execution prices
per (symbol, day) from the minute panel: open=first-min close, close=last-min,
twap=mean-min close, vwap=sum(close*vol)/sum(vol) (realized execution benchmark).

Per traded leg, benefit-vs-close = (buy: (close-exec)/close ; sell: (exec-close)/close).
Positive => the method fills better than close. Annualized impact = sum over days
of (sum_legs |dW| * benefit), reported by method, on OOS 2025-09..12 and 2026.
Implementation shortfall = exec vs decision price (prev close).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/intraday_2026/intraday_panel_675.parquet"
S1 = "runtime/reports/v89_closed_loop/stage1"
OUT = Path("runtime/reports/v89_closed_loop/stage3a"); OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = {"non2026": ("2025-09-01", "2025-12-31"), "y2026": ("2026-01-02", "2026-05-13")}
ANN = 244


def exec_prices(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[["symbol", "trade_date", "trade_time", "close", "volume"]].sort_values(["symbol", "trade_date", "trade_time"])
    g = p.groupby(["symbol", "trade_date"])
    out = g.agg(open_px=("close", "first"), close_px=("close", "last"), twap=("close", "mean")).reset_index()
    vw = p.assign(cv=p["close"] * p["volume"].fillna(0)).groupby(["symbol", "trade_date"]).agg(
        cv=("cv", "sum"), v=("volume", "sum")).reset_index()
    vw["vwap"] = vw["cv"] / vw["v"].replace(0, np.nan)
    out = out.merge(vw[["symbol", "trade_date", "vwap"]], on=["symbol", "trade_date"], how="left")
    out["vwap"] = out["vwap"].fillna(out["twap"])
    return out


def main() -> int:
    if not Path(PANEL).exists():
        print(f"FATAL: {PANEL} not built."); return 1
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "trade_time", "close", "volume"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    ep = exec_prices(panel)
    report = {}
    for book in ("w210_k10", "w111_k5"):
        pos = pd.read_parquet(f"{S1}/daily_{book}_positions.parquet"); pos["trade_date"] = pd.to_datetime(pos["trade_date"])
        wide = pos.pivot_table(index="trade_date", columns="symbol", values="weight", fill_value=0.0).sort_index()
        dW = wide.diff().fillna(wide.iloc[[0]].reindex(wide.index).fillna(0.0))
        legs = dW.reset_index().melt(id_vars="trade_date", var_name="symbol", value_name="dw")
        legs = legs[legs["dw"].abs() > 1e-9]
        legs = legs.merge(ep, on=["symbol", "trade_date"], how="inner")
        legs["side"] = np.where(legs["dw"] > 0, "buy", "sell")
        bres = {}
        for w, (a, b) in WINDOWS.items():
            L = legs[(legs["trade_date"] >= pd.Timestamp(a)) & (legs["trade_date"] <= pd.Timestamp(b))].copy()
            if L.empty:
                bres[w] = {"n_legs": 0}; continue
            row = {"n_legs": int(len(L)), "fill_rate": round(float(L["close_px"].notna().mean()), 3)}
            for meth, col in [("open", "open_px"), ("vwap", "vwap"), ("twap", "twap")]:
                buy = L["side"] == "buy"
                benefit = np.where(buy, (L["close_px"] - L[col]) / L["close_px"], (L[col] - L["close_px"]) / L["close_px"])
                daily = (pd.Series(L["dw"].abs().to_numpy() * benefit, index=L.index)
                         .groupby(L["trade_date"]).sum())
                ndays = L["trade_date"].nunique()
                ann_impact = float(daily.sum() / max(1, ndays) * ANN)
                row[f"{meth}_vs_close_ann_bps"] = round(ann_impact * 1e4, 1)
                row[f"{meth}_mean_benefit_bps"] = round(float(np.nanmean(benefit) * 1e4), 2)
            bres[w] = row
        report[book] = bres
        for w in WINDOWS:
            r = bres[w]
            print(f"{book} {w}: legs={r.get('n_legs')} | open {r.get('open_vs_close_ann_bps','-')}bps "
                  f"vwap {r.get('vwap_vs_close_ann_bps','-')}bps twap {r.get('twap_vs_close_ann_bps','-')}bps (ann vs close-fill)", flush=True)
    (OUT / "stage3a_execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "stage3a_execution_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
