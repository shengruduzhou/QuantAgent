#!/usr/bin/env python3
"""Stage 7 PRODUCT: risk-controlled beta = multi-factor long-only tilt + drawdown overlay.

Honest conclusion of the alpha search: no robustly tradable selection alpha beats
beta in A-share long-only. So the realistic product targets the SAME beta return
with LOWER drawdown (higher Calmar), via:

  * base book = equal-weight tradable universe, OR a mild DEFENSIVE tilt
    (eqw of the better half by low-volatility + value — a quality/defensive tilt
    that stays broad, not a concentrated alpha bet);
  * a CAUSAL drawdown overlay = the canonical 200-day trend filter on a broad
    index (CSI500): full exposure when the index is above its 200d MA, reduced
    exposure otherwise. Minimal parameters, not tuned to this window, time-tested
    to cut drawdown.

Exposure for day t uses only data through t-1 (no look-ahead). Overlay switches
are charged a turnover cost. Reports full-period + per-regime CAGR / maxDD /
Calmar / Sharpe vs the raw basket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
FUND = "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
TRADING_DAYS = 252.0


def _metrics(daily: pd.Series) -> dict:
    d = daily.dropna()
    if len(d) < 20:
        return {}
    nav = (1 + d).cumprod()
    cagr = float(nav.iloc[-1] ** (TRADING_DAYS / len(d)) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    sd = float(d.std(ddof=0))
    sharpe = float(d.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else float("nan")
    return {"cagr": round(cagr, 4), "maxDD": round(dd, 4),
            "calmar": round(cagr / abs(dd), 3) if dd < 0 else float("nan"),
            "sharpe": round(sharpe, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--ma-days", type=int, default=200)
    ap.add_argument("--risk-off-exposure", type=float, default=0.3)
    ap.add_argument("--switch-cost-bps", type=float, default=8.0)
    ap.add_argument("--tilt-top-frac", type=float, default=0.5, help="defensive tilt keeps the better half by lowvol+value")
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--output-dir", default="runtime/stage7_riskcontrolled_beta")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "amount", "is_st", "is_suspended"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[panel["trade_date"] >= args.start].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel["ret"] = panel.groupby("symbol", sort=False)["close"].pct_change()
    elig = ~(panel["is_st"].fillna(False).astype(bool) | panel["is_suspended"].fillna(False).astype(bool))
    el = panel[elig].copy()

    # base book: equal-weight tradable universe daily return.
    eqw = el.groupby("trade_date")["ret"].mean()

    # defensive tilt: keep the better half by (low 20d vol) + (value book-to-price).
    el["vol20"] = el.groupby("symbol", sort=False)["ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    fund = pd.read_parquet(FUND)[["symbol", "available_at", "bps"]].copy()
    fund["available_at"] = pd.to_datetime(fund["available_at"], errors="coerce")
    fund = fund.dropna(subset=["available_at"]).sort_values("available_at")
    el = pd.merge_asof(el.sort_values("trade_date"), fund, left_on="trade_date", right_on="available_at", by="symbol", direction="backward")
    el["bp"] = pd.to_numeric(el["bps"], errors="coerce") / pd.to_numeric(el["close"], errors="coerce")
    g = el.groupby("trade_date")
    el["score"] = g["bp"].rank(pct=True) + (1.0 - g["vol20"].rank(pct=True))   # high value + low vol
    # tilt uses the PREVIOUS day's score (causal) to weight today's return.
    el["score_lag"] = el.groupby("symbol", sort=False)["score"].shift(1)
    thr = el.groupby("trade_date")["score_lag"].transform(lambda s: s.quantile(1 - args.tilt_top_frac))
    tilt = el[el["score_lag"] >= thr].groupby("trade_date")["ret"].mean()

    # CAUSAL 200d-MA trend overlay on CSI500.
    idx = pd.read_parquet(INDEX); idx["observation_date"] = pd.to_datetime(idx["observation_date"], errors="coerce")
    csi = idx[idx["label"] == "csi500"].sort_values("observation_date").set_index("observation_date")["close"]
    csi = pd.to_numeric(csi, errors="coerce")
    ma = csi.rolling(args.ma_days, min_periods=args.ma_days // 2).mean()
    above = (csi > ma)
    exposure = above.map({True: 1.0, False: args.risk_off_exposure}).shift(1)   # decide from t-1, apply at t
    exposure = exposure.reindex(eqw.index).ffill().fillna(1.0)

    def _overlay(book: pd.Series, exp: pd.Series) -> pd.Series:
        e = exp.reindex(book.index).ffill().fillna(1.0)
        switch_cost = e.diff().abs().fillna(0.0) * (args.switch_cost_bps / 1e4)
        return e * book - switch_cost

    books = {
        "eqw_basket_raw": eqw,
        "eqw_basket_overlay": _overlay(eqw, exposure),
        "defensive_tilt_raw": tilt,
        "defensive_tilt_overlay": _overlay(tilt, exposure),
    }

    # windows for per-regime view
    dates = sorted(eqw.index)
    windows = [(dates[i], dates[min(i + args.window_days, len(dates)) - 1]) for i in range(0, len(dates), args.window_days)]
    windows = [(s, e) for (s, e) in windows if (pd.Index(dates).get_indexer([e])[0] - pd.Index(dates).get_indexer([s])[0]) >= 40]

    full = {name: _metrics(b) for name, b in books.items()}
    print("=== FULL PERIOD (2018-2026) ===")
    for name, mtr in full.items():
        print(f"  {name:26} CAGR {mtr.get('cagr',0):+.2%}  maxDD {mtr.get('maxDD',0):+.2%}  Calmar {mtr.get('calmar')}  Sharpe {mtr.get('sharpe')}")

    # per-window maxDD comparison: does the overlay cut drawdown?
    perwin = []
    for (ws, we) in windows:
        row = {"window": f"{ws.date()}..{we.date()}"}
        for name, b in books.items():
            bw = b[(b.index >= ws) & (b.index <= we)]
            m = _metrics(bw)
            row[f"{name}_cagr"] = m.get("cagr"); row[f"{name}_maxDD"] = m.get("maxDD")
        perwin.append(row)
    pw = pd.DataFrame(perwin)
    pw.to_csv(out / "per_window.csv", index=False)

    # summary: overlay vs raw (basket) — DD reduction + Calmar gain
    base_raw, base_ovl = full["eqw_basket_raw"], full["eqw_basket_overlay"]
    summary = {"full_period": full,
               "overlay_vs_raw_basket": {
                   "cagr_delta": round(base_ovl["cagr"] - base_raw["cagr"], 4),
                   "maxDD_improvement": round(base_ovl["maxDD"] - base_raw["maxDD"], 4),  # less negative = better
                   "calmar_raw": base_raw["calmar"], "calmar_overlay": base_ovl["calmar"]},
               "config": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== OVERLAY VALUE (eqw basket) ===")
    print(f"  raw     : CAGR {base_raw['cagr']:+.2%}  maxDD {base_raw['maxDD']:+.2%}  Calmar {base_raw['calmar']}")
    print(f"  overlay : CAGR {base_ovl['cagr']:+.2%}  maxDD {base_ovl['maxDD']:+.2%}  Calmar {base_ovl['calmar']}")
    print(f"  → CAGR Δ {summary['overlay_vs_raw_basket']['cagr_delta']:+.2%}, maxDD better by {summary['overlay_vs_raw_basket']['maxDD_improvement']:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
