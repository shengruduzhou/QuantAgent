#!/usr/bin/env python3
"""Stage 10.4 — forward paper-trading from PIT snapshots (NO backfill).

`generate`  : from a day's PIT snapshot, build the concept portfolio per the
              user's rules — top 2-5 strongest sub-concepts (主升/启动/高低切),
              top 2-5 概念硬度 stocks each, EXCLUDING 伪概念 / 业绩不兑现 / 估值极端 /
              (when 10.3 lands) unverifiable-order names; dedup across concepts;
              freeze entry marks + timestamp into the paper ledger.
`update`    : mark every open frozen portfolio forward using later snapshots'
              prices, compute after-cost paper PnL vs benchmarks (all-A eqw /
              concept eqw / selected-concept eqw), append to the track record.

Forward-only: a portfolio is only ever marked with prices dated AFTER its entry.
Designed to run daily; accrues a 60-90 trading-day track record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("runtime/stage10_concept")
SNAPS = ROOT / "snapshots"
PT = ROOT / "paper_trades"
COST_BPS = 18.0  # round-trip-ish per rebalance, after-cost


def _snap_dates():
    return sorted(d.name for d in SNAPS.glob("*") if (d / "manifest.json").exists())


def _spot_prices(date: str) -> pd.Series:
    sp = pd.read_parquet(SNAPS / date / "spot_all.parquet")
    sp["代码"] = sp["代码"].astype(str).str.zfill(6)
    px = pd.to_numeric(sp.set_index("代码")["最新价"].astype(str).replace({"-": np.nan}), errors="coerce")
    return px.dropna()


def generate(date: str, *, n_concepts: int, n_per: int, weighting: str):
    snap = SNAPS / date
    strength = pd.read_csv(snap / "concept_strength.csv")
    # prefer the 10.3 order-verified hardness if available
    hf = snap / "concept_hardness_verified.csv"
    hard = pd.read_csv(hf if hf.exists() else snap / "concept_hardness.csv")
    if "order_label" not in hard.columns:
        hard["order_label"] = "unverified"
    px = _spot_prices(date)

    concepts = strength[strength["state"].isin(["主升", "启动", "高低切"])].head(n_concepts)
    picks = []
    for _, cc in concepts.iterrows():
        sub = hard[hard["board"] == cc["board"]].copy()
        # exclusions per user rule
        sub = sub[sub["role"] != "伪概念"]
        sub = sub[~sub["name"].astype(str).str.contains("ST", case=False, na=False)]  # ST/*ST untradable
        sub = sub[~(sub["profit_yoy"].fillna(0) < -20)]      # 业绩不兑现
        sub = sub[~(sub["score_risk"].fillna(0) <= -6)]      # 估值极端
        sub = sub[~sub["order_label"].fillna("unverified").isin(
            ["fake_concept", "rumor_only", "performance_mismatch"])]  # 伪/传闻/概念强但业绩不兑现
        sub = sub.sort_values("hardness", ascending=False).head(n_per)
        for _, s in sub.iterrows():
            picks.append({"code": str(s["code"]).zfill(6), "name": s["name"], "concept": cc["board"],
                          "industry": cc["industry"], "segment": cc["segment"],
                          "concept_state": cc["state"], "concept_strength": round(cc["strength"], 2),
                          "role": s["role"], "hardness": round(s["hardness"], 1)})
    pf = pd.DataFrame(picks)
    if pf.empty:
        print("no qualifying picks"); return None
    # dedup: keep highest-hardness instance of a stock
    pf = pf.sort_values("hardness", ascending=False).drop_duplicates("code", keep="first")
    # weight
    if weighting == "equal":
        pf["weight"] = 1.0 / len(pf)
    else:  # hardness x concept-strength
        raw = pf["hardness"].clip(lower=0) * (pf["concept_strength"] - pf["concept_strength"].min() + 1)
        pf["weight"] = raw / raw.sum()
    pf["entry_price"] = pf["code"].map(px)
    pf = pf.dropna(subset=["entry_price"])
    pf["weight"] = pf["weight"] / pf["weight"].sum()

    PT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(PT / f"portfolio_{date}.csv", index=False)
    ledger = json.loads((PT / "ledger.json").read_text()) if (PT / "ledger.json").exists() else []
    if not any(e["entry_date"] == date for e in ledger):
        ledger.append({"entry_date": date, "n_stocks": len(pf), "n_concepts": int(pf["concept"].nunique()),
                       "weighting": weighting, "frozen_at": pd.Timestamp.now().isoformat(),
                       "concepts": pf["concept"].unique().tolist()})
        (PT / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
    print(f"[generate] {date}: {len(pf)} stocks / {pf['concept'].nunique()} concepts ({weighting})")
    with pd.option_context("display.width", 200, "display.unicode.east_asian_width", True):
        print(pf[["name", "concept", "segment", "role", "hardness", "entry_price", "weight"]].to_string(index=False))
    return pf


def update():
    """Roll the continuous daily-rebalanced strategy NAV forward + beta/alpha
    decompose it vs 4 benchmarks (all-A / same-universe / selected-concept /
    concept-index). Writes daily track + a beta-aware summary. Forward-only.

    plain-v8.9 is not markable forward (v8.9 score panel ends 2026-05-07);
    labeled n/a until a daily v8.9 inference feed exists.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quantagent.concept import paper_track as ptk
    HIST = ROOT / "raw" / "board_hist"
    res = ptk.build_track(SNAPS, PT, HIST)
    (PT / "track_summary.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "daily"}, ensure_ascii=False, indent=2, default=str))
    if res["status"] != "ok":
        print(f"=== paper-trade: {res['status']} ({res.get('snapshot_days',0)} snapshot day(s); "
              f"{res.get('need','')}) ===")
        print("  metrics activate once >=2 forward snapshot days accrue (beta/alpha need ~20d).")
        return
    res["daily"].to_csv(PT / "track_record.csv")
    p = res["panel"]
    print(f"=== paper-trade track ({res['days']} forward day(s)) — beta-aware ===")
    print(f"  absolute CAGR={p['cagr']:+.1%}  MaxDD={p['maxdd']:.1%}  Calmar={p['calmar']}  "
          f"Sharpe={p['sharpe']}  turnover={p['turnover']}  gross={res['gross_exposure']}")
    print(f"  vs all-A         : beta={p['beta_all_a']} alpha={p['alpha_all_a']:+.1%} excess={p['excess_all_a']:+.1%}")
    print(f"  vs same-universe : beta={p['beta_same_universe']} alpha={p['alpha_same_universe']:+.1%} excess={p['excess_same_universe']:+.1%}")
    print(f"  vs sel-concept   : beta={p['beta_selected_concept']} alpha={p['alpha_selected_concept']:+.1%} excess={p['excess_selected_concept']:+.1%}")
    print(f"  vs concept-index : beta={p['beta_concept_index']} alpha={p['alpha_concept_index']:+.1%} excess={p['excess_concept_index']:+.1%}")
    print(f"  plain-v8.9       : n/a (v8.9 panel ends 2026-05-07; needs daily inference)")
    print(f"  concept exposure : {res['concept_breakdown']}")
    print(f"  top contribution : {[(t['name'], round(t['contrib'],4)) for t in res['top_contribution']]}")
    if p["cagr"] and (p.get("alpha_all_a") or 0) <= 0.03 and (p.get("beta_all_a") or 0) > 0.8:
        print("  >> READ: return looks beta-driven (high beta to all-A, low beta-adjusted alpha)")
    elif (p.get("alpha_all_a") or 0) > 0.03:
        print("  >> READ: positive beta-adjusted alpha vs all-A")
    print(f"\n[write] {PT/'track_record.csv'} , {PT/'track_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["generate", "update", "both"], default="both")
    ap.add_argument("--date", default=None, help="snapshot date YYYYMMDD (default latest)")
    ap.add_argument("--n-concepts", type=int, default=4)
    ap.add_argument("--n-per", type=int, default=4)
    ap.add_argument("--weighting", default="hardness_strength", choices=["equal", "hardness_strength"])
    args = ap.parse_args()
    date = args.date or (_snap_dates()[-1] if _snap_dates() else None)
    if date is None:
        print("no snapshots; run stage10_daily_scan first"); return 1
    if args.action in ("generate", "both"):
        generate(date, n_concepts=args.n_concepts, n_per=args.n_per, weighting=args.weighting)
    if args.action in ("update", "both"):
        update()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
