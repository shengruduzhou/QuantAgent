#!/usr/bin/env python3
"""Forward paper-trade ledger — the ONLY truly hindsight-free OOS test.

gemma is a 2026 model, so any historical as-of (even "2026-03 picking April")
risks parametric memory.  The clean answer is to FREEZE today's pools and score
them only with prices that did not exist when we froze them.

Two modes:
  freeze  : append today's factor / chain / union pools (symbols + as_of +
            frozen_at wall-clock) to a ledger.  Each weekly re-run freezes a new
            row; the chained buy-hold of weekly rows == weekly rebalance.
  settle  : for ledger rows whose forward window has fully elapsed AND has
            prices, compute realized equal-weight return, benchmark (eqw all-A)
            and excess.  Never settles a row whose forward end > last price date
            (that would be lookahead).

Ledger: runtime/reports/forward_live/ledger.parquet
This is a research ledger, not an order book; it places no trades.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEDGER = Path("runtime/reports/forward_live/ledger.parquet")
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def _code6(s: str) -> str:
    return str(s).split(".")[0].zfill(6)


def _load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_parquet(LEDGER)
    return pd.DataFrame(columns=[
        "as_of", "frozen_at", "sleeve", "fw", "symbols", "n", "fwd_td",
        "fwd_end", "pool_ret", "bench_ret", "excess", "settled"])


def _save_ledger(df: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER, index=False)


def _pred_score_col(df: pd.DataFrame) -> str:
    for c in ("alpha_score", "prediction", "composite_score", "score"):
        if c in df.columns:
            return c
    raise ValueError("predictions need alpha_score/prediction/score")


def freeze(args) -> int:
    led = _load_ledger()
    as_of = args.as_of
    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    factor_syms: list[str] = []
    if args.predictions and Path(args.predictions).exists():
        p = pd.read_parquet(args.predictions)
        p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.normalize()
        sc = _pred_score_col(p)
        day = p[p["trade_date"] == pd.Timestamp(as_of)]
        if day.empty:  # fall back to the latest available date <= as_of
            avail = day = p[p["trade_date"] <= pd.Timestamp(as_of)]
            if not avail.empty:
                day = avail[avail["trade_date"] == avail["trade_date"].max()]
        factor_syms = day.sort_values(sc, ascending=False)["symbol"].astype(str).head(args.n_factor).tolist()

    chain_syms: list[str] = []
    if args.chain_pool and Path(args.chain_pool).exists():
        cp = pd.read_parquet(args.chain_pool)
        for sccol in ("chain_conviction", "mix_score", "conviction"):
            if sccol in cp.columns:
                cp = cp.sort_values(sccol, ascending=False)
                break
        if "source" in cp.columns and (cp["source"].astype(str).str.contains("LLM|产业链|chain", regex=True)).any():
            cp = cp[cp["source"].astype(str).str.contains("LLM|产业链|chain", regex=True)]
        chain_syms = cp["symbol"].astype(str).head(args.n_chain).tolist()

    union_syms = list(dict.fromkeys(factor_syms + chain_syms))

    for sleeve, syms, fw in (("factor", factor_syms, 1.0), ("chain", chain_syms, 0.0), ("union", union_syms, args.fw)):
        if not syms:
            print(f"  [{sleeve}] empty — not frozen (missing input?)")
            continue
        rows.append({"as_of": as_of, "frozen_at": now, "sleeve": sleeve, "fw": float(fw),
                     "symbols": json.dumps(syms, ensure_ascii=False), "n": len(syms),
                     "fwd_td": int(args.fwd_td), "fwd_end": None, "pool_ret": None,
                     "bench_ret": None, "excess": None, "settled": False})
    if not rows:
        print("nothing frozen (no factor preds and no chain pool found)"); return 1
    led = pd.concat([led, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
        ["as_of", "sleeve"], keep="last")
    _save_ledger(led)
    print(f"froze {len(rows)} sleeves at as_of={as_of}: " + ", ".join(f"{r['sleeve']}({r['n']})" for r in rows))
    print(f"ledger: {LEDGER} ({len(led)} rows)")
    return 0


def settle(args) -> int:
    led = _load_ledger()
    if led.empty:
        print("empty ledger"); return 0
    panel = pd.read_parquet(args.panel, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    tds = sorted(panel["trade_date"].unique())
    last_px = tds[-1]

    n_settled = 0
    for i, r in led.iterrows():
        if bool(r.get("settled")):
            continue
        as_of = pd.Timestamp(r["as_of"])
        future = [t for t in tds if t > as_of][: int(r["fwd_td"])]
        if len(future) < int(r["fwd_td"]) or future[-1] > last_px:
            continue  # window not fully elapsed → would be lookahead; skip
        end = future[-1]
        c0 = panel[panel.trade_date == as_of].set_index("symbol")["close"]
        c1 = panel[panel.trade_date == end].set_index("symbol")["close"]
        fwd = (c1 / c0 - 1.0).dropna()
        if fwd.empty:
            continue
        bench = float(fwd.mean())
        syms = [str(s) for s in json.loads(r["symbols"])]
        pr = fwd.reindex(syms).dropna()
        pool = float(pr.mean()) if not pr.empty else float("nan")
        led.at[i, "fwd_end"] = str(end.date())
        led.at[i, "pool_ret"] = round(pool, 6)
        led.at[i, "bench_ret"] = round(bench, 6)
        led.at[i, "excess"] = round(pool - bench, 6)
        led.at[i, "settled"] = True
        n_settled += 1
    _save_ledger(led)

    done = led[led["settled"] == True]  # noqa: E712
    print(f"settled {n_settled} new rows; ledger has {len(done)}/{len(led)} settled.")
    if not done.empty:
        agg = done.groupby("sleeve")["excess"].agg(["mean", "min", "count"])
        print("\n=== forward-live realized excess by sleeve ===")
        for sleeve, g in agg.iterrows():
            print(f"  {sleeve:<7} mean {g['mean']:+.4%}  worst {g['min']:+.4%}  n={int(g['count'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="append today's pools to the ledger")
    f.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    f.add_argument("--predictions", default="", help="latest factor preds parquet (alpha_score/prediction)")
    f.add_argument("--chain-pool", default="", help="live chain pool parquet for as_of")
    f.add_argument("--n-factor", type=int, default=20)
    f.add_argument("--n-chain", type=int, default=15)
    f.add_argument("--fw", type=float, default=0.6)
    f.add_argument("--fwd-td", type=int, default=5, help="forward holding (trading days) before settle")
    f.set_defaults(func=freeze)

    s = sub.add_parser("settle", help="score elapsed ledger rows with realized prices")
    s.add_argument("--panel", default=PANEL)
    s.set_defaults(func=settle)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
