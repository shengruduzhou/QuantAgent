#!/usr/bin/env python3
"""做T (T+0) intraday signals for a held pool, from TickFlow 1-minute bars.

For each stock in the pool, pull TickFlow 1-minute K (paid tier serves recent
intraday), compute 分时 features (VWAP / 日内位置 / 主动买卖 / 集合竞价缺口 /
放量异动), derive 做T band levels (加T buy_below, 减T sell_above), and overlay the
defensive microstructure guard (避开砸盘/压盘/对倒). Research output only — no orders.

Typical use (live, current trading day):
  intraday_dot_signals.py --pool runtime/reports/monthly/chain_pool_2026-04-30.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from quantagent.execution.intraday_features import features_frame
from quantagent.risk.microstructure_guard import microstructure_guard

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def _tf_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow
    return tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                             base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)


def _prev_close_map(symbols: list[str], as_of: str | None) -> dict[str, float]:
    try:
        mp = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
        mp["trade_date"] = pd.to_datetime(mp["trade_date"], errors="coerce")
        if as_of:
            mp = mp[mp["trade_date"] < pd.Timestamp(as_of)]
        mp = mp[mp["symbol"].isin(symbols)].sort_values("trade_date").groupby("symbol").tail(1)
        return {str(s): float(c) for s, c in zip(mp["symbol"], mp["close"]) if pd.notna(c)}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=None, help="parquet with a 'symbol' column")
    ap.add_argument("--symbols", default=None, help="comma-separated symbols (overrides --pool)")
    ap.add_argument("--as-of", default=None, help="target trading day (default: latest in the pulled bars)")
    ap.add_argument("--count", type=int, default=260, help="1-min bars to pull per symbol (≈1 session=240)")
    ap.add_argument("--out", type=Path, default=Path("runtime/reports/daily"))
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.pool and args.pool.exists():
        symbols = [str(s) for s in pd.read_parquet(args.pool)["symbol"].tolist()]
    else:
        raise SystemExit("provide --symbols or --pool")
    symbols = list(dict.fromkeys(symbols))

    tf = _tf_client()
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    target_day = args.as_of
    for sym in symbols:
        try:
            k = tf.klines.intraday(sym, period="1m", count=args.count, as_dataframe=True)
        except Exception as exc:
            print(f"  [{sym}] intraday err: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if k is None or getattr(k, "empty", True) or "trade_date" not in k.columns:
            continue
        day = target_day or str(pd.to_datetime(k["trade_date"]).max().date())
        target_day = target_day or day
        bars_by_symbol[sym] = k[k["trade_date"].astype(str) == day].copy()

    if not bars_by_symbol:
        raise SystemExit("no intraday bars pulled (market closed / wrong day / token?)")

    feats = features_frame(bars_by_symbol, prev_close=_prev_close_map(symbols, target_day))
    if feats.empty:
        raise SystemExit("no features computed")
    guarded = microstructure_guard(feats)

    # executable 做T action per stock (CAUSAL FSM on bars-so-far; held pool → in_position=True)
    from quantagent.execution.intraday_dot_strategy import live_dot_action
    act = {s: live_dot_action(b, symbol=s, in_position=True) for s, b in bars_by_symbol.items()}
    guarded["dot_exec"] = guarded["symbol"].map(lambda s: act.get(s, {}).get("action", "观望"))
    guarded["dot_level"] = guarded["symbol"].map(lambda s: act.get(s, {}).get("level"))

    cols = ["symbol", "trade_date", "last", "vwap", "intraday_range_pos", "net_buy_pressure",
            "open_auction_gap", "dot_exec", "dot_level", "dot_bias", "buy_below", "sell_above",
            "guard_action", "sweep_dump_risk"]
    out = guarded[[c for c in cols if c in guarded.columns]].sort_values(
        ["guard_action", "dot_bias"], ascending=[True, True])

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"intraday_dot_{target_day}".replace("-", "")
    out.to_csv(args.out / f"{stem}.csv", index=False)

    md = [f"# 做T 分时信号 — {target_day}", "",
          "*TickFlow 1分钟K → VWAP/日内位置/主动买卖/集合竞价 → 做T买卖带 + 盘口防护。研究参考,非交易指令。*", "",
          "| 代码 | 现价 | VWAP | 日内位置 | 主动买卖 | 竞价缺口 | 做T偏向 | 加T≤ | 减T≥ | 盘口 |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for _, r in out.iterrows():
        md.append(f"| {r['symbol']} | {r.get('last','-')} | {r.get('vwap','-')} | "
                  f"{r.get('intraday_range_pos','-'):.2f} | {r.get('net_buy_pressure','-'):.2f} | "
                  f"{r.get('open_auction_gap',0)*100:.2f}% | {r.get('dot_bias','-')} | "
                  f"{r.get('buy_below','-')} | {r.get('sell_above','-')} | {r.get('guard_action','-')} |")
    md += ["", "## 用法",
           "- **加T**: 偏多做T + 盘口ok 的票, 价格回到 `加T≤` 价位附近分批买回核心仓的T。",
           "- **减T**: 偏空做T 或盘口 caution/avoid 的票, 价格冲到 `减T≥` 价位附近减T锁利。",
           "- **避险**: guard_action=avoid (砸盘/压盘风险高) 当日不加仓, 优先减。"]
    (args.out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"做T signals for {len(out)} stocks @ {target_day} → {args.out / (stem + '.md')}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
