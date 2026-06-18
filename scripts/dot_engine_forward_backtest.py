#!/usr/bin/env python3
"""Forward 反T backtest of the NEW conf-gated 做T engine on the paper book.

The earlier ``dot_overlay_backtest.py`` validated the OLDER parametric
``intraday_dot_strategy`` (fixed target/stop) and found a net drag (~-14%/yr):
it entered ~87% of days at a 25% hit rate, so the ~26bps round-trip cost won.

This harness gives the PRODUCTION engine its own fair test: per (trade_date,
held symbol), it walks the day's real 1-minute bars and drives the actual
``compute_intraday_state`` + ``decide`` at a 5-minute cadence, executing the
反T (sell held-position high on SELL_HIGH @ conf>=75, buy it back low on
BUY_BACK) under T+1 (only the carried昨仓 sellable_qty is ever sold) and the
engine's built-in cost-edge gate. Any net-sold quantity is restored at the
close so the overnight position matches buy-and-hold; the incremental P&L vs
buy-and-hold is the 做T uplift.

Research only. No order intents. Outputs <out>/{trades.csv, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_protocol import _regime_label  # noqa: E402  (PIT 60d-trail regime)

from quantagent.execution.intraday_dot_engine import compute_intraday_state  # noqa: E402
from quantagent.execution.intraday_dot_decision import DecisionConfig, Position, decide  # noqa: E402

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MINUTE_DIR = Path("runtime/data/v7/silver/minute_bars")
ANN = 244


def _limit_band(symbol: str) -> float:
    s = str(symbol)
    if s.startswith(("30", "68")):  # ChiNext / STAR — 20% daily band
        return 0.20
    if s.startswith(("8", "4")):    # BSE — 30%
        return 0.30
    return 0.10


def _bench_regime() -> dict:
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    px = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    bench = px.pct_change(fill_method=None).mean(axis=1).dropna()
    lab = _regime_label(bench)
    return {pd.Timestamp(d).normalize(): r for d, r in lab.items()}


def _simulate_day(day_bars: pd.DataFrame, pre_close: float, symbol: str, *,
                  cfg: DecisionConfig, cadence: int, buy_cost: float, sell_cost: float,
                  engine_params=None) -> dict | None:
    """反T on a 1.0-notional held position; returns incremental uplift vs buy&hold."""
    b = day_bars.sort_values("trade_time").reset_index(drop=True)
    if len(b) < 30 or pre_close <= 0:
        return None
    band = _limit_band(symbol)
    limit_up = round(pre_close * (1 + band), 2)
    limit_down = round(pre_close * (1 - band), 2)
    close_px = float(b["close"].iloc[-1])

    held0 = 10000.0            # carried 昨仓 shares (all T+1-sellable); 1.0 notional ≈ held0*pre_close
    sold_total = 0.0           # of carried shares sold high
    bought_total = 0.0         # rebought low → become today_buy (NOT re-sellable today, T+1)
    cash = 0.0
    open_pair: str | None = None
    sold_legs = 0
    bought_legs = 0
    times = b["trade_time"].dt.strftime("%H:%M").tolist()

    for i in range(20, len(b), max(1, cadence)):
        state = compute_intraday_state(b.iloc[: i + 1], pre_close=pre_close, params=engine_params)
        if state is None:
            continue
        sellable = max(0.0, held0 - sold_total)          # only unsold carried shares
        today_buy = bought_total                          # rebought shares — cannot be sold today
        total = sellable + today_buy
        pos = Position(sellable_qty=int(sellable), today_buy_qty=int(today_buy), total_qty=int(total))
        d = decide(state, pos, symbol=symbol, current_time=times[i], pre_close=pre_close,
                   limit_up=limit_up, limit_down=limit_down, cash=cash,
                   config=cfg, pair_id_hint=open_pair)
        act = d.get("action")
        px = float(d.get("limit_price") or 0.0)
        qty = float(d.get("qty") or 0.0)
        if act == "SELL_HIGH" and qty > 0 and px > 0 and sellable >= qty:
            sold_total += qty
            cash += qty * px * (1 - sell_cost)
            open_pair = d.get("t_pair_id") or open_pair or f"{symbol}-{times[i]}-T"
            sold_legs += 1
        elif act == "BUY_BACK" and qty > 0 and px > 0 and cash >= qty * px * (1 + buy_cost):
            bought_total += qty
            cash -= qty * px * (1 + buy_cost)
            bought_legs += 1
            open_pair = None

    if sold_legs == 0:
        return {"executed": False, "uplift": 0.0, "sold_legs": 0, "bought_legs": 0}

    # Restore overnight position to held0 at close (apples-to-apples vs buy&hold of held0).
    net_shares = held0 - sold_total + bought_total
    if net_shares < held0:
        cash -= (held0 - net_shares) * close_px * (1 + buy_cost)
    elif net_shares > held0:
        cash += (net_shares - held0) * close_px * (1 - sell_cost)

    # After restoring to held0 shares overnight, the incremental vs buy&hold is exactly cash.
    uplift = cash / (held0 * pre_close)
    return {"executed": True, "uplift": float(uplift), "sold_legs": sold_legs, "bought_legs": bought_legs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdings-csv", default="runtime/paper/replay_2026/holdings_daily.csv")
    ap.add_argument("--output-dir", default="runtime/reports/v8/discovery/dot_engine_forward")
    ap.add_argument("--regimes", default="bull,sideways")
    ap.add_argument("--cadence", type=int, default=5, help="minutes between decision evaluations")
    ap.add_argument("--commission-bps", type=float, default=2.5)
    ap.add_argument("--stamp-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--conf-execute", type=float, default=75.0)
    ap.add_argument("--score-enter", type=float, default=None, help="override DotEngineParams.score_enter (高/低 A-signal threshold; default 80)")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    from quantagent.execution.intraday_dot_engine import DotEngineParams
    engine_params = None
    if args.score_enter is not None:
        _f = DotEngineParams.__dataclass_fields__
        kw = {"score_enter": args.score_enter}
        if "score_strong" in _f:
            kw["score_strong"] = max(args.score_enter + 6.0, args.score_enter)
        engine_params = DotEngineParams(**kw)

    buy_cost = (args.commission_bps + args.slippage_bps) / 1e4
    sell_cost = (args.commission_bps + args.stamp_bps + args.slippage_bps) / 1e4
    cfg = DecisionConfig(conf_execute=args.conf_execute) if "conf_execute" in DecisionConfig.__dataclass_fields__ else DecisionConfig()
    active = {r.strip() for r in args.regimes.split(",") if r.strip()}

    holdings = pd.read_csv(args.holdings_csv)
    holdings["trade_date"] = pd.to_datetime(holdings["trade_date"])
    holdings["symbol"] = holdings["symbol"].astype(str)
    regime = _bench_regime()

    rows: list[dict] = []
    n = 0
    for sym in sorted(holdings["symbol"].unique()):
        p = MINUTE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        bars = pd.read_parquet(p)
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        bars["d"] = bars["trade_time"].dt.normalize()
        by_day = dict(tuple(bars.groupby("d")))
        prev_close_by_day = {}
        for sub in holdings[holdings["symbol"] == sym].itertuples():
            d = pd.Timestamp(sub.trade_date).normalize()
            rg = regime.get(d, "sideways")
            if rg not in active:
                continue
            day_bars = by_day.get(d)
            if day_bars is None:
                continue
            # pre_close = last close strictly before d in the minute cache
            prior_days = [x for x in by_day if x < d]
            if not prior_days:
                continue
            pre_close = float(by_day[max(prior_days)]["close"].iloc[-1])
            res = _simulate_day(day_bars, pre_close, sym, cfg=cfg, cadence=args.cadence,
                                buy_cost=buy_cost, sell_cost=sell_cost, engine_params=engine_params)
            if res is None:
                continue
            rows.append({"trade_date": d, "symbol": sym, "weight": float(sub.weight), "regime": rg, **res})
            n += 1
            if args.max_pairs and n >= args.max_pairs:
                break
        if args.max_pairs and n >= args.max_pairs:
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(rows)
    if trades.empty:
        raise SystemExit("no pairs evaluated")
    trades.to_csv(out_dir / "trades.csv", index=False)

    ex = trades[trades["executed"]]
    # Weight uplift within each day by holding weight → daily book uplift.
    _tmp = trades.assign(wu=trades["uplift"] * trades["weight"]).groupby("trade_date")
    daily = _tmp["wu"].sum() / _tmp["weight"].sum().clip(lower=1e-9)
    daily_uplift_mean = float(daily.mean())
    summary = {
        "pairs": int(len(trades)),
        "executed_legs": int(len(ex)),
        "entry_rate": round(float(len(ex) / max(len(trades), 1)), 4),
        "sold_legs_total": int(trades["sold_legs"].sum()),
        "bought_legs_total": int(trades["bought_legs"].sum()),
        "avg_uplift_per_executed": round(float(ex["uplift"].mean()) if len(ex) else 0.0, 6),
        "daily_uplift_mean_bps": round(daily_uplift_mean * 1e4, 3),
        "annualized_uplift": round((1 + daily_uplift_mean) ** ANN - 1, 4),
        "days_covered": int(daily.shape[0]),
        "conf_execute": args.conf_execute,
        "cadence_min": args.cadence,
        "cost_roundtrip_bps": round((buy_cost + sell_cost) * 1e4, 2),
        "by_regime": {
            rg: {"pairs": int((trades["regime"] == rg).sum()),
                 "executed": int(((trades["regime"] == rg) & trades["executed"]).sum()),
                 "avg_uplift": round(float(trades.loc[trades["regime"] == rg, "uplift"].mean()), 6)}
            for rg in sorted(active)
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
