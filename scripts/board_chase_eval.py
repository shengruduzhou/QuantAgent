#!/usr/bin/env python3
"""打板 honest evaluation — what does board-chasing earn under break-only fills?

For every cached symbol-day whose minute bars touch the limit-up price:
  * reconstruct the board lifecycle (first touch / seal / breaks / close);
  * place the chase order at the FIRST SEAL, fill it with the honest queue
    model (filled only if the board breaks after the order);
  * exit at the NEXT day's open (T+1), full costs;
  * compare against the unfilled sealed boards — the unreachable alpha.

Conditioning cuts: 连板数 (prior consecutive limit-ups), seal-time bucket,
re-seal status. Output: runtime/reports/board_chase/{trades.csv, summary.json,
report.md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.board_fill_model import board_chase_fill, detect_board_day

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MINUTE_DIR = Path("runtime/data/v7/silver/minute_bars")


def seal_bucket(t: str | None) -> str:
    if not t:
        return "none"
    if t <= "10:00:00":
        return "early(<10:00)"
    if t <= "13:30:00":
        return "mid(10:00-13:30)"
    return "late(>13:30)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-12-15")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--commission-bps", type=float, default=2.5)
    ap.add_argument("--stamp-bps", type=float, default=5.0)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--output-dir", default="runtime/reports/board_chase")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    buy_cost = args.commission_bps / 1e4                      # passive limit buy: no slippage
    sell_cost = (args.commission_bps + args.stamp_bps + args.slippage_bps) / 1e4

    symbols = sorted(p.stem for p in MINUTE_DIR.glob("*.parquet"))
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "close",
                                            "is_st", "is_limit_up", "is_suspended"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["trade_date"] >= start - pd.Timedelta(days=40))
                  & (panel["trade_date"] <= end + pd.Timedelta(days=10))
                  & panel["symbol"].isin(symbols)].sort_values(["symbol", "trade_date"])
    g = panel.groupby("symbol", sort=False)
    panel["prev_close"] = g["close"].shift(1)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel["next_suspended"] = g["is_suspended"].shift(-1).fillna(False).astype(bool)
    # streak of consecutive limit-up closes ending the PREVIOUS day (0 ⇒ today would be 首板)
    lu = panel["is_limit_up"].fillna(False).astype(bool)
    streak = lu.groupby(panel["symbol"]).transform(
        lambda s: s.groupby((~s).cumsum()).cumsum())
    panel["prior_streak"] = streak.groupby(panel["symbol"]).shift(1).fillna(0).astype(int)
    pidx = panel.set_index(["symbol", "trade_date"])

    rows: list[dict] = []
    for i, sym in enumerate(symbols):
        path = MINUTE_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["trade_time", "open", "high", "low", "close", "volume"])
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        bars = bars[(bars["trade_time"] >= start) & (bars["trade_time"] <= end + pd.Timedelta(days=1))]
        for d, day_bars in bars.groupby(bars["trade_time"].dt.normalize()):
            try:
                meta = pidx.loc[(sym, d)]
            except KeyError:
                continue
            prev_close = float(meta["prev_close"]) if np.isfinite(meta["prev_close"]) else 0.0
            if prev_close <= 0 or len(day_bars) < 30:
                continue
            # cheap pre-filter: day high must reach the limit
            day_high = pd.to_numeric(day_bars["high"], errors="coerce").max()
            lim_approx = prev_close * (1.0 + 0.04)
            if not np.isfinite(day_high) or day_high < lim_approx:
                continue
            st = detect_board_day(day_bars, prev_close=prev_close, symbol=sym,
                                  is_st=bool(meta["is_st"]))
            if st is None or not st.touched:
                continue
            fill = board_chase_fill(st)
            rows.append({
                "symbol": sym, "trade_date": d,
                "limit_price": st.limit_price,
                "sealed": st.first_seal_time is not None,
                "seal_time": st.first_seal_time,
                "seal_bucket": seal_bucket(st.first_seal_time),
                "broke": st.broke_after_seal,
                "n_breaks": st.n_breaks,
                "closed_sealed": st.closed_sealed,
                "filled": fill.filled,
                "fill_reason": fill.reason,
                "prior_streak": int(meta["prior_streak"]),
                "next_open": float(meta["next_open"]) if np.isfinite(meta["next_open"]) else np.nan,
                "next_close": float(meta["next_close"]) if np.isfinite(meta["next_close"]) else np.nan,
                "next_suspended": bool(meta["next_suspended"]),
            })
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(symbols)} symbols, {len(rows)} board events", flush=True)

    if not rows:
        raise SystemExit("no board events found")
    df = pd.DataFrame(rows)
    df["entry"] = df["limit_price"] * (1 + buy_cost)
    df["net_ret_open"] = np.where(
        df["next_suspended"], np.nan,
        df["next_open"] * (1 - sell_cost) / df["entry"] - 1.0)
    df["net_ret_close"] = np.where(
        df["next_suspended"], np.nan,
        df["next_close"] * (1 - sell_cost) / df["entry"] - 1.0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "trades.csv", index=False)

    def agg(sub: pd.DataFrame) -> dict:
        s = sub["net_ret_open"].dropna()
        if s.empty:
            return {"n": int(len(sub)), "mean_open": None}
        return {
            "n": int(len(sub)),
            "mean_open": round(float(s.mean()), 5),
            "median_open": round(float(s.median()), 5),
            "win_rate_open": round(float((s > 0).mean()), 3),
            "mean_close": round(float(sub["net_ret_close"].dropna().mean()), 5),
            "t_stat": round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), 2)
            if len(s) > 2 and s.std(ddof=1) > 0 else None,
        }

    sealed = df[df["sealed"]]
    filled = sealed[sealed["filled"]]
    unfilled = sealed[~sealed["filled"]]
    summary = {
        "window": f"{args.start}..{args.end}",
        "n_touch_days": int(len(df)),
        "n_sealed": int(len(sealed)),
        "n_filled_on_break": int(len(filled)),
        "fill_rate_among_sealed": round(float(len(filled) / max(1, len(sealed))), 3),
        "filled_overall": agg(filled),
        "filled_resealed_into_close": agg(filled[filled["closed_sealed"]]),
        "filled_failed_board": agg(filled[~filled["closed_sealed"]]),
        "UNFILLED_sealed_boards (unreachable)": agg(unfilled),
        "filled_by_seal_bucket": {b: agg(g) for b, g in filled.groupby("seal_bucket")},
        "filled_by_prior_streak": {f"{int(k)}连板前": agg(g)
                                   for k, g in filled.groupby(filled["prior_streak"].clip(0, 3))},
        "costs": {"buy_bps": args.commission_bps,
                  "sell_bps": args.commission_bps + args.stamp_bps + args.slippage_bps},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    adv = None
    if summary["filled_overall"].get("mean_open") is not None \
            and summary["UNFILLED_sealed_boards (unreachable)"].get("mean_open") is not None:
        adv = summary["UNFILLED_sealed_boards (unreachable)"]["mean_open"] \
            - summary["filled_overall"]["mean_open"]
    lines = [
        "# 打板 honest-fill 评估（research only）", "",
        f"- 窗口 {args.start}..{args.end}；{len(df)} 触板日，{len(sealed)} 封板，"
        f"{len(filled)} 可成交（开板才轮到队尾）",
        f"- **逆向选择代价**: 封死不破的板（买不到）次日开盘 "
        f"{summary['UNFILLED_sealed_boards (unreachable)'].get('mean_open')}, "
        f"成交的板 {summary['filled_overall'].get('mean_open')}"
        + (f" → 差 {adv:+.4f}" if adv is not None else ""), "",
        "明细见 summary.json / trades.csv。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
