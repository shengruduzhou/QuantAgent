#!/usr/bin/env python3
"""Search for the best risk-adjusted strategy, without fooling ourselves.

The request is "find the optimal strategy: maximise excess and annualised
return, minimise drawdown". The hard part is not the search -- it is that
searching N configurations and reporting the winner's out-of-sample number is
itself a form of overfitting. The winner is the maximum of N noisy draws, so its
OOS figure is biased upward by construction, and quoting it as an expectation is
the single most common way a backtest lies.

Three guards, all measured rather than asserted:

1. NESTED SELECTION. Configs are ranked using ONLY windows whose test year is
   <= `--select-until`. The winner is then reported on the later windows, which
   the selection never saw. That later number is the one worth believing.
2. MULTIPLE-TESTING PENALTY. The selection-stage Sharpe is deflated for the
   number of configs tried, via the repo's own
   `probability_of_backtest_overfitting` and an E[max] adjustment. A winner that
   does not survive its own search is reported as such.
3. BENCHMARK IS EXPLICIT. Excess return is measured against the equal-weight
   investable universe of the same window -- not cash, and not a curated index.
   A long-only A-share book that merely tracks its universe has no excess, and
   an equal-weight benchmark is the honest opponent.

Signals are intentionally simple and few. A large grid would buy a higher
apparent maximum and a worse deflated one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.quant_math.purged_cv import probability_of_backtest_overfitting  # noqa: E402

PANEL = Path("data/raw/ashare_daily_full/panel_all.parquet")
OUT = Path("runtime/strategy_search")
BREADTH_FLOOR = 300

COMMISSION_BPS, STAMP_BPS, SLIP_BPS = 2.5, 50.0, 8.0


def _cost(turnover: float, sell: float) -> float:
    return (turnover * (COMMISSION_BPS + SLIP_BPS) + sell * STAMP_BPS) / 1e4


def _signal(hist: pd.DataFrame, kind: str, lookback: int) -> pd.Series:
    """All signals use ONLY data up to and including the rebalance bar."""
    if len(hist) <= lookback:
        return pd.Series(dtype=float)
    px = hist.iloc[-lookback - 1:]
    ret = px.iloc[-1] / px.iloc[0] - 1.0
    if kind == "momentum":
        s = ret
    elif kind == "reversal":
        s = -ret
    elif kind == "lowvol":
        s = -px.pct_change().std(ddof=1)          # prefer calm names
    elif kind == "mom_x_lowvol":
        v = px.pct_change().std(ddof=1)
        s = ret / v.replace(0.0, np.nan)          # risk-adjusted momentum
    else:
        raise ValueError(kind)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def run_window(tr: pd.DataFrame, te: pd.DataFrame, cfg: dict) -> dict:
    liq = tr.groupby("symbol")["amount"].median()
    tradable = set(liq[liq >= cfg["min_amount"]].index)

    tr_px = tr.pivot_table(index="trade_date", columns="symbol", values="close").sort_index()
    train_vol = tr_px.pct_change().mean(axis=1).std(ddof=1) * np.sqrt(252)
    vol_scale = 1.0
    if np.isfinite(train_vol) and train_vol > 1e-9:
        vol_scale = float(np.clip(cfg["target_vol"] / train_vol, 0.1, 1.0))

    px = te.pivot_table(index="trade_date", columns="symbol", values="close").sort_index()
    keep = [c for c in px.columns if c in tradable]
    if len(keep) < 20:
        return {"status": "too_few_tradable"}
    px = px[keep]
    lb = cfg["lookback"]
    if len(px) < lb + 10:
        return {"status": "too_short"}

    hist = tr_px.reindex(columns=px.columns).sort_index().tail(lb + 5)
    full = pd.concat([hist, px]).sort_index()

    nav, peak = 1.0, 1.0
    w = pd.Series(0.0, index=px.columns)
    navs, bench = [], []
    for i, d in enumerate(px.index):
        if i % cfg["rebal"] == 0:
            sig = _signal(full.loc[:d], cfg["signal"], lb)
            if len(sig) >= 2:
                picks = sig.nlargest(min(cfg["top_k"], len(sig))).index
                nw = pd.Series(0.0, index=px.columns)
                nw[picks] = 1.0 / len(picks)
                nw = nw.clip(upper=cfg["max_name"])
                dd = 1.0 - nav / max(peak, 1e-12)
                brake = cfg["brake_scale"] if dd > cfg["brake_dd"] else 1.0
                nw = nw * vol_scale * brake
                g = nw.abs().sum()
                if g > 1.0:
                    nw = nw / g
                turn = float((nw - w).abs().sum())
                sell = float((w - nw).clip(lower=0).sum())
                nav *= 1.0 - _cost(turn, sell)
                w = nw
        if i + 1 < len(px):
            r = (px.iloc[i + 1] / px.iloc[i] - 1.0).replace([np.inf, -np.inf], np.nan)
            live = w[r.notna()]
            if live.sum() > 1e-12:
                nav *= 1.0 + float((live * r[r.notna()]).sum())
            # Equal-weight investable universe: the honest long-only opponent.
            bench.append(float(r.dropna().mean()) if r.notna().any() else 0.0)
        peak = max(peak, nav)
        navs.append(nav)

    s = pd.Series(navs, index=px.index)
    r = s.pct_change().dropna()
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    mdd = float((s / s.cummax() - 1.0).min())
    bench_total = float(np.prod([1 + b for b in bench]) - 1.0) if bench else float("nan")
    total = float(s.iloc[-1] - 1.0)
    return {
        "status": "ok",
        "total_return": total,
        "benchmark_return": bench_total,
        "excess_return": total - bench_total,
        "cagr": float(s.iloc[-1] ** (1 / yrs) - 1.0) if s.iloc[-1] > 0 else float("nan"),
        "max_drawdown": mdd,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else float("nan"),
        "calmar": float((s.iloc[-1] ** (1 / yrs) - 1.0) / abs(mdd)) if mdd < -1e-9 else float("nan"),
        "median_symbols": int(px.notna().sum(axis=1).median()),
        "daily_returns": r,
    }


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--first-year", type=int, default=1998)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--select-until", type=int, default=2016,
                    help="configs are ranked on test years <= this; later years are held out")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    status = args.out / "status.json"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _write(status, {"run_id": run_id, "state": "loading_panel"})

    panel = pd.read_parquet(args.panel, columns=["symbol", "trade_date", "close", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["close"] > 0) & panel["close"].notna()]

    grid = [
        {"signal": s, "lookback": lb, "top_k": k, "target_vol": tv,
         "max_name": 0.05, "brake_dd": 0.15, "brake_scale": 0.5,
         "min_amount": 5_000_000.0, "rebal": 5}
        for s, lb, k, tv in itertools.product(
            ["momentum", "reversal", "lowvol", "mom_x_lowvol"],
            [20, 60, 120],
            [30, 50],
            [0.12, 0.20],
        )
    ]
    windows = [(y, y + args.train_years - 1, y + args.train_years)
               for y in range(args.first_year, args.last_year - args.train_years + 1)]

    print(f"{len(grid)} configs x {len(windows)} windows", flush=True)
    per_cfg: list[dict] = []
    t0 = time.time()

    for ci, cfg in enumerate(grid, start=1):
        rows = []
        for ts, tge, te_y in windows:
            tr = panel[(panel.trade_date >= f"{ts}-01-01") & (panel.trade_date <= f"{tge}-12-31")]
            te = panel[(panel.trade_date >= f"{te_y}-01-01") & (panel.trade_date <= f"{te_y}-12-31")]
            if tr.empty or te.empty:
                continue
            res = run_window(tr, te, cfg)
            if res.get("status") != "ok" or res["median_symbols"] < BREADTH_FLOOR:
                continue
            res["test_year"] = te_y
            rows.append(res)
        if not rows:
            continue
        sel = [r for r in rows if r["test_year"] <= args.select_until]
        hold = [r for r in rows if r["test_year"] > args.select_until]
        agg = lambda rs, k: float(np.mean([r[k] for r in rs if np.isfinite(r[k])])) if rs else float("nan")
        per_cfg.append({
            "config": cfg,
            "n_select": len(sel), "n_holdout": len(hold),
            "select": {k: agg(sel, k) for k in ("excess_return", "cagr", "max_drawdown", "sharpe", "calmar")},
            "holdout": {k: agg(hold, k) for k in ("excess_return", "cagr", "max_drawdown", "sharpe", "calmar")},
        })
        _write(status, {"run_id": run_id, "state": "running",
                        "configs_done": ci, "configs_total": len(grid),
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"  [{ci}/{len(grid)}] {cfg['signal']}/{cfg['lookback']}/k{cfg['top_k']}/v{cfg['target_vol']} "
              f"sel_calmar={per_cfg[-1]['select']['calmar']:.3f} "
              f"hold_calmar={per_cfg[-1]['holdout']['calmar']:.3f}", flush=True)

    if not per_cfg:
        _write(status, {"run_id": run_id, "state": "no_results"})
        return 1

    # --- selection on the EARLY windows only -----------------------------
    ranked = sorted(per_cfg, key=lambda c: (-(c["select"]["calmar"] if np.isfinite(c["select"]["calmar"]) else -9e9)))
    best = ranked[0]

    # --- multiple-testing penalty ----------------------------------------
    # PBO takes paired (in-sample, out-of-sample) Sharpes per config, which is
    # exactly what the nested design already produces: `select` is in-sample for
    # the ranking, `holdout` is the years the ranking never saw. It answers the
    # question that matters here -- does the config that won the search land
    # below median out-of-sample?
    paired = [(c["select"]["sharpe"], c["holdout"]["sharpe"]) for c in per_cfg
              if np.isfinite(c["select"]["sharpe"]) and np.isfinite(c["holdout"]["sharpe"])]
    pbo = float("nan")
    if len(paired) >= 4:
        try:
            pbo = float(probability_of_backtest_overfitting(
                np.array([a for a, _ in paired]), np.array([b for _, b in paired])))
        except Exception as exc:  # noqa: BLE001
            print(f"  PBO unavailable: {type(exc).__name__}: {exc}")
    sharpes = [c["select"]["sharpe"] for c in per_cfg if np.isfinite(c["select"]["sharpe"])]
    n_trials = len(sharpes)
    # E[max of N draws] ~ mean + sd*sqrt(2 ln N): how much of the winner's edge
    # is explained by having searched at all.
    expected_max = (np.mean(sharpes) + np.std(sharpes, ddof=1) * np.sqrt(2 * np.log(max(n_trials, 2)))
                    if n_trials > 1 else float("nan"))

    summary = {
        "run_id": run_id, "state": "complete",
        "configs_tried": len(per_cfg),
        "select_until": args.select_until,
        "breadth_floor": BREADTH_FLOOR,
        "best_config": best["config"],
        "best_select": best["select"], "best_holdout": best["holdout"],
        "n_select_windows": best["n_select"], "n_holdout_windows": best["n_holdout"],
        "search_penalty": {
            "pbo": pbo,
            "n_trials": n_trials,
            "winner_select_sharpe": best["select"]["sharpe"],
            "expected_max_sharpe_from_noise": expected_max,
            "survives_expected_max": bool(np.isfinite(expected_max)
                                          and best["select"]["sharpe"] > expected_max),
        },
        "leaderboard": [
            {"config": c["config"], "select": c["select"], "holdout": c["holdout"]}
            for c in ranked[:10]
        ],
        "note": ("Configs were ranked on test years <= select_until; `best_holdout` "
                 "is on later windows the ranking never saw and is the number worth "
                 "believing. `search_penalty.pbo` above 0.5, or a winner Sharpe below "
                 "expected_max_sharpe_from_noise, means the winner is not "
                 "distinguishable from the best of N noisy draws."),
        "elapsed_s": round(time.time() - t0, 1),
    }
    _write(status, summary)
    (args.out / f"run_{run_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nBEST (selected on <= {args.select_until}): {best['config']}")
    print(f"  select : {best['select']}")
    print(f"  holdout: {best['holdout']}")
    print(f"  PBO={pbo}  winner_sharpe={best['select']['sharpe']:.3f} vs E[max]={expected_max:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
