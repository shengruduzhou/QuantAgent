#!/usr/bin/env python3
"""Selective 做T walk-forward validation (the gate that decides if 做T ships).

Protocol (anti-overfit by construction):
  Phase 1  batch-simulate EVERY cached (symbol, day) × FSM-param combo
           (dip/target/stop/deadline × dip_buy/spike_sell). Selectivity gates
           are NOT applied here — the FSM outcome is independent of them.
  Phase 2  sweep the selectivity-gate grid as a pure filter over the Phase-1
           table, TRAIN window only. Keep configs whose research-universe
           per-leg net edge is positive with t ≥ t_min and enough legs,
           rank survivors by TRAIN book uplift.
  Phase 3  evaluate ONLY the chosen config on the untouched TEST window:
           research per-leg stats + holdings-book uplift + per-regime split.

Costs per round trip (both legs): 2×commission + sell stamp + 2×slippage.

Usage:
  selective_dot_walkforward.py --split 2026-03-31 [--max-symbols 50]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.selective_dot import (
    SelectiveDotParams,
    build_day_contexts,
    prepare_day_arrays,
    simulate_prepared,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MINUTE_DIR = Path("runtime/data/v7/silver/minute_bars")
HOLDINGS = "runtime/paper/replay_2026/holdings_daily.csv"
ANN = 244

FSM_GRID = {
    "mode": ["dip_buy", "spike_sell"],
    "dip_atr_mult": [0.20, 0.30, 0.45],
    "target_atr_mult": [0.40, 0.60],
    "stop_atr_mult": [0.40, 0.60],
    "morning_deadline": ["10:00:00", "10:30:00"],
}

MIN_ATR_GRID = [0.015, 0.025, 0.035]
MIN_MOM_GRID = [0.0, 0.02, 0.05]      # dip side: 5d trend must exceed
MAX_MOM_GRID = [0.0, -0.02]           # spike side: 5d trend must be below
MAX_ABS_GAP = 0.04


def load_day_arrays(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
                    min_bars: int = 30) -> dict[tuple[str, pd.Timestamp], dict]:
    out: dict[tuple[str, pd.Timestamp], dict] = {}
    for i, sym in enumerate(symbols):
        path = MINUTE_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["trade_time", "open", "high", "low", "close", "volume"])
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        bars = bars[(bars["trade_time"] >= start) & (bars["trade_time"] <= end + pd.Timedelta(days=1))]
        if bars.empty:
            continue
        for d, g in bars.groupby(bars["trade_time"].dt.normalize()):
            if len(g) < min_bars:
                continue
            day = prepare_day_arrays(g)
            if day is not None:
                out[(sym, d)] = day
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(symbols)} symbols, {len(out)} symbol-days", flush=True)
    return out


def phase1_results(day_arrays: dict, ctx_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    combos = [dict(zip(FSM_GRID, vals)) for vals in itertools.product(*FSM_GRID.values())]
    params_by_combo = [
        (i, c["mode"], SelectiveDotParams(mode=c["mode"], dip_atr_mult=c["dip_atr_mult"],
                                          target_atr_mult=c["target_atr_mult"],
                                          stop_atr_mult=c["stop_atr_mult"],
                                          morning_deadline=c["morning_deadline"]))
        for i, c in enumerate(combos)
    ]
    rows: list[tuple] = []
    n_days = len(day_arrays)
    for k, ((sym, d), day) in enumerate(day_arrays.items()):
        atr = ctx_map.get((sym, d))
        if atr is None or not np.isfinite(atr) or atr <= 0:
            continue
        for ci, mode, p in params_by_combo:
            state, _, _, ret, _, _ = simulate_prepared(day, atr, p, mode)
            rows.append((sym, d, ci, state, ret if ret is not None else np.nan))
        if (k + 1) % 10000 == 0:
            print(f"  simulated {k + 1}/{n_days} symbol-days", flush=True)
    res = pd.DataFrame(rows, columns=["symbol", "trade_date", "combo", "state", "gross_ret"])
    meta = pd.DataFrame(combos).reset_index().rename(columns={"index": "combo"})
    return res, meta


def _gate_mask(df: pd.DataFrame, side: str, min_atr: float, mom_thr: float) -> pd.Series:
    m = (df["atr_pct"] >= min_atr) \
        & (df["gap_open"].abs() <= MAX_ABS_GAP) \
        & df["regime"].isin(("bull", "sideways"))
    if side == "dip":
        return m & (df["mom_5d"] >= mom_thr)
    return m & (df["mom_5d"] <= mom_thr)


def eval_slices(slices: list[pd.DataFrame], cost_rt: float, dot_fraction: float,
                book_dates: list) -> dict:
    """Aggregate gated result slices into per-leg + book-uplift stats."""
    if not slices:
        return {"n_legs": 0}
    df = pd.concat(slices, ignore_index=True)
    if df.empty:
        return {"n_legs": 0}
    attempted = df[df["state"] != "waiting_no_entry"].copy()
    if attempted.empty:
        return {"n_legs": 0}
    attempted["net_ret"] = attempted["gross_ret"] - cost_rt
    n = len(attempted)
    mean_net = float(attempted["net_ret"].mean())
    sd = float(attempted["net_ret"].std(ddof=1)) if n > 1 else np.nan
    tstat = mean_net / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    out = {
        "n_days_gated_in": int(df.drop_duplicates(["symbol", "trade_date"]).shape[0]),
        "n_legs": n,
        "entry_rate": round(n / max(1, len(df)), 3),
        "hit_rate": round(float((attempted["state"] == "closed_profit").mean()), 3),
        "mean_gross": round(float(attempted["gross_ret"].mean()), 5),
        "mean_net": round(mean_net, 5),
        "t_stat": round(float(tstat), 2) if np.isfinite(tstat) else None,
        "total_net": round(float(attempted["net_ret"].sum()), 4),
        # 风控 tail stats per leg
        "worst_leg": round(float(attempted["net_ret"].min()), 5),
        "p5_leg": round(float(attempted["net_ret"].quantile(0.05)), 5),
        "eod_close_rate": round(float((attempted["state"] == "closed_eod").mean()), 3),
    }
    book = attempted[attempted["weight"].notna()]
    if not book.empty and book_dates:
        daily = book.assign(contrib=book["weight"] * dot_fraction * book["net_ret"]) \
                    .groupby("trade_date")["contrib"].sum()
        full = daily.reindex(book_dates, fill_value=0.0)
        ann = float((1 + full).prod() ** (ANN / max(1, len(book_dates))) - 1)
        out.update({
            "book_legs": int(len(book)),
            "book_daily_uplift_bps": round(float(full.mean()) * 1e4, 2),
            "book_ann_uplift": round(ann, 4),
        })
    return out


def sweep_gates(by_combo: dict[int, pd.DataFrame], meta: pd.DataFrame,
                cost_rt: float, dot_fraction: float, book_dates: list) -> pd.DataFrame:
    dip_combos = meta[meta["mode"] == "dip_buy"]["combo"].tolist()
    spike_combos = meta[meta["mode"] == "spike_sell"]["combo"].tolist()
    # auto pairs share identical FSM params on both sides
    key_cols = ["dip_atr_mult", "target_atr_mult", "stop_atr_mult", "morning_deadline"]
    keyed = {tuple(r[k] for k in key_cols): {} for _, r in meta.iterrows()}
    for _, r in meta.iterrows():
        keyed[tuple(r[k] for k in key_cols)][r["mode"]] = int(r["combo"])
    auto_pairs = [(v["dip_buy"], v["spike_sell"]) for v in keyed.values()
                  if "dip_buy" in v and "spike_sell" in v]

    rows = []
    for min_atr in MIN_ATR_GRID:
        for mom_thr in MIN_MOM_GRID:
            for dc in dip_combos:
                sl = by_combo[dc]
                stats = eval_slices([sl[_gate_mask(sl, "dip", min_atr, mom_thr)]],
                                    cost_rt, dot_fraction, book_dates)
                rows.append({"policy": "dip_only", "min_atr_pct": min_atr,
                             "min_mom_5d": mom_thr, "max_mom_5d": None,
                             "dip_combo": dc, "spike_combo": None, **stats})
        for mom_thr in MAX_MOM_GRID:
            for sc in spike_combos:
                sl = by_combo[sc]
                stats = eval_slices([sl[_gate_mask(sl, "spike", min_atr, mom_thr)]],
                                    cost_rt, dot_fraction, book_dates)
                rows.append({"policy": "spike_only", "min_atr_pct": min_atr,
                             "min_mom_5d": None, "max_mom_5d": mom_thr,
                             "dip_combo": None, "spike_combo": sc, **stats})
        for min_mom in MIN_MOM_GRID:
            for max_mom in MAX_MOM_GRID:
                for dc, sc in auto_pairs:
                    sld, sls = by_combo[dc], by_combo[sc]
                    stats = eval_slices(
                        [sld[_gate_mask(sld, "dip", min_atr, min_mom)],
                         sls[_gate_mask(sls, "spike", min_atr, max_mom)]],
                        cost_rt, dot_fraction, book_dates)
                    rows.append({"policy": "auto", "min_atr_pct": min_atr,
                                 "min_mom_5d": min_mom, "max_mom_5d": max_mom,
                                 "dip_combo": dc, "spike_combo": sc, **stats})
    return pd.DataFrame(rows)


def eval_chosen(by_combo: dict[int, pd.DataFrame], chosen: dict, cost_rt: float,
                dot_fraction: float, book_dates: list,
                regime: str | None = None) -> dict:
    slices = []
    if chosen["dip_combo"] is not None:
        sl = by_combo[int(chosen["dip_combo"])]
        m = _gate_mask(sl, "dip", chosen["min_atr_pct"], chosen["min_mom_5d"])
        if regime:
            m &= sl["regime"] == regime
        slices.append(sl[m])
    if chosen["spike_combo"] is not None:
        sl = by_combo[int(chosen["spike_combo"])]
        m = _gate_mask(sl, "spike", chosen["min_atr_pct"], chosen["max_mom_5d"])
        if regime:
            m &= sl["regime"] == regime
        slices.append(sl[m])
    return eval_slices(slices, cost_rt, dot_fraction, book_dates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="2026-03-31", help="train ≤ split < test")
    ap.add_argument("--start", default="2025-12-15")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--commission-bps", type=float, default=2.5)
    ap.add_argument("--stamp-bps", type=float, default=5.0)
    ap.add_argument("--dot-fraction", type=float, default=0.3)
    ap.add_argument("--min-train-legs", type=int, default=200)
    ap.add_argument("--t-min", type=float, default=2.0)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--reuse-phase1", action="store_true",
                    help="reuse <out>/phase1_results.parquet if present")
    ap.add_argument("--output-dir", default="runtime/reports/dot_selective")
    args = ap.parse_args()

    split = pd.Timestamp(args.split)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    cost_rt = (2 * args.commission_bps + args.stamp_bps + 2 * args.slippage_bps) / 1e4
    maker_rt = (2 * args.commission_bps + args.stamp_bps) / 1e4
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(p.stem for p in MINUTE_DIR.glob("*.parquet"))
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    print(f"=== Phase 0: contexts ({len(symbols)} symbols) ===", flush=True)
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["trade_date"] >= start - pd.Timedelta(days=120))
                  & (panel["trade_date"] <= end)]
    ctx = build_day_contexts(panel[panel["symbol"].isin(symbols)])
    ctx = ctx[ctx["trade_date"] >= start]
    ctx_map = {(r.symbol, r.trade_date): r.atr_pct for r in ctx.itertuples()}

    holdings = pd.read_csv(HOLDINGS)
    holdings["trade_date"] = pd.to_datetime(holdings["trade_date"])
    holdings["symbol"] = holdings["symbol"].astype(str)

    p1_path = out_dir / "phase1_results.parquet"
    if args.reuse_phase1 and p1_path.exists():
        print("=== Phase 1: reusing cached results ===", flush=True)
        res = pd.read_parquet(p1_path)
        meta = pd.read_csv(out_dir / "fsm_combos.csv")
    else:
        day_arrays = load_day_arrays(symbols, start, end)
        print(f"  {len(day_arrays)} symbol-days with minute bars", flush=True)
        print("=== Phase 1: batch FSM simulation ===", flush=True)
        res, meta = phase1_results(day_arrays, ctx_map)
        res.to_parquet(p1_path, index=False)
        meta.to_csv(out_dir / "fsm_combos.csv", index=False)
    print(f"  {len(res)} simulated rows across {len(meta)} FSM combos", flush=True)

    # one-time joins: contexts + book weights
    res = res.merge(ctx, on=["symbol", "trade_date"], how="inner")
    res = res.merge(holdings[["symbol", "trade_date", "weight"]],
                    on=["symbol", "trade_date"], how="left")

    train = res[res["trade_date"] <= split]
    test = res[res["trade_date"] > split]
    train_book_dates = sorted(holdings.loc[holdings["trade_date"] <= split, "trade_date"].unique())
    test_book_dates = sorted(holdings.loc[holdings["trade_date"] > split, "trade_date"].unique())
    by_combo_train = dict(tuple(train.groupby("combo")))
    by_combo_test = dict(tuple(test.groupby("combo")))

    print("=== Phase 2: gate grid on TRAIN ===", flush=True)
    grid = sweep_gates(by_combo_train, meta, cost_rt, args.dot_fraction, train_book_dates)
    grid = grid[grid["n_legs"] >= args.min_train_legs]
    grid.to_csv(out_dir / "grid_train.csv", index=False)
    if grid.empty:
        raise SystemExit("no gate config produced enough train legs")

    qualified = grid[(grid["mean_net"] > 0) & (grid["t_stat"].fillna(-9) >= args.t_min)
                     & grid.get("book_ann_uplift", pd.Series(dtype=float)).notna()]
    print(f"  {len(grid)} configs ≥{args.min_train_legs} legs; {len(qualified)} pass "
          f"net>0 & t≥{args.t_min}", flush=True)
    if qualified.empty:
        verdict = {"verdict": "NO_CONFIG_PASSES_TRAIN", "cost_per_roundtrip": cost_rt,
                   "best_unqualified_by_t": grid.sort_values("t_stat", ascending=False)
                                                .head(5).to_dict("records")}
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str),
                                              encoding="utf-8")
        print(json.dumps(verdict, indent=2, default=str))
        return 0

    chosen = qualified.sort_values("book_ann_uplift", ascending=False).iloc[0]
    chosen = {k: (None if pd.isna(v) else v) for k, v in chosen.items()}
    print("  chosen:", {k: chosen[k] for k in ("policy", "min_atr_pct", "min_mom_5d",
                                               "max_mom_5d", "dip_combo", "spike_combo",
                                               "mean_net", "t_stat", "book_ann_uplift")}, flush=True)

    print("=== Phase 3: OOS evaluation ===", flush=True)
    oos = eval_chosen(by_combo_test, chosen, cost_rt, args.dot_fraction, test_book_dates)
    oos_maker = eval_chosen(by_combo_test, chosen, maker_rt, args.dot_fraction, test_book_dates)
    regime_split = {rg: eval_chosen(by_combo_test, chosen, cost_rt, args.dot_fraction,
                                    test_book_dates, regime=rg)
                    for rg in ("bull", "sideways")}

    verdict = {
        "verdict": ("ENABLE" if oos.get("n_legs", 0) >= 30 and oos.get("mean_net", -1) > 0
                    and (oos.get("book_ann_uplift") or -1) > 0 else "DO_NOT_ENABLE"),
        "cost_per_roundtrip": cost_rt,
        "split": str(split.date()),
        "window": f"{args.start}..{args.end}",
        "chosen_config": {
            "policy": chosen["policy"], "min_atr_pct": chosen["min_atr_pct"],
            "min_mom_5d": chosen["min_mom_5d"], "max_mom_5d": chosen["max_mom_5d"],
            "max_abs_gap": MAX_ABS_GAP,
            "dip_fsm": None if chosen["dip_combo"] is None else meta.iloc[int(chosen["dip_combo"])].to_dict(),
            "spike_fsm": None if chosen["spike_combo"] is None else meta.iloc[int(chosen["spike_combo"])].to_dict(),
        },
        "train": {k: chosen.get(k) for k in ("n_days_gated_in", "n_legs", "entry_rate",
                                             "hit_rate", "mean_gross", "mean_net", "t_stat",
                                             "book_legs", "book_daily_uplift_bps",
                                             "book_ann_uplift")},
        "oos_taker": oos,
        "oos_maker_no_slip": oos_maker,
        "oos_by_regime": regime_split,
        "n_qualified_train_configs": int(len(qualified)),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps(verdict, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
