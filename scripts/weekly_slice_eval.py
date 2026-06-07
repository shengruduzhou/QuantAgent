#!/usr/bin/env python3
"""Fair weekly vertical-slice backtest: factor (rolling) ∪ LLM-chain (held).

This fixes the static-hold artifact that made the factor sleeve look weak in
``validate_llm_pit_selection.py`` / ``chain_oos_validation.py``.  Those scripts
froze the as-of top-N and held it for the whole forward window, throwing away
the 5d factor's edge (which needs daily rebalancing — reproduced +19.8%/yr only
with rolling rebalance, see overlay_backtest_ab.py).

Design (each sleeve traded on its NATIVE cadence):
  * factor sleeve : each forward day re-rank predictions → top-K equal weight
                    (rolling rebalance — the way the factor actually earns +α).
  * chain sleeve  : the LLM chain pool, equal weight, HELD across the window
                    (event/景气 driven — its native multi-week cadence).
  * union(fw)     : per day w = fw·factor_row + (1-fw)·chain_row (each row sums
                    to 1, so the blend also sums to 1 — no extra renorm).

All sleeves go through the SAME strict_v8 backtest (T+1, cost, slippage,
ST/suspension/limit gates) and are scored as excess over the equal-weight all-A
benchmark on the same forward dates.

Ablation (clean-OOS / anti-hindsight): pass several chain pools as
``--chain LABEL=path`` (e.g. real / nonews / scrambled).  The real-news chain
must beat the placebo chains OOS, otherwise the "edge" is parametric memory.

This script never calls an LLM; it only backtests pools already on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _pred_score_col(df: pd.DataFrame) -> str:
    for c in ("alpha_score", "prediction", "composite_score", "score"):
        if c in df.columns:
            return c
    raise ValueError("predictions need alpha_score/prediction/composite_score/score")


def _chain_score_col(df: pd.DataFrame) -> str | None:
    for c in ("chain_conviction", "mix_score", "conviction", "score"):
        if c in df.columns:
            return c
    return None


def _forward_dates(panel: pd.DataFrame, preds: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    pdates = set(pd.to_datetime(panel["trade_date"]).dt.normalize().unique())
    rdates = set(pd.to_datetime(preds["trade_date"]).dt.normalize().unique())
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    # require BOTH a price (panel) and a fresh prediction (so the factor sleeve
    # can actually rebalance each day — no silent static-hold tail).
    return sorted(d for d in (pdates & rdates) if lo <= d <= hi)


def _factor_weights_rolling(preds: pd.DataFrame, dates: list[pd.Timestamp], top_k: int) -> pd.DataFrame:
    sc = _pred_score_col(preds)
    d = preds.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
    d = d[d["trade_date"].isin(dates)].dropna(subset=[sc])
    d = d.sort_values(["trade_date", sc], ascending=[True, False])
    d["rank"] = d.groupby("trade_date").cumcount()
    d = d[d["rank"] < top_k].copy()
    d["w"] = 1.0 / float(top_k)
    return d.pivot_table(index="trade_date", columns="symbol", values="w", fill_value=0.0).reindex(dates).fillna(0.0)


def _held_weights(symbols: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    symbols = [s for s in dict.fromkeys(symbols) if isinstance(s, str) and s]
    if not symbols:
        return pd.DataFrame(index=pd.DatetimeIndex(dates))
    w = 1.0 / float(len(symbols))
    return pd.DataFrame({s: w for s in symbols}, index=pd.DatetimeIndex(dates))


def _blend(fac: pd.DataFrame, chain: pd.DataFrame, fw: float, dates: list[pd.Timestamp]) -> pd.DataFrame:
    cols = sorted(set(fac.columns) | set(chain.columns))
    f = fac.reindex(index=dates, columns=cols).fillna(0.0)
    c = chain.reindex(index=dates, columns=cols).fillna(0.0)
    out = fw * f + (1.0 - fw) * c
    s = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(s, axis=0).fillna(0.0)


def _benchmark(panel: pd.DataFrame, dates: list[pd.Timestamp]) -> float:
    px = panel[panel["trade_date"].isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    daily = px.pct_change(fill_method=None).mean(axis=1).dropna()
    return float((1.0 + daily).prod() - 1.0) if not daily.empty else 0.0


def _run(name: str, tw: pd.DataFrame, panel: pd.DataFrame, sector: pd.DataFrame,
         slip: float, bench: float, cash: float) -> dict:
    if tw.empty or tw.to_numpy().sum() == 0:
        return {"name": name, "pool": 0, "slip": slip, "total": 0.0, "excess": -bench,
                "maxDD": 0.0, "sharpe": 0.0, "turnover": 0.0}
    res = run_strict_backtest_v8(
        tw, panel, sector_map=sector if not sector.empty else None,
        config=AShareExecutionSimulationConfig(initial_cash=cash, slippage_bps=slip),
    )
    m = res.metrics.to_dict()
    tot = float(m["total_return"])
    return {
        "name": name, "pool": int((tw.iloc[-1] > 0).sum()), "slip": float(slip),
        "total": round(tot, 6), "excess": round(tot - bench, 6),
        "maxDD": round(float(m["max_drawdown"]), 6), "sharpe": round(float(m["sharpe"]), 6),
        "turnover": round(float(m.get("turnover", 0.0)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--predictions", required=True, help="real preds with alpha_score/prediction over the forward window")
    ap.add_argument("--chain", action="append", default=[], help="LABEL=path.parquet (repeatable: real/nonews/scrambled)")
    ap.add_argument("--panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--sector-map", default="runtime/data/v7/silver/sector_map/sector_map.parquet")
    ap.add_argument("--factor-topk", type=int, default=20)
    ap.add_argument("--chain-topn", type=int, default=15, help="cap on chain pool size used")
    ap.add_argument("--fw-grid", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0])
    ap.add_argument("--slippage-bps", nargs="+", type=float, default=[8.0, 16.0])
    ap.add_argument("--initial-cash", type=float, default=1_000_000.0)
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/llm_validation"))
    ap.add_argument("--tag", default="weekly_slice")
    args = ap.parse_args()

    if pd.Timestamp(args.as_of) >= pd.Timestamp(args.start_date):
        raise SystemExit("--as-of must be strictly before --start-date (forward validation)")

    preds = _read(Path(args.predictions))
    panel = _read(Path(args.panel))
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    sector = _read(Path(args.sector_map)) if Path(args.sector_map).exists() else pd.DataFrame()

    dates = _forward_dates(panel, preds, args.start_date, args.end_date)
    if not dates:
        raise SystemExit("no forward dates with BOTH a price and a prediction")
    panel_w = panel[panel["trade_date"].isin(dates)].copy()
    bench = _benchmark(panel_w, dates)

    fac = _factor_weights_rolling(preds, dates, args.factor_topk)

    chains: dict[str, pd.DataFrame] = {}
    for spec in args.chain:
        if "=" not in spec:
            raise SystemExit(f"--chain must be LABEL=path, got {spec!r}")
        label, path = spec.split("=", 1)
        cp = _read(Path(path))
        sccol = _chain_score_col(cp)
        if sccol:
            cp = cp.sort_values(sccol, ascending=False)
        chains[label] = cp.head(args.chain_topn)

    rows: list[dict] = []
    for slip in args.slippage_bps:
        # factor-only (rolling) — the honest baseline
        rows.append({**_run("factor_rolling", fac, panel_w, sector, slip, bench, args.initial_cash),
                     "label": "-", "fw": 1.0})
        for label, cp in chains.items():
            chw = _held_weights([str(s) for s in cp["symbol"].tolist()], dates)
            rows.append({**_run(f"chain_held[{label}]", chw, panel_w, sector, slip, bench, args.initial_cash),
                         "label": label, "fw": 0.0})
            for fw in args.fw_grid:
                if fw in (0.0, 1.0):  # 1.0==factor_rolling, 0.0==chain_held already covered
                    continue
                tw = _blend(fac, chw, fw, dates)
                rows.append({**_run(f"union[{label}]_fw{fw:.1f}", tw, panel_w, sector, slip, bench, args.initial_cash),
                             "label": label, "fw": fw})

    res = pd.DataFrame(rows).sort_values(["slip", "excess"], ascending=[True, False]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.as_of}_{args.start_date}_{args.end_date}".replace("-", "")
    (args.out_dir / f"{stem}.csv").write_text(res.to_csv(index=False), encoding="utf-8")

    base = res[(res.slip == min(args.slippage_bps))]
    factor_excess = float(base[base.name == "factor_rolling"]["excess"].iloc[0]) if not base.empty else 0.0
    md = [f"# Weekly Slice (FAIR: factor rolling ∪ chain held) — as_of {args.as_of}", "",
          f"- forward window: {args.start_date} → {args.end_date}  ({len(dates)} trading days)",
          f"- benchmark (eqw all-A) total: {bench:.4%}",
          f"- factor sleeve = rolling daily rebalance top-{args.factor_topk}; chain sleeve = equal-weight held",
          f"- factor_rolling excess @ {min(args.slippage_bps):.0f}bps: **{factor_excess:.2%}**", "",
          "| config | label | fw | pool | slip | total | excess | maxDD | sharpe |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in base.iterrows():
        md.append(f"| {r['name']} | {r['label']} | {r['fw']:.1f} | {int(r['pool'])} | {r['slip']:.0f} | "
                  f"{r['total']:.2%} | {r['excess']:.2%} | {r['maxDD']:.2%} | {r['sharpe']:.2f} |")
    md += ["", "## Reading guide",
           "- `factor_rolling` is the honest factor baseline (it rebalances daily, the way it earns +α).",
           "- 1+1>2 holds only if a `union[real]_fw*` beats BOTH `factor_rolling` and `chain_held[real]` at BOTH slippages.",
           "- anti-hindsight: `chain_held[real]` must beat `chain_held[nonews]`/`[scrambled]`; if not, the chain edge is memory."]
    (args.out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n=== Weekly slice as_of={args.as_of} fwd {args.start_date}..{args.end_date} "
          f"({len(dates)}d) bench={bench:.4%} ===")
    print(base[["name", "label", "fw", "pool", "total", "excess", "maxDD", "sharpe"]].to_string(index=False))
    print(json.dumps({"csv": str(args.out_dir / f'{stem}.csv'), "md": str(args.out_dir / f'{stem}.md'),
                      "factor_rolling_excess": factor_excess, "bench": bench, "dates": len(dates)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
