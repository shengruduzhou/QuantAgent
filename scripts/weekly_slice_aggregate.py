#!/usr/bin/env python3
"""Aggregate the FAIR weekly slice across many as-ofs → the real 1+1>2 verdict.

One 2-week window is pure noise (the factor's +20%/yr is the aggregate of ~100
rebalances; any single window can be deeply negative — e.g. early-April 2026).
So we run the fair strict backtest (factor rolling ∪ chain held, see
weekly_slice_eval.py) at MANY weekly as-ofs and aggregate:

  * mean / worst / win-rate of forward excess per sleeve & blend weight
  * hindsight ablation: chain[real] must beat chain[nonews]/[scrambled]
  * best union fw = the blend that beats BOTH factor and chain across windows

Chain pools are read from runtime/reports/monthly/chain_pool_<date>[_mode].parquet
(generate with batch_chain_pools.py).  Missing modes are skipped, not faked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weekly_slice_eval import (  # noqa: E402
    _benchmark, _blend, _factor_weights_rolling, _held_weights, _read, _run,
)

PRED_SRC = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
MON = Path("runtime/reports/monthly")
MODE_SFX = {"real": "", "nonews": "_nonews", "scrambled": "_scrambled"}


def _forward(panel: pd.DataFrame, preds: pd.DataFrame, as_of: pd.Timestamp, fwd_td: int) -> list[pd.Timestamp]:
    pdates = set(pd.to_datetime(panel["trade_date"]).dt.normalize().unique())
    rdates = set(pd.to_datetime(preds["trade_date"]).dt.normalize().unique())
    fut = sorted(d for d in (pdates & rdates) if d > as_of)
    return fut[:fwd_td]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", required=True, help="as-of dates (e.g. weekly Fridays)")
    ap.add_argument("--modes", nargs="+", default=["real", "nonews", "scrambled"])
    ap.add_argument("--fwd-td", type=int, default=10)
    ap.add_argument("--factor-topk", type=int, default=20)
    ap.add_argument("--chain-topn", type=int, default=15)
    ap.add_argument("--fw-grid", nargs="+", type=float, default=[0.8, 0.6, 0.5, 0.4, 0.2])
    ap.add_argument("--slippage-bps", nargs="+", type=float, default=[8.0, 16.0])
    ap.add_argument("--predictions", default=PRED_SRC)
    ap.add_argument("--out", default="runtime/reports/llm_validation/weekly_aggregate.json")
    args = ap.parse_args()

    allp = _read(Path(args.predictions))
    allp["trade_date"] = pd.to_datetime(allp["trade_date"]).dt.normalize()
    panel = _read(Path(PANEL)); panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    sector = _read(Path(SECTOR)) if Path(SECTOR).exists() else pd.DataFrame()

    rows: list[dict] = []
    used_dates: list[str] = []
    for d in args.dates:
        as_of = pd.Timestamp(d)
        dates = _forward(panel, allp, as_of, args.fwd_td)
        if len(dates) < args.fwd_td:
            print(f"[skip] {d}: only {len(dates)} forward dates (< {args.fwd_td})"); continue
        panel_w = panel[panel["trade_date"].isin(dates)].copy()
        bench = _benchmark(panel_w, dates)
        fac = _factor_weights_rolling(allp, dates, args.factor_topk)
        used_dates.append(d)

        for slip in args.slippage_bps:
            r = _run("factor_rolling", fac, panel_w, sector, slip, bench, 1_000_000.0)
            rows.append({**r, "as_of": d, "label": "-", "fw": 1.0, "bench": round(bench, 6)})
            for mode in args.modes:
                cp_path = MON / f"chain_pool_{d}{MODE_SFX[mode]}.parquet"
                if not cp_path.exists():
                    continue
                cp = _read(cp_path)
                for sccol in ("chain_conviction", "mix_score", "conviction"):
                    if sccol in cp.columns:
                        cp = cp.sort_values(sccol, ascending=False); break
                if "source" in cp.columns and cp["source"].astype(str).str.contains("LLM|产业链|chain", regex=True).any():
                    cp = cp[cp["source"].astype(str).str.contains("LLM|产业链|chain", regex=True)]
                syms = [str(s) for s in cp["symbol"].tolist()][: args.chain_topn]
                chw = _held_weights(syms, dates)
                rows.append({**_run(f"chain_held", chw, panel_w, sector, slip, bench, 1_000_000.0),
                             "as_of": d, "label": mode, "fw": 0.0, "bench": round(bench, 6)})
                for fw in args.fw_grid:
                    tw = _blend(fac, chw, fw, dates)
                    rows.append({**_run("union", tw, panel_w, sector, slip, bench, 1_000_000.0),
                                 "as_of": d, "label": mode, "fw": fw, "bench": round(bench, 6)})

    if not rows:
        print("no rows — generate chain pools first (batch_chain_pools.py)"); return 1
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)

    # ---- aggregate per (name,label,fw,slip) across as-ofs ----
    def agg(sub: pd.DataFrame) -> dict:
        ex = sub["excess"].to_numpy()
        return {"mean_excess": round(float(np.mean(ex)), 4), "worst": round(float(np.min(ex)), 4),
                "win_rate": round(float((ex > 0).mean()), 3), "n": int(len(ex)),
                "mean_maxDD": round(float(sub["maxDD"].mean()), 4), "mean_sharpe": round(float(sub["sharpe"].mean()), 3)}

    summary: dict = {"as_ofs": used_dates, "fwd_td": args.fwd_td, "factor_topk": args.factor_topk, "by_config": {}}
    for slip in args.slippage_bps:
        s = df[df["slip"] == slip]
        block: dict = {}
        block["factor_rolling"] = agg(s[s["name"] == "factor_rolling"])
        for mode in args.modes:
            sm = s[(s["name"] == "chain_held") & (s["label"] == mode)]
            if not sm.empty:
                block[f"chain[{mode}]"] = agg(sm)
            for fw in args.fw_grid:
                su = s[(s["name"] == "union") & (s["label"] == mode) & (np.isclose(s["fw"], fw))]
                if not su.empty:
                    block[f"union[{mode}]_fw{fw:.1f}"] = agg(su)
        summary["by_config"][f"slip{slip:.0f}"] = block

    # ---- verdicts ----
    def m(cfg, slip):
        return summary["by_config"].get(f"slip{slip:.0f}", {}).get(cfg, {}).get("mean_excess")
    stress = max(args.slippage_bps)
    fac_ex = m("factor_rolling", stress)
    chain_real = m("chain[real]", stress)
    chain_alt = [v for k in ("chain[nonews]", "chain[scrambled]") if (v := m(k, stress)) is not None]
    best_union = None
    for mode in ("real",):
        for fw in args.fw_grid:
            v = m(f"union[{mode}]_fw{fw:.1f}", stress)
            if v is not None and (best_union is None or v > best_union[1]):
                best_union = (f"union[{mode}]_fw{fw:.1f}", v)
    verdict = {
        "factor_rolling_mean_excess": fac_ex,
        "chain_real_mean_excess": chain_real,
        "chain_placebo_mean_excess": round(float(np.mean(chain_alt)), 4) if chain_alt else None,
        "edge_from_news": (round(chain_real - float(np.mean(chain_alt)), 4)
                           if chain_real is not None and chain_alt else None),
        "best_union": best_union,
        "one_plus_one_gt_two": (best_union is not None and fac_ex is not None and chain_real is not None
                                and best_union[1] > fac_ex and best_union[1] > chain_real),
    }
    summary["verdict_at_stress_slippage"] = verdict
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== weekly aggregate over {len(used_dates)} as-ofs, fwd {args.fwd_td}td, stress {stress:.0f}bps ===")
    for cfg, st in summary["by_config"][f"slip{stress:.0f}"].items():
        print(f"  {cfg:<22} mean {st['mean_excess']:+.4f}  worst {st['worst']:+.4f}  win {st['win_rate']:.0%}  n={st['n']}")
    print("\nVERDICT:", json.dumps(verdict, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
