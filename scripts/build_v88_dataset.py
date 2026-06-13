#!/usr/bin/env python3
"""Build the v8.8 training dataset: exec_v87 + discovered + GTJA survivors.

Adds, as new feature columns on top of the executable-label v8.7 dataset:
  * the GP/LLM accepted formulas (synth_* / llm_*) from the discovery run,
  * GTJA-191 tranche-1 factors whose unified-judgment verdict is
    all_weather / robust_4y / regime_specialist.

Values are computed full-history from the silver market panel (so rolling
windows see pre-2018 warmup), cast to float32, and merged on
(symbol, trade_date). Run AFTER judge_gtja191.py so the verdict file exists.

The v8.8 retrain should use --feature-policy judgment so every new column
is routed to its measured best horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXEC_V87 = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v87.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
GTJA_VERDICTS = "runtime/reports/v8/factor_full_judgment/gtja191_judgment.csv"
DISCOVERED = "runtime/reports/v8/discovery/eval_v87/accepted_definitions.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88.parquet")
    ap.add_argument("--warmup-days", type=int, default=400)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from quantagent.factors.factor_synthesis import load_definitions
    from quantagent.factors.gtja191 import compute_gtja191_factors

    meta = pq.ParquetFile(EXEC_V87)
    base_keys = pd.read_parquet(EXEC_V87, columns=["symbol", "trade_date"])
    base_keys["trade_date"] = pd.to_datetime(base_keys["trade_date"])
    symbols = set(base_keys["symbol"].unique())
    min_date = base_keys["trade_date"].min()
    print(f"exec_v87: {len(base_keys)} rows, {len(symbols)} symbols, from {min_date.date()}")

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                            "close", "volume", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[(panel["symbol"].isin(symbols))
                  & (panel["trade_date"] >= min_date - pd.Timedelta(days=args.warmup_days))]
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    print(f"panel slice: {len(panel)} rows")

    verdicts = pd.read_csv(GTJA_VERDICTS)
    keep_gtja = verdicts.loc[
        verdicts["verdict"].isin(["all_weather", "robust_4y", "regime_specialist"]), "factor"
    ].tolist()
    print(f"GTJA survivors: {len(keep_gtja)}")
    gtja_wide = compute_gtja191_factors(panel, names=keep_gtja, wide=True)
    gtja_wide["trade_date"] = pd.to_datetime(gtja_wide["trade_date"])
    for c in keep_gtja:
        gtja_wide[c] = gtja_wide[c].astype("float32")

    new_cols = dict.fromkeys(keep_gtja)
    disc_wide = panel[["symbol", "trade_date"]].copy()
    for definition in load_definitions(DISCOVERED):
        try:
            vals = pd.to_numeric(definition.expr.evaluate(panel), errors="coerce")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {definition.name}: {exc}")
            continue
        disc_wide[definition.name] = vals.replace([np.inf, -np.inf], np.nan).astype("float32").to_numpy()
        new_cols[definition.name] = None
        print(f"  computed {definition.name}")
    del panel

    print("loading exec_v87 full frame ...")
    df = pd.read_parquet(EXEC_V87)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.merge(gtja_wide, on=["symbol", "trade_date"], how="left")
    del gtja_wide
    df = df.merge(disc_wide, on=["symbol", "trade_date"], how="left")
    del disc_wide

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    coverage = {c: round(float(df[c].notna().mean()), 3) for c in list(new_cols)[:8]}
    schema = {
        "source": EXEC_V87,
        "added_columns": list(new_cols),
        "n_added": len(new_cols),
        "rows": int(len(df)),
        "coverage_sample": coverage,
        "note": "retrain with --feature-policy judgment (horizon_factor_assignment.json routing)",
    }
    Path(str(out).replace(".parquet", "_schema.json")).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(schema, ensure_ascii=False, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
