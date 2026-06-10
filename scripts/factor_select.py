#!/usr/bin/env python3
"""Scientific factor selection — between arbitrary top30 and the over-flat 168.

Too many factors flatten the latent space + overfit; too few (fixed top30) is arbitrary.
This selects the TRULY EFFECTIVE subset from factor_diagnostics using mainstream criteria:
  * keep |ICIR| >= --min-icir        (predictive AND stable, not just high mean IC)
  * keep regime specialists: |IC| in ANY regime >= --min-regime-ic (works somewhere)
  * always keep curated structural signals (old_dealer_risk = 避庄 hard gate, etc.)
  * factors are already de-redundified (governed set); we further prune by ICIR
Outputs the selected factor whitelist + a tiered report. Optionally writes a
selected training dataset (governed restricted to the whitelist + labels + meta).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DIAG = "runtime/reports/v8/factor_diagnostics/table.csv"
GOVERNED = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet"
OUT_DIR = Path("runtime/reports/v8/factor_select")
ALWAYS_KEEP = {"old_dealer_risk_score"}  # 避庄 hard-gate signal: keep regardless of IC


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-icir", type=float, default=0.30)
    ap.add_argument("--min-regime-ic", type=float, default=0.04)
    ap.add_argument("--max-factors", type=int, default=60, help="cap (model latent space)")
    ap.add_argument("--write-dataset", action="store_true", help="emit selected training dataset")
    ap.add_argument("--out-dataset", default="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_selected_v85.parquet")
    args = ap.parse_args()

    t = pd.read_csv(DIAG)
    governed = set(pd.read_parquet(GOVERNED, columns=None).columns) if Path(GOVERNED).exists() else set()
    # restrict to factors that survived neutralization/de-redundancy (the governed 168)
    if governed:
        t = t[t["factor"].isin(governed)]

    reg_cols = [c for c in ("ic_bull", "ic_sideways", "ic_bear") if c in t.columns]
    t["max_regime_ic"] = t[reg_cols].abs().max(axis=1) if reg_cols else 0.0
    t["stable"] = t["abs_icir"] >= args.min_icir
    t["regime_specialist"] = t["max_regime_ic"] >= args.min_regime_ic
    t["always"] = t["factor"].isin(ALWAYS_KEEP)
    t["selected"] = t["stable"] | t["regime_specialist"] | t["always"]

    sel = t[t["selected"]].sort_values("abs_icir", ascending=False)
    if len(sel) > args.max_factors:
        # cap: keep strongest |ICIR|, but guarantee the always-keep + best per-regime specialists
        head = sel.head(args.max_factors)
        must = sel[sel["always"]]
        sel = pd.concat([head, must]).drop_duplicates("factor").head(args.max_factors + len(must))
    whitelist = sel["factor"].tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selected_factors.json").write_text(json.dumps({
        "min_icir": args.min_icir, "min_regime_ic": args.min_regime_ic,
        "n_selected": len(whitelist), "n_candidates": int(len(t)),
        "whitelist": whitelist,
        "by_reason": {
            "stable_icir": int(t["stable"].sum()),
            "regime_specialist_only": int((t["regime_specialist"] & ~t["stable"]).sum()),
            "always_keep": sorted(ALWAYS_KEEP & set(t["factor"])),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    sel.round(4).to_csv(OUT_DIR / "selected_table.csv", index=False)

    print(f"candidates(governed)={len(t)} → selected={len(whitelist)} "
          f"(|ICIR|≥{args.min_icir}: {int(t['stable'].sum())}; regime-only: "
          f"{int((t['regime_specialist'] & ~t['stable']).sum())}; always: {len(ALWAYS_KEEP & set(t['factor']))})")
    print("top 20 selected:", whitelist[:20])

    if args.write_dataset and Path(GOVERNED).exists():
        df = pd.read_parquet(GOVERNED)
        meta = [c for c in df.columns if c.startswith(("forward_return", "label_end")) or c in {
            "symbol", "trade_date", "available_at", "open", "high", "low", "close", "volume", "amount",
            "is_suspended", "is_st", "is_limit_up", "is_limit_down"}]
        keep_cols = list(dict.fromkeys(meta + [f for f in whitelist if f in df.columns]))
        df[keep_cols].to_parquet(args.out_dataset, index=False)
        print(f"wrote {args.out_dataset}: {df.shape[0]} rows, {len(keep_cols)} cols ({len(whitelist)} factors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
