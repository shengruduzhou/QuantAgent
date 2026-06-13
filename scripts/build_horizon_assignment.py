#!/usr/bin/env python3
"""Rebuild horizon_factor_assignment.json from the master judgment table.

Run after every factor_full_judgment.py / judge_gtja191.py update so the
``--feature-policy judgment`` retrain routing stays in sync.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgment-dir", default="runtime/reports/v8/factor_full_judgment")
    args = ap.parse_args()

    out_dir = Path(args.judgment_dir)
    t = pd.read_csv(out_dir / "factor_judgment_table.csv")
    ok = t[t.verdict.isin(["all_weather", "robust_4y"])].copy()
    assign: dict[str, list[dict]] = {}
    for h in ["5d", "20d", "60d"]:
        sub = ok[ok.best_horizon == h].copy()
        sub["abs_icir"] = sub[f"icir_{h}"].abs()
        sub = sub.sort_values("abs_icir", ascending=False)
        assign[h] = [
            {"factor": r.factor, "family": r.family,
             "ic": float(getattr(r, f"ic_{h}")), "icir": float(getattr(r, f"icir_{h}")),
             "capacity_ratio": (None if pd.isna(r.capacity_ratio) else float(r.capacity_ratio))}
            for r in sub.itertuples()
        ]
    spec = t[t.verdict == "regime_specialist"]
    out = {
        "protocol": "all_weather/robust_4y factors assigned to their best (max |ICIR|) horizon; "
                    "use for 短线(5d)/中线(20d)/长线(60d) sleeves and --feature-policy judgment",
        "short_5d": assign["5d"],
        "mid_20d": assign["20d"],
        "long_60d": assign["60d"],
        "regime_specialists": [
            {"factor": r.factor, "family": r.family,
             "ic_bull": r.ic_bull, "ic_sideways": r.ic_sideways, "ic_bear": r.ic_bear}
            for r in spec.itertuples()
        ],
        "families_included": sorted(t["family"].unique().tolist()),
    }
    path = out_dir / "horizon_factor_assignment.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"short": len(assign["5d"]), "mid": len(assign["20d"]),
                      "long": len(assign["60d"]), "specialists": int(len(spec)),
                      "output": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
