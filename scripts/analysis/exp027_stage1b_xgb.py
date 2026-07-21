#!/usr/bin/env python3
"""H-027 stage 1b: X0/X1 re-run after the eval_qid API fix (code defect, not a
new trial — same preregistered candidates/params). Writes *_stage1b artifacts
picked up by exp027_gates."""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import rss_gib  # noqa: E402
from exp026_ablation import POOL7  # noqa: E402
from exp027_stage1 import OUT, OOF, XGB_PARAMS, load_frame, fold_masks, record  # noqa: E402


def main() -> int:
    import xgboost as xgbm
    t0 = time.time()
    df, base_xs = load_frame()
    store = {"ics": [], "rows": [], "oof": []}
    failures = {}
    for fi, trm, vm, tem in fold_masks(df):
        trn, val, te = df[trm & ~vm], df[vm], df[tem]
        qid_tr, qid_v = trn["trade_date"].factorize()[0], val["trade_date"].factorize()[0]
        for cand, feats in (("X0", base_xs), ("X1", base_xs + POOL7)):
            try:
                xr = xgbm.XGBRanker(**XGB_PARAMS)
                xr.fit(trn[feats], trn["grade"], qid=qid_tr,
                       eval_set=[(val[feats], val["grade"])], eval_qid=[qid_v], verbose=False)
                record(store, cand, fi, xr.predict(te[feats]), te)
                del xr; gc.collect()
            except Exception:
                failures[f"{cand}_F{fi}"] = traceback.format_exc()[-500:]
                print(f"  {cand} FAILED F{fi}", flush=True)
        del trn, val, te; gc.collect()
    pd.concat(store["ics"]).to_parquet(OUT / "daily_ic_stage1b.parquet", index=False)
    pd.concat(store["oof"]).to_parquet(OOF / "stage1b_oof.parquet", index=False)
    pd.DataFrame(store["rows"]).to_csv(OUT / "stage1b_fold_metrics.csv", index=False)
    (OUT / "stage1b_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"stage1b done, failures={len(failures)}, RSS {rss_gib():.2f} GiB, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
