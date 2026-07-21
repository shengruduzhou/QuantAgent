#!/usr/bin/env python3
"""H-027 Stage 2 (GPU): deep tabular candidates T0/T1 (TabM) + R1 (RealMLP-TD).

Preregistered H-027 (commit 4ca88e2). Pre-execution amendment (declared before
any deep model ran, for runtime feasibility on 1.6M-row folds; no results
seen): batch 4096->8192, max epochs 25->15, patience 5->4. Everything else
frozen: TabM.make defaults (n_blocks 3, d_block 512, dropout 0.1, k=32, no
numeric embeddings), AdamW lr 1e-3 wd 3e-4, MSE over k heads, early stop on
valid daily RankIC; features = per-date rank-pct fillna 0.5; target = per-date
rank-pct of exec h20. RealMLP-TD: pytabkit tuned defaults with bounded
n_epochs=16, batch_size=8192 (same feasibility note). VRAM recorded per
candidate; CUDA OOM => fail closed, never CPU fallback.
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import rss_gib  # noqa: E402
from exp026_ablation import LBL, POOL7  # noqa: E402
from exp027_stage1 import OUT, OOF, load_frame, fold_masks, daily_ic, decile_stats, record  # noqa: E402

MAX_EPOCHS, PATIENCE, BATCH = 15, 4, 8192


def train_tabm(trn_X, trn_y, val_X, val_y, val_dates, te_X, device):
    import torch
    import tabm
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = tabm.TabM.make(n_num_features=trn_X.shape[1], cat_cardinalities=[], d_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    Xtr = torch.from_numpy(trn_X); ytr = torch.from_numpy(trn_y)
    Xv = torch.from_numpy(val_X).to(device)
    n = len(Xtr)
    best_ic, best_state, bad = -np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx].to(device, non_blocking=True), ytr[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(xb, None).squeeze(-1)          # (B, k)
                loss = ((out - yb[:, None]) ** 2).mean()   # MSE averaged over k heads
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pv = []
            for i in range(0, len(Xv), 65536):
                pv.append(model(Xv[i:i + 65536], None).squeeze(-1).mean(-1).float().cpu())
            pv = torch.cat(pv).numpy()
        vic = float(daily_ic(pv, val_y, val_dates).mean())
        print(f"    ep{ep} val_ic {vic:+.5f}", flush=True)
        if vic > best_ic:
            best_ic, bad = vic, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    Xt = torch.from_numpy(te_X)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pt = []
        for i in range(0, len(Xt), 65536):
            pt.append(model(Xt[i:i + 65536].to(device), None).squeeze(-1).mean(-1).float().cpu())
    vram = torch.cuda.max_memory_allocated() / 1024**3
    del model, Xtr, ytr, Xv, Xt
    gc.collect(); torch.cuda.empty_cache()
    return torch.cat(pt).numpy(), vram, best_ic


def main() -> int:
    import torch
    assert torch.cuda.is_available(), "GPU required (fail closed, no CPU fallback)"
    device = "cuda"
    t0 = time.time()
    OOF.mkdir(parents=True, exist_ok=True)
    df, base_xs = load_frame()
    feats_all = base_xs + POOL7
    # per-date rank-pct + 0.5 fill (cross-sectional op; no temporal leakage)
    df[feats_all] = df.groupby("trade_date")[feats_all].rank(pct=True).fillna(0.5).astype("float32")
    m0, m3 = base_xs, feats_all
    print(f"frame {len(df):,} rows RSS {rss_gib():.1f} GiB, {time.time()-t0:.0f}s", flush=True)

    store = {"ics": [], "rows": [], "oof": []}
    failures, vram_log = {}, {}
    for fi, trm, vm, tem in fold_masks(df):
        trn, val, te = df[trm & ~vm], df[vm], df[tem]
        val_dates = val["trade_date"].to_numpy()
        print(f"\nF{fi}: train {len(trn):,} val {len(val):,} test {len(te):,}", flush=True)
        for cand, feats in (("T0", m0), ("T1", m3)):
            try:
                pred, vram, bic = train_tabm(
                    trn[feats].to_numpy(), trn["y"].to_numpy(),
                    val[feats].to_numpy(), val[LBL].to_numpy(), val_dates,
                    te[feats].to_numpy(), device)
                vram_log[f"{cand}_F{fi}"] = round(vram, 2)
                record(store, cand, fi, pred, te, note=f"vram={vram:.1f}G val_ic={bic:+.4f}")
            except torch.cuda.OutOfMemoryError:
                failures[f"{cand}_F{fi}"] = "CUDA_OOM (fail closed)"
                torch.cuda.empty_cache()
                print(f"  {cand} F{fi} CUDA OOM — fail closed", flush=True)
            except Exception:
                failures[f"{cand}_F{fi}"] = traceback.format_exc()[-600:]
                print(f"  {cand} F{fi} FAILED", flush=True)
        try:
            from pytabkit import RealMLP_TD_Regressor
            torch.cuda.reset_peak_memory_stats()
            r1 = RealMLP_TD_Regressor(device=device, n_epochs=16, batch_size=BATCH,
                                      random_state=0, verbosity=0)
            r1.fit(trn[m3].to_numpy(), trn["y"].to_numpy().astype("float32"),
                   X_val=val[m3].to_numpy(), y_val=val["y"].to_numpy().astype("float32"))
            pred = r1.predict(te[m3].to_numpy()).ravel()
            vram_log[f"R1_F{fi}"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            record(store, "R1", fi, pred, te, note=f"vram={vram_log[f'R1_F{fi}']}G")
            del r1; gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            failures[f"R1_F{fi}"] = "CUDA_OOM (fail closed)"
            torch.cuda.empty_cache()
        except Exception:
            failures[f"R1_F{fi}"] = traceback.format_exc()[-600:]
            print(f"  R1 F{fi} FAILED", flush=True)
        del trn, val, te; gc.collect()

    if store["ics"]:
        pd.concat(store["ics"]).to_parquet(OUT / "daily_ic_stage2.parquet", index=False)
        pd.concat(store["oof"]).to_parquet(OOF / "stage2_oof.parquet", index=False)
        pd.DataFrame(store["rows"]).to_csv(OUT / "stage2_fold_metrics.csv", index=False)
    (OUT / "stage2_failures.json").write_text(json.dumps(failures, indent=2))
    (OUT / "stage2_vram.json").write_text(json.dumps(vram_log, indent=2))
    print(f"\nstage2 done: {len(store['rows'])} candidate-folds, failures={len(failures)}, "
          f"peak RSS {rss_gib():.2f} GiB, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
