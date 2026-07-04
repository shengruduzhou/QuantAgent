#!/usr/bin/env python3
"""H-008 walk-forward retrain runner (WALK_FORWARD_PROTOCOL_H008.md §7).

Sequential GPU execution of 3 folds x 3 sleeves with hard guards:
  * RAM: kill the training process if RSS > 48 GiB (poll /proc every 30s)
  * CUDA OOM: one retry with train-micro-batch halved; second OOM aborts run
  * disk: abort if wf_h008 output tree exceeds 6 GiB or free space < 20 GiB
  * NaN / empty predictions: abort
  * abort marker file stops the chain; partial diagnostics retained

Every fold+sleeve appends a ledger row (command, git hash, runtime, RSS peak,
GPU peak if reported, artifact paths) to runner_ledger.jsonl.

F4 is NOT trained here — the protocol reuses retrain_plus7_20260620_0300.
Candidate evaluation happens afterwards in exp008_walkforward_eval.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / "AI_quant_venv/bin/python3")
OUT_ROOT = REPO / "runtime/reports/v89_closed_loop/wf_h008"
LEDGER = OUT_ROOT / "runner_ledger.jsonl"
ABORT = OUT_ROOT / "ABORT"
DATASET = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"

FOLDS = {
    "F1": {"train_end": "2022-12-30", "test_end": "2023-12-29"},
    "F2": {"train_end": "2023-06-30", "test_end": "2024-06-28"},
    "F3": {"train_end": "2023-12-29", "test_end": "2024-12-31"},
}
SLEEVES = ("short_5d", "mid_5d_30d", "long_30d_120d")
RAM_LIMIT_GIB = 48
DISK_TREE_LIMIT_GIB = 6
FREE_DISK_MIN_GIB = 20

# Production parity requirements discovered after the 2026-07-04 abort
# (F1 short/mid trained 17/20 features vs production 22 — schema violation):
# the plus7 production sleeves were trained with this env var exported by
# run_v89_plus7_retrain.sh / finish_long_plus7.sh. Mandatory for every fold.
HORIZON_ASSIGNMENT = "runtime/reports/v89_closed_loop/horizon_factor_assignment_plus7.json"
PROD_RUN = REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300"
# Long sleeve memory config: 8192/1024 and 4096/512 both FAILED in production
# history (_RETRAIN_COMPLETE_fail1 / _LONG_REDONE_fail); the only config that
# ever completed is batch 2048 + micro-batch 128 (finish_long_plus7.sh).
LONG_BATCH = {"batch_size": "2048", "micro_batch": 128}

COMMON = [
    "--dataset-path", DATASET,
    "--silver-panel-path", "runtime/data/v7/silver/market_panel/market_panel.parquet",
    "--symbols-file", "runtime/data/v7/universe_v88_comma.txt",
    "--train-start", "2018-01-02",
    "--embargo-days", "126",
    "--top-k", "30", "--max-epochs", "80",
    "--d-token", "256", "--n-blocks", "6", "--n-heads", "8", "--dates-per-step", "1",
    "--cross-sectional-norm", "rank", "--label-norm",
    "--attention-dropout", "0.25", "--ffn-dropout", "0.25", "--weight-decay", "0.001",
    "--early-stopping-patience", "8", "--learning-rate", "0.0005",
    "--feature-policy", "judgment", "--require-gpu",
]


def schema_parity_ok(fold: str, sleeve: str) -> tuple[bool, str]:
    """Protocol invariant: fold sleeve feature columns == production sleeve's."""
    try:
        prod = json.loads((PROD_RUN / sleeve / "ft" / "ft_transformer_feature_schema.json").read_text())
        mine = json.loads((OUT_ROOT / fold / sleeve / "ft" / "ft_transformer_feature_schema.json").read_text())
    except OSError as exc:
        return False, f"schema file missing: {exc}"
    if prod["feature_columns"] != mine["feature_columns"]:
        return False, (f"SCHEMA_MISMATCH prod={len(prod['feature_columns'])} "
                       f"fold={len(mine['feature_columns'])}")
    return True, "schema OK"


def git_hash() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=REPO).stdout.strip()


def rss_gib(pid: int) -> float:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return 0.0


def tree_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def log(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_one(fold: str, sleeve: str, micro_batch: int | None) -> tuple[bool, str, float, float]:
    """Return (ok, reason, runtime_s, rss_peak_gib)."""
    out_dir = OUT_ROOT / fold / sleeve
    batch_size = LONG_BATCH["batch_size"] if sleeve == "long_30d_120d" else "8192"
    cmd = [PY, "-m", "quantagent.cli", "train-v8-deep", "--horizon-class", sleeve,
           "--train-end", FOLDS[fold]["train_end"], "--test-end", FOLDS[fold]["test_end"],
           "--batch-size", batch_size,
           *COMMON, "--output-dir", str(out_dir)]
    if micro_batch:
        cmd += ["--train-micro-batch", str(micro_batch)]
    if sleeve == "long_30d_120d":
        # 90-feature sleeve sits at the 24G ceiling; checkpointing is the only
        # lever that reduces activation memory at dates_per_step=1 (see the
        # 2026-07-04 abort diagnoses in runner_ledger.jsonl).
        cmd += ["--activation-checkpointing"]
    logfile = OUT_ROOT / fold / f"{sleeve}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    # expandable_segments: the long sleeve (90 features → 91-token attention)
    # sits within ~400MB of the 24G VRAM ceiling and dies to allocator
    # fragmentation (observed: 22.72G allocated + 415MB reserved-unallocated,
    # 384MB request failed). NOTE --train-micro-batch is a NO-OP at
    # dates_per_step=1 (trainer splits by date group only) — kept for ledger
    # fidelity but it cannot reduce activation memory here.
    # QUANTAGENT_JUDGMENT_MAX_FACTORS=64: production parity — finish_long_plus7.sh
    # capped judgment factors to top-64 |ICIR| (178 -> 90 features for long;
    # no-op for short/mid which have <64). Without it the long sleeve trains a
    # DIFFERENT feature set (caught by the schema gate on 2026-07-04, 5.3h run
    # rejected: prod=90 fold=178).
    env = dict(os.environ,
               QUANTAGENT_HORIZON_ASSIGNMENT=HORIZON_ASSIGNMENT,
               QUANTAGENT_JUDGMENT_MAX_FACTORS="64",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    t0 = time.time()
    with logfile.open("w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, env=env)
        peak = 0.0
        while proc.poll() is None:
            time.sleep(30)
            r = rss_gib(proc.pid)
            peak = max(peak, r)
            if r > RAM_LIMIT_GIB:
                proc.kill()
                return False, f"RAM_GUARD {r:.1f}GiB", time.time() - t0, peak
    rt = time.time() - t0
    text = logfile.read_text(errors="ignore")
    if proc.returncode != 0:
        if "CUDA out of memory" in text or "OutOfMemoryError" in text:
            return False, "CUDA_OOM", rt, peak
        return False, f"EXIT_{proc.returncode}", rt, peak
    preds = out_dir / "predictions.parquet"
    if not preds.exists():
        return False, "NO_PREDICTIONS", rt, peak
    df = pd.read_parquet(preds, columns=["alpha_score"])
    if len(df) == 0 or df["alpha_score"].isna().all():
        return False, "EMPTY_OR_NAN_PREDICTIONS", rt, peak
    if df["alpha_score"].isna().mean() > 0.5:
        return False, "MAJORITY_NAN_PREDICTIONS", rt, peak
    ok_schema, schema_msg = schema_parity_ok(fold, sleeve)
    if not ok_schema:
        return False, schema_msg, rt, peak
    return True, "OK", rt, peak


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if ABORT.exists():
        print("ABORT marker present — refusing to start"); return 2
    gh = git_hash()
    for fold in FOLDS:
        for sleeve in SLEEVES:
            out_dir = OUT_ROOT / fold / sleeve
            if (out_dir / "predictions.parquet").exists():
                print(f"[skip] {fold}/{sleeve} already done", flush=True)
                continue
            free = shutil.disk_usage("/").free / (1024 ** 3)
            if free < FREE_DISK_MIN_GIB or tree_gib(OUT_ROOT) > DISK_TREE_LIMIT_GIB:
                ABORT.write_text("disk guard")
                print(f"DISK_GUARD free={free:.0f}G tree={tree_gib(OUT_ROOT):.1f}G", flush=True)
                return 3
            mb = LONG_BATCH["micro_batch"] if sleeve == "long_30d_120d" else None
            print(f"[start] {fold}/{sleeve} (micro_batch={mb})", flush=True)
            ok, reason, rt, peak = run_one(fold, sleeve, mb)
            if not ok and reason == "CUDA_OOM":
                retry_mb = (mb or 2048) // 2
                print(f"[retry] {fold}/{sleeve} after OOM with micro_batch={retry_mb}", flush=True)
                ok, reason, rt2, peak2 = run_one(fold, sleeve, retry_mb)
                rt += rt2; peak = max(peak, peak2)
                if not ok and reason == "CUDA_OOM":
                    reason = "DOUBLE_CUDA_OOM"
            gpu_peak = None
            mfile = out_dir / "ft" / "ft_transformer_metrics.json"
            if mfile.exists():
                try:
                    m = json.loads(mfile.read_text())
                    gpu_peak = m.get("gpu_peak_mb") or next(
                        (h.get("gpu_peak_mb") for h in m.get("training_history", [])
                         if isinstance(h, dict) and "gpu_peak_mb" in h), None)
                except Exception:
                    pass
            log({"fold": fold, "sleeve": sleeve, "ok": ok, "reason": reason,
                 "runtime_s": round(rt, 1), "rss_peak_gib": round(peak, 2),
                 "gpu_peak_mb": gpu_peak, "git": gh,
                 "train_end": FOLDS[fold]["train_end"], "test_end": FOLDS[fold]["test_end"],
                 "embargo_days": 126, "dataset": DATASET, "output": str(out_dir)})
            print(f"[done] {fold}/{sleeve} ok={ok} reason={reason} {rt/60:.0f}min "
                  f"rss {peak:.1f}G gpu {gpu_peak}MB", flush=True)
            if not ok:
                ABORT.write_text(f"{fold}/{sleeve}: {reason}")
                print(f"RUN ABORTED at {fold}/{sleeve}: {reason}", flush=True)
                return 4
    print("ALL FOLDS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
