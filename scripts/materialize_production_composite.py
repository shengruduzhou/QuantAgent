#!/usr/bin/env python3
"""Materialize the production composite score from configs/production_blend.json.

ONE command reproduces the production prediction artifact with full provenance:

    AI_quant_venv/bin/python3 scripts/materialize_production_composite.py \
        --verify-against runtime/reports/v89_closed_loop/ensemble_search_plus7/winner_predictions.parquet

Modes
-----
* default: read ``<model_run>/ensemble_composite.parquet`` per-sleeve score
  columns (exact frozen inputs of the original blend search).
* ``--from-sleeves``: rebuild the sleeve score columns from each sleeve's
  ``predictions.parquet`` (outer merge, NaN->0), replicating
  ``training.horizon_models.ensemble_horizon_predictions`` — for fresh
  closed-loop retrains where ensemble_composite.parquet does not exist yet.

The blend itself is an exact replica of ``ensemble_weight_search._ranked_sleeves``:
per trade_date percentile rank of each sleeve score, then weighted sum.

Every run writes ``<out>.manifest.json`` with git hash, argv, config echo,
input/output sha256, and the verification result. Trust class is copied from
the config — this artifact inherits the production selection's classification
(currently likely_overfit); materializing it does not launder it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "production_blend.json"
SLEEVE_ORDER = ("short_5d", "mid_5d_30d", "long_30d_120d")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_sleeve_frame(model_run: Path, cfg: dict, from_sleeves: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (frame with trade_date/symbol/{sleeve}_score columns, input hashes)."""
    inputs: dict[str, str] = {}
    if not from_sleeves:
        src = model_run / cfg.get("composite_source", "ensemble_composite.parquet")
        cols = ["trade_date", "symbol"] + [f"{s}_score" for s in SLEEVE_ORDER]
        frame = pd.read_parquet(src, columns=cols)
        inputs[str(src)] = sha256_file(src)
        return frame, inputs
    # Rebuild from per-sleeve predictions (replicates ensemble_horizon_predictions:
    # keep key+alpha_score, rename to {sleeve}_score, outer-merge in sleeve order,
    # numeric-coerce and fill NaN with 0.0).
    merged: pd.DataFrame | None = None
    for sleeve in SLEEVE_ORDER:
        rel = cfg.get("sleeve_predictions", {}).get(sleeve, f"{sleeve}/predictions.parquet")
        src = model_run / rel
        if not src.exists():
            print(f"[skip] {sleeve}: missing {src}", flush=True)
            continue
        f = pd.read_parquet(src, columns=["trade_date", "symbol", "alpha_score"])
        f = f.rename(columns={"alpha_score": f"{sleeve}_score"})
        inputs[str(src)] = sha256_file(src)
        merged = f if merged is None else merged.merge(f, on=["trade_date", "symbol"], how="outer")
    if merged is None:
        raise FileNotFoundError(f"no sleeve predictions found under {model_run}")
    for sleeve in SLEEVE_ORDER:
        col = f"{sleeve}_score"
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged, inputs


def blend(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Exact replica of ensemble_weight_search._ranked_sleeves + weighted sum."""
    out = frame[["trade_date", "symbol"]].copy()
    ranked: dict[str, pd.Series] = {}
    for sleeve in SLEEVE_ORDER:
        col = f"{sleeve}_score"
        if col in frame.columns:
            ranked[sleeve] = frame.groupby("trade_date")[col].rank(pct=True)
        else:
            ranked[sleeve] = pd.Series(0.0, index=frame.index)
    score = (weights.get("short_5d", 0.0) * ranked["short_5d"]
             + weights.get("mid_5d_30d", 0.0) * ranked["mid_5d_30d"]
             + weights.get("long_30d_120d", 0.0) * ranked["long_30d_120d"])
    out["composite_score"] = score.to_numpy()
    return out


def verify(produced: pd.DataFrame, reference_path: Path) -> dict:
    ref = pd.read_parquet(reference_path)
    report: dict[str, object] = {"reference": str(reference_path), "reference_sha256": sha256_file(reference_path)}
    a = produced.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    b = ref.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    if len(a) != len(b):
        report["identical_values"] = False
        report["diff"] = f"row count {len(a)} vs {len(b)}"
        return report
    keys_equal = a[["trade_date", "symbol"]].equals(b[["trade_date", "symbol"]])
    diff = np.abs(a["composite_score"].to_numpy() - b["composite_score"].to_numpy())
    report["keys_identical"] = bool(keys_equal)
    report["max_abs_score_diff"] = float(diff.max())
    report["identical_values"] = bool(keys_equal and diff.max() == 0.0)
    report["note"] = ("value-identical; parquet bytes may differ due to writer/compression metadata"
                      if report["identical_values"] else "VALUES DIFFER — investigate before use")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--model-run", default=None, help="override config model_run (e.g. a fresh retrain dir)")
    ap.add_argument("--out", default=None, help="output parquet (default <model_run>/production_composite.parquet)")
    ap.add_argument("--from-sleeves", action="store_true", help="rebuild sleeve scores from <sleeve>/predictions.parquet")
    ap.add_argument("--verify-against", default=None, help="compare produced values to a reference parquet")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_run = Path(args.model_run or cfg["model_run"])
    if not model_run.is_absolute():
        model_run = REPO / model_run
    out_path = Path(args.out) if args.out else model_run / "production_composite.parquet"

    frame, inputs = load_sleeve_frame(model_run, cfg, args.from_sleeves)
    produced = blend(frame, cfg["blend"]["weights"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    produced.to_parquet(out_path, index=False)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "argv": sys.argv,
        "config_path": str(args.config),
        "config_echo": {k: cfg[k] for k in ("model_run", "blend", "top_k", "evaluator", "trust") if k in cfg},
        "model_run": str(model_run),
        "mode": "from_sleeves" if args.from_sleeves else "from_composite",
        "inputs_sha256": inputs,
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "rows": int(len(produced)),
        "trust_class": cfg.get("trust", {}).get("class", "unknown"),
    }
    if args.verify_against:
        manifest["verification"] = verify(produced, Path(args.verify_against))
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out_path} ({len(produced):,} rows)")
    print(f"manifest {manifest_path}")
    if "verification" in manifest:
        v = manifest["verification"]
        print(f"verification: identical_values={v['identical_values']} "
              f"max_abs_diff={v.get('max_abs_score_diff')}")
        return 0 if v["identical_values"] else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
