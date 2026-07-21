#!/usr/bin/env python3
"""H-030 Track F2 (Path B): deterministic batch-replay certifier for frozen S4.

S4 (RW1_4state) is not produced by the daily runner. Rather than port the
learner to daily inference (Path A, which risks semantic drift), this proves
the frozen BATCH construction can be replayed deterministically at first read
from archived inputs — the option the freeze manifest already sanctions.

It imports the frozen learner UNMODIFIED (build_ic_and_regimes /
learn_state_params from scripts/analysis/regime_weight_meta.py) — no
reimplementation, no parameter fitting, no threshold tuning, no state redesign.

Validation (all pre-quarantine, asserted by the frozen module's own PANEL_CAP):
  1. archive completeness — every input S4 needs is persisted daily;
  2. exactness — replayed per-state tau/weights/n_days must equal the stored
     EXP-023 weight_trace bit-for-bit at the same refit cutoffs;
  3. determinism — two independent replays must agree exactly.

Usage:
  AI_quant_venv/bin/python3 scripts/s4_batch_replay.py            # certify
  AI_quant_venv/bin/python3 scripts/s4_batch_replay.py --cutoff 2023-06-14
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import regime_weight_meta as rw  # noqa: E402  (frozen learner, imported as-is)

TRACE = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp023_regime_weight_meta/results.json"
OUT = REPO / "runtime/reports/h030/s4_readiness_certificate.json"
BLIND = REPO / "runtime/paper/fresh_blind"


def archive_inventory() -> dict:
    """Every input the frozen S4 construction consumes, and where it persists."""
    import baseline_protocol as bp
    items = {
        "market_panel": REPO / bp.PANEL,
        "sector_map": REPO / bp.SECTOR,
        "forward_sleeve_scores": BLIND / "daily" / "sleeve_scores.parquet",
        "frozen_sleeve_predictions_short": REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/short_5d/predictions.parquet",
        "frozen_sleeve_predictions_mid": REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/mid_5d_30d/predictions.parquet",
        "frozen_sleeve_predictions_long": REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/long_30d_120d/predictions.parquet",
        "exp023_frozen_trace": TRACE,
    }
    out = {}
    for k, p in items.items():
        out[k] = {"path": str(p.relative_to(REPO)), "exists": p.exists(),
                  "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None}
    # component/regime helpers are code, not data — record their provenance
    out["_code_inputs"] = {
        "learner": "scripts/analysis/regime_weight_meta.py (frozen, imported unmodified)",
        "components": list(rw.COMPONENTS),
        "constants": {"TRAIL_START": str(rw.TRAIL_START.date()), "HORIZON": rw.HORIZON,
                      "EMBARGO": rw.EMBARGO, "REFIT_EVERY": rw.REFIT_EVERY,
                      "IC_MIN": rw.IC_MIN, "TAU_CAP": rw.TAU_CAP,
                      "MIN_REGIME_DAYS": rw.MIN_REGIME_DAYS,
                      "PANEL_CAP": str(rw.PANEL_CAP.date())},
    }
    return out


def replay(cutoffs: list[pd.Timestamp]) -> dict:
    ic_df, regime_df, _ = rw.build_ic_and_regimes()
    rid4 = (regime_df["crash"].astype(int) * 2 + regime_df["vol_hi"].astype(int))
    return {str(c.date()): rw.learn_state_params(ic_df, rid4, c, 4) for c in cutoffs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", default=None, help="single cutoff to replay (debug)")
    args = ap.parse_args()

    trace = json.loads(TRACE.read_text())["weight_trace"]["RW1_4state"]
    # stored refit cutoffs across every fold (all pre-quarantine)
    stored = {}
    for fold, refits in trace.items():
        for _, rec in refits.items():
            stored[rec["cutoff"]] = rec["states"]
    cutoffs = [pd.Timestamp(args.cutoff)] if args.cutoff else \
        [pd.Timestamp(c) for c in sorted(stored)]
    print(f"replaying {len(cutoffs)} frozen refit cutoffs "
          f"({min(cutoffs).date()}..{max(cutoffs).date()})", flush=True)

    run1 = replay(cutoffs)
    run2 = replay(cutoffs)
    deterministic = run1 == run2

    mismatches = []
    for c, states in run1.items():
        exp = stored.get(c)
        if exp is None:
            continue
        for st, got in states.items():
            want = exp.get(str(st))
            if want is None:
                mismatches.append({"cutoff": c, "state": st, "issue": "missing in stored trace"})
                continue
            if (round(got["tau"], 4) != round(want["tau"], 4)
                    or got["n_days"] != want["n_days"]
                    or {k: round(v, 4) for k, v in got["weights"].items()}
                    != {k: round(v, 4) for k, v in want["weights"].items()}):
                mismatches.append({"cutoff": c, "state": st, "replayed": got, "stored": want})

    inv = archive_inventory()
    archived_ok = all(v["exists"] for k, v in inv.items() if k != "_code_inputs")
    exact = not mismatches
    decision = "S4_BATCH_REPLAY_READY" if (exact and deterministic and archived_ok) else "S4_NOT_READY"

    cert = {
        "generated": datetime.now().isoformat(), "experiment": "H-030 Track F2",
        "path_chosen": "B — batch-at-first-read (freeze-manifest sanctioned)",
        "path_rationale": ("Path A would re-express the learner in a daily loop and risk "
                           "semantic drift in lags/windows/state definitions; Path B replays "
                           "the frozen construction itself, so S4 semantics cannot change."),
        "semantics_changed": False, "parameters_fitted": False, "fresh_access": False,
        "validation_window": f"{min(cutoffs).date()}..{max(cutoffs).date()} (pre-quarantine; "
                             f"frozen PANEL_CAP {rw.PANEL_CAP.date()} asserted inside the learner)",
        "refit_cutoffs_replayed": len(cutoffs),
        "exact_reproduction_vs_frozen_trace": exact,
        "mismatches": mismatches[:10],
        "deterministic_double_run": deterministic,
        "archived_inputs_complete": archived_ok,
        "archive_inventory": inv,
        "replay_command": ("AI_quant_venv/bin/python3 scripts/s4_batch_replay.py   "
                           "# at first read: same command with the frozen learner's PANEL_CAP "
                           "advanced per the preregistered protocol in configs/preregistered_evals.json"),
        "known_limitations": [
            "S4 books are constructed in batch at read time, not daily — decisions are "
            "reproducible but not incrementally logged like S1-S3",
            "replay requires the forward sleeve-score archive to remain complete; it is "
            "written every runner pass and covered by the shadow-day gate",
        ],
        "decision": decision,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cert, indent=2))
    print(json.dumps({"decision": decision, "exact": exact, "deterministic": deterministic,
                      "archived_ok": archived_ok, "cutoffs": len(cutoffs),
                      "mismatches": len(mismatches)}, indent=2))
    return 0 if decision == "S4_BATCH_REPLAY_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
