#!/usr/bin/env python3
"""Stage 10.4b — daily health check. Run after the cron pipeline.

Emits a one-glance status of the day's PIT snapshot + verification + paper-trade,
so silent degradation (throttled source, stale labels, failed portfolio) is
visible. Writes health.json into the snapshot and appends to a rolling health log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("runtime/stage10_concept")
SNAPS = ROOT / "snapshots"
PT = ROOT / "paper_trades"


def _latest():
    ds = sorted(d.name for d in SNAPS.glob("*") if (d / "manifest.json").exists())
    return ds[-1] if ds else None


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else _latest()
    if not date:
        print("HEALTH: no snapshot"); return 1
    snap = SNAPS / date
    man = json.loads((snap / "manifest.json").read_text()) if (snap / "manifest.json").exists() else {}
    h = {"asof": date, "checked_at": pd.Timestamp.now().isoformat()}

    h["concept_board_ok"] = (snap / "concept_boards.parquet").exists()
    h["n_concepts_scored"] = man.get("n_strength_scored")
    hf = snap / "concept_hardness.csv"
    h["n_hardness_stocks"] = int(len(pd.read_csv(hf))) if hf.exists() else 0

    vf = snap / "concept_hardness_verified.csv"
    if vf.exists():
        v = pd.read_csv(vf, dtype={"code": str})
        ls = v["label_status"].value_counts().to_dict() if "label_status" in v else {}
        h["announcements_verified"] = int(ls.get("verified", 0))
        h["labels_stale"] = int(ls.get("stale", 0))
        h["labels_offline"] = int(ls.get("offline", 0))
        conf = v.get("purity_confidence", pd.Series(dtype=str)).astype(str)
        src = v.get("concept_purity_source", pd.Series(dtype=str)).astype(str)
        h["purity_verified"] = int(conf.isin(["high", "medium", "low"]).sum())   # real exposure matched
        h["purity_fetched_nomatch"] = int(((src == "revenue_breakdown") & (conf == "none")).sum())
        h["purity_unknown"] = int((src == "unknown").sum())
        h["label_dist"] = v["order_label"].value_counts().to_dict() if "order_label" in v else {}
        h["performance_mismatch"] = int((v.get("order_label", pd.Series(dtype=str)) == "performance_mismatch").sum())
    else:
        h["announcements_verified"] = h["labels_stale"] = h["purity_verified"] = 0
        h["purity_unknown"] = h.get("n_hardness_stocks", 0)

    pf = PT / f"portfolio_{date}.csv"
    h["paper_portfolio_ok"] = pf.exists()
    h["paper_portfolio_n"] = int(len(pd.read_csv(pf))) if pf.exists() else 0

    # beta decomposition status (from paper-trade summary)
    ts = PT / "track_summary.json"
    beta_status = "not_run"
    if ts.exists():
        s = json.loads(ts.read_text())
        beta_status = "done" if s.get("status") == "ok" else s.get("status", "unknown")
    h["beta_decomposition"] = beta_status
    h["benchmark_updated"] = ts.exists()

    h["used_cache"] = bool(man.get("used_cache", False))
    # source-throttle heuristic: live attempted but nothing verified
    h["data_source_throttled"] = bool(man.get("used_cache") or
                                      (vf.exists() and h.get("announcements_verified", 0) == 0
                                       and h.get("purity_verified", 0) == 0))

    (snap / "health.json").write_text(json.dumps(h, ensure_ascii=False, indent=2))
    logf = ROOT / "health_log.jsonl"
    with logf.open("a") as fh:
        fh.write(json.dumps(h, ensure_ascii=False) + "\n")

    print(f"=== Stage10 HEALTH {date} ===")
    print(f"  concept board pulled : {h['concept_board_ok']}  ({h['n_concepts_scored']} concepts)")
    print(f"  hardness stocks      : {h['n_hardness_stocks']}")
    print(f"  公告 verified        : {h['announcements_verified']}  (stale {h['labels_stale']}, offline {h.get('labels_offline',0)})")
    print(f"  主营纯度 verified    : {h['purity_verified']}  (unknown {h['purity_unknown']})")
    print(f"  performance_mismatch : {h.get('performance_mismatch',0)}")
    print(f"  paper portfolio      : {h['paper_portfolio_ok']}  ({h['paper_portfolio_n']} stocks)")
    print(f"  benchmark updated    : {h['benchmark_updated']}  (beta decomp: {h['beta_decomposition']})")
    print(f"  cache reused         : {h['used_cache']}")
    print(f"  data source throttled: {h['data_source_throttled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
