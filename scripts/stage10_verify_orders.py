#!/usr/bin/env python3
"""Stage 10.3 — order/公告/主营 verification → hard labels on a PIT snapshot (fail-soft).

Assigns each top in-concept stock an `order_label` (confirmed_order /
confirmed_customer / revenue_exposure / earnings_verified / rumor_only /
fake_concept / performance_mismatch) and verified 概念纯度 fields, then re-scores
hardness. Designed to run inside the daily cron:

  * offline (default): earnings-only labels from cached yjbb/yjyg (no network)
  * --live: fetch 公告 (巨潮 primary, 东财 aux) + 主营构成 for the top names per
    concept; on any per-stock failure keep the PREVIOUS snapshot's label and mark
    label_status=stale; after repeated failures it declares the source throttled
    and stops hitting the network — the scan never breaks.

Writes concept_hardness_verified.csv into the same (PIT) snapshot.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.concept import announcements as anndb  # noqa: E402
from quantagent.concept import purity as puritydb  # noqa: E402
from quantagent.concept import scoring  # noqa: E402

SNAPS = Path("runtime/stage10_concept/snapshots")
TOP_PER_CONCEPT = 6
THROTTLE_AFTER = 3   # consecutive live failures -> declare source throttled, go fail-soft fast


def _snap_dates():
    return sorted(d.name for d in SNAPS.glob("*") if (d / "concept_hardness.csv").exists())


def _prev_labels(date: str) -> pd.DataFrame:
    prev = [d for d in _snap_dates() if d < date]
    for d in reversed(prev):
        f = SNAPS / d / "concept_hardness_verified.csv"
        if f.exists():
            return pd.read_csv(f, dtype={"code": str})
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    date = args.date or (_snap_dates()[-1] if _snap_dates() else None)
    if not date:
        print("no snapshots"); return 1
    snap = SNAPS / date
    h = pd.read_csv(snap / "concept_hardness.csv", dtype={"code": str})
    h["code"] = h["code"].str.zfill(6)

    # 1) offline earnings-based labels (always; no network)
    h["order_label"] = anndb.label_hardness_offline(h)
    h["label_status"] = "offline"
    for c in ["revenue_exposure_pct", "concept_purity_source", "purity_confidence", "last_verified_date"]:
        h[c] = [None] * len(h)
    h["concept_purity_source"] = "unknown"
    h["purity_confidence"] = "none"

    stats = {"verified": 0, "stale": 0, "purity_verified": 0, "throttled": False, "src_counts": {}}
    if args.live:
        prev = _prev_labels(date)
        prev_lab = prev.set_index("code")["order_label"].to_dict() if not prev.empty else {}
        prev_pur = prev.set_index("code")[["revenue_exposure_pct", "purity_confidence", "last_verified_date"]].to_dict("index") if not prev.empty and "revenue_exposure_pct" in prev.columns else {}
        targets = (h.sort_values("hardness", ascending=False)
                   .groupby("board", group_keys=False).head(TOP_PER_CONCEPT))
        fails = 0
        for code in targets["code"].unique():
            board = h[h["code"] == code].iloc[0]["board"]
            if stats["throttled"]:
                # source throttled: stale fallback
                if code in prev_lab:
                    h.loc[h["code"] == code, "order_label"] = prev_lab[code]
                    h.loc[h["code"] == code, "label_status"] = "stale"
                    stats["stale"] += 1
                continue
            titles, src = anndb.fetch_announcements(code, allow_network=True)
            zygc = puritydb.fetch_zygc(code, allow_network=True)
            if not titles and zygc is None:           # both failed -> count toward throttle
                fails += 1
                if code in prev_lab:
                    h.loc[h["code"] == code, "order_label"] = prev_lab[code]
                    if code in prev_pur:
                        h.loc[h["code"] == code, "revenue_exposure_pct"] = prev_pur[code].get("revenue_exposure_pct")
                        h.loc[h["code"] == code, "purity_confidence"] = prev_pur[code].get("purity_confidence")
                        h.loc[h["code"] == code, "last_verified_date"] = prev_pur[code].get("last_verified_date")
                        h.loc[h["code"] == code, "concept_purity_source"] = "stale"
                    h.loc[h["code"] == code, "label_status"] = "stale"
                    stats["stale"] += 1
                if fails >= THROTTLE_AFTER:
                    stats["throttled"] = True
                continue
            fails = 0
            row = h[h["code"] == code].iloc[0]
            pr = puritydb.purity_record(zygc, board, asof=date)
            ev = anndb.classify_announcement_titles(titles) if titles else {}
            lab = anndb.derive_order_label(ann_evidence=ev, profit_yoy=row.get("profit_yoy"),
                                           yj_forecast=row.get("yj_forecast"),
                                           purity_score=row.get("score_purity"),
                                           revenue_exposure_pct=pr["revenue_exposure_pct"])
            h.loc[h["code"] == code, "order_label"] = lab
            h.loc[h["code"] == code, "label_status"] = "verified"
            for k, v in pr.items():
                h.loc[h["code"] == code, k] = v
            stats["verified"] += 1
            if pr["concept_purity_source"] == "revenue_breakdown":
                stats["purity_verified"] += 1
            if src:
                stats["src_counts"][src] = stats["src_counts"].get(src, 0) + 1
            time.sleep(0.4)

    # 2) re-score with labels + verified purity
    h = scoring.score_hardness(h)
    h["role"] = scoring.classify_role(h)
    h["不买理由"] = h.apply(scoring.buy_reject_reason, axis=1)
    h = h.sort_values(["board", "hardness"], ascending=[True, False]).reset_index(drop=True)
    h.to_csv(snap / "concept_hardness_verified.csv", index=False)

    print(f"[verify {date}] labels: {h['order_label'].value_counts().to_dict()}")
    print(f"  status: {h['label_status'].value_counts().to_dict()}")
    print(f"  live verified={stats['verified']} stale={stats['stale']} "
          f"purity_verified={stats['purity_verified']} throttled={stats['throttled']} src={stats['src_counts']}")
    print(f"[write] {snap/'concept_hardness_verified.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
