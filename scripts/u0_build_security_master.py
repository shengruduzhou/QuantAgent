#!/usr/bin/env python3
"""H-031 Track U0: normalised historical security master (build-u0-security-master).

Normalises the authoritative exchange-derived master
(runtime/reports/h028/track_a/historical_security_master.parquet) into the
H-031 §7 schema and records, honestly, which point-in-time identity fields are
actually available vs BLOCKED_BY_DATA. It never fabricates missing ST,
suspension, delisting-date or corporate-action intervals.

Output: runtime/data/u0/historical_security_master.parquet
        runtime/data/u0/pit_field_availability.json

Usage: AI_quant_venv/bin/python3 scripts/u0_build_security_master.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
OUT = REPO / "runtime/data/u0"
IPO_SPECIAL_LIMIT_DAYS = 60   # preregistered H-028 IPO ineligibility window

BLOCKED = "BLOCKED_BY_DATA"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(SRC)
    m["symbol"] = m["symbol"].astype(str)
    for c in ("listing_date", "delisting_date", "st_start", "st_end", "available_at"):
        if c in m.columns:
            m[c] = pd.to_datetime(m[c], errors="coerce")
    src_hash = _sha_file(SRC)

    st_available = int(m["st_start"].notna().sum()) if "st_start" in m else 0
    delist_dates_available = int(m["delisting_date"].notna().sum())
    delisted_status = int((m["status"] == "delisted").sum())

    out = pd.DataFrame({
        "symbol": m["symbol"],
        # no historical code-migration table on disk (BSE/older-code remaps unknown)
        "historical_symbol": m["symbol"],
        "exchange": m["exchange"],
        "board": m["board"],
        "security_type": m["security_type"],
        "listing_date": m["listing_date"],
        "delisting_date": m["delisting_date"],
        # status_start = listing (only status transition we can date); status_end
        # = delisting_date when known, else BLOCKED_BY_DATA (status carried, date not)
        "status_start": m["listing_date"],
        "status_end": m["delisting_date"],
        "status_end_blocked": (m["status"] == "delisted") & m["delisting_date"].isna(),
        # ST / suspension intervals: no PIT source on disk -> empty + explicit block
        "st_intervals": [[] for _ in range(len(m))],
        "st_intervals_blocked": True if st_available == 0 else False,
        "suspension_intervals": [[] for _ in range(len(m))],
        "suspension_intervals_blocked": True,
        # price-limit rule: current rule known; time-varying history not tracked
        "historical_price_limit_rule": m["price_limit_rule"],
        "price_limit_rule_is_current_snapshot": True,
        "ipo_special_limit_rule": f"first_{IPO_SPECIAL_LIMIT_DAYS}_td_ineligible(preregistered_H028)",
        # corporate-action identity: not verified for this cohort
        "corporate_action_identity": BLOCKED,
        "available_at": m["available_at"] if "available_at" in m else pd.NaT,
        "source": m["source"],
        "source_hash": m["source_hash"] if "source_hash" in m else src_hash,
    })
    out.to_parquet(OUT / "historical_security_master.parquet", index=False)

    availability = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "H-031 Track U0 security master",
        "securities": int(len(out)),
        "master_source": str(SRC.relative_to(REPO)),
        "master_source_hash": src_hash,
        "by_board": out["board"].value_counts().to_dict(),
        "by_status": m["status"].value_counts().to_dict(),
        "pit_field_availability": {
            "identity_symbol": "AVAILABLE",
            "exchange_board_security_type": "AVAILABLE",
            "listing_date": f"AVAILABLE ({int(out['listing_date'].notna().sum())}/{len(out)})",
            "delisting_date": (f"{BLOCKED} — {delist_dates_available}/{delisted_status} delisted names "
                               f"carry a status but no date"),
            "st_intervals": f"{BLOCKED} — {st_available} rows with ST dates on disk (no PIT ST source)",
            "suspension_intervals": f"{BLOCKED} — no PIT suspension source on disk",
            "historical_price_limit_rule": "PARTIAL — current-snapshot rule only, not time-varying history",
            "ipo_special_limit_rule": "AVAILABLE (preregistered 60-td window)",
            "corporate_action_identity": f"{BLOCKED} — not verified for the full-universe cohort",
            "historical_symbol_migration": f"{BLOCKED} — no code-migration table (BSE/older-code remaps)",
        },
        "honesty_note": ("Missing ST / suspension / delisting-date / corporate-action intervals are marked "
                         "BLOCKED_BY_DATA, never fabricated or defaulted to false. This master is IDENTITY-"
                         "complete but PIT-execution-incomplete."),
    }
    (OUT / "pit_field_availability.json").write_text(json.dumps(availability, indent=2))
    return availability


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    a = build()
    print(json.dumps({"securities": a["securities"], "by_board": a["by_board"],
                      "pit_field_availability": a["pit_field_availability"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
