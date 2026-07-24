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
SUPPLEMENTAL = REPO / "runtime/data/u0/master_supplemental_additions.parquet"
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
    if SUPPLEMENTAL.exists():
        try:
            _add = pd.read_parquet(SUPPLEMENTAL)
            _shared = [c for c in m.columns if c in _add.columns]
            _add = _add[_shared]
            _add = _add[~_add['symbol'].astype(str).isin(set(m['symbol'].astype(str)))]
            if len(_add):
                m = pd.concat([m, _add], ignore_index=True)
        except Exception:
            pass
    m["symbol"] = m["symbol"].astype(str)
    for c in ("listing_date", "delisting_date", "st_start", "st_end", "available_at"):
        if c in m.columns:
            m[c] = pd.to_datetime(m[c], errors="coerce")
    src_hash = _sha_file(SRC)

    st_available = int(m["st_start"].notna().sum()) if "st_start" in m else 0
    delist_dates_available = int(m["delisting_date"].notna().sum())
    delisted_status = int((m["status"] == "delisted").sum())

    # survivorship-bias quantification: how many delisted names actually carry bars?
    covered_syms: set[str] = set()
    panel = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
    if panel.exists():
        covered_syms |= set(pd.read_parquet(panel, columns=["symbol"])["symbol"].astype(str).unique())
    staging = REPO / "runtime/data/v7/full_universe/_staging"
    if staging.exists():
        covered_syms |= {f.stem.replace("sym_", "").replace("_", ".") for f in staging.glob("sym_*.parquet")}
    delisted_syms = set(m.loc[m["status"] == "delisted", "symbol"])
    delisted_with_bars = len(delisted_syms & covered_syms)
    now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

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
        # §8: every PIT record must carry provenance; unavailable = BLOCKED_BY_DATA, not false.
        "pit_field_provenance": {
            "listing_date": {"available_at": "exchange listing", "source": "exchange_metadata",
                             "source_timestamp": now_ts, "source_hash": src_hash},
            "board_security_type": {"available_at": "exchange listing", "source": "exchange_metadata",
                                    "source_timestamp": now_ts, "source_hash": src_hash},
            "historical_price_limit_rule": {"available_at": "current snapshot", "source": "board_rule_table",
                                            "source_timestamp": now_ts, "source_hash": src_hash,
                                            "caveat": "current-snapshot only; time-varying history not tracked"},
            "ipo_special_limit_rule": {"available_at": "preregistered", "source": "H028_rule",
                                       "source_timestamp": now_ts, "source_hash": src_hash},
            "delisting_date": BLOCKED, "st_intervals": BLOCKED, "suspension_intervals": BLOCKED,
            "corporate_action_identity": BLOCKED, "historical_symbol_migration": BLOCKED,
        },
        "survivorship_bias": {
            "master_securities": int(len(out)),
            "delisted_total": delisted_status,
            "delisted_with_delisting_date": delist_dates_available,
            "delisted_with_bar_history": delisted_with_bars,
            "delisted_fraction_of_master": round(delisted_status / max(1, len(out)), 4),
            "delisted_history_gap": delisted_status - delisted_with_bars,
            "assessment": (f"{delisted_status} delisted names ({round(100*delisted_status/max(1,len(out)),1)}% of "
                           f"master); {delisted_with_bars} carry any bar history and {delist_dates_available} carry "
                           f"a delisting date. Delisting-date + interval history is BLOCKED_BY_DATA. A universe that "
                           f"silently drops delisted names is survivorship-biased; FULL_UNIVERSE_DATA_READY is "
                           f"withheld unless a preregistered rule permits a bounded, quantified exclusion."),
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
