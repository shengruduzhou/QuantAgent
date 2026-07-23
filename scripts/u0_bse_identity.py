#!/usr/bin/env python3
"""H-032B §6: BSE identity repair and code-mapping audit.

Corrects the H-032A over-classification. The master's 920xxx BSE codes are NOT
synthetic placeholders: the authoritative BSE name/code list (akshare, exchange-
derived) enumerates the CURRENT BSE universe entirely in the 920xxx series with
ZERO 8xxxxx codes, and TickFlow's instrument metadata recognises those 920xxx
symbols (e.g. 920002 = 万达轴承). BSE completed a code migration; 8xxxxx are
deprecated old codes. So an empty 8xxxxx response is NOT evidence that TickFlow
lacks BSE coverage — the current 920xxx code must be used.

Authoritative identity comes from the exchange list (akshare); TickFlow stays
the primary BAR provider. The only genuinely unavailable field is the old->new
code-migration table (historical_symbol), marked BLOCKED_BY_DATA — it does not
block current identity or bar coverage.

Outputs:
  runtime/data/u0/bse_code_mapping.parquet
  runtime/data/u0/bse_identity_audit.json

Usage: AI_quant_venv/bin/python3 scripts/u0_bse_identity.py --allow-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
OUT = REPO / "runtime/data/u0"
BLOCKED = "BLOCKED_BY_DATA"


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    import akshare as ak
    bj = ak.stock_info_bj_name_code()
    code_col = next(c for c in bj.columns if "代码" in c or "code" in c.lower())
    name_col = next((c for c in bj.columns if "简称" in c or "name" in c.lower()), None)
    list_col = next((c for c in bj.columns if "上市" in c or "list" in c.lower()), None)
    bj = bj.copy()
    bj["code"] = bj[code_col].astype(str).str.strip()
    bj["name"] = bj[name_col].astype(str) if name_col else None
    bj["listing_date"] = pd.to_datetime(bj[list_col], errors="coerce") if list_col else pd.NaT
    src_hash = _hash_df(bj[["code"]].sort_values("code"))

    master = pd.read_parquet(MASTER)
    mbse = master[master["board"] == "BSE"].copy()
    master_codes = set(mbse["code"].astype(str))
    ak_codes = set(bj["code"])

    all_920 = all(c.startswith("920") for c in ak_codes)
    any_8x = any(c.startswith("8") for c in ak_codes)

    rows = []
    for r in bj.itertuples():
        code = r.code
        cur_sym = f"{code}.BJ"
        in_master = code in master_codes
        # a code is a real BSE security iff it appears in the authoritative list;
        # none are rejected here — all 920xxx appear in the exchange list.
        identity_status = "AUTHORITATIVE_CURRENT" if all_920 else "UNVERIFIED"
        rows.append({
            "code": code,
            "old_code": BLOCKED,             # no authoritative old->new 8xxxxx map on hand
            "new_920_code": code,
            "current_symbol": cur_sym,
            "historical_symbol": BLOCKED,    # old 8xxxxx deprecated; migration table unavailable
            "listing_date": r.listing_date.date().isoformat() if pd.notna(r.listing_date) else None,
            "name": r.name,
            "in_u0_master": in_master,
            "identity_status": identity_status,
            "mapping_source": "akshare.stock_info_bj_name_code (exchange-derived)",
            "mapping_timestamp": now,
            "mapping_hash": src_hash,
        })
    mapping = pd.DataFrame.from_records(rows)
    mapping.to_parquet(OUT / "bse_code_mapping.parquet", index=False)

    missing_from_master = sorted(ak_codes - master_codes)
    placeholder_in_master = sorted(master_codes - ak_codes)   # codes NOT in authoritative list
    audit = {
        "generated": now, "experiment": "H-032B BSE identity repair",
        "authoritative_source": "akshare.stock_info_bj_name_code (exchange-derived)",
        "bar_provider": "TickFlow (920xxx current codes; primary bar source, unchanged)",
        "authoritative_bse_count": int(len(ak_codes)),
        "u0_master_bse_count": int(len(master_codes)),
        "all_authoritative_codes_are_920xxx": bool(all_920),
        "any_authoritative_8xxxxx_codes": bool(any_8x),
        "overlap": int(len(ak_codes & master_codes)),
        "in_authoritative_not_master": missing_from_master,
        "true_placeholder_codes_in_master": placeholder_in_master,
        "correction_of_h032a": (
            "H-032A marked BSE identity BLOCKED_BY_DATA because the master held only 920xxx and no "
            "8xxxxx, and probed 8xxxxx came back empty. That was an over-classification: the exchange "
            "list itself is entirely 920xxx with zero 8xxxxx — BSE migrated to the 920 series, so 920xxx "
            "IS the canonical current identity and the empty 8xxxxx responses are correct (deprecated "
            "codes), NOT evidence of missing BSE coverage."),
        "identity_decision": ("BSE_IDENTITY_CURRENT_RESOLVED" if all_920 and not placeholder_in_master
                              else "BSE_IDENTITY_PARTIAL"),
        "historical_symbol_migration": f"{BLOCKED} — old 8xxxxx->920xxx table not sourced (minor; does not block bars)",
        "dual_identity_prevented": ("mapping keys on the current 920xxx code only; old 8xxxxx is recorded "
                                    "as historical_symbol=BLOCKED, never emitted as a separate security"),
        "coverage_gap": {
            "recent_listings_missing_from_master": len(missing_from_master),
            "note": "add these currently-listed BSE names to the U0 master before bar backfill",
        },
        "blinding": "no candidate performance included",
    }
    (OUT / "bse_identity_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()
    if not args.allow_network:
        print("refusing: --allow-network not confirmed"); return 2
    a = build()
    print(json.dumps({k: a[k] for k in (
        "authoritative_bse_count", "u0_master_bse_count", "overlap",
        "all_authoritative_codes_are_920xxx", "any_authoritative_8xxxxx_codes",
        "true_placeholder_codes_in_master", "identity_decision",
        "in_authoritative_not_master")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
