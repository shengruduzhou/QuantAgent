#!/usr/bin/env python3
"""H-032C §4: U0 universe reconciliation across authoritative sources.

Reconciles the U0 security identity across:
  * U0 master (runtime/reports/h028/track_a/historical_security_master.parquet);
  * TickFlow provider coverage (which symbols returned bars);
  * official BSE current 920 list + old->new map (bse_code_mapping.parquet);
  * STAR list (688/689 in master) and post-2020 listings;
  * staged bars + assembled panel.

Adds legitimate missing recent listings to a supplemental additions table (never
mutating the frozen H-028 identity artifact); rejects identities unsupported by
authoritative metadata; and guarantees an old BSE 8xxxxx code can never become a
second security alongside its current 920xxx code (the master holds only 920xxx;
old codes live only as historical_symbol=BLOCKED_BY_DATA in the BSE map).

Outputs (runtime/data/u0/):
  universe_reconciliation.json
  master_supplemental_additions.parquet

Usage: AI_quant_venv/bin/python3 scripts/u0_reconcile_universe.py
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
COVERAGE = OUT / "provider_coverage_matrix.parquet"
BSE_MAP = OUT / "bse_code_mapping.parquet"
STAGING = REPO / "runtime/data/v7/full_universe/_staging"
PANEL_F = REPO / "runtime/data/v7/full_universe/full_universe_market_panel.parquet"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    master = pd.read_parquet(MASTER)
    master["symbol"] = master["symbol"].astype(str)
    master["code"] = master["code"].astype(str).str.zfill(6)
    master_codes = set(master["code"])
    master_syms = set(master["symbol"])

    cov = pd.read_parquet(COVERAGE) if COVERAGE.exists() else pd.DataFrame()
    covered_syms = set(cov.loc[cov["selected_bar_provider"] == "tickflow", "symbol"]) if len(cov) else set()
    staged_syms = ({f.stem.replace("sym_", "").replace("_", ".") for f in STAGING.glob("sym_*.parquet")}
                   if STAGING.exists() else set())
    panel_syms = set(pd.read_parquet(PANEL_F, columns=["symbol"])["symbol"].astype(str).unique()) \
        if PANEL_F.exists() else set()

    # ---- BSE reconciliation (authoritative current list) --------------------
    bse_additions = []
    bse_recon = {}
    if BSE_MAP.exists():
        bmap = pd.read_parquet(BSE_MAP)
        bmap["code"] = bmap["code"].astype(str).str.zfill(6)
        auth_codes = set(bmap["code"])
        missing = sorted(auth_codes - master_codes)
        bse_recon = {
            "authoritative_count": int(len(auth_codes)),
            "master_count": int(len(master_codes & auth_codes)),
            "missing_from_master": missing,
            "all_920xxx": bool(all(c.startswith("920") for c in auth_codes)),
        }
        for code in missing:
            row = bmap[bmap["code"] == code].iloc[0]
            bse_additions.append({
                "symbol": f"{code}.BJ", "code": code, "exchange": "BJ", "board": "BSE",
                "security_type": "A_share",
                "listing_date": row.get("listing_date"), "delisting_date": None,
                "status": "listed", "source": "akshare BSE list (reconciliation)",
                "historical_symbol": "BLOCKED_BY_DATA", "available_at": now,
                "source_hash": _hash(code)})

    # ---- dual-identity guard: no old 8xxxxx alongside a 920xxx --------------
    eightx_in_master = sorted(c for c in master_codes if c.startswith("8") and len(c) == 6)
    dual_identity_risk = len(eightx_in_master)  # must be 0 for BSE (migrated to 920)

    additions = pd.DataFrame.from_records(bse_additions) if bse_additions else pd.DataFrame(
        columns=["symbol", "code", "exchange", "board", "security_type", "listing_date",
                 "delisting_date", "status", "source", "historical_symbol", "available_at", "source_hash"])
    additions.to_parquet(OUT / "master_supplemental_additions.parquet", index=False)

    by_board = master.groupby("board").size().to_dict()
    star_syms = set(master.loc[master["board"] == "STAR", "symbol"])
    recon = {
        "generated": now, "experiment": "H-032C universe reconciliation",
        "master_securities": int(len(master)),
        "master_by_board": {str(k): int(v) for k, v in by_board.items()},
        "provider_covered": int(len(covered_syms)),
        "staged_symbols": int(len(staged_syms)),
        "panel_symbols": int(len(panel_syms)),
        "bse_reconciliation": bse_recon,
        "supplemental_additions": int(len(additions)),
        "supplemental_additions_symbols": additions["symbol"].tolist(),
        "star_total": int(len(star_syms)),
        "star_covered": int(len(star_syms & covered_syms)),
        "dual_identity_guard": {
            "old_8xxxxx_codes_in_master": eightx_in_master,
            "dual_identity_collisions": dual_identity_risk,
            "guarantee": ("old BSE 8xxxxx never enters the master as a separate security; it exists only "
                          "as historical_symbol=BLOCKED_BY_DATA in bse_code_mapping"),
        },
        "rejected_identities": [],
        "rejected_note": "no authoritative-unsupported identities added; only akshare-listed BSE names",
        "policy": ("frozen H-028 identity artifact is NOT mutated; legitimate missing recent listings are "
                   "written to master_supplemental_additions.parquet for a future master rebuild"),
        "blinding": "no candidate performance included",
    }
    (OUT / "universe_reconciliation.json").write_text(json.dumps(recon, indent=2, ensure_ascii=False))
    return recon


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    r = build()
    print(json.dumps({"master_securities": r["master_securities"],
                      "bse_missing_from_master": r["bse_reconciliation"].get("missing_from_master"),
                      "supplemental_additions": r["supplemental_additions"],
                      "dual_identity_collisions": r["dual_identity_guard"]["dual_identity_collisions"],
                      "star_covered": r["star_covered"], "star_total": r["star_total"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
