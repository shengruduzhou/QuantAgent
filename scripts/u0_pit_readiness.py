#!/usr/bin/env python3
"""H-032B §9/§10: PIT field source audit + strict point-in-time readiness.

Independently classifies whether TickFlow (the primary bar provider) supplies
each historical PIT field, using the live capability findings recorded in
runtime/reports/h032b/. It NEVER infers historical intervals from current
metadata and NEVER defaults an unavailable status to false — absent sources are
BLOCKED_BY_DATA or ALTERNATIVE_SOURCE_REQUIRED.

Verified TickFlow capability (2026-07-23, paid API):
  * klines carry only OHLCV + identity (symbol/name/timestamp/trade_date) — no
    ST/suspension/limit/status flags;
  * instrument.ext is a CURRENT snapshot (listing_date, current limit_up/down,
    shares, tick_size) — no delisting date, no historical intervals;
  * ex_factors (adjustment / corporate-action identity) is NOT entitled
    (PermissionError 无除权因子查询权限).

Outputs:
  runtime/data/u0/pit_source_audit.json
  runtime/data/u0/pit_field_availability.json   (rewritten with §10 provenance)
  runtime/data/u0/u0_strict_pit_certificate.json

Usage: AI_quant_venv/bin/python3 scripts/u0_pit_readiness.py
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
BENCH = REPO / "runtime/reports/h032b/tickflow_capability_benchmark.json"
BSE_AUDIT = OUT / "bse_identity_audit.json"
BLOCKED = "BLOCKED_BY_DATA"

# §9 vocabulary
TICKFLOW_AVAILABLE = "TICKFLOW_AVAILABLE"
TICKFLOW_CURRENT_ONLY = "TICKFLOW_CURRENT_ONLY"
EXCHANGE_SOURCE_AVAILABLE = "EXCHANGE_SOURCE_AVAILABLE"
ALTERNATIVE_SOURCE_REQUIRED = "ALTERNATIVE_SOURCE_REQUIRED"

# Verified field-by-field source classification (from the live capability probe).
PIT_SOURCE = {
    "st_intervals": {
        "tickflow": ALTERNATIVE_SOURCE_REQUIRED,
        "reason": "no ST flag in klines; instrument.ext is current-only",
        "alternative": "akshare/exchange ST history (not yet materialised)",
        "readiness": BLOCKED},
    "suspension_intervals": {
        "tickflow": TICKFLOW_CURRENT_ONLY,
        "reason": "suspensions are only inferable from bar gaps; no authoritative interval field",
        "alternative": "exchange suspension notices / akshare (not yet materialised)",
        "readiness": BLOCKED},
    "delisting_intervals": {
        "tickflow": ALTERNATIVE_SOURCE_REQUIRED,
        "reason": "instrument.ext carries no delisting date; klines end silently",
        "alternative": "akshare/exchange delisting list (not yet materialised)",
        "readiness": BLOCKED},
    "historical_price_limit_regimes": {
        "tickflow": TICKFLOW_CURRENT_ONLY,
        "reason": "instrument.ext exposes CURRENT limit_up/limit_down only, not the time-varying regime",
        "alternative": EXCHANGE_SOURCE_AVAILABLE + ": board rule timeline is derivable from board+date",
        "readiness": "PARTIAL"},
    "ipo_special_limit_intervals": {
        "tickflow": TICKFLOW_CURRENT_ONLY,
        "reason": "TickFlow provides listing_date; the IPO no-limit window is derivable from it",
        "alternative": EXCHANGE_SOURCE_AVAILABLE + ": listing_date + preregistered 60-td rule",
        "readiness": "AVAILABLE"},
    "corporate_action_identity": {
        "tickflow": ALTERNATIVE_SOURCE_REQUIRED,
        "reason": "ex_factors endpoint NOT entitled (PermissionError 无除权因子查询权限)",
        "alternative": "akshare adjustment factors (not yet materialised); raw bars use adjust=none",
        "readiness": BLOCKED},
}


def _sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    master = pd.read_parquet(MASTER)
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")
    src_hash = _sha_file(MASTER)
    bench_hash = _sha_file(BENCH)
    bse = json.loads(BSE_AUDIT.read_text()) if BSE_AUDIT.exists() else {}

    # §9 source audit
    source_audit = {
        "generated": now, "experiment": "H-032B PIT field source audit",
        "primary_bar_provider": "TickFlow (unchanged)",
        "capability_evidence": str(BENCH.relative_to(REPO)),
        "capability_evidence_hash": bench_hash,
        "fields": PIT_SOURCE,
        "principle": ("historical intervals are NEVER inferred from current metadata; unavailable "
                      "sources are BLOCKED_BY_DATA / ALTERNATIVE_SOURCE_REQUIRED, never default-false"),
    }
    (OUT / "pit_source_audit.json").write_text(json.dumps(source_audit, indent=2, ensure_ascii=False))

    # §10 provenance-carrying availability (rewrite pit_field_availability.json)
    delisted_status = int((master["status"] == "delisted").sum())
    delist_dates = int(master["delisting_date"].notna().sum())
    covered_syms: set[str] = set()
    panel = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
    if panel.exists():
        covered_syms |= set(pd.read_parquet(panel, columns=["symbol"])["symbol"].astype(str).unique())
    staging = REPO / "runtime/data/v7/full_universe/_staging"
    if staging.exists():
        covered_syms |= {f.stem.replace("sym_", "").replace("_", ".") for f in staging.glob("sym_*.parquet")}
    delisted_syms = set(master.loc[master["status"] == "delisted", "symbol"].astype(str))
    delisted_with_bars = len(delisted_syms & covered_syms)

    def prov(field: str):
        info = PIT_SOURCE[field]
        return {"tickflow_status": info["tickflow"], "readiness": info["readiness"],
                "reason": info["reason"], "alternative_source": info["alternative"],
                "available_at": None if info["readiness"] in (BLOCKED,) else "derivable/current",
                "source": info["alternative"], "source_timestamp": now, "source_hash": src_hash}

    availability = {
        "generated": now, "experiment": "H-032B strict PIT availability",
        "securities": int(len(master)),
        "pit_field_availability": {
            "identity_symbol": "AVAILABLE",
            "exchange_board_security_type": "AVAILABLE",
            "listing_date": f"AVAILABLE (TickFlow instrument.ext + exchange; {int(master['listing_date'].notna().sum() if 'listing_date' in master else 0)}/{len(master)})",
            "delisting_date": f"{BLOCKED} — {delist_dates}/{delisted_status} delisted carry a date; TickFlow has none",
            "st_intervals": f"{BLOCKED} — {PIT_SOURCE['st_intervals']['tickflow']}",
            "suspension_intervals": f"{BLOCKED} — {PIT_SOURCE['suspension_intervals']['tickflow']}",
            "historical_price_limit_rule": f"PARTIAL — {PIT_SOURCE['historical_price_limit_regimes']['tickflow']}",
            "ipo_special_limit_rule": "AVAILABLE (listing_date + preregistered 60-td)",
            "corporate_action_identity": f"{BLOCKED} — {PIT_SOURCE['corporate_action_identity']['tickflow']}",
            "historical_symbol_migration": (
                f"{BLOCKED} for old-code map; BSE CURRENT identity RESOLVED "
                f"({bse.get('identity_decision', 'n/a')})"),
        },
        "pit_field_provenance": {f: prov(f) for f in PIT_SOURCE},
        "survivorship_bias": {
            "master_securities": int(len(master)), "delisted_total": delisted_status,
            "delisted_with_delisting_date": delist_dates, "delisted_with_bar_history": delisted_with_bars,
            "delisted_fraction_of_master": round(delisted_status / max(1, len(master)), 4),
            "assessment": (f"{delisted_status} delisted ({round(100*delisted_status/max(1,len(master)),1)}%); "
                           f"{delisted_with_bars} carry bars, {delist_dates} carry a delisting date. "
                           f"Delisting/ST/suspension/corporate-action intervals BLOCKED_BY_DATA."),
        },
        "honesty_note": ("TickFlow is the primary BAR provider and remains so; several PIT metadata fields "
                         "need an alternative authoritative source that is not yet materialised. Unavailable "
                         "= BLOCKED_BY_DATA / ALTERNATIVE_SOURCE_REQUIRED, never default-false."),
    }

    # H-032C: fold in the sourced PIT metadata tables (delisting/price-limit/IPO
    # closed with authoritative provenance) so the decision reflects real closures.
    meta = REPO / "runtime/data/u0/pit/pit_metadata_manifest.json"
    sourced = {}
    if meta.exists():
        mj = json.loads(meta.read_text())
        sourced = mj.get("field_status", {})
        availability["pit_metadata_sourcing"] = {
            "closed_fields": mj.get("closed_fields", []),
            "blocked_fields": mj.get("blocked_fields", []),
            "delisting_dates_sourced": mj.get("delisting_dates_sourced"),
        }
        # reflect closures into the field availability + provenance
        if sourced.get("delisting_intervals", {}).get("status") == "AVAILABLE":
            availability["pit_field_availability"]["delisting_date"] = (
                f"AVAILABLE — {mj.get('delisting_dates_sourced')} delisting dates sourced "
                f"(akshare SH/SZ lists); PIT interval table runtime/data/u0/pit/delisting_intervals.parquet")
        if sourced.get("price_limit_regimes", {}).get("status") == "AVAILABLE":
            availability["pit_field_availability"]["historical_price_limit_rule"] = (
                "AVAILABLE — deterministic exchange rule intervals "
                "(runtime/data/u0/pit/price_limit_regimes.parquet)")

    (OUT / "pit_field_availability.json").write_text(json.dumps(availability, indent=2, ensure_ascii=False))

    # §10 strict PIT decision — blocked = fields with no authoritative source yet
    if sourced:
        blocked_fields = [f for f, v in sourced.items()
                          if str(v.get("status")) in (BLOCKED, "ALTERNATIVE_SOURCE_REQUIRED")]
    else:
        blocked_fields = [f for f, i in PIT_SOURCE.items() if i["readiness"] == BLOCKED]
    # identity is resolved for BSE current codes; the binding blocker is PIT metadata
    bse_ok = bse.get("identity_decision") == "BSE_IDENTITY_CURRENT_RESOLVED"
    if not bse_ok:
        decision = "FULL_UNIVERSE_DATA_NOT_READY_IDENTITY"
    elif blocked_fields:
        decision = "FULL_UNIVERSE_DATA_NOT_READY_PIT"
    else:
        decision = "FULL_UNIVERSE_DATA_READY"
    cert = {
        "generated": now, "experiment": "H-032B strict PIT certificate",
        "decision": decision,
        "training_permitted": decision == "FULL_UNIVERSE_DATA_READY",
        "blocked_pit_fields": blocked_fields,
        "bse_identity_decision": bse.get("identity_decision"),
        "primary_bar_provider": "TickFlow (unchanged)",
        "allowed_decisions": ["FULL_UNIVERSE_DATA_READY", "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
                              "FULL_UNIVERSE_DATA_NOT_READY_PIT", "FULL_UNIVERSE_DATA_NOT_READY_IDENTITY",
                              "FULL_UNIVERSE_DATA_NOT_READY_PROVIDER"],
        "pit_source_audit": str((OUT / "pit_source_audit.json").relative_to(REPO)),
        "blinding": "no candidate performance included",
    }
    (OUT / "u0_strict_pit_certificate.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False))
    return {"pit_source_audit": source_audit, "availability": availability, "strict_pit": cert}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    r = build()
    print(json.dumps({"strict_pit_decision": r["strict_pit"]["decision"],
                      "training_permitted": r["strict_pit"]["training_permitted"],
                      "blocked_pit_fields": r["strict_pit"]["blocked_pit_fields"],
                      "bse_identity": r["strict_pit"]["bse_identity_decision"]}, indent=2))
    return 0 if r["strict_pit"]["training_permitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
