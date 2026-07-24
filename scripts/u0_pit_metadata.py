#!/usr/bin/env python3
"""H-032C §7: strict PIT metadata sourcing (interval tables with provenance).

Builds auditable historical PIT interval tables. TickFlow stays the primary BAR
provider; these tables come from authoritative NON-bar sources:

  * price_limit_regimes    — DETERMINISTIC from documented exchange rules
                             (board + rule-change dates); source = exchange rule.
  * ipo_special_limit      — DETERMINISTIC from listing_date + board IPO rule.
  * delisting_intervals    — akshare SH/SZ delisting lists (authoritative, bulk).
  * st_intervals           — no bulk historical interval source -> BLOCKED_BY_DATA.
  * suspension_intervals   — no authoritative per-symbol interval source ->
                             ALTERNATIVE_SOURCE_REQUIRED (bar-gap inference is not
                             authoritative and is NOT used).
  * corporate_action_ident — TickFlow ex_factors NOT entitled ->
                             ALTERNATIVE_SOURCE_REQUIRED (never fabricated).

Every emitted row carries: symbol, effective_start, effective_end, available_at,
source, source_timestamp, source_hash. History is NEVER inferred from current
metadata; unknown status is NEVER defaulted to false.

Outputs (runtime/data/u0/pit/):
  price_limit_regimes.parquet, ipo_special_limit_intervals.parquet,
  delisting_intervals.parquet, pit_metadata_manifest.json

Usage: AI_quant_venv/bin/python3 scripts/u0_pit_metadata.py --allow-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
OUT = REPO / "runtime/data/u0/pit"
BLOCKED = "BLOCKED_BY_DATA"
ALT = "ALTERNATIVE_SOURCE_REQUIRED"

# Documented, authoritative exchange price-limit regimes (board -> ordered
# (effective_start, effective_end|None, normal_pct, st_pct, note)).
PRICE_LIMIT_RULES = {
    "SH_Main": [("1996-12-16", None, 10, 5, "10% (ST 5%) since 1996-12-16")],
    "SZ_Main": [("1996-12-16", None, 10, 5, "10% (ST 5%) since 1996-12-16")],
    "ChiNext": [("2009-10-30", "2020-08-23", 10, 5, "10% pre-reform"),
                ("2020-08-24", None, 20, 20, "20% from registration reform 2020-08-24")],
    "STAR":    [("2019-07-22", None, 20, 20, "20% since inception; first 5 td no limit")],
    "BSE":     [("2021-11-15", None, 30, 30, "30% since inception; first day no limit")],
    "OTHER":   [("1996-12-16", None, 10, 5, "assumed main-board 10%")],
}
# IPO no-limit / special-limit window (trading days from listing) by board.
IPO_WINDOW_TD = {"STAR": 5, "ChiNext": 5, "BSE": 1, "SH_Main": 1, "SZ_Main": 1, "OTHER": 1}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _akshare_delist(now: str) -> tuple[pd.DataFrame, dict]:
    import akshare as ak
    frames, provenance = [], {}
    for market, fn in (("SH", "stock_info_sh_delist"), ("SZ", "stock_info_sz_delist")):
        rows, err = None, None
        for attempt in range(3):
            try:
                df = getattr(ak, fn)()
                rows = df
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}:{str(e)[:80]}"
                time.sleep((3, 8, 15)[attempt])
        provenance[market] = {"function": fn, "rows": 0 if rows is None else int(len(rows)),
                              "error": err}
        if rows is None or not len(rows):
            continue
        code_col = next((c for c in rows.columns if "代码" in c), rows.columns[0])
        list_col = next((c for c in rows.columns if "上市" in c), None)
        del_col = next((c for c in rows.columns if "暂停" in c or "终止" in c or "退市" in c), None)
        out = pd.DataFrame({
            "code": rows[code_col].astype(str).str.zfill(6),
            "listing_date": pd.to_datetime(rows[list_col], errors="coerce") if list_col else pd.NaT,
            "delisting_date": pd.to_datetime(rows[del_col], errors="coerce") if del_col else pd.NaT,
            "market": market,
        })
        frames.append(out)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "listing_date", "delisting_date", "market"])
    return combined, provenance


def build(allow_network: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    master = pd.read_parquet(MASTER)
    master["symbol"] = master["symbol"].astype(str)
    master["code"] = master["code"].astype(str).str.zfill(6)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")

    # ---- price_limit_regimes (deterministic) --------------------------------
    plr_rows = []
    rule_hash = _hash(PRICE_LIMIT_RULES)
    for m in master.itertuples():
        for (start, end, npct, stpct, note) in PRICE_LIMIT_RULES.get(m.board, PRICE_LIMIT_RULES["OTHER"]):
            eff_start = max(pd.Timestamp(start), m.listing_date) if pd.notna(m.listing_date) else pd.Timestamp(start)
            plr_rows.append({
                "symbol": m.symbol, "board": m.board,
                "effective_start": eff_start.date().isoformat(),
                "effective_end": end, "normal_limit_pct": npct, "st_limit_pct": stpct,
                "regime_note": note,
                "available_at": eff_start.date().isoformat(),
                "source": "exchange_price_limit_rule (documented)",
                "source_timestamp": now, "source_hash": rule_hash})
    plr = pd.DataFrame(plr_rows)
    plr.to_parquet(OUT / "price_limit_regimes.parquet", index=False)

    # ---- ipo_special_limit_intervals (deterministic from listing_date) ------
    ipo_rows = []
    for m in master.itertuples():
        if pd.isna(m.listing_date):
            continue
        td = IPO_WINDOW_TD.get(m.board, 1)
        # approximate window end by calendar days (td*~1.5) — the exact end uses the
        # trading calendar at assembly; this table records the rule + listing anchor.
        end = (m.listing_date + pd.Timedelta(days=int(td * 1.6))).date().isoformat()
        ipo_rows.append({
            "symbol": m.symbol, "board": m.board,
            "effective_start": m.listing_date.date().isoformat(),
            "effective_end": end, "special_window_trading_days": td,
            "rule": f"first {td} trading day(s) no/relaxed price limit",
            "available_at": m.listing_date.date().isoformat(),
            "source": "exchange_ipo_rule + listing_date", "source_timestamp": now,
            "source_hash": _hash(IPO_WINDOW_TD)})
    ipo = pd.DataFrame(ipo_rows)
    ipo.to_parquet(OUT / "ipo_special_limit_intervals.parquet", index=False)

    # ---- delisting_intervals (akshare authoritative, bulk) ------------------
    delist_prov = {}
    delist_matched = 0
    if allow_network:
        ak_del, delist_prov = _akshare_delist(now)
        if len(ak_del):
            code_to_sym = dict(zip(master["code"], master["symbol"]))
            ak_del["symbol"] = ak_del["code"].map(code_to_sym)
            src_hash = _hash(sorted(ak_del["code"].tolist()))
            rows = []
            for r in ak_del.itertuples():
                if pd.isna(r.symbol) or pd.isna(r.delisting_date):
                    continue
                rows.append({
                    "symbol": r.symbol, "code": r.code,
                    "effective_start": r.delisting_date.date().isoformat(),
                    "effective_end": None, "status": "delisted",
                    "available_at": r.delisting_date.date().isoformat(),
                    "source": f"akshare.stock_info_{r.market.lower()}_delist",
                    "source_timestamp": now, "source_hash": src_hash})
            dl = pd.DataFrame(rows)
            delist_matched = len(dl)
            dl.to_parquet(OUT / "delisting_intervals.parquet", index=False)
    else:
        pd.DataFrame(columns=["symbol", "code", "effective_start", "effective_end", "status",
                              "available_at", "source", "source_timestamp", "source_hash"]
                     ).to_parquet(OUT / "delisting_intervals.parquet", index=False)

    field_status = {
        "price_limit_regimes": {"status": "AVAILABLE", "rows": int(len(plr)),
                                "source": "exchange documented rule (deterministic)"},
        "ipo_special_limit_intervals": {"status": "AVAILABLE", "rows": int(len(ipo)),
                                        "source": "exchange IPO rule + listing_date"},
        "delisting_intervals": {"status": "AVAILABLE" if delist_matched else ALT,
                                "rows": int(delist_matched),
                                "source": "akshare SH/SZ delisting lists",
                                "provenance": delist_prov,
                                "note": "SZ endpoint intermittently unreachable; retried 3x"},
        "st_intervals": {"status": BLOCKED,
                         "reason": "no authoritative bulk historical ST-interval source; current ST "
                                   "list is not intervals; name-change history not materialised"},
        "suspension_intervals": {"status": ALT,
                                 "reason": "no authoritative per-symbol suspension-interval source; "
                                           "bar-gap inference is not authoritative and is not used"},
        "corporate_action_identity": {"status": ALT,
                                      "reason": "TickFlow ex_factors NOT entitled; akshare adjustment "
                                                "factors not materialised; never fabricated"},
    }
    manifest = {
        "generated": now, "experiment": "H-032C strict PIT metadata sourcing",
        "primary_bar_provider": "TickFlow (unchanged)",
        "principle": ("history never inferred from current metadata; unknown status never default-false; "
                      "absent authoritative source = BLOCKED_BY_DATA / ALTERNATIVE_SOURCE_REQUIRED"),
        "field_status": field_status,
        "closed_fields": [f for f, v in field_status.items() if v["status"] == "AVAILABLE"],
        "blocked_fields": [f for f, v in field_status.items() if v["status"] in (BLOCKED, ALT)],
        "delisted_in_master": int((master["status"] == "delisted").sum()),
        "delisting_dates_sourced": delist_matched,
    }
    (OUT / "pit_metadata_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()
    m = build(args.allow_network)
    print(json.dumps({"closed_fields": m["closed_fields"], "blocked_fields": m["blocked_fields"],
                      "delisting_dates_sourced": m["delisting_dates_sourced"],
                      "price_limit_rows": m["field_status"]["price_limit_regimes"]["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
