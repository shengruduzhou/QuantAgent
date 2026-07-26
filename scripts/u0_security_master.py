#!/usr/bin/env python3
"""Build the historically-aware A-share security master from live sources.

Three independent inputs are unioned, never averaged, and every row keeps the
source that contributed it:

1. **TickFlow ``exchanges.get_instruments``** — the entitled bulk instrument
   listing for SH / SZ / BJ. Authoritative for currently-listed identity:
   name (including the ST / *ST prefix), listing date, share counts, tick size
   and the vendor's current price limits.
2. **Exchange delisting lists (akshare over SSE/SZSE)** — securities that no
   longer appear in any live instrument listing. Without these the universe is
   survivorship-biased by construction.
3. **The frozen H-028 master already on disk** — retains historical names that
   neither live source still publishes.

Board and exchange are always re-derived with
``quantagent.data.ashare.symbols`` so a vendor suffix can never place a security
on the wrong exchange, and BSE legacy (8xxxxx) versus current (920xxx) codes are
counted separately because vendors serve them differently.

Output: runtime/data/u0/security_master.parquet
        runtime/data/u0/security_master_manifest.json

Usage: AI_quant_venv/bin/python3 scripts/u0_security_master.py --allow-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.symbols import SymbolError, identify, is_bse_legacy_code  # noqa: E402

OUT = REPO / "runtime/data/u0"
FROZEN_MASTER = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
SUPPLEMENTAL = OUT / "master_supplemental_additions.parquet"
EXCHANGES = ("SH", "SZ", "BJ")
TICKFLOW_PACE_S = 6.5           # measured hard limit: 10 requests / minute


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _from_tickflow() -> tuple[pd.DataFrame, dict]:
    """Bulk instrument listing per exchange (entitled endpoint)."""
    from quantagent.data.ashare.sources import build_tickflow_client

    client = build_tickflow_client()
    frames, provenance = [], {}
    for index, exchange in enumerate(EXCHANGES):
        if index:
            time.sleep(TICKFLOW_PACE_S)
        try:
            raw = pd.DataFrame(client.exchanges.get_instruments(exchange))
        except Exception as exc:  # noqa: BLE001
            provenance[exchange] = {"rows": 0, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            continue
        provenance[exchange] = {"rows": int(len(raw)), "error": None}
        if raw.empty:
            continue
        ext = raw["ext"].apply(lambda v: v if isinstance(v, dict) else {})
        frames.append(pd.DataFrame({
            "symbol": raw["symbol"].astype(str),
            "code": raw["code"].astype(str).str.zfill(6),
            "name": raw["name"].astype(str),
            "instrument_type": raw["type"].astype(str),
            "listing_date": pd.to_datetime(ext.apply(lambda d: d.get("listing_date")), errors="coerce"),
            "total_shares": pd.to_numeric(ext.apply(lambda d: d.get("total_shares")), errors="coerce"),
            "float_shares": pd.to_numeric(ext.apply(lambda d: d.get("float_shares")), errors="coerce"),
            "tick_size": pd.to_numeric(ext.apply(lambda d: d.get("tick_size")), errors="coerce"),
            "vendor_limit_up": pd.to_numeric(ext.apply(lambda d: d.get("limit_up")), errors="coerce"),
            "vendor_limit_down": pd.to_numeric(ext.apply(lambda d: d.get("limit_down")), errors="coerce"),
            "status": "listed",
            "source": "tickflow.exchanges.get_instruments",
        }))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, provenance


def _from_delisting_lists() -> tuple[pd.DataFrame, dict]:
    """Delisted securities published by SSE and SZSE (via the akshare wrappers)."""
    import akshare as ak

    frames, provenance = [], {}
    for market, fn in (("SH", "stock_info_sh_delist"), ("SZ", "stock_info_sz_delist")):
        raw, error = None, None
        for attempt in range(3):
            try:
                raw = getattr(ak, fn)()
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:120]}"
                time.sleep((3, 8, 15)[attempt])
        provenance[market] = {"function": fn, "rows": 0 if raw is None else int(len(raw)),
                              "error": error}
        if raw is None or not len(raw):
            continue
        code_col = next((c for c in raw.columns if "代码" in c), raw.columns[0])
        name_col = next((c for c in raw.columns if "简称" in c or "名称" in c), None)
        list_col = next((c for c in raw.columns if "上市" in c), None)
        del_col = next((c for c in raw.columns if any(k in c for k in ("暂停", "终止", "退市"))), None)
        codes = raw[code_col].astype(str).str.extract(r"(\d{6})")[0].dropna()
        frame = pd.DataFrame({
            "code": codes,
            "name": raw.loc[codes.index, name_col].astype(str) if name_col else "",
            "listing_date": pd.to_datetime(raw.loc[codes.index, list_col], errors="coerce")
            if list_col else pd.NaT,
            "delisting_date": pd.to_datetime(raw.loc[codes.index, del_col], errors="coerce")
            if del_col else pd.NaT,
            "status": "delisted",
            "source": f"akshare.{fn}",
        })
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        symbols, keep = [], []
        for code in combined["code"]:
            try:
                symbols.append(identify(code).symbol)
                keep.append(True)
            except SymbolError:
                keep.append(False)
        combined = combined[keep].copy()
        combined["symbol"] = symbols
    return combined, provenance


def _from_frozen_master() -> pd.DataFrame:
    frames = []
    for path in (FROZEN_MASTER, SUPPLEMENTAL):
        if not path.exists():
            continue
        raw = pd.read_parquet(path)
        raw = raw.assign(source=f"frozen_master:{path.name}")
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("symbol", "listing_date", "delisting_date", "status", "source")
            if c in combined.columns]
    combined = combined[keep].copy()
    combined["symbol"] = combined["symbol"].astype(str)
    return combined


def build(allow_network: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = _now()
    live, tickflow_prov = (_from_tickflow() if allow_network else (pd.DataFrame(), {"skipped": True}))
    delisted, delist_prov = (_from_delisting_lists() if allow_network else (pd.DataFrame(), {"skipped": True}))
    frozen = _from_frozen_master()

    parts = []
    for frame, origin in ((live, "tickflow_live"), (delisted, "exchange_delisting"),
                          (frozen, "frozen_master")):
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["origin"] = origin
        parts.append(frame)
    if not parts:
        raise RuntimeError("no security-master source produced rows; refusing to write an empty master")
    master = pd.concat(parts, ignore_index=True)

    # canonical identity, dropping anything that is not an A-share security code
    rows = []
    dropped: dict[str, int] = {}
    for record in master.to_dict("records"):
        try:
            ident = identify(str(record.get("symbol") or record.get("code")))
        except SymbolError:
            dropped["unresolvable_code"] = dropped.get("unresolvable_code", 0) + 1
            continue
        if str(record.get("instrument_type", "stock")).lower() not in {"stock", "nan", ""}:
            dropped["non_equity_instrument"] = dropped.get("non_equity_instrument", 0) + 1
            continue
        if ident.board == "OTHER":
            dropped["non_a_share_board"] = dropped.get("non_a_share_board", 0) + 1
            continue
        record.update({
            "symbol": ident.symbol, "code": ident.code, "exchange": ident.exchange,
            "board": ident.board, "security_type": ident.security_type,
            "bse_legacy_code": is_bse_legacy_code(ident.symbol),
        })
        rows.append(record)
    frame = pd.DataFrame(rows)

    # Deduplicate with an explicit precedence: a live listing wins identity, a
    # delisting record wins the delisting date, the frozen master fills gaps.
    order = {"tickflow_live": 0, "exchange_delisting": 1, "frozen_master": 2}
    frame["_rank"] = frame["origin"].map(order).fillna(9)
    frame = frame.sort_values(["symbol", "_rank"])
    listed_symbols = set(frame.loc[frame["origin"] == "tickflow_live", "symbol"])
    delisting_dates = (frame.loc[frame["origin"] == "exchange_delisting"]
                       .dropna(subset=["delisting_date"])
                       .drop_duplicates("symbol").set_index("symbol")["delisting_date"])
    merged = frame.drop_duplicates("symbol", keep="first").drop(columns=["_rank"]).copy()
    merged["delisting_date"] = merged["symbol"].map(delisting_dates).fillna(
        pd.to_datetime(merged.get("delisting_date"), errors="coerce"))
    merged["status"] = merged.apply(
        lambda r: "listed" if r["symbol"] in listed_symbols else
        ("delisted" if pd.notna(r["delisting_date"]) or r.get("status") == "delisted" else "unknown"),
        axis=1)
    merged["listing_date"] = pd.to_datetime(merged["listing_date"], errors="coerce")
    # a name carrying the risk-warning prefix is the vendor's CURRENT ST state
    merged["current_st"] = merged["name"].astype(str).str.upper().str.replace(" ", "").str.startswith(("ST", "*ST"))
    merged["available_at"] = merged["listing_date"]
    merged["source_timestamp"] = now
    merged["master_build"] = "u0_security_master"
    columns = ["symbol", "code", "exchange", "board", "security_type", "name", "status",
               "listing_date", "delisting_date", "current_st", "bse_legacy_code",
               "total_shares", "float_shares", "tick_size", "vendor_limit_up",
               "vendor_limit_down", "origin", "source", "available_at", "source_timestamp",
               "master_build"]
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged = merged[columns].sort_values("symbol").reset_index(drop=True)
    merged.to_parquet(OUT / "security_master.parquet", index=False)

    by_board = merged["board"].value_counts().to_dict()
    manifest = {
        "generated": now,
        "securities": int(len(merged)),
        "by_board": by_board,
        "by_status": merged["status"].value_counts().to_dict(),
        "by_origin": merged["origin"].value_counts().to_dict(),
        "bse_legacy_codes": int(merged["bse_legacy_code"].fillna(False).astype(bool).sum()),
        "bse_current_920": int(((merged["board"] == "BSE") &
                                (~merged["bse_legacy_code"].fillna(False).astype(bool))).sum()),
        "current_st_names": int(merged["current_st"].fillna(False).astype(bool).sum()),
        "listing_date_coverage": int(merged["listing_date"].notna().sum()),
        "delisting_date_coverage": int(merged["delisting_date"].notna().sum()),
        "dropped": dropped,
        "sources": {
            "tickflow_instruments": tickflow_prov,
            "exchange_delisting_lists": delist_prov,
            "frozen_master": {"rows": int(len(frozen)), "path": str(FROZEN_MASTER.relative_to(REPO))},
        },
        "precedence": "tickflow_live > exchange_delisting > frozen_master (identity); "
                      "delisting date always taken from the exchange delisting list when present",
        "content_hash": _hash(sorted(merged["symbol"].tolist())),
        "honesty_note": ("Historical NAME/ST/suspension intervals are NOT in this file; they live in "
                         "runtime/data/u0/pit/ interval tables built by u0_pit_intervals.py."),
    }
    (OUT / "security_master_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="required to contact TickFlow and the exchange delisting lists")
    args = parser.parse_args()
    load_repo_env()
    manifest = build(args.allow_network)
    print(json.dumps({k: manifest[k] for k in
                      ("securities", "by_board", "by_status", "bse_legacy_codes",
                       "bse_current_920", "current_st_names", "delisting_date_coverage")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
