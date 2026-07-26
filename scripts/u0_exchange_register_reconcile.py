#!/usr/bin/env python3
"""Reconcile the U0 security master against the exchanges' own listed registers.

A vendor instrument listing is not proof that a security trades. The SZSE and
SSE each publish the authoritative register of currently listed A-shares with
their listing dates; a code that a vendor serves but the exchange does not list
has not begun trading, and a code the exchange lists but no provider serves is a
real acquisition gap. Distinguishing those two is the only way to close the
coverage gate honestly instead of inventing placeholder bars.

For every master security this produces exactly one disposition:

``LISTED_WITH_HISTORY``        exchange lists it and we hold bars
``LISTED_NO_HISTORY``          exchange lists it, no provider served bars — a gap
``PRE_LISTING_NO_SESSIONS``    vendor carries the instrument, the exchange register
                               does not, and no provider has ever returned a bar:
                               an approved issue that has not started trading
``DELISTED_WITH_HISTORY``      not in the live register, we hold historical bars
``DELISTED_NO_HISTORY``        not in the live register and no bars anywhere
``EXCHANGE_ONLY_NOT_IN_MASTER`` exchange lists a code the master lacks

Outputs (runtime/data/u0/):
  exchange_register.parquet            raw exchange rows with provenance
  master_disposition.parquet           one disposition per master security
  exchange_reconciliation.json         counts, mismatches, listing-date deltas

Usage:
  AI_quant_venv/bin/python3 scripts/u0_exchange_register_reconcile.py --allow-network
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.ashare.symbols import SymbolError, identify  # noqa: E402

U0 = REPO / "runtime/data/u0"
MASTER = U0 / "security_master.parquet"
BARS_DIRS = (
    U0 / "bars/tickflow_raw",
    U0 / "bars/tencent_raw",
    U0 / "bars/sina_delisted",
    REPO / "runtime/data/v7/full_universe/_staging",
)
SZSE_URL = "https://www.szse.cn/api/report/ShowReport/data"
SSE_URL = "https://query.sse.com.cn/sseQuery/commonQuery.do"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_szse(client: HttpClient) -> tuple[pd.DataFrame, dict]:
    """SZSE 'A股列表' — the exchange's own register of listed Shenzhen A-shares."""
    rows: list[dict] = []
    page, pages, as_of, error = 1, None, None, None
    while True:
        outcome = client.get_json(SZSE_URL, headers={
            "Referer": "https://www.szse.cn/market/stock/list/index.html"}, params={
            "SHOWTYPE": "JSON", "CATALOGID": "1110", "TABKEY": "tab1",
            "PAGENO": str(page), "random": f"0.{page}"})
        if not outcome.ok:
            error = outcome.error
            break
        body = outcome.payload
        node = body[0] if isinstance(body, list) and body else {}
        meta = node.get("metadata") or {}
        pages = pages or int(meta.get("pagecount") or 0)
        as_of = as_of or str(meta.get("subname") or "").strip()
        chunk = node.get("data") or []
        if not chunk:
            break
        rows.extend(chunk)
        if pages and page >= pages:
            break
        page += 1
        if page > 400:                       # safety valve
            break
    if not rows:
        return pd.DataFrame(), {"exchange": "SZSE", "rows": 0, "error": error}
    frame = pd.DataFrame(rows)
    # the register embeds the code in an anchor tag on some rows
    codes = frame["agdm"].astype(str).str.extract(r"(\d{6})")[0]
    names = frame["agjc"].astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip()
    out = pd.DataFrame({
        "code": codes,
        "exchange_name": names,
        "exchange_listing_date": pd.to_datetime(frame["agssrq"], errors="coerce"),
        "exchange": "SZSE",
        "source": "szse.cn A股列表 (CATALOGID=1110)",
        "as_of": as_of,
    }).dropna(subset=["code"])
    return out, {"exchange": "SZSE", "rows": int(len(out)), "pages": pages,
                 "as_of": as_of, "error": error}


def fetch_sse(client: HttpClient) -> tuple[pd.DataFrame, dict]:
    """SSE listed-security query. Optional: the SSE endpoint is intermittent."""
    rows: list[dict] = []
    page, error, total = 1, None, None
    while True:
        outcome = client.get_json(SSE_URL, headers={
            "Referer": "https://www.sse.com.cn/assortment/stock/list/share/"}, params={
            "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "STOCK_TYPE": "1",
            "REG_PROVINCE": "", "CSRC_CODE": "", "STOCK_CODE": "",
            "pageHelp.pageSize": "100", "pageHelp.pageNo": str(page),
            "pageHelp.beginPage": str(page), "pageHelp.cacheSize": "1",
            "pageHelp.endPage": str(page)})
        if not outcome.ok:
            error = outcome.error
            break
        helper = (outcome.payload or {}).get("pageHelp") or {}
        chunk = helper.get("data") or []
        total = total or helper.get("total")
        if not chunk:
            break
        rows.extend(chunk)
        if total and len(rows) >= int(total):
            break
        page += 1
        if page > 100:
            break
    if not rows:
        return pd.DataFrame(), {"exchange": "SSE", "rows": 0, "error": error,
                                "note": "SSE query endpoint intermittent in this runtime"}
    frame = pd.DataFrame(rows)
    code_col = next((c for c in frame.columns if c.upper() in {"A_STOCK_CODE", "SECURITY_CODE"}),
                    frame.columns[0])
    name_col = next((c for c in frame.columns if "ABBR" in c.upper() or "NAME" in c.upper()), None)
    date_col = next((c for c in frame.columns if "LIST" in c.upper() and "DATE" in c.upper()), None)
    out = pd.DataFrame({
        "code": frame[code_col].astype(str).str.extract(r"(\d{6})")[0],
        "exchange_name": frame[name_col].astype(str) if name_col else "",
        "exchange_listing_date": pd.to_datetime(frame[date_col], errors="coerce")
        if date_col else pd.NaT,
        "exchange": "SSE",
        "source": "query.sse.com.cn COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
        "as_of": "",
    }).dropna(subset=["code"])
    return out, {"exchange": "SSE", "rows": int(len(out)), "total": total, "error": error}


def symbols_with_bars() -> set[str]:
    held: set[str] = set()
    for directory in BARS_DIRS:
        if directory.exists():
            held |= {p.stem.replace("sym_", "").replace("_", ".")
                     for p in directory.glob("sym_*.parquet")}
    return held


def build(allow_network: bool) -> dict:
    master = pd.read_parquet(MASTER)
    master["symbol"] = master["symbol"].astype(str)
    master["code"] = master["code"].astype(str).str.zfill(6)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")

    client = HttpClient(timeout=25, max_attempts=3)
    frames, provenance = [], []
    if allow_network:
        for fetcher in (fetch_szse, fetch_sse):
            frame, info = fetcher(client)
            provenance.append(info)
            if len(frame):
                frames.append(frame)
    register = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "exchange_name", "exchange_listing_date", "exchange", "source", "as_of"])
    if len(register):
        register = register.drop_duplicates("code")
        register.to_parquet(U0 / "exchange_register.parquet", index=False)

    held = symbols_with_bars()
    listed_codes = set(register["code"]) if len(register) else set()
    # Only exchanges we actually pulled can inform a disposition; a security on an
    # exchange whose register we could not read stays UNVERIFIED rather than being
    # declared pre-listing on missing evidence.
    covered_exchanges = set(register["exchange"]) if len(register) else set()
    exchange_of = {"SSE": "SSE", "SZSE": "SZSE", "BSE": "BSE"}

    listing_lookup = register.set_index("code")["exchange_listing_date"] if len(register) else pd.Series(dtype="datetime64[ns]")
    records = []
    for row in master.itertuples():
        has_bars = row.symbol in held
        register_known = exchange_of.get(row.exchange) in covered_exchanges
        in_register = row.code in listed_codes
        vendor_listing_absent = pd.isna(row.listing_date) or row.listing_date <= pd.Timestamp("1970-01-02")
        if row.status == "delisted":
            disposition = "DELISTED_WITH_HISTORY" if has_bars else "DELISTED_NO_HISTORY"
        elif not register_known:
            disposition = "LISTED_WITH_HISTORY" if has_bars else "UNVERIFIED_NO_HISTORY"
        elif in_register:
            disposition = "LISTED_WITH_HISTORY" if has_bars else "LISTED_NO_HISTORY"
        elif not has_bars and vendor_listing_absent:
            disposition = "PRE_LISTING_NO_SESSIONS"
        elif not has_bars:
            disposition = "LISTED_NO_HISTORY"
        else:
            disposition = "LISTED_WITH_HISTORY"
        exchange_listing = listing_lookup.get(row.code, pd.NaT) if len(register) else pd.NaT
        records.append({
            "symbol": row.symbol, "code": row.code, "exchange": row.exchange,
            "board": row.board, "status": row.status, "name": row.name,
            "vendor_listing_date": row.listing_date,
            "exchange_listing_date": exchange_listing,
            "in_exchange_register": in_register,
            "exchange_register_available": register_known,
            "has_bars": has_bars,
            "disposition": disposition,
        })
    disposition = pd.DataFrame(records)
    disposition.to_parquet(U0 / "master_disposition.parquet", index=False)

    # listing-date disagreements between the vendor master and the exchange
    both = disposition.dropna(subset=["exchange_listing_date"])
    both = both[both["vendor_listing_date"].notna()]
    delta = both[both["vendor_listing_date"].dt.date != both["exchange_listing_date"].dt.date]

    master_codes = set(master["code"])
    exchange_only = sorted(listed_codes - master_codes)

    payload = {
        "generated": _now(),
        "master_securities": int(len(master)),
        "exchange_register_rows": int(len(register)),
        "register_provenance": provenance,
        "disposition_counts": disposition["disposition"].value_counts().to_dict(),
        "pre_listing_symbols": disposition.loc[
            disposition["disposition"] == "PRE_LISTING_NO_SESSIONS", "symbol"].tolist(),
        "listed_no_history_symbols": disposition.loc[
            disposition["disposition"] == "LISTED_NO_HISTORY", "symbol"].tolist()[:50],
        "listing_date_disagreements": int(len(delta)),
        "listing_date_disagreement_sample": delta[
            ["symbol", "vendor_listing_date", "exchange_listing_date"]].head(10).astype(str).to_dict("records"),
        "exchange_only_codes_not_in_master": exchange_only[:50],
        "exchange_only_count": len(exchange_only),
        "principle": ("a vendor instrument listing is not proof of trading; a code the exchange "
                      "does not list and no provider ever served has not begun trading and never "
                      "receives placeholder bars"),
    }
    (U0 / "exchange_reconciliation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="required: reads the SZSE and SSE listed registers")
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to reconcile: --allow-network was not confirmed")
        return 2
    load_repo_env()
    payload = build(args.allow_network)
    print(json.dumps({k: payload[k] for k in
                      ("master_securities", "exchange_register_rows", "disposition_counts",
                       "pre_listing_symbols", "listing_date_disagreements",
                       "exchange_only_count", "register_provenance")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
