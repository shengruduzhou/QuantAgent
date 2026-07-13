#!/usr/bin/env python3
"""H-028 Track A step 1: historical A-share security master (listing/delisting).

Sources (akshare, per-source fail-loud, provenance recorded):
  - SH main + STAR:   stock_info_sh_name_code (listing dates)
  - SZ main + ChiNext: stock_info_sz_name_code (listing dates)
  - BSE:               stock_info_bj_name_code
  - delisted:          stock_info_sh_delist / stock_info_sz_delist
Board classified from code ranges (deterministic, zero survivorship).
ST history and price-limit rule are field-stubbed where no PIT source exists
on disk (recorded as gaps — never fabricated).

Writes runtime/reports/h028/track_a/{historical_security_master.parquet,
universe_manifest.json} and quantifies the panel-vs-reality gap. The daily
bar backfill for post-2020 listings is a separate bounded network job
(commands in universe_repair_audit.md) — WITHOUT it Track A remains
INCOMPLETE_HISTORICAL_UNIVERSE.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runtime/reports/h028/track_a"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"


def board_of(code: str) -> str:
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301", "302")):
        return "ChiNext"
    if code.startswith(("8", "43", "92")):
        return "BSE"
    if code.startswith("60"):
        return "SH_Main"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ_Main"
    return "OTHER"


def limit_rule(board: str) -> str:
    return {"STAR": "20pct", "ChiNext": "20pct", "BSE": "30pct",
            "SH_Main": "10pct(5pct ST)", "SZ_Main": "10pct(5pct ST)"}.get(board, "unknown")


def main() -> int:
    import akshare as ak
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frames, sources = [], {}

    def pull(tag, fn, code_col, date_col, status, exchange):
        try:
            df = fn()
            code = df[code_col].astype(str).str.zfill(6)
            listing = pd.to_datetime(df[date_col], errors="coerce")
            out = pd.DataFrame({"code": code, "listing_date": listing})
            out["status"], out["exchange"] = status, exchange
            out["source"] = tag
            frames.append(out)
            sources[tag] = {"rows": len(out), "ok": True}
            print(f"{tag}: {len(out)} rows", flush=True)
        except Exception as e:  # fail-loud per source, recorded
            sources[tag] = {"ok": False, "error": str(e)[:200]}
            print(f"{tag}: FAILED {str(e)[:120]}", flush=True)

    pull("sh_main", lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
         "证券代码", "上市日期", "listed", "SH")
    pull("sh_star", lambda: ak.stock_info_sh_name_code(symbol="科创板"),
         "证券代码", "上市日期", "listed", "SH")
    pull("sz_all", lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
         "A股代码", "A股上市日期", "listed", "SZ")
    pull("bj_all", lambda: ak.stock_info_bj_name_code, "证券代码", "上市日期", "listed", "BJ") \
        if False else pull("bj_all", ak.stock_info_bj_name_code, "证券代码", "上市日期", "listed", "BJ")
    pull("sh_delist", lambda: ak.stock_info_sh_delist(symbol="全部"),
         "证券代码", "上市日期", "delisted", "SH")
    pull("sz_delist", lambda: ak.stock_info_sz_delist(),
         "证券代码", "上市日期", "delisted", "SZ")

    m = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"], keep="first")
    m["board"] = m["code"].map(board_of)
    m["security_type"] = "A_share"
    m["symbol"] = m.apply(lambda r: f"{r['code']}.{'SH' if r['exchange']=='SH' else ('BJ' if r['exchange']=='BJ' else 'SZ')}", axis=1)
    m["delisting_date"] = pd.NaT  # delist DATE column differs per source; status carries the fact
    m["st_start"], m["st_end"] = pd.NaT, pd.NaT  # gap: no PIT ST history source on disk
    m["price_limit_rule"] = m["board"].map(limit_rule)
    m["available_at"] = pd.Timestamp.now().normalize()
    m["source_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(m[["code", "listing_date"]]).values.tobytes()).hexdigest()[:16]
    m.to_parquet(OUT / "historical_security_master.parquet", index=False)

    panel_syms = set(pd.read_parquet(PANEL, columns=["symbol"])["symbol"].unique())
    m["in_panel"] = m["symbol"].isin(panel_syms)
    listed = m[m["status"] == "listed"]
    post2020 = listed[listed["listing_date"] >= "2020-07-01"]
    gap = {
        "master_rows": len(m),
        "listed_now": int(len(listed)),
        "delisted_rows": int((m["status"] == "delisted").sum()),
        "panel_symbols": len(panel_syms),
        "listed_missing_from_panel": int((~listed["in_panel"]).sum()),
        "post2020H2_listings": int(len(post2020)),
        "post2020H2_missing_from_panel": int((~post2020["in_panel"]).sum()),
        "by_board_missing": {k: int(v) for k, v in
                             listed[~listed["in_panel"]]["board"].value_counts().items()},
        "known_gaps": ["ST PIT history (st_start/st_end unfilled — no source on disk)",
                       "per-source delisting DATE column not normalized (status only)",
                       "daily bars for missing symbols NOT backfilled (separate bounded job)"],
        "sources": sources,
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "universe_manifest.json").write_text(json.dumps(gap, indent=2, ensure_ascii=False))
    print(json.dumps(gap, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
