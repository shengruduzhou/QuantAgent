#!/usr/bin/env python3
"""Probe the live TickFlow API to map exactly what our subscription tier permits.

Read-only. Calls each SDK namespace method with a tiny real request and records
whether it returns data or an error (403 / permission / rate-limit). NEVER prints
the API key. Writes a machine-readable JSON + human summary so Stage B can decide
which data-trust gaps (vwap / adj_factor / turnover / minute / L2 order-flow) are
actually closeable on this account.

SDK 0.1.22 signatures (verified via inspect):
    klines.get(symbol, *, period, count, start_time, end_time, adjust, as_dataframe)
    klines.batch(symbols, *, period, count, ..., adjust, as_dataframe)
    klines.ex_factors(symbols: List[str], ...)         # plural!
    klines.intraday(symbol, *, period, count, as_dataframe)
    quotes.get(*, symbols=, universes=, as_dataframe)   # keyword-only
    financials.metrics(symbols: List[str], *, latest, as_dataframe)  # plural!
    depth.get(symbol)
AdjustType = forward(qfq) | backward(hfq) | forward_additive | backward_additive | none

Usage:
    AI_quant_venv/bin/python3 scripts/probe_tickflow_capabilities.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# --- load .env without echoing secrets ------------------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import tickflow  # noqa: E402

TOKEN = os.environ.get("TICKFLOW_API_KEY", "")
ENDPOINT = os.environ.get("TICKFLOW_API_ENDPOINT") or "https://api.tickflow.org"
SYM = "600519.SH"           # Kweichow Moutai, main board, always trading
SYM2 = "300750.SZ"          # CATL, ChiNext (20% board) — tier-sensitive


def _summarise(obj):
    """Compact description of a return value: type + shape + a few columns."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            cols = list(obj.columns)[:14]
            head = obj.head(1).to_dict("records")
            return {"kind": "DataFrame", "rows": int(len(obj)), "cols": cols,
                    "first_row": head[0] if head else None}
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        head = obj[0] if obj else None
        sample = None
        if head is not None:
            sample = list(head.keys())[:14] if isinstance(head, dict) else str(type(head).__name__)
        return {"kind": "list", "len": len(obj), "item": sample}
    if isinstance(obj, dict):
        return {"kind": "dict", "keys": list(obj.keys())[:14]}
    return {"kind": type(obj).__name__, "repr": str(obj)[:160]}


def probe(name, fn):
    rec = {"call": name, "status": None, "detail": None}
    try:
        out = fn()
        rec["status"] = "OK"
        rec["detail"] = _summarise(out)
    except Exception as e:  # noqa: BLE001 — we want every failure classified
        etype = type(e).__name__
        msg = str(e)
        low = msg.lower()
        if "403" in msg or "permission" in low or "无权限" in msg or "权限" in msg:
            rec["status"] = "FORBIDDEN(tier)"
        elif "401" in msg or "auth" in low:
            rec["status"] = "AUTH"
        elif "429" in msg or "rate" in low:
            rec["status"] = "RATE_LIMIT"
        elif "404" in msg or "not found" in low:
            rec["status"] = "NOT_FOUND"
        else:
            rec["status"] = f"ERR:{etype}"
        rec["detail"] = msg[:200]
    print(f"  [{rec['status']:>16}] {name}  ::  {json.dumps(rec['detail'], ensure_ascii=False, default=str)[:150]}")
    return rec


def main():
    print(f"TickFlow probe :: endpoint={ENDPOINT} :: token={'SET' if TOKEN else 'MISSING'} :: sdk={getattr(tickflow,'__version__','?')}")
    tf = tickflow.TickFlow(api_key=TOKEN, base_url=ENDPOINT) if TOKEN else tickflow.TickFlow.free()

    results = []
    print("\n== klines (daily, raw + adjusted) ==")
    results.append(probe("klines.get(1d,count=5,adjust=none)",
                         lambda: tf.klines.get(SYM, period="1d", count=5, adjust="none", as_dataframe=True)))
    results.append(probe("klines.get(1d,count=5,adjust=forward[qfq])",
                         lambda: tf.klines.get(SYM, period="1d", count=5, adjust="forward", as_dataframe=True)))
    results.append(probe("klines.get(1d,count=5,adjust=backward[hfq])",
                         lambda: tf.klines.get(SYM, period="1d", count=5, adjust="backward", as_dataframe=True)))
    results.append(probe("klines.batch(1d,count=5,adjust=forward)",
                         lambda: tf.klines.batch([SYM, SYM2], period="1d", count=5, adjust="forward",
                                                 as_dataframe=True, show_progress=False)))
    results.append(probe("klines.ex_factors([sym])",
                         lambda: tf.klines.ex_factors([SYM], as_dataframe=True)))

    print("\n== klines (intraday / minute) ==")
    results.append(probe("klines.intraday(1m,count=5)",
                         lambda: tf.klines.intraday(SYM, period="1m", count=5, as_dataframe=True)))
    results.append(probe("klines.get(5m,count=5)",
                         lambda: tf.klines.get(SYM, period="5m", count=5, as_dataframe=True)))
    results.append(probe("klines.get(15m,count=5)",
                         lambda: tf.klines.get(SYM, period="15m", count=5, as_dataframe=True)))
    results.append(probe("klines.get(60m,count=5)",
                         lambda: tf.klines.get(SYM, period="60m", count=5, as_dataframe=True)))
    results.append(probe("klines.intraday_batch(1m,count=5)",
                         lambda: tf.klines.intraday_batch([SYM, SYM2], period="1m", count=5, as_dataframe=True)))

    print("\n== quotes (realtime) ==")
    results.append(probe("quotes.get(symbols=)",
                         lambda: tf.quotes.get(symbols=[SYM, SYM2], as_dataframe=True)))
    results.append(probe("quotes.get_by_symbols",
                         lambda: tf.quotes.get_by_symbols([SYM, SYM2], as_dataframe=True)))
    results.append(probe("quotes.get_by_universes(CN_Equity_A)",
                         lambda: tf.quotes.get_by_universes(["CN_Equity_A"], as_dataframe=True)))

    print("\n== depth (L2 order book) ==")
    results.append(probe("depth.get", lambda: tf.depth.get(SYM)))

    print("\n== financials ==")
    results.append(probe("financials.metrics([sym])", lambda: tf.financials.metrics([SYM], latest=True, as_dataframe=True)))
    results.append(probe("financials.income([sym])", lambda: tf.financials.income([SYM], latest=True, as_dataframe=True)))
    results.append(probe("financials.balance_sheet([sym])", lambda: tf.financials.balance_sheet([SYM], latest=True, as_dataframe=True)))
    results.append(probe("financials.cash_flow([sym])", lambda: tf.financials.cash_flow([SYM], latest=True, as_dataframe=True)))
    results.append(probe("financials.shares([sym])", lambda: tf.financials.shares([SYM], latest=True, as_dataframe=True)))

    print("\n== instruments / exchanges / universes ==")
    results.append(probe("exchanges.list", lambda: tf.exchanges.list()))
    results.append(probe("exchanges.get_instruments(SH,stock)", lambda: tf.exchanges.get_instruments("SH", "stock")))
    results.append(probe("instruments.get", lambda: tf.instruments.get(SYM)))
    results.append(probe("instruments.batch", lambda: tf.instruments.batch([SYM, SYM2])))
    results.append(probe("universes.list", lambda: tf.universes.list()))
    results.append(probe("universes.get(CN_Equity_A)", lambda: tf.universes.get("CN_Equity_A")))

    out_path = Path(__file__).resolve().parents[1] / "reports" / "data" / "tickflow_capability_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probed_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": ENDPOINT,
        "token_present": bool(TOKEN),
        "sdk_version": getattr(tickflow, "__version__", "?"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    ok = [r["call"] for r in results if r["status"] == "OK"]
    forbidden = [r["call"] for r in results if "FORBIDDEN" in (r["status"] or "")]
    other = [(r["call"], r["status"]) for r in results if r["status"] != "OK" and "FORBIDDEN" not in (r["status"] or "")]
    print(f"\nOK ({len(ok)}): {ok}")
    print(f"FORBIDDEN ({len(forbidden)}): {forbidden}")
    print(f"OTHER ({len(other)}): {other}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
