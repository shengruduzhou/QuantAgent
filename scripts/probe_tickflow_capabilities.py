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


def _ws_quotes_smoke(tf, wait_seconds: float = 8.0) -> dict:
    """Bounded WebSocket quotes smoke: connect, subscribe 1 symbol, wait, close.

    Never logs the authenticated URL or token. Outside trading hours zero
    pushed messages is expected — a clean subscribe (no error callback, no
    exception) is the entitlement signal; message count is informational.
    """
    import time

    got = {"messages": 0, "errors": []}
    tf.stream.on_quotes(lambda rows: got.__setitem__("messages", got["messages"] + len(rows or ())))
    tf.stream.on_error(lambda msg: got["errors"].append(str(msg)[:160]))
    tf.stream.connect(block=False)
    try:
        tf.stream.subscribe("quotes", [SYM])
        time.sleep(wait_seconds)
        tf.stream.unsubscribe("quotes", [SYM])
    finally:
        tf.stream.close()
    if got["errors"]:
        raise RuntimeError("; ".join(got["errors"]))
    return got


# ---------------------------------------------------------------------------
# Capability manifest (machine-readable registry per prompt/mission 2026-07-12)
# ---------------------------------------------------------------------------

# operation inventory verified against https://api.tickflow.org/openapi.json
# (openapi 3.1.0, 19 paths / 21 ops, fetched 2026-07-12 — matches baseline).
# probe_key maps each operation to the probe() call that evidences its
# permission status; "inherit:<key>" reuses another op's evidence when the
# server gates them with the same permission (verified message text).
_OPERATIONS: list[dict] = [
    dict(operation_id="get_klines_daily", method="GET", path="/v1/klines", domain="klines",
         probe_key="klines.get(1d,count=5,adjust=none)", semantics="historical",
         pit="PIT-safe (historical bars; available_at=trade_date+1d convention)",
         batch_limit="count<=10000/request",
         implemented="TickflowProvider.daily_ohlcv/adjusted_prices; scripts/update_market_panel_daily.py; scripts/fetch_tickflow_daily_klines.py",
         notes="raw/qfq/hfq via adjust=; minute periods separately gated"),
    dict(operation_id="get_klines_minute", method="GET", path="/v1/klines?period=5m|15m|30m|60m", domain="klines",
         probe_key="klines.get(5m,count=5)", semantics="historical",
         pit="would be PIT-safe", batch_limit="count<=10000/request",
         implemented="none (gated); minute source = runtime/data/raw/qlib/cn_data_1min (2020-09..2021-06) + silver/minute_bars",
         notes="无分钟K线查询权限"),
    dict(operation_id="get_klines_batch", method="GET", path="/v1/klines/batch", domain="klines",
         probe_key="klines.batch(1d,count=5,adjust=forward)", semantics="historical",
         pit="PIT-safe", batch_limit="symbols comma-list; separately priced tier",
         implemented="transparent per-symbol fallback in TickflowProvider._fetch_daily",
         notes="performance-only gap; fallback loop is the production path"),
    dict(operation_id="get_ex_factors", method="GET", path="/v1/klines/ex-factors", domain="klines",
         probe_key="klines.ex_factors([sym])", semantics="historical",
         pit="PIT-safe", batch_limit="symbols list",
         implemented="WORKAROUND: klines.get(adjust=forward/backward) returns server-side adjusted OHLC",
         notes="无除权因子查询权限 — adjusted series obtainable, raw factors not"),
    dict(operation_id="get_intraday", method="GET", path="/v1/klines/intraday", domain="klines",
         probe_key="klines.intraday(1m,count=5)", semantics="current-day snapshot",
         pit="current-day only", batch_limit="count param",
         implemented="none (gated)", notes="无日内分时查询权限"),
    dict(operation_id="get_intraday_batch", method="GET", path="/v1/klines/intraday/batch", domain="klines",
         probe_key="klines.intraday_batch(1m,count=5)", semantics="current-day snapshot",
         pit="current-day only", batch_limit="symbols list",
         implemented="none (gated)", notes="无日内分时查询批量查询权限"),
    dict(operation_id="get_quotes", method="GET", path="/v1/quotes", domain="quotes",
         probe_key="quotes.get(symbols=)", semantics="realtime snapshot (NOT historical)",
         pit="NOT for backtests; forward collection only", batch_limit="symbols comma-list",
         implemented="scripts/intraday_dot_signals.py + live forward loop; volume+amount verified present",
         notes="quote volume is cumulative-day, not bar volume"),
    dict(operation_id="post_quotes", method="POST", path="/v1/quotes", domain="quotes",
         probe_key="quotes.get_by_symbols", semantics="realtime snapshot",
         pit="NOT for backtests", batch_limit="JSON symbol list",
         implemented="SDK get_by_symbols", notes=""),
    dict(operation_id="get_quotes_by_universe", method="GET", path="/v1/quotes?universes=", domain="quotes",
         probe_key="quotes.get_by_universes(CN_Equity_A)", semantics="realtime snapshot",
         pit="NOT for backtests", batch_limit="universe ids",
         implemented="none (gated); full-universe snapshots = explicit symbol batches",
         notes="无标的池查询权限"),
    dict(operation_id="get_depth", method="GET", path="/v1/depth", domain="depth",
         probe_key="depth.get", semantics="realtime L2 5-level snapshot (NO history)",
         pit="forward collection only", batch_limit="single symbol",
         implemented="scripts/collect_tickflow_depth.py (forward collector, BLOCKED by tier)",
         notes="无市场深度查询权限（市场: CN）"),
    dict(operation_id="get_depth_batch", method="GET", path="/v1/depth/batch", domain="depth",
         probe_key="inherit:depth.get", semantics="realtime snapshot",
         pit="forward collection only", batch_limit="symbols comma-list",
         implemented="none; SDK 0.1.22 lacks depth.batch (REST-only op)",
         notes="same CN-market depth permission as /v1/depth"),
    dict(operation_id="list_exchanges", method="GET", path="/v1/exchanges", domain="instruments",
         probe_key="exchanges.list", semantics="snapshot",
         pit="metadata", batch_limit="n/a",
         implemented="TickflowProvider._ensure_all_instruments", notes="SH/SZ/BJ/US/HK"),
    dict(operation_id="get_exchange_instruments", method="GET", path="/v1/exchanges/{exchange}/instruments", domain="instruments",
         probe_key="exchanges.get_instruments(SH,stock)", semantics="current snapshot",
         pit="listing_date in ext; delistings NOT retained by endpoint", batch_limit="per exchange",
         implemented="TickflowProvider.stock_basic; scripts/fetch_sector_map_tickflow.py",
         notes="survivorship: current listing snapshot only"),
    dict(operation_id="get_instruments", method="GET", path="/v1/instruments", domain="instruments",
         probe_key="instruments.get", semantics="current snapshot", pit="metadata",
         batch_limit="symbols comma-list", implemented="SDK instruments.get", notes=""),
    dict(operation_id="post_instruments", method="POST", path="/v1/instruments", domain="instruments",
         probe_key="instruments.batch", semantics="current snapshot", pit="metadata",
         batch_limit="JSON list (documented ~1000)", implemented="SDK instruments.batch", notes=""),
    dict(operation_id="list_universes", method="GET", path="/v1/universes", domain="universes",
         probe_key="universes.list", semantics="current snapshot",
         pit="membership is CURRENT — survivorship_safe=false for history", batch_limit="n/a",
         implemented="TickflowProvider._ensure_industry_map (SW1/SW2)", notes="1013 universes"),
    dict(operation_id="get_universe", method="GET", path="/v1/universes/{id}", domain="universes",
         probe_key="universes.get(CN_Equity_A)", semantics="current snapshot",
         pit="survivorship_safe=false for history", batch_limit="n/a",
         implemented="sector_map builder", notes=""),
    dict(operation_id="batch_universes", method="POST", path="/v1/universes/batch", domain="universes",
         probe_key="universes.batch", semantics="current snapshot",
         pit="survivorship_safe=false for history", batch_limit="JSON id list",
         implemented="SDK universes.batch", notes=""),
    dict(operation_id="get_income", method="GET", path="/v1/financials/income", domain="financials",
         probe_key="financials.income([sym])", semantics="historical statements",
         pit="filter is period_end — publication time must come from announce_date field; never use period_end as availability",
         batch_limit="symbols comma-list",
         implemented="TickflowProvider.financials_income (fail-loud); on-disk history silver/fundamentals + tickflow_fin_quarterly pulled under an EARLIER entitlement (~<=2026-05); refresh = akshare/tushare/baostock",
         notes="无公司财务数据查询权限 (entitlement since revoked)"),
    dict(operation_id="get_balance_sheet", method="GET", path="/v1/financials/balance-sheet", domain="financials",
         probe_key="financials.balance_sheet([sym])", semantics="historical statements",
         pit="same period_end caveat", batch_limit="symbols comma-list",
         implemented="TickflowProvider.financials_balance_sheet (fail-loud); alternatives as income",
         notes="无公司财务数据查询权限"),
    dict(operation_id="get_cash_flow", method="GET", path="/v1/financials/cash-flow", domain="financials",
         probe_key="financials.cash_flow([sym])", semantics="historical statements",
         pit="same period_end caveat", batch_limit="symbols comma-list",
         implemented="TickflowProvider.financials_cash_flow (fail-loud)", notes="无公司财务数据查询权限"),
    dict(operation_id="get_metrics", method="GET", path="/v1/financials/metrics", domain="financials",
         probe_key="financials.metrics([sym])", semantics="historical statements",
         pit="announce_date present when entitled (verified in on-disk history)", batch_limit="symbols comma-list",
         implemented="TickflowProvider.financials_metrics (fail-loud); on-disk metrics_panel.parquet is PIT-audited (EXP-020)",
         notes="无公司财务数据查询权限"),
    dict(operation_id="get_shares", method="GET", path="/v1/financials/shares", domain="financials",
         probe_key="financials.shares([sym])", semantics="historical share-capital",
         pit="same period_end caveat", batch_limit="symbols comma-list",
         implemented="none; share-capital absent on disk (why market_cap/turnover_rate factors are excluded — see H-020)",
         notes="无公司财务数据查询权限"),
    dict(operation_id="ws_stream_quotes", method="WS", path="wss://.../v1/ws/stream#quotes", domain="stream",
         probe_key="stream.quotes(subscribe/unsubscribe)", semantics="realtime push",
         pit="forward collection only", batch_limit="server-side subscription limits",
         implemented="SDK stream namespace; no production collector (optional)", notes=""),
    dict(operation_id="ws_stream_depth", method="WS", path="wss://.../v1/ws/stream#depth", domain="stream",
         probe_key="inherit:depth.get", semantics="realtime push",
         pit="forward collection only", batch_limit="server-side subscription limits",
         implemented="none; gated with REST depth", notes="depth permission gates the channel"),
]


def _classify(status: str | None, implemented: str) -> str:
    if status == "OK":
        return "SUPPORTED" if implemented and not implemented.startswith("none") else "NOT_IMPLEMENTED"
    if status and "FORBIDDEN" in status:
        return "UNAUTHORIZED"
    if status == "NOT_FOUND":
        return "UNSUPPORTED_BY_API"
    if status in ("RATE_LIMIT", "AUTH"):
        return "TEMPORARILY_UNAVAILABLE"
    return "BLOCKED_BY_MISSING_SPEC" if status is None else "TEMPORARILY_UNAVAILABLE"


def build_capability_manifest(results: list[dict], probed_at: str, sdk_version: str) -> dict:
    by_key = {r["call"]: r for r in results}

    def resolve(probe_key: str) -> tuple[str | None, str]:
        if probe_key.startswith("inherit:"):
            parent = by_key.get(probe_key.split(":", 1)[1])
            status = parent["status"] if parent else None
            return status, f"inferred_from:{probe_key.split(':', 1)[1]}"
        rec = by_key.get(probe_key)
        return (rec["status"] if rec else None), "direct"

    ops = []
    for op in _OPERATIONS:
        status, evidence = resolve(op["probe_key"])
        ops.append({**{k: v for k, v in op.items() if k != "probe_key"},
                    "permission_status": status or "NOT_PROBED",
                    "probe_evidence": evidence,
                    "classification": _classify(status, op["implemented"]),
                    "last_probe": probed_at})
    return {"schema_version": 1, "probed_at": probed_at, "sdk_version": sdk_version,
            "openapi_verified": "2026-07-12 (19 paths / 21 ops, matches baseline inventory)",
            "operations": ops}


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
    results.append(probe("universes.batch", lambda: tf.universes.batch(["CN_Equity_A"])))

    print("\n== websocket stream (quotes channel, bounded smoke) ==")
    results.append(probe("stream.quotes(subscribe/unsubscribe)", lambda: _ws_quotes_smoke(tf)))

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

    manifest = build_capability_manifest(results, payload["probed_at"], payload["sdk_version"])
    manifest_path = out_path.with_name("tickflow_capability_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    counts: dict[str, int] = {}
    for op in manifest["operations"]:
        counts[op["classification"]] = counts.get(op["classification"], 0) + 1
    print(f"Wrote {manifest_path} :: {counts}")
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
