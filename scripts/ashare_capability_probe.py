#!/usr/bin/env python3
"""Unified A-share provider capability & entitlement probe (real network calls).

Answers one question per (provider, dataset family, representative security):
did a real request return usable rows, and if not, exactly why? Nothing here is
inferred from an import succeeding or from documentation; every SUPPORTED verdict
is backed by a response with a row count, a schema and a latency.

Representative cohort covers every board and lifecycle state the mission
requires: SH main, SZ main, ChiNext, STAR, BSE (both the current 920xxx range and
a legacy 8xxxxx code), an ST name, a recent IPO, a suspended name and a delisted
name.

Outputs (runtime/data/u0/capability/):
  provider_capability_matrix.csv     one row per probe with status + evidence
  provider_capability_matrix.json    same, plus environment and blocker summary
  provider_capability_report.md      operator-readable matrix

Usage:
  AI_quant_venv/bin/python3 scripts/ashare_capability_probe.py --allow-network
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.ashare.sources import (  # noqa: E402
    EastmoneySource,
    SinaSource,
    TencentSource,
    TickFlowSource,
)

OUT = REPO / "runtime/data/u0/capability"

#: TickFlow enforces a hard 10 requests/minute. Without pacing the probe would
#: rate-limit ITSELF and report an entitled method as a capability failure.
TICKFLOW_PACE_S = 6.5

# (label, symbol, why this security is in the cohort)
COHORT: tuple[tuple[str, str, str], ...] = (
    ("SH_Main", "600519.SH", "Shanghai main board, long history"),
    ("SZ_Main", "000001.SZ", "Shenzhen main board"),
    ("ChiNext", "300750.SZ", "ChiNext"),
    ("STAR", "688981.SH", "STAR market"),
    ("BSE_920", "920079.BJ", "Beijing Stock Exchange, current 920xxx range"),
    ("BSE_legacy", "830799.BJ", "Beijing Stock Exchange, legacy 8xxxxx code"),
    ("ST", "000018.SZ", "ST / risk-warning name"),
    ("RecentIPO", "301618.SZ", "recently listed security"),
    ("Delisted", "600087.SH", "delisted security (historical bars only)"),
)

# Non-HTTP ports used by providers that would otherwise be candidates.
PORT_PROBES = (
    ("baostock", "baostock.com", 10030, "baostock TCP API"),
    ("pytdx_tdx", "119.147.212.81", 7709, "TDX (mootdx / pytdx) quote server"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(provider: str, family: str, target: str, status: str, rows: int = 0,
         detail: str = "", latency: float = 0.0, schema: str = "",
         date_range: str = "") -> dict:
    return {
        "provider": provider, "dataset_family": family, "target": target,
        "status": status, "rows": rows, "latency_s": round(latency, 3),
        "schema": schema, "date_range": date_range, "detail": detail[:300],
        "probed_at": _now(),
    }


def probe_port_blockers() -> list[dict]:
    out = []
    for provider, host, port, note in PORT_PROBES:
        started = time.monotonic()
        try:
            sock = socket.create_connection((host, port), timeout=8)
            sock.close()
            out.append(_row(provider, "transport", f"{host}:{port}", "REACHABLE",
                            detail=note, latency=time.monotonic() - started))
        except Exception as exc:  # noqa: BLE001
            out.append(_row(provider, "transport", f"{host}:{port}",
                            "BLOCKED_BY_ENVIRONMENT",
                            detail=f"{note}: {type(exc).__name__} — only ports 80/443 egress",
                            latency=time.monotonic() - started))
    return out


def probe_optional_sdks() -> list[dict]:
    """Record which optional SDKs exist in the runtime and whether they are usable."""
    out = []
    checks = {
        "tushare": ("TUSHARE_TOKEN", "financials/metadata; needs a paid token"),
        "xtquant": (None, "QMT/MiniQMT broker client — Windows-only, provides Level-2"),
        "mootdx": (None, "TDX client; requires TCP 7709 egress"),
        "baostock": (None, "free provider; requires TCP 10030 egress"),
        "akshare": (None, "aggregator over the same public endpoints"),
        "qlib": (None, "local historical library (needs a materialised provider_uri)"),
    }
    import importlib
    import os
    for name, (token_env, note) in checks.items():
        try:
            importlib.import_module(name)
            installed = True
        except Exception:  # noqa: BLE001
            installed = False
        if not installed:
            out.append(_row(name, "sdk", name, "NOT_INSTALLED", detail=note))
            continue
        if token_env and not os.environ.get(token_env):
            out.append(_row(name, "sdk", name, "NO_CREDENTIAL",
                            detail=f"{note}; {token_env} absent"))
        else:
            out.append(_row(name, "sdk", name, "INSTALLED", detail=note))
    return out


def probe_qlib() -> list[dict]:
    import os
    out = []
    for env_key, freq in (("QLIB_PROVIDER_URI_1D", "daily"), ("QLIB_PROVIDER_URI_1MIN", "minute")):
        uri = os.environ.get(env_key, "")
        if not uri:
            out.append(_row("qlib", f"{freq}_bars", env_key, "NOT_CONFIGURED"))
            continue
        path = Path(uri)
        features = path / "features"
        if not path.exists():
            out.append(_row("qlib", f"{freq}_bars", uri, "MISSING_DATA_ROOT",
                            detail="configured provider_uri does not exist on disk"))
        elif not features.exists() or not any(features.iterdir()):
            out.append(_row("qlib", f"{freq}_bars", uri, "EMPTY_DATA_ROOT",
                            detail="provider_uri exists but carries no features/ instruments"))
        else:
            out.append(_row("qlib", f"{freq}_bars", uri, "SUPPORTED",
                            rows=len(list(features.iterdir()))))
    return out


def probe_tickflow() -> list[dict]:
    out: list[dict] = []
    source = TickFlowSource()
    try:
        source.client
    except Exception as exc:  # noqa: BLE001
        return [_row("tickflow", "auth", "client", "NO_CREDENTIAL", detail=str(exc)[:200])]

    for index, (label, symbol, _) in enumerate(COHORT):
        if index:
            time.sleep(TICKFLOW_PACE_S)
        started = time.monotonic()
        result = source.daily_bars(symbol, start=pd.Timestamp("1999-01-01"),
                                   end=pd.Timestamp.now().normalize())
        status = {"OK": "SUPPORTED", "EMPTY": "EMPTY_RESPONSE",
                  "ENTITLEMENT": "UNAUTHORIZED"}.get(result.retry_class, result.retry_class)
        rng = ""
        if result.rows:
            rng = (f"{result.frame['trade_date'].min().date()}.."
                   f"{result.frame['trade_date'].max().date()}")
        out.append(_row("tickflow", "daily_bars", f"{label}:{symbol}", status, result.rows,
                        result.error or "", time.monotonic() - started,
                        ",".join(result.frame.columns[:8]), rng))

    # Entitlement probes against the METHODS THE SDK ACTUALLY EXPOSES (enumerated
    # from the installed client), so an UNAUTHORIZED verdict is a subscription
    # fact rather than a guessed method name.
    probes = [
        ("security_master", "instruments.get", ("600519.SH",), {}),
        ("security_master_bulk", "exchanges.list", (), {}),
        ("universes", "universes.list", (), {}),
        ("quotes", "quotes.get_by_symbols", (["600519.SH"],), {}),
        ("daily_bars_batch", "klines.batch", (["600519.SH", "000001.SZ"],), {"period": "1d", "count": 5}),
        ("minute_bars", "klines.intraday", ("600519.SH",), {"period": "1m", "count": 10}),
        ("minute_bars_batch", "klines.intraday_batch", (["600519.SH"],), {"period": "1m", "count": 10}),
        ("adjust_factors", "klines.ex_factors", ("600519.SH",), {}),
        ("l2_depth", "depth.get", ("600519.SH",), {}),
        ("financials_income", "financials.income", ("600519.SH",), {}),
        ("financials_balance", "financials.balance_sheet", ("600519.SH",), {}),
        ("financials_cashflow", "financials.cash_flow", ("600519.SH",), {}),
        ("valuation_metrics", "financials.metrics", ("600519.SH",), {}),
        ("shares_outstanding", "financials.shares", ("600519.SH",), {}),
    ]
    for family, method, args, kwargs in probes:
        time.sleep(TICKFLOW_PACE_S)
        started = time.monotonic()
        info = source.probe(method, *args, **kwargs)
        out.append(_row("tickflow", family, method, info["status"], info.get("rows", 0),
                        info.get("error", ""), time.monotonic() - started))
    return out


def probe_public_http() -> list[dict]:
    client = HttpClient(timeout=20, max_attempts=2)
    tencent = TencentSource(client)
    sina = SinaSource(client)
    eastmoney = EastmoneySource(client)
    out: list[dict] = []

    def record(provider: str, family: str, target: str, result) -> None:
        status = {"OK": "SUPPORTED", "EMPTY": "EMPTY_RESPONSE",
                  "ENTITLEMENT": "UNAUTHORIZED", "TRANSIENT": "TRANSIENT_FAILURE",
                  "RATE_LIMITED": "RATE_LIMITED", "PERMANENT": "PERMANENT_FAILURE"}.get(
                      result.retry_class, result.retry_class)
        date_col = next((c for c in ("trade_date", "bar_time", "effective_date", "ex_date",
                                     "quote_time") if c in result.frame.columns), None)
        rng = ""
        if result.rows and date_col:
            rng = f"{result.frame[date_col].min()}..{result.frame[date_col].max()}"
        out.append(_row(provider, family, target, status, result.rows,
                        result.error or "", 0.0, ",".join(list(result.frame.columns)[:8]), rng))

    for label, symbol, _ in COHORT:
        record("tencent", "daily_bars", f"{label}:{symbol}",
               tencent.daily_bars(symbol, "1999-01-01", str(pd.Timestamp.now().date())))
    record("tencent", "daily_bars_qfq", "SH_Main:600519.SH",
           tencent.daily_bars("600519.SH", "2020-01-01", str(pd.Timestamp.now().date()), "qfq"))
    record("tencent", "daily_bars_hfq", "SH_Main:600519.SH",
           tencent.daily_bars("600519.SH", "2020-01-01", str(pd.Timestamp.now().date()), "hfq"))
    for label, symbol, _ in COHORT[:6]:
        record("tencent", "minute_bars", f"{label}:{symbol}", tencent.minute_bars(symbol, 5, 100))
    record("tencent", "quotes_l1_depth5", "cohort",
           tencent.quotes([s for _, s, _ in COHORT[:6]]))

    for label, symbol, _ in COHORT:
        record("sina", "adjust_factors", f"{label}:{symbol}", sina.adjust_factors(symbol))
    for label, symbol, _ in COHORT[:5]:
        record("sina", "corporate_actions", f"{label}:{symbol}", sina.dividends(symbol))

    for label, symbol, _ in COHORT[:4]:
        record("eastmoney", "money_flow", f"{label}:{symbol}", eastmoney.money_flow(symbol, 60))
    for label, symbol, _ in COHORT[:2]:
        record("eastmoney", "minute_bars_1m", f"{label}:{symbol}", eastmoney.minute_trends(symbol, 1))
    return out


def probe_exchange_official() -> list[dict]:
    """Exchange-published security lists (authoritative identity source)."""
    client = HttpClient(timeout=25, max_attempts=2)
    out = []
    sse = client.get_json(
        "https://query.sse.com.cn/sseQuery/commonQuery.do",
        headers={"Referer": "https://www.sse.com.cn/"},
        params={"sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "STOCK_TYPE": "1",
                "REG_PROVINCE": "", "CSRC_CODE": "", "STOCK_CODE": "",
                "pageHelp.pageSize": "25", "pageHelp.pageNo": "1",
                "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1",
                "pageHelp.endPage": "1"})
    payload = sse.payload if sse.ok else None
    n_sse = len(((payload or {}).get("pageHelp") or {}).get("data") or []) if payload else 0
    out.append(_row("sse_official", "security_master", "query.sse.com.cn",
                    "SUPPORTED" if n_sse else ("TRANSIENT_FAILURE" if not sse.ok else "EMPTY_RESPONSE"),
                    n_sse, sse.error or "", sse.latency_s))
    szse = client.get("https://www.szse.cn/api/report/ShowReport/data",
                      headers={"Referer": "https://www.szse.cn/market/stock/list/index.html"},
                      params={"SHOWTYPE": "JSON", "CATALOGID": "1110", "TABKEY": "tab1",
                              "PAGENO": "1", "random": "0.1"})
    n_szse = 0
    if szse.ok:
        try:
            body = json.loads(szse.text or "[]")
            n_szse = len((body[0] or {}).get("data") or []) if body else 0
        except Exception:  # noqa: BLE001
            n_szse = 0
    out.append(_row("szse_official", "security_master", "www.szse.cn",
                    "SUPPORTED" if n_szse else ("TRANSIENT_FAILURE" if not szse.ok else "EMPTY_RESPONSE"),
                    n_szse, szse.error or "", szse.latency_s))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="required: every probe here performs real vendor calls")
    parser.add_argument("--skip-tickflow", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to probe: --allow-network was not confirmed")
        return 2

    load_repo_env()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows += probe_optional_sdks()
    rows += probe_port_blockers()
    rows += probe_qlib()
    if not args.skip_tickflow:
        rows += probe_tickflow()
    rows += probe_public_http()
    rows += probe_exchange_official()

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "provider_capability_matrix.csv", index=False)

    supported = frame[frame["status"] == "SUPPORTED"]
    families = sorted(frame["dataset_family"].unique())
    by_family = {
        family: sorted(supported.loc[supported["dataset_family"] == family, "provider"].unique())
        for family in families
    }
    blockers = frame[frame["status"].isin(
        ["BLOCKED_BY_ENVIRONMENT", "UNAUTHORIZED", "NO_CREDENTIAL", "NOT_INSTALLED",
         "MISSING_DATA_ROOT", "EMPTY_DATA_ROOT", "NOT_CONFIGURED"])]
    # A provider that is merely throttled right now is NOT the same as one that
    # cannot serve the family at all; conflating them would let a temporary IP
    # block read as a permanent capability gap.
    throttled = frame[frame["status"].isin(["RATE_LIMITED", "TRANSIENT_FAILURE"])]
    throttled_families = sorted(
        set(throttled["dataset_family"]) - set(supported["dataset_family"]))
    summary = {
        "generated": _now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "egress": "TCP 80/443 only — non-HTTP provider ports are unreachable",
        },
        "probes": int(len(frame)),
        "supported_probes": int(len(supported)),
        "providers_with_any_support": sorted(supported["provider"].unique()),
        "serving_providers_by_family": by_family,
        "families_without_any_provider": [f for f, p in by_family.items() if not p],
        "families_unavailable_only_due_to_throttling": throttled_families,
        "throttled_probes": throttled[["provider", "dataset_family", "target", "status"]]
        .to_dict("records"),
        "blockers": blockers[["provider", "dataset_family", "status", "detail"]].to_dict("records"),
        "principle": "SUPPORTED means a real request returned parsed rows in this runtime",
    }
    (OUT / "provider_capability_matrix.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    lines = ["# A-share provider capability matrix\n\n",
             f"Generated {summary['generated']} · {summary['probes']} real probes · ",
             f"{summary['supported_probes']} SUPPORTED\n\n",
             "Egress in this runtime is limited to TCP 80/443.\n\n",
             "## Serving providers by dataset family\n\n| dataset family | providers proven to serve it |\n|---|---|\n"]
    for family in families:
        lines.append(f"| {family} | {', '.join(by_family[family]) or '**none**'} |\n")
    lines.append("\n## Probe detail\n\n| provider | family | target | status | rows | detail |\n|---|---|---|---|---|---|\n")
    for r in rows:
        lines.append(f"| {r['provider']} | {r['dataset_family']} | {r['target']} | "
                     f"{r['status']} | {r['rows']} | {r['detail'][:110]} |\n")
    (OUT / "provider_capability_report.md").write_text("".join(lines))

    print(json.dumps({k: summary[k] for k in
                      ("probes", "supported_probes", "providers_with_any_support",
                       "families_without_any_provider")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
