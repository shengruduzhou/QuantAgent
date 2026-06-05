"""verify-datasets-v8 — actually call every data provider + report coverage.

Spec section 1 + 12.6 require that every production training run is
backed by real PIT-correct data, not synthetic fallback. This CLI is
the operator's pre-flight: it tries every connected data source for
a short live sample and produces a markdown report listing:

* What works right now (with sample shape + date range)
* What is missing (network / token / package)
* What we can actually train on (recommended scope)

The command is **read-only** — it does not mutate any silver / gold
layer; it only fetches a few rows per source to confirm reachability.
"""

from __future__ import annotations

import importlib
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root


def _try_qlib_local(provider_uri: Path, sample_symbols: list[str]) -> dict[str, Any]:
    """Read the qlib bin files directly — no network."""
    out: dict[str, Any] = {"name": "qlib_local", "kind": "local_snapshot"}
    try:
        if not provider_uri.exists():
            return {**out, "status": "missing", "reason": f"path absent: {provider_uri}"}
        cal = (provider_uri / "calendars" / "day.txt").read_text().strip().splitlines()
        out["calendar_first"] = cal[0]
        out["calendar_last"] = cal[-1]
        out["calendar_days"] = len(cal)
        feats = provider_uri / "features"
        symbols_present = sorted(p.name for p in feats.iterdir() if p.is_dir())
        out["symbols_total"] = len(symbols_present)
        # check the sample symbols
        from quantagent.data.providers.qlib_provider import QlibProvider
        from quantagent.data.providers.base import ProviderRequest

        qp = QlibProvider(provider_uri=str(provider_uri))
        req = ProviderRequest(
            start_date=(pd.Timestamp(cal[-1]) - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
            end_date=cal[-1],
            symbols=tuple(sample_symbols),
        )
        t0 = time.time()
        res = qp.daily_ohlcv(req)
        out["status"] = "ok" if not res.frame.empty else "empty_slice"
        out["latency_sec"] = round(time.time() - t0, 3)
        out["sample_rows"] = int(len(res.frame))
        out["sample_symbols"] = list(res.frame["symbol"].unique()[:5]) if not res.frame.empty else []
        out["fields"] = list(res.frame.columns)
        if not res.frame.empty:
            out["sample_first_date"] = str(res.frame["trade_date"].min())
            out["sample_last_date"] = str(res.frame["trade_date"].max())
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"
        out["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    return out


def _try_silver_panel(panel_path: Path, sample_symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "silver_panel", "kind": "local_silver"}
    try:
        if not panel_path.exists():
            return {**out, "status": "missing", "reason": f"absent: {panel_path}"}
        t0 = time.time()
        df = pd.read_parquet(panel_path, columns=[
            "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "available_at",
        ])
        out["latency_sec"] = round(time.time() - t0, 3)
        out["total_rows"] = int(len(df))
        out["symbols_total"] = int(df["symbol"].nunique())
        out["first_date"] = str(df["trade_date"].min())
        out["last_date"] = str(df["trade_date"].max())
        sample = df[df["symbol"].isin(sample_symbols)]
        out["sample_rows"] = int(len(sample))
        out["sample_first_date"] = str(sample["trade_date"].min()) if not sample.empty else None
        out["sample_last_date"] = str(sample["trade_date"].max()) if not sample.empty else None
        out["status"] = "ok" if not sample.empty else "missing_symbols"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _try_akshare(sample_symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "akshare", "kind": "network_free"}
    try:
        ak = importlib.import_module("akshare")
    except ImportError as exc:
        return {**out, "status": "missing", "reason": f"package not installed: {exc}"}
    out["package_version"] = getattr(ak, "__version__", "?")
    probes: dict[str, dict[str, Any]] = {}
    sym = sample_symbols[0]
    ak_sym = sym.split(".")[0]  # akshare uses bare 6-digit code for A-share
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
        t0 = time.time()
        df = ak.stock_zh_a_hist(
            symbol=ak_sym, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        probes["stock_zh_a_hist_daily"] = {
            "status": "ok" if df is not None and not df.empty else "empty",
            "rows": int(len(df)) if df is not None else 0,
            "fields": list(df.columns) if df is not None else [],
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        probes["stock_zh_a_hist_daily"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        t0 = time.time()
        df = ak.stock_zh_a_hist_min_em(
            symbol=ak_sym, period="60",
            start_date=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d 09:30:00"),
            end_date=datetime.now().strftime("%Y-%m-%d 15:00:00"),
            adjust="qfq",
        )
        probes["stock_zh_a_hist_min_60"] = {
            "status": "ok" if df is not None and not df.empty else "empty",
            "rows": int(len(df)) if df is not None else 0,
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        probes["stock_zh_a_hist_min_60"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        t0 = time.time()
        df = ak.stock_financial_em(symbol=ak_sym)
        probes["stock_financial_em"] = {
            "status": "ok" if df is not None and not df.empty else "empty",
            "rows": int(len(df)) if df is not None else 0,
            "fields": list(df.columns)[:10] if df is not None and not df.empty else [],
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        probes["stock_financial_em"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        t0 = time.time()
        df = ak.stock_lhb_detail_daily_sina(
            date=(datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
        )
        probes["stock_lhb_detail_daily"] = {
            "status": "ok" if df is not None and not df.empty else "empty",
            "rows": int(len(df)) if df is not None else 0,
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        probes["stock_lhb_detail_daily"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        t0 = time.time()
        df = ak.stock_individual_fund_flow(stock=ak_sym, market="sh" if sym.endswith(".SH") else "sz")
        probes["stock_individual_fund_flow"] = {
            "status": "ok" if df is not None and not df.empty else "empty",
            "rows": int(len(df)) if df is not None else 0,
            "latency_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        probes["stock_individual_fund_flow"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    ok_count = sum(1 for v in probes.values() if v.get("status") == "ok")
    out["status"] = "ok" if ok_count > 0 else "all_failed"
    out["probes"] = probes
    out["probes_ok"] = ok_count
    out["probes_total"] = len(probes)
    return out


def _try_tushare(token: str | None, sample_symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "tushare", "kind": "network_token"}
    try:
        ts = importlib.import_module("tushare")
    except ImportError as exc:
        return {**out, "status": "missing", "reason": f"package not installed: {exc}"}
    out["package_version"] = getattr(ts, "__version__", "?")
    if not token:
        return {**out, "status": "no_token", "reason": "TUSHARE_TOKEN env / .env missing"}
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        sym = sample_symbols[0]
        # ts uses '600519.SH' (matches our format)
        t0 = time.time()
        df = pro.daily(
            ts_code=sym,
            start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        out["daily_status"] = "ok" if df is not None and not df.empty else "empty"
        out["daily_rows"] = int(len(df)) if df is not None else 0
        out["daily_fields"] = list(df.columns) if df is not None else []
        out["latency_sec"] = round(time.time() - t0, 3)
        out["status"] = out["daily_status"]
    except Exception as exc:
        out["status"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _try_baostock(sample_symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": "baostock", "kind": "network_free"}
    try:
        bs = importlib.import_module("baostock")
    except ImportError as exc:
        return {**out, "status": "missing", "reason": f"package not installed: {exc}"}
    out["package_version"] = getattr(bs, "__version__", "?")
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return {**out, "status": "login_failed", "reason": lg.error_msg}
        sym = sample_symbols[0]
        # baostock uses 'sh.600519' format
        head, tail = sym.split(".")
        bs_sym = f"{tail.lower()}.{head}"
        t0 = time.time()
        rs = bs.query_history_k_data_plus(
            bs_sym, "date,open,high,low,close,volume,amount,turn",
            start_date=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d", adjustflag="2",
        )
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        out["status"] = "ok" if rows else "empty"
        out["rows"] = len(rows)
        out["latency_sec"] = round(time.time() - t0, 3)
        if rows:
            out["sample_row"] = rows[0]
    except Exception as exc:
        out["status"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _try_alpha_factor_lake() -> dict[str, Any]:
    """Check the pre-built alpha181 + training dataset gold layer."""
    out: dict[str, Any] = {"name": "alpha181_gold", "kind": "local_gold"}
    paths = {
        "training_dataset_alpha181_full_nosynth": Path(
            "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
        ),
        "factors_alpha181_full": Path("runtime/data/v7/silver/factors/alpha181_full_nosynth.parquet"),
        "sector_map": Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        "fundamentals": Path("runtime/data/v7/silver/fundamentals"),
        "st_flags": Path("runtime/data/v7/silver/st_flags/st_flags.parquet"),
        "valuation": Path("runtime/data/v7/silver/valuation"),
    }
    for k, p in paths.items():
        if not p.exists():
            out[k] = {"status": "missing"}
            continue
        if p.is_dir():
            files = list(p.glob("*.parquet"))
            out[k] = {"status": "ok", "n_files": len(files)}
            continue
        try:
            head = pd.read_parquet(p).head(2)
            out[k] = {
                "status": "ok",
                "bytes": p.stat().st_size,
                "fields_count": len(head.columns),
                "fields_first10": list(head.columns[:10]),
            }
        except Exception as exc:
            out[k] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    out["status"] = "ok"
    return out


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset verification report\n",
        f"_Generated: {report['generated_at']}_\n",
        f"Sample symbols probed: `{', '.join(report['sample_symbols'])}`\n\n",
    ]
    # summary
    lines.append("## Summary\n\n| source | status | notes |\n|---|---|---|")
    for src in report["sources"]:
        notes = src.get("reason", "")
        if not notes and src.get("status") == "ok":
            notes = f"rows={src.get('sample_rows', src.get('total_rows', src.get('rows', '?')))}"
        lines.append(f"| {src['name']} | **{src['status']}** | {notes} |")
    lines.append("")
    # per-source detail
    for src in report["sources"]:
        lines.append(f"\n## {src['name']} ({src['kind']})\n")
        lines.append("```json")
        lines.append(json.dumps(src, indent=2, default=str))
        lines.append("```\n")
    return "\n".join(lines)


@app.command("verify-datasets-v8")
def verify_datasets_v8(
    sample_symbols: str = typer.Option(
        "600519.SH,000001.SZ,600036.SH",
        help="comma-separated symbols to probe across providers",
    ),
    qlib_provider_uri: Path = typer.Option(
        Path("runtime/data/raw/qlib/cn_data"),
        help="local qlib bin root",
    ),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        help="v7 silver market panel parquet",
    ),
    tushare_token: Optional[str] = typer.Option(
        None,
        help="TuShare API token. If absent the CLI reads TUSHARE_TOKEN env var.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help="markdown report destination (default: runtime/reports/v8/datasets_verification.md)",
    ),
    json_path: Optional[Path] = typer.Option(
        None,
        help="raw JSON report destination",
    ),
):
    """Ping every data source and emit a coverage report."""
    import os

    syms = [s.strip() for s in sample_symbols.split(",") if s.strip()]
    typer.echo(f"[verify] probing {len(syms)} symbols across providers …")

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_symbols": syms,
        "sources": [],
    }

    typer.echo("[1/5] qlib local …")
    report["sources"].append(_try_qlib_local(qlib_provider_uri, syms))

    typer.echo("[2/5] silver panel …")
    report["sources"].append(_try_silver_panel(silver_panel_path, syms))

    typer.echo("[3/5] akshare (network) …")
    report["sources"].append(_try_akshare(syms))

    typer.echo("[4/5] tushare (network + token) …")
    token = tushare_token or os.environ.get("TUSHARE_TOKEN")
    report["sources"].append(_try_tushare(token, syms))

    typer.echo("[5/5] baostock (network) …")
    report["sources"].append(_try_baostock(syms))

    typer.echo("[+] alpha181 gold lake …")
    report["sources"].append(_try_alpha_factor_lake())

    # Recommendation: which source can drive a full training run
    candidate = next(
        (s for s in report["sources"] if s["name"] == "alpha181_gold" and s.get("training_dataset_alpha181_full_nosynth", {}).get("status") == "ok"),
        None,
    )
    if candidate:
        report["recommended_training_input"] = (
            "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
        )
    else:
        # fall back to silver panel
        silver = next((s for s in report["sources"] if s["name"] == "silver_panel" and s["status"] == "ok"), None)
        report["recommended_training_input"] = (
            "runtime/data/v7/silver/market_panel/market_panel.parquet" if silver else None
        )

    out_md = output_path or (default_reports_root() / "v8" / "datasets_verification.md")
    out_json = json_path or (default_reports_root() / "v8" / "datasets_verification.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    typer.echo(f"\nreport → {out_md}\njson  → {out_json}")
    typer.echo(f"recommended_training_input: {report['recommended_training_input']}")
    return out_md


__all__ = ["verify_datasets_v8"]
