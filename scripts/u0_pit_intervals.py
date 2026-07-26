#!/usr/bin/env python3
"""Point-in-time interval tables sourced from real, dated vendor records.

Sub-commands (all resumable, all writing provenance with every row):

``calendar``    exchange trading calendar (Sina, 1990-12-19 onward). Required to
                tell a missing session apart from a halted one.
``factors``     cumulative backward-adjustment factor series per symbol (Sina).
                A factor step IS an ex-rights event, so this doubles as the
                machine-readable corporate-action identity.
``dividends``   dividend / bonus / rights records with announce, record and
                ex-dates (Sina F10).
``suspension``  per-trading-date halt snapshots from the Eastmoney suspension
                report, folded into ``(symbol, start, end)`` intervals. Each
                snapshot carries the halt's own start date, so intervals are
                vendor-dated rather than inferred from bar gaps.
``st``          risk-warning (ST / *ST) state. The current state comes from the
                exchange instrument name; dated episodes are folded out of the
                SZSE short-name register. Exchanges with no dated register are
                reported as blocked rather than assumed never-ST.

Outputs land in runtime/data/u0/pit/.

Usage:
  AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py calendar --allow-network
  AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py factors --allow-network --max-minutes 120
  AI_quant_venv/bin/python3 scripts/u0_pit_intervals.py suspension --allow-network \\
      --start 2015-01-01 --max-minutes 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.ashare.sources import SinaSource  # noqa: E402
from quantagent.data.ashare.symbols import SymbolError, identify  # noqa: E402

OUT = REPO / "runtime/data/u0/pit"
MASTER = REPO / "runtime/data/u0/security_master.parquet"
BLOCKED = "BLOCKED_BY_DATA"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_master() -> pd.DataFrame:
    if not MASTER.exists():
        raise SystemExit(f"missing {MASTER.relative_to(REPO)} — run u0_security_master.py first")
    return pd.read_parquet(MASTER)


# ---------------------------------------------------------------------------
def cmd_calendar(args: argparse.Namespace) -> int:
    import akshare as ak

    raw = ak.tool_trade_date_hist_sina()
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce").dropna()
    frame = pd.DataFrame({
        "trade_date": dates.values, "exchange": "SSE_SZSE", "is_open": True,
        "source": "akshare.tool_trade_date_hist_sina", "source_endpoint": "sina trading calendar",
        "retrieved_at": _now(), "available_at": dates.dt.strftime("%Y-%m-%d").values,
        "quality_status": "OK",
    }).sort_values("trade_date").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "trading_calendar.parquet", index=False)
    payload = {"rows": int(len(frame)), "first": str(frame["trade_date"].min().date()),
               "last": str(frame["trade_date"].max().date()),
               "source": "akshare.tool_trade_date_hist_sina", "generated": _now()}
    (OUT / "trading_calendar_manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------------
def _per_symbol(args: argparse.Namespace, kind: str) -> int:
    master = _load_master()
    symbols = sorted(master["symbol"].astype(str).unique())
    if args.limit:
        symbols = symbols[:args.limit]
    staging = OUT / f"_{kind}"
    staging.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / f"{kind}_ledger.csv"
    done = {p.stem.replace("sym_", "").replace("_", ".") for p in staging.glob("sym_*.parquet")}
    source = SinaSource(HttpClient(timeout=20, max_attempts=3))
    fetch = source.adjust_factors if kind == "adjust_factors" else source.dividends

    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    ledger_rows, written, empty, failed = [], 0, 0, 0
    for index, symbol in enumerate(symbols):
        if time.monotonic() > deadline:
            print(f"time budget reached after {index} symbols", flush=True)
            break
        if symbol in done:
            continue
        result = fetch(symbol)
        if result.rows:
            result.frame.to_parquet(staging / f"sym_{symbol.replace('.', '_')}.parquet", index=False)
            written += 1
        elif result.retry_class == "EMPTY":
            empty += 1
        else:
            failed += 1
        ledger_rows.append({"symbol": symbol, "rows": result.rows,
                            "retry_class": result.retry_class, "detail": (result.error or "")[:160],
                            "recorded_at": _now()})
        if len(ledger_rows) >= 50:
            pd.DataFrame(ledger_rows).to_csv(ledger_path, mode="a", index=False,
                                             header=not ledger_path.exists())
            ledger_rows = []
            print(f"  {index + 1}/{len(symbols)} written={written} empty={empty} failed={failed} "
                  f"{time.monotonic() - started:.0f}s", flush=True)
    if ledger_rows:
        pd.DataFrame(ledger_rows).to_csv(ledger_path, mode="a", index=False,
                                         header=not ledger_path.exists())

    files = sorted(staging.glob("sym_*.parquet"))
    if files:
        frames = [pd.read_parquet(p) for p in files]
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(OUT / f"{kind}.parquet", index=False)
    else:
        combined = pd.DataFrame()
    payload = {"dataset": kind, "symbols_with_data": len(files), "rows": int(len(combined)),
               "universe": len(symbols), "written_this_run": written, "empty": empty,
               "failed": failed, "source": "sina", "generated": _now()}
    (OUT / f"{kind}_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_factors(args: argparse.Namespace) -> int:
    return _per_symbol(args, "adjust_factors")


def cmd_dividends(args: argparse.Namespace) -> int:
    return _per_symbol(args, "corporate_actions")


# ---------------------------------------------------------------------------
def cmd_suspension(args: argparse.Namespace) -> int:
    """Per-trading-date halt snapshots folded into intervals."""
    import akshare as ak

    calendar_path = OUT / "trading_calendar.parquet"
    if not calendar_path.exists():
        raise SystemExit("run `u0_pit_intervals.py calendar --allow-network` first")
    calendar = pd.read_parquet(calendar_path)
    today = pd.Timestamp.now().normalize()
    dates = calendar.loc[(calendar["trade_date"] >= pd.Timestamp(args.start)) &
                         (calendar["trade_date"] <= today), "trade_date"]
    dates = sorted(dates, reverse=True)          # newest first: most useful data lands early

    staging = OUT / "_suspension"
    staging.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in staging.glob("*.parquet")}
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    fetched = empty = failed = 0
    for index, date in enumerate(dates):
        if time.monotonic() > deadline:
            print(f"time budget reached after {index} dates", flush=True)
            break
        stamp = date.strftime("%Y%m%d")
        if stamp in done:
            continue
        try:
            raw = ak.stock_tfp_em(date=stamp)
        except Exception as exc:  # noqa: BLE001 - throttling is expected and recorded
            failed += 1
            if failed % 20 == 0:
                print(f"  {stamp}: {type(exc).__name__} (failures={failed})", flush=True)
            time.sleep(2.0)
            continue
        if raw is None or not len(raw):
            pd.DataFrame(columns=["code", "snapshot_date"]).to_parquet(staging / f"{stamp}.parquet")
            empty += 1
            continue
        frame = pd.DataFrame({
            "code": raw["代码"].astype(str).str.zfill(6),
            "name": raw.get("名称", pd.Series([""] * len(raw))).astype(str),
            "suspend_start": pd.to_datetime(raw.get("停牌时间"), errors="coerce"),
            "suspend_end": pd.to_datetime(raw.get("停牌截止时间"), errors="coerce"),
            "suspension_reason": raw.get("停牌原因", pd.Series([""] * len(raw))).astype(str),
            "snapshot_date": date,
        })
        frame.to_parquet(staging / f"{stamp}.parquet", index=False)
        fetched += 1
        if fetched % 25 == 0:
            print(f"  {fetched} snapshots · {stamp} · {time.monotonic() - started:.0f}s", flush=True)

    # fold every snapshot into (symbol, start, end) intervals
    files = sorted(staging.glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in files]
    frames = [f for f in frames if len(f) and "suspend_start" in f.columns]
    intervals = pd.DataFrame()
    if frames:
        snapshots = pd.concat(frames, ignore_index=True)
        symbols, keep = [], []
        for code in snapshots["code"]:
            try:
                symbols.append(identify(code).symbol)
                keep.append(True)
            except SymbolError:
                keep.append(False)
        snapshots = snapshots[keep].copy()
        snapshots["symbol"] = symbols
        snapshots["effective_start"] = snapshots["suspend_start"].fillna(snapshots["snapshot_date"])
        grouped = snapshots.groupby(["symbol", "effective_start"], as_index=False).agg(
            effective_end=("suspend_end", "max"),
            last_seen=("snapshot_date", "max"),
            suspension_reason=("suspension_reason", "first"))
        # a halt with no published resume date is closed at the last date it was
        # still observed, never left open-ended into the future
        grouped["effective_end"] = grouped["effective_end"].fillna(grouped["last_seen"])
        intervals = pd.DataFrame({
            "symbol": grouped["symbol"],
            "effective_start": grouped["effective_start"],
            "effective_end": grouped["effective_end"],
            "suspension_reason": grouped["suspension_reason"],
            "evidence": "eastmoney_suspension_snapshot",
            "source": "akshare.stock_tfp_em",
            "source_endpoint": "eastmoney RPT_CUSTOM_SUSPEND_DATA_INTERFACE",
            "retrieved_at": _now(),
            "available_at": grouped["effective_start"].dt.strftime("%Y-%m-%d"),
            "quality_status": "OK",
        }).sort_values(["symbol", "effective_start"])
        intervals.to_parquet(OUT / "suspension_intervals.parquet", index=False)

    covered = sorted({p.stem for p in files})
    payload = {
        "dataset": "suspension_intervals",
        "snapshot_dates_on_disk": len(covered),
        "snapshot_date_range": [covered[0], covered[-1]] if covered else None,
        "requested_range": [args.start, str(today.date())],
        "trading_dates_in_range": int(len(dates)),
        "intervals": int(len(intervals)),
        "symbols_with_halts": int(intervals["symbol"].nunique()) if len(intervals) else 0,
        "fetched_this_run": fetched, "empty_snapshots": empty, "failed_requests": failed,
        "source": "akshare.stock_tfp_em (Eastmoney suspension report)",
        "coverage_note": ("Only trading dates whose snapshot is on disk contribute intervals; "
                          "dates outside snapshot_date_range are NOT claimed as halt-free."),
        "generated": _now(),
    }
    (OUT / "suspension_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
#: Risk-warning prefixes as they appear in exchange security short names.
#: Order matters — the longest marker must be tested first.
ST_PREFIXES: tuple[tuple[str, str], ...] = (
    ("S*ST", "s_star_st"),      # pre-2007: unreformed share structure + delisting risk
    ("*ST", "star_st"),         # delisting risk warning
    ("SST", "s_st"),            # pre-2007: unreformed + other risk
    ("ST", "st"),               # other risk warning
)


def _st_kind(name: str) -> str | None:
    """Return the risk-warning kind encoded in a security short name, or None."""
    cleaned = str(name).upper().replace(" ", "").replace("　", "")
    for marker, kind in ST_PREFIXES:
        if cleaned.startswith(marker):
            return kind
    return None


def _szse_name_history() -> tuple[pd.DataFrame, dict]:
    """Dated SZSE security short-name changes (the exchange's own register)."""
    import akshare as ak

    error = None
    for attempt in range(3):
        try:
            raw = ak.stock_info_sz_change_name(symbol="简称变更")
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
            raw = None
            time.sleep((3, 8, 15)[attempt])
    provenance = {"function": "akshare.stock_info_sz_change_name(简称变更)",
                  "endpoint": "https://www.szse.cn/api/report/ShowReport",
                  "rows": 0 if raw is None else int(len(raw)), "error": error}
    if raw is None or not len(raw):
        return pd.DataFrame(), provenance
    frame = pd.DataFrame({
        "code": raw["证券代码"].astype(str).str.zfill(6),
        "change_date": pd.to_datetime(raw["变更日期"], errors="coerce"),
        "name_before": raw["变更前简称"].astype(str),
        "name_after": raw["变更后简称"].astype(str),
    }).dropna(subset=["change_date"])
    return frame.sort_values(["code", "change_date"]), provenance


def _st_intervals_from_name_history(history: pd.DataFrame, master: pd.DataFrame,
                                    now: str) -> pd.DataFrame:
    """Fold a dated name history into closed ST episodes.

    A security is in a risk-warning state from the date its short name gains an
    ST marker until the date the name loses it. The episode that is still open
    at the end of the register stays open (``effective_end`` NaT), which is the
    honest representation of "still ST as far as the register shows".
    """
    code_to_symbol = dict(zip(master["code"].astype(str), master["symbol"].astype(str)))
    records: list[dict] = []
    for code, group in history.groupby("code"):
        symbol = code_to_symbol.get(code)
        if symbol is None:
            continue                      # a code the current master does not carry
        group = group.sort_values("change_date")
        # timeline: the name in force before the first change, then each change
        timeline = [(pd.NaT, group.iloc[0]["name_before"])]
        timeline += [(row.change_date, row.name_after) for row in group.itertuples()]
        open_start: pd.Timestamp | None = None
        open_name = open_kind = None
        for change_date, name in timeline:
            kind = _st_kind(name)
            if kind and open_start is None:
                if pd.isna(change_date):
                    continue              # ST before the register starts: undated, skipped
                open_start, open_name, open_kind = change_date, name, kind
            elif not kind and open_start is not None:
                records.append({"symbol": symbol, "effective_start": open_start,
                                "effective_end": change_date, "security_name": open_name,
                                "st_flag": True, "st_kind": open_kind})
                open_start = open_name = open_kind = None
        if open_start is not None:
            records.append({"symbol": symbol, "effective_start": open_start,
                            "effective_end": pd.NaT, "security_name": open_name,
                            "st_flag": True, "st_kind": open_kind})
    return pd.DataFrame(records, columns=["symbol", "effective_start", "effective_end",
                                          "security_name", "st_flag", "st_kind"])


def cmd_st(args: argparse.Namespace) -> int:
    """Risk-warning (ST / *ST) state: dated episodes where a register exists."""
    master = _load_master()
    master["code"] = master["code"].astype(str).str.zfill(6)
    now = _now()
    today = pd.Timestamp.now().normalize()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- current snapshot (authoritative for today, from the instrument name) --
    current = master[master["current_st"].fillna(False).astype(bool)].copy()
    snapshot = pd.DataFrame({
        "symbol": current["symbol"],
        "observed_on": today,
        "security_name": current["name"],
        "st_kind": current["name"].astype(str).map(lambda n: _st_kind(n) or "st"),
        "source": "tickflow.exchanges.get_instruments (instrument name)",
        "retrieved_at": now, "available_at": today.strftime("%Y-%m-%d"), "quality_status": "OK",
    })
    snapshot.to_parquet(OUT / "st_current_snapshot.parquet", index=False)

    # --- dated episodes from the SZSE short-name register ---------------------
    history, provenance = (_szse_name_history() if args.allow_network
                           else (pd.DataFrame(), {"skipped": True}))
    intervals = pd.DataFrame()
    if len(history):
        intervals = _st_intervals_from_name_history(history, master, now)
        if len(intervals):
            intervals["source"] = "akshare.stock_info_sz_change_name (SZSE short-name register)"
            intervals["source_endpoint"] = provenance.get("endpoint", "")
            intervals["retrieved_at"] = now
            intervals["available_at"] = intervals["effective_start"].dt.strftime("%Y-%m-%d")
            intervals["quality_status"] = "OK"
            intervals.to_parquet(OUT / "st_intervals.parquet", index=False)

    covered_exchanges = sorted(
        master.loc[master["symbol"].isin(set(intervals["symbol"])), "exchange"].unique()
    ) if len(intervals) else []
    uncovered = sorted(set(master["exchange"].unique()) - set(covered_exchanges))
    payload = {
        "dataset": "st_intervals",
        "current_st_names": int(len(snapshot)),
        "dated_episodes": int(len(intervals)),
        "securities_with_dated_episodes": int(intervals["symbol"].nunique()) if len(intervals) else 0,
        "name_change_records": provenance.get("rows", 0),
        "episode_date_range": [str(intervals["effective_start"].min().date()),
                               str(intervals["effective_start"].max().date())] if len(intervals) else None,
        "exchanges_with_dated_history": covered_exchanges,
        "exchanges_without_dated_history": uncovered,
        "historical_intervals_status": "AVAILABLE" if len(intervals) else BLOCKED,
        "coverage_caveat": ("The SZSE publishes a dated short-name register, so Shenzhen main-board "
                            "and ChiNext episodes are exchange-sourced. No equivalent bulk register "
                            "was found for the SSE or the BSE, so risk-warning history for those "
                            f"exchanges ({', '.join(uncovered) or 'none'}) remains {BLOCKED}."),
        "sources_tested_for_history": [
            {"source": "akshare.stock_info_sz_change_name (简称变更)",
             "result": f"WORKS — {provenance.get('rows', 0)} dated SZSE short-name changes"},
            {"source": "query.sse.com.cn common-query name-change reports",
             "result": "no bulk dated short-name register found for the SSE"},
            {"source": "akshare.stock_info_change_name (Sina former names)",
             "result": "ordered former names WITHOUT change dates; cannot form intervals"},
            {"source": "akshare.stock_profile_cninfo (曾用简称)",
             "result": "former names without dates; empty for delisted names"},
            {"source": "akshare.stock_zh_a_st_em",
             "result": "Eastmoney current risk-warning board only, and IP-throttled here"},
            {"source": "TickFlow instrument listing",
             "result": "current name only — no name history in the entitled API"},
            {"source": "baostock query_history_k_data_plus (isST per trading day)",
             "result": "would cover every exchange, but baostock needs TCP 10030 and this "
                       "runtime only has 80/443 egress"},
        ],
        "why_not_inferred": ("A +/-5% limit signature would let ST days be guessed from bars, but a "
                             "guessed regulatory state is not a point-in-time fact and is never "
                             "written into a PIT table."),
        "provenance": provenance,
        "generated": now,
    }
    (OUT / "st_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                    default=str))
    print(json.dumps({k: payload[k] for k in
                      ("current_st_names", "dated_episodes", "securities_with_dated_episodes",
                       "name_change_records", "episode_date_range", "exchanges_with_dated_history",
                       "exchanges_without_dated_history", "historical_intervals_status")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["calendar", "factors", "dividends", "suspension", "st"])
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to run: --allow-network was not confirmed")
        return 2
    load_repo_env()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"calendar": cmd_calendar, "factors": cmd_factors, "dividends": cmd_dividends,
            "suspension": cmd_suspension, "st": cmd_st}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
