"""ENG_ST31 — forward-anchored risk-warning (ST) register for the SSE and the BSE.

ENG_ST30 tried to rebuild ten years of SSE/BSE risk-warning history from cninfo
announcements alone and failed its pre-registered HOLDOUT. ENG_ST31 only needs
the frozen fresh window, so it starts from a dated full-universe instrument-name
snapshot (TickFlow ``exchanges.get_instruments``; 2026-07-26 19:39 CST) and
carries the state forward:

  snapshot       fetch today's full-universe instrument names (a new dated anchor;
                 never overwrites the security master);
  company-names  per-company cninfo announcement lists over the window for the
                 symbols the register has to reason about — the short name cninfo
                 prints on each disclosure is the name in force on that date;
  build          fold anchors + ENG_ST30-classified events + dated names into
                 TRUE / FALSE / UNKNOWN spans per symbol (``st_forward``);
  evaluate       SZSE run of the identical pipeline against the SZSE register
                 (``--split dev`` = historical windows, even codes;
                 ``--split validation`` = 2026-07-27..2026-11-24, once);
  publish        apply the decision rule and write the strict PIT evidence.

Method, thresholds and decision rule are pre-registered in
``docs/research/preregistrations/ENG_ST31_preregistration.md``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CST = dt.timezone(dt.timedelta(hours=8))
EXCHANGES = ("SH", "SZ", "BJ")
TICKFLOW_PACE_S = 6.5           # measured hard limit: 10 requests / minute


def _u0_root() -> Path:
    from quantagent.config.paths import quant_paths

    return quant_paths().data_root / "u0"


# Resolved at import, BEFORE any credential file is loaded: the repository .env
# may set QUANTAGENT_DATA_ROOT, which must not redirect where this script writes.
U0 = _u0_root()
CACHE = U0 / "pit" / "_st_forward"


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Full-universe instrument names now, with the retrieval timestamp.

    The snapshot is an observation of the short name in force at
    ``retrieved_at``; outside trading hours it describes the previous session
    or (if the vendor already rolled over) the next one — ``st_forward`` treats
    it that way. Nothing is overwritten: each fetch is a new file.
    """
    import pandas as pd

    from quantagent.data.ashare.env import load_repo_env
    from quantagent.data.ashare.sources import build_tickflow_client

    if args.env_file:
        load_repo_env(Path(args.env_file))
    client = build_tickflow_client()
    frames, provenance = [], {}
    started = dt.datetime.now(dt.timezone.utc)
    for index, exchange in enumerate(EXCHANGES):
        if index:
            time.sleep(TICKFLOW_PACE_S)
        try:
            raw = pd.DataFrame(client.exchanges.get_instruments(exchange))
        except Exception as exc:  # noqa: BLE001 - recorded; an incomplete snapshot is refused
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
            "vendor_limit_up": pd.to_numeric(ext.apply(lambda d: d.get("limit_up")), errors="coerce"),
            "vendor_limit_down": pd.to_numeric(ext.apply(lambda d: d.get("limit_down")), errors="coerce"),
            "exchange_code": exchange,
        }))
    finished = dt.datetime.now(dt.timezone.utc)
    if any(p["error"] or not p["rows"] for p in provenance.values()):
        raise SystemExit(f"incomplete instrument snapshot, nothing written: {provenance}")
    frame = pd.concat(frames, ignore_index=True)
    frame["source"] = "tickflow.exchanges.get_instruments"
    frame["retrieved_at"] = finished.isoformat(timespec="seconds")
    out = CACHE / "instrument_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"instruments_{finished.astimezone(CST):%Y%m%dT%H%M%S}.parquet"
    frame.to_parquet(target, index=False)
    print(json.dumps({"written": str(target), "rows": int(len(frame)), "by_exchange": provenance,
                      "retrieval_started": started.isoformat(timespec="seconds"),
                      "retrieved_at": finished.isoformat(timespec="seconds")}, ensure_ascii=False))
    return 0



# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

ST30_CACHE = U0 / "pit" / "_st_announcements"
PREREG = "docs/research/preregistrations/ENG_ST31_preregistration.md"
A1_FILE = "instruments_20261124T231441.parquet"            # ENG_ST31 §2.1
V_WINDOW = ("2026-07-27", "2026-11-24")                     # ENG_ST31 §5
V_CUTOFF = "2026-11-24"          # information cut-off of the validation run (A1 retrieval day)
DEV_WINDOWS = (("2025-07-28", "2025-11-24"), ("2024-07-29", "2024-11-22"),
               ("2025-03-03", "2025-07-25"))                # ENG_ST31 §4
EVENT_LOOKBACK_DAYS = 45
NAME_LOOKBACK_DAYS = 10
SUFFIX_OF = {"SSE": ".SH", "SZSE": ".SZ", "BSE": ".BJ"}
EXCHANGE_OF = {".SH": "SSE", ".SZ": "SZSE", ".BJ": "BSE"}
QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STOCK_LISTS = ("szse_stock.json", "bj_stock.json")
METHOD_FILES = ("src/quantagent/data/ashare/st_forward.py", "scripts/u0_st_forward.py",
                "src/quantagent/data/ashare/st_announcements.py", "scripts/u0_st_announcements.py")


def _st30():
    import u0_st_announcements as st30

    return st30


def _fwd():
    from quantagent.data.ashare import st_forward

    return st_forward


def _calendar():
    import pandas as pd

    from quantagent.data.ashare.st_announcements import Calendar

    cal = pd.read_parquet(U0 / "pit/trading_calendar.parquet")
    return Calendar.from_dates(cal.loc[cal["is_open"].astype(bool), "trade_date"])


def _today_cst() -> "dt.date":
    return dt.datetime.now(CST).date()


def _default_as_of():
    """A session's state is settled by its 09:25 pre-open; before that only the
    previous day is vouched for."""
    import pandas as pd

    now = dt.datetime.now(CST)
    day = pd.Timestamp(now.date())
    return day - pd.Timedelta(days=1) if (now.hour, now.minute) < (9, 25) else day


def _master():
    import pandas as pd

    return pd.read_parquet(U0 / "security_master.parquet")


def _a0_time():
    import pandas as pd

    stamps = _master()["source_timestamp"].dropna().unique()
    if len(stamps) != 1:
        raise SystemExit(f"security master has {len(stamps)} source timestamps; A0 is ambiguous")
    return pd.Timestamp(stamps[0])


def _snapshot_files(cutoff=None) -> list[Path]:
    files = sorted((CACHE / "instrument_snapshots").glob("instruments_*.parquet"))
    if cutoff is not None:
        files = [f for f in files if f.name <= f"instruments_{cutoff:%Y%m%d}T235959.parquet"]
    return files


def _snapshots(files) -> "pd.DataFrame":
    import pandas as pd

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame = frame[frame["instrument_type"].eq("stock")]
        frames.append(frame.assign(snapshot_file=path.name))
    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "listing_date", "retrieved_at", "snapshot_file"])
    return pd.concat(frames, ignore_index=True)


def _universe(exchanges, snapshots, calendar) -> "pd.DataFrame":
    """A0's listed A-shares (state from the A0 name) + listings after A0 seen in
    a later snapshot (not ST at listing). ENG_ST31 §3.1."""
    import pandas as pd

    fwd = _fwd()
    master = _master()
    a0 = master[master["origin"].eq("tickflow_live") & master["status"].eq("listed")
                & master["security_type"].eq("A_share") & master["exchange"].isin(exchanges)]
    a0_sessions = fwd.snapshot_sessions(_a0_time(), calendar)
    base = pd.DataFrame({"symbol": a0["symbol"].astype(str), "anchor_session": a0_sessions[0],
                         "start_session": a0_sessions[-1], "start_name": a0["name"].astype(str),
                         "origin": "A0"})
    suffixes = tuple(SUFFIX_OF[e] for e in exchanges)
    known = set(master["symbol"].astype(str))
    later = snapshots[snapshots["symbol"].str[-3:].isin(suffixes) & ~snapshots["symbol"].isin(known)]
    later = later[pd.to_datetime(later["listing_date"]) > _a0_time().tz_convert(None).normalize()]
    later = later.sort_values("retrieved_at").groupby("symbol").first().reset_index()
    rows = []
    for row in later.itertuples(index=False):
        session = calendar.on_or_after(pd.Timestamp(row.listing_date))
        if session is None:
            continue
        rows.append({"symbol": row.symbol, "anchor_session": session, "start_session": session,
                     "start_name": "", "origin": f"listed_after_A0:{row.snapshot_file}"})
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def _events(cutoff=None):
    """ENG_ST30-classified, dated cninfo events + every crawled announcement as a
    dated short name (withdrawn filings excluded)."""
    import pandas as pd

    st30 = _st30()
    ann, events = st30.build_events(ST30_CACHE, fallback_sessions=1)
    if cutoff is not None:
        ann = ann[ann["announcement_date"] <= cutoff]
        events = events[events["announcement_date"] <= cutoff]
    live = ann[~ann["title"].astype(str).str.contains("已取消")]
    names = pd.DataFrame({"symbol": live["symbol"], "kind": "name",
                          "observed_at": live["announcement_date"], "sec_name": live["sec_name"],
                          "source": "cninfo:" + live["source_query"].astype(str),
                          "ref": live["announcementId"].astype(str)})
    return events, names


def _snapshot_obs(snapshots, after):
    import pandas as pd

    later = snapshots[pd.to_datetime(snapshots["retrieved_at"]) > after]
    return pd.DataFrame({"symbol": later["symbol"], "kind": "snapshot",
                         "observed_at": pd.to_datetime(later["retrieved_at"]),
                         "sec_name": later["name"], "source": "tickflow.get_instruments",
                         "ref": later["snapshot_file"]})


def _active(universe, events, snapshot_obs, *, since):
    """ENG_ST31 §2.3: ST-marked in any anchor, any non-IGNORE event announced
    since ``since``, or ST-marking that differs between anchors."""
    import pandas as pd

    fwd = _fwd()
    marked0 = universe.set_index("symbol")["start_name"].map(fwd.is_marked)
    snap = snapshot_obs.assign(marked=snapshot_obs["sec_name"].map(fwd.is_marked))
    snap = snap[snap["symbol"].isin(universe["symbol"])]
    marked_any = set(marked0[marked0].index) | set(snap.loc[snap["marked"], "symbol"])
    per = snap.groupby("symbol")["marked"].agg(lambda m: set(m))
    differ = {s for s, values in per.items()
              if len(values | {bool(marked0.get(s, False))}) > 1}
    pre = fwd.prefilter_events(events)
    recent = pre[(pre["announcement_date"] >= since) & pre["event"].ne("IGNORE")]
    with_events = set(recent["symbol"]) & set(universe["symbol"])
    return sorted(marked_any | differ | with_events)


# ---------------------------------------------------------------------------
# Per-company dated short names (cninfo, all categories)
# ---------------------------------------------------------------------------

def _org_ids() -> dict:
    import requests

    st30 = _st30()
    out_dir = CACHE / "cninfo_stock_lists"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for name in STOCK_LISTS:
        path = out_dir / name
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < 7 * 86400
        if not fresh:
            resp = requests.get(f"http://www.cninfo.com.cn/new/data/{name}",
                                headers={"User-Agent": st30.HEADERS["User-Agent"]}, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        for row in json.loads(path.read_text(encoding="utf-8")).get("stockList", []):
            mapping.setdefault(str(row["code"]).zfill(6), row["orgId"])
    return mapping


def _fetch_company(symbol: str, org_id: str, lo, hi, sleep: float) -> list[dict]:
    st30 = _st30()
    payload = {"pageNum": "1", "pageSize": str(st30.PAGE_SIZE), "column": "szse",
               "tabName": "fulltext", "plate": "", "searchkey": "", "secid": "",
               "stock": f"{symbol[:6]},{org_id}", "category": "", "trade": "",
               "seDate": f"{lo:%Y-%m-%d}~{hi:%Y-%m-%d}", "sortName": "", "sortType": "",
               "isHLtitle": "true"}
    body = st30._post(payload, sleep)
    total = int(body.get("totalAnnouncement") or 0)
    rows = list(body.get("announcements") or [])
    for page in range(2, max(1, -(-total // st30.PAGE_SIZE)) + 1):
        payload["pageNum"] = str(page)
        rows.extend(st30._post(payload, sleep).get("announcements") or [])
    keep = [{k: r.get(k) for k in ("announcementId", "announcementTime", "secName", "secCode",
                                   "announcementTitle")} for r in rows]
    if len({r["announcementId"] for r in keep}) < total:
        raise RuntimeError(f"{symbol}: {len(keep)} of {total} announcements returned")
    return keep


def _company_targets(mode: str, calendar):
    """[(tag, symbol, lo, hi)] the name fetch must cover for ``mode``."""
    import pandas as pd

    events, _ = _events()
    if mode == "production":
        snaps = _snapshots(_snapshot_files())
        universe = _universe(("SSE", "SZSE", "BSE"), snaps, calendar)
        a0 = _a0_time().tz_convert(None).normalize()
        active = _active(universe, events, _snapshot_obs(snaps, _a0_time()),
                         since=a0 - pd.Timedelta(days=EVENT_LOOKBACK_DAYS))
        lo = a0 - pd.Timedelta(days=NAME_LOOKBACK_DAYS)
        return [("prod", s, lo, pd.Timestamp(_today_cst())) for s in active]
    targets = []
    for start, end in DEV_WINDOWS:
        universe, snap_obs = _dev_inputs(start, end, calendar)
        active = _active(universe, events, snap_obs,
                         since=pd.Timestamp(start) - pd.Timedelta(days=EVENT_LOOKBACK_DAYS))
        lo = pd.Timestamp(start) - pd.Timedelta(days=NAME_LOOKBACK_DAYS)
        hi = pd.Timestamp(end) + pd.Timedelta(days=3)
        targets += [(f"dev_{start}", s, lo, hi) for s in active]
    return targets


def cmd_company_names(args: argparse.Namespace) -> int:
    import hashlib

    import pandas as pd

    calendar = _calendar()
    orgs = _org_ids()
    targets = _company_targets(args.mode, calendar)
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        targets = [t for t in targets
                   if int(hashlib.md5(t[1].encode()).hexdigest(), 16) % count == index]
    fetched = skipped = failed = 0
    for tag, symbol, lo, hi in targets:
        out = CACHE / "company_names" / tag
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{symbol}.json"
        old = json.loads(target.read_text()) if target.exists() else None
        fetch_lo = lo
        if old is not None:
            old_hi = pd.Timestamp(old["window"][1])
            settled = pd.Timestamp(old["retrieved_at"]).tz_convert("Asia/Shanghai").tz_localize(None) \
                > old_hi + pd.Timedelta(days=1)
            if old_hi >= hi and settled:
                skipped += 1
                continue
            fetch_lo = max(lo, old_hi - pd.Timedelta(days=3))
        org_id = orgs.get(symbol[:6])
        if not org_id:
            failed += 1
            print(f"no cninfo orgId for {symbol}", flush=True)
            continue
        try:
            rows = _fetch_company(symbol, org_id, fetch_lo, hi, args.sleep)
        except Exception as exc:  # noqa: BLE001 - recorded; the symbol keeps its older cache
            failed += 1
            print(f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        if old is not None:
            seen = {r["announcementId"] for r in rows}
            rows += [r for r in old["rows"] if r["announcementId"] not in seen]
        target.write_text(json.dumps({
            "symbol": symbol, "org_id": org_id, "window": [str(lo.date()), str(hi.date())],
            "rows": rows, "endpoint": QUERY_URL, "category": "all",
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat()}, ensure_ascii=False))
        fetched += 1
        if fetched % 50 == 0:
            print(f"company-names fetched={fetched} skipped={skipped} failed={failed} of {len(targets)}",
                  flush=True)
    print(json.dumps({"mode": args.mode, "targets": len(targets), "fetched": fetched,
                      "skipped": skipped, "failed": failed}), flush=True)
    return 0 if not failed else 3


def _company_name_obs(tag: str, cutoff=None):
    import pandas as pd

    frames = []
    for path in sorted((CACHE / "company_names" / tag).glob("*.json")):
        record = json.loads(path.read_text())
        frame = pd.DataFrame(record.get("rows") or [])
        if frame.empty:
            continue
        frame = frame[~frame["announcementTitle"].astype(str).str.contains("已取消")]
        observed = (pd.to_datetime(frame["announcementTime"], unit="ms", utc=True)
                    .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize())
        frames.append(pd.DataFrame({
            "symbol": record["symbol"], "kind": "name", "observed_at": observed,
            "sec_name": frame["secName"].astype(str).str.replace(r"</?em>", "", regex=True),
            "source": "cninfo:company", "ref": frame["announcementId"].astype(str)}))
    if not frames:
        return pd.DataFrame(columns=["symbol", "kind", "observed_at", "sec_name", "source", "ref"])
    out = pd.concat(frames, ignore_index=True)
    if cutoff is not None:
        out = out[out["observed_at"] <= cutoff]
    return out


# ---------------------------------------------------------------------------
# Dev windows (SZSE even codes; anchors simulated from the register)
# ---------------------------------------------------------------------------

def _szse_names_in_force(day) -> dict:
    """{symbol: short name in force on ``day``} from the SZSE name register."""
    import pandas as pd

    path = sorted(ST30_CACHE.glob("szse_name_history_*.parquet"))[-1]
    hist = pd.read_parquet(path)
    hist["change_date"] = pd.to_datetime(hist["change_date"])
    day = pd.Timestamp(day)
    master = _master()
    names = {str(s): str(n) for s, n in zip(master["symbol"], master["name"]) if str(s).endswith(".SZ")}
    for code, group in hist.sort_values("change_date").groupby("code"):
        symbol = f"{str(code).zfill(6)}.SZ"
        before = group[group["change_date"] <= day]
        names[symbol] = (str(before.iloc[-1]["name_after"]) if len(before)
                         else str(group.iloc[0]["name_before"]))
    return names


def _dev_inputs(start: str, end: str, calendar):
    """DEV-half universe with a register-simulated start anchor (the evening
    before ``start``) and end anchor (the evening of ``end``)."""
    import pandas as pd

    fwd = _fwd()
    master = _master()
    sz = master[master["exchange"].eq("SZSE") & master["security_type"].eq("A_share")]
    sz = sz[sz["code"].astype(str).str[5].astype(int) % 2 == 0]
    first = pd.Timestamp(start)
    listed = pd.to_datetime(sz["listing_date"]) < first
    alive = sz["delisting_date"].isna() | (pd.to_datetime(sz["delisting_date"]) > first)
    sz = sz[listed & alive]
    pos = calendar.sessions.searchsorted(first, side="left") - 1
    prev = calendar.sessions[pos]
    start_t = pd.Timestamp(f"{prev:%Y-%m-%d} 20:00").tz_localize("Asia/Shanghai")
    end_t = pd.Timestamp(f"{end} 23:00").tz_localize("Asia/Shanghai")
    names0 = _szse_names_in_force(prev)
    names1 = _szse_names_in_force(pd.Timestamp(end))
    sessions = fwd.snapshot_sessions(start_t, calendar)
    universe = pd.DataFrame({"symbol": sz["symbol"].astype(str), "anchor_session": sessions[0],
                             "start_session": sessions[-1],
                             "start_name": sz["symbol"].map(names0).fillna(""), "origin": "dev_anchor"})
    snap_obs = pd.DataFrame({"symbol": universe["symbol"], "kind": "snapshot", "observed_at": end_t,
                             "sec_name": universe["symbol"].map(names1).fillna(""),
                             "source": "register_simulated_end_anchor", "ref": f"dev:{end}"})
    return universe, snap_obs


# ---------------------------------------------------------------------------
# Build / evaluate / publish
# ---------------------------------------------------------------------------

def _forward(universe, events, observations, calendar, as_of, *, emit_from_anchor=False):
    fwd = _fwd()
    if emit_from_anchor:
        universe = universe.assign(start_session=universe["anchor_session"])
    return fwd.run_forward(universe=universe, events=events, observations=observations,
                           calendar=calendar, as_of=as_of)


def _production(calendar, as_of, *, exchanges=("SSE", "SZSE", "BSE"), cutoff=None,
                snapshot_files=None, emit_from_anchor=False):
    import pandas as pd

    snaps = _snapshots(snapshot_files if snapshot_files is not None else _snapshot_files())
    universe = _universe(exchanges, snaps, calendar)
    events, ann_names = _events(cutoff)
    obs = pd.concat([_snapshot_obs(snaps, _a0_time()), ann_names, _company_name_obs("prod", cutoff)],
                    ignore_index=True)
    obs = obs[obs["symbol"].isin(universe["symbol"])]
    rows, log = _forward(universe, events, obs, calendar, as_of, emit_from_anchor=emit_from_anchor)
    return universe, rows, log


def cmd_build(args: argparse.Namespace) -> int:
    import pandas as pd

    calendar = _calendar()
    as_of = pd.Timestamp(args.as_of) if args.as_of else _default_as_of()
    exchanges = tuple(args.exchanges.split(","))
    if "SZSE" in exchanges and not (CACHE / "validation_result.json").exists():
        raise SystemExit("ENG_ST31 §4: no SZSE state may be computed on the validation window "
                         "before the validation run; use --exchanges SSE,BSE")
    universe, rows, log = _production(calendar, as_of, exchanges=exchanges)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "_".join(e.lower() for e in exchanges)
    rows.to_parquet(CACHE / f"build_rows_{tag}.parquet", index=False)
    log.to_parquet(CACHE / f"build_restarts_{tag}.parquet", index=False)
    rows["exchange"] = rows["symbol"].str[-3:].map(EXCHANGE_OF)
    summary = {
        "as_of": str(as_of.date()), "symbols": int(rows["symbol"].nunique()),
        "states_by_exchange": rows.groupby(["exchange", "state"]).size().unstack(fill_value=0)
        .to_dict(orient="index"),
        "restarts_by_exchange": (log["symbol"].str[-3:].map(EXCHANGE_OF).value_counts().to_dict()
                                 if len(log) else {}),
        "unknown_symbols_by_exchange": rows[rows["state"].eq("UNKNOWN")].groupby("exchange")["symbol"]
        .nunique().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _register_truth(register, symbols, calendar, lo, hi):
    import pandas as pd

    from quantagent.data.ashare.st_announcements import merge_contiguous

    fwd = _fwd()
    sessions = calendar.sessions[(calendar.sessions >= pd.Timestamp(lo))
                                 & (calendar.sessions <= pd.Timestamp(hi))]
    reg = register[register["symbol"].isin(set(symbols))]
    return fwd.register_states(merge_contiguous(reg), symbols, sessions)


def _git(*args) -> str:
    import subprocess

    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def cmd_evaluate(args: argparse.Namespace) -> int:
    import hashlib

    import pandas as pd

    fwd = _fwd()
    calendar = _calendar()
    record: dict = {"split": args.split, "preregistration": PREREG,
                    "method_sha256": {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
                                      for f in METHOD_FILES},
                    "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.split == "dev":
        truth_all = pd.read_parquet(U0 / "pit/st_intervals.parquet")
        events, ann_names = _events()
        record["windows"] = {}
        chosen = [w for w in DEV_WINDOWS if not args.windows or w[0] in args.windows.split(",")]
        for start, end in chosen:
            universe, snap_obs = _dev_inputs(start, end, calendar)
            obs = pd.concat([snap_obs, ann_names, _company_name_obs(f"dev_{start}")], ignore_index=True)
            is_name = obs["kind"].eq("name")
            name_day = pd.to_datetime(obs["observed_at"].where(is_name))
            obs = obs[obs["symbol"].isin(universe["symbol"])
                      & (~is_name | (name_day <= pd.Timestamp(end)))]
            ev = events[events["announcement_date"] <= pd.Timestamp(end)]
            rows, log = _forward(universe, ev, obs, calendar, pd.Timestamp(end), emit_from_anchor=True)
            truth = _register_truth(truth_all, universe["symbol"], calendar,
                                    universe["anchor_session"].min(), end)
            metrics = fwd.evaluate_forward(rows, truth, calendar, window=(start, end))
            metrics["restarts"] = int(len(log))
            metrics["restart_examples"] = log.head(10).astype(str).to_dict("records") if len(log) else []
            record["windows"][start] = {"window": [start, end], "metrics": metrics,
                                        "acceptance": fwd.acceptance(metrics)}
            print(start, json.dumps({k: v for k, v in metrics.items()
                                     if k not in ("wrong_rows_by_symbol", "restart_examples")},
                                    default=str), flush=True)
        # vendor-anchor check (DEV half, 2026-07-24): A0 names vs the register
        master = _master()
        a0 = master[master["origin"].eq("tickflow_live") & master["exchange"].eq("SZSE")
                    & master["security_type"].eq("A_share")]
        a0 = a0[a0["code"].astype(str).str[5].astype(int) % 2 == 0]
        truth = _register_truth(truth_all, a0["symbol"], calendar, "2026-07-24", "2026-07-24")
        vendor = a0.set_index("symbol")["name"].map(fwd.is_marked)
        truth["vendor"] = truth["symbol"].map(vendor)
        disagree = truth[truth["vendor"] != truth["truth"]]
        record["vendor_anchor_dev_20260724"] = {
            "symbols": int(len(truth)), "register_true": int(truth["truth"].sum()),
            "vendor_true": int(truth["vendor"].sum()), "disagree": int(len(disagree)),
            "disagree_symbols": disagree["symbol"].tolist()[:30]}
        print(json.dumps(record["vendor_anchor_dev_20260724"], ensure_ascii=False), flush=True)
        target = CACHE / f"dev_eval_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    else:
        target = CACHE / "validation_result.json"
        if target.exists():
            raise SystemExit(f"validation was already evaluated once ({target}); ENG_ST31 §5 "
                             "forbids a second evaluation.")
        dirty = _git("status", "--porcelain", "--", *METHOD_FILES, PREREG)
        if dirty:
            raise SystemExit("method code is not committed; freeze it before validation:\n" + dirty)
        record["method_commit"] = _git("rev-parse", "HEAD")
        sys.path.insert(0, str(REPO / "scripts"))
        import u0_pit_intervals as pit

        master = _master()
        master["code"] = master["code"].astype(str).str.zfill(6)
        history, provenance = pit._szse_name_history()
        if not len(history):
            raise SystemExit(f"SZSE register fetch failed: {provenance}")
        register = pit._st_intervals_from_name_history(history, master,
                                                       dt.datetime.now(dt.timezone.utc).isoformat())
        stamp = dt.datetime.now(CST)
        history.to_parquet(CACHE / f"validation_szse_name_history_{stamp:%Y%m%dT%H%M%S}.parquet",
                           index=False)
        record["register"] = {"retrieved_at": stamp.isoformat(), "name_changes": int(len(history)),
                              "latest_change": str(history["change_date"].max().date()),
                              "endpoint": provenance.get("endpoint", "")}
        cutoff = pd.Timestamp(V_CUTOFF)
        snap_files = [CACHE / "instrument_snapshots" / A1_FILE]
        universe, rows, log = _production(calendar, pd.Timestamp(V_WINDOW[1]), exchanges=("SZSE",),
                                          cutoff=cutoff, snapshot_files=snap_files,
                                          emit_from_anchor=True)
        truth = _register_truth(register, universe["symbol"], calendar,
                                universe["anchor_session"].min(), V_WINDOW[1])
        # a symbol listed after A0 has truth rows only from its listing session
        first = universe.set_index("symbol")["anchor_session"]
        truth = truth[truth["trade_date"] >= truth["symbol"].map(first)]
        metrics = fwd.evaluate_forward(rows, truth, calendar, window=V_WINDOW)
        metrics["restarts"] = int(len(log))
        metrics["restart_log"] = log.astype(str).to_dict("records") if len(log) else []
        record["metrics"] = metrics
        record["acceptance"] = fwd.acceptance(metrics)
        rows.to_parquet(CACHE / "validation_rows.parquet", index=False)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    if args.split == "validation":
        print(json.dumps({k: v for k, v in record["metrics"].items() if k != "restart_log"},
                         ensure_ascii=False, indent=2, default=str))
        print(json.dumps(record["acceptance"], indent=2, default=str))
    print(f"written {target}")
    return 0


MONITORED = ("V1", "V2", "V4", "V5")                       # ENG_ST31 §7


def _seed_coverage(evidence_path: Path, as_of) -> dict:
    """Would the fresh-window catch-up accept the table? (reported, not gated)"""
    import pandas as pd

    from quantagent.data.providers.st_pit import (
        HistoricalSTCoverageError,
        load_historical_st_evidence,
        require_symbol_coverage,
    )

    panel_path = U0.parent / "v7/silver/market_panel/market_panel.parquet"
    if not panel_path.exists():
        return {"status": "no_panel"}
    panel = pd.read_parquet(panel_path, columns=["symbol", "trade_date"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    pmax = panel["trade_date"].max()
    seed = sorted(panel.loc[panel["trade_date"] == pmax, "symbol"].astype(str).unique())
    evidence = load_historical_st_evidence(evidence_path)
    start = pmax + pd.Timedelta(days=1)
    report = {"panel_max": str(pmax.date()), "seed_symbols": len(seed),
              "window": [str(start.date()), str(pd.Timestamp(as_of).date())]}
    uncovered = []
    for symbol in seed:
        try:
            require_symbol_coverage(evidence, [symbol], start, as_of)
        except HistoricalSTCoverageError:
            uncovered.append(symbol)
    report["uncovered"] = len(uncovered)
    report["uncovered_by_suffix"] = pd.Series([u[-3:] for u in uncovered]).value_counts().to_dict()
    report["uncovered_examples"] = uncovered[:20]
    report["catchup_would_accept"] = not uncovered
    return report


def cmd_publish(args: argparse.Namespace) -> int:
    """ENG_ST31 §6–§7: SZSE from the register (ENG_ST30 publish, unchanged);
    SSE/BSE forward rows only if validation passed and today's SZSE monitor holds."""
    import pandas as pd

    from quantagent.data.ashare.st_announcements import STATE_TRUE

    st30 = _st30()
    fwd = _fwd()
    calendar = _calendar()
    st30.cmd_publish(argparse.Namespace(cache=str(ST30_CACHE), szse_register=None, as_of=None))
    out = U0 / "pit"
    manifest_path = out / "st_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text())
    evidence = pd.read_parquet(out / "st_pit_evidence.parquet")
    if evidence["symbol"].str[-3:].isin([".SH", ".BJ"]).any():
        raise SystemExit("ENG_ST30 publish wrote SSE/BSE rows; ENG_ST31 refuses to mix methods")
    as_of = pd.Timestamp(args.as_of) if args.as_of else _default_as_of()
    section: dict = {"preregistration": PREREG, "as_of": str(as_of.date()),
                     "generated": dt.datetime.now(dt.timezone.utc).isoformat()}
    vpath = CACHE / "validation_result.json"
    status = "NOT_VALIDATED"
    if vpath.exists():
        validation = json.loads(vpath.read_text())
        section["validation"] = {"acceptance": validation["acceptance"],
                                 "method_commit": validation.get("method_commit")}
        status = "VALIDATION_FAILED" if not validation["acceptance"]["all_pass"] else "VALIDATED"
    if status == "VALIDATED":
        universe, rows, log = _production(calendar, as_of)
        rows.to_parquet(CACHE / "forward_rows.parquet", index=False)
        log.to_parquet(CACHE / "forward_restarts.parquet", index=False)
        register = pd.read_parquet(manifest["szse_register"])
        reg_as_of = pd.Timestamp(manifest["szse_register_as_of"])
        mon_end = min(as_of, reg_as_of)
        sz = universe[universe["symbol"].str.endswith(".SZ")]
        truth = _register_truth(register, sz["symbol"], calendar, V_WINDOW[0], mon_end)
        truth = truth[truth["trade_date"] >= truth["symbol"].map(sz.set_index("symbol")["start_session"])]
        metrics = fwd.evaluate_forward(rows[rows["symbol"].str.endswith(".SZ")], truth, calendar,
                                       window=(V_WINDOW[0], mon_end))
        acc = fwd.acceptance(metrics)
        monitor = {k: acc[k] for k in MONITORED}
        section["monitor"] = {"window": [V_WINDOW[0], str(mon_end.date())], "checks": monitor,
                              "rows": metrics["rows"], "false_true_rows": metrics["false_true_rows"],
                              "missed_true_rows": metrics["missed_true_rows"],
                              "unknown_rows": metrics["unknown_rows"]}
        if all(v["pass"] for v in monitor.values()):
            status = "PUBLISHED"
            keep = rows[rows["symbol"].str[-3:].isin([".SH", ".BJ"])]
            intervals = fwd.rows_to_intervals(keep, calendar, as_of=as_of,
                                              source="ENG_ST31 forward register (TickFlow names + cninfo)")
            evidence = pd.concat([evidence, intervals], ignore_index=True)
            evidence.to_parquet(out / "st_pit_evidence.parquet", index=False)
            for exch, sfx in (("SSE", ".SH"), ("BSE", ".BJ")):
                manifest["accepted_exchanges"][exch] = True
                part = keep[keep["symbol"].str.endswith(sfx)]
                section[exch] = {"symbols": int(part["symbol"].nunique()),
                                 "sessions_by_state": part["state"].value_counts().to_dict(),
                                 "symbols_with_unknown": int(part.loc[part["state"].eq("UNKNOWN"),
                                                                      "symbol"].nunique()),
                                 "first_session": str(part["trade_date"].min().date()),
                                 "st_symbols_on_as_of": int(part[(part["trade_date"] == part["trade_date"].max())
                                                                 & part["state"].eq(STATE_TRUE)]
                                                            ["symbol"].nunique())}
            manifest["exchanges_blocked"] = [e for e, ok in manifest["accepted_exchanges"].items() if not ok]
            manifest["pit_evidence"]["rows"] = int(len(evidence))
            manifest["pit_evidence"]["symbols"] = int(evidence["symbol"].nunique())
            manifest["pit_evidence"]["coverage_start_sse_bse"] = V_WINDOW[0]
        else:
            status = "WITHHELD_BY_MONITOR"
    section["status"] = status
    section["seed_coverage"] = _seed_coverage(out / "st_pit_evidence.parquet", as_of)
    manifest["eng_st31"] = section
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(section, ensure_ascii=False, indent=2, default=str))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--env-file", default=None,
                      help="credential file to load (names only; values are never printed)")
    snap.set_defaults(func=cmd_snapshot)
    cn = sub.add_parser("company-names")
    cn.add_argument("--mode", choices=("production", "dev"), required=True)
    cn.add_argument("--sleep", type=float, default=1.0)
    cn.add_argument("--shard", default="", help="i/n: this worker's share of the symbols")
    cn.set_defaults(func=cmd_company_names)
    bd = sub.add_parser("build")
    bd.add_argument("--as-of", default=None)
    bd.add_argument("--exchanges", default="SSE,BSE")
    bd.set_defaults(func=cmd_build)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", choices=("dev", "validation"), required=True)
    ev.add_argument("--windows", default="", help="dev only: comma-separated window starts")
    ev.set_defaults(func=cmd_evaluate)
    pb = sub.add_parser("publish")
    pb.add_argument("--as-of", default=None)
    pb.set_defaults(func=cmd_publish)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
