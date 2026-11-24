"""Risk-warning (ST) episodes from cninfo 特别处理和退市 announcements.

The SZSE publishes a dated short-name register; the SSE and the BSE do not.
Every listed company on all three exchanges must, however, disclose the
implementation and the revocation of a risk warning on cninfo (巨潮资讯网, the
CSRC-designated disclosure platform), and the body of that announcement states
the session from which the warning applies. This script

  crawl     pages the cninfo announcement search for category 特别处理和退市
            (``category_tbclts_szsh``) per plate and month, caching raw JSON;
  pdf       fetches the PDFs of the announcements the title classifier marks as
            candidate state changes and caches their extracted text;
  build     folds the events into dated episodes (see
            ``quantagent.data.ashare.st_announcements``);

Every response is cached under ``--cache`` and the crawl is resumable. It is
polite by construction: one request at a time with a sleep between requests.

Method and acceptance thresholds are pre-registered in
``docs/research/preregistrations/ENG_ST30_preregistration.md``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_URL = "http://static.cninfo.com.cn/"
CATEGORY = "category_tbclts_szsh"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-crawler",
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
}
PAGE_SIZE = 30            # cninfo ignores larger values


def _months(start: str, end: str):
    cur = dt.date.fromisoformat(start).replace(day=1)
    stop = dt.date.fromisoformat(end)
    while cur <= stop:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield cur, min(nxt - dt.timedelta(days=1), stop)
        cur = nxt


def _post(payload: dict, sleep: float, retries: int = 4) -> dict:
    err = None
    for attempt in range(retries):
        time.sleep(sleep * (1 + 2 * attempt))
        try:
            resp = requests.post(QUERY_URL, data=payload, headers=HEADERS, timeout=40)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
    raise RuntimeError(f"cninfo query failed after {retries} attempts: {err}")


def _window_settled(done: Path) -> bool:
    """A cached window is final only if it was fetched after the window closed.

    A month fetched while still open (the current month on a daily refresh) is
    fetched again, so an announcement published later in that month is seen.
    """
    try:
        record = json.loads(done.read_text())
        closed = dt.date.fromisoformat(record["window"][1])
        fetched = dt.datetime.fromisoformat(record["retrieved_at"]).astimezone(
            dt.timezone(dt.timedelta(hours=8))).date()
    except Exception:  # noqa: BLE001 - an unreadable marker is not trusted
        return False
    return fetched > closed


def _crawl_windows(cache: Path, *, tag: str, plate: str, first: str, end: str,
                   extra: dict, sleep: float) -> int:
    """Page every monthly window of one (plate, query) and cache the raw JSON."""
    new_windows = 0
    for lo, hi in _months(first, end):
        window_tag = f"{tag}_{lo:%Y%m}"
        done = cache / f"{window_tag}.done.json"
        if done.exists() and _window_settled(done):
            continue
        payload = {"pageNum": "1", "pageSize": str(PAGE_SIZE), "column": "szse",
                   "tabName": "fulltext", "plate": plate, "stock": "", "searchkey": "",
                   "secid": "", "category": "", "trade": "",
                   "seDate": f"{lo:%Y-%m-%d}~{hi:%Y-%m-%d}", "sortName": "",
                   "sortType": "", "isHLtitle": "true"}
        payload.update(extra)
        first_page = _post(payload, sleep)
        total = int(first_page.get("totalAnnouncement") or 0)
        pages = max(1, -(-total // PAGE_SIZE))
        rows = list(first_page.get("announcements") or [])
        (cache / f"{window_tag}_p1.json").write_text(json.dumps(first_page, ensure_ascii=False))
        for page in range(2, pages + 1):
            payload["pageNum"] = str(page)
            body = _post(payload, sleep)
            (cache / f"{window_tag}_p{page}.json").write_text(json.dumps(body, ensure_ascii=False))
            rows.extend(body.get("announcements") or [])
        ids = {r.get("announcementId") for r in rows}
        record = {"plate": plate, "query": extra, "window": [str(lo), str(hi)],
                  "total": total, "pages": pages, "rows": len(rows), "unique_ids": len(ids),
                  "complete": len(ids) >= total,
                  "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "endpoint": QUERY_URL}
        done.write_text(json.dumps(record, ensure_ascii=False))
        new_windows += 1
        if total:
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return new_windows


PLATES = (("sh", None), ("sz", None), ("bj", "bj_start"))


def cmd_crawl(args: argparse.Namespace) -> int:
    """cninfo category 特别处理和退市 (pre-registered source, ENG_ST30 §1)."""
    cache = Path(args.cache) / "query"
    cache.mkdir(parents=True, exist_ok=True)
    total = 0
    for plate, start_attr in PLATES:
        first = getattr(args, start_attr) if start_attr else args.start
        total += _crawl_windows(cache, tag=plate, plate=plate, first=first, end=args.end,
                                extra={"category": CATEGORY}, sleep=args.sleep)
    print(f"crawl complete: {total} new windows", flush=True)
    return 0


#: Title keywords for the second query (ENG_ST30 amendment A1): BSE risk-warning
#: notices are not filed under category 特别处理和退市, and neither are some
#: SSE/SZSE ones. cninfo ``searchkey`` matches announcement titles only.
TITLE_KEYWORDS = ("风险警示", "特别处理", "退市整理")


def cmd_keyword_crawl(args: argparse.Namespace) -> int:
    cache = Path(args.cache) / "query_kw"
    cache.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, keyword in enumerate(TITLE_KEYWORDS):
        for plate, start_attr in PLATES:
            first = getattr(args, start_attr) if start_attr else args.start
            total += _crawl_windows(cache, tag=f"{plate}_kw{index}", plate=plate, first=first,
                                    end=args.end, extra={"searchkey": keyword},
                                    sleep=args.sleep)
    print(f"keyword crawl complete: {total} new windows", flush=True)
    return 0


def load_announcements(cache_dir: str | Path):
    """All cached announcements, de-duplicated by announcementId, with plate."""
    import pandas as pd

    records = {}
    for sub in ("query", "query_kw"):
        for path in sorted((Path(cache_dir) / sub).glob("*_p*.json")):
            plate = path.name.split("_", 1)[0]
            body = json.loads(path.read_text())
            for row in body.get("announcements") or []:
                row = dict(row)
                key = row["announcementId"]
                if key in records:
                    records[key]["source_query"] += f"|{sub}"
                    continue
                row["plate"] = plate
                row["raw_file"] = f"{sub}/{path.name}"
                row["source_query"] = sub
                records[key] = row
    frame = pd.DataFrame(list(records.values()))
    if frame.empty:
        return frame
    frame["announcement_date"] = (
        pd.to_datetime(frame["announcementTime"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    )
    # keyword queries return titles/names with <em> highlight markup
    for raw, clean in (("announcementTitle", "title"), ("secName", "sec_name")):
        frame[clean] = frame[raw].astype(str).str.replace(r"</?em>", "", regex=True)
    return frame


def cmd_pdf(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quantagent.data.ashare import st_announcements as sa

    frame = load_announcements(args.cache)
    frame["event"] = frame["title"].map(sa.classify_title)
    wanted = frame[frame["event"].isin(sa.STATE_EVENTS)]
    if args.plates:
        wanted = wanted[wanted["plate"].isin(args.plates.split(","))]
    out = Path(args.cache) / "pdf_text"
    out.mkdir(parents=True, exist_ok=True)
    import io

    import pypdf

    fetched = failed = 0
    for row in wanted.itertuples():
        target = out / f"{row.announcementId}.json"
        if target.exists():
            continue
        url = STATIC_URL + str(row.adjunctUrl)
        record = {"announcementId": row.announcementId, "url": url}
        try:
            time.sleep(args.sleep)
            resp = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=60)
            resp.raise_for_status()
            record["sha256_16"] = hashlib.sha256(resp.content).hexdigest()[:16]
            record["bytes"] = len(resp.content)
            if str(row.adjunctType).upper() == "PDF" or resp.content[:4] == b"%PDF":
                reader = pypdf.PdfReader(io.BytesIO(resp.content))
                record["text"] = "\n".join((page.extract_text() or "")
                                           for page in reader.pages[:6])
            else:
                record["text"] = resp.content.decode("utf-8", "replace")
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            failed += 1
        record["retrieved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        target.write_text(json.dumps(record, ensure_ascii=False))
        if (fetched + failed) % 100 == 0:
            print(f"pdf fetched={fetched} failed={failed}", flush=True)
    print(f"pdf complete fetched={fetched} failed={failed} wanted={len(wanted)}", flush=True)
    return 0


SSE_BULLETIN_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
SSE_KEYWORDS = ("风险警示", "特别处理", "退市整理")


def _sse_get(params: dict, sleep: float, retries: int = 4) -> dict:
    err = None
    for attempt in range(retries):
        time.sleep(sleep * (1 + 2 * attempt))
        try:
            resp = requests.get(SSE_BULLETIN_URL, params=params, timeout=40,
                                headers={"User-Agent": HEADERS["User-Agent"],
                                         "Referer": "https://www.sse.com.cn/"})
            resp.raise_for_status()
            text = resp.text
            return json.loads(text[text.index("(") + 1:text.rindex(")")])
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
    raise RuntimeError(f"SSE bulletin query failed after {retries} attempts: {err}")


def cmd_sse_crawl(args: argparse.Namespace) -> int:
    """The SSE's own company-bulletin search, title keywords, monthly windows.

    An independent second source for SSE events (exchange website rather than
    cninfo). The endpoint returns nothing for windows longer than ~2 months, so
    the crawl is monthly.
    """
    cache = Path(args.cache) / "sse_bulletin"
    cache.mkdir(parents=True, exist_ok=True)
    for keyword in SSE_KEYWORDS:
        for lo, hi in _months(args.start, args.end):
            tag = f"{hashlib.md5(keyword.encode()).hexdigest()[:6]}_{lo:%Y%m}"
            done = cache / f"{tag}.done.json"
            if done.exists():
                continue
            rows, page, total = [], 1, None
            while True:
                params = {"jsonCallBack": "cb", "isPagination": "true",
                          "pageHelp.pageSize": "100", "pageHelp.pageNo": str(page),
                          "pageHelp.beginPage": str(page), "pageHelp.cacheSize": "1",
                          "pageHelp.endPage": str(page), "START_DATE": f"{lo:%Y-%m-%d}",
                          "END_DATE": f"{hi:%Y-%m-%d}", "SECURITY_CODE": "", "TITLE": keyword,
                          "BULLETIN_TYPE": "", "stockType": ""}
                body = _sse_get(params, args.sleep)
                (cache / f"{tag}_p{page}.json").write_text(json.dumps(body, ensure_ascii=False))
                help_ = body.get("pageHelp") or {}
                total = int(help_.get("total") or 0)
                data = help_.get("data") or []
                rows.extend(data)
                if len(rows) >= total or not data:
                    break
                page += 1
            done.write_text(json.dumps({"keyword": keyword, "window": [str(lo), str(hi)],
                                        "total": total, "rows": len(rows),
                                        "endpoint": SSE_BULLETIN_URL,
                                        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                                       ensure_ascii=False))
            if total:
                print(json.dumps({"keyword": keyword, "month": f"{lo:%Y-%m}", "total": total,
                                  "rows": len(rows)}, ensure_ascii=False), flush=True)
    print("sse crawl complete", flush=True)
    return 0


def load_sse_bulletins(cache_dir: str | Path):
    import pandas as pd

    records = {}
    for path in sorted((Path(cache_dir) / "sse_bulletin").glob("*_p*.json")):
        body = json.loads(path.read_text())
        for item in (body.get("pageHelp") or {}).get("data") or []:
            row = item[0] if isinstance(item, list) and item else item
            if not isinstance(row, dict):
                continue
            key = row.get("ORG_BULLETIN_ID") or (row.get("SECURITY_CODE"), row.get("SSEDATE"),
                                                 row.get("TITLE"))
            records[key] = dict(row, raw_file=path.name)
    frame = pd.DataFrame(list(records.values()))
    if not frame.empty:
        frame["announcement_date"] = pd.to_datetime(frame["SSEDATE"], errors="coerce")
    return frame


# ---------------------------------------------------------------------------
# build / evaluate / checks / refresh-szse / publish
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
SUFFIX = {"sh": ".SH", "sz": ".SZ", "bj": ".BJ"}
CRAWL_START = "2001-01-01"
WINDOW = ("2016-01-04", "2026-07-24")            # ENG_ST30 §3
C_WINDOW = ("2016-01-04", "2026-07-03")          # ENG_ST30 §5
SNAPSHOT_DAY = "2026-07-24"                      # ENG_ST30 §5 C3
PIT_COVERAGE_START = "2016-01-04"


def _sa():
    sys.path.insert(0, str(REPO / "src"))
    from quantagent.data.ashare import st_announcements as sa
    return sa


def _calendar():
    import pandas as pd

    sa = _sa()
    cal = pd.read_parquet(U0 / "pit/trading_calendar.parquet")
    return sa.Calendar.from_dates(cal.loc[cal["is_open"].astype(bool), "trade_date"])


def _pdf_texts(cache: Path) -> dict:
    texts = {}
    for path in (cache / "pdf_text").glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a torn cache file is refetched by `pdf`
            continue
        if record.get("text"):
            texts[str(record["announcementId"])] = record["text"]
    return texts


def _symbols(ann):
    """secCode + exchange suffix; BSE companies keyed by cninfo orgId so a
    pre-migration 8xxxxx/43xxxx code resolves to the company's 920xxx code."""
    sym = ann["secCode"].astype(str).str.zfill(6) + ann["plate"].map(SUFFIX)
    bj = ann["plate"].eq("bj")
    if bj.any():
        latest = (ann[bj & ann["secCode"].astype(str).str.startswith("92")]
                  .sort_values("announcement_date").groupby("orgId")["secCode"].last())
        mapped = ann.loc[bj, "orgId"].map(latest)
        sym.loc[bj] = mapped.fillna(ann.loc[bj, "secCode"]).astype(str).str.zfill(6) + ".BJ"
    return sym


def build_events(cache: Path, *, fallback_sessions: int | None):
    sa = _sa()
    ann = load_announcements(cache)
    ann["symbol"] = _symbols(ann)
    ann["event"] = ann["title"].map(sa.classify_title)
    ann["announcement_id"] = ann["announcementId"].astype(str)
    texts = _pdf_texts(cache)
    events = ann[ann["event"].isin(sa.STATE_EVENTS)].copy()
    events["text"] = events["announcement_id"].map(texts)
    events = sa.resolve_effective_dates(events, _calendar(), fallback_sessions=fallback_sessions)
    return ann, events


def cmd_build(args: argparse.Namespace) -> int:
    import pandas as pd

    sa = _sa()
    cache = Path(args.cache)
    fallback = None if args.fallback_sessions < 0 else args.fallback_sessions
    ann, events = build_events(cache, fallback_sessions=fallback)
    episodes = sa.build_episodes(events, coverage_start=pd.Timestamp(CRAWL_START))
    episodes["available_at"] = episodes["start_announcement_date"]
    episodes["end_available_at"] = episodes["end_announcement_date"]
    episodes["source"] = "cninfo announcements (ENG_ST30)"
    episodes["evidence_id"] = episodes["start_event_id"]
    keep = ["announcement_id", "symbol", "plate", "orgId", "announcement_date", "title",
            "sec_name", "event", "adjunctUrl", "source_query", "raw_file"]
    ann[keep].to_parquet(cache / "announcements.parquet", index=False)
    events.drop(columns=["text"]).to_parquet(cache / "events.parquet", index=False)
    episodes.to_parquet(cache / "episodes.parquet", index=False)
    summary = {
        "announcements": int(len(ann)), "state_events": int(len(events)),
        "events_by_kind": events["event"].value_counts().to_dict(),
        "effective_basis": events["effective_basis"].str.split(":").str[0].value_counts().to_dict(),
        "events_with_pdf_text": int(events["announcement_id"].isin(_pdf_texts(cache)).sum()),
        "episodes_by_state": episodes["st_state"].value_counts().to_dict() if len(episodes) else {},
        "fallback_sessions": fallback,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _panel_rows(suffixes: tuple[str, ...], columns=("symbol", "trade_date")):
    import pandas as pd
    import pyarrow.parquet as pq

    table = pq.read_table(U0 / "panel/daily_bars_raw.parquet", columns=list(columns))
    frame = table.to_pandas()
    frame = frame[frame["symbol"].str[-3:].isin(suffixes)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame


def _szse_split(split: str) -> list[str]:
    import pandas as pd

    master = pd.read_parquet(U0 / "security_master.parquet")
    sz = master[master["exchange"].eq("SZSE")]["symbol"].astype(str)
    parity = sz.str[5].astype(int) % 2                    # last digit of the 6-digit code
    return sorted(sz[parity.eq(0 if split == "dev" else 1)])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_evaluate(args: argparse.Namespace) -> int:
    import subprocess

    import pandas as pd

    sa = _sa()
    cache = Path(args.cache)
    method_files = [REPO / "src/quantagent/data/ashare/st_announcements.py", Path(__file__).resolve()]
    record: dict = {"split": args.split, "window": WINDOW,
                    "method_sha256": {p.name: _sha(p) for p in method_files},
                    "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.split == "holdout":
        target = cache / "holdout_result.json"
        if target.exists():
            raise SystemExit(f"HOLDOUT was already evaluated once ({target}); ENG_ST30 §3 "
                             "forbids a second evaluation.")
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--"]
                               + [str(p) for p in method_files],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise SystemExit("method code is not committed; freeze it before HOLDOUT:\n" + dirty)
        record["method_commit"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    else:
        target = cache / f"dev_eval_{dt.datetime.now():%Y%m%dT%H%M%S}.json"
    episodes = pd.read_parquet(cache / "episodes.parquet")
    truth = pd.read_parquet(U0 / "pit/st_intervals.parquet")
    rows = _panel_rows((".SZ",))
    symbols = _szse_split(args.split)
    metrics = sa.evaluate_reconstruction(
        episodes, truth, rows, _calendar(),
        window=(pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])), symbols=symbols)
    record["metrics"] = metrics
    record["acceptance"] = sa.acceptance(metrics)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "examples"}, indent=2,
                     default=str))
    print(json.dumps({k: v for k, v in record["acceptance"].items()}, indent=2, default=str))
    print(f"written {target}")
    return 0


def _round_half_up(values):
    import numpy as np

    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0


def _band_frame(suffix: str, board: str, states_for):
    """Valid rows (ENG_ST30 §5) with 5% band flags and a TRUE/FALSE/UNKNOWN state."""
    import numpy as np
    import pandas as pd

    master = pd.read_parquet(U0 / "security_master.parquet")
    symbols = set(master.loc[master["board"].eq(board), "symbol"].astype(str))
    bars = _panel_rows((suffix,), columns=("symbol", "trade_date", "high", "low", "close", "volume"))
    bars = bars[bars["symbol"].isin(symbols)]
    lo, hi = pd.Timestamp(C_WINDOW[0]), pd.Timestamp(C_WINDOW[1])
    bars = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    cal = _calendar().sessions
    pos = pd.Series(np.arange(len(cal)), index=cal)
    bars["session_pos"] = bars["trade_date"].map(pos)
    g = bars.groupby("symbol", sort=False)
    prev_close = g["close"].shift(1)
    prev_pos = g["session_pos"].shift(1)
    prev_vol = g["volume"].shift(1)
    ca = pd.read_parquet(U0 / "pit/corporate_actions.parquet", columns=["symbol", "ex_date"])
    ex = set(zip(ca["symbol"].astype(str), pd.to_datetime(ca["ex_date"]).dt.normalize()))
    is_ex = np.fromiter(((s, d) in ex for s, d in zip(bars["symbol"], bars["trade_date"])),
                        dtype=bool, count=len(bars))
    valid = ((bars["volume"] > 0) & (prev_vol > 0) & (bars["session_pos"] - prev_pos == 1)
             & ~is_ex & bars["trade_date"].between(lo, hi))
    bars = bars[valid.to_numpy()].copy()
    prev = prev_close[valid.to_numpy()].to_numpy()
    up, dn = _round_half_up(prev * 1.05), _round_half_up(prev * 0.95)
    bars["outside"] = (bars["high"].to_numpy() > up + 1e-6) | (bars["low"].to_numpy() < dn - 1e-6)
    close = bars["close"].to_numpy()
    bars["at_band"] = (np.abs(close - up) < 1e-6) | (np.abs(close - dn) < 1e-6)
    bars["state"] = states_for(bars)
    return bars


def _c_metrics(bars) -> dict:
    true = bars["state"].eq("TRUE")
    hard = float(bars.loc[true, "outside"].mean()) if true.any() else None
    month = bars.assign(month=bars["trade_date"].dt.to_period("M"))
    agg = month.groupby(["symbol", "month"]).agg(n=("outside", "size"), outside=("outside", "sum"),
                                                 at_band=("at_band", "sum"),
                                                 true=("state", lambda s: (s == "TRUE").mean()))
    st_like = agg[(agg["n"] >= 10) & (agg["at_band"] >= 2) & (agg["outside"] == 0)]
    unexplained = st_like[st_like["true"] < 0.5]
    return {"true_rows": int(true.sum()), "hard_contradiction_share": hard,
            "st_like_months": int(len(st_like)), "unexplained_months": int(len(unexplained)),
            "unexplained_share": (len(unexplained) / len(st_like)) if len(st_like) else None,
            "unexplained_examples": [f"{s}@{m}" for s, m in unexplained.index[:30]]}


def cmd_checks(args: argparse.Namespace) -> int:
    import pandas as pd

    sa = _sa()
    cache = Path(args.cache)
    episodes = pd.read_parquet(cache / "episodes.parquet")
    truth = pd.read_parquet(U0 / "pit/st_intervals.parquet")
    truth_eps = sa.merge_contiguous(truth).assign(st_state=sa.STATE_TRUE)

    sse = _band_frame(".SH", "SH_Main", lambda b: sa.row_states(b, episodes))
    szse = _band_frame(".SZ", "SZ_Main", lambda b: sa.row_states(b, truth_eps))
    c_sse, c_sz = _c_metrics(sse), _c_metrics(szse)
    c1_bound = max(2 * (c_sz["hard_contradiction_share"] or 0.0), 0.005)
    c2_bound = max(2 * (c_sz["unexplained_share"] or 0.0), 0.05)
    c1 = c_sse["hard_contradiction_share"] is not None and c_sse["hard_contradiction_share"] <= c1_bound
    c2 = c_sse["unexplained_share"] is not None and c_sse["unexplained_share"] <= c2_bound

    snap = pd.read_parquet(U0 / "pit/st_current_snapshot.parquet")
    day = pd.Timestamp(SNAPSHOT_DAY)
    rows = _panel_rows((".SH", ".BJ"))
    rows = rows[rows["trade_date"].eq(day)].copy()
    rows["state"] = sa.row_states(rows, episodes)
    c3 = {}
    for suffix in (".SH", ".BJ"):
        part = rows[rows["symbol"].str.endswith(suffix)]
        snap_syms = set(snap.loc[snap["symbol"].str.endswith(suffix), "symbol"])
        in_snap = part[part["symbol"].isin(snap_syms)]
        rec_true = part[part["state"].eq("TRUE")]
        c3[suffix] = {
            "snapshot_names_with_row": int(len(in_snap)),
            "snapshot_true_share": float(in_snap["state"].eq("TRUE").mean()) if len(in_snap) else None,
            "snapshot_not_true": sorted(in_snap.loc[~in_snap["state"].eq("TRUE"), "symbol"])[:40],
            "reconstructed_true": int(len(rec_true)),
            "true_not_in_snapshot_share": (float((~rec_true["symbol"].isin(snap_syms)).mean())
                                           if len(rec_true) else None),
            "true_not_in_snapshot": sorted(set(rec_true["symbol"]) - snap_syms)[:40],
        }
    sh3 = c3[".SH"]
    c3_sse = (sh3["snapshot_true_share"] is not None and sh3["snapshot_true_share"] >= 0.95
              and (sh3["true_not_in_snapshot_share"] or 0.0) <= 0.05)
    bj3 = c3[".BJ"]
    c3_bse = (bj3["snapshot_names_with_row"] > 0 and bj3["snapshot_true_share"] == 1.0
              and not bj3["true_not_in_snapshot"])

    ann = pd.read_parquet(cache / "announcements.parquet")
    d = {}
    for kind in (sa.IMPLEMENT, sa.REVOKE):
        part = ann[ann["event"].eq(kind) & ann["plate"].eq("sh")]
        has_st = part["sec_name"].str.upper().str.contains(r"ST", regex=False)
        d[kind] = {"n": int(len(part)), "sec_name_has_ST": float(has_st.mean()) if len(part) else None}

    record = {
        "C1": {"sse": c_sse["hard_contradiction_share"], "szse_register": c_sz["hard_contradiction_share"],
               "bound": c1_bound, "pass": bool(c1)},
        "C2": {"sse": c_sse["unexplained_share"], "szse_register": c_sz["unexplained_share"],
               "bound": c2_bound, "pass": bool(c2),
               "sse_detail": c_sse, "szse_detail": c_sz},
        "C3": {"detail": c3, "sse_pass": bool(c3_sse), "bse_pass": bool(c3_bse)},
        "D_informational": d,
        "bse_announcements": int(ann["plate"].eq("bj").sum()),
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    target = cache / "checks_result.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({"C1": record["C1"], "C2": {k: v for k, v in record["C2"].items()
                                                 if not k.endswith("detail")},
                      "C3": {"sse_pass": c3_sse, "bse_pass": c3_bse,
                             "SH": {k: v for k, v in sh3.items() if not isinstance(v, list)},
                             "BJ": {k: v for k, v in bj3.items() if not isinstance(v, list)}},
                      "D": d}, ensure_ascii=False, indent=2, default=str))
    print(f"written {target}")
    return 0


def cmd_refresh_szse(args: argparse.Namespace) -> int:
    """Fetch the SZSE dated short-name register now (forward coverage).

    The certified ``pit/st_intervals.parquet`` (retrieved 2026-07-26) is never
    overwritten; the fresh register is written next to the crawl cache.
    """
    import pandas as pd

    sys.path.insert(0, str(REPO / "scripts"))
    import u0_pit_intervals as pit

    master = pd.read_parquet(U0 / "security_master.parquet")
    master["code"] = master["code"].astype(str).str.zfill(6)
    history, provenance = pit._szse_name_history()
    if not len(history):
        raise SystemExit(f"SZSE register fetch failed: {provenance}")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    intervals = pit._st_intervals_from_name_history(history, master, now)
    intervals["source"] = "akshare.stock_info_sz_change_name (SZSE short-name register)"
    intervals["source_endpoint"] = provenance.get("endpoint", "")
    intervals["retrieved_at"] = now
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
    target = Path(args.cache) / f"szse_register_{stamp:%Y%m%d}.parquet"
    intervals.to_parquet(target, index=False)
    history.to_parquet(Path(args.cache) / f"szse_name_history_{stamp:%Y%m%d}.parquet", index=False)
    print(json.dumps({"episodes": int(len(intervals)), "name_changes": int(len(history)),
                      "latest_change": str(history["change_date"].max().date()),
                      "written": str(target)}, ensure_ascii=False))
    return 0


def _latest(cache: Path, pattern: str):
    found = sorted(cache.glob(pattern))
    return found[-1] if found else None


def cmd_publish(args: argparse.Namespace) -> int:
    """ENG_ST30 §7 decision rule → st_intervals_v2 + the strict PIT evidence table."""
    import pandas as pd

    sa = _sa()
    cache = Path(args.cache)
    holdout = cache / "holdout_result.json"
    checks = cache / "checks_result.json"
    if not holdout.exists() or not checks.exists():
        raise SystemExit("publish needs holdout_result.json and checks_result.json")
    hold = json.loads(holdout.read_text())
    chk = json.loads(checks.read_text())
    holdout_pass = bool(hold["acceptance"]["all_pass"])
    sse_ok = holdout_pass and chk["C1"]["pass"] and chk["C2"]["pass"] and chk["C3"]["sse_pass"]
    bse_ok = holdout_pass and chk["C3"]["bse_pass"] and chk["bse_announcements"] > 0
    accepted = {"SZSE": True, "SSE": bool(sse_ok), "BSE": bool(bse_ok)}

    szse_path = Path(args.szse_register) if args.szse_register else _latest(cache, "szse_register_*.parquet")
    if szse_path is None:
        raise SystemExit("no SZSE register; run refresh-szse first")
    szse = pd.read_parquet(szse_path)
    szse_as_of = pd.to_datetime(szse["retrieved_at"].iloc[0]).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    szse_eps = szse.assign(st_state=sa.STATE_TRUE,
                           available_at=pd.to_datetime(szse["effective_start"]),
                           end_available_at=pd.to_datetime(szse["effective_end"]),
                           evidence_id=szse["security_name"])
    szse_eps = szse_eps[["symbol", "effective_start", "effective_end", "st_state", "available_at",
                         "end_available_at", "source", "evidence_id"]]

    episodes = pd.read_parquet(cache / "episodes.parquet")
    keep_suffix = [sfx for exch, sfx in (("SSE", ".SH"), ("BSE", ".BJ")) if accepted[exch]]
    recon = episodes[episodes["symbol"].str[-3:].isin(keep_suffix)]
    recon = recon[["symbol", "effective_start", "effective_end", "st_state", "available_at",
                   "end_available_at", "source", "evidence_id", "start_basis", "end_basis",
                   "start_title", "end_title", "note"]]
    v2 = pd.concat([szse_eps.assign(method="exchange_register"),
                    recon.assign(method="cninfo_announcements_ENG_ST30")], ignore_index=True)

    crawl_as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(
        dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date())
    as_of = min(crawl_as_of, szse_as_of) if accepted["SSE"] or accepted["BSE"] else szse_as_of
    master = pd.read_parquet(U0 / "security_master.parquet")
    master = master[master["security_type"].eq("A_share")]
    exchanges = [e for e, ok in accepted.items() if ok]
    life = master[master["exchange"].isin(exchanges)]
    life = pd.DataFrame({"symbol": life["symbol"].astype(str),
                         "first_day": pd.to_datetime(life["listing_date"]),
                         "last_day": pd.to_datetime(life["delisting_date"]) - pd.Timedelta(days=1)})
    pit_eps = v2[v2["st_state"].isin([sa.STATE_TRUE, sa.STATE_UNKNOWN])]
    evidence = sa.pit_intervals(pit_eps, life, as_of=as_of,
                                coverage_start=pd.Timestamp(PIT_COVERAGE_START))
    out = U0 / "pit"
    v2.to_parquet(out / "st_intervals_v2.parquet", index=False)
    evidence.to_parquet(out / "st_pit_evidence.parquet", index=False)
    unknown = v2[v2["st_state"].eq(sa.STATE_UNKNOWN)]
    manifest = {
        "dataset": "st_intervals_v2", "preregistration": "docs/research/preregistrations/ENG_ST30_preregistration.md",
        "accepted_exchanges": accepted, "holdout_acceptance": hold["acceptance"],
        "holdout_method_commit": hold.get("method_commit"),
        "checks": {"C1": chk["C1"], "C2": {k: v for k, v in chk["C2"].items() if not k.endswith("detail")},
                   "C3": {"sse_pass": chk["C3"]["sse_pass"], "bse_pass": chk["C3"]["bse_pass"]}},
        "as_of": str(as_of.date()), "szse_register": str(szse_path), "szse_register_as_of": str(szse_as_of.date()),
        "pit_evidence": {"path": "runtime/data/u0/pit/st_pit_evidence.parquet",
                         "coverage_start": PIT_COVERAGE_START, "rows": int(len(evidence)),
                         "symbols": int(evidence["symbol"].nunique()),
                         "contract": "quantagent.data.providers.st_pit (interval mode, end inclusive)"},
        "unknown_spans": int(len(unknown)),
        "unknown_spans_after_coverage_start": int((unknown["effective_end"].fillna(pd.Timestamp.max)
                                                   > pd.Timestamp(PIT_COVERAGE_START)).sum()),
        "exchanges_blocked": [e for e, ok in accepted.items() if not ok],
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out / "st_manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("accepted_exchanges", "as_of", "pit_evidence", "unknown_spans",
                                              "unknown_spans_after_coverage_start", "exchanges_blocked")},
                     ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sse = sub.add_parser("sse-crawl")
    sse.add_argument("--cache", required=True)
    sse.add_argument("--start", default="2001-01-01")
    sse.add_argument("--end", default="2026-07-31")
    sse.add_argument("--sleep", type=float, default=1.2)
    sse.set_defaults(func=cmd_sse_crawl)
    crawl = sub.add_parser("crawl")
    crawl.add_argument("--cache", required=True)
    crawl.add_argument("--start", default="2001-01-01")
    crawl.add_argument("--bj-start", default="2021-11-01")
    crawl.add_argument("--end", default="2026-07-31")
    crawl.add_argument("--sleep", type=float, default=1.0)
    crawl.set_defaults(func=cmd_crawl)
    kw = sub.add_parser("keyword-crawl")
    kw.add_argument("--cache", required=True)
    kw.add_argument("--start", default="2001-01-01")
    kw.add_argument("--bj-start", default="2021-11-01")
    kw.add_argument("--end", default="2026-07-31")
    kw.add_argument("--sleep", type=float, default=1.0)
    kw.set_defaults(func=cmd_keyword_crawl)
    pdf = sub.add_parser("pdf")
    pdf.add_argument("--cache", required=True)
    pdf.add_argument("--plates", default="")
    pdf.add_argument("--sleep", type=float, default=1.0)
    pdf.set_defaults(func=cmd_pdf)
    build = sub.add_parser("build")
    build.add_argument("--cache", required=True)
    build.add_argument("--fallback-sessions", type=int, default=1,
                       help="sessions after the announcement when the body states no date; -1 = UNKNOWN")
    build.set_defaults(func=cmd_build)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--cache", required=True)
    ev.add_argument("--split", choices=("dev", "holdout"), required=True)
    ev.set_defaults(func=cmd_evaluate)
    ck = sub.add_parser("checks")
    ck.add_argument("--cache", required=True)
    ck.set_defaults(func=cmd_checks)
    rs = sub.add_parser("refresh-szse")
    rs.add_argument("--cache", required=True)
    rs.set_defaults(func=cmd_refresh_szse)
    pb = sub.add_parser("publish")
    pb.add_argument("--cache", required=True)
    pb.add_argument("--szse-register", default=None)
    pb.add_argument("--as-of", default=None)
    pb.set_defaults(func=cmd_publish)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
