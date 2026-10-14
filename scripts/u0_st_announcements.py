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
``docs/audits/round30/roles/ENG_ST30_method.md``.
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


def cmd_crawl(args: argparse.Namespace) -> int:
    cache = Path(args.cache) / "query"
    cache.mkdir(parents=True, exist_ok=True)
    log = []
    for plate, first in (("sh", args.start), ("sz", args.start), ("bj", args.bj_start)):
        for lo, hi in _months(first, args.end):
            tag = f"{plate}_{lo:%Y%m}"
            done = cache / f"{tag}.done.json"
            if done.exists():
                continue
            payload = {"pageNum": "1", "pageSize": str(PAGE_SIZE), "column": "szse",
                       "tabName": "fulltext", "plate": plate, "stock": "", "searchkey": "",
                       "secid": "", "category": CATEGORY, "trade": "",
                       "seDate": f"{lo:%Y-%m-%d}~{hi:%Y-%m-%d}", "sortName": "",
                       "sortType": "", "isHLtitle": "true"}
            first_page = _post(payload, args.sleep)
            total = int(first_page.get("totalAnnouncement") or 0)
            pages = max(1, -(-total // PAGE_SIZE))
            rows = list(first_page.get("announcements") or [])
            (cache / f"{tag}_p1.json").write_text(json.dumps(first_page, ensure_ascii=False))
            for page in range(2, pages + 1):
                payload["pageNum"] = str(page)
                body = _post(payload, args.sleep)
                (cache / f"{tag}_p{page}.json").write_text(json.dumps(body, ensure_ascii=False))
                rows.extend(body.get("announcements") or [])
            ids = {r.get("announcementId") for r in rows}
            record = {"plate": plate, "window": [str(lo), str(hi)], "total": total,
                      "pages": pages, "rows": len(rows), "unique_ids": len(ids),
                      "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "endpoint": QUERY_URL, "category": CATEGORY}
            done.write_text(json.dumps(record))
            log.append(record)
            if total:
                print(json.dumps(record), flush=True)
    print(f"crawl complete: {len(log)} new windows", flush=True)
    return 0


def load_announcements(cache_dir: str | Path):
    """All cached announcements, de-duplicated by announcementId, with plate."""
    import pandas as pd

    cache = Path(cache_dir) / "query"
    records = {}
    for path in sorted(cache.glob("*_p*.json")):
        plate = path.name.split("_", 1)[0]
        body = json.loads(path.read_text())
        for row in body.get("announcements") or []:
            row = dict(row)
            row["plate"] = plate
            row["raw_file"] = path.name
            records[row["announcementId"]] = row
    frame = pd.DataFrame(list(records.values()))
    if frame.empty:
        return frame
    frame["announcement_date"] = (
        pd.to_datetime(frame["announcementTime"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    )
    return frame


def cmd_pdf(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from quantagent.data.ashare import st_announcements as sa

    frame = load_announcements(args.cache)
    frame["event"] = frame["announcementTitle"].map(sa.classify_title)
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
    pdf = sub.add_parser("pdf")
    pdf.add_argument("--cache", required=True)
    pdf.add_argument("--plates", default="")
    pdf.add_argument("--sleep", type=float, default=1.0)
    pdf.set_defaults(func=cmd_pdf)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
