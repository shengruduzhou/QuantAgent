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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--env-file", default=None,
                      help="credential file to load (names only; values are never printed)")
    snap.set_defaults(func=cmd_snapshot)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
