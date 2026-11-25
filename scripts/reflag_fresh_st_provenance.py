#!/usr/bin/env python3
"""Re-derive ``is_st`` for silver-panel rows whose ST flag was a current-snapshot broadcast.

Rows 2026-05-19..2026-08-05 of ``runtime/data/v7/silver/market_panel`` carry
``is_st_provenance = 'current_snapshot_broadcast'``: today's ST names were
painted backwards. This script replaces that state **only where dated PIT ST
evidence covers the row** (``st_pit_evidence.parquet``: SZSE throughout, SSE/BSE
from 2026-07-27 per ENG_ST31), through the same
:func:`quantagent.data.providers.st_pit.attach_historical_st` the catch-up uses.

Rows the evidence does not cover keep their ``is_st`` value but are relabelled
``current_snapshot_broadcast_unverified_no_dated_evidence`` -- explicitly
unknown provenance. The label keeps the ``current_snapshot``/``broadcast``
markers on purpose: the v7 dataset builder refuses ``is_st`` as a feature when
any row carries them, and a reflag must never make a consumer *more* willing to
trust these rows. No ST state is invented. The label protects ONLY the v7
training builder: FRESH decisions and fills still read the (current-snapshot)
``is_st`` value of these rows -- see docs/DATA.md §5.

Default is a dry run (report only). ``--output PATH`` writes a NEW panel version
plus ``PATH.manifest.json``; it refuses to write the live panel path. Swapping
the new version in is an operator action after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.providers.st_pit import (  # noqa: E402
    attach_historical_st,
    load_historical_st_evidence,
)

LIVE_RUNTIME = Path("/home/shanhefu/QuantAgent/runtime")
DEFAULT_PANEL = LIVE_RUNTIME / "data/v7/silver/market_panel/market_panel.parquet"
DEFAULT_EVIDENCE = LIVE_RUNTIME / "data/u0/pit/st_pit_evidence.parquet"
BROADCAST = "current_snapshot_broadcast"
DATED = "dated_pit_st_evidence"
UNKNOWN = "current_snapshot_broadcast_unverified_no_dated_evidence"


def check_output_path(output: Path, panel_path: Path) -> None:
    """Refuse any target that could overwrite live or existing data."""
    target = Path(output).resolve()
    live_dir = DEFAULT_PANEL.parent.resolve()
    if target in (DEFAULT_PANEL.resolve(), Path(panel_path).resolve()):
        raise SystemExit("refusing to overwrite the input/live panel; write a new version")
    if target == live_dir or live_dir in target.parents:
        raise SystemExit(f"refusing to write inside the live panel directory {live_dir}")
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing {target}; choose a new path")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_mask(rows: pd.DataFrame, evidence_frame: pd.DataFrame) -> pd.Series:
    """True where exactly the interval lookup ``attach_historical_st`` does resolves."""
    left = rows[["symbol", "trade_date"]].copy()
    left["_i"] = rows.index
    left = left.sort_values(["trade_date", "symbol"], kind="mergesort")
    right = evidence_frame[["symbol", "start_date", "end_date"]].sort_values(
        ["start_date", "symbol"], kind="mergesort")
    hit = pd.merge_asof(left, right, left_on="trade_date", right_on="start_date",
                        by="symbol", direction="backward", allow_exact_matches=True)
    ok = hit["end_date"].notna() & (hit["trade_date"] <= hit["end_date"])
    return pd.Series(ok.to_numpy(), index=hit["_i"].to_numpy()).reindex(rows.index).fillna(False)


def reflag(panel: pd.DataFrame, evidence) -> tuple[pd.DataFrame, dict]:
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    target = out["is_st_provenance"].astype(str) == BROADCAST
    rows = out.loc[target, ["symbol", "trade_date", "is_st"]].copy()
    rows["exchange"] = rows["symbol"].astype(str).str[-2:]
    covered = coverage_mask(rows, evidence.frame)

    report: dict = {
        "rows_with_broadcast_provenance": int(target.sum()),
        "broadcast_date_range": [str(rows["trade_date"].min().date()),
                                 str(rows["trade_date"].max().date())] if len(rows) else None,
        "evidence_sha256": evidence.source_sha256,
    }
    attached = attach_historical_st(rows.loc[covered, ["symbol", "trade_date"]], evidence)
    new_st = pd.Series(attached["is_st"].to_numpy(), index=rows.index[covered.to_numpy()])
    old_st = rows.loc[covered, "is_st"].astype(bool)
    by_ex = {}
    for exchange, group in rows.groupby("exchange"):
        cov = covered.loc[group.index]
        flips = (old_st.reindex(group.index[cov.to_numpy()]) != new_st.reindex(
            group.index[cov.to_numpy()]))
        by_ex[exchange] = {
            "rows": int(len(group)),
            "dated_evidence_rows": int(cov.sum()),
            "unknown_rows": int((~cov).sum()),
            "unknown_date_range": [str(group.loc[~cov, "trade_date"].min().date()),
                                   str(group.loc[~cov, "trade_date"].max().date())]
            if (~cov).any() else None,
            "broadcast_is_st_true_rows": int(group["is_st"].astype(bool).sum()),
            "broadcast_is_st_true_unknown_rows": int(group.loc[~cov, "is_st"].astype(bool).sum()),
            "dated_is_st_true_rows": int(new_st.reindex(group.index[cov.to_numpy()]).sum()),
            "flipped_rows": int(flips.sum()),
            "flipped_true_to_false": int((old_st.reindex(flips.index) & ~new_st.reindex(
                flips.index).astype(bool)).sum()),
            "flipped_false_to_true": int((~old_st.reindex(flips.index) & new_st.reindex(
                flips.index).astype(bool)).sum()),
            "symbols_with_flips": int(group.loc[flips[flips].index, "symbol"].nunique())
            if flips.any() else 0,
        }
    report["by_exchange"] = by_ex
    report["dated_evidence_rows"] = int(covered.sum())
    report["unknown_rows"] = int((~covered).sum())

    idx_cov = rows.index[covered.to_numpy()]
    idx_unk = rows.index[~covered.to_numpy()]
    out.loc[idx_cov, "is_st"] = new_st.astype(bool).to_numpy()
    out.loc[idx_cov, "is_st_provenance"] = DATED
    out.loc[idx_cov, "st_evidence_sha256"] = evidence.source_sha256
    out.loc[idx_unk, "is_st_provenance"] = UNKNOWN
    out.loc[idx_unk, "st_evidence_sha256"] = None
    out["is_st"] = out["is_st"].astype(bool)
    report["provenance_after"] = {str(k): int(v) for k, v in
                                  out["is_st_provenance"].value_counts(dropna=False).items()}
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    ap.add_argument("--output", default=None,
                    help="write a NEW panel version here (never the live path)")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args()
    panel_path = Path(args.panel)
    if args.output:
        check_output_path(Path(args.output), panel_path)
    evidence = load_historical_st_evidence(args.evidence)
    panel = pd.read_parquet(panel_path)
    original_dtypes = panel.dtypes.to_dict()
    out, report = reflag(panel, evidence)
    report.update({"panel": str(panel_path), "panel_sha256": _sha(panel_path),
                   "panel_rows": int(len(panel)), "mode": "apply" if args.output else "dry_run",
                   "generated_at": datetime.now(timezone.utc).isoformat()})
    if len(out) != len(panel) or not out[["symbol", "trade_date"]].equals(
            panel[["symbol", "trade_date"]].assign(trade_date=pd.to_datetime(panel["trade_date"]))):
        raise SystemExit("row identity changed during reflag; aborting")
    if args.output:
        out = out.astype({k: v for k, v in original_dtypes.items()
                          if k not in ("is_st_provenance", "st_evidence_sha256")})
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(target)
        report["output"] = str(target)
        report["output_sha256"] = _sha(target)
        Path(str(target) + ".manifest.json").write_text(json.dumps(report, indent=1))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
