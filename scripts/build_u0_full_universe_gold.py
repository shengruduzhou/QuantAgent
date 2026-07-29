#!/usr/bin/env python3
"""Build the U0 full-universe Gold dataset and its quality certificate.

Replaces the frozen-cohort path for full-universe work. The old V7 dataset is
never used as a fallback: if this build cannot produce a structurally valid
dataset it fails, rather than silently degrading to 3,872 securities.

Emits the ten artifacts the research workflow depends on::

    manifest.json                  build parameters, hashes, source commit
    dataset.parquet                features + labels + masks, one row per eligible security-day
    adjusted_market_panel.parquet  the adjusted OHLCV panel the features derive from
    eligibility.parquet            per security-day tradability and why
    labels.parquet                 delay-1 executable forward returns
    feature_coverage.parquet       per-feature, per-date availability
    missingness_masks.parquet      explicit NaN masks, so missing is never read as zero
    folds.json                     purged expanding walk-forward with embargo
    lineage.json                   inputs, hashes, exact rebuild command
    quality_certificate.json       structural checks and the FULL_UNIVERSE_GOLD_READY verdict

Design rules that are enforced rather than documented:

* raw and adjusted prices are never mixed -- adjustment is applied once, to all
  price columns, with the mode recorded; volume stays in shares and amount in CNY;
* ST and suspension state are point-in-time, and where no dated register exists
  the mask is ``UNKNOWN`` rather than ``FALSE``;
* IPO seasoning uses the preregistered 60-trading-day rule, counted in observed
  sessions rather than calendar days;
* labels are delay-1 executable, and a row whose t+1 entry was infeasible
  (suspended, sealed limit-up) is dropped rather than kept at a price nobody
  could have paid;
* the rebuild is deterministic and the dataset content hash is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantagent.data.ashare import contracts, gold_bridge  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime" / "data" / "u0"

#: Preregistered seasoning rule. Not a tunable: changing it changes the universe.
IPO_SEASONING_TRADING_DAYS = 60
#: Label horizons in trading days.
HORIZONS = (1, 5, 20)
#: Embargo must be at least the longest label horizon, or a fold's training data
#: overlaps the very returns the next fold is scored on.
EMBARGO_DAYS = max(HORIZONS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    payload = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _schema_hash(frame: pd.DataFrame) -> str:
    spec = json.dumps(
        {c: str(frame[c].dtype) for c in sorted(frame.columns)}, sort_keys=True
    )
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_parquet(path, **kw) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional daily features from the adjusted panel.

    Deliberately a small, transparent set built only from prices and volume the
    panel already carries. This mission validates the frozen model surface on
    the real universe; it is not a factor search, so no new signal families are
    introduced here.

    Every feature uses only information available up to and including ``t``.
    """
    frame = panel.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", sort=False)

    close = frame["close"]
    frame["ret_1d"] = grouped["close"].pct_change(1)
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["ret_20d"] = grouped["close"].pct_change(20)
    frame["ret_60d"] = grouped["close"].pct_change(60)

    for window in (5, 20, 60):
        frame[f"ma_{window}"] = grouped["close"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )
        frame[f"px_to_ma_{window}"] = close / frame[f"ma_{window}"] - 1.0

    frame["vol_20d"] = grouped["close"].transform(
        lambda s: s.pct_change().rolling(20, min_periods=20).std()
    )
    frame["vol_60d"] = grouped["close"].transform(
        lambda s: s.pct_change().rolling(60, min_periods=60).std()
    )

    frame["turnover_20d"] = grouped["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    frame["volume_ratio_5_20"] = (
        grouped["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
        / grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    )
    # Amihud illiquidity: |return| per unit turnover. Computed via a precomputed
    # column plus transform rather than groupby.apply -- apply materialises a
    # frame per symbol and is orders of magnitude slower across ~5,900 symbols.
    frame["_illiq"] = frame["ret_1d"].abs() / frame["amount"].replace(0, np.nan)
    frame["amihud_20d"] = frame.groupby("symbol", sort=False)["_illiq"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    frame = frame.drop(columns=["_illiq"])

    frame["high_low_range_20d"] = (
        grouped["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
        / grouped["low"].transform(lambda s: s.rolling(20, min_periods=20).min()) - 1.0
    )
    frame["gap_open"] = frame["open"] / grouped["close"].shift(1) - 1.0
    frame["intraday_range"] = (frame["high"] - frame["low"]) / frame["close"]

    return frame


FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "px_to_ma_5", "px_to_ma_20", "px_to_ma_60",
    "vol_20d", "vol_60d", "turnover_20d", "volume_ratio_5_20", "amihud_20d",
    "high_low_range_20d", "gap_open", "intraday_range",
)


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def build_folds(dates: pd.Series, *, n_folds: int, embargo: int) -> list[dict]:
    """Purged expanding walk-forward folds with an embargo gap.

    Expanding, not rolling: each fold trains on everything before its test
    window. The embargo removes the ``embargo`` sessions immediately before the
    test window from training, because a label observed at ``t`` looks forward
    up to ``max(HORIZONS)`` sessions and would otherwise overlap the test period.
    """
    unique = pd.Index(sorted(pd.unique(dates)))
    if len(unique) < (n_folds + 1) * (embargo + 20):
        n_folds = max(1, len(unique) // (embargo + 60))
    block = len(unique) // (n_folds + 1)
    folds: list[dict] = []
    for i in range(n_folds):
        train_end_idx = block * (i + 1)
        test_start_idx = train_end_idx + embargo
        test_end_idx = min(test_start_idx + block, len(unique) - 1)
        if test_start_idx >= test_end_idx:
            break
        folds.append({
            "fold": i,
            "train_start": str(unique[0])[:10],
            "train_end": str(unique[train_end_idx - 1])[:10],
            "embargo_days": embargo,
            "embargo_start": str(unique[train_end_idx])[:10],
            "embargo_end": str(unique[test_start_idx - 1])[:10],
            "test_start": str(unique[test_start_idx])[:10],
            "test_end": str(unique[test_end_idx])[:10],
        })
    return folds


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------
def run_quality_checks(dataset: pd.DataFrame, master: pd.DataFrame) -> dict:
    """Structural checks that gate FULL_UNIVERSE_GOLD_READY."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, evidence=None) -> None:
        checks.append({"check": name, "verdict": "PASS" if ok else "FAIL",
                       "detail": detail, "evidence": evidence})

    duplicates = int(dataset.duplicated(subset=["symbol", "trade_date"]).sum())
    add("no_duplicate_security_dates", duplicates == 0,
        "each (symbol, trade_date) appears once", {"duplicates": duplicates})

    identity = master.set_index("symbol")
    dates = pd.to_datetime(dataset["trade_date"])
    listing = pd.to_datetime(
        dataset["symbol"].map(identity.get("listing_date", pd.Series(dtype=object))),
        errors="coerce")
    delisting = pd.to_datetime(
        dataset["symbol"].map(identity.get("delisting_date", pd.Series(dtype=object))),
        errors="coerce")
    pre = int(((dates < listing) & listing.notna()).sum())
    post = int(((dates > delisting) & delisting.notna()).sum())
    add("no_pre_listing_rows", pre == 0, "no row precedes its listing date",
        {"rows": pre})
    add("no_post_delisting_rows", post == 0, "no row follows its delisting date",
        {"rows": post})

    add("adjustment_mode_declared",
        dataset["adjustment_method"].nunique() == 1,
        "exactly one adjustment mode across the dataset",
        {"modes": sorted(map(str, dataset["adjustment_method"].unique()))})

    negative_volume = int((pd.to_numeric(dataset["volume"], errors="coerce") < 0).sum())
    add("volume_non_negative", negative_volume == 0, "volume is non-negative in shares",
        {"rows": negative_volume})

    non_positive_close = int((pd.to_numeric(dataset["close"], errors="coerce") <= 0).sum())
    add("close_positive", non_positive_close == 0, "adjusted close is strictly positive",
        {"rows": non_positive_close})

    label_columns = [c for c in dataset.columns if c.startswith("forward_return_")]
    add("labels_present", bool(label_columns), "delay-1 executable labels emitted",
        {"labels": label_columns})

    mask_columns = [c for c in dataset.columns if c.startswith("mask_")]
    add("masks_present", bool(mask_columns), "explicit eligibility masks emitted",
        {"masks": mask_columns})

    infeasible_kept = (
        int((~dataset["entry_feasible"]).sum()) if "entry_feasible" in dataset else 0
    )
    add("no_infeasible_entries", infeasible_kept == 0,
        "every retained row had a feasible t+1 entry", {"rows": infeasible_kept})

    failed = [c["check"] for c in checks if c["verdict"] == "FAIL"]
    return {
        "generated": _now(),
        "checks": checks,
        "failed_checks": failed,
        "structurally_valid": not failed,
        # Flat counts consumed by ReadinessEvaluator.full_universe_gold(). The
        # evaluator defines the interface contract, so the builder satisfies it
        # exactly rather than the tier being taught a second shape.
        "duplicate_security_dates": duplicates,
        "out_of_life_rows": pre + post,
        "pre_listing_rows": pre,
        "post_delisting_rows": post,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runtime/data/gold/full_universe")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="0 builds the whole universe; >0 is a bounded build")
    parser.add_argument("--adjustment", default=contracts.ADJUST_HFQ,
                        choices=list(gold_bridge.ADJUSTMENT_METHODS))
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    rebuild_command = " ".join(["python", "scripts/build_u0_full_universe_gold.py", *sys.argv[1:]])

    print("[1/8] loading U0 inputs ...", flush=True)
    panel = pd.read_parquet(
        U0 / "panel/daily_bars_raw.parquet",
        columns=["symbol", "trade_date", "open", "high", "low", "close",
                 "volume", "amount", "serving_provider"])
    if args.start_date:
        panel = panel[panel["trade_date"] >= args.start_date]
    if args.end_date:
        panel = panel[panel["trade_date"] <= args.end_date]

    master = _read(U0 / "security_master.parquet")
    factors = _read(U0 / "pit/adjust_factors.parquet")
    suspension = _read(U0 / "pit/suspension_intervals.parquet")
    st = _read(U0 / "pit/st_intervals.parquet")

    pit_certificate = json.loads((U0 / "u0_strict_pit_certificate.json").read_text("utf-8"))
    st_field = pit_certificate.get("pit_field_availability", {}).get("st_intervals", "")
    st_available = st_field.startswith("AVAILABLE")

    if args.max_symbols:
        merged = master.merge(panel[["symbol"]].drop_duplicates(), on="symbol")
        per_board = max(1, args.max_symbols // max(1, merged["board"].nunique()))
        picks: list[str] = []
        for _, group in merged.groupby("board"):
            picks.extend(sorted(group["symbol"])[:per_board])
        panel = panel[panel["symbol"].isin(picks[:args.max_symbols])]

    print(f"      panel rows={len(panel):,} symbols={panel.symbol.nunique():,}", flush=True)

    print("[2/8] applying adjustment and eligibility masks ...", flush=True)
    adjusted = gold_bridge.apply_adjustment(panel, factors, method=args.adjustment)
    masked = gold_bridge.build_masks(
        adjusted, master=master, suspension=suspension, st=st,
        st_available=st_available, seasoning_days=IPO_SEASONING_TRADING_DAYS)

    print("[3/8] computing features ...", flush=True)
    featured = build_features(masked)

    print("[4/8] building delay-1 executable labels ...", flush=True)
    labelled, dropped = gold_bridge.build_labels(featured, horizons=HORIZONS)

    label_columns = [c for c in labelled.columns if c.startswith("forward_return_")]
    mask_columns = [c for c in labelled.columns if c.startswith("mask_")]
    feature_columns = [c for c in FEATURE_COLUMNS if c in labelled.columns]

    # Rows with no usable features are structurally useless; dropping them here
    # keeps the dataset honest about what it can actually train on.
    dataset = labelled.dropna(subset=feature_columns, how="all").reset_index(drop=True)

    print("[5/8] writing artifacts ...", flush=True)
    dataset.to_parquet(target / "dataset.parquet", index=False)

    panel_columns = ["symbol", "trade_date", "open", "high", "low", "close",
                     "volume", "amount", "adjust_factor", "adjustment_method"]
    dataset[[c for c in panel_columns if c in dataset.columns]].to_parquet(
        target / "adjusted_market_panel.parquet", index=False)

    eligibility_columns = (["symbol", "trade_date", "eligible_for_training",
                            "entry_feasible"] + mask_columns
                           + [c for c in dataset.columns if c.startswith("has_")])
    dataset[[c for c in eligibility_columns if c in dataset.columns]].to_parquet(
        target / "eligibility.parquet", index=False)

    dataset[["symbol", "trade_date", "entry_close_t1", *label_columns]].to_parquet(
        target / "labels.parquet", index=False)

    masks = dataset[["symbol", "trade_date"]].copy()
    for column in feature_columns:
        masks[f"missing_{column}"] = dataset[column].isna()
    masks.to_parquet(target / "missingness_masks.parquet", index=False)

    coverage_rows = []
    by_date = dataset.groupby("trade_date")
    for column in feature_columns:
        present = by_date[column].apply(lambda s: float(s.notna().mean()))
        coverage_rows.append(pd.DataFrame({
            "feature": column, "trade_date": present.index, "coverage": present.values}))
    coverage = pd.concat(coverage_rows, ignore_index=True)
    coverage.to_parquet(target / "feature_coverage.parquet", index=False)

    print("[6/8] building purged walk-forward folds ...", flush=True)
    folds = build_folds(dataset["trade_date"], n_folds=args.folds, embargo=EMBARGO_DAYS)
    (target / "folds.json").write_text(json.dumps({
        "scheme": "purged expanding walk-forward",
        "embargo_days": EMBARGO_DAYS,
        "embargo_rationale": (
            "at least the longest label horizon, so a training label cannot look "
            "forward into its own test window"),
        "horizons": list(HORIZONS),
        "folds": folds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[7/8] running structural quality checks ...", flush=True)
    quality = run_quality_checks(dataset, master)

    dataset_hash = _frame_hash(dataset)
    schema_hash = _schema_hash(dataset)
    feature_hash = hashlib.sha256(
        json.dumps(sorted(feature_columns)).encode()).hexdigest()[:16]
    label_hash = hashlib.sha256(
        json.dumps(sorted(label_columns)).encode()).hexdigest()[:16]
    fold_hash = hashlib.sha256(
        json.dumps(folds, sort_keys=True).encode()).hexdigest()[:16]

    warnings: list[str] = []
    if not st_available:
        warnings.append(
            "ST intervals are not a complete dated register (SZSE only); mask_is_st "
            "is UNKNOWN for exchanges without one. This dataset is therefore NOT "
            "point-in-time complete for ST, and FULL_UNIVERSE_RESEARCH_READY must "
            "stay withheld.")

    granted = quality["structurally_valid"]
    certificate = {
        "certificate": "FULL_UNIVERSE_GOLD_READY",
        "granted": granted,
        "generated": _now(),
        "source_commit": _source_commit(),
        "dataset_hash": dataset_hash,
        "schema_hash": schema_hash,
        "rows": int(len(dataset)),
        "symbols": int(dataset.symbol.nunique()),
        "date_range": [str(dataset.trade_date.min())[:10], str(dataset.trade_date.max())[:10]],
        "quality": quality,
        # Promoted to the top level because ReadinessEvaluator reads the
        # certificate root; nesting them made the tier report UNMET on a
        # dataset that had actually passed.
        "duplicate_security_dates": quality["duplicate_security_dates"],
        "out_of_life_rows": quality["out_of_life_rows"],
        "warnings": warnings,
        "scope_note": (
            "Structural readiness only. It permits full-universe training; it does "
            "NOT permit formal research claims, which require "
            "FULL_UNIVERSE_RESEARCH_READY."),
    }
    (target / "quality_certificate.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2), encoding="utf-8")

    boards = master.set_index("symbol").get("board", pd.Series(dtype=object))
    board_counts = dataset["symbol"].map(boards).value_counts().to_dict()

    manifest = {
        "generated": _now(),
        "source_commit": _source_commit(),
        "rebuild_command": rebuild_command,
        "adjustment_method": args.adjustment,
        "adjustment_note": "applied once to all price columns; volume stays shares, amount stays CNY",
        "ipo_seasoning_trading_days": IPO_SEASONING_TRADING_DAYS,
        "horizons": list(HORIZONS),
        "embargo_days": EMBARGO_DAYS,
        "rows": int(len(dataset)),
        "symbols": int(dataset.symbol.nunique()),
        "date_range": [str(dataset.trade_date.min())[:10], str(dataset.trade_date.max())[:10]],
        "boards": {str(k): int(v) for k, v in board_counts.items()},
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "mask_columns": mask_columns,
        "rows_dropped": dropped,
        "dataset_hash": dataset_hash,
        # Alias consumed by ReadinessEvaluator, which names this field
        # content_hash. Emitting both keeps one source of truth for the value.
        "content_hash": dataset_hash,
        "schema_hash": schema_hash,
        "feature_hash": feature_hash,
        "label_hash": label_hash,
        "fold_hash": fold_hash,
        "st_pit_complete": st_available,
        "warnings": warnings,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    (target / "lineage.json").write_text(json.dumps({
        "generated": _now(),
        "source_commit": _source_commit(),
        "rebuild_command": rebuild_command,
        "inputs": {
            "panel": "runtime/data/u0/panel/daily_bars_raw.parquet",
            "security_master": "runtime/data/u0/security_master.parquet",
            "adjust_factors": "runtime/data/u0/pit/adjust_factors.parquet",
            "suspension_intervals": "runtime/data/u0/pit/suspension_intervals.parquet",
            "st_intervals": "runtime/data/u0/pit/st_intervals.parquet",
            "u0_strict_pit_certificate": "runtime/data/u0/u0_strict_pit_certificate.json",
        },
        "upstream_decisions": {
            "u0_bar": pit_certificate.get("bar_decision"),
            "u0_pit": pit_certificate.get("decision"),
            "blocked_pit_fields": pit_certificate.get("blocked_pit_fields", []),
        },
        "hashes": {"dataset": dataset_hash, "schema": schema_hash,
                   "feature": feature_hash, "label": label_hash, "fold": fold_hash},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[8/8] done", flush=True)
    print(json.dumps({
        "granted": granted,
        "certificate": "FULL_UNIVERSE_GOLD_READY",
        "rows": manifest["rows"], "symbols": manifest["symbols"],
        "date_range": manifest["date_range"], "boards": manifest["boards"],
        "features": len(feature_columns), "labels": label_columns,
        "dataset_hash": dataset_hash, "schema_hash": schema_hash,
        "folds": len(folds), "embargo_days": EMBARGO_DAYS,
        "rows_dropped": dropped,
        "failed_checks": quality["failed_checks"],
        "warnings": warnings,
        "output": str(target),
    }, ensure_ascii=False, indent=2))
    return 0 if granted else 2


if __name__ == "__main__":
    raise SystemExit(main())
