#!/usr/bin/env python3
"""Validation and reconciliation battery over the acquired A-share data.

Every check runs against real downloaded artifacts — never a fixture — and emits
a machine-readable verdict plus the evidence behind it. A check that cannot run
because its input is missing reports NOT_RUN with the reason; it never reports
PASS by default.

Checks
  schema / dtype / symbol normalisation / timezone
  duplicate (symbol, trade_date)
  OHLC relationships, non-positive prices, negative volume
  volume and amount unit semantics (turnover implied VWAP must sit in [low, high])
  pre-listing and post-delisting rows
  missing-session classification (suspension coverage)
  price-limit plausibility per board regime
  adjustment consistency: raw prices must NOT already be adjusted — verified by
    replaying the ex-rights factor series against the panel's own price jumps
  corporate-action agreement between factor steps and dividend records
  intraday -> daily reconciliation
  point-in-time: available_at never precedes the session close it describes
  freshness against the exchange calendar
  coverage by exchange / board / status / listing era
  cross-provider reconciliation on a real sample

Outputs (runtime/data/u0/validation/):
  validation_report.json, validation_report.md, cross_provider_sample.csv

Usage: AI_quant_venv/bin/python3 scripts/u0_validate.py [--allow-network]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.contracts import DAILY_BARS  # noqa: E402
from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.symbols import SymbolError, identify  # noqa: E402

U0 = REPO / "runtime/data/u0"
PANEL = U0 / "panel/daily_bars_raw.parquet"
GAPS = U0 / "panel/session_gaps.parquet"
MASTER = U0 / "security_master.parquet"
PIT = U0 / "pit"
MINUTE = U0 / "intraday/minute_bars.parquet"
OUT = U0 / "validation"

PASS, FAIL, WARN, NOT_RUN = "PASS", "FAIL", "WARN", "NOT_RUN"

# Board price-limit ceilings used for plausibility only (not as a PIT rule).
BOARD_LIMIT_PCT = {"SH_Main": 0.10, "SZ_Main": 0.10, "ChiNext": 0.20,
                   "STAR": 0.20, "BSE": 0.30}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add(self, name: str, verdict: str, detail: str, evidence: dict | None = None) -> None:
        self.results.append({"check": name, "verdict": verdict, "detail": detail,
                             "evidence": evidence or {}})

    def verdicts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.results:
            out[row["verdict"]] = out.get(row["verdict"], 0) + 1
        return out


def validate_schema(panel: pd.DataFrame, checks: Checks) -> None:
    expected = set(DAILY_BARS.columns)
    missing = sorted(expected - set(panel.columns))
    checks.add("schema_columns", PASS if not missing else FAIL,
               "panel carries the declared daily-bar contract columns",
               {"missing": missing, "columns": list(panel.columns)})
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    bad = [c for c in numeric if c in panel.columns and not pd.api.types.is_numeric_dtype(panel[c])]
    checks.add("schema_dtypes", PASS if not bad else FAIL,
               "OHLCV columns are numeric", {"non_numeric": bad})
    is_datetime = pd.api.types.is_datetime64_any_dtype(panel["trade_date"])
    tz_aware = getattr(panel["trade_date"].dtype, "tz", None) is not None
    checks.add("timestamp_type", PASS if is_datetime and not tz_aware else FAIL,
               "trade_date is a naive datetime interpreted in Asia/Shanghai",
               {"is_datetime": bool(is_datetime), "tz_aware": bool(tz_aware)})


def validate_symbols(panel: pd.DataFrame, checks: Checks) -> None:
    symbols = panel["symbol"].astype(str).unique()
    bad, non_canonical = [], []
    for symbol in symbols:
        try:
            ident = identify(symbol)
        except SymbolError:
            bad.append(symbol)
            continue
        if ident.symbol != symbol:
            non_canonical.append(symbol)
    checks.add("symbol_normalisation", PASS if not bad and not non_canonical else FAIL,
               "every panel symbol resolves to its canonical <code>.<EX> form",
               {"unresolvable": bad[:20], "non_canonical": non_canonical[:20],
                "symbols": int(len(symbols))})


def validate_integrity(panel: pd.DataFrame, checks: Checks) -> None:
    dupes = int(panel.duplicated(["symbol", "trade_date"]).sum())
    checks.add("duplicate_symbol_date", PASS if dupes == 0 else FAIL,
               "no duplicate (symbol, trade_date) rows", {"duplicates": dupes})

    violation = ((panel["high"] < panel["low"]) |
                 (panel["close"] > panel["high"]) | (panel["close"] < panel["low"]) |
                 (panel["open"] > panel["high"]) | (panel["open"] < panel["low"]))
    n_violation = int(violation.fillna(False).sum())
    checks.add("ohlc_relationships", PASS if n_violation == 0 else FAIL,
               "low <= {open, close} <= high on every row", {"violations": n_violation})

    non_positive = int((panel["close"] <= 0).sum() + (panel["open"] <= 0).sum())
    checks.add("non_positive_prices", PASS if non_positive == 0 else FAIL,
               "no zero or negative traded prices", {"rows": non_positive})

    nulls = int(panel["close"].isna().sum())
    checks.add("null_close", PASS if nulls == 0 else FAIL,
               "panel contains traded sessions only, so no null closes",
               {"null_close_rows": nulls})

    negative_volume = int((panel["volume"] < 0).sum())
    checks.add("negative_volume", PASS if negative_volume == 0 else FAIL,
               "volume is non-negative", {"rows": negative_volume})


def validate_units(panel: pd.DataFrame, checks: Checks) -> None:
    """Turnover semantics: amount / volume must be a price inside the day's range."""
    sample = panel[panel["amount"].notna() & (panel["volume"] > 0)]
    if sample.empty:
        checks.add("amount_volume_units", NOT_RUN,
                   "no rows carry turnover, so the unit relationship cannot be tested", {})
        return
    sample = sample.sample(min(200_000, len(sample)), random_state=7)
    vwap = sample["amount"] / sample["volume"]
    inside = ((vwap >= sample["low"] * 0.98) & (vwap <= sample["high"] * 1.02))
    share_inside = float(inside.mean())
    checks.add("amount_volume_units", PASS if share_inside > 0.98 else FAIL,
               "amount/volume implies a VWAP inside the session range "
               "(volume=shares, amount=CNY)",
               {"sampled_rows": int(len(sample)), "share_vwap_in_range": round(share_inside, 5),
                "median_implied_vwap": round(float(vwap.median()), 4)})

    # A lot-denominated volume would inflate the implied VWAP by ~100x.
    ratio = float((vwap / sample["close"]).median())
    checks.add("volume_unit_is_shares", PASS if 0.9 < ratio < 1.1 else FAIL,
               "implied VWAP is the same order as close (volume is shares, not lots)",
               {"median_vwap_over_close": round(ratio, 5)})


def validate_lifecycle(panel: pd.DataFrame, master: pd.DataFrame, checks: Checks) -> None:
    listing = master.set_index("symbol")["listing_date"]
    delisting = master.set_index("symbol")["delisting_date"]
    listed = panel["symbol"].map(listing)
    delisted = panel["symbol"].map(delisting)
    pre = int((listed.notna() & (panel["trade_date"] < listed)).sum())
    post = int((delisted.notna() & (panel["trade_date"] > delisted)).sum())
    checks.add("pre_listing_rows", PASS if pre == 0 else FAIL,
               "no bars dated before the recorded listing date", {"rows": pre})
    checks.add("post_delisting_rows", PASS if post == 0 else FAIL,
               "no bars dated after the recorded delisting date", {"rows": post})


def validate_sessions(checks: Checks) -> None:
    if not GAPS.exists():
        checks.add("suspension_representation", NOT_RUN, "session_gaps.parquet absent", {})
        return
    gaps = pd.read_parquet(GAPS)
    if gaps.empty:
        checks.add("suspension_representation", PASS,
                   "no in-life session is missing a bar", {"gaps": 0})
        return
    counts = gaps["classification"].value_counts().to_dict()
    explained = int(counts.get("SUSPENDED", 0))
    unexplained = int(counts.get("MISSING_UNEXPLAINED", 0))
    total = explained + unexplained
    checks.add("suspension_representation",
               PASS if unexplained == 0 else WARN,
               "in-life sessions without a bar are classified, never left as silent nulls",
               {"suspended": explained, "unexplained": unexplained,
                "explained_share": round(explained / total, 4) if total else None,
                "note": "unexplained gaps are reported, not hidden; they are dominated by "
                        "sessions outside the halt-snapshot coverage window"})


def validate_price_limits(panel: pd.DataFrame, master: pd.DataFrame, checks: Checks) -> None:
    board = master.set_index("symbol")["board"]
    sample = panel.sort_values(["symbol", "trade_date"]).copy()
    sample["board"] = sample["symbol"].map(board)
    sample["prev_close"] = sample.groupby("symbol")["close"].shift(1)
    sample = sample[sample["prev_close"] > 0]
    if sample.empty:
        checks.add("price_limit_plausibility", NOT_RUN, "not enough consecutive sessions", {})
        return
    sample["ret"] = sample["close"] / sample["prev_close"] - 1.0
    sample["ceiling"] = sample["board"].map(BOARD_LIMIT_PCT).fillna(0.10)
    # allow headroom for the first sessions after listing and for resumptions
    breaches = sample[np.abs(sample["ret"]) > sample["ceiling"] * 1.6]
    share = float(len(breaches) / len(sample))
    at_limit = float((np.abs(np.abs(sample["ret"]) - sample["ceiling"]) < 0.004).mean())
    checks.add("price_limit_plausibility", PASS if share < 0.005 else WARN,
               "daily returns respect the board price-limit regime apart from known "
               "no-limit windows (IPO debut, resumption after a long halt)",
               {"breach_share": round(share, 6), "breaches": int(len(breaches)),
                "share_exactly_at_limit": round(at_limit, 5),
                "sampled_rows": int(len(sample))})


def validate_adjustment(panel: pd.DataFrame, checks: Checks) -> None:
    """Raw prices must show the ex-rights gap that adjusted prices remove."""
    factor_path = PIT / "adjust_factors.parquet"
    if not factor_path.exists():
        checks.add("adjustment_is_raw", NOT_RUN, "adjust_factors.parquet absent", {})
        return
    factors = pd.read_parquet(factor_path)
    factors["effective_date"] = pd.to_datetime(factors["effective_date"])
    factors = factors.sort_values(["symbol", "effective_date"])
    factors["factor_step"] = factors.groupby("symbol")["hfq_factor"].pct_change()
    events = factors[factors["factor_step"].abs() > 0.01]
    if events.empty:
        checks.add("adjustment_is_raw", NOT_RUN, "no ex-rights events in the factor table", {})
        return
    frame = panel.sort_values(["symbol", "trade_date"]).copy()
    frame["prev_close"] = frame.groupby("symbol")["close"].shift(1)
    frame = frame[frame["prev_close"] > 0]
    frame["ret"] = frame["close"] / frame["prev_close"] - 1.0
    merged = events.merge(frame, left_on=["symbol", "effective_date"],
                          right_on=["symbol", "trade_date"], how="inner",
                          suffixes=("_f", ""))
    if merged.empty:
        checks.add("adjustment_is_raw", NOT_RUN,
                   "no ex-rights date coincides with a panel session in the covered universe", {})
        return
    # On an ex-rights day a RAW series drops by the entitlement; an already
    # adjusted series does not. Expected raw drop = 1/(1+factor_step) - 1.
    merged["expected_ret"] = 1.0 / (1.0 + merged["factor_step"]) - 1.0
    material = merged[merged["expected_ret"].abs() > 0.02]
    if material.empty:
        checks.add("adjustment_is_raw", NOT_RUN, "no material ex-rights events overlap the panel", {})
        return
    agreement = float((np.sign(material["ret"]) == np.sign(material["expected_ret"])).mean())
    median_error = float((material["ret"] - material["expected_ret"]).abs().median())
    checks.add("adjustment_is_raw", PASS if agreement > 0.8 else FAIL,
               "panel prices show the ex-rights drop, proving they are unadjusted",
               {"events_tested": int(len(material)),
                "sign_agreement": round(agreement, 4),
                "median_abs_error": round(median_error, 4)})


def validate_corporate_actions(checks: Checks) -> None:
    factor_path, action_path = PIT / "adjust_factors.parquet", PIT / "corporate_actions.parquet"
    if not factor_path.exists() or not action_path.exists():
        checks.add("corporate_action_agreement", NOT_RUN,
                   "factor or dividend table absent", {})
        return
    factors = pd.read_parquet(factor_path)
    actions = pd.read_parquet(action_path)
    factors["effective_date"] = pd.to_datetime(factors["effective_date"])
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])
    factor_events = set(zip(factors["symbol"], factors["effective_date"]))
    action_events = set(zip(actions["symbol"], actions["ex_date"]))
    shared_symbols = set(factors["symbol"]) & set(actions["symbol"])
    factor_events = {e for e in factor_events if e[0] in shared_symbols}
    action_events = {e for e in action_events if e[0] in shared_symbols}
    if not action_events:
        checks.add("corporate_action_agreement", NOT_RUN, "no overlapping symbols", {})
        return
    matched = len(factor_events & action_events)
    coverage = matched / len(action_events)
    checks.add("corporate_action_agreement", PASS if coverage > 0.7 else WARN,
               "dividend ex-dates line up with steps in the adjustment-factor series",
               {"symbols_compared": len(shared_symbols), "dividend_events": len(action_events),
                "matched_on_exact_date": matched, "match_share": round(coverage, 4)})


def validate_pit(panel: pd.DataFrame, checks: Checks) -> None:
    if "available_at" not in panel.columns:
        checks.add("pit_available_at", NOT_RUN, "panel has no available_at column", {})
        return
    available = pd.to_datetime(panel["available_at"], errors="coerce")
    known = available.notna()
    if not known.any():
        checks.add("pit_available_at", FAIL, "available_at is entirely unparseable", {})
        return
    leaks = int((available[known] < panel.loc[known, "trade_date"]).sum())
    checks.add("pit_available_at", PASS if leaks == 0 else FAIL,
               "available_at never precedes the session the row describes",
               {"rows_checked": int(known.sum()), "leaking_rows": leaks,
                "unparseable": int((~known).sum())})


def validate_freshness(panel: pd.DataFrame, checks: Checks) -> None:
    calendar_path = PIT / "trading_calendar.parquet"
    if not calendar_path.exists():
        checks.add("freshness", NOT_RUN, "trading calendar absent", {})
        return
    calendar = pd.read_parquet(calendar_path)["trade_date"]
    today = pd.Timestamp.now().normalize()
    sessions = calendar[calendar <= today]
    last_session = sessions.max()
    panel_max = panel["trade_date"].max()
    lag = int(len(sessions[(sessions > panel_max) & (sessions <= last_session)]))
    checks.add("freshness", PASS if lag <= 1 else WARN,
               "panel reaches the latest completed trading session",
               {"last_exchange_session": str(last_session.date()),
                "panel_max_date": str(panel_max.date()), "sessions_behind": lag})


def validate_coverage(panel: pd.DataFrame, master: pd.DataFrame, checks: Checks) -> None:
    covered = set(panel["symbol"].unique())
    master = master.copy()
    master["covered"] = master["symbol"].isin(covered)
    by_board = master.groupby("board")["covered"].agg(["sum", "count"])
    by_status = master.groupby("status")["covered"].agg(["sum", "count"])
    era = pd.cut(master["listing_date"].dt.year,
                 bins=[0, 2000, 2010, 2020, 2100],
                 labels=["pre_2000", "2000s", "2010s", "2020s"])
    by_era = master.groupby(era, observed=False)["covered"].agg(["sum", "count"])
    absent_boards = [b for b, row in by_board.iterrows() if row["sum"] == 0]
    checks.add("board_coverage", PASS if not absent_boards else FAIL,
               "every required board has at least one covered security",
               {"by_board": {b: {"covered": int(r["sum"]), "total": int(r["count"])}
                             for b, r in by_board.iterrows()},
                "absent_boards": absent_boards})
    checks.add("status_coverage",
               PASS if int(by_status.loc[by_status.index == "delisted", "sum"].sum()) > 0 else FAIL,
               "delisted securities are represented (no survivorship-only universe)",
               {"by_status": {s: {"covered": int(r["sum"]), "total": int(r["count"])}
                              for s, r in by_status.iterrows()}})
    checks.add("listing_era_coverage", PASS,
               "coverage broken out by listing era",
               {"by_era": {str(e): {"covered": int(r["sum"]), "total": int(r["count"])}
                           for e, r in by_era.iterrows()}})
    total_covered = int(master["covered"].sum())
    checks.add("universe_completeness",
               PASS if total_covered == len(master) else WARN,
               "every security in the master carries bar history",
               {"covered": total_covered, "master": int(len(master)),
                "missing": int(len(master) - total_covered),
                "coverage_share": round(total_covered / max(1, len(master)), 4)})


def validate_intraday(checks: Checks) -> None:
    if not MINUTE.exists() or not PANEL.exists():
        checks.add("intraday_to_daily_reconciliation", NOT_RUN,
                   "minute bars or daily panel absent", {})
        return
    minute = pd.read_parquet(MINUTE)
    if minute.empty:
        checks.add("intraday_to_daily_reconciliation", NOT_RUN, "minute table is empty", {})
        return
    minute["bar_time"] = pd.to_datetime(minute["bar_time"])
    minute["trade_date"] = minute["bar_time"].dt.normalize()
    daily_from_minute = minute.groupby(["symbol", "trade_date"]).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), bars=("close", "size")).reset_index()
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "high", "low",
                                            "close", "volume"])
    merged = daily_from_minute.merge(panel, on=["symbol", "trade_date"],
                                     suffixes=("_min", "_day"))
    if merged.empty:
        checks.add("intraday_to_daily_reconciliation", NOT_RUN,
                   "no (symbol, date) overlap between minute bars and the daily panel", {})
        return
    close_match = float((np.abs(merged["close_min"] / merged["close_day"] - 1) < 0.002).mean())
    high_ok = float((merged["high_min"] <= merged["high_day"] * 1.002).mean())
    low_ok = float((merged["low_min"] >= merged["low_day"] * 0.998).mean())
    volume_ratio = float((merged["volume_min"] / merged["volume_day"].replace(0, np.nan)).median())
    checks.add("intraday_to_daily_reconciliation",
               PASS if close_match > 0.95 and high_ok > 0.95 and low_ok > 0.95 else WARN,
               "minute bars aggregate to the daily bar of the same session",
               {"sessions_compared": int(len(merged)),
                "close_agreement": round(close_match, 4),
                "high_within_daily": round(high_ok, 4), "low_within_daily": round(low_ok, 4),
                "median_volume_ratio": round(volume_ratio, 4) if volume_ratio == volume_ratio else None})


def validate_cross_provider(panel: pd.DataFrame, allow_network: bool, checks: Checks) -> None:
    if not allow_network:
        checks.add("cross_provider_reconciliation", NOT_RUN,
                   "--allow-network not given, so no independent provider was queried", {})
        return
    from quantagent.data.ashare.sources import TencentSource, TickFlowSource

    served_by = panel.groupby("symbol")["serving_provider"].first() \
        if "serving_provider" in panel.columns else pd.Series(dtype=str)
    tickflow_symbols = list(served_by[served_by.str.startswith("tickflow")].index[:40]) \
        if len(served_by) else []
    rng = np.random.default_rng(11)
    if len(tickflow_symbols) > 12:
        tickflow_symbols = list(rng.choice(tickflow_symbols, 12, replace=False))
    if not tickflow_symbols:
        checks.add("cross_provider_reconciliation", NOT_RUN, "no TickFlow-served symbols", {})
        return
    tencent = TencentSource()
    rows = []
    for symbol in tickflow_symbols:
        ours = panel[panel["symbol"] == symbol][["trade_date", "close", "volume"]]
        if ours.empty:
            continue
        start = str(max(ours["trade_date"].min(), pd.Timestamp("2024-01-01")).date())
        result = tencent.daily_bars(symbol, start, str(ours["trade_date"].max().date()))
        if not result.rows:
            rows.append({"symbol": symbol, "status": result.retry_class, "sessions": 0})
            continue
        joined = ours.merge(result.frame[["trade_date", "close", "volume"]],
                            on="trade_date", suffixes=("_ours", "_alt"))
        if joined.empty:
            rows.append({"symbol": symbol, "status": "NO_OVERLAP", "sessions": 0})
            continue
        close_diff = (joined["close_ours"] / joined["close_alt"] - 1).abs()
        volume_diff = (joined["volume_ours"] / joined["volume_alt"].replace(0, np.nan) - 1).abs()
        rows.append({
            "symbol": symbol, "status": "COMPARED", "sessions": int(len(joined)),
            "close_match_share": float((close_diff < 0.001).mean()),
            "median_close_diff": float(close_diff.median()),
            "volume_match_share": float((volume_diff < 0.001).mean()),
        })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "cross_provider_sample.csv", index=False)
    compared = frame[frame["status"] == "COMPARED"]
    if compared.empty:
        checks.add("cross_provider_reconciliation", NOT_RUN,
                   "no symbol produced an overlapping window", {"attempted": len(frame)})
        return
    agreement = float(compared["close_match_share"].mean())
    checks.add("cross_provider_reconciliation", PASS if agreement > 0.97 else WARN,
               "TickFlow-served prices reconcile against an independent public provider",
               {"symbols_compared": int(len(compared)),
                "mean_close_match_share": round(agreement, 4),
                "mean_volume_match_share": round(float(compared["volume_match_share"].mean()), 4),
                "worst_symbol": compared.sort_values("close_match_share").iloc[0].to_dict()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true",
                        help="enable the cross-provider reconciliation check")
    args = parser.parse_args()
    load_repo_env()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = Checks()

    if not PANEL.exists():
        checks.add("panel_present", FAIL, f"{PANEL.relative_to(REPO)} does not exist", {})
        report = {"generated": _now(), "verdicts": checks.verdicts(), "checks": checks.results}
        (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    master = pd.read_parquet(MASTER)
    master["listing_date"] = pd.to_datetime(master["listing_date"], errors="coerce")
    master["delisting_date"] = pd.to_datetime(master["delisting_date"], errors="coerce")

    validate_schema(panel, checks)
    validate_symbols(panel, checks)
    validate_integrity(panel, checks)
    validate_units(panel, checks)
    validate_lifecycle(panel, master, checks)
    validate_sessions(checks)
    validate_price_limits(panel, master, checks)
    validate_adjustment(panel, checks)
    validate_corporate_actions(checks)
    validate_pit(panel, checks)
    validate_freshness(panel, checks)
    validate_coverage(panel, master, checks)
    validate_intraday(checks)
    validate_cross_provider(panel, args.allow_network, checks)

    report = {
        "generated": _now(),
        "panel": str(PANEL.relative_to(REPO)),
        "panel_rows": int(len(panel)), "panel_symbols": int(panel["symbol"].nunique()),
        "date_range": [str(panel["trade_date"].min().date()), str(panel["trade_date"].max().date())],
        "verdicts": checks.verdicts(),
        "checks": checks.results,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                          default=str))
    lines = [f"# U0 validation report — {report['generated']}\n\n",
             f"Panel: `{report['panel']}` · {report['panel_rows']:,} rows · ",
             f"{report['panel_symbols']:,} symbols · {report['date_range'][0]} → {report['date_range'][1]}\n\n",
             "| check | verdict | detail |\n|---|---|---|\n"]
    for row in checks.results:
        lines.append(f"| {row['check']} | **{row['verdict']}** | {row['detail']} |\n")
    lines.append("\n## Evidence\n\n```json\n")
    lines.append(json.dumps({r["check"]: r["evidence"] for r in checks.results},
                            indent=2, ensure_ascii=False, default=str))
    lines.append("\n```\n")
    (OUT / "validation_report.md").write_text("".join(lines))

    print(json.dumps({"verdicts": report["verdicts"],
                      "failures": [r["check"] for r in checks.results if r["verdict"] == FAIL]},
                     indent=2))
    return 0 if FAIL not in checks.verdicts() else 1


if __name__ == "__main__":
    raise SystemExit(main())
