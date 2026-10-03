#!/usr/bin/env python3
"""Single trusted baseline evaluator for factor sleeves (A-share strict).

Every factor/model comparison routed through this script uses the same
position-carrying strict A-share simulator. Target-weight indices are **signal
sessions**; this script never pre-shifts them. The strict simulator is the sole
owner of the canonical T-close -> next-global-session execution mapping.

The historical variant names are preserved for artifact compatibility, but
execution timing is now one consistent contract for every variant:

  variant A  flags ON,  next-session execution, raw ranking      (slot-wasting)
  variant B  flags ON,  next-session execution, eligible ranking (smart slots)
  variant C  flags ON,  next-session execution, eligible ranking (legacy trusted alias)
  variant D  flags OFF, next-session execution, raw ranking      (forensic/legacy flags-off)

``C_flags_eligible_delay1`` keeps its old identifier because downstream v8.9
artifacts and scripts reference it. The ``delay1`` suffix is legacy naming only:
there is no extra target-date shift. A signal at T is mapped exactly once by the
strict simulator and executes on T+1 global market session when executable.

"eligible ranking" excludes, at signal time, names you provably cannot or
should not buy that day: suspended, ST (the strategy's own hard risk gate),
and limit-up-sealed closes. This converts rejected *signal-date eligibility*
into next-best picks using information available at the signal close. Actual
next-session order feasibility still belongs to the strict broker/simulator.

Excess is vs the frictionless equal-weight all-A benchmark (close-to-close
mean), per the project's stated target. The benchmark pays no costs, so excess
is a conservative bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.backtest.quarantine import (
    FORENSICS_TRUST_CLASS,
    QuarantineViolation,
    check_window,
    clamp_panel_window,
    load_windows,
    log_access,
    violation_message,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
#: Certified raw execution panel (round 29). The default whenever it exists.
CERTIFIED_PANEL = "runtime/data/gold/full_universe_r29/execution_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
ANN = 244

#: How the legacy default panel is described in every output that used it. It is
#: kept as the default only for backward compatibility: qfq price levels with raw
#: volume/amount, vendor-stitched without SourceBoundary, no rows for suspended
#: sessions after 2020 (R1-F01/F02). Pass ``--panel`` with a certified raw
#: execution panel for a canonical number.
LEGACY_PANEL_NOTE = (
    "legacy v7 silver panel: not verified (qfq price levels with raw volume, "
    "vendor-stitched, suspended sessions absent, is_st = the 2026-05-31 ST list "
    "broadcast to every historical date); not a certified execution panel"
)
#: Trust class stamped on any number produced from the legacy panel. Its ST flag
#: is today's list applied to the past, which excludes names that became ST
#: LATER - future losers - so historical results carry look-ahead (round-29 R4-F01).
LEGACY_PANEL_TRUST_CLASS = "unverified_legacy_panel_lookahead_st"


def _bench_daily(panel: pd.DataFrame, dates) -> pd.Series:
    px = panel[panel["trade_date"].isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    return px.pct_change(fill_method=None).mean(axis=1).dropna()


def _bench_sessions_total_return(panel: pd.DataFrame, start, end) -> pd.Series:
    """Equal-weight all-A total return on every valuation session.

    For a raw execution panel the price is ``close x adjust_factor`` (hfq), so
    ex-rights dates are not read as losses, and the benchmark is computed on the
    sessions the NAV is valued on -- not on signal dates, which for a weekly
    prediction file would compound weekly returns as if they were daily.
    """
    frame = panel[(panel["trade_date"] >= pd.Timestamp(start))
                  & ((panel["trade_date"] <= pd.Timestamp(end)) if end else True)]
    price = frame["close"] * frame.get("adjust_factor", 1.0)
    px = frame.assign(_px=price).pivot_table(index="trade_date", columns="symbol", values="_px")
    return px.pct_change(fill_method=None).mean(axis=1).dropna()


def _load_verified_panel(panel_path: str, p_start, p_end) -> tuple[pd.DataFrame, dict]:
    """Read a certified raw execution panel and refuse anything else."""
    from quantagent.data.ashare.execution_panel import verify_execution_panel

    path = Path(panel_path)
    filters = [("trade_date", ">=", pd.Timestamp(p_start))]
    if p_end is not None:
        filters.append(("trade_date", "<=", pd.Timestamp(p_end)))
    panel = pd.read_parquet(path, filters=filters)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    boundaries_path = path.parent / "source_boundaries.parquet"
    boundaries = pd.read_parquet(boundaries_path) if boundaries_path.exists() else None
    verification = verify_execution_panel(panel, source_boundaries=boundaries)
    meta: dict = {"path": str(path), "verified": True, **verification}
    manifest_path = path.parent / "execution_panel_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["schema"] = manifest.get("schema")
        meta["content_hash"] = manifest.get("content_hash")
    return panel, meta


def _regime_label(bench_daily: pd.Series) -> pd.Series:
    cum = (1 + bench_daily).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(
        np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
        index=bench_daily.index,
    )


def _regime_excess(nav: pd.Series, bench_daily: pd.Series) -> dict:
    strat = nav.pct_change().dropna()
    idx = strat.index.intersection(bench_daily.index)
    strat, bench = strat.reindex(idx), bench_daily.reindex(idx)
    regime = _regime_label(bench).reindex(idx)
    rows = {}
    for rg in ["all", "bull", "sideways", "bear"]:
        mask = pd.Series(True, index=idx) if rg == "all" else (regime == rg)
        n = int(mask.sum())
        if n < 3:
            continue
        s, b = strat[mask], bench[mask]
        ann_s = float((1 + s).prod() ** (ANN / n) - 1)
        ann_b = float((1 + b).prod() ** (ANN / n) - 1)
        rows[rg] = {"days": n, "strat_ann": round(ann_s, 4), "bench_ann": round(ann_b, 4),
                    "excess_ann": round(ann_s - ann_b, 4)}
    return rows


def _target_weights(
    preds: pd.DataFrame,
    score_col: str,
    top_k: int,
    *,
    eligible_only: bool,
) -> pd.DataFrame:
    """Build signal-date target weights; never shift execution dates here."""
    d = preds.copy()
    if eligible_only:
        bad = (
            d.get("is_suspended", pd.Series(False, index=d.index)).fillna(False).astype(bool)
            | d.get("is_st", pd.Series(False, index=d.index)).fillna(False).astype(bool)
            | d.get("is_limit_up", pd.Series(False, index=d.index)).fillna(False).astype(bool)
        )
        d = d[~bad]
    d = d.sort_values(["trade_date", score_col], ascending=[True, False])
    d["rank"] = d.groupby("trade_date").cumcount()
    d = d[d["rank"] < top_k]
    d["w"] = 1.0 / float(top_k)
    return d.pivot_table(
        index="trade_date", columns="symbol", values="w", fill_value=0.0
    ).sort_index()


#: Stamped when the certified panel's ST status is UNKNOWN for names the book can
#: buy (no dated SSE/BSE register exists): those names are treated as non-ST, so
#: an ST name may be held. Round-29 R7-F1.
ST_INCOMPLETE_TRUST_CLASS = "research_certified_panel_st_incomplete"


def _st_unknown_buy_share(trades: pd.DataFrame | None, panel: pd.DataFrame) -> float | None:
    """Share of filled buy notional in names whose ST status was UNKNOWN that day."""
    if trades is None or trades.empty or "st_status" not in panel.columns:
        return None
    fills = trades[(trades["side"].astype(str).str.lower() == "buy")
                   & (pd.to_numeric(trades["filled_quantity"], errors="coerce") > 0)].copy()
    if fills.empty:
        return None
    fills["trade_date"] = pd.to_datetime(fills["trade_date"]).dt.normalize()
    status = panel[["symbol", "trade_date", "st_status"]].copy()
    status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.normalize()
    fills = fills.merge(status, on=["symbol", "trade_date"], how="left")
    notional = pd.to_numeric(fills["filled_quantity"], errors="coerce") * pd.to_numeric(
        fills["avg_price"], errors="coerce")
    total = float(notional.sum())
    if total <= 0:
        return None
    unknown = ~fills["st_status"].astype(str).isin(["FALSE", "TRUE"])
    return round(float(notional[unknown].sum()) / total, 4)


def _sharpe_uncertainty(nav: pd.Series) -> dict[str, object]:
    """Point estimates alone overstate certainty: publish PSR, MinTRL and a
    dependence-preserving bootstrap interval next to the Sharpe ratio."""
    from quantagent.quant_math.performance import (
        minimum_track_record_length,
        probabilistic_sharpe_ratio,
        sharpe_bootstrap_interval,
    )

    daily = pd.Series(nav).sort_index().pct_change().dropna()
    low, high = sharpe_bootstrap_interval(daily)
    min_trl = minimum_track_record_length(daily)
    return {
        "observed_sessions": int(len(daily)),
        "psr_vs_zero": None if not np.isfinite(psr := probabilistic_sharpe_ratio(daily)) else round(psr, 4),
        "min_track_record_sessions": (
            None if not np.isfinite(min_trl) else int(np.ceil(min_trl))
        ),
        "min_track_record_status": (
            "sharpe_not_above_zero" if min_trl == float("inf") else "measured"
        ),
        "sharpe_ci95_stationary_bootstrap": [
            None if not np.isfinite(low) else round(low, 3),
            None if not np.isfinite(high) else round(high, 3),
        ],
        "multiple_testing": "not_deflated: DSR needs the declared trial family (see research gates)",
    }


def _save_ui_backtest(base_dir: str, variant: str, res, m, bench, bench_ann: float,
                      start: str, end: str | None, top_k: int,
                      trust_class: str | None = None) -> str:
    """Emit a UI-discoverable backtest artifact (metrics.json + nav.csv)."""
    d = Path(base_dir) / "backtest"
    d.mkdir(parents=True, exist_ok=True)
    nav = res.nav.copy()
    nav.index = pd.to_datetime(nav.index)
    aligned = bench.reindex(nav.index)
    # A missing benchmark session is unmeasured, not a 0% day: leave the
    # benchmark/excess columns empty from the first gap on instead of compounding
    # a fabricated flat return into the comparison (DEF-022 shape).
    benchmark_gaps = int(aligned.isna().sum())
    bnav = (1.0 + aligned).cumprod()
    navdf = pd.DataFrame({
        "trade_date": nav.index.strftime("%Y-%m-%d"),
        "nav": nav.to_numpy(),
        "daily_return": nav.pct_change().to_numpy(),
        "benchmark_nav": (bnav / bnav.iloc[0]).to_numpy(),
        "excess_nav": (nav / nav.iloc[0]).to_numpy() - (bnav / bnav.iloc[0]).to_numpy(),
    })
    navdf.to_csv(d / "nav.csv", index=False)
    calmar = (m.annualized_return / abs(m.max_drawdown)) if m.max_drawdown else None
    metrics = {
        "start_date": start,
        "end_date": end or str(nav.index.max().date()),
        "variant": variant,
        "top_k": top_k,
        "universe_size": top_k,
        "total_return": round(float(m.total_return), 6),
        "annualized_return": round(float(m.annualized_return), 6),
        "max_drawdown": round(float(m.max_drawdown), 6),
        "sharpe": round(float(m.sharpe), 4),
        "calmar": round(float(calmar), 4) if calmar is not None else None,
        "benchmark_annualized_return": round(float(bench_ann), 6),
        "benchmark_nav_gap_sessions": benchmark_gaps,
        "total_cost_cny": round(float(m.total_cost), 2),
        "slippage_cost_cny": round(float(m.slippage_cost), 2),
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "target_index_semantics": "signal_date_not_pre_shifted",
    }
    if trust_class:
        metrics["trust_class"] = trust_class
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "run_config.json").write_text(json.dumps({
        "strategy_version": "v89_closed_loop", "feature_policy": "judgment",
        "initial_cash": 1_000_000.0, "horizon": variant, "top_k": top_k,
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "target_index_semantics": "signal_date_not_pre_shifted",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(d)


def evaluate(preds_path: str, *, top_k: int, start: str, end: str | None,
             slippage_bps: float, variants: list[str], score_column: str = "alpha_score",
             save_backtest_dir: str | None = None,
             save_variant: str = "C_flags_eligible_delay1",
             allow_quarantined: str | None = None,
             panel_path: str | None = None,
             allow_unverified_panel: str | None = None) -> dict:
    # ---- quarantine guard (fail closed, BEFORE any data is read) ----------
    q_windows, q_log_path = load_windows()
    q_hit = check_window(start, end, q_windows)
    q_record = None
    if q_hit is not None:
        if not (allow_quarantined and allow_quarantined.strip()):
            raise QuarantineViolation(violation_message(start, end, q_hit), q_hit)
        q_record = log_access(q_hit, allow_quarantined.strip(), start, end, q_log_path)
        print(f"[quarantine] FORENSIC OVERRIDE — outputs stamped trust_class={FORENSICS_TRUST_CLASS}",
              flush=True)

    preds = pd.read_parquet(preds_path)
    if score_column != "alpha_score":
        if score_column not in preds.columns:
            raise KeyError(f"--score-column '{score_column}' not in predictions {list(preds.columns)}")
        preds = preds.rename(columns={score_column: "alpha_score"})
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= pd.Timestamp(start)]
    if end:
        preds = preds[preds["trade_date"] <= pd.Timestamp(end)]

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    p_start = pd.Timestamp(start) - pd.Timedelta(days=10)
    p_end = pd.Timestamp(end) + pd.Timedelta(days=10) if end else None
    if q_record is None:
        # Keep the +/-10d execution buffers out of quarantine too: the strict
        # simulator consumes the next global session after each signal date.
        p_start, p_end = clamp_panel_window(p_start, p_end, q_windows)
    universe_note: dict = {}
    if panel_path is None and Path(CERTIFIED_PANEL).exists():
        panel_path = CERTIFIED_PANEL
    if panel_path is None and not (allow_unverified_panel and allow_unverified_panel.strip()):
        raise ValueError(
            "no certified execution panel found at "
            f"{CERTIFIED_PANEL}; the legacy v7 panel broadcasts today's ST list to "
            "every historical date (look-ahead). Pass --panel <certified panel>, or "
            "--allow-unverified-panel '<reason>' to stamp the output "
            f"trust_class={LEGACY_PANEL_TRUST_CLASS}."
        )
    if panel_path:
        # Certified raw execution panel: verified at entry (adjustment 'none',
        # one provider per symbol or a declared SourceBoundary, corporate-action
        # credits present) -- refused otherwise.
        panel, panel_meta = _load_verified_panel(panel_path, p_start, p_end)
        # A prediction with no execution row is not a signal anyone could act on.
        keyed = preds.merge(panel[["symbol", "trade_date"]], on=["symbol", "trade_date"])
        universe_note["predictions_without_execution_row"] = int(len(preds) - len(keyed))
        preds = keyed
        # The strict simulator refuses NaN amount; the vendor lacks it for whole
        # symbols, so those names cannot be executed and are excluded up front
        # (disclosed, not filled with an estimate).
        unmeasured = set(panel.loc[~panel["amount_measured"].astype(bool), "symbol"])
        universe_note["excluded_symbols_amount_unmeasured"] = len(unmeasured)
        universe_note["excluded_prediction_rows_amount_unmeasured"] = int(
            preds["symbol"].isin(unmeasured).sum())
        preds = preds[~preds["symbol"].isin(unmeasured)]
        panel = panel[~panel["symbol"].isin(unmeasured)]
        if "st_status" in panel.columns:
            unknown_rows = float(panel["st_status"].astype(str).eq("UNKNOWN").mean())
            panel_meta["st_unknown_row_share"] = round(unknown_rows, 4)
            if unknown_rows > 0:
                panel_meta["trust_class"] = ST_INCOMPLETE_TRUST_CLASS
                print(f"[panel] ST status UNKNOWN on {unknown_rows:.1%} of rows (no dated "
                      "SSE/BSE register): treated as non-ST; output stamped "
                      f"trust_class={ST_INCOMPLETE_TRUST_CLASS}", file=sys.stderr, flush=True)
    else:
        panel = pd.read_parquet(PANEL, columns=panel_cols)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel[panel["trade_date"] >= p_start]
        if p_end is not None:
            panel = panel[panel["trade_date"] <= p_end]
        panel_meta = {
            "path": PANEL, "verified": False, "note": LEGACY_PANEL_NOTE,
            "trust_class": LEGACY_PANEL_TRUST_CLASS,
            "unverified_reason": allow_unverified_panel.strip(),
        }
        print(f"[panel] WARNING {LEGACY_PANEL_NOTE}", file=sys.stderr, flush=True)
    sector = pd.read_parquet(SECTOR) if Path(SECTOR).exists() else pd.DataFrame()

    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")

    if panel_path:
        last = pd.Timestamp(end) if end else preds["trade_date"].max()
        bench = _bench_sessions_total_return(panel, preds["trade_date"].min(), last)
        bench_basis = "valuation_sessions_total_return_hfq"
    else:
        bench = _bench_daily(panel, sorted(preds["trade_date"].unique()))
        bench_basis = "signal_dates_close_to_close"
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)

    panel_noflags = panel.drop(columns=["is_suspended", "is_st", "is_limit_up", "is_limit_down"])

    # All targets stay on their signal dates. The strict simulator is the sole
    # owner of the next-session execution clock. Variant C keeps its old name
    # only for backward-compatible artifact discovery and is an alias of B.
    spec = {
        "A_flags_raw": dict(eligible=False, flags=True),
        "B_flags_eligible": dict(eligible=True, flags=True),
        "C_flags_eligible_delay1": dict(eligible=True, flags=True),
        "D_noflags_raw": dict(eligible=False, flags=False),
    }
    out: dict = {
        "bench_ann": round(bench_ann, 4),
        "predictions": preds_path,
        "top_k": top_k,
        "start": start,
        "end": end,
        "slippage_bps": slippage_bps,
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "target_index_semantics": "signal_date_not_pre_shifted",
        "legacy_variant_aliases": {"C_flags_eligible_delay1": "B_flags_eligible"},
        "panel": panel_meta,
        "universe_restrictions": universe_note,
        "benchmark_basis": bench_basis,
        "variants": {},
    }
    if q_record is not None:
        out["trust_class"] = FORENSICS_TRUST_CLASS
        out["quarantine_override"] = q_record
    elif panel_meta.get("trust_class"):
        out["trust_class"] = panel_meta["trust_class"]
    for name in variants:
        v = spec[name]
        tw = _target_weights(
            preds,
            "alpha_score",
            top_k,
            eligible_only=v["eligible"],
        )
        use_panel = panel if v["flags"] else panel_noflags
        if panel_path:
            # The simulator only reads rows of names it may hold, and every held
            # name was a target; gap rows keep each of their sessions present.
            use_panel = use_panel[use_panel["symbol"].isin(set(tw.columns))]
        res = run_strict_backtest_v8(
            tw, use_panel, sector_map=sector,
            config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=slippage_bps),
        )
        m = res.metrics
        ca_audit = getattr(res, "corporate_action_audit", None)
        corporate_actions = (
            {"credits": int((ca_audit["basis"] != "delisting_writeoff").sum()),
             "cash_credit_cny": round(float(ca_audit["cash_credit"].sum()), 2),
             "bonus_shares": int(ca_audit.loc[ca_audit["basis"] != "delisting_writeoff",
                                              "bonus_shares"].sum()),
             "delisting_writeoffs": int((ca_audit["basis"] == "delisting_writeoff").sum()),
             "delisting_writeoff_value_cny": round(float(
                 ca_audit.get("written_off_value", pd.Series(dtype=float)).fillna(0.0).sum()), 2)}
            if isinstance(ca_audit, pd.DataFrame) and not ca_audit.empty
            else {"credits": 0}
        )
        rec = {
            **_sharpe_uncertainty(res.nav),
            "ann": round(m.annualized_return, 4),
            "excess_ann": round(m.annualized_return - bench_ann, 4),
            "total": round(m.total_return, 4),
            "sharpe": round(m.sharpe, 3),
            "maxDD": round(m.max_drawdown, 4),
            "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            "regime": _regime_excess(res.nav, bench),
            "corporate_action_credits_applied": corporate_actions,
            # Point estimates of risk and cost belong next to the return
            # (round-29 R7-F3: these were dropped from out.json).
            "volatility": round(float(m.volatility), 4),
            "turnover": round(float(m.turnover), 4),
            "total_cost_cny": round(float(m.total_cost), 2),
            "explicit_fees_cny": round(float(m.explicit_fees), 2),
            "impact_cost_cny": round(float(m.impact_cost), 2),
            "slippage_cost_cny": round(float(m.slippage_cost), 2),
            "st_unknown_buy_value_share": _st_unknown_buy_share(res.trades, panel),
        }
        out["variants"][name] = rec
        if save_backtest_dir and name == save_variant:
            saved = _save_ui_backtest(save_backtest_dir, name, res, m, bench, bench_ann, start, end, top_k,
                                      trust_class=(FORENSICS_TRUST_CLASS if q_record is not None
                                                   else panel_meta.get("trust_class")))
            out["ui_backtest_dir"] = saved
        print(f"{name:28} ann {m.annualized_return:+8.2%} | excess {m.annualized_return - bench_ann:+8.2%} | "
              f"sharpe {m.sharpe:5.2f} | maxDD {m.max_drawdown:6.2%}")
    print(f"{'eqw_all_A_bench':28} ann {bench_ann:+8.2%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--start", default="2024-08-28")
    ap.add_argument("--end", default=None)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--variants", default="A_flags_raw,B_flags_eligible,C_flags_eligible_delay1,D_noflags_raw")
    ap.add_argument("--score-column", default="alpha_score",
                    help="Prediction column to rank on (e.g. composite_score). Renamed to alpha_score internally.")
    ap.add_argument("--save-backtest-dir", default=None,
                    help="If set, write a UI-discoverable <dir>/backtest/{metrics.json,nav.csv} for --save-variant.")
    ap.add_argument("--save-variant", default="C_flags_eligible_delay1",
                    help=(
                        "Which variant to export. Historical default C is retained as a compatibility alias "
                        "for eligible ranking under the canonical strict next-session simulator."
                    ))
    ap.add_argument("--allow-quarantined", default=None, metavar="REASON",
                    help="Forensic override for quarantined windows (configs/quarantined_windows.json). "
                         "Requires a non-empty justification; access is logged and outputs are "
                         "stamped trust_class=contaminated_holdout_forensics.")
    ap.add_argument("--panel", default=None,
                    help=("Certified raw execution panel (e.g. runtime/data/gold/full_universe_r29/"
                          "execution_panel.parquet). Verified at entry: adjustment_method 'none', one "
                          "provider per symbol or a declared SourceBoundary, corporate-action credits. "
                          f"Default: {CERTIFIED_PANEL} when present."))
    ap.add_argument("--allow-unverified-panel", default=None, metavar="REASON",
                    help=("Use the legacy v7 silver panel (today's ST list broadcast to history: "
                          "look-ahead). Requires a justification; outputs are stamped "
                          f"trust_class={LEGACY_PANEL_TRUST_CLASS}."))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    try:
        out = evaluate(args.predictions, top_k=args.top_k, start=args.start, end=args.end,
                       score_column=args.score_column,
                       slippage_bps=args.slippage_bps,
                       save_backtest_dir=args.save_backtest_dir,
                       save_variant=args.save_variant,
                       variants=[v.strip() for v in args.variants.split(",") if v.strip()],
                       allow_quarantined=args.allow_quarantined,
                       panel_path=args.panel,
                       allow_unverified_panel=args.allow_unverified_panel)
    except QuarantineViolation as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
