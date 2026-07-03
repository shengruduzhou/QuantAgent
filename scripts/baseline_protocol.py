#!/usr/bin/env python3
"""THE single trusted baseline evaluator for factor sleeves (A-share strict).

Every factor/model comparison must run through this protocol so numbers are
comparable. It decomposes where returns come from:

  variant A  flags ON,  t-close fill,  raw ranking      (strict, slot-wasting)
  variant B  flags ON,  t-close fill,  eligible ranking (strict + smart slots)
  variant C  flags ON,  t+1 fill,      eligible ranking (honest deliverable)
  variant D  flags OFF, t-close fill,  raw ranking      (legacy phantom number)

"eligible ranking" excludes, at signal time, names you provably cannot or
should not buy that day: suspended, ST (the strategy's own hard risk gate),
and limit-up-sealed closes. This converts rejected orders into next-best
picks instead of cash drag — implementable live because all three states
are observable at the close.

Excess is vs the frictionless equal-weight all-A benchmark (close-to-close
mean), per the project's stated target. The benchmark pays no costs, so
excess is a conservative bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
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
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
ANN = 244


def _bench_daily(panel: pd.DataFrame, dates) -> pd.Series:
    px = panel[panel["trade_date"].isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    return px.pct_change(fill_method=None).mean(axis=1).dropna()


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


def _target_weights(preds: pd.DataFrame, score_col: str, top_k: int, *, eligible_only: bool,
                    delay_days: int, trade_dates: list[pd.Timestamp]) -> pd.DataFrame:
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
    tw = d.pivot_table(index="trade_date", columns="symbol", values="w", fill_value=0.0).sort_index()
    if delay_days > 0:
        # Signal at t is executed on the (t + delay)-th trading day.
        date_index = pd.DatetimeIndex(sorted(trade_dates))
        positions = date_index.searchsorted(tw.index) + delay_days
        keep = positions < len(date_index)
        tw = tw.iloc[keep]
        tw.index = date_index[positions[keep]]
        tw = tw[~tw.index.duplicated(keep="last")].sort_index()
    return tw


def _save_ui_backtest(base_dir: str, variant: str, res, m, bench, bench_ann: float,
                      start: str, end: str | None, top_k: int,
                      trust_class: str | None = None) -> str:
    """Emit a UI-discoverable backtest artifact (metrics.json + nav.csv) so the
    real-test CAGR/Calmar surfaces in the quant UI (`services/quant_api`).

    The indexer classifies anything under a ``/backtest/`` path as kind=backtest;
    the adapter needs metrics.json + a sibling nav.csv and reads `calmar` directly.
    """
    d = Path(base_dir) / "backtest"
    d.mkdir(parents=True, exist_ok=True)
    nav = res.nav.copy()
    nav.index = pd.to_datetime(nav.index)
    bnav = (1.0 + bench.reindex(nav.index).fillna(0.0)).cumprod()
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
    }
    if trust_class:
        metrics["trust_class"] = trust_class
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "run_config.json").write_text(json.dumps({
        "strategy_version": "v89_closed_loop", "feature_policy": "judgment",
        "initial_cash": 1_000_000.0, "horizon": variant, "top_k": top_k,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(d)


def evaluate(preds_path: str, *, top_k: int, start: str, end: str | None,
             slippage_bps: float, variants: list[str], score_column: str = "alpha_score",
             save_backtest_dir: str | None = None,
             save_variant: str = "C_flags_eligible_delay1",
             allow_quarantined: str | None = None) -> dict:
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
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    p_start = pd.Timestamp(start) - pd.Timedelta(days=10)
    p_end = pd.Timestamp(end) + pd.Timedelta(days=10) if end else None
    if q_record is None:
        # Clean eval window: keep the +/-10d fill buffers out of quarantine too
        # (delay-1 fills otherwise execute inside the burned window).
        p_start, p_end = clamp_panel_window(p_start, p_end, q_windows)
    panel = panel[panel["trade_date"] >= p_start]
    if p_end is not None:
        panel = panel[panel["trade_date"] <= p_end]
    sector = pd.read_parquet(SECTOR)

    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel["trade_date"].unique())

    bench = _bench_daily(panel, sorted(preds["trade_date"].unique()))
    bench_ann = float((1 + bench).prod() ** (ANN / max(1, len(bench))) - 1)

    panel_noflags = panel.drop(columns=["is_suspended", "is_st", "is_limit_up", "is_limit_down"])

    spec = {
        "A_flags_raw": dict(eligible=False, delay=0, flags=True),
        "B_flags_eligible": dict(eligible=True, delay=0, flags=True),
        "C_flags_eligible_delay1": dict(eligible=True, delay=1, flags=True),
        "D_noflags_raw": dict(eligible=False, delay=0, flags=False),
    }
    out: dict = {"bench_ann": round(bench_ann, 4), "predictions": preds_path,
                 "top_k": top_k, "start": start, "end": end, "slippage_bps": slippage_bps,
                 "variants": {}}
    if q_record is not None:
        out["trust_class"] = FORENSICS_TRUST_CLASS
        out["quarantine_override"] = q_record
    for name in variants:
        v = spec[name]
        tw = _target_weights(preds, "alpha_score", top_k, eligible_only=v["eligible"],
                             delay_days=v["delay"], trade_dates=trade_dates)
        use_panel = panel if v["flags"] else panel_noflags
        res = run_strict_backtest_v8(
            tw, use_panel, sector_map=sector,
            config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=slippage_bps),
        )
        m = res.metrics
        rec = {
            "ann": round(m.annualized_return, 4),
            "excess_ann": round(m.annualized_return - bench_ann, 4),
            "total": round(m.total_return, 4),
            "sharpe": round(m.sharpe, 3),
            "maxDD": round(m.max_drawdown, 4),
            "regime": _regime_excess(res.nav, bench),
        }
        out["variants"][name] = rec
        if save_backtest_dir and name == save_variant:
            saved = _save_ui_backtest(save_backtest_dir, name, res, m, bench, bench_ann, start, end, top_k,
                                      trust_class=(FORENSICS_TRUST_CLASS if q_record is not None else None))
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
                    help="Which variant to export as the UI backtest (default = honest variant C).")
    ap.add_argument("--allow-quarantined", default=None, metavar="REASON",
                    help="Forensic override for quarantined windows (configs/quarantined_windows.json). "
                         "Requires a non-empty justification; access is logged and outputs are "
                         "stamped trust_class=contaminated_holdout_forensics.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    try:
        out = evaluate(args.predictions, top_k=args.top_k, start=args.start, end=args.end,
                       score_column=args.score_column,
                       slippage_bps=args.slippage_bps,
                       save_backtest_dir=args.save_backtest_dir,
                       save_variant=args.save_variant,
                       variants=[v.strip() for v in args.variants.split(",") if v.strip()],
                       allow_quarantined=args.allow_quarantined)
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
