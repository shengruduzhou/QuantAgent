#!/usr/bin/env python3
"""EXP-008 fold evaluation (WALK_FORWARD_PROTOCOL_H008.md §3-5).

Pre-registered candidates ONLY (N=6, no additions, no per-fold tuning):
  C1_apriori_avg   0.30*short + 0.45*mid + 0.25*long on raw scores (NaN->0)
  C2_prod_rank110  per-date pct-rank sum, weights (1,1,0)
  C3_rank_median   per-date median of the three sleeve pct-ranks
  C3_ema{0.3,0.5,0.7}  per-symbol ewm(adjust=False) on C3 (warmup = fold's
                       pre-OOS prediction days only; identical across folds)

Folds F1-F3 come from wf_h008 retrains; F4 reuses retrain_plus7_20260620_0300.
Every evaluation is strict variant C (eligible + delay-1, k=10, 8bps) on the
fold's OOS window — all pre-quarantine; the quarantine guard stays armed.

Outputs: wf_h008/wf_summary.json + candidate_fold_metrics.csv (+ stitched
daily returns for DSR), fold-block CSCV PBO across the 6 candidates.
"""
from __future__ import annotations

import itertools
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import baseline_protocol as bp  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT_ROOT = REPO / "runtime/reports/v89_closed_loop/wf_h008"
F4_RUN = REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300"
QUARANTINE_START = pd.Timestamp("2025-09-01")
TOP_K = 10
ANN = 244
EULER = 0.5772156649015329
SLEEVES = ("short_5d", "mid_5d_30d", "long_30d_120d")
C1_W = {"short_5d": 0.30, "mid_5d_30d": 0.45, "long_30d_120d": 0.25}

FOLDS = {
    "F1": {"dir": OUT_ROOT / "F1", "oos": ("2023-07-03", "2023-12-29")},
    "F2": {"dir": OUT_ROOT / "F2", "oos": ("2024-01-02", "2024-06-28")},
    "F3": {"dir": OUT_ROOT / "F3", "oos": ("2024-07-01", "2024-12-31")},
    "F4": {"dir": F4_RUN, "oos": ("2025-01-02", "2025-08-29")},
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def cagr(r: np.ndarray) -> float:
    nav = float(np.prod(1.0 + r))
    return nav ** (ANN / len(r)) - 1.0 if len(r) and nav > 0 else -1.0


def max_dd(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max()) if len(r) else 0.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sleeve_frame(fold: str) -> pd.DataFrame:
    """Outer-merge sleeve predictions -> {sleeve}_score columns (NaN->0)."""
    merged: pd.DataFrame | None = None
    for sl in SLEEVES:
        p = FOLDS[fold]["dir"] / sl / "predictions.parquet"
        f = pd.read_parquet(p, columns=["trade_date", "symbol", "alpha_score"])
        f["trade_date"] = pd.to_datetime(f["trade_date"])
        f = f.rename(columns={"alpha_score": f"{sl}_score"})
        merged = f if merged is None else merged.merge(f, on=["trade_date", "symbol"], how="outer")
    assert merged is not None
    for sl in SLEEVES:
        merged[f"{sl}_score"] = pd.to_numeric(merged[f"{sl}_score"], errors="coerce").fillna(0.0)
    # F4 reuses the production run whose prediction FILES extend into the
    # quarantined window (scores only). Truncate at the boundary on read —
    # rows beyond it are discarded unread-for-evaluation (same treatment as
    # the EXP-000 _tmp frames) — then assert the invariant on what remains.
    merged = merged[merged["trade_date"] < QUARANTINE_START].reset_index(drop=True)
    assert merged["trade_date"].max() < QUARANTINE_START, f"{fold}: quarantine breach in predictions"
    return merged


def build_candidates(frame: pd.DataFrame, oos_start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    ranks = {sl: frame.groupby("trade_date")[f"{sl}_score"].rank(pct=True) for sl in SLEEVES}
    keys = frame[["trade_date", "symbol"]].copy()

    def make(score) -> pd.DataFrame:
        f = keys.copy()
        f["alpha_score"] = np.asarray(score)
        return f

    c3_full = make(np.median(np.column_stack([ranks[sl].to_numpy() for sl in SLEEVES]), axis=1))
    cands = {
        "C1_apriori_avg": make(sum(C1_W[sl] * frame[f"{sl}_score"] for sl in SLEEVES).to_numpy()),
        "C2_prod_rank110": make((ranks["short_5d"] + ranks["mid_5d_30d"]).to_numpy()),
        "C3_rank_median": c3_full,
    }
    for a in (0.3, 0.5, 0.7):
        sm = c3_full.sort_values(["symbol", "trade_date"]).copy()
        sm["alpha_score"] = (sm.groupby("symbol")["alpha_score"]
                             .transform(lambda s: s.ewm(alpha=a, adjust=False).mean()))
        cands[f"C3_ema{a}"] = sm
    # slice to OOS window AFTER EMA so pre-OOS prediction days serve as warmup
    return {k: v[v["trade_date"] >= oos_start].reset_index(drop=True) for k, v in cands.items()}


def main() -> int:
    t0 = time.time()
    results: dict[str, dict] = {}
    daily: dict[str, dict[str, pd.Series]] = {}
    sector = pd.read_parquet(REPO / bp.SECTOR)
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START
        frame = sleeve_frame(fold)
        cands = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        bench = bp._bench_daily(panel, trade_dates)
        n_bench = len(bench.loc[oos_s:oos_e])
        bench_ann = float((1 + bench.loc[oos_s:oos_e]).prod() ** (ANN / max(1, n_bench)) - 1)
        smap = dict(zip(sector["symbol"].astype(str), sector.iloc[:, 1].astype(str))) if len(sector.columns) > 1 else {}
        cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)

        results[fold] = {"oos": spec["oos"], "bench_ann": round(bench_ann, 4), "candidates": {}}
        for name, preds in cands.items():
            p = preds.merge(flags, on=["symbol", "trade_date"], how="left")
            tw = bp._target_weights(p, "alpha_score", TOP_K, eligible_only=True,
                                    delay_days=1, trade_dates=trade_dates)
            res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
            nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
            r = nav.pct_change().dropna()
            daily.setdefault(name, {})[fold] = r
            m = res.metrics
            half = len(r) // 2
            halves = [cagr(r.iloc[:half].to_numpy()), cagr(r.iloc[half:].to_numpy())]
            # sector concentration: mean daily max sector weight of the book
            tw_long = tw.stack()
            tw_long = tw_long[tw_long > 0].rename("w").reset_index()
            tw_long.columns = ["trade_date", "symbol", "w"]
            tw_long["sec"] = tw_long["symbol"].astype(str).map(smap).fillna("?")
            sec_max = float(tw_long.groupby(["trade_date", "sec"])["w"].sum()
                            .groupby("trade_date").max().mean()) if len(tw_long) else 0.0
            results[fold]["candidates"][name] = {
                "cagr": round(m.annualized_return, 4), "excess_ann": round(m.annualized_return - bench_ann, 4),
                "sharpe": round(m.sharpe, 3), "maxdd": round(m.max_drawdown, 4),
                "calmar": round(m.annualized_return / m.max_drawdown, 3) if m.max_drawdown else None,
                "turnover": round(float(m.turnover), 4), "n_days": int(len(r)),
                "half_cagrs": [round(x, 4) for x in halves], "worst_half": round(min(halves), 4),
                "mean_max_sector_weight": round(sec_max, 3),
            }
            print(f"{fold} {name:16s} CAGR {m.annualized_return:+.1%} (bench {bench_ann:+.1%}) "
                  f"DD {m.max_drawdown:.1%} turn {m.turnover:.3f}", flush=True)

    names = list(next(iter(results.values()))["candidates"].keys())
    folds = list(FOLDS)
    # per-candidate aggregates
    agg = {}
    for n in names:
        cs = [results[f]["candidates"][n]["cagr"] for f in folds]
        ex = [results[f]["candidates"][n]["excess_ann"] for f in folds]
        dd = [results[f]["candidates"][n]["maxdd"] for f in folds]
        to = [results[f]["candidates"][n]["turnover"] for f in folds]
        agg[n] = {"fold_cagrs": cs, "median_cagr": round(float(np.median(cs)), 4),
                  "min_cagr": round(min(cs), 4), "loss_folds": int(sum(c < 0 for c in cs)),
                  "median_excess": round(float(np.median(ex)), 4),
                  "median_maxdd": round(float(np.median(dd)), 4), "worst_maxdd": round(max(dd), 4),
                  "max_turnover": round(max(to), 4)}

    # fold-block CSCV PBO (4 blocks, C(4,2)=6 splits, select by summed log-growth)
    growth = np.array([[float(np.log1p(daily[n][f]).sum()) for n in names] for f in folds])
    lam = []
    for combo in itertools.combinations(range(len(folds)), 2):
        mask = np.zeros(len(folds), dtype=bool); mask[list(combo)] = True
        tr, te = growth[mask].sum(0), growth[~mask].sum(0)
        w = int(np.argmax(tr))
        omega = float((te <= te[w]).sum()) / (len(names) + 1)
        lam.append(math.log(omega / (1 - omega)))
    pbo = round(float((np.array(lam) <= 0).mean()), 3)

    # DSR on stitched daily returns, cumulative N=50
    def dsr(name: str, n_trials: int = 50) -> float:
        r = pd.concat([daily[name][f] for f in folds]).to_numpy()
        sr = float(r.mean() / r.std(ddof=1))
        z = (r - r.mean()) / r.std(ddof=1)
        g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
        srs = []
        for n2 in names:
            rr = pd.concat([daily[n2][f] for f in folds]).to_numpy()
            srs.append(float(rr.mean() / rr.std(ddof=1)))
        v = float(np.var(srs, ddof=1))
        sr0 = math.sqrt(v) * ((1 - EULER) * norm_ppf(1 - 1 / n_trials)
                              + EULER * norm_ppf(1 - 1 / (n_trials * math.e)))
        denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        return round(norm_cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)

    summary = {
        "protocol": "WALK_FORWARD_PROTOCOL_H008.md", "n_candidates": len(names),
        "cumulative_trials_N": 50, "folds": {f: results[f] for f in folds},
        "aggregates": agg, "fold_block_pbo": pbo,
        "dsr_stitched": {n: dsr(n) for n in names},
        "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "wf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for f in folds:
        for n in names:
            rows.append({"fold": f, "candidate": n, **results[f]["candidates"][n]})
    pd.DataFrame(rows).to_csv(OUT_ROOT / "candidate_fold_metrics.csv", index=False)
    stitched = pd.DataFrame({n: pd.concat([daily[n][f] for f in folds]) for n in names})
    stitched.to_csv(OUT_ROOT / "stitched_daily_returns.csv")
    print(json.dumps({"aggregates": agg, "pbo": pbo, "dsr": summary["dsr_stitched"]}, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
