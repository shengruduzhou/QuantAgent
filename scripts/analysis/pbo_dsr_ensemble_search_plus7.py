#!/usr/bin/env python3
"""Retroactive PBO / DSR diagnostics for the ensemble_search_plus7 candidates (Phase 2.5).

ANALYSIS-ONLY. Not wired into any pipeline. Never reads dates >= 2025-09-01
(the quarantined holdout): the market panel is filter-read up to 2025-08-31 and
an assertion enforces it.

What it does
------------
1. Reconstructs the 27 blend x top_k candidates' VALIDATION-window daily
   returns by replaying the exact evaluation the original search ran
   (same score frames, same `baseline_protocol` target-weight rule
   [variant C: eligible ranking + delay-1], same deterministic strict engine,
   same cost config). This is a replay, not a new search: no new candidates,
   no selection, no window changes.
2. Fidelity-gates the replay against the recorded summary.json val CAGRs.
3. Computes overfit diagnostics on the T x 27 daily-return matrix:
   - CSCV-based Probability of Backtest Overfitting (PBO), S=8 and S=16
   - Deflated Sharpe Ratio for the adopted winner w=(1,1,0) k=10,
     with trial-count sensitivity (N = 27 / 54 / 100 / 200)
   - sub-period rank stability (halves, thirds; Spearman)
   - dispersion / degradation statistics
4. Writes PBO_DSR_RESULTS.csv (repo root) and small artifacts under
   runtime/reports/v89_closed_loop/pbo_dsr_retro/.

Memory: panel is column-pruned + date-filter read (~1.4M rows); peak RSS is
logged and expected well under 4 GiB (phase budget 16 GiB).
"""
from __future__ import annotations

import argparse
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

SEARCH_DIR = REPO / "runtime/reports/v89_closed_loop/ensemble_search_plus7"
OUT_DIR = REPO / "runtime/reports/v89_closed_loop/pbo_dsr_retro"
VAL_START = pd.Timestamp("2024-08-28")
VAL_END = pd.Timestamp("2025-08-31")
QUARANTINE_START = pd.Timestamp("2025-09-01")
ANN = 244
EULER_GAMMA = 0.5772156649015329
FIDELITY_TOL = 0.02  # |replayed - recorded| val CAGR tolerance (2pp)

PANEL_COLS = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
              "amount", "available_at", "is_suspended", "is_st", "is_limit_up",
              "is_limit_down"]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def cagr(r: np.ndarray) -> float:
    nav = float(np.prod(1.0 + r))
    return nav ** (ANN / len(r)) - 1.0 if nav > 0 else -1.0


def sharpe_daily(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s) if s > 0 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max())


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    # Acklam rational approximation, adequate for the quantiles used here.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        return -norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def replay_candidates(limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_returns TxN DataFrame, per-candidate stats DataFrame)."""
    summary = json.loads((SEARCH_DIR / "summary.json").read_text())
    cands = summary["all_results"][: limit or None]

    print(f"[{time.strftime('%H:%M:%S')}] loading panel (pruned, <= {VAL_END.date()}) ...", flush=True)
    panel = pd.read_parquet(
        REPO / bp.PANEL, columns=PANEL_COLS,
        filters=[("trade_date", ">=", VAL_START - pd.Timedelta(days=10)),
                 ("trade_date", "<=", VAL_END)],
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUARANTINE_START, "quarantine breach in panel read"
    sector = pd.read_parquet(REPO / bp.SECTOR)
    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    trade_dates = sorted(panel["trade_date"].unique())
    print(f"  panel rows {len(panel):,}  dates {len(trade_dates)}  RSS {rss_gib():.2f} GiB", flush=True)

    rets: dict[str, pd.Series] = {}
    stats_rows = []
    cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)
    for i, rec in enumerate(cands, 1):
        ws, wm, wl = rec["weights"]
        k = rec["top_k"]
        cid = f"w{ws}_{wm}_{wl}_k{k}"
        t0 = time.time()
        preds = pd.read_parquet(
            SEARCH_DIR / "_tmp" / f"{cid}.parquet",
            filters=[("trade_date", ">=", VAL_START), ("trade_date", "<=", VAL_END)],
        )
        preds["trade_date"] = pd.to_datetime(preds["trade_date"])
        assert preds["trade_date"].max() < QUARANTINE_START, "quarantine breach in preds read"
        preds = preds.rename(columns={"composite_score": "alpha_score"})
        preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")
        tw = bp._target_weights(preds, "alpha_score", k, eligible_only=True,
                                delay_days=1, trade_dates=trade_dates)
        res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
        nav = res.nav.copy()
        nav.index = pd.to_datetime(nav.index)
        r = nav.pct_change().dropna()
        assert np.isfinite(r.to_numpy()).all(), f"non-finite returns for {cid}"
        rets[cid] = r
        m = res.metrics
        rec_cagr = rec["val"]["cagr"]
        delta = m.annualized_return - rec_cagr
        stats_rows.append({
            "candidate_id": cid, "w_short": ws, "w_mid": wm, "w_long": wl, "top_k": k,
            "n_days": len(r), "replay_cagr": round(m.annualized_return, 4),
            "recorded_val_cagr": rec_cagr, "fidelity_delta": round(delta, 4),
            "replay_sharpe_ann": round(sharpe_daily(r.to_numpy()) * math.sqrt(ANN), 3),
            "replay_maxdd": round(max_drawdown(r.to_numpy()), 4),
            "turnover": round(float(m.turnover), 4),
            "is_production_winner": (ws, wm, wl, k) == (1, 1, 0, 10),
        })
        print(f"  [{i:2d}/{len(cands)}] {cid:16s} replay {m.annualized_return:+.1%} "
              f"(recorded {rec_cagr:+.1%}, d {delta:+.3f}) turn {m.turnover:.2f} "
              f"{time.time()-t0:.0f}s RSS {rss_gib():.2f} GiB", flush=True)

    matrix = pd.DataFrame(rets)
    matrix = matrix.dropna(how="any")
    return matrix, pd.DataFrame(stats_rows)


def cscv_pbo(matrix: pd.DataFrame, n_blocks: int) -> dict:
    """CSCV PBO (Bailey et al.): contiguous blocks, select-on-train by total
    log growth (rank-equivalent to CAGR on equal day counts)."""
    log1p = np.log1p(matrix.to_numpy())
    T, N = log1p.shape
    bounds = np.linspace(0, T, n_blocks + 1, dtype=int)
    blocks = np.array([log1p[bounds[i]:bounds[i + 1]].sum(axis=0) for i in range(n_blocks)])
    lambdas, oos_ranks = [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        mask = np.zeros(n_blocks, dtype=bool)
        mask[list(combo)] = True
        train = blocks[mask].sum(axis=0)
        test = blocks[~mask].sum(axis=0)
        w = int(np.argmax(train))
        rank = float((test <= test[w]).sum())          # 1 = worst ... N = best
        omega = rank / (N + 1)
        lambdas.append(math.log(omega / (1 - omega)))
        oos_ranks.append(rank)
    lam = np.array(lambdas)
    return {
        "n_blocks": n_blocks, "n_splits": len(lam),
        "pbo": round(float((lam <= 0).mean()), 4),
        "median_logit": round(float(np.median(lam)), 4),
        "mean_oos_rank_of_is_winner": round(float(np.mean(oos_ranks)), 2),
        "n_candidates": N,
    }


def dsr(matrix: pd.DataFrame, winner: str, n_trials: int) -> dict:
    r = matrix[winner].to_numpy()
    T = len(r)
    sr = sharpe_daily(r)                      # daily, non-annualised
    mu, sd = r.mean(), r.std(ddof=1)
    z = (r - mu) / sd
    g3 = float((z ** 3).mean())
    g4 = float((z ** 4).mean())               # raw kurtosis (normal = 3)
    all_sr = np.array([sharpe_daily(matrix[c].to_numpy()) for c in matrix.columns])
    v = float(all_sr.var(ddof=1))
    sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n_trials)
                          + EULER_GAMMA * norm_ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    stat = (sr - sr0) * math.sqrt(T - 1) / denom
    return {"n_trials": n_trials, "sr_daily": round(sr, 4),
            "sr_annualized": round(sr * math.sqrt(ANN), 3),
            "skew": round(g3, 3), "kurtosis_raw": round(g4, 3),
            "expected_max_sr_daily": round(sr0, 4), "dsr": round(norm_cdf(stat), 4)}


def rank_stability(matrix: pd.DataFrame, n_parts: int, winner: str) -> dict:
    T = len(matrix)
    bounds = np.linspace(0, T, n_parts + 1, dtype=int)
    part_cagr = []
    for i in range(n_parts):
        seg = matrix.iloc[bounds[i]:bounds[i + 1]]
        part_cagr.append(seg.apply(lambda s: cagr(s.to_numpy())))
    ranks = [p.rank() for p in part_cagr]
    rhos = []
    for i in range(n_parts):
        for j in range(i + 1, n_parts):
            rhos.append(float(ranks[i].corr(ranks[j], method="spearman")))
    return {
        "n_parts": n_parts,
        "spearman_between_parts": [round(x, 3) for x in rhos],
        "winner_rank_per_part": [int(r.rank(ascending=False)[winner]) for r in part_cagr],
        "winner_cagr_per_part": [round(float(p[winner]), 4) for p in part_cagr],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="smoke: replay only first N candidates")
    ap.add_argument("--skip-replay", action="store_true", help="reuse saved daily returns matrix")
    args = ap.parse_args()
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT_DIR / "candidate_val_daily_returns.csv"
    stats_path = OUT_DIR / "candidate_replay_stats.csv"

    if args.skip_replay and matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0, parse_dates=True)
        stats = pd.read_csv(stats_path)
    else:
        matrix, stats = replay_candidates(args.limit)
        matrix.to_csv(matrix_path)
        stats.to_csv(stats_path, index=False)
    print(f"matrix: {matrix.shape[0]} days x {matrix.shape[1]} candidates", flush=True)

    if args.limit:
        print("smoke mode: stopping before diagnostics")
        return 0

    winner = "w1_1_0_k10"
    fid_bad = stats[stats["fidelity_delta"].abs() > FIDELITY_TOL]
    fidelity_ok = fid_bad.empty

    diag: dict[str, object] = {
        "val_window": f"{matrix.index.min().date()}..{matrix.index.max().date()}",
        "n_days": int(len(matrix)),
        "fidelity_ok": bool(fidelity_ok),
        "fidelity_max_abs_delta": float(stats["fidelity_delta"].abs().max()),
        "mean_pairwise_corr": round(float(matrix.corr().to_numpy()[np.triu_indices(matrix.shape[1], 1)].mean()), 3),
        "cagr_best": round(float(stats["replay_cagr"].max()), 4),
        "cagr_median": round(float(stats["replay_cagr"].median()), 4),
        "cagr_degradation_best_to_median": round(float(stats["replay_cagr"].max() - stats["replay_cagr"].median()), 4),
        "pbo_s8": cscv_pbo(matrix, 8),
        "pbo_s16": cscv_pbo(matrix, 16),
        "dsr_sensitivity": [dsr(matrix, winner, n) for n in (27, 54, 100, 200)],
        "rank_stability_halves": rank_stability(matrix, 2, winner),
        "rank_stability_thirds": rank_stability(matrix, 3, winner),
        "peak_rss_gib": round(rss_gib(), 2),
        "runtime_sec": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # flat results csv at repo root
    rows = [("candidate", r["candidate_id"],
             f"cagr={r['replay_cagr']} sharpe={r['replay_sharpe_ann']} maxdd={r['replay_maxdd']} "
             f"turnover={r['turnover']} fidelity_delta={r['fidelity_delta']}")
            for r in stats.to_dict("records")]
    for key in ("pbo_s8", "pbo_s16"):
        for kk, vv in diag[key].items():
            rows.append((key, kk, vv))
    for d in diag["dsr_sensitivity"]:
        rows.append(("dsr", f"N={d['n_trials']}", f"dsr={d['dsr']} sr_ann={d['sr_annualized']} "
                     f"exp_max_sr_daily={d['expected_max_sr_daily']}"))
    for key in ("rank_stability_halves", "rank_stability_thirds"):
        rows.append((key, "spearman", str(diag[key]["spearman_between_parts"])))
        rows.append((key, "winner_rank_per_part", str(diag[key]["winner_rank_per_part"])))
        rows.append((key, "winner_cagr_per_part", str(diag[key]["winner_cagr_per_part"])))
    for kk in ("val_window", "n_days", "fidelity_ok", "fidelity_max_abs_delta",
               "mean_pairwise_corr", "cagr_best", "cagr_median",
               "cagr_degradation_best_to_median", "peak_rss_gib", "runtime_sec"):
        rows.append(("global", kk, diag[kk]))
    pd.DataFrame(rows, columns=["section", "name", "value"]).to_csv(
        REPO / "PBO_DSR_RESULTS.csv", index=False)

    print(json.dumps(diag, indent=2), flush=True)
    print(f"wrote {OUT_DIR/'diagnostics.json'} and PBO_DSR_RESULTS.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
