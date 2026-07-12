#!/usr/bin/env python3
"""H-015 dual-track (L low-turnover vs H high-turnover) governed comparison.

Common harness (DUAL_TURNOVER_STRATEGY_PROTOCOL.md). Frozen candidates
(HYPOTHESIS_REGISTRY.md H-015). Corrected simulator, strict variant-C, H-008
folds, 8/15/25 bps, net metrics decide. Zero retrain, zero fresh-holdout.
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
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import (  # noqa: E402
    FOLDS, TOP_K, build_candidates, norm_cdf, norm_ppf, sleeve_frame,
)
from exp009_exposure_overlay import bench_series  # noqa: E402
from exp011_book_churn import eligible_rank_lists  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp015_dual_track"
QUARANTINE_START = pd.Timestamp("2025-09-01")
EULER = 0.5772156649015329
ANN = 244
N_TRIALS = 73
BPS = (8.0, 15.0, 25.0)
CARRIER_REF = "C3_ema0.7"

# (id, track, carrier_spec, rule, params) -- carrier_spec is a build_candidates
# key OR dict{sleeves, ema}. FROZEN (registry H-015).
CANDIDATES = [
    ("L1_c3ema07_minhold10", "L", "C3_ema0.7", "minhold", {"n": 10}),
    ("L2_midlong_ema07", "L", {"sleeves": ["mid_5d_30d", "long_30d_120d"], "ema": 0.7}, "plain", {}),
    ("L3_midlong_minhold10", "L", {"sleeves": ["mid_5d_30d", "long_30d_120d"], "ema": 0.7}, "minhold", {"n": 10}),
    ("L4_c3ema07_reb10", "L", "C3_ema0.7", "reb", {"n": 10}),
    ("H1_short_fast", "H", {"sleeves": ["short_5d"], "ema": None}, "plain", {}),
    ("H2_short_hyst", "H", {"sleeves": ["short_5d"], "ema": None}, "keepzone", {"buffer_mult": 2}),
    ("H3_c2_fast", "H", "C2_prod_rank110", "plain", {}),
    ("H4_short_minhold3", "H", {"sleeves": ["short_5d"], "ema": None}, "minhold", {"n": 3}),
]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def cagr(r: np.ndarray) -> float:
    nav = float(np.prod(1.0 + r))
    return nav ** (ANN / len(r)) - 1.0 if len(r) and nav > 0 else -1.0


def max_dd(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(((peak - nav) / peak).max()) if len(r) else 0.0


def sortino(r: np.ndarray) -> float:
    dn = r[r < 0]
    dstd = dn.std(ddof=1) if len(dn) > 1 else np.nan
    return float(r.mean() / dstd * math.sqrt(ANN)) if dstd and dstd > 0 else float("nan")


def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * math.sqrt(ANN)) if s > 0 else float("nan")


def make_carrier(frame: pd.DataFrame, spec, oos_s: pd.Timestamp) -> pd.DataFrame:
    """Custom carrier = rank-mean of chosen sleeves, optional per-symbol EMA."""
    sleeves, ema = spec["sleeves"], spec["ema"]
    ranks = [frame.groupby("trade_date")[f"{sl}_score"].rank(pct=True).to_numpy() for sl in sleeves]
    f = frame[["trade_date", "symbol"]].copy()
    f["alpha_score"] = np.mean(np.column_stack(ranks), axis=1)
    if ema:
        f = f.sort_values(["symbol", "trade_date"]).copy()
        f["alpha_score"] = f.groupby("symbol")["alpha_score"].transform(
            lambda s: s.ewm(alpha=ema, adjust=False).mean())
    return f[f["trade_date"] >= oos_s].reset_index(drop=True)


def build_book(days: dict[pd.Timestamp, list[str]], rule: str, params: dict, k: int = TOP_K) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    held: dict[str, float] = {}
    ages: dict[str, int] = {}
    for i, (ts, order) in enumerate(days.items()):
        rank = {s: j for j, s in enumerate(order)}
        if rule == "plain":
            book = order[:k]
            held = {s: 1.0 / k for s in book}
        elif rule == "minhold":
            n = params["n"]
            locked = [s for s in held if ages.get(s, 99) < n and s in rank]
            fills = [s for s in order if s not in locked][: k - len(locked)]
            book = locked + fills
            ages = {s: ages.get(s, 0) + 1 for s in book}
            held = {s: 1.0 / k for s in book}
        elif rule == "keepzone":
            bm = params["buffer_mult"]
            keep = [s for s in held if rank.get(s, 1 << 30) < bm * k]
            book = keep + [s for s in order if s not in held][: k - len(keep)]
            held = {s: 1.0 / k for s in book}
        elif rule == "reb":
            n = params["n"]
            if i % n == 0 or not held:
                held = {s: 1.0 / k for s in order[:k]}
        else:
            raise ValueError(rule)
        assert len(held) <= 500 and abs(sum(held.values())) <= 1.0 + 1e-9
        rows[ts] = dict(held)
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    tw.index.name = "trade_date"
    return tw


def avg_holding_days(book: pd.DataFrame) -> float:
    arr = book.to_numpy() > 0
    eps, tot = 0, 0
    for j in range(arr.shape[1]):
        idx = np.where(arr[:, j])[0]
        if len(idx) == 0:
            continue
        for run in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            eps += 1
            tot += len(run)
    return round(tot / eps, 2) if eps else 0.0


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    smap = dict(zip(sector["symbol"].astype(str), sector.iloc[:, 1].astype(str))) if len(sector.columns) > 1 else {}
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfgs = {b: AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=b) for b in BPS}

    # per candidate/fold/bps metrics; daily returns at 8bps for PBO/DSR
    met: dict[str, dict] = {c[0]: {} for c in CANDIDATES}
    daily8: dict[str, dict[str, pd.Series]] = {c[0]: {} for c in CANDIDATES}
    bench_ann: dict[str, float] = {}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START, f"{fold} quarantine breach"
        frame = sleeve_frame(fold)
        std_cands = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        bench = bench_series(oos_s, oos_e)
        n_b = len(bench.loc[oos_s:oos_e])
        bench_ann[fold] = float((1 + bench.loc[oos_s:oos_e]).prod() ** (ANN / max(1, n_b)) - 1)

        carrier_cache: dict[str, pd.DataFrame] = {}
        for cid, track, cspec, rule, params in CANDIDATES:
            key = cspec if isinstance(cspec, str) else json.dumps(cspec, sort_keys=True)
            if key not in carrier_cache:
                carrier_cache[key] = (std_cands[cspec] if isinstance(cspec, str)
                                      else make_carrier(frame[frame["trade_date"] <= oos_e], cspec, oos_s))
            carrier = carrier_cache[key]
            p = carrier.merge(flags, on=["symbol", "trade_date"], how="left")
            days = eligible_rank_lists(p)
            book = build_book(days, rule, params)
            tw = bp._apply_delay(book, trade_dates, 1)
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all(), f"{cid} leverage breach"
            hold_days = avg_holding_days(book)
            tw_long = tw.stack()
            tw_long = tw_long[tw_long > 0].rename("w").reset_index()
            tw_long.columns = ["trade_date", "symbol", "w"]
            tw_long["sec"] = tw_long["symbol"].astype(str).map(smap).fillna("?")
            sec_max = float(tw_long.groupby(["trade_date", "sec"])["w"].sum()
                            .groupby("trade_date").max().mean()) if len(tw_long) else 0.0
            per_bps = {}
            for b in BPS:
                res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfgs[b])
                nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
                r = nav.pct_change().dropna()
                rr = r.to_numpy()
                per_bps[b] = {"cagr": round(cagr(rr), 4), "maxdd": round(max_dd(rr), 4),
                              "sharpe": round(sharpe(rr), 3), "sortino": round(sortino(rr), 3),
                              "turnover": round(float(res.metrics.turnover), 4)}
                if b == 8.0:
                    daily8[cid][fold] = r
            met[cid][fold] = {**per_bps[8.0], "sortino8": per_bps[8.0]["sortino"],
                              "hold_days": hold_days, "sec_max": round(sec_max, 3),
                              "cagr15": per_bps[15.0]["cagr"], "cagr25": per_bps[25.0]["cagr"],
                              "dd25": per_bps[25.0]["maxdd"]}
            print(f"{fold} {cid:24s}[{track}] CAGR8 {per_bps[8.0]['cagr']:+.1%} "
                  f"15 {per_bps[15.0]['cagr']:+.1%} 25 {per_bps[25.0]['cagr']:+.1%} "
                  f"DD {per_bps[8.0]['maxdd']:.1%} turn {per_bps[8.0]['turnover']:.3f} hold {hold_days:.1f}d",
                  flush=True)

    # corrected carrier daily from EXP-008 stitched (reference in PBO/DSR)
    stitched = pd.read_csv(REPO / "runtime/reports/v89_closed_loop/wf_h008/stitched_daily_returns.csv",
                           index_col=0, parse_dates=True)
    carrier_daily = {f: stitched[CARRIER_REF].loc[pd.Timestamp(s["oos"][0]):pd.Timestamp(s["oos"][1])].dropna()
                     for f, s in FOLDS.items()}
    names = [c[0] for c in CANDIDATES] + [CARRIER_REF]
    daily_all = {**daily8, CARRIER_REF: carrier_daily}
    folds = list(FOLDS)

    # fold-block CSCV PBO
    growth = np.array([[float(np.log1p(daily_all[n][f]).sum()) for n in names] for f in folds])
    lam = []
    for combo in itertools.combinations(range(len(folds)), 2):
        mask = np.zeros(len(folds), dtype=bool); mask[list(combo)] = True
        tr, te = growth[mask].sum(0), growth[~mask].sum(0)
        w = int(np.argmax(tr))
        omega = float((te <= te[w]).sum()) / (len(names) + 1)
        lam.append(math.log(omega / (1 - omega)))
    pbo = round(float((np.array(lam) <= 0).mean()), 3)

    def dsr(name: str) -> float:
        r = pd.concat([daily_all[name][f] for f in folds]).to_numpy()
        sr = float(r.mean() / r.std(ddof=1))
        z = (r - r.mean()) / r.std(ddof=1)
        g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
        srs = [float(pd.concat([daily_all[n][f] for f in folds]).to_numpy().mean()
                     / pd.concat([daily_all[n][f] for f in folds]).to_numpy().std(ddof=1)) for n in names]
        v = float(np.var(srs, ddof=1))
        sr0 = math.sqrt(v) * ((1 - EULER) * norm_ppf(1 - 1 / N_TRIALS)
                              + EULER * norm_ppf(1 - 1 / (N_TRIALS * math.e)))
        denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        return round(norm_cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)

    dsrs = {n: dsr(n) for n in names}

    # aggregate per candidate
    agg: dict[str, dict] = {}
    for cid, track, *_ in CANDIDATES:
        cs = [met[cid][f]["cagr"] for f in folds]
        dd = [met[cid][f]["maxdd"] for f in folds]
        to = [met[cid][f]["turnover"] for f in folds]
        c15 = [met[cid][f]["cagr15"] for f in folds]
        c25 = [met[cid][f]["cagr25"] for f in folds]
        hd = [met[cid][f]["hold_days"] for f in folds]
        sec = [met[cid][f]["sec_max"] for f in folds]
        sh = [met[cid][f]["sharpe"] for f in folds]
        so = [met[cid][f]["sortino8"] for f in folds]
        med = float(np.median(cs))
        excess = [cs[i] - bench_ann[f] for i, f in enumerate(folds)]
        agg[cid] = {
            "track": track, "fold_cagrs8": cs, "median_cagr8": round(med, 4),
            "worst_fold8": round(min(cs), 4), "f2_cagr8": round(met[cid]["F2"]["cagr"], 4),
            "median_cagr15": round(float(np.median(c15)), 4), "median_cagr25": round(float(np.median(c25)), 4),
            "cost_drag_8_25": round(med - float(np.median(c25)), 4),
            "worst_dd8": round(max(dd), 4), "calmar": round(med / max(dd) if max(dd) > 0 else 0, 3),
            "max_turnover": round(max(to), 4), "avg_hold_days": round(float(np.mean(hd)), 2),
            "median_excess": round(float(np.median(excess)), 4),
            "sharpe_med": round(float(np.median(sh)), 3), "sortino_med": round(float(np.median(so)), 3),
            "max_sector": round(max(sec), 3), "dsr": dsrs[cid],
        }

    car = json.loads((REPO / "runtime/reports/v89_closed_loop/wf_h008/wf_summary.json").read_text())["aggregates"][CARRIER_REF]
    carrier_row = {"track": "REF", "median_cagr8": car["median_cagr"], "worst_fold8": car["min_cagr"],
                   "f2_cagr8": car["fold_cagrs"][1], "worst_dd8": car["worst_maxdd"],
                   "max_turnover": car["max_turnover"], "dsr": dsrs[CARRIER_REF]}

    summary = {"protocol": "DUAL_TURNOVER_STRATEGY_PROTOCOL.md", "hypothesis": "H-015",
               "cumulative_trials_N": N_TRIALS, "bench_ann": {f: round(bench_ann[f], 4) for f in folds},
               "carrier_ref": carrier_row, "pbo": pbo, "dsr": dsrs, "candidates": agg,
               "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [{"candidate": cid, **agg[cid]} for cid, *_ in CANDIDATES]
    pd.DataFrame(rows).to_csv(OUT / "dual_track_metrics.csv", index=False)
    print("\n=== PBO", pbo, "| N", N_TRIALS, "| carrier DSR", dsrs[CARRIER_REF], "===")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
