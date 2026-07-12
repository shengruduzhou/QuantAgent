#!/usr/bin/env python3
"""H-023 / EXP-023: learned regime->tilt-weight meta-model (CPU).

Replaces the hand-designed tilt sequence (EXP-016..019: factor, weight and
regime structure all chosen by a human who had seen prior fold results) with a
fully causal learner: at each monthly refit date t inside a fold, per-regime
component weights and the tilt fraction tau_s are derived ONLY from trailing
data (2018-01-02 .. t-11 trading days, embargo >= label horizon 10d).

  blend_t = (1 - tau_s(t)) * carrier_rank + tau_s(t) * tilt_rank_s(t)

Components (frozen): D1 low-vol / quality / sector_rs (reused from the
dual-track tilt harness). Momentum proxy rank-mean(ret5, ret20, ret60) is used
only to scale tau, never inside the tilt.  tau_s = 0.5*IC_tilt+/(IC_tilt+ +
IC_mom+ + 0.01), cap 0.5 inherited from EXP-019 (spent dof, not searched).

Candidates (pre-registered, frozen):
  RW1_4state  regime = R2a trend (2) x bench 20d ann. vol >= 0.25 (2)
  RW2_2state  regime = R2a trend only (ablation: does the vol split add value)

Carrier / book / folds / sim / bps identical to EXP-016..019 (corrected sim,
min-hold-10 top-10, H-008 folds, 8/15/25 bps). Zero retrain, zero fresh-window
contact: the component/label panel is hard-capped before the 2025-09-01
quarantine, so forward-10d labels never read quarantined prices.
"""
from __future__ import annotations

import json
import sys
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import FOLDS, build_candidates, sleeve_frame, cagr, max_dd  # noqa: E402
from exp011_book_churn import eligible_rank_lists  # noqa: E402
from exp010_hysteresis_overlay import gross_series  # noqa: E402
from dual_track_eval import build_book, avg_holding_days  # noqa: E402
from dual_track_d1_integration import tilt_series  # noqa: E402
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

QUAR = pd.Timestamp("2025-09-01")
BPS = (8.0, 15.0, 25.0)
OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp023_regime_weight_meta"
COMPONENTS = ("d1", "quality", "sector_rs")
# frozen protocol constants (HYPOTHESIS_REGISTRY.md H-023; none searched)
TRAIL_START = pd.Timestamp("2018-01-02")
PANEL_CAP = pd.Timestamp("2025-08-29")   # hard cap: labels never touch quarantine
HORIZON = 10                             # label horizon, matches min-hold-10 book
EMBARGO = 11                             # trading days, >= HORIZON + 1
REFIT_EVERY = 21                         # trading days (~monthly)
MIN_REGIME_DAYS = 60                     # else fall back to unconditional ICs
IC_MIN = 0.01                            # component enters tilt only above this
TAU_CAP = 0.5                            # inherited from EXP-019 (spent dof)
VOL_TH = 0.25                            # ann. vol threshold, existing R3 rule
ANN = 244


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def per_date_spearman(df: pd.DataFrame, xcol: str, ycol: str) -> pd.Series:
    """Daily cross-sectional Spearman IC of df[xcol] vs df[ycol]."""
    d = df[["trade_date", xcol, ycol]].dropna()
    rx = d.groupby("trade_date")[xcol].rank(pct=True)
    ry = d.groupby("trade_date")[ycol].rank(pct=True)
    t = pd.DataFrame({"td": d["trade_date"].to_numpy(),
                      "x": rx.to_numpy(), "y": ry.to_numpy()})
    t["xy"] = t["x"] * t["y"]
    t["x2"] = t["x"] ** 2
    t["y2"] = t["y"] ** 2
    g = t.groupby("td").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                            sxy=("xy", "sum"), sx2=("x2", "sum"), sy2=("y2", "sum"))
    g = g[g["n"] >= 50]
    cov = g["sxy"] / g["n"] - (g["sx"] / g["n"]) * (g["sy"] / g["n"])
    vx = g["sx2"] / g["n"] - (g["sx"] / g["n"]) ** 2
    vy = g["sy2"] / g["n"] - (g["sy"] / g["n"]) ** 2
    return cov / np.sqrt(vx * vy)


def build_ic_and_regimes() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Trailing component-IC panel + regime states over 2018..PANEL_CAP.

    Returns (ic_df[date x component ICs], regime_df[date -> crash, vol_hi],
    all trade dates array for trading-day arithmetic).
    """
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
    pan = pd.read_parquet(REPO / bp.PANEL, columns=pcols,
                          filters=[("trade_date", ">=", TRAIL_START - pd.Timedelta(days=180)),
                                   ("trade_date", "<=", PANEL_CAP)])
    pan["trade_date"] = pd.to_datetime(pan["trade_date"])
    assert pan["trade_date"].max() <= PANEL_CAP < QUAR, "quarantine breach in IC panel"
    pan = pan.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = pan.groupby("symbol", sort=False)["close"]
    for w in (5, 20, 60):
        pan[f"ret{w}"] = g.pct_change(w)
    pan["fwd"] = g.shift(-HORIZON) / pan["close"] - 1.0

    # momentum proxy = per-date rank-mean of ret5/ret20/ret60
    for w in (5, 20, 60):
        pan[f"_r{w}"] = pan.groupby("trade_date")[f"ret{w}"].rank(pct=True)
    pan["mom"] = pan[["_r5", "_r20", "_r60"]].mean(axis=1)
    pan.drop(columns=["_r5", "_r20", "_r60", "ret5", "ret20", "ret60"], inplace=True)

    # components via the SAME tilt functions used at fold time (identical
    # definitions and PIT lags as EXP-016..019)
    for comp in COMPONENTS:
        pan[comp] = tilt_series(pan[pcols], comp)["tilt"].to_numpy()

    ic = pd.DataFrame({c: per_date_spearman(pan, c, "fwd")
                       for c in (*COMPONENTS, "mom")})
    ic = ic[ic.index >= TRAIL_START].dropna(how="all")

    # regimes from the eqw-all-A bench over the full range (t-1 -> t shift:
    # gross_series shifts internally; the vol state is shifted here)
    all_dates = np.sort(pan["trade_date"].unique())
    bench = bp._bench_daily(pan[["symbol", "trade_date", "close"]], list(all_dates))
    crash = (gross_series(bench, "R2a_confirm5") < 1.0)
    vol_hi = (bench.rolling(20, min_periods=15).std() * np.sqrt(ANN) >= VOL_TH).shift(1).fillna(False)
    regime = pd.DataFrame({"crash": crash.astype(bool), "vol_hi": vol_hi.astype(bool)})
    return ic, regime, all_dates


def learn_state_params(ic_df: pd.DataFrame, regime_ids: pd.Series,
                       cutoff: pd.Timestamp, n_states: int) -> dict[int, dict]:
    """Per-regime tilt weights + tau from trailing ICs (dates <= cutoff only)."""
    trail = ic_df[ic_df.index <= cutoff]
    rid = regime_ids.reindex(trail.index)
    out: dict[int, dict] = {}
    uncond = trail.mean()
    for st in range(n_states):
        sub = trail[rid == st]
        mean_ic = sub.mean() if len(sub) >= MIN_REGIME_DAYS else uncond
        pos = {c: float(mean_ic[c]) for c in COMPONENTS if mean_ic[c] >= IC_MIN}
        if not pos:
            out[st] = {"tau": 0.0, "weights": {}, "n_days": int(len(sub)),
                       "fallback": bool(len(sub) < MIN_REGIME_DAYS)}
            continue
        tot = sum(pos.values())
        weights = {c: v / tot for c, v in pos.items()}
        ic_tilt = max(0.0, sum(weights[c] * pos[c] for c in pos))
        ic_mom = max(0.0, float(mean_ic["mom"]))
        tau = min(TAU_CAP, TAU_CAP * ic_tilt / (ic_tilt + ic_mom + 0.01))
        out[st] = {"tau": round(tau, 4), "weights": {c: round(w, 4) for c, w in weights.items()},
                   "n_days": int(len(sub)), "fallback": bool(len(sub) < MIN_REGIME_DAYS)}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    ap.add_argument("--bps", default="8,15,25")
    args = ap.parse_args()
    run_folds = args.folds.split(",")
    run_bps = tuple(float(b) for b in args.bps.split(","))
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("building trailing IC panel + regimes (2018..cap)...", flush=True)
    ic_df, regime_df, all_dates = build_ic_and_regimes()
    print(f"  IC panel {ic_df.index.min().date()}..{ic_df.index.max().date()} "
          f"({len(ic_df)}d)  crash days {int(regime_df['crash'].sum())}  "
          f"vol_hi days {int(regime_df['vol_hi'].sum())}  RSS {rss_gib():.1f}G", flush=True)

    cand_states = {"RW1_4state": 4, "RW2_2state": 2}
    rid4 = (regime_df["crash"].astype(int) * 2 + regime_df["vol_hi"].astype(int))
    rid2 = regime_df["crash"].astype(int)
    regime_ids = {"RW1_4state": rid4, "RW2_2state": rid2}

    sector = pd.read_parquet(REPO / bp.SECTOR)
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
             "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfgs = {b: AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=b) for b in run_bps}
    res: dict[str, dict] = {k: {} for k in cand_states}
    weight_trace: dict[str, dict] = {k: {} for k in cand_states}

    for fold in run_folds:
        spec = FOLDS[fold]
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUAR
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)["C3_ema0.7"].copy()
        panel = pd.read_parquet(REPO / bp.PANEL, columns=pcols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=210)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        pan = panel.sort_values(["symbol", "trade_date"]).copy()
        comp = pan[["symbol", "trade_date"]].copy()
        for c in COMPONENTS:
            comp[c] = tilt_series(pan, c)["tilt"].to_numpy()
        bt_panel = panel[panel["trade_date"] >= oos_s - pd.Timedelta(days=10)].copy()
        flags = bt_panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(bt_panel["trade_date"].unique())

        c = carrier.merge(comp, on=["symbol", "trade_date"], how="left")
        c["rc"] = c.groupby("trade_date")["alpha_score"].rank(pct=True)
        for k in COMPONENTS:
            c[f"r_{k}"] = c.groupby("trade_date")[k].rank(pct=True)

        oos_dates = [d for d in sorted(c["trade_date"].unique()) if d >= oos_s]
        refit_pos = range(0, len(oos_dates), REFIT_EVERY)

        for name, n_states in cand_states.items():
            rid = regime_ids[name]
            # per-date (tau, weights) from the refit schedule — strictly causal
            tau_by_date: dict[pd.Timestamp, float] = {}
            w_by_date: dict[pd.Timestamp, dict[str, float]] = {}
            trace = {}
            for pi in refit_pos:
                t = oos_dates[pi]
                tpos = int(np.searchsorted(all_dates, np.datetime64(t)))
                assert tpos >= EMBARGO, "refit before trailing history exists"
                cutoff = pd.Timestamp(all_dates[tpos - EMBARGO])
                params = learn_state_params(ic_df, rid, cutoff, n_states)
                trace[str(t.date())] = {"cutoff": str(cutoff.date()),
                                        "states": {str(s): p for s, p in params.items()}}
                for d in oos_dates[pi:pi + REFIT_EVERY]:
                    st = int(rid.get(d, 0))
                    p = params.get(st, {"tau": 0.0, "weights": {}})
                    tau_by_date[d] = p["tau"]
                    w_by_date[d] = p["weights"]
            weight_trace[name][fold] = trace

            cc = c.copy()
            tau_col = cc["trade_date"].map(tau_by_date).fillna(0.0).to_numpy()
            # weighted tilt rank; missing component values fall back to the
            # carrier rank so the book universe is unchanged (same as EXP-016..19)
            tilt = np.zeros(len(cc))
            for k in COMPONENTS:
                wk = cc["trade_date"].map(lambda d, k=k: w_by_date.get(d, {}).get(k, 0.0)).to_numpy()
                tilt = tilt + wk * cc[f"r_{k}"].fillna(cc["rc"]).to_numpy()
            cc["blend"] = (1 - tau_col) * cc["rc"].to_numpy() + tau_col * tilt
            score = cc[["trade_date", "symbol"]].copy()
            score["alpha_score"] = cc["blend"].to_numpy()
            p = score.merge(flags, on=["symbol", "trade_date"], how="left")
            days = eligible_rank_lists(p)
            book = build_book(days, "minhold", {"n": 10})
            tw = bp._apply_delay(book, trade_dates, 1)
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all()
            row = {"hold_days": avg_holding_days(book),
                   "mean_tau": round(float(np.mean([tau_by_date[d] for d in oos_dates])), 4)}
            for b in run_bps:
                r = run_strict_backtest_v8(tw, bt_panel, sector_map=sector, config=cfgs[b])
                nav = r.nav.copy(); nav.index = pd.to_datetime(nav.index)
                rr = nav.pct_change().dropna().to_numpy()
                row[f"cagr{int(b)}"] = round(cagr(rr), 4)
                if b == 8.0:
                    row["maxdd"] = round(max_dd(rr), 4)
                    row["turnover"] = round(float(r.metrics.turnover), 4)
            res[name][fold] = row
            print(f"{fold} {name:12s} CAGR8 {row.get('cagr8', float('nan')):+.1%} "
                  f"25 {row.get('cagr25', float('nan')):+.1%} DD {row.get('maxdd', float('nan')):.1%} "
                  f"turn {row.get('turnover', float('nan')):.3f} mean_tau {row['mean_tau']:.3f}", flush=True)

    summary = {"hypothesis": "H-023", "candidates": list(cand_states),
               "frozen": {"trail_start": str(TRAIL_START.date()), "horizon": HORIZON,
                          "embargo": EMBARGO, "refit_every": REFIT_EVERY,
                          "min_regime_days": MIN_REGIME_DAYS, "ic_min": IC_MIN,
                          "tau_cap": TAU_CAP, "vol_th": VOL_TH},
               "comparators_spent_dof": {
                   "L1_baseline": {"median_cagr8": 0.364, "worst_dd8": 0.366, "calmar": 0.99},
                   "EXP019_d1_regime": {"median_cagr8": 0.253, "worst_dd8": 0.221, "calmar": 1.14}},
               "per_fold": res, "weight_trace": weight_trace, "agg": {}}
    folds_run = [f for f in run_folds if f in next(iter(res.values()))]
    if len(folds_run) == len(FOLDS):
        for name in cand_states:
            cs = [res[name][f]["cagr8"] for f in folds_run]
            wdd = max(res[name][f]["maxdd"] for f in folds_run)
            med = float(np.median(cs))
            summary["agg"][name] = {
                "median_cagr8": round(med, 4), "worst_fold8": round(min(cs), 4),
                "f2_cagr8": round(res[name]["F2"]["cagr8"], 4),
                "median_cagr25": round(float(np.median([res[name][f]["cagr25"] for f in folds_run])), 4),
                "worst_dd8": round(wdd, 4), "calmar": round(med / wdd, 3) if wdd else None,
                "max_turnover": round(max(res[name][f]["turnover"] for f in folds_run), 4),
            }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    if summary["agg"]:
        print("\n=== EXP-023 aggregates (gates: A median>36.4% & DD<=36.6%; "
              "B calmar>1.14 & median>=25.3%) ===")
        for name, a in summary["agg"].items():
            print(f"  {name:12s} med {a['median_cagr8']:+.1%} DD {a['worst_dd8']:.1%} "
                  f"calmar {a['calmar']} F2 {a['f2_cagr8']:+.1%} med@25 {a['median_cagr25']:+.1%} "
                  f"turn {a['max_turnover']:.3f}")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
