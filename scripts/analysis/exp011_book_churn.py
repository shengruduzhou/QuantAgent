#!/usr/bin/env python3
"""EXP-011 / H-011: book-construction churn control (Track A, first batch).

Pre-registered candidates (BOOK_CHURN_CONTROL_EXPERIMENT.md, frozen at commit
1994cd4 BEFORE any evaluation run — no additions, no post-hoc tuning):
  B1_buffer30        enter top-10, retain while eligible rank < 30
  B2_minhold10       10-trading-day minimum hold (slot lock), age=1 on entry
  B3_partial30       w_t = 0.7*w_{t-1} + 0.3*target_t, prune <0.005, renorm
  B4_reb5d           recompute top-10 every 5 trading days, hold in between
  B5_buffer_r2a_ramp B1 book x R2a confirm-5 MA60 gross{1.0,0.5}, ramp 0.1/day

Carrier: C3_ema0.7 scores rebuilt exactly as EXP-008/009/010. Eligibility and
delay-1 semantics identical to variant-C `_target_weights` (same force-out).
Evaluation: strict variant C on the four H-008 folds, 8bps + 15bps sensitivity.
Statistics: fold-block CSCV PBO across {B1..B5 + carrier}, DSR at N=60.
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
from exp009_exposure_overlay import CARRIER, bench_series  # noqa: E402
from exp010_hysteresis_overlay import gross_series  # noqa: E402

from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008/exp011_book_churn"
QUARANTINE_START = pd.Timestamp("2025-09-01")
EULER = 0.5772156649015329
RULES = ("B1_buffer30", "B2_minhold10", "B3_partial30", "B4_reb5d", "B5_buffer_r2a_ramp")
N_TRIALS = 60  # cumulative pre-registered trial count (55 prior + 5 here)

BASE = {  # frozen EXP-008 C3_ema0.7 baselines (wf_summary.json)
    "worst_dd": 0.2503, "f2_floor": -0.249, "turn_cap": 0.10,
    "median_floor": 0.2802, "sector_cap": 0.33,
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def eligible_rank_lists(p: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    """Per-day symbols in eligible rank order — filter+sort identical to
    `_target_weights(eligible_only=True)` so tie-handling matches the carrier."""
    bad = (
        p.get("is_suspended", pd.Series(False, index=p.index)).fillna(False).astype(bool)
        | p.get("is_st", pd.Series(False, index=p.index)).fillna(False).astype(bool)
        | p.get("is_limit_up", pd.Series(False, index=p.index)).fillna(False).astype(bool)
    )
    d = p[~bad].sort_values(["trade_date", "alpha_score"], ascending=[True, False])
    return {ts: g["symbol"].tolist() for ts, g in d.groupby("trade_date", sort=True)}


def build_book(days: dict[pd.Timestamp, list[str]], rule: str, k: int = TOP_K) -> pd.DataFrame:
    """Iterative book construction -> weights indexed by score date."""
    prune = 0.05 / k  # 0.005 at k=10; same 5%-of-slot proportion at other k
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    held: dict[str, float] = {}
    ages: dict[str, int] = {}
    for i, (ts, order) in enumerate(days.items()):
        rank = {s: j for j, s in enumerate(order)}
        if rule == "B1_buffer30":
            keep = [s for s in held if rank.get(s, 1 << 30) < 3 * k]
            book = keep + [s for s in order if s not in held][: k - len(keep)]
            held = {s: 1.0 / k for s in book}
        elif rule == "B2_minhold10":
            locked = [s for s in held if ages.get(s, 99) < 10 and s in rank]
            free = k - len(locked)
            fills = [s for s in order if s not in locked][:free]
            book = locked + fills
            ages = {s: ages.get(s, 0) + 1 for s in book}
            held = {s: 1.0 / k for s in book}
        elif rule == "B3_partial30":
            target = {s: 1.0 / k for s in order[:k]}
            if not held:
                held = target
            else:
                nw = {s: 0.7 * w for s, w in held.items()}
                for s, w in target.items():
                    nw[s] = nw.get(s, 0.0) + 0.3 * w
                nw = {s: w for s, w in nw.items() if w >= prune}
                tot = sum(nw.values())
                held = {s: w / tot for s, w in nw.items()} if tot > 0 else target
        elif rule == "B4_reb5d":
            if i % 5 == 0 or not held:
                held = {s: 1.0 / k for s in order[:k]}
        else:
            raise ValueError(rule)
        # anti-runaway guard only — B3's decay tail has no registered size cap
        assert len(held) <= 500 and abs(sum(held.values())) <= 1.0 + 1e-9
        assert all(w >= 0 for w in held.values())
        rows[ts] = dict(held)
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    tw.index.name = "trade_date"
    return tw


def ramped_gross(state: pd.Series) -> pd.Series:
    """Move gross toward the (already t-1 shifted) R2a state by at most 0.1/day."""
    g = np.empty(len(state))
    cur = 1.0
    tgt = state.to_numpy()
    for i in range(len(state)):
        cur += float(np.clip(tgt[i] - cur, -0.1, 0.1))
        g[i] = cur
    out = pd.Series(g, index=state.index)
    assert out.between(0.5 - 1e-9, 1.0 + 1e-9).all(), "gross out of [0.5, 1.0]"
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sector = pd.read_parquet(REPO / bp.SECTOR)
    smap = dict(zip(sector["symbol"].astype(str), sector.iloc[:, 1].astype(str))) if len(sector.columns) > 1 else {}
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    cfg8 = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0)
    cfg15 = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=15.0)

    results: dict[str, dict] = {r: {} for r in RULES}
    sens15: dict[str, dict] = {r: {} for r in RULES}
    daily: dict[str, dict[str, pd.Series]] = {r: {} for r in RULES}

    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        assert oos_e < QUARANTINE_START
        frame = sleeve_frame(fold)
        carrier = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)[CARRIER]
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        p = carrier.merge(flags, on=["symbol", "trade_date"], how="left")
        days = eligible_rank_lists(p)
        bench = bench_series(oos_s, oos_e)
        b1_tw: pd.DataFrame | None = None

        for rule in RULES:
            if rule == "B5_buffer_r2a_ramp":
                assert b1_tw is not None
                state = gross_series(bench, "R2a_confirm5").reindex(b1_tw.index).fillna(1.0)
                tw = b1_tw.mul(ramped_gross(state), axis=0)
            else:
                book = build_book(days, rule)
                tw = bp._apply_delay(book, trade_dates, 1)
                if rule == "B1_buffer30":
                    b1_tw = tw
            assert (tw.sum(axis=1) <= 1.0 + 1e-6).all(), "leverage breach"

            res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg8)
            nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
            r = nav.pct_change().dropna()
            daily[rule][fold] = r
            m = res.metrics
            tw_long = tw.stack()
            tw_long = tw_long[tw_long > 0].rename("w").reset_index()
            tw_long.columns = ["trade_date", "symbol", "w"]
            tw_long["sec"] = tw_long["symbol"].astype(str).map(smap).fillna("?")
            sec_max = float(tw_long.groupby(["trade_date", "sec"])["w"].sum()
                            .groupby("trade_date").max().mean()) if len(tw_long) else 0.0
            results[rule][fold] = {
                "cagr": round(m.annualized_return, 4), "maxdd": round(m.max_drawdown, 4),
                "sharpe": round(m.sharpe, 3), "turnover": round(float(m.turnover), 4),
                "mean_max_sector_weight": round(sec_max, 3),
            }
            res15 = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg15)
            sens15[rule][fold] = {"cagr": round(res15.metrics.annualized_return, 4),
                                  "maxdd": round(res15.metrics.max_drawdown, 4)}
            print(f"{fold} {rule:18s} CAGR {m.annualized_return:+.1%} DD {m.max_drawdown:.1%} "
                  f"turn {m.turnover:.3f} sec {sec_max:.2f} | 15bps {res15.metrics.annualized_return:+.1%}",
                  flush=True)

    # ---- carrier daily returns from frozen EXP-008 stitched file (no rerun)
    stitched = pd.read_csv(REPO / "runtime/reports/v89_closed_loop/wf_h008/stitched_daily_returns.csv",
                           index_col=0, parse_dates=True)
    carrier_daily = {
        f: stitched[CARRIER].loc[pd.Timestamp(s["oos"][0]):pd.Timestamp(s["oos"][1])].dropna()
        for f, s in FOLDS.items()
    }
    books = list(RULES) + [CARRIER]
    daily_all = {**daily, CARRIER: carrier_daily}
    folds = list(FOLDS)

    growth = np.array([[float(np.log1p(daily_all[n][f]).sum()) for n in books] for f in folds])
    lam = []
    for combo in itertools.combinations(range(len(folds)), 2):
        mask = np.zeros(len(folds), dtype=bool); mask[list(combo)] = True
        tr, te = growth[mask].sum(0), growth[~mask].sum(0)
        w = int(np.argmax(tr))
        omega = float((te <= te[w]).sum()) / (len(books) + 1)
        lam.append(math.log(omega / (1 - omega)))
    pbo = round(float((np.array(lam) <= 0).mean()), 3)

    def dsr(name: str) -> float:
        r = pd.concat([daily_all[name][f] for f in folds]).to_numpy()
        sr = float(r.mean() / r.std(ddof=1))
        z = (r - r.mean()) / r.std(ddof=1)
        g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
        srs = [float(pd.concat([daily_all[n][f] for f in folds]).to_numpy().mean()
                     / pd.concat([daily_all[n][f] for f in folds]).to_numpy().std(ddof=1))
               for n in books]
        v = float(np.var(srs, ddof=1))
        sr0 = math.sqrt(v) * ((1 - EULER) * norm_ppf(1 - 1 / N_TRIALS)
                              + EULER * norm_ppf(1 - 1 / (N_TRIALS * math.e)))
        denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        return round(norm_cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4)

    summary: dict[str, object] = {
        "protocol": "BOOK_CHURN_CONTROL_EXPERIMENT.md", "carrier": CARRIER,
        "baseline_frozen": BASE, "cumulative_trials_N": N_TRIALS, "rules": {},
        "fold_block_pbo_6books": pbo, "dsr_stitched": {n: dsr(n) for n in books},
        "sensitivity_15bps": sens15,
    }
    for rule in RULES:
        cs = [results[rule][f]["cagr"] for f in folds]
        dd = [results[rule][f]["maxdd"] for f in folds]
        to = [results[rule][f]["turnover"] for f in folds]
        sc = [results[rule][f]["mean_max_sector_weight"] for f in folds]
        med = float(np.median(cs))
        gates = {
            "G1_turnover": max(to) <= BASE["turn_cap"],
            "G2_worst_dd": max(dd) <= BASE["worst_dd"],
            "G3_f2_material": results[rule]["F2"]["cagr"] >= BASE["f2_floor"],
            "G4_median": med >= BASE["median_floor"],
            "G5_sector": max(sc) <= BASE["sector_cap"],
            "G6_no_leverage": True,  # asserted at construction
            "G7_quarantine": True,   # asserted per fold
        }
        summary["rules"][rule] = {
            "fold_cagrs": cs, "median_cagr": round(med, 4), "min_cagr": round(min(cs), 4),
            "worst_dd": round(max(dd), 4), "max_turnover": round(max(to), 4),
            "max_sector": round(max(sc), 3), "per_fold": results[rule],
            "gates": gates, "all_gates": all(gates.values()),
            "churn_solved_crash_unsolved": (gates["G1_turnover"] and gates["G2_worst_dd"]
                                            and gates["G4_median"] and gates["G5_sector"]
                                            and not gates["G3_f2_material"]),
        }
    summary["peak_rss_gib"] = round(rss_gib(), 2)
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for rule in RULES:
        for f in folds:
            rows.append({"rule": rule, "fold": f, **results[rule][f],
                         "cagr_15bps": sens15[rule][f]["cagr"]})
    pd.DataFrame(rows).to_csv(OUT / "book_fold_metrics.csv", index=False)
    pd.DataFrame({n: pd.concat([daily_all[n][f] for f in folds]) for n in books}) \
        .to_csv(OUT / "stitched_daily_returns.csv")
    print(json.dumps({r: {"gates": summary["rules"][r]["gates"],
                          "all": summary["rules"][r]["all_gates"]} for r in RULES}, indent=2))
    print(json.dumps({"pbo": pbo, "dsr": summary["dsr_stitched"]}, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
