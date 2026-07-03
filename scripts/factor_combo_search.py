#!/usr/bin/env python3
"""Re-select the factor COMBINATION + WEIGHTS across ALL factors to MAX absolute CAGR.

Pool = the 3 model sleeve scores (short/mid/long) + every dataset factor
(alpha*/gtja*/llm_*). Each candidate is oriented by its validation IC sign and
cross-sectionally ranked per date. Greedy forward selection (with replacement →
integer weights, sparse + overfit-resistant) maximizes a fast top-k daily-return
proxy CAGR on the VALIDATION window; the winning combination is then confirmed
with the trusted strict backtest (baseline_protocol variant C) on the unseen
HELD-OUT window. Contamination-safe: held-out never used for selection.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp  # noqa: E402

DATASET = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
ENSEMBLE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
ANN = 244


def _proxy_cagr(df: pd.DataFrame, score: np.ndarray, k: int, round_trip: float = 0.0025) -> float:
    """Turnover-AWARE proxy: daily eligible top-k, NET of rebalance cost.

    A cost-blind proxy rewards high-turnover daily churn that A-share t+1 fill +
    costs destroy (proxy +266% -> strict -0.5%). Here daily return is reduced by
    (fraction of the k names swapped vs yesterday) * round_trip cost, so the
    search optimizes net tradable return and the proxy tracks the strict backtest.
    """
    s = pd.DataFrame({"d": df["trade_date"].to_numpy(), "sym": df["symbol"].to_numpy(),
                      "r": df["fwd1d"].to_numpy(), "s": score})
    s = s.sort_values("s", ascending=False)
    top = s.groupby("d", sort=False).head(k)
    ret = top.groupby("d")["r"].mean()
    sets = top.groupby("d")["sym"].apply(frozenset)
    days = list(ret.index)
    prev: frozenset = frozenset()
    net = []
    for d in days:
        cur = sets.loc[d]
        turn = len(cur - prev) / k if prev else 1.0  # fraction swapped in
        net.append(float(ret.loc[d]) - turn * round_trip)
        prev = cur
    net = pd.Series(net).dropna()
    if len(net) < 20:
        return -1.0
    nav = float((1.0 + net).prod())
    return nav ** (ANN / len(net)) - 1.0 if nav > 0 else -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-start", default="2024-08-28")
    ap.add_argument("--val-end", default="2025-08-31")
    # No defaults: the old 2025-09-01+ defaults silently consumed the (now
    # quarantined) holdout. bp.evaluate() fails closed on quarantined windows.
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-units", type=int, default=12, help="greedy steps = max total weight units")
    ap.add_argument("--n-library", type=int, default=24, help="top library factors by |val IC| to admit")
    ap.add_argument("--output-dir", default="runtime/reports/v89_closed_loop/factor_combo_search")
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    cols = pd.read_parquet(DATASET, columns=None, engine="pyarrow").columns if False else None
    import pyarrow.parquet as pq
    all_cols = pq.ParquetFile(DATASET).schema.names
    factor_cols = [c for c in all_cols if re.match(r"alpha\d", c) or c.startswith("gtja") or c.startswith("llm_")]
    need = ["symbol", "trade_date", "forward_return_1d", "is_st", "is_suspended", "is_limit_up", *factor_cols]
    df = pd.read_parquet(DATASET, columns=[c for c in need if c in all_cols])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= pd.Timestamp(args.val_start)) & (df["trade_date"] <= pd.Timestamp(args.test_end))]
    df = df.rename(columns={"forward_return_1d": "fwd1d"})
    df["eligible"] = ~(df["is_st"].fillna(False).astype(bool)
                       | df["is_suspended"].fillna(False).astype(bool)
                       | df["is_limit_up"].fillna(False).astype(bool))

    # merge the 3 model sleeve scores
    ens = pd.read_parquet(ENSEMBLE); ens["trade_date"] = pd.to_datetime(ens["trade_date"])
    sleeve_cols = [c for c in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score") if c in ens.columns]
    df = df.merge(ens[["symbol", "trade_date", *sleeve_cols]], on=["symbol", "trade_date"], how="left")

    candidates = sleeve_cols + factor_cols
    df = df[df["eligible"]].reset_index(drop=True)
    val_mask = (df["trade_date"] <= pd.Timestamp(args.val_end)).to_numpy()
    ho_mask = (df["trade_date"] >= pd.Timestamp(args.test_start)).to_numpy()

    # Orient each candidate by validation IC sign and cross-sectionally rank per date.
    print(f"ranking + orienting {len(candidates)} candidates on validation...", flush=True)
    dval = df[val_mask]
    ic = {}
    for c in candidates:
        x = pd.to_numeric(df[c], errors="coerce")
        # validation IC (Spearman vs fwd1d), used for sign + library pre-screen
        v = pd.DataFrame({"d": dval["trade_date"].to_numpy(), "x": x[val_mask].to_numpy(), "r": dval["fwd1d"].to_numpy()}).dropna()
        if len(v) < 1000:
            ic[c] = 0.0; continue
        daily = v.groupby("d").apply(lambda g: g["x"].corr(g["r"], method="spearman"))
        ic[c] = float(daily.mean()) if len(daily) else 0.0
    # admit all sleeves + llm, plus top-N library by |IC|
    lib = [c for c in factor_cols if not c.startswith("llm_")]
    lib_top = sorted(lib, key=lambda c: -abs(ic[c]))[: args.n_library]
    pool = sleeve_cols + [c for c in factor_cols if c.startswith("llm_")] + lib_top
    print(f"pool = {len(pool)} candidates (3 sleeves + llm + top-{args.n_library} library)", flush=True)

    # precompute oriented per-date ranks for the pool (sign * pct-rank)
    R = {}
    for c in pool:
        sign = 1.0 if ic[c] >= 0 else -1.0
        r = df.groupby("trade_date")[c].rank(pct=True).to_numpy()
        R[c] = sign * np.nan_to_num(r, nan=0.5)

    # greedy forward selection with replacement on VALIDATION proxy CAGR
    cum = np.zeros(len(df))
    weights: dict[str, int] = {}
    best_val = -1.0
    df_val = df[val_mask]
    val_idx = np.where(val_mask)[0]
    for step in range(args.max_units):
        gains = []
        for c in pool:
            trial = (cum + R[c])[val_idx]
            v = _proxy_cagr(df_val, trial, args.top_k)
            gains.append((v, c))
        v, c = max(gains, key=lambda t: t[0])
        if v <= best_val + 1e-4:
            print(f"step {step}: no improvement (best val CAGR {best_val:+.2%}); stop", flush=True)
            break
        best_val = v; cum = cum + R[c]; weights[c] = weights.get(c, 0) + 1
        print(f"step {step}: +{c}  -> val proxy CAGR {v:+.2%}  (weights={weights})", flush=True)

    # held-out proxy + strict confirmation
    ho_proxy = _proxy_cagr(df[ho_mask], cum[ho_mask], args.top_k)
    # build composite predictions for strict backtest (combined score per symbol/date)
    comp = df[["symbol", "trade_date"]].copy(); comp["composite_score"] = cum
    comp_path = outdir / "combo_predictions.parquet"; comp.to_parquet(comp_path, index=False)
    strict = bp.evaluate(str(comp_path), top_k=args.top_k, start=args.test_start, end=args.test_end,
                         slippage_bps=8.0, variants=["C_flags_eligible_delay1"], score_column="composite_score",
                         save_backtest_dir=str(outdir / "combo_heldout"))
    sc = strict["variants"]["C_flags_eligible_delay1"]
    summary = {
        "selected_weights": weights, "top_k": args.top_k,
        "val_proxy_cagr": best_val, "heldout_proxy_cagr": ho_proxy,
        "heldout_strict": {"cagr": sc["ann"], "maxDD": sc["maxDD"], "sharpe": sc["sharpe"],
                            "calmar": (sc["ann"] / sc["maxDD"] if sc["maxDD"] else 0.0)},
        "candidate_val_ic": {c: round(ic[c], 4) for c in pool},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== RESULT ===\nselected: {weights}")
    print(f"val proxy CAGR {best_val:+.2%} | held-out proxy {ho_proxy:+.2%} | held-out STRICT CAGR {sc['ann']:+.2%} "
          f"(maxDD {sc['maxDD']:.2%}, Calmar {summary['heldout_strict']['calmar']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
