#!/usr/bin/env python3
"""H-029 Stage 2/3: layer-by-layer forward-inference fidelity diagnosis.

Golden window 2025-06-02..2025-08-29 (preregistered 1b47c3d; pre-quarantine
asserted; NO labels/returns read — pure score fidelity).

Truth anchors: gold dataset (exact training feature values) + stored sleeve
predictions + 0.30/0.45/0.25 blend. Layers:
  L0 row population (gold keys vs forward keys)
  L1 raw features   (recomputed vs gold, per column)
  L2 normalized     (rank semantics, population effects)
  L3a inference     (checkpoints on gold features/population vs stored preds)
  L3b end-to-end    (forward path vs stored preds, per sleeve)
  L4 composite      (blend fidelity + top-K overlap)
Stage-3 population test: checkpoints on gold VALUES with the forward
population mask (isolates rank-universe effect from value drift).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from forward_daily_inference import (  # noqa: E402
    SLEEVES, _sleeve_features, build_feature_frame, rank_normalize,
)

RUN = REPO / "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300"
GOLD = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
OUT = REPO / "runtime/reports/h029"
W0, W1 = pd.Timestamp("2025-06-02"), pd.Timestamp("2025-08-29")
QUAR = pd.Timestamp("2025-09-01")
BLEND_W = {"short_5d": 0.30, "mid_5d_30d": 0.45, "long_30d_120d": 0.25}


def daily_spear(a: pd.Series, b: pd.Series, dates) -> pd.Series:
    df = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy(), "d": dates}).dropna()
    return df.groupby("d", group_keys=False)[["a", "b"]].apply(
        lambda g: g["a"].rank().corr(g["b"].rank()) if len(g) >= 30 else np.nan).dropna()


def col_stats(name, gold_v, fwd_v, dates):
    al = pd.DataFrame({"g": gold_v, "f": fwd_v, "d": dates}).dropna()
    if len(al) < 100:
        return {"column": name, "n": len(al), "note": "too_few_aligned"}
    sp = daily_spear(al["g"], al["f"], al["d"])
    return {"column": name, "n": len(al),
            "mae": float((al["g"] - al["f"]).abs().mean()),
            "max_abs_err": float((al["g"] - al["f"]).abs().max()),
            "pearson": float(al["g"].corr(al["f"])),
            "daily_median_spearman": float(sp.median()),
            "daily_min_spearman": float(sp.min())}


def predict_frame(feats_norm: pd.DataFrame, sleeve_feats: dict, device: str) -> dict:
    from quantagent.training.ft_transformer_trainer import predict_ft_transformer_artifact
    out = {}
    for sl in SLEEVES:
        res = predict_ft_transformer_artifact(
            RUN / sl / "ft", feats_norm[["symbol", "trade_date", *sleeve_feats[sl]]],
            device=device)
        p = res.predictions[["symbol", "trade_date", "prediction"]].rename(
            columns={"prediction": f"pred_{sl}"})
        out[sl] = p
    return out


def score_vs_stored(preds: dict, stored: dict, tag: str, rows: list, daily_rows: list):
    per_sleeve = {}
    for sl in SLEEVES:
        m = preds[sl].merge(stored[sl], on=["symbol", "trade_date"], how="inner")
        sp = daily_spear(m[f"pred_{sl}"], m["alpha_score"], m["trade_date"].to_numpy())
        per_sleeve[sl] = {"median": float(sp.median()), "min": float(sp.min()),
                          "n_dates": int(len(sp)), "n_rows": len(m)}
        for d, v in sp.items():
            daily_rows.append({"layer": tag, "sleeve": sl, "date": str(pd.Timestamp(d).date()),
                               "spearman": round(float(v), 5)})
        rows.append({"layer": f"{tag}:{sl}", **{k: round(v, 5) if isinstance(v, float) else v
                                                for k, v in per_sleeve[sl].items()}})
        print(f"  {tag} {sl}: median {per_sleeve[sl]['median']:.4f} min {per_sleeve[sl]['min']:.4f}", flush=True)
    return per_sleeve


def blend(preds: dict) -> pd.DataFrame:
    b = None
    for sl in SLEEVES:
        p = preds[sl].copy()
        p["r"] = p.groupby("trade_date")[f"pred_{sl}"].rank(pct=True) * BLEND_W[sl]
        p = p[["symbol", "trade_date", "r"]].rename(columns={"r": f"r_{sl}"})
        b = p if b is None else b.merge(p, on=["symbol", "trade_date"], how="inner")
    b["composite"] = sum(b[f"r_{sl}"] for sl in SLEEVES)
    return b[["symbol", "trade_date", "composite"]]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sleeve_feats = _sleeve_features(RUN)
    union = sorted({c for cols in sleeve_feats.values() for c in cols})

    # ---- truth anchors
    gold = pd.read_parquet(GOLD, columns=["symbol", "trade_date"] + union,
                           filters=[("trade_date", ">=", W0), ("trade_date", "<=", W1)])
    gold["trade_date"] = pd.to_datetime(gold["trade_date"])
    assert gold["trade_date"].max() < QUAR
    stored = {}
    for sl in SLEEVES:
        s = pd.read_parquet(RUN / sl / "predictions.parquet")
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        stored[sl] = s[(s["trade_date"] >= W0) & (s["trade_date"] <= W1)]
        assert stored[sl]["trade_date"].max() < QUAR

    # ---- forward feature recompute (script-identical universe + warmup)
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                            "close", "volume", "amount"],
                            filters=[("trade_date", ">=", W0 - pd.Timedelta(days=420)),
                                     ("trade_date", "<=", W1)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUAR
    target = pd.DatetimeIndex(sorted(gold["trade_date"].unique()))
    active = panel.loc[panel["trade_date"].isin(target) & panel["close"].gt(0), "symbol"].unique()
    p_slice = panel[panel["symbol"].isin(active)].copy()
    print(f"recompute: universe {len(active)}, slice {len(p_slice):,}", flush=True)
    fwd_raw = build_feature_frame(p_slice, union, target)
    flags = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "is_st", "is_suspended"])
    flags["trade_date"] = pd.to_datetime(flags["trade_date"])
    fwd_raw = fwd_raw.merge(flags, on=["symbol", "trade_date"], how="left")
    bad = fwd_raw["is_st"].fillna(False).astype(bool) | fwd_raw["is_suspended"].fillna(False).astype(bool)
    fwd_pop = fwd_raw[~bad].drop(columns=["is_st", "is_suspended"]).copy()

    # ---- L0 population
    kg = set(map(tuple, gold[["symbol", "trade_date"]].itertuples(index=False)))
    kf = set(map(tuple, fwd_pop[["symbol", "trade_date"]].itertuples(index=False)))
    l0 = {"gold_rows": len(kg), "fwd_rows": len(kf),
          "missing_from_fwd": len(kg - kf), "extra_in_fwd": len(kf - kg),
          "jaccard": round(len(kg & kf) / len(kg | kf), 4)}
    print("L0 population:", json.dumps(l0), flush=True)

    # ---- L1 raw feature values on intersection
    inter = gold.merge(fwd_raw, on=["symbol", "trade_date"], suffixes=("_g", "_f"), how="inner")
    dates_i = inter["trade_date"].to_numpy()
    l1 = [col_stats(c, inter[f"{c}_g"], inter[f"{c}_f"], dates_i) for c in union]
    l1df = pd.DataFrame(l1).sort_values("daily_median_spearman")
    l1df.to_csv(OUT / "layer_comparison.csv", index=False)
    worst = l1df.head(15)[["column", "daily_median_spearman", "pearson", "mae", "n"]]
    print("L1 worst raw columns:\n", worst.to_string(index=False), flush=True)

    rows, daily_rows = [], []
    # ---- L3a: inference fidelity — gold values, gold population
    gold_norm = rank_normalize(gold.copy(), union)
    pa = predict_frame(gold_norm, sleeve_feats, args.device)
    l3a = score_vs_stored(pa, stored, "L3a_gold_values_gold_pop", rows, daily_rows)

    # ---- Stage-3 population test: gold values, FORWARD population mask
    gpop = gold.merge(fwd_pop[["symbol", "trade_date"]], on=["symbol", "trade_date"], how="inner")
    gp_norm = rank_normalize(gpop.copy(), union)
    pb = predict_frame(gp_norm, sleeve_feats, args.device)
    l3p = score_vs_stored(pb, stored, "L3pop_gold_values_fwd_pop", rows, daily_rows)

    # ---- L3b: end-to-end forward
    fwd_norm = rank_normalize(fwd_pop.copy(), union)
    pc = predict_frame(fwd_norm, sleeve_feats, args.device)
    l3b = score_vs_stored(pc, stored, "L3b_forward_end_to_end", rows, daily_rows)

    # ---- L4 composite + top-K overlap
    comp_stored = None
    for sl in SLEEVES:
        s = stored[sl].copy()
        s["r"] = s.groupby("trade_date")["alpha_score"].rank(pct=True) * BLEND_W[sl]
        s = s[["symbol", "trade_date", "r"]].rename(columns={"r": f"r_{sl}"})
        comp_stored = s if comp_stored is None else comp_stored.merge(s, on=["symbol", "trade_date"])
    comp_stored["composite"] = sum(comp_stored[f"r_{sl}"] for sl in SLEEVES)
    ce = blend(pc).merge(comp_stored[["symbol", "trade_date", "composite"]],
                         on=["symbol", "trade_date"], suffixes=("_f", "_s"))
    sp4 = daily_spear(ce["composite_f"], ce["composite_s"], ce["trade_date"].to_numpy())
    ov_rows = []
    for d, g in ce.groupby("trade_date"):
        tf10 = set(g.nlargest(10, "composite_f")["symbol"]); ts10 = set(g.nlargest(10, "composite_s")["symbol"])
        tf50 = set(g.nlargest(50, "composite_f")["symbol"]); ts50 = set(g.nlargest(50, "composite_s")["symbol"])
        ov_rows.append({"date": str(pd.Timestamp(d).date()),
                        "top10_overlap": len(tf10 & ts10) / 10, "top50_overlap": len(tf50 & ts50) / 50})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(OUT / "topk_overlap.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUT / "daily_score_fidelity.csv", index=False)

    summary = {"golden_window": f"{W0.date()}..{W1.date()}", "l0_population": l0,
               "l1_worst_columns": worst.to_dict("records"),
               "l3a_inference_gold_pop": l3a, "l3_population_test": l3p,
               "l3b_end_to_end": l3b,
               "l4_composite": {"median": round(float(sp4.median()), 5),
                                "min": round(float(sp4.min()), 5),
                                "top10_overlap_mean": round(float(ov["top10_overlap"].mean()), 3),
                                "top50_overlap_mean": round(float(ov["top50_overlap"].mean()), 3)},
               "runtime_s": round(time.time() - t0, 1)}
    (OUT / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "l1_worst_columns"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
