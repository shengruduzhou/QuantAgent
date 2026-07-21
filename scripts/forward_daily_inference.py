#!/usr/bin/env python3
"""Forward daily inference — score NEW panel dates with the frozen v8.8 models.

Predictions in the v8.8 run end 2026-05-07; the forward loop needs scores
for every later close. This script recomputes EXACTLY the features the three
sleeve checkpoints were trained on (schemas read from the artifacts):

  base px features  return_1d / momentum_{5,20}d / volatility_20d /
                    amount_mean_20d / volume_mean_20d / intraday_return
  alphaXXX          quantagent.factors.alpha181 (computed on a trailing
                    warmup slice so rolling windows are exact)
  gtjaXXX           quantagent.factors.gtja191
  synth_* / llm_*   GA/LLM discovered formulas (accepted_definitions.json)
  idx_*             akshare index/commodity/treasury closes + 5d log
                    returns, as-of merged (same rule as the dataset builder)

then runs each sleeve's FT-Transformer checkpoint and blends with the run's
ensemble weights into ``composite_score``, appending to
runtime/reports/v8/forward/ensemble_forward.parquet.

--validate replays an OVERLAP window and reports (a) per-date rank
correlation against the training run's own ensemble_composite and (b) the
5d rank-IC of both score sets.

KNOWN FIDELITY LIMIT (2026-06-12): mean spearman ≈ 0.71, not ~1.0. Root
cause: 11/145 feature columns (alpha003/013/015/016/027/042/050/061/062/
065/095) are NOT reproducible with current code — the v8.2 alpha101
performance refactor changed their numerical behaviour after the gold
dataset was built. The forward scores are self-consistent across days but
are a noisy variant of what the checkpoints were trained on (overlap-window
rank-IC −0.040 vs −0.010 for the original). Full fidelity requires fixing
those factor implementations or a v8.9 retrain on rebuilt features.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
RUN_DIR = Path("runtime/reports/v8/deep/v88_judgment_20260611_2015")
DISCOVERED = "runtime/reports/v8/discovery/eval_v87/accepted_definitions.json"
# H-029 fidelity repair (frozen-semantics restore): the v8.9 "+7" llm_* factor
# definitions live in the closed-loop pooled_eval_clean file, NOT in the
# v8.7-era file above. Before this fix the seven llm_* sleeve features were
# silently NaN-filled -> missing-mask tokens -> short/mid sleeve fidelity 0.93/0.91.
DISCOVERED_PLUS7 = "runtime/reports/v89_closed_loop/pooled_eval_clean/accepted_definitions.json"
INDEX_ROOT = Path("runtime/data/v7/raw/akshare/index")
OUT_PATH = Path("runtime/reports/v8/forward/ensemble_forward.parquet")
SLEEVES = ("short_5d", "mid_5d_30d", "long_30d_120d")
ANN = 244


def _sleeve_features(run_dir: Path) -> dict[str, list[str]]:
    out = {}
    for sl in SLEEVES:
        schema = json.loads((run_dir / sl / "ft" / "ft_transformer_feature_schema.json")
                            .read_text(encoding="utf-8"))
        out[sl] = [str(c) for c in schema["feature_columns"]]
    return out


def _base_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    group = data.groupby("symbol", sort=False)
    data["return_1d"] = group["close"].pct_change()
    data["momentum_5d"] = group["close"].pct_change(5)
    data["momentum_20d"] = group["close"].pct_change(20)
    data["volatility_20d"] = data.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(20, min_periods=5).std())
    data["amount_mean_20d"] = group["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["volume_mean_20d"] = group["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    data["intraday_return"] = data["close"] / data["open"].replace(0, np.nan) - 1.0
    return data


def _index_features(needed: list[str]) -> pd.DataFrame:
    """Replicate dataset_builder._load_index_wide: closes + 5d log returns."""
    frames = []
    for table in ("equity_index", "commodity_main", "treasury_future"):
        p = INDEX_ROOT / f"{table}.parquet"
        if not p.exists():
            continue
        frame = pd.read_parquet(p)
        if frame.empty or "close" not in frame.columns:
            continue
        wide = frame.pivot_table(index="available_at", columns="label",
                                 values="close", aggfunc="last")
        wide.columns = [f"idx_{str(c).lower()}_close" for c in wide.columns]
        wide = wide.reset_index()
        for col in [c for c in wide.columns if c != "available_at"]:
            series = pd.to_numeric(wide[col], errors="coerce")
            wide[f"{col[:-len('_close')]}_ret5"] = np.log(series / series.shift(5))
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for piece in frames[1:]:
        combined = combined.merge(piece, on="available_at", how="outer")
    combined = combined.sort_values("available_at").reset_index(drop=True)
    keep = ["available_at"] + [c for c in combined.columns if c in needed]
    return combined[keep]


def build_feature_frame(panel: pd.DataFrame, needed: list[str],
                        target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute the union feature frame for the target dates (warmup included
    in ``panel``; rows outside target_dates are dropped at the end)."""
    need = set(needed)
    alpha_names = sorted(n for n in need if re.fullmatch(r"alpha\d+", n))
    gtja_names = sorted(n for n in need if n.startswith("gtja"))
    synth_names = sorted(n for n in need if n.startswith(("synth_", "llm_")))
    idx_names = sorted(n for n in need if n.startswith("idx_"))

    feats = _base_features(panel)

    if alpha_names:
        from quantagent.factors.alpha181 import compute_alpha181
        a = compute_alpha181(panel, names=alpha_names, wide=True)
        feats = feats.merge(a, on=["symbol", "trade_date"], how="left")
        print(f"  alpha181: {len(alpha_names)} columns", flush=True)
    if gtja_names:
        from quantagent.factors.gtja191 import compute_gtja191_factors
        g = compute_gtja191_factors(panel, names=gtja_names, wide=True)
        feats = feats.merge(g, on=["symbol", "trade_date"], how="left")
        print(f"  gtja191: {len(gtja_names)} columns", flush=True)
    if synth_names:
        from quantagent.factors.factor_synthesis import compute_synthesized_factors
        pieces = []
        for defs in (DISCOVERED, DISCOVERED_PLUS7):
            part = compute_synthesized_factors(panel, defs)
            if not part.empty:
                pieces.append(part)
        s_long = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        if not s_long.empty:
            s_long = s_long.drop_duplicates(subset=["symbol", "trade_date", "factor_name"])
        if not s_long.empty:
            s_wide = s_long[s_long["factor_name"].isin(synth_names)].pivot_table(
                index=["symbol", "trade_date"], columns="factor_name",
                values="factor_value", aggfunc="last").reset_index()
            feats = feats.merge(s_wide, on=["symbol", "trade_date"], how="left")
        print(f"  synthesized: {len(synth_names)} columns", flush=True)
    if idx_names:
        idx_wide = _index_features(idx_names)
        if not idx_wide.empty:
            idx_wide["available_at"] = pd.to_datetime(idx_wide["available_at"])
            # dataset builder merges on the ROW's available_at (= t+1), so a
            # row dated t carries the index close OF t (available_at t+1)
            feats["available_at"] = feats["trade_date"] + pd.Timedelta(days=1)
            feats = pd.merge_asof(
                feats.sort_values("available_at"),
                idx_wide.sort_values("available_at"),
                on="available_at", direction="backward").drop(columns=["available_at"])
        missing_idx = [c for c in idx_names if c not in feats.columns]
        if missing_idx:
            print(f"  WARN idx columns unavailable (filled 0): {missing_idx}", flush=True)
        print(f"  idx: {len(idx_names) - len(missing_idx)} columns", flush=True)

    feats = feats[feats["trade_date"].isin(target_dates)]
    for col in needed:
        if col not in feats.columns:
            feats[col] = np.nan
    return feats[["symbol", "trade_date", *needed]].reset_index(drop=True)


# Must mirror cli/v8_deep.py exactly — the checkpoints were trained on
# per-date rank-transformed features (cross_sectional_norm="rank").
_NO_CROSS_SECTIONAL_NORM = {
    "core_policy_score",
    "core_sentiment_score",
    "flow_north_total",
    "flow_margin_sh",
    "idx_csi300_ret5",
}


def rank_normalize(feats: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    """Per-date percentile rank centred to [-0.5, 0.5]; NaN -> 0 (median)."""
    normalize_cols = [c for c in needed if c not in _NO_CROSS_SECTIONAL_NORM]
    passthrough = [c for c in needed if c in _NO_CROSS_SECTIONAL_NORM]
    for c in passthrough:
        feats[c] = pd.to_numeric(feats[c], errors="coerce") \
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    ranked = feats.groupby("trade_date", sort=False)[normalize_cols].rank(pct=True) - 0.5
    for c in normalize_cols:
        feats[c] = ranked[c].fillna(0.0).astype("float32")
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(RUN_DIR))
    ap.add_argument("--start", default=None,
                    help="first date to score (default: after last forward/base prediction)")
    ap.add_argument("--end", default=None, help="last date to score (default: panel max)")
    ap.add_argument("--warmup-days", type=int, default=420)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--validate", action="store_true",
                    help="score an overlap window and rank-correlate vs the training run")
    ap.add_argument("--validate-days", type=int, default=8)
    ap.add_argument("--output", default=str(OUT_PATH))
    ap.add_argument("--sleeve-scores-output", default=None,
                    help="also append per-sleeve raw scores (wide parquet) — "
                         "needed by the frozen candidates S2-S4 (H-029)")
    args = ap.parse_args()

    import sys as _sys
    print(
        "[forward_daily_inference] WARNING: pinned to the v8.8 run "
        f"({args.run_dir}) — a SUPERSEDED, data-corrupted generation whose "
        "3-sleeve blend does NOT match production (configs/production_blend.json), "
        "with 11 known non-reproducible feature columns (overlap spearman ~0.71). "
        "Scores must NOT be used as production or trusted-evaluation evidence. "
        "See PRODUCTION_REPRODUCIBILITY_AUDIT.md Q6 / patch P6.",
        file=_sys.stderr, flush=True,
    )

    run_dir = Path(args.run_dir)
    sleeve_feats = _sleeve_features(run_dir)
    union_feats = sorted({c for cols in sleeve_feats.values() for c in cols})
    summary = json.loads((run_dir / "ensemble_summary.json").read_text(encoding="utf-8"))
    weights = summary["weights"]

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                            "close", "volume", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel_dates = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))

    base = pd.read_parquet(run_dir / "ensemble_composite.parquet",
                           columns=["trade_date", "symbol", "composite_score"])
    base["trade_date"] = pd.to_datetime(base["trade_date"])

    if args.validate:
        t_end = base["trade_date"].max()
        if args.end:  # honor --end in validate mode (H-028 guard-gap fix)
            t_end = min(t_end, pd.Timestamp(args.end))
        target = panel_dates[(panel_dates <= t_end)][-args.validate_days:]
        # P4-style quarantine guard (previously missing here; incident logged
        # 2026-07-13 in runtime/state/holdout_access_log.jsonl — EXP-028)
        qcfg = json.loads(Path("configs/quarantined_windows.json").read_text())
        for w in qcfg["windows"]:
            ws, we = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            hit = [d for d in target if ws <= d <= we]
            if hit and not os.environ.get("QUANTAGENT_ALLOW_QUARANTINED_VALIDATE"):
                raise SystemExit(
                    f"REFUSED: validate window intersects quarantined window "
                    f"{w['start']}..{w['end']} ({len(hit)} dates, e.g. {hit[0].date()}). "
                    f"Choose --end <= pre-quarantine or set "
                    f"QUANTAGENT_ALLOW_QUARANTINED_VALIDATE='<reason>' (logged, per "
                    f"EVALUATION_PROTOCOL_V2 section 3 exception process).")
    else:
        out_path = Path(args.output)
        last_scored = base["trade_date"].max()
        if out_path.exists():
            prev = pd.read_parquet(out_path, columns=["trade_date"])
            if len(prev):
                last_scored = max(last_scored, pd.to_datetime(prev["trade_date"]).max())
        lo = pd.Timestamp(args.start) if args.start else last_scored + pd.Timedelta(days=1)
        hi = pd.Timestamp(args.end) if args.end else panel_dates.max()
        target = panel_dates[(panel_dates >= lo) & (panel_dates <= hi)]
    if len(target) == 0:
        print("nothing to score (panel has no new dates) — refresh the panel first")
        return 0
    print(f"scoring {len(target)} dates: {target.min().date()}..{target.max().date()}", flush=True)

    warmup_lo = target.min() - pd.Timedelta(days=args.warmup_days)
    p_slice = panel[(panel["trade_date"] >= warmup_lo) & (panel["trade_date"] <= target.max())]
    active = p_slice.loc[p_slice["trade_date"].isin(target) & p_slice["close"].gt(0),
                         "symbol"].unique()
    p_slice = p_slice[p_slice["symbol"].isin(active)].copy()
    print(f"universe {len(active)} symbols, panel slice {len(p_slice)} rows", flush=True)

    feats = build_feature_frame(p_slice, union_feats, target)

    # Replicate the training population (exec dataset drops st(t)|susp(t)
    # rows; the t+1-flag part of that filter is unknowable at signal time)
    # so the per-date rank cross-section matches what the model saw.
    flags_panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date",
                                                  "is_st", "is_suspended"])
    flags_panel["trade_date"] = pd.to_datetime(flags_panel["trade_date"])
    feats = feats.merge(flags_panel, on=["symbol", "trade_date"], how="left")
    bad = feats["is_st"].fillna(False).astype(bool) \
        | feats["is_suspended"].fillna(False).astype(bool)
    feats = feats[~bad].drop(columns=["is_st", "is_suspended"])

    feats = rank_normalize(feats, union_feats)
    print(f"feature frame: {feats.shape} (rank-normalized, st/susp dropped)", flush=True)

    from quantagent.training.ft_transformer_trainer import predict_ft_transformer_artifact

    sleeve_preds: dict[str, pd.DataFrame] = {}
    for sl in SLEEVES:
        res = predict_ft_transformer_artifact(
            run_dir / sl / "ft", feats[["symbol", "trade_date", *sleeve_feats[sl]]],
            device=args.device)
        f = res.predictions[["symbol", "trade_date", "prediction"]].rename(
            columns={"prediction": "alpha_score"})
        sleeve_preds[sl] = f
        print(f"  {sl}: {len(f)} predictions", flush=True)

    if args.sleeve_scores_output:
        wide = None
        for sl in SLEEVES:
            p = sleeve_preds[sl].rename(columns={"alpha_score": f"score_{sl}"})
            wide = p if wide is None else wide.merge(p, on=["symbol", "trade_date"], how="outer")
        sp_path = Path(args.sleeve_scores_output)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        if sp_path.exists():
            prev = pd.read_parquet(sp_path)
            prev["trade_date"] = pd.to_datetime(prev["trade_date"])
            wide = pd.concat([prev[~prev["trade_date"].isin(wide["trade_date"].unique())], wide],
                             ignore_index=True)
        wide.to_parquet(sp_path, index=False)
        print(f"sleeve scores -> {sp_path} ({len(wide)} rows total)", flush=True)

    from quantagent.training.horizon_models import (
        HorizonClass, HorizonEnsembleWeights, ensemble_horizon_predictions)
    blended = ensemble_horizon_predictions(
        {HorizonClass(k): v for k, v in sleeve_preds.items()},
        weights=HorizonEnsembleWeights(**{
            {"short_5d": "short", "mid_5d_30d": "mid", "long_30d_120d": "long"}[k]: float(v)
            for k, v in weights.items()}))

    if args.validate:
        merged = blended.merge(base, on=["trade_date", "symbol"], suffixes=("_fwd", "_base"))
        score_fwd = [c for c in merged.columns if c.endswith("_fwd") and "composite" in c]
        score_base = [c for c in merged.columns if c.endswith("_base") and "composite" in c]
        sf = score_fwd[0] if score_fwd else "composite_score_fwd"
        sb = score_base[0] if score_base else "composite_score_base"
        corr = merged.groupby("trade_date").apply(
            lambda g: g[sf].corr(g[sb], method="spearman"), include_groups=False)
        print("=== validation: per-date spearman vs training-run composite ===")
        print(corr.round(4).to_string())
        print(f"mean {corr.mean():.4f}  min {corr.min():.4f}  (n_dates={len(corr)})")

        # the metric that matters: does the forward path predict returns as
        # well as the training-run composite did on the same dates?
        px = panel.pivot_table(index="trade_date", columns="symbol", values="close",
                               aggfunc="last").sort_index()
        fwd5 = (px.shift(-5) / px - 1.0).stack().rename("ret5").reset_index()
        merged = merged.merge(fwd5, on=["trade_date", "symbol"], how="left")
        ic = merged.dropna(subset=["ret5"]).groupby("trade_date").apply(
            lambda g: pd.Series({
                "ic_fwd": g[sf].corr(g["ret5"], method="spearman"),
                "ic_base": g[sb].corr(g["ret5"], method="spearman"),
            }), include_groups=False)
        print("=== rank-IC (5d fwd return) — forward path vs training composite ===")
        print(ic.round(4).to_string())
        print(f"mean ic_fwd {ic['ic_fwd'].mean():.4f}  mean ic_base {ic['ic_base'].mean():.4f}")
        merged.to_parquet(Path(args.output).parent / "validate_merged.parquet", index=False)
        return 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prev["trade_date"] = pd.to_datetime(prev["trade_date"])
        blended = pd.concat([prev, blended], ignore_index=True).drop_duplicates(
            ["trade_date", "symbol"], keep="last")
    blended.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(blended)} rows, "
          f"max date {pd.to_datetime(blended['trade_date']).max().date()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
