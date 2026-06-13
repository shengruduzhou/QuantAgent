#!/usr/bin/env python3
"""Factor orthogonalization / neutralization / de-redundancy — governance step before retrain.

Mainstream AI-quant preprocessing applied cross-sectionally PER DAY to every kept factor:
  1. winsorize (1%/99% clip)              — 去极值
  2. cross-sectional z-score               — 标准化
  3. industry-neutralize (申万一级 demean)  — 行业中性
  4. size-neutralize (residual vs log size) — 市值中性 (proxy: log amount_mean_20d; no mktcap field)
  5. re-z-score
De-redundancy: drop the redundant members of each correlation cluster (factor_diagnostics
redundancy.json), keeping the ICIR-strongest head. Labels + meta (is_st/is_suspended/limits)
are passed through unchanged so train-v8-deep's filters still work.

Output: runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUG = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_aug_v85.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
REDUN = "runtime/reports/v8/factor_diagnostics/redundancy.json"
OUT = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet"

_NON_FACTOR = {
    "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "available_at",
    "source", "source_type", "source_reliability", "point_in_time_valid", "label",
    "is_suspended", "is_st", "is_limit_up", "is_limit_down",
    "missing_fundamentals", "missing_valuation", "missing_disclosures",
}


def _factor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _NON_FACTOR and not c.startswith(("forward_return", "label_end", "return_"))
            and pd.api.types.is_numeric_dtype(df[c])]


def _neutralize_day(F: pd.DataFrame, sector: pd.Series, size: pd.Series) -> pd.DataFrame:
    # 1) winsorize 1/99
    lo, hi = F.quantile(0.01), F.quantile(0.99)
    F = F.clip(lower=lo, upper=hi, axis=1)
    # 2) z-score
    F = (F - F.mean()) / F.std(ddof=0).replace(0, np.nan)
    # 3) industry demean (residual after 申万一级 dummies)
    F = F - F.groupby(sector, observed=True).transform("mean")
    # 4) size residual: resid_ij = F_ij - s_i * beta_j (single regressor, per factor)
    s = size.copy()
    s = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    s = s.fillna(0.0)
    svar = float((s * s).sum())
    if svar > 1e-9:
        beta = F.mul(s, axis=0).sum() / svar
        F = F.sub(pd.DataFrame(np.outer(s.to_numpy(), beta.to_numpy()), index=F.index, columns=F.columns))
    # 5) re-z-score
    F = (F - F.mean()) / F.std(ddof=0).replace(0, np.nan)
    return F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keep-redundant", action="store_true", help="neutralize only, don't drop clusters")
    args = ap.parse_args()

    df = pd.read_parquet(AUG)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= pd.Timestamp(args.start)) & (df["trade_date"] <= pd.Timestamp(args.end))].copy()
    # Tradability flags must always come from the silver panel: gold inputs
    # have carried stale all-False flags before (2026-06-11), and passing
    # them through silently re-contaminates every downstream dataset.
    flags = pd.read_parquet(
        "runtime/data/v7/silver/market_panel/market_panel.parquet",
        columns=["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up", "is_limit_down"],
    )
    flags["trade_date"] = pd.to_datetime(flags["trade_date"])
    df = df.drop(columns=[c for c in ("is_st", "is_suspended", "is_limit_up", "is_limit_down") if c in df.columns])
    df = df.merge(flags, on=["symbol", "trade_date"], how="left")
    for c in ("is_st", "is_suspended", "is_limit_up", "is_limit_down"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    del flags
    factors = _factor_cols(df)

    redundant: set[str] = set()
    if not args.keep_redundant and Path(REDUN).exists():
        red = json.load(open(REDUN))
        for c in red.get("clusters", []):
            redundant.update(c.get("redundant_with", []))
    keep = [f for f in factors if f not in redundant]
    drop = [f for f in factors if f in redundant]
    print(f"factors={len(factors)} keep={len(keep)} drop_redundant={len(drop)}", flush=True)

    sm = pd.read_parquet(SECTOR, columns=["symbol", "sector_level_1"])
    df = df.merge(sm, on="symbol", how="left")
    df["sector_level_1"] = df["sector_level_1"].fillna("UNKNOWN").astype("category")
    size_col = "amount_mean_20d" if "amount_mean_20d" in df.columns else "amount"
    df["_logsize"] = np.log1p(pd.to_numeric(df[size_col], errors="coerce").clip(lower=0))

    df[keep] = df[keep].astype("float32")
    parts = []
    dates = sorted(df["trade_date"].unique())
    for i, d in enumerate(dates):
        m = df["trade_date"] == d
        sub = df.loc[m]
        if len(sub) < 30:
            parts.append(sub[keep])
            continue
        parts.append(_neutralize_day(sub[keep], sub["sector_level_1"], sub["_logsize"]).astype("float32"))
        if (i + 1) % 300 == 0:
            print(f"  neutralized {i+1}/{len(dates)} days", flush=True)
    neut = pd.concat(parts).reindex(df.index)
    df[keep] = neut

    out_cols = [c for c in df.columns if c not in set(drop) | {"sector_level_1", "_logsize"}]
    df = df[out_cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    cov = {f: round(float(df[f].notna().mean()), 3) for f in keep[:1]}
    print(f"wrote {args.out}: {df.shape[0]} rows, {df.shape[1]} cols "
          f"(kept {len(keep)} neutralized factors, dropped {len(drop)} redundant)", flush=True)
    print(f"sanity (first kept factor coverage): {cov}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
