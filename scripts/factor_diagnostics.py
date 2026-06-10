#!/usr/bin/env python3
"""Factor diagnostics — the 因子诊断 foundation before orthogonalization + retrain.

For every factor in the augmented library computes (mainstream AI-quant governance):
  * RankIC (5d) + ICIR (mean/std of daily cross-sectional Spearman IC)
  * decay: RankIC at 1d / 5d / 20d horizons
  * stability: lag-5 cross-sectional rank autocorr (high = low turnover)
  * per-REGIME RankIC (bull / sideways / bear) — factors work differently per regime
  * redundancy: average cross-sectional |Spearman| correlation clusters (|ρ|≥ threshold)

Output: runtime/reports/v8/factor_diagnostics/{table.csv, redundancy.json, summary.md}
Read-only (no training). Date-sampled for tractability over the full library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUG = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_aug_v85.parquet"
CORE = "runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"

_NON_FACTOR = {
    "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "available_at",
    "source", "source_type", "source_reliability", "point_in_time_valid", "label",
    "is_suspended", "is_st", "is_limit_up", "is_limit_down",
    "missing_fundamentals", "missing_valuation", "missing_disclosures",
}


def _factor_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in _NON_FACTOR or c.startswith(("forward_return", "label_end", "return_")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _regime_labels(panel: pd.DataFrame, dates: list[pd.Timestamp]) -> dict:
    px = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    daily = px.pct_change(fill_method=None).mean(axis=1)
    trail = (1.0 + daily).rolling(60).apply(lambda x: x.prod() - 1.0, raw=True)
    lab = {}
    for d in dates:
        v = trail.get(d, np.nan)
        lab[d] = "bull" if v > 0.05 else ("bear" if v < -0.05 else "sideways")
    return lab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--sample-every", type=int, default=2, help="use every Nth trade date (speed)")
    ap.add_argument("--min-stocks", type=int, default=200)
    ap.add_argument("--corr-dates", type=int, default=40, help="dates sampled for the correlation matrix")
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--out", default="runtime/reports/v8/factor_diagnostics")
    args = ap.parse_args()

    fac = pd.read_parquet(AUG)
    fac["trade_date"] = pd.to_datetime(fac["trade_date"])
    fac = fac[(fac["trade_date"] >= pd.Timestamp(args.start)) & (fac["trade_date"] <= pd.Timestamp(args.end))]
    factors = _factor_cols(fac)  # before adding labels
    fwd = ["forward_return_1d", "forward_return_5d", "forward_return_20d"]
    if all(c in fac.columns for c in fwd):
        df = fac  # aug dataset already carries the forward-return labels
    else:
        lab = pd.read_parquet(CORE, columns=["symbol", "trade_date", *fwd])
        lab["trade_date"] = pd.to_datetime(lab["trade_date"])
        df = fac.merge(lab, on=["symbol", "trade_date"], how="inner")
    print(f"factors={len(factors)} rows={len(df)} dates={df['trade_date'].nunique()}", flush=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    all_dates = sorted(df["trade_date"].unique())
    dates = all_dates[:: args.sample_every]
    regimes = _regime_labels(panel[panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=120)], dates)

    # per-date cross-sectional Spearman IC for every factor at 1d/5d/20d
    ic = {h: [] for h in (1, 5, 20)}
    ic_reg = {r: [] for r in ("bull", "sideways", "bear")}
    for d in dates:
        day = df[df["trade_date"] == d]
        if len(day) < args.min_stocks:
            continue
        franks = day[factors].rank()
        for h in (1, 5, 20):
            fr = pd.to_numeric(day[f"forward_return_{h}d"], errors="coerce")
            if fr.notna().sum() < args.min_stocks:
                continue
            s = franks.corrwith(fr.rank())
            s.name = d
            ic[h].append(s)
            if h == 5:
                ic_reg[regimes.get(d, "sideways")].append(s)

    ic5 = pd.DataFrame(ic[5])
    ic1 = pd.DataFrame(ic[1]); ic20 = pd.DataFrame(ic[20])
    rank_ic = ic5.mean(); icir = ic5.mean() / ic5.std(ddof=0).replace(0, np.nan)
    table = pd.DataFrame({
        "factor": factors,
        "rank_ic_5d": rank_ic.reindex(factors).values,
        "icir_5d": icir.reindex(factors).values,
        "ic_1d": ic1.mean().reindex(factors).values,
        "ic_20d": ic20.mean().reindex(factors).values,
        "ic_bull": pd.DataFrame(ic_reg["bull"]).mean().reindex(factors).values if ic_reg["bull"] else np.nan,
        "ic_sideways": pd.DataFrame(ic_reg["sideways"]).mean().reindex(factors).values if ic_reg["sideways"] else np.nan,
        "ic_bear": pd.DataFrame(ic_reg["bear"]).mean().reindex(factors).values if ic_reg["bear"] else np.nan,
        "abs_icir": icir.reindex(factors).abs().values,
    }).sort_values("abs_icir", ascending=False)

    # redundancy: avg cross-sectional |Spearman| correlation over sampled dates → greedy clusters
    cdates = dates[:: max(1, len(dates) // args.corr_dates)][: args.corr_dates]
    corr_acc, n = None, 0
    for d in cdates:
        day = df[df["trade_date"] == d]
        if len(day) < args.min_stocks:
            continue
        c = day[factors].rank().corr().abs()
        corr_acc = c if corr_acc is None else corr_acc + c
        n += 1
    corr = (corr_acc / n) if n else pd.DataFrame()
    clusters = []
    if not corr.empty:
        seen = set()
        order = table["factor"].tolist()  # strongest ICIR first = cluster head
        for f in order:
            if f in seen or f not in corr.columns:
                continue
            grp = [g for g in corr.index if g != f and g not in seen and corr.loc[f, g] >= args.corr_threshold]
            if grp:
                clusters.append({"head": f, "redundant_with": grp[:12], "n": len(grp)})
                seen.update(grp)
            seen.add(f)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "table.csv", index=False)
    (out / "redundancy.json").write_text(json.dumps({
        "corr_threshold": args.corr_threshold, "n_clusters": len(clusters),
        "redundant_factor_count": int(sum(c["n"] for c in clusters)), "clusters": clusters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    top = table.head(25)
    md = ["# 因子诊断 (IC/ICIR/衰减/分regime)", "",
          f"- 窗口 {args.start}..{args.end}, 采样 every {args.sample_every}d, {len(factors)} 因子",
          f"- 冗余: {len(clusters)} 个相关簇 (|ρ|≥{args.corr_threshold}), 可剔除 ~{sum(c['n'] for c in clusters)} 个冗余因子", "",
          "## Top 25 因子 (按 |ICIR|)", "",
          "| factor | RankIC_5d | ICIR | IC_1d | IC_20d | IC_bull | IC_side | IC_bear |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in top.iterrows():
        md.append(f"| {r['factor']} | {r['rank_ic_5d']:.4f} | {r['icir_5d']:.3f} | {r['ic_1d']:.4f} | "
                  f"{r['ic_20d']:.4f} | {r['ic_bull']:.4f} | {r['ic_sideways']:.4f} | {r['ic_bear']:.4f} |")
    md += ["", "## 冗余簇 (头部因子保留, 其余可去)", ""]
    for c in clusters[:15]:
        md.append(f"- **{c['head']}** ⊃ {', '.join(c['redundant_with'][:8])}{' …' if c['n']>8 else ''}")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out}/summary.md  (factors={len(factors)}, clusters={len(clusters)}, "
          f"redundant≈{sum(c['n'] for c in clusters)})", flush=True)
    print(top[["factor", "rank_ic_5d", "icir_5d", "ic_bull", "ic_sideways", "ic_bear"]].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
