#!/usr/bin/env python3
"""Full-universe OOS evaluation + accept/reject gates for discovered formula alphas.

Takes synthesized_definitions.json files (from GP seeds and/or the LLM
generator), evaluates each formula over the whole market panel, and grades
it on the SAME protocol used for the human/alpha181 library:

  train   <= --train-end          (what the search saw)
  oos     >  --train-end          (never seen by any search)

Gates (all must pass for "accepted"):
  - finite ratio >= 0.5 on OOS
  - oos RankIC >= --min-oos-ic (default 0.015), same sign as train
  - oos daily ICIR >= --min-icir-vs-reference x median reference ICIR
  - |spearman corr| to every reference factor < --max-reference-correlation
  - |spearman corr| to every previously accepted candidate < --max-correlation
  - OOS quintile monotonicity: top-bottom spread > 0 and rank-corr >= 0.6

Rejected factors keep a machine-readable reason so the LLM memory loop can
learn from them. Outputs:
  <out>/factor_eval_table.csv
  <out>/accepted_definitions.json   (feed to materialize/retrain)
  <out>/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.factors import expr as E
from quantagent.factors.factor_synthesis import _node_count, load_definitions

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
LABELS = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
DEFAULT_REFERENCES = "alpha016,alpha015,alpha050,alpha044,alpha040,alpha161,alpha163,alpha088,alpha145"


def _daily_ic(values: pd.Series, labels: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"d": dates.values, "f": values.values, "y": labels.values}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby("d")
    df = df.assign(rf=g["f"].rank(), ry=g["y"].rank())
    daily = df.groupby("d")[["rf", "ry"]].corr().unstack().iloc[:, 1].dropna()
    return daily


def _ic_stats(values, labels, dates) -> tuple[float, float, int]:
    daily = _daily_ic(values, labels, dates)
    if daily.empty:
        return 0.0, 0.0, 0
    mean = float(daily.mean())
    std = float(daily.std(ddof=0))
    return mean, (mean / std if std > 1e-12 else 0.0), int(len(daily))


def _quintile_monotonicity(values: pd.Series, labels: pd.Series, dates: pd.Series, q: int = 5) -> tuple[float, float]:
    df = pd.DataFrame({"d": dates.values, "f": values.values, "y": labels.values}).dropna()
    if df.empty:
        return 0.0, 0.0
    df["bucket"] = df.groupby("d")["f"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop")
    )
    means = df.groupby("bucket")["y"].mean()
    if len(means) < q:
        return 0.0, 0.0
    spread = float(means.iloc[-1] - means.iloc[0])
    rank_corr = float(pd.Series(means.values).corr(pd.Series(range(len(means))), method="spearman"))
    return spread, rank_corr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--definitions", nargs="+", required=True,
                    help="One or more synthesized_definitions.json paths.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--market-panel", default=PANEL)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--label-column", default="forward_return_5d")
    ap.add_argument("--train-end", default="2024-07-31")
    ap.add_argument("--reference-columns", default=DEFAULT_REFERENCES)
    ap.add_argument("--min-oos-ic", type=float, default=0.015)
    ap.add_argument("--min-icir-vs-reference", type=float, default=0.5)
    ap.add_argument("--max-reference-correlation", type=float, default=0.6)
    ap.add_argument("--max-correlation", type=float, default=0.7)
    ap.add_argument("--min-monotonicity-corr", type=float, default=0.6)
    ap.add_argument("--sample-symbols", type=int, default=0, help="0 = full universe")
    args = ap.parse_args()

    references = [c.strip() for c in args.reference_columns.split(",") if c.strip()]
    panel = pd.read_parquet(
        args.market_panel,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                 "is_st", "is_suspended", "is_limit_up"],
    )
    labels = pd.read_parquet(
        args.labels, columns=["symbol", "trade_date", args.label_column, *references]
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
    merged = panel.merge(labels, on=["symbol", "trade_date"], how="inner")
    if args.sample_symbols:
        rng = np.random.default_rng(7)
        keep = set(rng.choice(sorted(merged["symbol"].unique()), size=args.sample_symbols, replace=False))
        merged = merged[merged["symbol"].isin(keep)]
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    # Score factors on the ELIGIBLE pool only (matching the executable
    # selection protocol): IC earned on ST/suspended/limit-up names is
    # alpha the strategy is never allowed to act on.
    eligible = ~(
        merged["is_st"].fillna(False).astype(bool)
        | merged["is_suspended"].fillna(False).astype(bool)
        | merged["is_limit_up"].fillna(False).astype(bool)
    )
    # NOTE: factor expressions still see the full per-symbol history (rolling
    # windows need it); only the IC/label rows are restricted to eligible.
    train_end = pd.Timestamp(args.train_end)
    is_oos = (merged["trade_date"] > train_end) & eligible
    is_train = (merged["trade_date"] <= train_end) & eligible
    y = merged[args.label_column]
    d = merged["trade_date"]
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    universe_median_amount = float(amount[is_oos].median())

    # Reference factor ICIR scale (same metric, same OOS window).
    ref_icirs = []
    ref_values_oos: dict[str, pd.Series] = {}
    for ref in references:
        if ref not in merged.columns:
            continue
        vals = pd.to_numeric(merged[ref], errors="coerce")
        _, icir, _ = _ic_stats(vals[is_oos], y[is_oos], d[is_oos])
        ref_icirs.append(abs(icir))
        ref_values_oos[ref] = vals[is_oos]
    ref_icir_median = float(np.median(ref_icirs)) if ref_icirs else 0.0
    icir_gate = args.min_icir_vs_reference * ref_icir_median
    print(f"reference OOS |ICIR| median = {ref_icir_median:.3f} -> candidate gate >= {icir_gate:.3f}")

    # Load and dedupe candidate definitions.
    candidates: list[E.FactorDefinition] = []
    seen: set[str] = set()
    for path in args.definitions:
        for definition in load_definitions(path):
            key = repr(definition.expr)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(definition)
    print(f"{len(candidates)} unique candidate formulas from {len(args.definitions)} files")

    rows: list[dict] = []
    accepted: list[E.FactorDefinition] = []
    accepted_values: list[pd.Series] = []
    for i, definition in enumerate(candidates):
        name = f"{definition.name}_{i:03d}"
        row: dict = {"name": name, "expression": repr(definition.expr),
                     "description": definition.description,
                     "complexity": _node_count(definition.expr)}
        try:
            vals = pd.to_numeric(definition.expr.evaluate(merged), errors="coerce").replace([np.inf, -np.inf], np.nan)
        except Exception as exc:  # noqa: BLE001
            rows.append({**row, "status": "evaluate_failed", "error": str(exc)})
            continue
        finite_oos = float(vals[is_oos].notna().mean())
        train_ic, train_icir, _ = _ic_stats(vals[is_train], y[is_train], d[is_train])
        oos_ic, oos_icir, oos_days = _ic_stats(vals[is_oos], y[is_oos], d[is_oos])
        spread, mono = _quintile_monotonicity(vals[is_oos], y[is_oos], d[is_oos])
        # Capacity: liquidity of the names the factor actually favours.
        oos_df = pd.DataFrame({
            "d": d[is_oos].values, "f": vals[is_oos].values, "amt": amount[is_oos].values,
        }).dropna()
        if not oos_df.empty:
            sign = 1.0 if (train_ic >= 0) else -1.0
            oos_df["r"] = oos_df.groupby("d")["f"].rank(pct=True, ascending=(sign > 0))
            top_amount = float(oos_df.loc[oos_df["r"] >= 0.9, "amt"].median())
            capacity_ratio = top_amount / universe_median_amount if universe_median_amount > 0 else np.nan
        else:
            capacity_ratio = np.nan
        capacity_risk = bool(np.isfinite(capacity_ratio) and capacity_ratio < 0.2)
        ref_corr = 0.0
        for ref_vals in ref_values_oos.values():
            c = abs(vals[is_oos].corr(ref_vals, method="spearman"))
            if np.isfinite(c):
                ref_corr = max(ref_corr, float(c))
        cand_corr = 0.0
        for prev in accepted_values:
            c = abs(vals[is_oos].corr(prev, method="spearman"))
            if np.isfinite(c):
                cand_corr = max(cand_corr, float(c))
        row.update(
            train_rank_ic=round(train_ic, 4), train_icir=round(train_icir, 3),
            oos_rank_ic=round(oos_ic, 4), oos_icir=round(oos_icir, 3), oos_days=oos_days,
            oos_finite_ratio=round(finite_oos, 3),
            oos_quintile_spread=round(spread, 5), oos_monotonicity=round(mono, 3),
            max_reference_corr=round(ref_corr, 3), max_accepted_corr=round(cand_corr, 3),
            capacity_ratio=round(float(capacity_ratio), 3) if np.isfinite(capacity_ratio) else np.nan,
            capacity_risk=capacity_risk,
        )
        status = "accepted"
        if finite_oos < 0.5:
            status = "low_finite_ratio"
        elif np.sign(train_ic) != np.sign(oos_ic) or abs(oos_ic) < args.min_oos_ic:
            status = "oos_ic_failed"
        elif abs(oos_icir) < icir_gate:
            status = "oos_icir_failed"
        elif ref_corr > args.max_reference_correlation:
            status = "high_corr_to_library"
        elif cand_corr > args.max_correlation:
            status = "high_corr_to_accepted"
        elif (np.sign(oos_ic) * spread) <= 0 or abs(mono) < args.min_monotonicity_corr:
            status = "no_monotonicity"
        row["status"] = status
        rows.append(row)
        flag = "ACCEPT" if status == "accepted" else "reject"
        print(f"[{i+1}/{len(candidates)}] {flag:6} {name:28} train {train_ic:+.4f} | "
              f"oos {oos_ic:+.4f} icir {oos_icir:+.3f} | mono {mono:+.2f} | refcorr {ref_corr:.2f} | {status}")
        if status == "accepted":
            accepted.append(E.FactorDefinition(name=name, expr=definition.expr, description=definition.description))
            accepted_values.append(vals[is_oos])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "factor_eval_table.csv", index=False)
    from quantagent.factors.factor_synthesis import save_definitions

    save_definitions(accepted, out_dir / "accepted_definitions.json")
    summary = {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected_by_reason": table[table["status"] != "accepted"]["status"].value_counts().to_dict() if not table.empty else {},
        "reference_oos_icir_median": ref_icir_median,
        "train_end": args.train_end,
        "table": str(out_dir / "factor_eval_table.csv"),
        "accepted_definitions": str(out_dir / "accepted_definitions.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
