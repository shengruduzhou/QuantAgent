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

Each factor also gets a recorded (non-gating) economic profile on the full
panel — the place where day continuity is real, so decay/capacity are valid:
  - IC-decay curve: oos_ic_h{1,3,5,10,20} + ic_peak_horizon
  - top-bucket / long-short return: oos_top_bucket_return, oos_long_short_return,
    oos_long_short_after_cost (turnover × --profile-cost-bps), oos_turnover
  - capacity: capacity_ratio, capacity_risk
Field semantics match the discovery-time leaderboard
(factor_synthesis._factor_economic_profile) for a consistent schema.

Rejected factors keep a machine-readable reason so the LLM memory loop can
learn from them. Outputs:
  <out>/factor_eval_table.csv       (every candidate: full metrics + status)
  <out>/factor_leaderboard.csv      (accepted library only, sorted by |OOS IC|)
  <out>/factor_report.md            (human-readable run report — Stage 3.4)
  <out>/rejected_factors.csv|.md    (structured reject reasons — Stage 3.3)
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


def _ic_decay_curve(
    values: pd.Series,
    fwd_by_h: dict[int, pd.Series],
    dates: pd.Series,
    mask: pd.Series,
) -> dict[int, float]:
    """OOS rank-IC at each forward horizon (IC-decay profile).

    Computed on the FULL panel where per-symbol day continuity holds (forward
    returns are derived by shifting ``close`` ``h`` trading days), so it is
    methodologically valid — unlike the sub-sampled factor-acceptance panel.
    """
    return {h: _ic_stats(values[mask], fwd[mask], dates[mask])[0] for h, fwd in fwd_by_h.items()}


def _economic_profile(
    values: pd.Series,
    labels: pd.Series,
    dates: pd.Series,
    symbols: pd.Series,
    mask: pd.Series,
    sign: float,
    cost_bps: float,
    q: int = 5,
) -> dict[str, float]:
    """Top-bucket / long-short return + turnover, factor oriented to +IC.

    Reuses ``evaluation.quantile_group_backtest`` so the field semantics match
    the discovery-time leaderboard profile (``factor_synthesis._factor_economic_profile``).
    Best-effort: any failure returns NaNs.
    """
    out = {
        "oos_top_bucket_return": float("nan"),
        "oos_long_short_return": float("nan"),
        "oos_turnover": float("nan"),
        "oos_long_short_after_cost": float("nan"),
    }
    try:
        from quantagent.factors import evaluation as _eval

        frame = pd.DataFrame({
            "trade_date": dates[mask].values,
            "symbol": symbols[mask].values,
            "factor": values[mask].values * sign,
            "fwd": labels[mask].values,
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["factor", "fwd"])
        if frame.empty or frame["trade_date"].nunique() < 2:
            return out
        qb = _eval.quantile_group_backtest(frame, "factor", "fwd", quantiles=q, cost_bps=cost_bps)
        gr = qb.group_returns
        if q in gr.columns:
            out["oos_top_bucket_return"] = float(gr[q].mean())
        if not qb.long_short.dropna().empty:
            out["oos_long_short_return"] = float(qb.long_short.mean())
        if not qb.turnover.dropna().empty:
            out["oos_turnover"] = float(qb.turnover.mean())
        if not qb.cost_adjusted_long_short.dropna().empty:
            out["oos_long_short_after_cost"] = float(qb.cost_adjusted_long_short.mean())
    except Exception:  # noqa: BLE001 — profiling must never break the eval run
        pass
    return out


def _write_rejected_export(table: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Structured export of rejected factors: machine-readable CSV + Markdown.

    Gives the LLM memory loop and a human reviewer a specific reason per rejected
    formula, grouped by failing gate — the "rejection reason if rejected"
    acceptance-criteria requirement, at the authoritative full-panel gate.
    """
    rej = table[table["status"] != "accepted"].copy() if not table.empty else table.iloc[0:0].copy()
    cols = [c for c in ("name", "status", "reject_detail", "complexity",
                        "train_rank_ic", "oos_rank_ic", "oos_icir", "oos_monotonicity",
                        "oos_long_short_return", "oos_turnover", "max_reference_corr",
                        "max_accepted_corr", "error", "expression") if c in rej.columns]
    csv_path = out_dir / "rejected_factors.csv"
    rej[cols].to_csv(csv_path, index=False)

    md_path = out_dir / "rejected_factors.md"
    lines = ["# Rejected Factors", "", f"Total rejected: **{len(rej)}**", ""]
    if rej.empty:
        lines.append("_No rejected factors._")
    else:
        counts = rej["status"].value_counts()
        lines += ["## By reason", "", "| reason | count |", "|---|---|"]
        lines += [f"| {reason} | {n} |" for reason, n in counts.items()]
        lines.append("")
        for reason in counts.index:
            grp = rej[rej["status"] == reason]
            lines += [f"## {reason} ({len(grp)})", ""]
            for _, r in grp.iterrows():
                detail = str(r.get("reject_detail") or r.get("error") or "")
                lines.append(f"- **{r['name']}** — {detail}")
                expr = str(r.get("expression", "") or "")
                if expr:
                    lines.append(f"  - `{expr[:160]}`")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Render a small DataFrame as GitHub-flavoured Markdown table lines."""
    cols = [c for c in columns if c in df.columns]
    if df.empty or not cols:
        return ["_(none)_"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    return [head, sep, *body]


def _write_standard_report(
    table: pd.DataFrame, meta: dict, out_dir: Path,
) -> tuple[Path, Path]:
    """Standard deliverables: factor_leaderboard.csv (accepted library) +
    factor_report.md (human-readable run report). Stage 3.4."""
    acc = table[table["status"] == "accepted"].copy() if not table.empty else table.iloc[0:0].copy()
    if not acc.empty and "oos_rank_ic" in acc.columns:
        acc = acc.reindex(acc["oos_rank_ic"].abs().sort_values(ascending=False).index)

    lb_cols = [c for c in (
        "name", "oos_rank_ic", "oos_icir", "ic_peak_horizon",
        "oos_top_bucket_return", "oos_long_short_return", "oos_long_short_after_cost",
        "oos_turnover", "capacity_ratio", "capacity_risk",
        "max_reference_corr", "max_accepted_corr", "complexity",
        "description", "expression",
    ) if c in acc.columns]
    lb_path = out_dir / "factor_leaderboard.csv"
    acc[lb_cols].to_csv(lb_path, index=False)

    rej = table[table["status"] != "accepted"] if not table.empty else table.iloc[0:0]
    md = [
        "# Factor Discovery Report", "",
        f"- train ≤ `{meta.get('train_end')}` · OOS > train_end"
        + (f" ≤ `{meta.get('oos_end')}`" if meta.get("oos_end") else ""),
        f"- candidates: **{meta.get('candidates', 0)}** · accepted: **{len(acc)}** · rejected: **{len(rej)}**",
        f"- reference OOS |ICIR| median: `{meta.get('reference_oos_icir_median', 0):.3f}` "
        f"→ candidate ICIR gate `{meta.get('icir_gate', 0):.3f}`",
        f"- gates: min OOS |IC| `{meta.get('min_oos_ic')}`, max ref corr `{meta.get('max_reference_correlation')}`, "
        f"max accepted corr `{meta.get('max_correlation')}`, min |monotonicity| `{meta.get('min_monotonicity_corr')}`",
        "",
        "## Accepted factor library (sorted by |OOS RankIC|)", "",
    ]
    md += _md_table(acc, [
        "name", "oos_rank_ic", "oos_icir", "ic_peak_horizon",
        "oos_long_short_return", "oos_long_short_after_cost", "oos_turnover",
        "capacity_ratio", "max_reference_corr",
    ])
    md += ["", "Full metrics: `factor_leaderboard.csv`. Each accepted factor carries the full "
           "economic profile (IC-decay curve `oos_ic_h*`, top-bucket/long-short return, turnover, capacity).", ""]
    if not rej.empty:
        counts = rej["status"].value_counts()
        md += ["## Rejections by reason", "", "| reason | count |", "|---|---|"]
        md += [f"| {reason} | {n} |" for reason, n in counts.items()]
        md += ["", "Per-factor reasons: `rejected_factors.md` / `rejected_factors.csv`.", ""]
    report_path = out_dir / "factor_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return lb_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--definitions", nargs="+", required=True,
                    help="One or more synthesized_definitions.json paths.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--market-panel", default=PANEL)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--label-column", default="forward_return_5d")
    ap.add_argument("--train-end", default="2024-07-31")
    ap.add_argument("--oos-end", default=None,
                    help="Cap the OOS selection window at this date so a later tail stays a clean, "
                         "never-seen final-test window (avoids selection-bias contamination).")
    ap.add_argument("--reference-columns", default=DEFAULT_REFERENCES)
    ap.add_argument("--min-oos-ic", type=float, default=0.015)
    ap.add_argument("--min-icir-vs-reference", type=float, default=0.5)
    ap.add_argument("--max-reference-correlation", type=float, default=0.6)
    ap.add_argument("--max-correlation", type=float, default=0.7)
    ap.add_argument("--min-monotonicity-corr", type=float, default=0.6)
    ap.add_argument("--sample-symbols", type=int, default=0, help="0 = full universe")
    ap.add_argument("--decay-horizons", default="1,3,5,10,20",
                    help="Forward-return horizons (trading days) for the IC-decay curve.")
    ap.add_argument("--profile-cost-bps", type=float, default=15.0,
                    help="Round-trip cost (bps) applied to the long-short turnover for the after-cost spread.")
    args = ap.parse_args()
    decay_horizons = tuple(int(h) for h in args.decay_horizons.split(",") if h.strip())

    references = [c.strip() for c in args.reference_columns.split(",") if c.strip()]
    import pyarrow.parquet as _pq

    _base_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "is_st", "is_suspended", "is_limit_up"]
    # Include any PIT fundamental columns the panel carries so valuation/quality
    # factors (OptionalColumn pb/roe/gross_margin/debt_to_asset/...) can be scored
    # instead of silently evaluating to NaN on a price-volume-only panel.
    _panel_cols = set(_pq.read_schema(args.market_panel).names)
    _fund_cols = [c for c in ("pb", "roe", "gross_margin", "debt_to_asset", "pe_ttm",
                              "turnover_rate", "revenue", "operating_cash_flow")
                  if c in _panel_cols]
    panel = pd.read_parquet(
        args.market_panel,
        columns=[*_base_cols, *_fund_cols],
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
    if args.oos_end:
        is_oos = is_oos & (merged["trade_date"] <= pd.Timestamp(args.oos_end))
    is_train = (merged["trade_date"] <= train_end) & eligible
    y = merged[args.label_column]
    d = merged["trade_date"]
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    universe_median_amount = float(amount[is_oos].median())

    # Forward returns at each decay horizon, derived once on the full panel.
    # merged is sorted by (symbol, trade_date) so the per-symbol shift respects
    # day order; continuity here is real (unlike the sub-sampled accept panel).
    close = pd.to_numeric(merged["close"], errors="coerce")
    grp_close = merged.assign(_c=close).groupby("symbol", sort=False)["_c"]
    fwd_by_h: dict[int, pd.Series] = {
        h: (grp_close.shift(-h) / close - 1.0).replace([np.inf, -np.inf], np.nan)
        for h in decay_horizons
    }

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
        # IC-decay curve + full economic profile (top-bucket/long-short/turnover),
        # factor oriented to positive train-IC so bucket Q5 is the favoured side.
        sign = 1.0 if (train_ic >= 0) else -1.0
        decay = _ic_decay_curve(vals, fwd_by_h, d, is_oos)
        peak_h = max(decay, key=lambda h: abs(decay[h])) if decay else 0
        profile = _economic_profile(vals, y, d, merged["symbol"], is_oos, sign, args.profile_cost_bps)
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
            oos_top_bucket_return=round(profile["oos_top_bucket_return"], 5) if np.isfinite(profile["oos_top_bucket_return"]) else np.nan,
            oos_long_short_return=round(profile["oos_long_short_return"], 5) if np.isfinite(profile["oos_long_short_return"]) else np.nan,
            oos_long_short_after_cost=round(profile["oos_long_short_after_cost"], 5) if np.isfinite(profile["oos_long_short_after_cost"]) else np.nan,
            oos_turnover=round(profile["oos_turnover"], 4) if np.isfinite(profile["oos_turnover"]) else np.nan,
            ic_peak_horizon=int(peak_h),
            **{f"oos_ic_h{h}": round(float(decay[h]), 4) for h in decay_horizons},
        )
        status = "accepted"
        reject_detail = ""
        if finite_oos < 0.5:
            status = "low_finite_ratio"
            reject_detail = f"OOS finite ratio {finite_oos:.3f} < 0.5 (formula evaluates to NaN too often)"
        elif np.sign(train_ic) != np.sign(oos_ic):
            status = "oos_ic_failed"
            reject_detail = f"sign flip: train IC {train_ic:+.4f} vs OOS IC {oos_ic:+.4f} (edge did not persist out-of-sample)"
        elif abs(oos_ic) < args.min_oos_ic:
            status = "oos_ic_failed"
            reject_detail = f"|OOS RankIC| {abs(oos_ic):.4f} < {args.min_oos_ic:.4f} min"
        elif abs(oos_icir) < icir_gate:
            status = "oos_icir_failed"
            reject_detail = f"|OOS ICIR| {abs(oos_icir):.3f} < {icir_gate:.3f} gate ({args.min_icir_vs_reference:g}× reference median; unstable day-to-day)"
        elif ref_corr > args.max_reference_correlation:
            status = "high_corr_to_library"
            reject_detail = f"max corr to reference library {ref_corr:.3f} > {args.max_reference_correlation:.3f} (duplicates an existing factor)"
        elif cand_corr > args.max_correlation:
            status = "high_corr_to_accepted"
            reject_detail = f"max corr to already-accepted {cand_corr:.3f} > {args.max_correlation:.3f} (redundant with this batch)"
        elif (np.sign(oos_ic) * spread) <= 0 or abs(mono) < args.min_monotonicity_corr:
            status = "no_monotonicity"
            reject_detail = f"weak/non-monotone quantiles: rank-corr {mono:+.3f} (need |·|≥{args.min_monotonicity_corr:.2f}), spread {spread:+.5f} vs IC sign {np.sign(oos_ic):+.0f}"
        row["status"] = status
        row["reject_detail"] = reject_detail
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
    rejected_csv, rejected_md = _write_rejected_export(table, out_dir)
    report_meta = {
        "candidates": len(candidates),
        "train_end": args.train_end,
        "oos_end": args.oos_end,
        "reference_oos_icir_median": ref_icir_median,
        "icir_gate": icir_gate,
        "min_oos_ic": args.min_oos_ic,
        "max_reference_correlation": args.max_reference_correlation,
        "max_correlation": args.max_correlation,
        "min_monotonicity_corr": args.min_monotonicity_corr,
    }
    leaderboard_csv, report_md = _write_standard_report(table, report_meta, out_dir)
    summary = {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "rejected_by_reason": table[table["status"] != "accepted"]["status"].value_counts().to_dict() if not table.empty else {},
        "reference_oos_icir_median": ref_icir_median,
        "train_end": args.train_end,
        "table": str(out_dir / "factor_eval_table.csv"),
        "accepted_definitions": str(out_dir / "accepted_definitions.json"),
        "rejected_factors_csv": str(rejected_csv),
        "rejected_factors_md": str(rejected_md),
        "factor_leaderboard_csv": str(leaderboard_csv),
        "factor_report_md": str(report_md),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
