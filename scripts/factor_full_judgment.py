#!/usr/bin/env python3
"""因子判决明细 — unified verdict for EVERY factor family.

Covers, in one table with one protocol:
  * the 156 alpha181 columns materialised in the training dataset
    (alpha001-101 = WorldQuant Alpha101 approximations,
     alpha102-181 = CICC-inspired A-share price/volume templates),
  * the 25 alphas missing from the dataset (recomputed from the panel to
    document whether they are recoverable or degenerate),
  * GP/LLM discovered formulas (accepted_definitions.json),
  * any other numeric factor column present in the dataset.

Protocol (per user spec 2026-06-11):
  - eligible pool only (~is_st & ~is_suspended & ~is_limit_up) — IC earned
    on names the strategy cannot buy is phantom;
  - horizons 5d / 20d / 60d (short / mid / long sleeves);
  - per-year RankIC for 2022..2026 — a factor must be effective in each of
    the last 4 full years (sign-consistent, not just one era);
  - per-regime RankIC (bull / sideways / bear via benchmark 60d trailing);
  - capacity ratio (median amount of top-decile picks vs universe).

Verdicts:
  all_weather       4y sign-consistent + works in all 3 regimes
  robust_4y         4y sign-consistent + >=2/3 regimes
  regime_specialist fails 4y overall but |IC|>=0.03 in one regime with
                    yearly sign consistency inside that regime
  weak              some signal but fails the gates
  dead              |IC| < 0.005 everywhere
  not_cross_sectional / not_computable  (documented, not judged)

Implementation: daily Spearman = Pearson on within-date ranks, computed
with np.bincount over date codes (fast enough for 537 factor-horizon
combinations on 7M rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
HORIZONS = (5, 20, 60)
YEARS = (2022, 2023, 2024, 2025, 2026)
GATE_YEARS = (2022, 2023, 2024, 2025)


def set_gate_years(years: tuple[int, ...], gate_years: tuple[int, ...]) -> None:
    """Override the judgment year windows (used for as-of/PIT selection tests)."""
    global YEARS, GATE_YEARS
    YEARS = years
    GATE_YEARS = gate_years


def _regime_from_bench(bench: pd.Series) -> pd.Series:
    cum = (1 + bench).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=bench.index)


class DailyICEngine:
    """Vectorised daily rank-IC over (date_code, value, label_rank) arrays."""

    def __init__(self, dates: pd.Series, labels: dict[int, pd.Series], eligible: np.ndarray):
        self.date_index, self.date_codes = np.unique(dates.values, return_inverse=True)
        self.n_dates = len(self.date_index)
        self.eligible = eligible
        self.dates = dates
        self.label_ranks: dict[int, np.ndarray] = {}
        self.label_valid: dict[int, np.ndarray] = {}
        for h, lab in labels.items():
            ranks = lab.groupby(dates, sort=False).rank(method="average")
            arr = ranks.to_numpy(dtype=float)
            self.label_ranks[h] = arr
            self.label_valid[h] = np.isfinite(arr)

    def daily_ic(self, values: pd.Series, horizon: int) -> pd.Series:
        v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        f_rank = v.groupby(self.dates, sort=False).rank(method="average").to_numpy(dtype=float)
        y = self.label_ranks[horizon]
        ok = np.isfinite(f_rank) & self.label_valid[horizon] & self.eligible
        if ok.sum() < 100:
            return pd.Series(dtype=float)
        c = self.date_codes[ok]
        x = f_rank[ok]
        yy = y[ok]
        n = np.bincount(c, minlength=self.n_dates).astype(float)
        sx = np.bincount(c, weights=x, minlength=self.n_dates)
        sy = np.bincount(c, weights=yy, minlength=self.n_dates)
        sxx = np.bincount(c, weights=x * x, minlength=self.n_dates)
        syy = np.bincount(c, weights=yy * yy, minlength=self.n_dates)
        sxy = np.bincount(c, weights=x * yy, minlength=self.n_dates)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * sy / np.where(n > 0, n, np.nan)
            vx = sxx - sx * sx / np.where(n > 0, n, np.nan)
            vy = syy - sy * sy / np.where(n > 0, n, np.nan)
            corr = cov / np.sqrt(vx * vy)
        valid = (n >= 30) & np.isfinite(corr)
        return pd.Series(corr[valid], index=pd.DatetimeIndex(self.date_index[valid]))


def _stats(daily: pd.Series) -> tuple[float, float]:
    if daily.empty:
        return 0.0, 0.0
    m = float(daily.mean())
    s = float(daily.std(ddof=0))
    return m, (m / s if s > 1e-12 else 0.0)


def _judge_factor(engine: DailyICEngine, values: pd.Series, regime_by_date: pd.Series,
                  amount: pd.Series, eligible: np.ndarray, dates: pd.Series) -> dict:
    row: dict = {}
    # cross-sectional sanity: needs within-date variance
    sample_dates = engine.date_index[:: max(1, engine.n_dates // 25)]
    cs_std = values[dates.isin(sample_dates)].groupby(dates[dates.isin(sample_dates)]).std()
    if not np.isfinite(cs_std).any() or float(np.nanmedian(cs_std)) < 1e-12:
        row["verdict"] = "not_cross_sectional"
        return row

    best_h, best_icir, best_daily = None, 0.0, pd.Series(dtype=float)
    for h in HORIZONS:
        daily = engine.daily_ic(values, h)
        m, icir = _stats(daily)
        row[f"ic_{h}d"] = round(m, 4)
        row[f"icir_{h}d"] = round(icir, 3)
        if abs(icir) >= abs(best_icir):
            best_h, best_icir, best_daily = h, icir, daily
    row["best_horizon"] = f"{best_h}d"
    if best_daily.empty:
        row["verdict"] = "not_computable"
        return row

    sign = 1.0 if best_daily.mean() >= 0 else -1.0
    year_ics = {}
    for yr in YEARS:
        sub = best_daily[best_daily.index.year == yr]
        year_ics[yr] = float(sub.mean()) if len(sub) >= 20 else np.nan
        row[f"ic_{yr}"] = round(year_ics[yr], 4) if np.isfinite(year_ics[yr]) else np.nan
    regime_ics = {}
    reg = regime_by_date.reindex(best_daily.index)
    for rg in ("bull", "sideways", "bear"):
        sub = best_daily[reg == rg]
        regime_ics[rg] = float(sub.mean()) if len(sub) >= 20 else np.nan
        row[f"ic_{rg}"] = round(regime_ics[rg], 4) if np.isfinite(regime_ics[rg]) else np.nan

    # capacity
    v = pd.to_numeric(values, errors="coerce")
    ok = np.isfinite(v.to_numpy()) & eligible
    df = pd.DataFrame({"d": dates[ok].values, "f": sign * v[ok].values, "amt": amount[ok].values}).dropna()
    if not df.empty:
        df["r"] = df.groupby("d")["f"].rank(pct=True)
        top_amt = float(df.loc[df["r"] >= 0.9, "amt"].median())
        uni_amt = float(df["amt"].median())
        row["capacity_ratio"] = round(top_amt / uni_amt, 3) if uni_amt > 0 else np.nan
    else:
        row["capacity_ratio"] = np.nan

    gate_year_vals = [year_ics[y] for y in GATE_YEARS if np.isfinite(year_ics.get(y, np.nan))]
    year_ok = (len(gate_year_vals) >= len(GATE_YEARS)
               and all(np.sign(x) == sign and abs(x) >= 0.005 for x in gate_year_vals)
               and abs(np.mean(gate_year_vals)) >= 0.015)
    reg_vals = {k: x for k, x in regime_ics.items() if np.isfinite(x)}
    reg_same = [k for k, x in reg_vals.items() if np.sign(x) == sign and abs(x) >= 0.005]
    reg_bad = [k for k, x in reg_vals.items() if np.sign(x) != sign and abs(x) >= 0.02]

    if year_ok and len(reg_same) == 3:
        verdict = "all_weather"
    elif year_ok and len(reg_same) >= 2 and not reg_bad:
        verdict = "robust_4y"
    else:
        specialist = any(np.sign(x) == sign and abs(x) >= 0.03 for x in reg_vals.values())
        max_ic = max((abs(x) for x in [row.get(f"ic_{h}d", 0.0) for h in HORIZONS]), default=0.0)
        if specialist and max_ic >= 0.02:
            verdict = "regime_specialist"
        elif max_ic >= 0.005:
            verdict = "weak"
        else:
            verdict = "dead"
    row["verdict"] = verdict
    row["years_passed"] = sum(1 for x in gate_year_vals if np.sign(x) == sign and abs(x) >= 0.005)
    row["regimes_passed"] = len(reg_same)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--output-dir", default="runtime/reports/v8/factor_full_judgment")
    ap.add_argument("--start", default="2021-06-01", help="judgment window start (covers last ~4y + warmup)")
    ap.add_argument("--end", default=None, help="judgment window end (as-of/PIT selection tests)")
    ap.add_argument("--gate-years", default=None,
                    help="comma-separated years for the sign-consistency gate (default: module GATE_YEARS)")
    ap.add_argument("--definitions", nargs="*", default=["runtime/reports/v8/discovery/eval_v87/accepted_definitions.json"],
                    help="GP/LLM definition JSONs to include")
    ap.add_argument("--missing-symbols-sample", type=int, default=1200,
                    help="symbol sample for recomputing dataset-missing alphas (0 skips)")
    ap.add_argument("--batch", type=int, default=60, help="factor columns loaded per batch")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_names = pq.read_schema(args.dataset).names
    base_cols = ["symbol", "trade_date", "return_1d", "amount",
                 "is_st", "is_suspended", "is_limit_up",
                 "forward_return_5d", "forward_return_20d", "forward_return_60d"]
    skip = set(base_cols) | {"open", "high", "low", "close", "volume", "available_at", "source",
                             "source_type", "source_reliability", "point_in_time_valid",
                             "is_limit_down", "missing_fundamentals", "missing_valuation",
                             "missing_disclosures"}
    factor_cols = [c for c in schema_names
                   if c not in skip and not c.startswith(("forward_return", "label_end", "__"))]

    if args.gate_years:
        gates = tuple(int(y) for y in args.gate_years.split(",") if y.strip())
        set_gate_years(tuple(sorted(set(gates) | set(YEARS))), gates)
        print(f"gate years overridden: {GATE_YEARS}")

    print(f"loading base frame ({len(base_cols)} cols) ...")
    base = pd.read_parquet(args.dataset, columns=base_cols)
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base = base[base["trade_date"] >= pd.Timestamp(args.start)].reset_index(drop=True)
    if args.end:
        base = base[base["trade_date"] <= pd.Timestamp(args.end)].reset_index(drop=True)
    eligible = ~(
        base["is_st"].fillna(False).astype(bool)
        | base["is_suspended"].fillna(False).astype(bool)
        | base["is_limit_up"].fillna(False).astype(bool)
    ).to_numpy()
    dates = base["trade_date"]
    labels = {h: base[f"forward_return_{h}d"] for h in HORIZONS}
    engine = DailyICEngine(dates, labels, eligible)
    bench = base.groupby("trade_date")["return_1d"].mean()
    regime_by_date = _regime_from_bench(bench)
    amount = pd.to_numeric(base["amount"], errors="coerce")
    print(f"rows={len(base)} dates={engine.n_dates} eligible={eligible.mean():.1%}")

    rows: list[dict] = []
    for i in range(0, len(factor_cols), args.batch):
        chunk = factor_cols[i : i + args.batch]
        block = pd.read_parquet(args.dataset, columns=["symbol", "trade_date", *chunk])
        block["trade_date"] = pd.to_datetime(block["trade_date"])
        block = block[block["trade_date"] >= pd.Timestamp(args.start)].reset_index(drop=True)
        if args.end:
            block = block[block["trade_date"] <= pd.Timestamp(args.end)].reset_index(drop=True)
        assert len(block) == len(base), "row alignment mismatch between batches"
        for col in chunk:
            family = ("alpha101_worldquant" if col.startswith("alpha") and col[5:8].isdigit() and int(col[5:8]) <= 101
                      else "cicc_ashare80" if col.startswith("alpha")
                      else "macro_idx_flow" if col.startswith(("macro_", "idx_", "flow_"))
                      else "other_dataset")
            row = {"factor": col, "family": family, "source": "dataset"}
            row.update(_judge_factor(engine, block[col], regime_by_date, amount, eligible, dates))
            rows.append(row)
            print(f"[{len(rows)}/{len(factor_cols)}] {col:28} {row.get('verdict','?'):18} "
                  f"best={row.get('best_horizon','-'):4} ic5={row.get('ic_5d', float('nan'))}")
        del block

    # ---- GP/LLM discovered definitions on the panel sample -------------------
    from quantagent.factors.factor_synthesis import load_definitions

    if args.missing_symbols_sample:
        print("preparing panel sample for missing alphas + discovered formulas ...")
        panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                                "close", "volume", "amount"])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel[panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=200)]
        if args.end:
            panel = panel[panel["trade_date"] <= pd.Timestamp(args.end)]
        rng = np.random.default_rng(7)
        syms = sorted(panel["symbol"].unique())
        keep = set(rng.choice(syms, size=min(args.missing_symbols_sample, len(syms)), replace=False))
        panel = panel[panel["symbol"].isin(keep)].sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        sub_idx = base["symbol"].isin(keep).to_numpy()
        sub = base[sub_idx].reset_index(drop=True)
        sub_eligible = eligible[sub_idx]
        sub_engine = DailyICEngine(sub["trade_date"], {h: sub[f"forward_return_{h}d"] for h in HORIZONS}, sub_eligible)
        sub_amount = pd.to_numeric(sub["amount"], errors="coerce")

        def _judge_panel_values(name: str, family: str, source: str, wide_vals: pd.DataFrame) -> None:
            merged = sub[["symbol", "trade_date"]].merge(wide_vals, on=["symbol", "trade_date"], how="left")
            row = {"factor": name, "family": family, "source": source}
            row.update(_judge_factor(sub_engine, merged[name], regime_by_date, sub_amount,
                                     sub_eligible, sub["trade_date"]))
            rows.append(row)
            print(f"[+] {name:34} {row.get('verdict','?'):18} best={row.get('best_horizon','-')}")

        # missing alphas
        present = {c for c in factor_cols if c.startswith("alpha")}
        missing = [f"alpha{i:03d}" for i in range(1, 182) if f"alpha{i:03d}" not in present]
        if missing:
            try:
                from quantagent.factors.alpha181 import compute_alpha181
                wide = compute_alpha181(panel, names=missing, wide=True)
                wide["trade_date"] = pd.to_datetime(wide["trade_date"])
                for name in missing:
                    if name in wide.columns:
                        _judge_panel_values(name, "alpha181_missing_from_dataset", "recomputed",
                                            wide[["symbol", "trade_date", name]])
                    else:
                        rows.append({"factor": name, "family": "alpha181_missing_from_dataset",
                                     "source": "recomputed", "verdict": "not_computable"})
            except Exception as exc:  # noqa: BLE001
                print(f"missing-alpha recompute failed: {exc}")
                for name in missing:
                    rows.append({"factor": name, "family": "alpha181_missing_from_dataset",
                                 "source": "recompute_failed", "verdict": "not_computable",
                                 "error": str(exc)[:120]})

        for def_path in args.definitions:
            if not Path(def_path).exists():
                continue
            for definition in load_definitions(def_path):
                try:
                    vals = pd.to_numeric(definition.expr.evaluate(panel), errors="coerce")
                except Exception:  # noqa: BLE001
                    rows.append({"factor": definition.name, "family": "discovered_gp_llm",
                                 "source": def_path, "verdict": "not_computable"})
                    continue
                wide_vals = panel[["symbol", "trade_date"]].copy()
                wide_vals[definition.name] = vals.to_numpy()
                _judge_panel_values(definition.name, "discovered_gp_llm", def_path, wide_vals)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "factor_judgment_table.csv", index=False)
    summary = {
        "n_factors": len(table),
        "verdict_counts": table["verdict"].value_counts().to_dict(),
        "by_family": {fam: grp["verdict"].value_counts().to_dict()
                      for fam, grp in table.groupby("family")},
        "accepted_all_weather": table.loc[table.verdict == "all_weather", "factor"].tolist(),
        "accepted_robust_4y": table.loc[table.verdict == "robust_4y", "factor"].tolist(),
        "regime_specialists": table.loc[table.verdict == "regime_specialist", "factor"].tolist(),
        "window": args.start,
        "protocol": "eligible pool, horizons 5/20/60d, per-year 2022-2026, per-regime, capacity",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_family"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
