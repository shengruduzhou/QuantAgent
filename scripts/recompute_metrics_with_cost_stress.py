"""Re-compute metrics from existing walk_forward_predictions.csv under varying
cost assumptions, properly scaling per-horizon returns to daily-equivalent
before annualising.

Each row in the prediction CSV represents one (symbol, trade_date, horizon)
where ``forward_return_{horizon}d`` is the H-day forward close-to-close return.
For an H-day held cross-section evaluated every trading day:

    avg_period_return  = mean of long-short returns per date
    avg_daily_return   = avg_period_return / H   (since each period spans H days)
    annualised_return  = (1 + avg_daily_return) ** 252 - 1

The 5d horizon is treated as the headline; per-horizon breakdown is also printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _rank_ic(label_col: str):
    def inner(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return float("nan")
        p = frame["prediction"].rank()
        r = frame[label_col].rank()
        if p.nunique() < 2 or r.nunique() < 2:
            return float("nan")
        return float(p.corr(r))
    return inner


def _long_short(label_col: str):
    def inner(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return 0.0
        ranks = frame["prediction"].rank(pct=True) - 0.5
        gross = ranks.abs().sum()
        if gross <= 0:
            return 0.0
        weights = ranks / gross
        return float((weights * frame[label_col]).sum())
    return inner


def metrics_per_horizon(preds: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    cost = cost_bps / 10_000.0
    rows: list[dict] = []
    for (fold, horizon), grp in preds.groupby(["fold_id", "horizon"]):
        h = int(horizon)
        label_col = f"forward_return_{h}d"
        if label_col not in grp.columns:
            continue
        sub = grp.dropna(subset=[label_col, "prediction"])
        if sub.empty:
            continue
        ic_by_date = sub.groupby("trade_date").apply(_rank_ic(label_col)).dropna()
        ret_by_date = sub.groupby("trade_date").apply(_long_short(label_col)).fillna(0.0)
        net_period = ret_by_date - cost
        n = max(int(len(net_period)), 1)
        avg_period = float(net_period.mean()) if not net_period.empty else 0.0
        avg_daily_eq = avg_period / h
        ann_ret = (1.0 + avg_daily_eq) ** 252 - 1.0 if avg_daily_eq > -1 else float("nan")
        nav = (1.0 + net_period).cumprod()
        dd = nav / nav.cummax() - 1.0 if not nav.empty else pd.Series(dtype=float)
        std_period = float(net_period.std(ddof=1)) if n > 1 else 0.0
        sharpe = (avg_daily_eq * 252) / (std_period * np.sqrt(252 / h) + 1e-12) if std_period > 0 else 0.0
        rows.append({
            "fold": int(fold), "horizon": h,
            "rank_ic_mean": float(ic_by_date.mean()) if not ic_by_date.empty else 0.0,
            "avg_period_return_bps": avg_period * 1e4,
            "avg_daily_return_bps": avg_daily_eq * 1e4,
            "annualised_return_pct": ann_ret * 100 if not np.isnan(ann_ret) else float("nan"),
            "annualised_sharpe": sharpe,
            "max_drawdown_pct": float(dd.min()) * 100 if not dd.empty else 0.0,
            "n_days": n,
        })
    return pd.DataFrame(rows)


def aggregate(per_horizon: pd.DataFrame, horizon: int) -> dict:
    """Headline = mean over folds for one horizon."""
    sub = per_horizon[per_horizon["horizon"] == horizon]
    if sub.empty:
        return {}
    return {
        "horizon": horizon,
        "rank_ic_mean": float(sub["rank_ic_mean"].mean()),
        "ICIR": float(sub["rank_ic_mean"].mean() / (sub["rank_ic_mean"].std(ddof=1) + 1e-12)),
        "avg_daily_return_bps": float(sub["avg_daily_return_bps"].mean()),
        "annualised_return_pct": float(sub["annualised_return_pct"].mean()),
        "annualised_sharpe": float(sub["annualised_sharpe"].mean()),
        "max_drawdown_pct": float(sub["max_drawdown_pct"].min()),
        "fold_count": int(len(sub)),
        "evaluated_days": int(sub["n_days"].sum()),
    }


def main() -> None:
    runs = [
        ("iter1 dropout=0.10 wd=1e-4",
         Path("runtime/models/v7_alpha_iter1/walk_forward_predictions.csv")),
        ("iter2 dropout=0.20 wd=1e-3",
         Path("runtime/models/v7_alpha_iter2/walk_forward_predictions.csv")),
    ]
    cost_grid = [0.0, 12.0, 25.0, 50.0, 100.0]
    horizons = [1, 5, 20, 60, 126]

    headline_rows: list[dict] = []
    full_records: list[dict] = []

    for run_name, path in runs:
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        preds = pd.read_csv(path, parse_dates=["trade_date"])
        for cost in cost_grid:
            per_h = metrics_per_horizon(preds, cost)
            for h in horizons:
                agg = aggregate(per_h, h)
                if agg:
                    agg["run"] = run_name
                    agg["cost_bps"] = cost
                    headline_rows.append(agg)
            full_records.append({
                "run": run_name,
                "cost_bps": cost,
                "per_fold_per_horizon": per_h.to_dict(orient="records"),
            })

    table = pd.DataFrame(headline_rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 25)
    for run_name, _ in runs:
        sub = table[table["run"] == run_name]
        if sub.empty:
            continue
        print(f"\n=== {run_name} ===")
        pivot_cols = ["cost_bps", "horizon", "rank_ic_mean", "ICIR",
                      "avg_daily_return_bps", "annualised_return_pct",
                      "annualised_sharpe", "max_drawdown_pct", "evaluated_days"]
        view = sub[pivot_cols].copy()
        view["rank_ic_mean"] = view["rank_ic_mean"].round(4)
        view["ICIR"] = view["ICIR"].round(3)
        view["avg_daily_return_bps"] = view["avg_daily_return_bps"].round(2)
        view["annualised_return_pct"] = view["annualised_return_pct"].round(2)
        view["annualised_sharpe"] = view["annualised_sharpe"].round(2)
        view["max_drawdown_pct"] = view["max_drawdown_pct"].round(1)
        print(view.to_string(index=False))

    out_path = Path("runtime/reports/v7/cost_stress.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "headline": headline_rows,
        "full": full_records,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
