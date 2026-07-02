"""Walk-forward OOS evaluation: per-fold + overall IC / RankIC / ICIR + coverage.

Consumes the self-describing OOS predictions emitted by
``run_walk_forward_deep_training`` (columns ``symbol, trade_date, fold_id,
alpha_{h}d, ...``) joined against the dataset's realised forward-return labels
(``forward_return_{h}d``). Metrics are cross-sectional per trading day, then
aggregated by fold and overall — the standard rank-IC protocol used elsewhere in
the codebase, so numbers are comparable.

This is evaluation only: no look-ahead, no trading. Predictions are paired with
their realised forward return at the same ``(symbol, trade_date)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardEvalResult:
    metrics_by_fold: pd.DataFrame   # fold_id, horizon, n_obs, n_days, mean_ic, rank_ic, rank_icir, hit_rate
    overall: pd.DataFrame           # horizon, n_obs, n_days, mean_ic, rank_ic, rank_icir, hit_rate
    coverage: dict = field(default_factory=dict)


def _daily_rank_ic(frame: pd.DataFrame, alpha_col: str, label_col: str) -> pd.Series:
    """Per-day cross-sectional Spearman rank correlation of alpha vs label."""
    df = frame[["trade_date", alpha_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby("trade_date")
    df = df.assign(_ra=g[alpha_col].rank(), _rl=g[label_col].rank())
    daily = (
        df.groupby("trade_date")[["_ra", "_rl"]].corr().unstack().iloc[:, 1].dropna()
    )
    return daily


def _daily_pearson_ic(frame: pd.DataFrame, alpha_col: str, label_col: str) -> pd.Series:
    df = frame[["trade_date", alpha_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby("trade_date").apply(
        lambda s: s[alpha_col].corr(s[label_col]) if len(s) > 1 else np.nan,
        include_groups=False,
    ).dropna()


def _metrics(frame: pd.DataFrame, alpha_col: str, label_col: str) -> dict[str, float]:
    rank_daily = _daily_rank_ic(frame, alpha_col, label_col)
    ic_daily = _daily_pearson_ic(frame, alpha_col, label_col)
    paired = frame[[alpha_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    rank_mean = float(rank_daily.mean()) if not rank_daily.empty else float("nan")
    rank_std = float(rank_daily.std(ddof=0)) if len(rank_daily) > 1 else float("nan")
    rank_icir = rank_mean / rank_std if rank_std and rank_std > 1e-12 else float("nan")
    hit_rate = float((rank_daily > 0).mean()) if not rank_daily.empty else float("nan")
    return {
        "n_obs": int(len(paired)),
        "n_days": int(len(rank_daily)),
        "mean_ic": float(ic_daily.mean()) if not ic_daily.empty else float("nan"),
        "rank_ic": rank_mean,
        "rank_icir": float(rank_icir),
        "hit_rate": hit_rate,
    }


def evaluate_walk_forward_oos(
    oos_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    *,
    alpha_prefix: str = "alpha_",
    label_prefix: str = "forward_return_",
    output_dir: str | Path | None = None,
) -> WalkForwardEvalResult:
    """Per-fold + overall rank-IC/ICIR for each horizon, plus OOS coverage.

    ``labels`` only needs ``symbol, trade_date, forward_return_{h}d``; it is
    joined onto the predictions at the same key (no shifting — the forward
    return is already the realised future outcome from that date).
    """
    if oos_predictions is None or oos_predictions.empty:
        raise ValueError("evaluate_walk_forward_oos: empty OOS predictions")
    preds = oos_predictions.copy()
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce")
    lab = labels.copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"], errors="coerce")
    label_cols = [f"{label_prefix}{h}d" for h in horizons if f"{label_prefix}{h}d" in lab.columns]
    merged = preds.merge(
        lab[["symbol", "trade_date", *label_cols]], on="symbol trade_date".split(), how="left"
    )

    fold_rows: list[dict] = []
    overall_rows: list[dict] = []
    for h in horizons:
        alpha_col, label_col = f"{alpha_prefix}{h}d", f"{label_prefix}{h}d"
        if alpha_col not in merged.columns or label_col not in merged.columns:
            continue
        overall_rows.append({"horizon": h, **_metrics(merged, alpha_col, label_col)})
        if "fold_id" in merged.columns:
            for fold_id, grp in merged.groupby("fold_id"):
                fold_rows.append({"fold_id": int(fold_id), "horizon": h, **_metrics(grp, alpha_col, label_col)})

    metrics_by_fold = pd.DataFrame(fold_rows)
    overall = pd.DataFrame(overall_rows)
    coverage = {
        "n_oos_rows": int(len(preds)),
        "n_symbols": int(preds["symbol"].nunique()),
        "n_dates": int(preds["trade_date"].nunique()),
        "date_start": str(preds["trade_date"].min()),
        "date_end": str(preds["trade_date"].max()),
        "n_folds": int(preds["fold_id"].nunique()) if "fold_id" in preds.columns else 0,
        "label_match_rate": {
            f"{label_prefix}{h}d": float(merged[f"{label_prefix}{h}d"].notna().mean())
            for h in horizons if f"{label_prefix}{h}d" in merged.columns
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not metrics_by_fold.empty:
            metrics_by_fold.to_csv(out / "metrics_by_fold.csv", index=False)
        overall.to_csv(out / "metrics_overall.csv", index=False)
        (out / "oos_coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    return WalkForwardEvalResult(metrics_by_fold=metrics_by_fold, overall=overall, coverage=coverage)


__all__ = ["WalkForwardEvalResult", "evaluate_walk_forward_oos"]
