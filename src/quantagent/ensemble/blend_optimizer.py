"""Tune horizon-ensemble blend weights from OOS predictions.

Given the three per-horizon ``predictions.parquet`` files produced by
``train-v8-deep`` and the realised forward returns from the gold
dataset, this module fits a non-negative simplex of three weights
(short / mid / long) that maximises an OOS objective. Default
objective is the average per-date Spearman rank IC against
``forward_return_20d`` — a horizon-agnostic, scale-invariant target
that rewards correct cross-sectional ranking, which is exactly what
the downstream top-K selector consumes.

A simple grid + local refinement search over the 2-simplex is enough:
the objective is non-convex but smooth in 0.05 increments, and we
care about robustness across an OOS date split, not finding a
parameter-tuned local optimum. Production callers should split the
OOS window into K date folds and use the mean fold IC; this module
exposes both single-shot and K-fold variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HORIZONS = ("short_5d", "mid_5d_30d", "long_30d_120d")


@dataclass(frozen=True)
class BlendObjective:
    """Configurable scoring rule for blend candidate weights."""

    metric: str = "rank_ic"
    target_label: str = "forward_return_20d"
    top_k: int = 30
    drawdown_penalty: float = 1.0
    turnover_penalty: float = 0.05

    def score(
        self,
        composite: pd.DataFrame,
        realized: pd.DataFrame,
        *,
        key_columns: tuple[str, ...] = ("trade_date", "symbol"),
    ) -> float:
        """Score one blend output against realised returns."""
        if composite.empty or realized.empty:
            return float("nan")
        realized_clean = realized.drop(
            columns=[c for c in ("composite_score", "alpha_score") if c in realized.columns],
            errors="ignore",
        )
        merged = composite.merge(realized_clean, on=list(key_columns), how="inner")
        if merged.empty:
            return float("nan")
        if self.metric == "rank_ic":
            return _per_date_rank_ic(merged, "composite_score", self.target_label)
        if self.metric == "topk_return":
            return _per_date_topk_return(merged, "composite_score", self.target_label, k=self.top_k)
        if self.metric == "topk_excess_return":
            return _per_date_topk_excess_return(merged, "composite_score", self.target_label, k=self.top_k)
        if self.metric == "topk_utility":
            return _per_date_topk_utility(
                merged,
                "composite_score",
                self.target_label,
                k=self.top_k,
                drawdown_penalty=self.drawdown_penalty,
                turnover_penalty=self.turnover_penalty,
            )
        raise ValueError(f"unknown metric: {self.metric}")


def _per_date_rank_ic(
    merged: pd.DataFrame,
    score_col: str,
    label_col: str,
) -> float:
    if score_col not in merged.columns or label_col not in merged.columns:
        return float("nan")
    sub = merged[["trade_date", score_col, label_col]].dropna()
    if sub.empty:
        return float("nan")
    grouped = sub.groupby("trade_date", sort=False)
    rows: list[float] = []
    for _, g in grouped:
        if len(g) < 5:
            continue
        s = g[score_col].rank(method="average")
        r = g[label_col].rank(method="average")
        c = s.corr(r, method="pearson")
        if pd.notna(c):
            rows.append(float(c))
    return float(np.mean(rows)) if rows else float("nan")


def _per_date_topk_return(
    merged: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    k: int = 30,
) -> float:
    if score_col not in merged.columns or label_col not in merged.columns:
        return float("nan")
    sub = merged[["trade_date", score_col, label_col]].dropna()
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(["trade_date", score_col], ascending=[True, False])
    sub["rank"] = sub.groupby("trade_date").cumcount()
    selected = sub[sub["rank"] < k]
    if selected.empty:
        return float("nan")
    per_date = selected.groupby("trade_date")[label_col].mean()
    if per_date.empty:
        return float("nan")
    # Annualise: 20-day fwd return × ~12.6 trading periods/year
    mean = float(per_date.mean())
    return mean * (252.0 / 20.0)


def _per_date_topk_excess_return(
    merged: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    k: int = 30,
) -> float:
    """Annualised top-K return minus same-date cross-sectional benchmark.

    This is a return-first objective for horizon fusion. It rewards blends
    whose selected names beat the available universe on the same dates,
    rather than blends that merely produce a high rank correlation.
    """
    if score_col not in merged.columns or label_col not in merged.columns:
        return float("nan")
    sub = merged[["trade_date", score_col, label_col]].dropna()
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(["trade_date", score_col], ascending=[True, False])
    sub["rank"] = sub.groupby("trade_date").cumcount()
    selected = sub[sub["rank"] < k]
    if selected.empty:
        return float("nan")
    top = selected.groupby("trade_date")[label_col].mean()
    bench = sub.groupby("trade_date")[label_col].mean()
    aligned = pd.concat([top.rename("top"), bench.rename("bench")], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    # Infer annualisation from the label name when possible.
    horizon = 20.0
    if label_col.startswith("forward_return_") and label_col.endswith("d"):
        try:
            horizon = float(label_col.removeprefix("forward_return_").removesuffix("d"))
        except ValueError:
            horizon = 20.0
    return float((aligned["top"] - aligned["bench"]).mean() * (252.0 / max(1.0, horizon)))


def _per_date_topk_utility(
    merged: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    k: int = 30,
    drawdown_penalty: float = 1.0,
    turnover_penalty: float = 0.05,
) -> float:
    """Return/drawdown/turnover utility for choosing top-K names.

    This is still a label-based proxy, not a full execution simulation. It is
    intentionally more portfolio-like than RankIC: selected names form an
    equal-weight sleeve per date, same-date universe mean is the benchmark,
    drawdown is computed on the sleeve excess-return path, and turnover is
    approximated by selected-set churn across consecutive dates.
    """
    if score_col not in merged.columns or label_col not in merged.columns:
        return float("nan")
    sub = merged[["trade_date", "symbol", score_col, label_col]].dropna()
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(["trade_date", score_col], ascending=[True, False])
    sub["rank"] = sub.groupby("trade_date").cumcount()
    selected = sub[sub["rank"] < k].copy()
    if selected.empty:
        return float("nan")
    top = selected.groupby("trade_date")[label_col].mean()
    bench = sub.groupby("trade_date")[label_col].mean()
    excess = (top - bench).dropna().sort_index()
    if excess.empty:
        return float("nan")
    horizon = _infer_label_horizon(label_col)
    ann_excess = float(excess.mean() * (252.0 / max(1.0, horizon)))
    nav = (1.0 + excess.fillna(0.0)).cumprod()
    drawdown = float(-(nav / nav.cummax() - 1.0).min()) if len(nav) else 0.0
    turnover = _selected_set_turnover(selected)
    return float(ann_excess - drawdown_penalty * drawdown - turnover_penalty * turnover)


def _selected_set_turnover(selected: pd.DataFrame) -> float:
    sets: list[set[str]] = []
    for _, g in selected.groupby("trade_date", sort=True):
        sets.append(set(g["symbol"].astype(str)))
    if len(sets) <= 1:
        return 0.0
    rows = []
    for prev, cur in zip(sets, sets[1:]):
        denom = max(1, len(prev | cur))
        rows.append(1.0 - len(prev & cur) / denom)
    return float(np.mean(rows)) if rows else 0.0


def _infer_label_horizon(label_col: str) -> float:
    horizon = 20.0
    if label_col.startswith("forward_return_") and label_col.endswith("d"):
        try:
            horizon = float(label_col.removeprefix("forward_return_").removesuffix("d"))
        except ValueError:
            horizon = 20.0
    return horizon


def _simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Enumerate the 2-simplex at the given step in 3 dims."""
    out: list[tuple[float, float, float]] = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


def _apply_weights(
    per_horizon: dict[str, pd.DataFrame],
    weights: tuple[float, float, float],
) -> pd.DataFrame:
    """Blend per-horizon prediction frames into a composite_score frame."""
    wshort, wmid, wlong = weights
    frames = []
    for hz, w in zip(HORIZONS, (wshort, wmid, wlong)):
        if hz not in per_horizon or per_horizon[hz].empty:
            continue
        f = per_horizon[hz][["trade_date", "symbol", "alpha_score"]].copy()
        f = f.rename(columns={"alpha_score": f"score_{hz}"})
        f[f"score_{hz}"] = pd.to_numeric(f[f"score_{hz}"], errors="coerce").fillna(0.0)
        frames.append((f, w))
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "composite_score"])
    merged: pd.DataFrame | None = None
    score_cols: list[tuple[str, float]] = []
    for f, w in frames:
        score_col = [c for c in f.columns if c.startswith("score_")][0]
        score_cols.append((score_col, w))
        merged = f if merged is None else merged.merge(f, on=["trade_date", "symbol"], how="outer")
    assert merged is not None
    for col, _ in score_cols:
        merged[col] = merged[col].fillna(0.0)
    merged["composite_score"] = sum(merged[col] * w for col, w in score_cols)
    return merged[["trade_date", "symbol", "composite_score"] + [c for c, _ in score_cols]]


@dataclass(frozen=True)
class BlendSearchResult:
    """Result of a blend-weight grid search."""

    best_weights: tuple[float, float, float]
    best_score: float
    grid_scores: list[tuple[tuple[float, float, float], float]] = field(default_factory=list)
    fold_scores: dict[tuple[float, float, float], float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "best_weights": {
                "short_5d": self.best_weights[0],
                "mid_5d_30d": self.best_weights[1],
                "long_30d_120d": self.best_weights[2],
            },
            "best_score": self.best_score,
            "top10_grid": [
                {"weights": list(w), "score": s}
                for w, s in sorted(self.grid_scores, key=lambda kv: -kv[1])[:10]
            ],
        }


@dataclass(frozen=True)
class RegimeBlendSearchResult:
    """Regime-specific horizon blend result with a robust global fallback."""

    global_result: BlendSearchResult
    regime_results: dict[str, BlendSearchResult] = field(default_factory=dict)
    regime_days: dict[str, int] = field(default_factory=dict)
    skipped_regimes: dict[str, str] = field(default_factory=dict)
    min_regime_days: int = 40

    def weights_for(self, regime: str | None) -> tuple[float, float, float]:
        if regime and regime in self.regime_results:
            return self.regime_results[regime].best_weights
        return self.global_result.best_weights

    def as_dict(self) -> dict[str, object]:
        return {
            "global": self.global_result.as_dict(),
            "regimes": {
                regime: result.as_dict()
                for regime, result in sorted(self.regime_results.items())
            },
            "regime_days": dict(sorted(self.regime_days.items())),
            "skipped_regimes": dict(sorted(self.skipped_regimes.items())),
            "min_regime_days": self.min_regime_days,
        }


def load_predictions(deep_run_dir: Path) -> dict[str, pd.DataFrame]:
    """Read predictions.parquet for each of the three horizons."""
    out: dict[str, pd.DataFrame] = {}
    for hz in HORIZONS:
        p = deep_run_dir / hz / "predictions.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "alpha_score" not in df.columns:
            score_cols = [c for c in df.columns if c not in ("trade_date", "symbol")]
            if score_cols:
                df = df.rename(columns={score_cols[0]: "alpha_score"})
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        out[hz] = df[["trade_date", "symbol", "alpha_score"]]
    return out


def load_realized_returns(
    gold_path: Path,
    *,
    target_label: str = "forward_return_20d",
    symbol_universe: Iterable[str] | None = None,
    date_min: pd.Timestamp | None = None,
    date_max: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read realised forward returns from the gold dataset for the OOS window."""
    df = pd.read_parquet(gold_path, columns=["symbol", "trade_date", target_label])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if symbol_universe is not None:
        df = df[df["symbol"].isin(list(symbol_universe))]
    if date_min is not None:
        df = df[df["trade_date"] >= date_min]
    if date_max is not None:
        df = df[df["trade_date"] <= date_max]
    return df.dropna(subset=[target_label]).reset_index(drop=True)


def _prediction_dates_and_symbols(
    per_horizon: Mapping[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, set[str]]:
    all_dates = pd.Index([], dtype="datetime64[ns]")
    for frame in per_horizon.values():
        all_dates = all_dates.union(pd.Index(frame["trade_date"].unique()))
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    symbols = {sym for frame in per_horizon.values() for sym in frame["symbol"].unique()}
    return all_dates, symbols


def _fold_edges_from_dates(
    dates: pd.DatetimeIndex,
    *,
    n_folds: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if dates.empty:
        return []
    if n_folds <= 1:
        return [(dates[0], dates[-1])]
    fold_edges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    step_size = max(1, len(dates) // n_folds)
    for i in range(n_folds):
        lo_idx = i * step_size
        hi_idx = min(len(dates) - 1, (i + 1) * step_size - 1)
        if lo_idx >= len(dates):
            break
        fold_edges.append((dates[lo_idx], dates[hi_idx]))
    return fold_edges


def _score_blend_grid(
    per_horizon: Mapping[str, pd.DataFrame],
    realized: pd.DataFrame,
    *,
    objective: BlendObjective,
    dates: pd.DatetimeIndex,
    step: float,
    n_folds: int,
) -> BlendSearchResult:
    """Score simplex grid on a chosen date subset."""
    fold_edges = _fold_edges_from_dates(dates, n_folds=n_folds)
    base = _apply_weights(dict(per_horizon), (1.0, 0.0, 0.0))
    base = base[base["trade_date"].isin(dates)].copy()
    realized_clean = realized.drop(
        columns=[c for c in ("composite_score", "alpha_score") if c in realized.columns],
        errors="ignore",
    )
    merged_base = base.merge(realized_clean, on=["trade_date", "symbol"], how="inner")
    if merged_base.empty:
        raise ValueError("no OOS overlap between predictions and realised labels")
    for hz in HORIZONS:
        col = f"score_{hz}"
        if col not in merged_base.columns:
            merged_base[col] = 0.0
        merged_base[col] = pd.to_numeric(merged_base[col], errors="coerce").fillna(0.0)
    grid = _simplex_grid(step=step)
    grid_scores: list[tuple[tuple[float, float, float], float]] = []
    fold_scores: dict[tuple[float, float, float], float] = {}
    for weights in grid:
        if not any(weights):
            continue
        scored = merged_base.copy()
        scored["composite_score"] = sum(
            scored[f"score_{hz}"] * weight
            for hz, weight in zip(HORIZONS, weights)
        )
        per_fold = []
        for lo, hi in fold_edges:
            window = scored[
                (scored["trade_date"] >= lo) & (scored["trade_date"] <= hi)
            ]
            score = _score_merged_for_objective(objective, window)
            if pd.notna(score):
                per_fold.append(score)
        if not per_fold:
            continue
        mean_score = float(np.mean(per_fold))
        grid_scores.append((weights, mean_score))
        fold_scores[weights] = mean_score

    if not grid_scores:
        raise ValueError("no grid points produced a finite score — check OOS overlap")
    grid_scores.sort(key=lambda kv: -kv[1])
    best_weights, best_score = grid_scores[0]
    return BlendSearchResult(
        best_weights=best_weights,
        best_score=best_score,
        grid_scores=grid_scores,
        fold_scores=fold_scores,
    )


def _score_merged_for_objective(objective: BlendObjective, merged: pd.DataFrame) -> float:
    """Score a frame that already contains composite_score and the target label."""
    if merged.empty:
        return float("nan")
    if objective.metric == "rank_ic":
        return _per_date_rank_ic(merged, "composite_score", objective.target_label)
    if objective.metric == "topk_return":
        return _per_date_topk_return(merged, "composite_score", objective.target_label, k=objective.top_k)
    if objective.metric == "topk_excess_return":
        return _per_date_topk_excess_return(merged, "composite_score", objective.target_label, k=objective.top_k)
    if objective.metric == "topk_utility":
        return _per_date_topk_utility(
            merged,
            "composite_score",
            objective.target_label,
            k=objective.top_k,
            drawdown_penalty=objective.drawdown_penalty,
            turnover_penalty=objective.turnover_penalty,
        )
    raise ValueError(f"unknown metric: {objective.metric}")


def _normalise_regime_by_date(regime_by_date: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a Timestamp-indexed Series of regime labels."""
    if isinstance(regime_by_date, pd.DataFrame):
        if "regime" not in regime_by_date.columns:
            raise ValueError("regime_by_date DataFrame must include a 'regime' column")
        series = regime_by_date["regime"].copy()
        if "trade_date" in regime_by_date.columns:
            series.index = pd.to_datetime(regime_by_date["trade_date"], errors="coerce")
    else:
        series = regime_by_date.copy()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()].dropna()
    return series.astype(str)


def optimize_blend_weights(
    deep_run_dir: Path,
    *,
    gold_path: Path,
    objective: BlendObjective | None = None,
    step: float = 0.05,
    n_folds: int = 3,
) -> BlendSearchResult:
    """Search over the 3-horizon simplex and return the OOS-best weight.

    K-fold means: split the OOS dates into ``n_folds`` contiguous
    chunks and score each weight by the mean fold score, so that no
    single window dominates. ``n_folds=1`` falls back to single-shot
    scoring.
    """
    objective = objective or BlendObjective()
    per_horizon = load_predictions(deep_run_dir)
    if not per_horizon:
        raise FileNotFoundError(f"no predictions.parquet under {deep_run_dir}/{{horizon}}/")
    # Intersect the OOS date range
    all_dates, symbols = _prediction_dates_and_symbols(per_horizon)
    if all_dates.empty:
        raise ValueError("no OOS dates found in predictions")
    realized = load_realized_returns(
        gold_path,
        target_label=objective.target_label,
        symbol_universe=symbols,
        date_min=all_dates[0], date_max=all_dates[-1],
    )
    return _score_blend_grid(
        per_horizon,
        realized,
        objective=objective,
        dates=all_dates,
        step=step,
        n_folds=n_folds,
    )


def save_blend_result(
    result: BlendSearchResult,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return output_path


def optimize_regime_aware_blend_weights(
    deep_run_dir: Path,
    *,
    gold_path: Path,
    regime_by_date: pd.Series | pd.DataFrame,
    objective: BlendObjective | None = None,
    step: float = 0.05,
    n_folds: int = 3,
    min_regime_days: int = 40,
) -> RegimeBlendSearchResult:
    """Search global and per-regime horizon weights.

    The regime layer changes alpha composition, not gross exposure. Sparse
    regimes deliberately fall back to the global result to avoid a brittle
    one-window optimum.
    """
    objective = objective or BlendObjective()
    per_horizon = load_predictions(deep_run_dir)
    if not per_horizon:
        raise FileNotFoundError(f"no predictions.parquet under {deep_run_dir}/{{horizon}}/")
    all_dates, symbols = _prediction_dates_and_symbols(per_horizon)
    if all_dates.empty:
        raise ValueError("no OOS dates found in predictions")
    realized = load_realized_returns(
        gold_path,
        target_label=objective.target_label,
        symbol_universe=symbols,
        date_min=all_dates[0],
        date_max=all_dates[-1],
    )
    global_result = _score_blend_grid(
        per_horizon,
        realized,
        objective=objective,
        dates=all_dates,
        step=step,
        n_folds=n_folds,
    )

    regimes = _normalise_regime_by_date(regime_by_date)
    aligned = pd.Series(index=all_dates, dtype="object")
    aligned.loc[aligned.index.intersection(regimes.index)] = regimes.reindex(
        aligned.index.intersection(regimes.index)
    )
    regime_results: dict[str, BlendSearchResult] = {}
    regime_days: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for regime in sorted(v for v in aligned.dropna().unique()):
        regime_dates = pd.DatetimeIndex(aligned[aligned == regime].index)
        regime_days[str(regime)] = int(len(regime_dates))
        if len(regime_dates) < int(min_regime_days):
            skipped[str(regime)] = f"only {len(regime_dates)} OOS dates"
            continue
        try:
            regime_results[str(regime)] = _score_blend_grid(
                per_horizon,
                realized,
                objective=objective,
                dates=regime_dates,
                step=step,
                n_folds=n_folds,
            )
        except ValueError as exc:
            skipped[str(regime)] = str(exc)

    return RegimeBlendSearchResult(
        global_result=global_result,
        regime_results=regime_results,
        regime_days=regime_days,
        skipped_regimes=skipped,
        min_regime_days=int(min_regime_days),
    )


def save_regime_blend_result(
    result: RegimeBlendSearchResult,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return output_path


def write_blended_composite(
    deep_run_dir: Path,
    *,
    weights: tuple[float, float, float],
    output_path: Path,
) -> Path:
    """Apply weights and write the resulting composite_score parquet."""
    per_horizon = load_predictions(deep_run_dir)
    composite = _apply_weights(per_horizon, weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.to_parquet(output_path, index=False)
    return output_path


def write_regime_aware_composite(
    deep_run_dir: Path,
    *,
    regime_by_date: pd.Series | pd.DataFrame,
    result: RegimeBlendSearchResult,
    output_path: Path,
) -> Path:
    """Write a composite score that uses per-date regime-specific weights."""
    per_horizon = load_predictions(deep_run_dir)
    composite = _apply_weights(per_horizon, (1.0, 0.0, 0.0))
    regimes = _normalise_regime_by_date(regime_by_date)
    composite["trade_date"] = pd.to_datetime(composite["trade_date"], errors="coerce")
    composite["regime"] = composite["trade_date"].map(regimes).fillna("global")
    weights_by_regime = {
        regime: result.weights_for(regime)
        for regime in sorted(composite["regime"].dropna().unique())
    }
    for idx, hz in enumerate(HORIZONS):
        col = f"score_{hz}"
        if col not in composite.columns:
            composite[col] = 0.0
        weight_col = f"weight_{hz}"
        composite[weight_col] = composite["regime"].map(
            {regime: weights[idx] for regime, weights in weights_by_regime.items()}
        ).fillna(result.global_result.best_weights[idx])
    composite["composite_score"] = sum(
        composite[f"score_{hz}"].fillna(0.0) * composite[f"weight_{hz}"]
        for hz in HORIZONS
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.to_parquet(output_path, index=False)
    return output_path


__all__ = [
    "BlendObjective",
    "RegimeBlendSearchResult",
    "BlendSearchResult",
    "HORIZONS",
    "load_predictions",
    "load_realized_returns",
    "optimize_blend_weights",
    "optimize_regime_aware_blend_weights",
    "save_blend_result",
    "save_regime_blend_result",
    "write_blended_composite",
    "write_regime_aware_composite",
]
