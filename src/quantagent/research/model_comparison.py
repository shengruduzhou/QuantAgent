"""One protocol, several model classes, one verdict.

The question this module exists to answer is narrow and the repository could not
previously answer it: *does admitting factor interactions buy stable
out-of-sample alpha over the additive rank-weighted blend that is already in
production?* Answering it needs every arm to see the same folds, the same
features, the same costs and the same evaluation, because almost every way of
getting this wrong makes the complex model look better.

Arms, and what each one is (the taxonomy lives in
:mod:`quantagent.models.interactions`):

===========================  =====================================  ================
arm                          model class                            role
===========================  =====================================  ================
``linear_baseline``          ``rank_weighted_additive``             control
``linear_pair_interaction``  ``factor_interaction``                 ``x_i x_j``
``linear_regime_interaction````regime_interaction``                 ``x_j ⊗ m_t``
``linear_all_interaction``   ``factor_interaction``                 both
``gbm``                      ``nonlinear_learner``                  trees
``ensemble_stack``           ``ensemble``                           OOF stack
===========================  =====================================  ================

``linear_baseline`` is kept deliberately, and kept simple. Gu, Kelly & Xiu's
whole result is a *comparison*: trees and neural nets reach a monthly R² of
0.33-0.40% against 0.11% for elastic net, and their spline GLM — nonlinear in
each factor but additive across factors — manages only 0.19%. Those numbers mean
nothing without the linear column, and neither does anything measured here.
Their pairwise Diebold-Mariano table is worth remembering too: most of the
tree-versus-dimension-reduction gaps are *not* statistically significant. The
honest prior is that interaction gains are real but small and hard to establish,
not that a more expressive model wins by default.

Four evaluation dimensions, all reported, none optimised alone:

``prediction``
    Rank IC, ICIR, and out-of-sample R² benchmarked against a forecast of zero
    rather than the historical mean — Gu, Kelly & Xiu's convention, because for
    individual stock returns the historical mean is so noisy that benchmarking
    against it inflates R² by roughly three percentage points.
``economic``
    Long-only top-K net return, Sharpe and Calmar *after* costs. The return
    series overlaps by construction (every date carries an H-day label), so its
    Sharpe is reported as ``net_sharpe_overlapping`` and is optimistic in
    absolute terms — it is a valid basis for *comparing* arms that share the
    identical overlap, not a standalone claim. Target membership is determined
    from prediction only; whether a future label happens to be observable is
    never allowed to backfill a lower-ranked name into today's target.
``robustness``
    Per-fold consistency, PBO across the arm family, and a deflated Sharpe whose
    trial count is the number of arms actually run.
``trading``
    Turnover, cost drag, and the share of selected names that were untradable on
    the day they were selected.

Selection discipline: arms are ranked on the **selection** folds only. The final
contiguous ``holdout_folds`` are scored for every arm and reported, but no arm is
chosen using them. That ordering is the point — a holdout consulted during
selection is not a holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from quantagent.models.interactions import (
    InteractionPair,
    ModelClass,
    cross_sectional_rank_normalise,
    describe_feature_block,
    pairwise_interaction_features,
    regime_interaction_features,
    select_interaction_pairs,
)
from quantagent.quant_math.performance import (
    deflated_sharpe_ratio,
    max_drawdown,
    newey_west_t_stat,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from quantagent.training.splitters import WalkForwardSplitConfig, split_walk_forward

Verdict = Literal[
    "production_accepted",
    "hypothesis_rejected",
    "pipeline_failed",
    "data_invalid",
    "model_invalid",
]

LINEAR_BASELINE = "linear_baseline"


class ModelInvalid(Exception):
    """The estimator ran to completion and produced an unusable prediction.

    Distinct from an ordinary exception, which means the *pipeline* broke. A
    constant or non-finite prediction is a statement about the model: it fitted,
    it returned, and what it returned cannot be ranked. Collapsing the two into
    one bucket is how "the code crashed" and "this model class does not work
    here" end up looking the same in a research log.
    """


@dataclass(frozen=True)
class ComparisonConfig:
    """Everything the operator controls. Trial count is derived, never set."""

    label_column: str = "forward_return_5d"
    horizon_days: int = 5
    top_k: int = 30
    cost_bps: float = 12.0
    n_folds: int = 6
    #: Trailing folds reserved from arm selection entirely.
    holdout_folds: int = 2
    valid_size_days: int = 40
    min_train_days: int = 250
    embargo_days: int = 5
    ridge_alpha: float = 10.0
    max_interaction_pairs: int = 12
    #: Minimum Newey-West t-statistic on the per-date IC *difference* against
    #: the linear baseline before an arm counts as an improvement.
    min_incremental_t: float = 2.0
    #: Minimum *size* of the IC improvement, independent of its significance.
    #: A paired daily test over hundreds of dates will certify an IC gain of
    #: 0.001 as significant; that gain is real and worthless. Cross-sectional
    #: equity IC lives around 0.02-0.05, so 0.005 is roughly a tenth of the
    #: signal — below it, added model complexity is not paying rent.
    min_ic_delta: float = 0.005
    #: Minimum improvement in net annualised top-K return, after costs.
    min_net_return_delta: float = 0.01
    #: Minimum fraction of selection folds in which the arm must beat the
    #: baseline. A single fold cannot establish stability.
    min_fold_win_rate: float = 0.60
    max_pbo: float = 0.50
    min_symbols_per_date: int = 30
    gbm_estimators: int = 300
    gbm_leaves: int = 31
    gbm_learning_rate: float = 0.05
    #: Threads per LightGBM fit. ``-1`` uses every core, which is right for a
    #: single research run and wrong under a test suite or alongside another
    #: job — oversubscribing a 20-core box turned a 2-minute fit into 25.
    gbm_n_jobs: int = -1
    random_state: int = 1729

    def __post_init__(self) -> None:
        if self.n_folds <= self.holdout_folds:
            raise ValueError("n_folds must exceed holdout_folds")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

    @property
    def selection_folds(self) -> int:
        return self.n_folds - self.holdout_folds

    @property
    def periods_per_year(self) -> int:
        return max(1, int(round(252 / self.horizon_days)))


@dataclass(frozen=True)
class ArmResult:
    """One arm's out-of-sample record, split into selection and holdout."""

    name: str
    model_class: str
    feature_summary: dict[str, object]
    status: Literal["measured", "failed"]
    selection_metrics: dict[str, float] = field(default_factory=dict)
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    fold_ic: dict[int, float] = field(default_factory=dict)
    daily_ic: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "modelClass": self.model_class,
            "status": self.status,
            "featureSummary": self.feature_summary,
            "selection": {k: _round(v) for k, v in self.selection_metrics.items()},
            "holdout": {k: _round(v) for k, v in self.holdout_metrics.items()},
            "foldIc": {str(k): _round(v) for k, v in self.fold_ic.items()},
            "error": self.error,
        }


@dataclass(frozen=True)
class IncrementalTest:
    """Arm versus the linear baseline, on the selection folds."""

    arm: str
    ic_delta: float
    ic_delta_t_stat: float
    fold_win_rate: float
    net_return_delta: float
    passes: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "icDelta": _round(self.ic_delta),
            "icDeltaTStat": _round(self.ic_delta_t_stat),
            "foldWinRate": _round(self.fold_win_rate),
            "netReturnDelta": _round(self.net_return_delta),
            "passes": bool(self.passes),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ComparisonReport:
    config: ComparisonConfig
    arms: list[ArmResult]
    incremental: list[IncrementalTest]
    champion: str
    verdict: Verdict
    verdict_reasons: tuple[str, ...]
    n_trials: int
    pbo: float
    dsr_probability: float
    fold_windows: list[dict[str, str]]
    generated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "generatedAt": self.generated_at,
            "verdict": self.verdict,
            "verdictReasons": list(self.verdict_reasons),
            "champion": self.champion,
            "nTrials": int(self.n_trials),
            "pbo": _round(self.pbo),
            "dsrProbability": _round(self.dsr_probability),
            "config": {
                "labelColumn": self.config.label_column,
                "horizonDays": self.config.horizon_days,
                "topK": self.config.top_k,
                "costBps": self.config.cost_bps,
                "nFolds": self.config.n_folds,
                "holdoutFolds": self.config.holdout_folds,
                "minIncrementalT": self.config.min_incremental_t,
                "minFoldWinRate": self.config.min_fold_win_rate,
            },
            "foldWindows": self.fold_windows,
            "arms": [arm.as_dict() for arm in self.arms],
            "incrementalTests": [test.as_dict() for test in self.incremental],
        }


def _round(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(float(value)) else round(float(value), 6)
    return value


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge on an already-standardised design.

    Features arriving here are cross-sectional ranks in ``[-1, 1]`` and products
    of such ranks, so they are already on one scale; the residual centring below
    is what makes the penalty comparable across columns anyway.
    """
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    z_train = (train_x - center) / scale
    z_test = (test_x - center) / scale
    mean_y = float(train_y.mean())
    gram = z_train.T @ z_train + alpha * np.eye(z_train.shape[1])
    try:
        coef = np.linalg.solve(gram, z_train.T @ (train_y - mean_y))
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, z_train.T @ (train_y - mean_y), rcond=None)[0]
    return z_test @ coef + mean_y


def _gbm_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: ComparisonConfig,
) -> np.ndarray:
    import lightgbm as lgb  # type: ignore

    model = lgb.LGBMRegressor(
        n_estimators=config.gbm_estimators,
        num_leaves=config.gbm_leaves,
        learning_rate=config.gbm_learning_rate,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=config.random_state,
        n_jobs=config.gbm_n_jobs,
        verbose=-1,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=float)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _daily_rank_ic(frame: pd.DataFrame, label_column: str, min_names: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("trade_date", sort=True):
        usable = group[["prediction", label_column]].dropna()
        if len(usable) < min_names:
            continue
        if usable["prediction"].nunique() < 2 or usable[label_column].nunique() < 2:
            continue
        correlation = usable["prediction"].rank().corr(usable[label_column].rank())
        if pd.notna(correlation):
            values[pd.Timestamp(date)] = float(correlation)
    return pd.Series(values, dtype=float).sort_index()


def _topk_daily_returns(
    frame: pd.DataFrame,
    label_column: str,
    config: ComparisonConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Equal-weight top-K net daily-equivalent returns, turnover, untradable share.

    Target membership is based on information available at selection time:
    predictions (and, upstream, the PIT research universe). A future outcome
    label is never used as a membership filter. If any selected name lacks the
    outcome needed to evaluate that target, the economic observation for that
    date is omitted rather than replacing the name with a lower-ranked stock.
    The target state itself is still carried forward for subsequent turnover
    comparisons, so censoring cannot reset the portfolio history either.

    The label is an ``H``-day cumulative return, so it is divided by ``H`` to get
    a daily-equivalent rate before annualising — otherwise 252-fold compounding
    is applied to a 5-day number. The rebalance cost is spread over the same
    ``H`` days for the same reason. Cost is then charged in proportion to the
    one-way top-K replacement turnover actually implied by that rebalance; a
    book that does not trade must not pay the same cost as a fully replaced book.

    **The returned series overlaps.** Every date carries an ``H``-day forward
    return, so consecutive entries share ``H-1`` days of realisation and are
    strongly autocorrelated. The mean is unbiased; the *dispersion* is not, so
    any Sharpe computed from this series is optimistic and is named
    ``net_sharpe_overlapping`` to say so. ``quantagent.fusion.evaluation``
    decomposes into non-overlapping cohorts (``dates[offset::H]``) when a
    defensible standalone Sharpe is needed. Here the series exists to *compare*
    arms that all carry the identical overlap, which it supports, and the paired
    incremental test uses a HAC standard error precisely because of it.
    """
    horizon = max(1, config.horizon_days)
    cost_per_day = (config.cost_bps / 10_000.0) / horizon
    gross: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    untradable: dict[pd.Timestamp, float] = {}
    previous: set[str] = set()
    tradability_columns = [
        column
        for column in ("is_suspended", "is_limit_up", "is_st")
        if column in frame.columns
    ]
    for date, group in frame.groupby("trade_date", sort=True):
        rankable = group.dropna(subset=["prediction"])
        if len(rankable) < config.min_symbols_per_date:
            continue
        picked = rankable.nlargest(config.top_k, "prediction")
        if picked.empty:
            continue

        names = set(picked["symbol"].astype(str))
        target_turnover = (
            1.0 if not previous else len(names - previous) / max(1, len(names))
        )
        previous = names

        if tradability_columns:
            blocked = np.zeros(len(picked), dtype=bool)
            for column in tradability_columns:
                blocked |= picked[column].fillna(False).astype(bool).to_numpy()
            untradable[pd.Timestamp(date)] = float(blocked.mean())

        selected_returns = (
            pd.to_numeric(picked[label_column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        if selected_returns.isna().any():
            continue
        gross[pd.Timestamp(date)] = float(selected_returns.mean()) / horizon
        turnover[pd.Timestamp(date)] = target_turnover

    gross_series = pd.Series(gross, dtype=float).sort_index()
    turnover_series = pd.Series(turnover, dtype=float).sort_index()
    daily_cost = turnover_series.reindex(gross_series.index).fillna(0.0) * cost_per_day
    net = gross_series - daily_cost
    return net, turnover_series, pd.Series(untradable, dtype=float).sort_index()


def _metrics(
    frame: pd.DataFrame,
    label_column: str,
    config: ComparisonConfig,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    ic = _daily_rank_ic(frame, label_column, config.min_symbols_per_date)
    net, turnover, untradable = _topk_daily_returns(frame, label_column, config)
    predictions = pd.to_numeric(frame["prediction"], errors="coerce")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid = predictions.notna() & labels.notna()
    # Gu, Kelly & Xiu's R^2: denominator is the sum of squared realised returns
    # WITHOUT demeaning, because the historical mean is a far weaker benchmark
    # than a forecast of zero for individual stock returns.
    ss_res = float(((labels[valid] - predictions[valid]) ** 2).sum())
    ss_tot = float((labels[valid] ** 2).sum())
    r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")

    nav = (1.0 + net).cumprod() if not net.empty else pd.Series(dtype=float)
    drawdown = abs(max_drawdown(nav)) if len(nav) > 1 else float("nan")
    mean_daily = float(net.mean()) if not net.empty else float("nan")
    annual = (1.0 + mean_daily) ** 252 - 1.0 if np.isfinite(mean_daily) else float("nan")
    metrics = {
        # prediction quality
        "rank_ic_mean": float(ic.mean()) if not ic.empty else float("nan"),
        "rank_ic_ir": (
            float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else float("nan")
        ),
        "rank_ic_positive_rate": float((ic > 0).mean()) if not ic.empty else float("nan"),
        "r2_oos_vs_zero": r2_oos,
        "ic_observations": int(len(ic)),
        # economic value
        "net_annual_return": annual,
        "net_sharpe_overlapping": sharpe_ratio(net, periods_per_year=252) if len(net) > 1 else float("nan"),
        "max_drawdown": drawdown,
        "calmar": annual / drawdown if drawdown and np.isfinite(drawdown) and drawdown > 1e-9 else float("nan"),
        # trading reality
        "avg_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "cost_drag_annual": float(config.cost_bps / 10_000.0)
        * float(turnover.mean() if not turnover.empty else 0.0)
        * (252.0 / max(1, config.horizon_days)),
        "untradable_selection_rate": (
            float(untradable.mean()) if not untradable.empty else float("nan")
        ),
        "trading_days": int(len(net)),
    }
    return metrics, ic, net


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmSpec:
    name: str
    model_class: ModelClass
    use_pair_interactions: bool = False
    use_regime_interactions: bool = False
    estimator: Literal["ridge", "gbm", "stack"] = "ridge"


def _default_arms(include_gbm: bool) -> list[_ArmSpec]:
    arms = [
        _ArmSpec(LINEAR_BASELINE, ModelClass.RANK_WEIGHTED_ADDITIVE),
        _ArmSpec("linear_pair_interaction", ModelClass.FACTOR_INTERACTION, use_pair_interactions=True),
        _ArmSpec("linear_regime_interaction", ModelClass.REGIME_INTERACTION, use_regime_interactions=True),
        _ArmSpec(
            "linear_all_interaction",
            ModelClass.FACTOR_INTERACTION,
            use_pair_interactions=True,
            use_regime_interactions=True,
        ),
    ]
    if include_gbm:
        arms.append(_ArmSpec("gbm", ModelClass.NONLINEAR_LEARNER, estimator="gbm"))
        arms.append(_ArmSpec("ensemble_stack", ModelClass.ENSEMBLE, estimator="stack"))
    return arms


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


def _prepare_comparison_panel(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    config: ComparisonConfig,
) -> pd.DataFrame:
    """Prepare model input without conditioning inference rows on future labels.

    The target column is numeric-normalised but deliberately not part of the
    global ``dropna``. Training folds remove unknown outcomes immediately before
    fitting; validation/inference folds retain them so target construction never
    learns whether a future outcome will later be observable.
    """
    columns = ["trade_date", "symbol", config.label_column, *factor_columns] + [
        column
        for column in ("is_suspended", "is_limit_up", "is_st")
        if column in panel.columns
    ]
    work = panel[columns].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["symbol"] = work["symbol"].astype(str)
    work[config.label_column] = pd.to_numeric(work[config.label_column], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol"])
    return work.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def run_model_comparison(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    config: ComparisonConfig | None = None,
    regime_by_date: pd.Series | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> ComparisonReport:
    """Fit every arm on identical folds and return one verdict.

    ``panel`` must carry ``trade_date``, ``symbol``, ``config.label_column`` and
    the ``factor_columns``. Outcome labels may be missing on inference dates;
    they are required only for training and for an economic/predictive metric to
    be observed. They never determine target membership. Tradability flags
    (``is_suspended``, ``is_limit_up``, ``is_st``) are used when present only as
    diagnostics in this research evaluator; production economics remain blocked
    until targets are evaluated through the strict A-share execution simulator.
    """
    cfg = config or ComparisonConfig()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- data validity -------------------------------------------------
    missing = {"trade_date", "symbol", cfg.label_column} - set(panel.columns)
    if missing:
        return _invalid_report(
            cfg, generated_at, "data_invalid", (f"panel missing columns: {sorted(missing)}",)
        )
    usable_factors = [name for name in factor_columns if name in panel.columns]
    if len(usable_factors) < 2:
        return _invalid_report(
            cfg,
            generated_at,
            "data_invalid",
            (f"need >= 2 usable factor columns, found {len(usable_factors)}",),
        )

    work = _prepare_comparison_panel(panel, usable_factors, cfg)
    if work.empty or not bool(work[cfg.label_column].notna().any()):
        return _invalid_report(
            cfg,
            generated_at,
            "data_invalid",
            ("panel has no usable labelled observations for model fitting",),
        )

    folds = split_walk_forward(
        work,
        config=WalkForwardSplitConfig(
            n_splits=cfg.n_folds,
            valid_size_days=cfg.valid_size_days,
            min_train_days=cfg.min_train_days,
            embargo_days=cfg.embargo_days,
            # Purge the label horizon: a 5-day forward return computed on the
            # last training day resolves inside the validation window.
            purge_days=cfg.horizon_days,
            mode="expanding",
            anchor="end",
        ),
    )
    if len(folds) < cfg.holdout_folds + 2:
        return _invalid_report(
            cfg,
            generated_at,
            "data_invalid",
            (
                f"walk-forward produced {len(folds)} folds; need at least "
                f"{cfg.holdout_folds + 2} (>=2 selection + {cfg.holdout_folds} holdout)",
            ),
        )
    selection_fold_ids = {fold.fold_id for fold in folds[: len(folds) - cfg.holdout_folds]}
    fold_windows = [
        {
            "foldId": str(fold.fold_id),
            "role": "selection" if fold.fold_id in selection_fold_ids else "holdout",
            "trainStart": str(fold.train_dates[0].date()),
            "trainEnd": str(fold.train_dates[1].date()),
            "validStart": str(fold.valid_dates[0].date()),
            "validEnd": str(fold.valid_dates[1].date()),
        }
        for fold in folds
    ]

    arms = _default_arms(include_gbm=_lightgbm_available())
    n_trials = len(arms)
    predictions: dict[str, list[pd.DataFrame]] = {arm.name: [] for arm in arms}
    failures: dict[str, str] = {}
    #: Arms whose estimator returned an unusable prediction, as opposed to arms
    #: whose machinery broke. Kept apart so the verdict can name the right one.
    invalid_models: set[str] = set()
    total_steps = max(1, len(folds) * len(arms))
    completed = 0

    for fold in folds:
        train = work.iloc[fold.train_idx].dropna(subset=[cfg.label_column])
        test = work.iloc[fold.valid_idx]
        if train.empty or test.empty:
            continue

        # Pair selection is fitted on this fold's labelled training segment only.
        pairs: list[InteractionPair] = select_interaction_pairs(
            train,
            usable_factors,
            cfg.label_column,
            top_n=cfg.max_interaction_pairs,
        )
        blocks = _build_feature_blocks(
            train, test, usable_factors, pairs, regime_by_date, cfg
        )

        for arm in arms:
            completed += 1
            if progress is not None:
                progress(f"fold {fold.fold_id} · {arm.name}", completed, total_steps)
            if arm.name in failures:
                continue
            try:
                frame = _fit_arm(arm, blocks, train, test, cfg)
            except Exception as exc:  # noqa: BLE001 - classified below
                failures[arm.name] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, ModelInvalid):
                    invalid_models.add(arm.name)
                continue
            frame["fold_id"] = fold.fold_id
            frame["role"] = "selection" if fold.fold_id in selection_fold_ids else "holdout"
            predictions[arm.name].append(frame)

    # ---- assemble arm results ------------------------------------------
    results: list[ArmResult] = []
    for arm in arms:
        summary = _arm_feature_summary(arm, usable_factors)
        if arm.name in failures:
            results.append(
                ArmResult(
                    name=arm.name,
                    model_class=arm.model_class.value,
                    feature_summary=summary,
                    status="failed",
                    error=failures[arm.name],
                )
            )
            continue
        frames = predictions[arm.name]
        if not frames:
            results.append(
                ArmResult(
                    name=arm.name,
                    model_class=arm.model_class.value,
                    feature_summary=summary,
                    status="failed",
                    error="arm produced no fold predictions",
                )
            )
            continue
        combined = pd.concat(frames, ignore_index=True)
        selection = combined[combined["role"] == "selection"]
        holdout = combined[combined["role"] == "holdout"]
        selection_metrics, selection_ic, selection_net = _metrics(
            selection, cfg.label_column, cfg
        )
        holdout_metrics, _, _ = (
            _metrics(holdout, cfg.label_column, cfg)
            if not holdout.empty
            else ({}, pd.Series(dtype=float), pd.Series(dtype=float))
        )
        fold_ic = {
            int(fold_id): float(
                _daily_rank_ic(group, cfg.label_column, cfg.min_symbols_per_date).mean()
            )
            for fold_id, group in selection.groupby("fold_id")
        }
        results.append(
            ArmResult(
                name=arm.name,
                model_class=arm.model_class.value,
                feature_summary=summary,
                status="measured",
                selection_metrics=selection_metrics,
                holdout_metrics=holdout_metrics,
                fold_ic=fold_ic,
                daily_ic=selection_ic,
                daily_returns=selection_net,
            )
        )

    return _decide(
        results, cfg, n_trials, fold_windows, generated_at, invalid_models=invalid_models
    )


def _build_feature_blocks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor_columns: Sequence[str],
    pairs: Sequence[InteractionPair],
    regime_by_date: pd.Series | None,
    config: ComparisonConfig,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Rank-normalise once per fold and build every optional block from it.

    Normalisation is cross-sectional and per date, so train and test can be
    transformed independently without either seeing the other. That is exactly
    why the rank convention is used here rather than a fitted z-score: a scaler
    fitted on train and applied to test would be correct but would also make the
    interaction terms depend on the training distribution, which muddies the
    attribution of any gain.
    """
    blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    blocks["main"] = (
        cross_sectional_rank_normalise(train, factor_columns),
        cross_sectional_rank_normalise(test, factor_columns),
    )
    if pairs:
        blocks["pair"] = (
            pairwise_interaction_features(train, pairs),
            pairwise_interaction_features(test, pairs),
        )
    if regime_by_date is not None and len(regime_by_date):
        train_regime = regime_interaction_features(train, factor_columns, regime_by_date)
        test_regime = regime_interaction_features(test, factor_columns, regime_by_date)
        # Both segments must expose the same columns or the fitted coefficient
        # vector will not line up. A regime absent from training gets a zero
        # column in test rather than a coefficient invented from nothing.
        shared = sorted(set(train_regime.columns) | set(test_regime.columns))
        if shared:
            blocks["regime"] = (
                train_regime.reindex(columns=shared, fill_value=0.0),
                test_regime.reindex(columns=shared, fill_value=0.0),
            )
    return blocks


def _stack_blocks(
    blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_parts = [blocks[name][0] for name in names if name in blocks]
    test_parts = [blocks[name][1] for name in names if name in blocks]
    train_frame = pd.concat(train_parts, axis=1)
    test_frame = pd.concat(test_parts, axis=1)
    train_values = np.nan_to_num(train_frame.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    test_values = np.nan_to_num(test_frame.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return train_values, test_values, list(train_frame.columns)


def _fit_arm(
    arm: _ArmSpec,
    blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ComparisonConfig,
) -> pd.DataFrame:
    names = ["main"]
    if arm.use_pair_interactions:
        names.append("pair")
    if arm.use_regime_interactions:
        names.append("regime")
    train_x, test_x, _ = _stack_blocks(blocks, names)
    train_y = train[config.label_column].to_numpy(dtype=float)

    if arm.estimator == "ridge":
        prediction = _ridge_fit_predict(train_x, train_y, test_x, config.ridge_alpha)
    elif arm.estimator == "gbm":
        prediction = _gbm_fit_predict(train_x, train_y, test_x, config)
    else:
        prediction = _stack_predict(blocks, train, test, config)

    if not np.all(np.isfinite(prediction)):
        raise ModelInvalid(f"{arm.name} produced non-finite predictions")
    if float(np.nanstd(prediction)) <= 1e-15:
        raise ModelInvalid(
            f"{arm.name} produced a constant prediction; rank IC is undefined and "
            "the metric layer would report it as 0.0"
        )

    out = test[["trade_date", "symbol", config.label_column]].copy()
    for column in ("is_suspended", "is_limit_up", "is_st"):
        if column in test.columns:
            out[column] = test[column].to_numpy()
    out["prediction"] = prediction
    return out


def _stack_predict(
    blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ComparisonConfig,
) -> np.ndarray:
    """OOF stack of the additive linear model and the tree model.

    The meta-weights are fitted on out-of-fold predictions produced *inside* the
    training segment by a further time-ordered split. Fitting them on in-sample
    predictions would hand the blend to whichever base learner overfits harder,
    which is the classic way a stack ends up worse than its own components.
    """
    dates = np.sort(train["trade_date"].unique())
    # The inner split needs room for a purge gap on top of both segments.
    if len(dates) < 4 + config.horizon_days:
        raise ValueError("stacking needs at least four distinct training dates plus the purge gap")
    cut_index = int(len(dates) * 0.7)
    # Purge the label horizon at the inner seam too. The meta-learner is fitted
    # on the inner-validation predictions, so without this its weights are
    # chosen partly on labels that overlap the inner-training window — the same
    # leak the outer folds already purge, reintroduced one level down.
    inner_train_end = dates[cut_index]
    inner_valid_start = dates[min(len(dates) - 1, cut_index + config.horizon_days)]
    inner_train_mask = train["trade_date"].to_numpy() <= inner_train_end
    inner_valid_mask = train["trade_date"].to_numpy() >= inner_valid_start
    if not inner_train_mask.any() or not inner_valid_mask.any():
        raise ValueError("inner stacking split produced an empty segment")

    # The outer blocks are already index-aligned to ``train``, so the inner
    # split is a positional mask over them. Re-normalising the inner segment
    # separately would give the meta-learner a different feature space from the
    # one the base learners are refitted on below.
    train_main, test_main, _ = _stack_blocks(blocks, ["main"])
    x_inner_train = train_main[inner_train_mask]
    x_inner_valid = train_main[inner_valid_mask]
    y_inner_train = train[config.label_column].to_numpy(dtype=float)[inner_train_mask]
    y_inner_valid = train[config.label_column].to_numpy(dtype=float)[inner_valid_mask]

    oof_linear = _ridge_fit_predict(
        x_inner_train, y_inner_train, x_inner_valid, config.ridge_alpha
    )
    oof_gbm = _gbm_fit_predict(x_inner_train, y_inner_train, x_inner_valid, config)
    meta_design = np.column_stack(
        [np.ones(len(oof_linear)), _zscore(oof_linear), _zscore(oof_gbm)]
    )
    weights, *_ = np.linalg.lstsq(meta_design, y_inner_valid, rcond=None)

    full_y = train[config.label_column].to_numpy(dtype=float)
    base_linear = _ridge_fit_predict(train_main, full_y, test_main, config.ridge_alpha)
    base_gbm = _gbm_fit_predict(train_main, full_y, test_main, config)
    return (
        weights[0] + weights[1] * _zscore(base_linear) + weights[2] * _zscore(base_gbm)
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 1e-15:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _arm_feature_summary(arm: _ArmSpec, factor_columns: Sequence[str]) -> dict[str, object]:
    summary = describe_feature_block(list(factor_columns))
    summary["declaredModelClass"] = arm.model_class.value
    summary["representsInteraction"] = bool(
        arm.use_pair_interactions
        or arm.use_regime_interactions
        or arm.model_class is ModelClass.NONLINEAR_LEARNER
        or arm.model_class is ModelClass.ENSEMBLE
    )
    summary["estimator"] = arm.estimator
    return summary


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _incremental_test(
    arm: ArmResult,
    baseline: ArmResult,
    config: ComparisonConfig,
) -> IncrementalTest:
    """Paired per-date IC difference with a HAC t-statistic.

    Pairing on the date is what makes this test sharp: both arms saw the same
    cross-section, so the common component of daily IC — which is mostly market
    state, and is enormous relative to the difference — cancels. Gu, Kelly & Xiu
    reach for the same device, adapting Diebold-Mariano to cross-sectional
    average errors for exactly this reason.
    """
    joined = pd.concat(
        {"arm": arm.daily_ic, "baseline": baseline.daily_ic}, axis=1, join="inner"
    ).dropna()
    reasons: list[str] = []
    if len(joined) < 20:
        return IncrementalTest(
            arm=arm.name,
            ic_delta=float("nan"),
            ic_delta_t_stat=float("nan"),
            fold_win_rate=float("nan"),
            net_return_delta=float("nan"),
            passes=False,
            reasons=("fewer than 20 paired observation dates",),
        )
    delta = (joined["arm"] - joined["baseline"]).astype(float)
    t_stat = newey_west_t_stat(delta)
    wins = [
        1.0 if arm.fold_ic.get(fold, float("-inf")) > baseline.fold_ic.get(fold, float("inf")) else 0.0
        for fold in sorted(set(arm.fold_ic) & set(baseline.fold_ic))
    ]
    win_rate = float(np.mean(wins)) if wins else float("nan")
    net_delta = float(
        arm.selection_metrics.get("net_annual_return", float("nan"))
        - baseline.selection_metrics.get("net_annual_return", float("nan"))
    )

    mean_delta = float(delta.mean())
    if not np.isfinite(t_stat) or t_stat < config.min_incremental_t:
        reasons.append(
            f"IC-difference t={t_stat:.3f} below the pre-registered {config.min_incremental_t:.2f}"
        )
    if not np.isfinite(mean_delta) or mean_delta < config.min_ic_delta:
        # Significance and size are separate questions and both have to be
        # answered. A stable +0.001 IC over 400 paired dates clears any t-test
        # and does not justify carrying a second model into production.
        reasons.append(
            f"IC improvement {mean_delta:+.4f} below the materiality floor "
            f"{config.min_ic_delta:.4f}"
        )
    if not np.isfinite(win_rate) or win_rate < config.min_fold_win_rate:
        reasons.append(
            f"beat the baseline in {win_rate:.0%} of selection folds, below "
            f"{config.min_fold_win_rate:.0%}"
        )
    if not np.isfinite(net_delta) or net_delta < config.min_net_return_delta:
        reasons.append(
            f"net top-K return improvement {net_delta:+.4f} below "
            f"{config.min_net_return_delta:.4f} after costs"
        )
    return IncrementalTest(
        arm=arm.name,
        ic_delta=float(delta.mean()),
        ic_delta_t_stat=float(t_stat),
        fold_win_rate=win_rate,
        net_return_delta=net_delta,
        passes=not reasons,
        reasons=tuple(reasons),
    )


def _decide(
    results: list[ArmResult],
    config: ComparisonConfig,
    n_trials: int,
    fold_windows: list[dict[str, str]],
    generated_at: str,
    invalid_models: set[str] | None = None,
) -> ComparisonReport:
    measured = [arm for arm in results if arm.status == "measured"]
    failed = [arm for arm in results if arm.status == "failed"]
    invalid = set(invalid_models or ())
    baseline = next((arm for arm in measured if arm.name == LINEAR_BASELINE), None)

    if baseline is None:
        # Without the control there is no comparison to make, whatever the other
        # arms did. Which of the two failures it was still matters: a baseline
        # that returned a constant is a model problem, a baseline that threw is
        # a pipeline problem.
        return ComparisonReport(
            config=config,
            arms=results,
            incremental=[],
            champion="",
            verdict="model_invalid" if LINEAR_BASELINE in invalid else "pipeline_failed",
            verdict_reasons=tuple(
                [f"{arm.name}: {arm.error}" for arm in failed]
                or ["linear baseline produced no result"]
            ),
            n_trials=n_trials,
            pbo=float("nan"),
            dsr_probability=float("nan"),
            fold_windows=fold_windows,
            generated_at=generated_at,
        )

    incremental = [
        _incremental_test(arm, baseline, config)
        for arm in measured
        if arm.name != LINEAR_BASELINE
    ]

    # PBO and DSR over the arm family, on selection-fold returns only.
    returns_matrix = pd.DataFrame(
        {arm.name: arm.daily_returns for arm in measured}
    ).dropna(how="all")
    pbo = float("nan")
    if returns_matrix.shape[1] >= 2 and len(returns_matrix) >= 8:
        partitions = min(16, len(returns_matrix))
        partitions -= partitions % 2
        if partitions >= 4:
            try:
                pbo = float(
                    probability_of_backtest_overfitting(
                        returns_matrix.fillna(0.0), n_partitions=partitions
                    )
                )
            except ValueError:
                pbo = float("nan")

    passing = [test for test in incremental if test.passes]
    if passing:
        best = max(passing, key=lambda test: test.ic_delta_t_stat)
        champion = best.arm
    else:
        champion = LINEAR_BASELINE

    dsr = float("nan")
    champion_arm = next((arm for arm in measured if arm.name == champion), None)
    if champion_arm is not None and len(returns_matrix.columns) >= 2:
        per_period = np.asarray(
            [
                float(sharpe_ratio(returns_matrix[column].dropna(), periods_per_year=252))
                / np.sqrt(252.0)
                for column in returns_matrix.columns
                if np.isfinite(sharpe_ratio(returns_matrix[column].dropna(), periods_per_year=252))
            ],
            dtype=float,
        )
        if per_period.size >= 2:
            dsr = float(
                deflated_sharpe_ratio(
                    champion_arm.daily_returns,
                    per_period,
                    periods_per_year=252,
                    n_trials=max(n_trials, per_period.size),
                )
            )

    reasons: list[str] = []
    verdict: Verdict
    every_challenger_failed = bool(failed) and all(
        arm.status == "failed" for arm in results if arm.name != LINEAR_BASELINE
    )
    if every_challenger_failed:
        # Nothing was compared. Report *why* nothing was compared: every
        # challenger returning an unusable prediction says something about the
        # model class on this data; every challenger throwing says something
        # about the code.
        failed_names = {arm.name for arm in failed}
        verdict = "model_invalid" if failed_names <= invalid else "pipeline_failed"
        reasons = [f"{arm.name}: {arm.error}" for arm in failed]
    elif not passing:
        # Every nonlinear arm ran and none cleared the pre-registered bar. That
        # is a *result*: the interaction hypothesis was tested and refused.
        verdict = "hypothesis_rejected"
        reasons = [
            f"{test.arm}: " + "; ".join(test.reasons) for test in incremental
        ] or ["no non-baseline arm was measured"]
        if failed:
            reasons.extend(f"{arm.name} did not run: {arm.error}" for arm in failed)
    elif np.isfinite(pbo) and pbo > config.max_pbo:
        verdict = "hypothesis_rejected"
        reasons = [
            f"{champion} beat the baseline but the search PBO is {pbo:.3f}, "
            f"above the pre-registered {config.max_pbo:.2f}"
        ]
    else:
        verdict = "production_accepted"
        reasons = [
            f"{champion} beat {LINEAR_BASELINE} on paired IC "
            f"(t={next(t.ic_delta_t_stat for t in passing if t.arm == champion):.2f}), "
            "in a majority of selection folds, and on net return after costs"
        ]
        if failed:
            reasons.append(
                "note: " + ", ".join(f"{arm.name} did not run" for arm in failed)
            )

    return ComparisonReport(
        config=config,
        arms=results,
        incremental=incremental,
        champion=champion,
        verdict=verdict,
        verdict_reasons=tuple(reasons),
        n_trials=n_trials,
        pbo=pbo,
        dsr_probability=dsr,
        fold_windows=fold_windows,
        generated_at=generated_at,
    )


def _invalid_report(
    config: ComparisonConfig,
    generated_at: str,
    verdict: Verdict,
    reasons: tuple[str, ...],
) -> ComparisonReport:
    return ComparisonReport(
        config=config,
        arms=[],
        incremental=[],
        champion="",
        verdict=verdict,
        verdict_reasons=reasons,
        n_trials=0,
        pbo=float("nan"),
        dsr_probability=float("nan"),
        fold_windows=[],
        generated_at=generated_at,
    )


def save_comparison_report(report: ComparisonReport, output_dir: str | Path) -> Path:
    """Persist the report so every claim in a write-up has a file behind it."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "model_comparison.json"
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


__all__ = [
    "ArmResult",
    "ComparisonConfig",
    "ComparisonReport",
    "IncrementalTest",
    "LINEAR_BASELINE",
    "Verdict",
    "run_model_comparison",
    "save_comparison_report",
]
