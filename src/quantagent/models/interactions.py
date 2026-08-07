"""Explicit nonlinearity: what kind, built how, and measured against what.

Five things get called "nonlinear multi-factor" and only two of them are the
same thing. This module names them, and builds the two that were missing.

``ModelClass`` is the taxonomy. It exists because the repository already
contained a beam search whose stage was labelled ``interaction_beam`` and whose
configuration flag was ``interaction_search``, but which searched *subsets* of
factors and scored each subset with an equal-weighted mean of centred
cross-sectional ranks. No product of two factors was ever formed. The search was
honest; the name was not, and a name that overstates the model class is how an
additive book ends up defended as a nonlinear one.

The distinction is not pedantry — it is the single sharpest empirical result in
Gu, Kelly & Xiu (2020). Their generalized linear model expands every predictor
into a second-order spline basis, which is a genuine *factor nonlinear
transform*, and it scores a monthly out-of-sample R² of 0.19% against 0.11% for
plain elastic net: essentially nothing. Their trees and neural networks, which
differ from the GLM precisely by admitting *interactions between* predictors,
score 0.33–0.40%. The paper states it directly: the GLM "uses spline functions
of individual features, but includes no interaction among features", and that is
why it fails to improve on linear methods. Curvature in one factor is cheap.
Conditioning one factor on another is where the gains were.

Two constructions are provided, both of which a linear estimator can consume, so
that "does interaction help?" can be answered without confounding it with "does
a tree help?":

``pairwise_interaction_features``
    ``x_i x_j`` on rank-normalised factors — the term a tree approximates with
    axis-aligned splits and a neural network with hidden units, made explicit so
    a ridge can price it and so its incremental contribution is attributable.

``regime_interaction_features``
    ``x_j ⊗ m_t`` — the same idea against the market state rather than another
    stock-level factor. This is Gu, Kelly & Xiu's baseline feature set, which is
    not the raw characteristics but their Kronecker product with the macro state
    vector (their equation 21, ``z_{i,t} = x_t ⊗ c_{i,t}``). The repository
    conditioned on regime in three places before this — a position-size
    multiplier, a sleeve blend weight, and a date filter — and all three act
    *after* the score exists. None of them let the model learn that a factor
    changes sign between regimes, because none of them reach the coefficient.

Pair selection is deliberately not "take the products with the highest IC". A
product of two predictive factors is itself predictive almost by construction,
so ranking on raw IC re-selects the main effects with extra steps. What is
selected on here is the IC of the product *after* both parents are projected out
cross-sectionally, which is zero unless the interaction carries something
neither parent does.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DATE_COLUMN = "trade_date"
SYMBOL_COLUMN = "symbol"
#: Separator for generated interaction column names. Chosen so a generated name
#: can never collide with a source factor name and can be split back apart.
INTERACTION_SEPARATOR = "__x__"
REGIME_SEPARATOR = "__regime__"


class ModelClass(str, Enum):
    """What a scoring component actually is.

    Components declare one of these so a downstream reader never has to infer
    the model class from a function name. The order is roughly increasing
    expressive power, but nothing depends on the order.
    """

    #: ``score = Σ_j w_j x_j`` on raw factor values.
    LINEAR_ADDITIVE = "linear_additive"
    #: ``score = Σ_j w_j rank(x_j)``. Still additive: the rank is a monotone
    #: per-factor transform and the blend across factors is a weighted sum.
    RANK_WEIGHTED_ADDITIVE = "rank_weighted_additive"
    #: ``score = Σ_j f_j(x_j)`` with ``f_j`` nonlinear — splines, buckets,
    #: winsorised ranks. Nonlinear in each factor, still additive *across*
    #: factors. Gu, Kelly & Xiu's GLM sits here and buys ~nothing.
    FACTOR_NONLINEAR_TRANSFORM = "factor_nonlinear_transform"
    #: Contains at least one ``x_i x_j`` term with ``i != j``.
    FACTOR_INTERACTION = "factor_interaction"
    #: Contains at least one ``x_j × m_t`` term against a market/regime state.
    REGIME_INTERACTION = "regime_interaction"
    #: A learner that can represent interactions without being told which ones
    #: (trees, boosted trees, neural networks).
    NONLINEAR_LEARNER = "nonlinear_learner"
    #: The *objective* is nonlinear (Huber, rank/IC loss, drawdown-penalised),
    #: independent of whether the functional form is.
    NONLINEAR_OBJECTIVE = "nonlinear_objective"
    #: A combination of separately fitted models (stacking, averaging, blending).
    #: Ensembling additive models yields an additive model.
    ENSEMBLE = "ensemble"

    @property
    def represents_interaction(self) -> bool:
        """True only for classes that can express ``∂²f/∂x_i∂x_j ≠ 0``."""
        return self in {
            ModelClass.FACTOR_INTERACTION,
            ModelClass.REGIME_INTERACTION,
            ModelClass.NONLINEAR_LEARNER,
        }

    @property
    def is_additive(self) -> bool:
        """True when the score is a sum of per-factor terms."""
        return self in {
            ModelClass.LINEAR_ADDITIVE,
            ModelClass.RANK_WEIGHTED_ADDITIVE,
            ModelClass.FACTOR_NONLINEAR_TRANSFORM,
        }


@dataclass(frozen=True)
class InteractionPair:
    """One selected ``x_i x_j`` term and the evidence that selected it."""

    left: str
    right: str
    #: Rank IC of the raw product against the forward return.
    raw_ic: float
    #: Rank IC of the product *after* both parents are projected out
    #: cross-sectionally. This is the number the pair was ranked on.
    orthogonal_ic: float
    #: Fraction of dates on which ``orthogonal_ic`` kept the sign of its mean.
    sign_stability: float

    @property
    def column(self) -> str:
        return f"{self.left}{INTERACTION_SEPARATOR}{self.right}"

    def as_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "left": self.left,
            "right": self.right,
            "rawIc": round(float(self.raw_ic), 6),
            "orthogonalIc": round(float(self.orthogonal_ic), 6),
            "signStability": round(float(self.sign_stability), 6),
        }


def cross_sectional_rank_normalise(
    panel: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
    """Per-date uniform rank of each column mapped onto ``[-1, 1]``.

    This is the Gu, Kelly & Xiu (2020) convention — they "cross-sectionally rank
    all stock characteristics period-by-period and map these ranks into the
    [-1,1] interval" — and it is what makes a product of two factors mean
    something. On raw scales ``x_i x_j`` is dominated by whichever factor has the
    larger units, so the "interaction" would mostly re-express one main effect.

    The mapping is ``2(r - 1)/(n - 1) - 1`` on the integer rank rather than the
    more obvious ``2(rank_pct - 0.5)``. ``rank(pct=True)`` returns ``r/n``, which
    spans ``(0, 1]`` and therefore centres on ``1/n``, not zero. That residual
    offset is harmless in an additive blend and is not harmless here: with a
    per-factor offset ``c``, the product ``(x_i + c)(x_j + c)`` carries
    ``c(x_i + x_j)`` of pure main effect, so an "interaction" column would
    smuggle in its own parents — exactly what the orthogonality test in
    :func:`select_interaction_pairs` exists to rule out. On a 40-name date the
    contamination is 2.5% of each parent.

    Only rows sharing a ``date_column`` value are compared, so this introduces no
    lookahead. Missing values map to 0.0, the cross-sectional median rank.
    """
    missing = [name for name in columns if name not in panel.columns]
    if missing:
        raise KeyError(f"columns missing from panel: {sorted(missing)}")
    if date_column not in panel.columns:
        raise KeyError(f"panel must contain {date_column}")
    if not len(columns):
        return pd.DataFrame(index=panel.index)

    numeric = panel[list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby(panel[date_column], sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    # A single usable name on a date has no cross-section to rank against.
    spread = (counts - 1.0).where(counts > 1)
    centred = 2.0 * (ranks - 1.0) / spread - 1.0
    return centred.fillna(0.0)


def _residualise_by_date(
    product: pd.Series,
    left: pd.Series,
    right: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Per-date residual of ``product`` on ``[1, left, right]``, in closed form.

    Two regressors plus an intercept is a 2x2 normal-equation system, so the
    whole panel resolves in a handful of grouped sums rather than one
    ``lstsq`` per date per pair. On the real panel that is the difference
    between a few seconds and a few hours — the loop version was O(pairs x
    dates) least-squares solves and made the honest selection rule too
    expensive to actually run, which is how selection rules quietly get
    replaced by cheaper dishonest ones.
    """
    frame = pd.DataFrame({"p": product, "a": left, "b": right})
    grouped = frame.groupby(dates, sort=False)
    # Within-date centring absorbs the intercept.
    centred = frame - grouped.transform("mean")
    products = pd.DataFrame(
        {
            "aa": centred["a"] * centred["a"],
            "bb": centred["b"] * centred["b"],
            "ab": centred["a"] * centred["b"],
            "ap": centred["a"] * centred["p"],
            "bp": centred["b"] * centred["p"],
        }
    )
    sums = products.groupby(dates, sort=False).transform("sum")
    determinant = sums["aa"] * sums["bb"] - sums["ab"] ** 2
    safe = determinant.abs() > 1e-12
    beta_a = np.where(safe, (sums["bb"] * sums["ap"] - sums["ab"] * sums["bp"]) / determinant, 0.0)
    beta_b = np.where(safe, (sums["aa"] * sums["bp"] - sums["ab"] * sums["ap"]) / determinant, 0.0)
    return centred["p"] - beta_a * centred["a"] - beta_b * centred["b"]


def _daily_rank_ic(
    scores: pd.Series,
    labels: pd.Series,
    dates: pd.Series,
    *,
    min_names: int = 5,
) -> pd.Series:
    """Per-date Spearman rank IC. Dates with too few names produce no entry."""
    frame = pd.DataFrame({"score": scores, "label": labels, "date": dates}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    values: dict[object, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        if group["score"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        correlation = group["score"].rank().corr(group["label"].rank())
        if pd.notna(correlation):
            values[date] = float(correlation)
    return pd.Series(values, dtype=float)


def select_interaction_pairs(
    train_panel: pd.DataFrame,
    columns: Sequence[str],
    label_column: str,
    *,
    top_n: int = 12,
    date_column: str = DATE_COLUMN,
    min_sign_stability: float = 0.55,
    min_abs_orthogonal_ic: float = 0.002,
    max_candidates: int = 400,
) -> list[InteractionPair]:
    """Rank ``x_i x_j`` terms by the IC they carry *beyond* their parents.

    Must be handed a training segment only. Nothing here reads a label outside
    the rows it is given, but the caller owns the fold boundary; passing the
    whole panel would select interactions using the period they are later
    scored on, which is the leak this whole exercise exists to avoid.

    Ranking on the raw IC of the product would be close to meaningless. If
    ``x_i`` and ``x_j`` both predict returns then so does their product, and the
    "interaction" would be selected for information its parents already carry.
    Each product is therefore regressed on its two parents within each date and
    scored on the residual, so a pair survives only when the *conditioning* —
    "``x_i`` matters more when ``x_j`` is high" — predicts something on its own.

    ``min_sign_stability`` additionally requires the residual IC to keep one
    sign across dates. A pair whose conditional effect flips sign every other
    day has an impressive absolute IC and no usable content.
    """
    if label_column not in train_panel.columns:
        raise KeyError(f"train_panel must contain {label_column}")
    usable = [name for name in columns if name in train_panel.columns]
    if len(usable) < 2:
        return []

    work = train_panel[[date_column, label_column, *usable]].copy()
    work = work.dropna(subset=[date_column, label_column])
    if work.empty:
        return []
    ranked = cross_sectional_rank_normalise(work, usable, date_column=date_column)
    labels = pd.to_numeric(work[label_column], errors="coerce")
    dates = work[date_column]

    candidates = list(combinations(usable, 2))
    if len(candidates) > max_candidates:
        # Deterministic truncation: keep the pairs among the factors with the
        # strongest standalone IC, so the shortlist is reproducible rather than
        # dependent on column order in the source parquet.
        standalone = {
            name: abs(float(_daily_rank_ic(ranked[name], labels, dates).mean() or 0.0))
            for name in usable
        }
        ordered = sorted(usable, key=lambda name: (-standalone.get(name, 0.0), name))
        keep = ordered[: max(2, int(np.sqrt(2 * max_candidates)) + 1)]
        candidates = [pair for pair in combinations(keep, 2)]

    results: list[InteractionPair] = []
    for left, right in candidates:
        product = ranked[left] * ranked[right]
        raw = _daily_rank_ic(product, labels, dates)
        # Project the parents out one date at a time: the conditional effect is
        # a cross-sectional statement, so residualising over the pooled panel
        # would leave time-series variation in the residual.
        residual = _residualise_by_date(product, ranked[left], ranked[right], dates)
        orthogonal = _daily_rank_ic(residual, labels, dates)
        if orthogonal.empty:
            continue
        mean_ic = float(orthogonal.mean())
        stability = float((np.sign(orthogonal) == np.sign(mean_ic)).mean())
        results.append(
            InteractionPair(
                left=left,
                right=right,
                raw_ic=float(raw.mean()) if not raw.empty else float("nan"),
                orthogonal_ic=mean_ic,
                sign_stability=stability,
            )
        )

    survivors = [
        pair
        for pair in results
        if abs(pair.orthogonal_ic) >= min_abs_orthogonal_ic
        and pair.sign_stability >= min_sign_stability
    ]
    survivors.sort(key=lambda pair: (-abs(pair.orthogonal_ic), pair.left, pair.right))
    return survivors[: max(0, int(top_n))]


def pairwise_interaction_features(
    panel: pd.DataFrame,
    pairs: Iterable[InteractionPair | tuple[str, str]],
    *,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
    """Materialise ``x_i x_j`` columns from rank-normalised parents.

    Returns a frame aligned to ``panel.index`` holding only the generated
    columns, so the caller decides whether to concatenate them onto the feature
    matrix or evaluate them alone.
    """
    normalised: list[tuple[str, str]] = []
    for pair in pairs:
        if isinstance(pair, InteractionPair):
            normalised.append((pair.left, pair.right))
        else:
            left, right = pair
            normalised.append((str(left), str(right)))
    if not normalised:
        return pd.DataFrame(index=panel.index)

    needed = sorted({name for pair in normalised for name in pair})
    ranked = cross_sectional_rank_normalise(panel, needed, date_column=date_column)
    out = pd.DataFrame(index=panel.index)
    for left, right in normalised:
        out[f"{left}{INTERACTION_SEPARATOR}{right}"] = ranked[left] * ranked[right]
    return out


def regime_interaction_features(
    panel: pd.DataFrame,
    columns: Sequence[str],
    regime_by_date: pd.Series,
    *,
    date_column: str = DATE_COLUMN,
    states: Sequence[str] | None = None,
    drop_reference_state: bool = True,
) -> pd.DataFrame:
    """Materialise ``x_j ⊗ 1[regime_t = s]`` — the ``x × regime`` term.

    This is the discrete form of Gu, Kelly & Xiu's ``z_{i,t} = x_t ⊗ c_{i,t}``:
    every factor is crossed with the market state so a linear estimator can
    learn a *different coefficient per regime*, including a sign flip. That is
    strictly more than a regime-dependent blend weight can express — a blend
    weight scales a factor's contribution by a positive number chosen per
    regime, so it can turn a factor down but never turn it around.

    One state is dropped by default. Keeping every state alongside the
    unconditional factors makes the design matrix exactly collinear (the state
    indicators sum to one), and the dropped state becomes the reference whose
    effect the unconditional factor carries.

    ``regime_by_date`` is indexed by date. Dates absent from it are assigned the
    reference state, which yields zeros in every generated column — the model
    then falls back to the unconditional coefficient rather than to a guess.
    """
    if date_column not in panel.columns:
        raise KeyError(f"panel must contain {date_column}")
    usable = [name for name in columns if name in panel.columns]
    if not usable:
        return pd.DataFrame(index=panel.index)

    regimes = pd.Series(regime_by_date).copy()
    regimes.index = pd.to_datetime(regimes.index, errors="coerce")
    regimes = regimes[~regimes.index.isna()].astype(str)
    panel_dates = pd.to_datetime(panel[date_column], errors="coerce")
    assigned = panel_dates.map(regimes)

    if states is None:
        observed = sorted(assigned.dropna().unique())
    else:
        observed = [str(state) for state in states]
    if not observed:
        return pd.DataFrame(index=panel.index)
    # Deterministic reference: the alphabetically first observed state, so the
    # generated column set does not depend on dictionary iteration order.
    reference = observed[0] if drop_reference_state else None
    emitted = [state for state in observed if state != reference]
    if not emitted:
        return pd.DataFrame(index=panel.index)

    ranked = cross_sectional_rank_normalise(panel, usable, date_column=date_column)
    out = pd.DataFrame(index=panel.index)
    for state in emitted:
        indicator = (assigned == state).astype(float).to_numpy()
        for name in usable:
            out[f"{name}{REGIME_SEPARATOR}{state}"] = ranked[name].to_numpy() * indicator
    return out


def describe_feature_block(columns: Sequence[str]) -> dict[str, object]:
    """Classify a feature list into main effects, pair terms and regime terms.

    Used by reports so a model card can state its own model class from the
    columns it was fitted on rather than from a label somebody typed.
    """
    pair_terms = [name for name in columns if INTERACTION_SEPARATOR in name]
    regime_terms = [name for name in columns if REGIME_SEPARATOR in name]
    main_terms = [
        name
        for name in columns
        if INTERACTION_SEPARATOR not in name and REGIME_SEPARATOR not in name
    ]
    classes: list[str] = []
    if main_terms:
        classes.append(ModelClass.RANK_WEIGHTED_ADDITIVE.value)
    if pair_terms:
        classes.append(ModelClass.FACTOR_INTERACTION.value)
    if regime_terms:
        classes.append(ModelClass.REGIME_INTERACTION.value)
    return {
        "mainEffectCount": len(main_terms),
        "pairInteractionCount": len(pair_terms),
        "regimeInteractionCount": len(regime_terms),
        "modelClasses": classes,
        "representsInteraction": bool(pair_terms or regime_terms),
    }


__all__ = [
    "DATE_COLUMN",
    "INTERACTION_SEPARATOR",
    "InteractionPair",
    "ModelClass",
    "REGIME_SEPARATOR",
    "SYMBOL_COLUMN",
    "cross_sectional_rank_normalise",
    "describe_feature_block",
    "pairwise_interaction_features",
    "regime_interaction_features",
    "select_interaction_pairs",
]
