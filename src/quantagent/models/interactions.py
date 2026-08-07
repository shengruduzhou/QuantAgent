"""Leakage-safe nonlinear factor interaction construction.

The module keeps model-class semantics explicit. A rank transform, spline, or
ensemble of additive scores is not called a factor interaction. Genuine
interaction blocks are either ``x_i * x_j`` or ``x_j * market_state`` and are
selected/constructed without reading rows outside the caller-supplied training
segment.

Design anchors:
- Gu, Kelly & Xiu (2020): cross-sectional rank normalisation and the empirical
  importance of interactions rather than per-feature curvature alone.
- QuantAgent invariant: selection is train-only; transformation is same-date;
  generated features remain index-aligned with the source panel.
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
INTERACTION_SEPARATOR = "__x__"
REGIME_SEPARATOR = "__regime__"


class ModelClass(str, Enum):
    LINEAR_ADDITIVE = "linear_additive"
    RANK_WEIGHTED_ADDITIVE = "rank_weighted_additive"
    FACTOR_NONLINEAR_TRANSFORM = "factor_nonlinear_transform"
    FACTOR_INTERACTION = "factor_interaction"
    REGIME_INTERACTION = "regime_interaction"
    NONLINEAR_LEARNER = "nonlinear_learner"
    NONLINEAR_OBJECTIVE = "nonlinear_objective"
    ENSEMBLE = "ensemble"

    @property
    def represents_interaction(self) -> bool:
        return self in {
            ModelClass.FACTOR_INTERACTION,
            ModelClass.REGIME_INTERACTION,
            ModelClass.NONLINEAR_LEARNER,
        }

    @property
    def is_additive(self) -> bool:
        return self in {
            ModelClass.LINEAR_ADDITIVE,
            ModelClass.RANK_WEIGHTED_ADDITIVE,
            ModelClass.FACTOR_NONLINEAR_TRANSFORM,
        }


@dataclass(frozen=True)
class InteractionPair:
    left: str
    right: str
    raw_ic: float
    orthogonal_ic: float
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
    """Map same-date cross-sectional ranks exactly onto ``[-1, 1]``.

    Missing/singleton values map to 0 (neutral). No statistic is pooled across
    dates, so future rows cannot move an earlier row's representation.
    """
    missing = [name for name in columns if name not in panel.columns]
    if missing:
        raise KeyError(f"columns missing from panel: {sorted(missing)}")
    if date_column not in panel.columns:
        raise KeyError(f"panel must contain {date_column}")
    if not columns:
        return pd.DataFrame(index=panel.index)

    numeric = panel[list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby(panel[date_column], sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    spread = (counts - 1.0).where(counts > 1)
    centred = 2.0 * (ranks - 1.0) / spread - 1.0
    out = centred.fillna(0.0)
    if not out.index.equals(panel.index):
        raise RuntimeError("rank normalisation broke source-panel index alignment")
    return out


def _residualise_by_date(
    product: pd.Series,
    left: pd.Series,
    right: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Residualise product on [1,left,right] independently within each date."""
    frame = pd.DataFrame({"p": product, "a": left, "b": right}, index=product.index)
    grouped = frame.groupby(dates, sort=False)
    centred = frame - grouped.transform("mean")
    products = pd.DataFrame(
        {
            "aa": centred["a"] * centred["a"],
            "bb": centred["b"] * centred["b"],
            "ab": centred["a"] * centred["b"],
            "ap": centred["a"] * centred["p"],
            "bp": centred["b"] * centred["p"],
        },
        index=product.index,
    )
    sums = products.groupby(dates, sort=False).transform("sum")
    determinant = sums["aa"] * sums["bb"] - sums["ab"] ** 2
    safe = determinant.abs() > 1e-12
    beta_a = np.where(
        safe,
        (sums["bb"] * sums["ap"] - sums["ab"] * sums["bp"]) / determinant,
        0.0,
    )
    beta_b = np.where(
        safe,
        (sums["aa"] * sums["bp"] - sums["ab"] * sums["ap"]) / determinant,
        0.0,
    )
    residual = centred["p"] - beta_a * centred["a"] - beta_b * centred["b"]
    return pd.Series(residual, index=product.index, dtype=float)


def _daily_rank_ic(
    scores: pd.Series,
    labels: pd.Series,
    dates: pd.Series,
    *,
    min_names: int = 5,
) -> pd.Series:
    frame = pd.DataFrame(
        {"score": scores, "label": labels, "date": dates}, index=scores.index
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    values: dict[object, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        if group["score"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        correlation = group["score"].rank().corr(group["label"].rank())
        if pd.notna(correlation) and np.isfinite(correlation):
            values[date] = float(correlation)
    return pd.Series(values, dtype=float)


def _finite_abs_mean(series: pd.Series) -> float:
    """Stable sort score: empty/NaN IC is zero, never a NaN sort key."""
    if series.empty:
        return 0.0
    value = float(series.mean())
    return abs(value) if np.isfinite(value) else 0.0


def _candidate_pairs(
    ranked: pd.DataFrame,
    labels: pd.Series,
    dates: pd.Series,
    usable: Sequence[str],
    max_candidates: int,
) -> list[tuple[str, str]]:
    """Build a deterministic pair budget that never exceeds max_candidates."""
    budget = max(0, int(max_candidates))
    if budget == 0 or len(usable) < 2:
        return []
    all_pairs = list(combinations(sorted(usable), 2))
    if len(all_pairs) <= budget:
        return all_pairs

    standalone = {
        name: _finite_abs_mean(_daily_rank_ic(ranked[name], labels, dates))
        for name in usable
    }
    # Finite score first, factor name second: independent of input column order.
    ordered = sorted(set(usable), key=lambda name: (-standalone[name], name))

    # Find the smallest top-factor set that covers at least the budget, then
    # truncate the deterministic pair list. The previous implementation could
    # emit more pairs than max_candidates and allowed NaN IC into the sort key.
    keep_count = 2
    while keep_count < len(ordered) and keep_count * (keep_count - 1) // 2 < budget:
        keep_count += 1
    pairs = list(combinations(ordered[:keep_count], 2))
    return pairs[:budget]


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
    """Select train-only interactions by incremental IC beyond both parents.

    The product is residualised cross-sectionally on its two parents before
    scoring. This prevents a predictive main effect from being re-labelled as a
    nonlinear interaction. Candidate generation is finite, deterministic, and
    strictly bounded by ``max_candidates``.
    """
    if label_column not in train_panel.columns:
        raise KeyError(f"train_panel must contain {label_column}")
    if date_column not in train_panel.columns:
        raise KeyError(f"train_panel must contain {date_column}")
    usable = sorted({name for name in columns if name in train_panel.columns})
    if len(usable) < 2 or top_n <= 0 or max_candidates <= 0:
        return []

    work = train_panel[[date_column, label_column, *usable]].copy()
    work = work.dropna(subset=[date_column, label_column])
    if work.empty:
        return []
    ranked = cross_sectional_rank_normalise(work, usable, date_column=date_column)
    labels = pd.to_numeric(work[label_column], errors="coerce")
    dates = work[date_column]
    candidates = _candidate_pairs(ranked, labels, dates, usable, max_candidates)

    results: list[InteractionPair] = []
    for left, right in candidates:
        product = ranked[left] * ranked[right]
        raw = _daily_rank_ic(product, labels, dates)
        residual = _residualise_by_date(product, ranked[left], ranked[right], dates)
        orthogonal = _daily_rank_ic(residual, labels, dates)
        if orthogonal.empty:
            continue
        mean_ic = float(orthogonal.mean())
        if not np.isfinite(mean_ic):
            continue
        nonzero_sign = np.sign(mean_ic)
        stability = float((np.sign(orthogonal) == nonzero_sign).mean()) if nonzero_sign else 0.0
        raw_mean = float(raw.mean()) if not raw.empty else float("nan")
        results.append(
            InteractionPair(
                left=left,
                right=right,
                raw_ic=raw_mean,
                orthogonal_ic=mean_ic,
                sign_stability=stability,
            )
        )

    survivors = [
        pair for pair in results
        if abs(pair.orthogonal_ic) >= min_abs_orthogonal_ic
        and pair.sign_stability >= min_sign_stability
    ]
    survivors.sort(key=lambda pair: (-abs(pair.orthogonal_ic), pair.left, pair.right))
    return survivors[: int(top_n)]


def pairwise_interaction_features(
    panel: pd.DataFrame,
    pairs: Iterable[InteractionPair | tuple[str, str]],
    *,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
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
    if not out.index.equals(panel.index):
        raise RuntimeError("pairwise interaction block lost source-panel alignment")
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
    """Construct factor x market-state terms with one reference state dropped."""
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

    observed = sorted(assigned.dropna().unique()) if states is None else sorted({str(s) for s in states})
    if not observed:
        return pd.DataFrame(index=panel.index)
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
    if not out.index.equals(panel.index):
        raise RuntimeError("regime interaction block lost source-panel alignment")
    return out


def describe_feature_block(columns: Sequence[str]) -> dict[str, object]:
    pair_terms = [name for name in columns if INTERACTION_SEPARATOR in name]
    regime_terms = [name for name in columns if REGIME_SEPARATOR in name]
    main_terms = [
        name for name in columns
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
