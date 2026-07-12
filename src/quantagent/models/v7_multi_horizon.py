"""Hand-weighted heuristic multi-horizon baseline.

.. warning:: STATUS (2026-07-03, ARCHITECTURE_AUDIT.md §2)
   NOT the production model and NOT trained — fixed hand-tuned weights for
   the agentic V7 pipeline fallback. Production = FT-Transformer sleeves +
   configs/production_blend.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.v7.schemas import FactorApplicability, MultiHorizonAlpha, ThematicUniverseMember


V7_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120, 126)


@dataclass(frozen=True)
class V7MultiHorizonConfig:
    horizons: tuple[int, ...] = V7_HORIZONS
    volatility_floor: float = 0.05
    interval_width_multiplier: float = 1.65


class V7MultiHorizonBaselineModel:
    """Feature-driven V7 alpha baseline with separate horizon feature groups."""

    def __init__(self, config: V7MultiHorizonConfig | None = None) -> None:
        self.config = config or V7MultiHorizonConfig()

    def predict(
        self,
        feature_frame: pd.DataFrame,
        universe_members: list[ThematicUniverseMember],
        factor_applicability: Iterable[FactorApplicability] = (),
    ) -> dict[str, MultiHorizonAlpha]:
        if feature_frame.empty:
            return {}
        latest = _latest_rows(feature_frame)
        members = {member.symbol: member for member in universe_members}
        applicability = list(factor_applicability)
        output: dict[str, MultiHorizonAlpha] = {}
        for symbol, row in latest.iterrows():
            member = members.get(str(symbol))
            if member is None:
                continue
            horizon_scores = {
                horizon: _horizon_score(row, member, horizon, applicability)
                for horizon in self.config.horizons
            }
            volatility = max(self.config.volatility_floor, _safe_float(row.get("volatility_20d", row.get("realized_vol_20d", 0.20))))
            risk_penalty = min(1.0, member.fraud_risk_score / 100.0 * 0.45 + max(0.0, volatility - 0.25))
            expected = float(np.mean(list(horizon_scores.values())))
            interval = self.config.interval_width_multiplier * volatility / np.sqrt(252 / 20)
            confidence = float(np.clip(member.source_confidence * (1.0 - risk_penalty) * _factor_support(applicability, member), 0.05, 0.95))
            output[str(symbol)] = MultiHorizonAlpha(
                symbol=str(symbol),
                alpha_1d=horizon_scores.get(1, 0.0),
                alpha_5d=horizon_scores.get(5, 0.0),
                alpha_20d=horizon_scores.get(20, 0.0),
                alpha_60d=horizon_scores.get(60, 0.0),
                alpha_120d=horizon_scores.get(120, 0.0),
                alpha_126d=horizon_scores.get(126, 0.0),
                expected_return=expected,
                expected_excess_return=expected - _safe_float(row.get("benchmark_expected_return", 0.0)),
                volatility_forecast=volatility,
                downside_risk=max(0.0, volatility * 0.60 + risk_penalty * 0.20),
                confidence=confidence,
                conformal_confidence=float(np.clip(1.0 - interval, 0.05, 0.95)),
                prediction_interval_low=expected - interval,
                prediction_interval_high=expected + interval,
                rank_score=float(np.clip(_cross_section_rank(latest, symbol, "v7_alpha_seed") * 100.0, 0.0, 100.0)),
                regime_adjusted_score=float(np.clip(expected * 100.0 * confidence, -100.0, 100.0)),
                factor_contribution=_factor_contribution(row),
                evidence_contribution={member.theme: member.exposure_score / 100.0},
                risk_penalty=risk_penalty,
                final_alpha_score=float(np.clip(expected * 100.0 - risk_penalty * 25.0, -100.0, 100.0)),
            )
        return output


def predict_v7_multi_horizon_alpha(
    feature_frame: pd.DataFrame,
    universe_members: list[ThematicUniverseMember],
    factor_applicability: Iterable[FactorApplicability] = (),
    config: V7MultiHorizonConfig | None = None,
) -> dict[str, MultiHorizonAlpha]:
    return V7MultiHorizonBaselineModel(config).predict(feature_frame, universe_members, factor_applicability)


def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False).tail(1)
    data = data.set_index("symbol", drop=False)
    data["v7_alpha_seed"] = data.apply(_seed_score, axis=1)
    return data


def _horizon_score(
    row: pd.Series,
    member: ThematicUniverseMember,
    horizon: int,
    applicability: list[FactorApplicability],
) -> float:
    direct = row.get(f"model_alpha_{horizon}d", row.get(f"alpha_{horizon}d"))
    if direct is not None and not pd.isna(direct):
        return float(np.clip(direct, -1.0, 1.0))
    short = _score_columns(row, ("ret_1d", "ret_5d", "momentum_5d", "fund_flow_5d", "news_sentiment_score"))
    medium = _score_columns(row, ("ret_20d", "momentum_20d", "sector_rotation_score", "theme_strength", "earnings_revision_score"))
    long = _score_columns(row, ("policy_strength", "industry_fundamental_strength", "exposure_score", "fundamental_score", "quality_score", "margin_of_safety"))
    valuation = _safe_float(row.get("valuation_score", member.valuation_score)) / 100.0
    fraud_penalty = member.fraud_risk_score / 100.0
    support = _factor_support(applicability, member, horizon)
    if horizon <= 5:
        raw = 0.55 * short + 0.25 * medium + 0.10 * long + 0.10 * valuation
    elif horizon <= 20:
        raw = 0.25 * short + 0.45 * medium + 0.20 * long + 0.10 * valuation
    else:
        raw = 0.10 * short + 0.25 * medium + 0.50 * long + 0.15 * valuation
    raw *= support
    raw -= 0.25 * fraud_penalty
    return float(np.clip(raw, -1.0, 1.0))


def _score_columns(row: pd.Series, columns: tuple[str, ...]) -> float:
    values = [_normalize_feature(row.get(column)) for column in columns if column in row.index and not pd.isna(row.get(column))]
    if not values:
        return 0.0
    return float(np.mean(values))


def _normalize_feature(value: object) -> float:
    numeric = _safe_float(value)
    if abs(numeric) > 2.0:
        numeric = numeric / 100.0
    return float(np.tanh(numeric))


def _seed_score(row: pd.Series) -> float:
    return _score_columns(row, ("theme_strength", "fundamental_score", "exposure_score", "momentum_20d", "policy_strength"))


def _factor_support(
    applicability: Iterable[FactorApplicability],
    member: ThematicUniverseMember | None = None,
    horizon: int | None = None,
) -> float:
    applicable = list(applicability)
    if not applicable:
        return 0.80
    matched = []
    for item in applicable:
        if horizon is not None and item.horizon_days != horizon:
            continue
        if member is not None and item.applicable_theme and member.theme not in item.applicable_theme:
            continue
        if item.factor_lifecycle_stage in {"production", "validation"}:
            matched.append(max(0.0, item.rank_icir) + item.hit_rate)
    if not matched:
        return 0.65
    return float(np.clip(0.65 + np.mean(matched) * 0.20, 0.65, 1.10))


def _cross_section_rank(frame: pd.DataFrame, symbol: object, column: str) -> float:
    if column not in frame.columns or symbol not in frame.index:
        return 0.5
    ranks = frame[column].rank(pct=True)
    value = ranks.loc[symbol]
    return float(value) if np.isfinite(value) else 0.5


def _factor_contribution(row: pd.Series) -> dict[str, float]:
    return {
        "short_timing": _score_columns(row, ("ret_1d", "ret_5d", "momentum_5d", "news_sentiment_score")),
        "medium_theme": _score_columns(row, ("ret_20d", "sector_rotation_score", "theme_strength")),
        "long_fundamental": _score_columns(row, ("policy_strength", "industry_fundamental_strength", "fundamental_score", "quality_score")),
    }


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
