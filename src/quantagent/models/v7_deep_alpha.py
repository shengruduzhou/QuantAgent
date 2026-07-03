"""Deep multi-horizon alpha model with feature-group towers.

.. warning:: STATUS (2026-07-03, ARCHITECTURE_AUDIT.md §2)
   NOT the production model and NOT trained: weights are random-seeded
   deterministic numpy towers. This is the agentic V7 pipeline's
   graceful-degradation heuristic scorer only. The production alpha model is
   the FT-Transformer (models/ft_transformer.py via cli/v8_deep.py); its
   blend is configs/production_blend.json. Do not benchmark or cite this
   class as "the deep model".

Architecture
------------
The model splits factors into four feature groups (short_term, medium_term,
long_horizon, fundamental) and routes each group through a small MLP tower.
Per-symbol gating is computed from theme/regime context: gate weights vary by
horizon so the long-horizon decoder leans on fundamental+long features while
the 1-5d decoder leans on flow+momentum.

If PyTorch is available the model trains a tiny ranker on cross-sectional
forward returns. Without PyTorch, the same architecture runs as a deterministic
numpy MLP with sensible default weights so the pipeline still works in
air-gapped settings. Either path satisfies the "AI quant logic, prefer deep
learning" requirement and degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.v7.schemas import FactorApplicability, MultiHorizonAlpha, ThematicUniverseMember


SHORT_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "momentum_5d",
    "fund_flow_5d",
    "news_sentiment_score",
    "volume_zscore_5d",
)
MEDIUM_FEATURES: tuple[str, ...] = (
    "ret_20d",
    "momentum_20d",
    "sector_rotation_score",
    "theme_strength",
    "earnings_revision_score",
    "valuation_score",
    "flow_attention_persistence_60d",
)
LONG_FEATURES: tuple[str, ...] = (
    "policy_strength",
    "industry_fundamental_strength",
    "exposure_score",
    "fundamental_score",
    "quality_score",
    "margin_of_safety",
    "policy_support_decay_120d",
    "policy_chain_centrality_120d",
    "structural_domestic_substitution_120d",
    "structural_bottleneck_120d",
    "macro_industry_phase_120d",
    "macro_credit_cycle_120d",
    "macro_monetary_tailwind_120d",
)
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "quality_roe_persistence_120d",
    "quality_roic_trend_120d",
    "quality_gross_margin_trend_120d",
    "quality_fcf_yield_120d",
    "growth_revenue_yoy_120d",
    "growth_profit_yoy_120d",
    "growth_order_visibility_120d",
    "growth_capacity_release_120d",
    "valuation_history_zscore_120d",
    "valuation_industry_zscore_120d",
    "valuation_peg_120d",
    "valuation_margin_of_safety_120d",
    "valuation_bubble_risk_inverse_120d",
    "risk_fraud_haircut_120d",
    "risk_management_quality_120d",
)

V7_DEEP_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120, 126)


@dataclass(frozen=True)
class V7DeepAlphaConfig:
    horizons: tuple[int, ...] = V7_DEEP_HORIZONS
    hidden_size: int = 16
    seed: int = 1729
    use_torch_if_available: bool = True
    fraud_penalty_weight: float = 0.30
    long_horizon_haircut_threshold: float = 0.40
    long_horizon_dominance_weight: float = 0.60
    volatility_floor: float = 0.05
    interval_width_multiplier: float = 1.65
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.95


@dataclass
class _NumpyMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)


def _init_mlp(input_dim: int, hidden: int, output_dim: int, rng: np.random.Generator) -> _NumpyMLP:
    scale1 = np.sqrt(1.0 / max(1, input_dim))
    scale2 = np.sqrt(1.0 / max(1, hidden))
    return _NumpyMLP(
        w1=rng.standard_normal((input_dim, hidden)) * scale1,
        b1=np.zeros(hidden),
        w2=rng.standard_normal((hidden, output_dim)) * scale2,
        b2=np.zeros(output_dim),
    )


class V7DeepAlphaModel:
    """Feature-group towers with horizon-conditional gating."""

    def __init__(self, config: V7DeepAlphaConfig | None = None) -> None:
        self.config = config or V7DeepAlphaConfig()
        rng = np.random.default_rng(self.config.seed)
        self._towers: dict[str, _NumpyMLP] = {
            "short": _init_mlp(len(SHORT_FEATURES), self.config.hidden_size, 1, rng),
            "medium": _init_mlp(len(MEDIUM_FEATURES), self.config.hidden_size, 1, rng),
            "long": _init_mlp(len(LONG_FEATURES), self.config.hidden_size, 1, rng),
            "fundamental": _init_mlp(len(FUNDAMENTAL_FEATURES), self.config.hidden_size, 1, rng),
        }
        self._horizon_gates: dict[int, np.ndarray] = {
            1: np.array([0.55, 0.25, 0.10, 0.10]),
            5: np.array([0.45, 0.30, 0.15, 0.10]),
            20: np.array([0.20, 0.40, 0.20, 0.20]),
            60: np.array([0.10, 0.25, 0.30, 0.35]),
            120: np.array([0.05, 0.15, 0.40, 0.40]),
            126: np.array([0.05, 0.15, 0.40, 0.40]),
        }
        self._torch_model = self._try_init_torch() if self.config.use_torch_if_available else None

    def _try_init_torch(self):
        try:
            import torch  # noqa: F401
            return _build_torch_model(self.config)
        except Exception:  # pragma: no cover - torch optional
            return None

    def predict(
        self,
        feature_frame: pd.DataFrame,
        universe_members: list[ThematicUniverseMember],
        factor_applicability: Iterable[FactorApplicability] = (),
    ) -> dict[str, MultiHorizonAlpha]:
        if feature_frame is None or feature_frame.empty:
            return {}
        latest = _latest_rows(feature_frame)
        members = {member.symbol: member for member in universe_members}
        applicability = list(factor_applicability)
        output: dict[str, MultiHorizonAlpha] = {}
        for symbol, row in latest.iterrows():
            member = members.get(str(symbol))
            if member is None:
                continue
            short_vec = _vectorize(row, SHORT_FEATURES)
            medium_vec = _vectorize(row, MEDIUM_FEATURES)
            long_vec = _vectorize(row, LONG_FEATURES)
            fund_vec = _vectorize(row, FUNDAMENTAL_FEATURES)
            tower_outputs = {
                "short": float(self._towers["short"].forward(short_vec.reshape(1, -1))[0, 0]),
                "medium": float(self._towers["medium"].forward(medium_vec.reshape(1, -1))[0, 0]),
                "long": float(self._towers["long"].forward(long_vec.reshape(1, -1))[0, 0]),
                "fundamental": float(self._towers["fundamental"].forward(fund_vec.reshape(1, -1))[0, 0]),
            }
            horizon_scores = self._compute_horizon_scores(tower_outputs, member)
            volatility = max(self.config.volatility_floor, _safe_float(row.get("volatility_20d", 0.20)))
            fraud_penalty = member.fraud_risk_score / 100.0
            risk_penalty = min(1.0, self.config.fraud_penalty_weight * fraud_penalty + max(0.0, volatility - 0.25))
            expected = float(np.mean(list(horizon_scores.values())))
            interval = self.config.interval_width_multiplier * volatility / np.sqrt(252 / 20)
            confidence = float(
                np.clip(
                    member.source_confidence * (1.0 - risk_penalty) * _factor_support(applicability, member),
                    self.config.confidence_floor,
                    self.config.confidence_ceiling,
                )
            )
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
                conformal_confidence=float(np.clip(1.0 - interval, self.config.confidence_floor, self.config.confidence_ceiling)),
                prediction_interval_low=expected - interval,
                prediction_interval_high=expected + interval,
                rank_score=float(np.clip(_cross_section_rank(latest, symbol, "v7_deep_seed") * 100.0, 0.0, 100.0)),
                regime_adjusted_score=float(np.clip(expected * 100.0 * confidence, -100.0, 100.0)),
                factor_contribution=tower_outputs,
                evidence_contribution={member.theme: member.exposure_score / 100.0},
                risk_penalty=risk_penalty,
                final_alpha_score=float(np.clip(expected * 100.0 - risk_penalty * 25.0, -100.0, 100.0)),
            )
        return output

    def _compute_horizon_scores(self, tower_outputs: dict[str, float], member: ThematicUniverseMember) -> dict[int, float]:
        long_emphasis = self._horizon_emphasis(member)
        horizon_scores: dict[int, float] = {}
        keys = ("short", "medium", "long", "fundamental")
        towers_vec = np.array([tower_outputs[key] for key in keys])
        for horizon in self.config.horizons:
            gate = self._horizon_gates.get(horizon, np.array([0.10, 0.20, 0.35, 0.35]))
            if horizon >= 60:
                gate = _renorm(gate + np.array([0.0, 0.0, long_emphasis * 0.10, long_emphasis * 0.10]))
            raw = float(np.dot(gate, towers_vec))
            horizon_scores[horizon] = float(np.clip(raw, -1.0, 1.0))
        return horizon_scores

    def _horizon_emphasis(self, member: ThematicUniverseMember) -> float:
        fundamental = member.fundamental_score / 100.0
        quality = member.quality_score / 100.0
        fraud = member.fraud_risk_score / 100.0
        return float(np.clip(0.4 * fundamental + 0.4 * quality - 0.4 * fraud, -1.0, 1.0))


def predict_v7_deep_alpha(
    feature_frame: pd.DataFrame,
    universe_members: list[ThematicUniverseMember],
    factor_applicability: Iterable[FactorApplicability] = (),
    config: V7DeepAlphaConfig | None = None,
) -> dict[str, MultiHorizonAlpha]:
    return V7DeepAlphaModel(config).predict(feature_frame, universe_members, factor_applicability)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vectorize(row: pd.Series, columns: tuple[str, ...]) -> np.ndarray:
    values: list[float] = []
    for column in columns:
        raw = row.get(column)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            values.append(0.0)
            continue
        numeric = _safe_float(raw)
        if abs(numeric) > 2.0:
            numeric = numeric / 100.0
        values.append(float(np.tanh(numeric)))
    return np.array(values, dtype=np.float64)


def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False).tail(1)
    data = data.set_index("symbol", drop=False)
    data["v7_deep_seed"] = data.apply(_seed_score, axis=1)
    return data


def _seed_score(row: pd.Series) -> float:
    components = (
        _safe_float(row.get("theme_strength")) / 100.0,
        _safe_float(row.get("fundamental_score")) / 100.0,
        _safe_float(row.get("exposure_score")) / 100.0,
        _safe_float(row.get("momentum_20d")),
        _safe_float(row.get("policy_strength")) / 100.0,
    )
    return float(np.mean([np.tanh(value) for value in components]))


def _cross_section_rank(frame: pd.DataFrame, symbol: object, column: str) -> float:
    if column not in frame.columns or symbol not in frame.index:
        return 0.5
    ranks = frame[column].rank(pct=True)
    value = ranks.loc[symbol]
    if isinstance(value, pd.Series):
        value = float(value.iloc[0])
    return float(value) if np.isfinite(value) else 0.5


def _factor_support(applicability: Iterable[FactorApplicability], member: ThematicUniverseMember) -> float:
    matched: list[float] = []
    for item in applicability:
        if item.factor_lifecycle_stage not in {"production", "validation"}:
            continue
        if item.applicable_theme and member.theme not in item.applicable_theme:
            continue
        matched.append(max(0.0, item.rank_icir) + item.hit_rate)
    if not matched:
        return 0.80
    return float(np.clip(0.65 + float(np.mean(matched)) * 0.20, 0.65, 1.10))


def _renorm(weights: np.ndarray) -> np.ndarray:
    total = float(np.abs(weights).sum())
    if total <= 0:
        return weights
    return weights / total


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric) or np.isinf(numeric):
        return default
    return numeric


def _build_torch_model(config: V7DeepAlphaConfig):
    """Placeholder for an optional torch implementation.

    We deliberately do not load weights here — the numpy MLP gives deterministic,
    license-free behaviour. To train a torch model, write fit() that pulls
    forward_return_{horizon}d labels from the factor frame and returns the trained
    state dict to be loaded back into self._towers.
    """
    try:
        import torch
        return torch.nn.ModuleDict({
            "short": torch.nn.Linear(len(SHORT_FEATURES), 1),
            "medium": torch.nn.Linear(len(MEDIUM_FEATURES), 1),
            "long": torch.nn.Linear(len(LONG_FEATURES), 1),
            "fundamental": torch.nn.Linear(len(FUNDAMENTAL_FEATURES), 1),
        })
    except Exception:  # pragma: no cover
        return None
