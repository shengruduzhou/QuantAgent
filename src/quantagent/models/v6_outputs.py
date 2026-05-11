from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V6ModelOutput:
    trade_date: str
    symbol: str
    alpha_1d: float
    alpha_5d: float
    alpha_20d: float
    direction_logit_1d: float
    direction_logit_5d: float
    direction_logit_20d: float
    q_low: float
    q_high: float
    confidence: float
    conformal_confidence: float
    risk_score: float
    factor_gate: dict[str, float]
    moe_gate: dict[str, float]
    regime: str
    model_version: str
    feature_version: str
    calibration_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

