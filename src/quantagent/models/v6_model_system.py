from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1

import numpy as np
import pandas as pd

from quantagent.models.v6_outputs import V6ModelOutput
from quantagent.training.conformal_calibrator import ConformalCalibrator


FACTOR_GROUPS = ("momentum", "reversal", "flow", "event", "quality", "liquidity")
MOE_GROUPS = ("sequence", "snapshot", "event")


@dataclass
class V6ModelSystem:
    model_version: str = "v6.smoke"
    calibration_version: str = "calib.synthetic.v1"
    feature_version: str = "v6.0"
    factor_groups: tuple[str, ...] = FACTOR_GROUPS
    conformal: ConformalCalibrator = field(default_factory=lambda: ConformalCalibrator(alpha=0.1).fit(np.array([0.0, 0.01, -0.01]), np.array([0.002, 0.0, -0.004])))

    def infer(self, features: pd.DataFrame, trade_date: str | None = None) -> list[V6ModelOutput]:
        if features.empty:
            return []
        data = features.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        date_value = pd.Timestamp(trade_date) if trade_date else data["trade_date"].max()
        latest = data[data["trade_date"] == date_value].copy()
        if latest.empty:
            latest = data[data["trade_date"] <= date_value].groupby("symbol", sort=False).tail(1).copy()
        raw = self._raw_predictions(latest)
        calibrated = self.conformal.attach_interval(raw.rename(columns={"alpha_5d": "alpha"}), pred_column="alpha", vol_column=None)
        outputs: list[V6ModelOutput] = []
        for idx, row in raw.iterrows():
            alpha_5d = float(row["alpha_5d"])
            factor_gate = self._factor_gate(row)
            outputs.append(
                V6ModelOutput(
                    trade_date=str(pd.Timestamp(row["trade_date"]).date()),
                    symbol=str(row["symbol"]),
                    alpha_1d=float(alpha_5d / 3.0),
                    alpha_5d=alpha_5d,
                    alpha_20d=float(alpha_5d * 1.8),
                    direction_logit_1d=float(np.tanh(alpha_5d / 0.02)),
                    direction_logit_5d=float(np.tanh(alpha_5d / 0.01)),
                    direction_logit_20d=float(np.tanh(alpha_5d / 0.03)),
                    q_low=float(calibrated.loc[idx, "alpha_lower"]),
                    q_high=float(calibrated.loc[idx, "alpha_upper"]),
                    confidence=float(row["confidence"]),
                    conformal_confidence=float(calibrated.loc[idx, "conformal_confidence"]),
                    risk_score=float(row["risk_score"]),
                    factor_gate=factor_gate,
                    moe_gate=self._moe_gate(row),
                    regime=str(row.get("regime", "range_bound")),
                    model_version=self.model_version,
                    feature_version=str(row.get("feature_version", self.feature_version)),
                    calibration_version=self.calibration_version,
                )
            )
        return outputs

    def infer_frame(self, features: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
        rows = [output.to_dict() for output in self.infer(features, trade_date)]
        return pd.DataFrame(rows)

    def feature_fingerprint(self, frame: pd.DataFrame) -> str:
        cols = sorted(str(c) for c in frame.columns)
        sample = frame[cols].head(64).to_json(date_format="iso", default_handler=str)
        return sha1((self.feature_version + sample).encode("utf-8")).hexdigest()[:16]

    def _raw_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        components: list[pd.Series] = []
        for column in ("ret_5d", "ma_gap_20d", "northbound_flow", "main_money_flow", "event_sentiment", "roe", "cashflow_quality"):
            if column in data.columns:
                values = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
                components.append(_zscore(values))
        if not components:
            score = pd.Series(0.0, index=data.index)
        else:
            score = sum(components) / len(components)
        data["alpha_5d"] = (0.01 * np.tanh(score)).astype(float)
        data["confidence"] = (0.45 + 0.35 * (1.0 - data["alpha_5d"].abs() / 0.02).clip(lower=0.0)).clip(0.05, 0.95)
        volatility = pd.to_numeric(data.get("volatility_20d", pd.Series(0.02, index=data.index)), errors="coerce").fillna(0.02)
        data["risk_score"] = np.clip(volatility / max(float(volatility.quantile(0.9)), 1e-6), 0.0, 1.0)
        data["regime"] = np.where(data["risk_score"] > 0.7, "high_volatility", "range_bound")
        return data

    def _factor_gate(self, row: pd.Series) -> dict[str, float]:
        raw = pd.Series(
            {
                "momentum": abs(float(row.get("ret_5d", 0.0))),
                "reversal": abs(float(row.get("ma_gap_20d", 0.0))),
                "flow": abs(float(row.get("northbound_flow", 0.0))) / 1_000_000.0,
                "event": abs(float(row.get("event_sentiment", 0.0))),
                "quality": max(float(row.get("roe", 0.0)), 0.0),
                "liquidity": float(row.get("amount", 0.0)) / 1_000_000_000.0,
            }
        ).clip(lower=0.0)
        if raw.sum() <= 0:
            raw[:] = 1.0
        weights = raw / raw.sum()
        return {key: float(value) for key, value in weights.items()}

    def _moe_gate(self, row: pd.Series) -> dict[str, float]:
        event = abs(float(row.get("event_sentiment", 0.0)))
        snapshot = abs(float(row.get("roe", 0.0))) + abs(float(row.get("northbound_flow", 0.0))) / 1_000_000.0
        seq = abs(float(row.get("ret_5d", 0.0))) + abs(float(row.get("ma_gap_20d", 0.0)))
        raw = pd.Series({"sequence": seq, "snapshot": snapshot, "event": event}).clip(lower=0.0)
        if raw.sum() <= 0:
            raw[:] = 1.0
        weights = raw / raw.sum()
        return {key: float(value) for key, value in weights.items()}


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std
