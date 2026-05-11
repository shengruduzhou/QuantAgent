from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.factors.composite import combine_with_model_gate, composite_factor_score


@dataclass(frozen=True)
class FactorPipelineResult:
    frame: pd.DataFrame
    composite_weights: pd.Series
    audit: dict[str, object]


class FactorPipeline:
    def run(
        self,
        frame: pd.DataFrame,
        factor_columns: list[str],
        statistical_weights: pd.Series,
        model_gate: pd.Series,
        lifecycle_scores: pd.Series | None = None,
        crowding_penalty: pd.Series | None = None,
        gate_strength: float = 1.0,
    ) -> FactorPipelineResult:
        weights = combine_with_model_gate(
            statistical_weights=statistical_weights,
            model_gate=model_gate,
            lifecycle_scores=lifecycle_scores,
            crowding_penalty=crowding_penalty,
            gate_strength=gate_strength,
        )
        result = composite_factor_score(frame, factor_columns, weights=weights, output_column="composite_factor_score")
        audit = {
            "factor_columns": tuple(factor_columns),
            "statistical_weights": statistical_weights.to_dict(),
            "model_gate": model_gate.to_dict(),
            "composite_weights": weights.to_dict(),
            "point_in_time_note": "model_gate must be lagged before this pipeline consumes it",
        }
        return FactorPipelineResult(result, weights, audit)

