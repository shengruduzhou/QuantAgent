"""V5 online per-agent reliability tracker.

Each agent emits views; later we observe realized returns. The tracker keeps a
rolling, exponentially-weighted information coefficient (EWMA) per agent and
returns a reliability score in (0, 1) that the AgentRouter can use to scale
view confidence (q) and inverse uncertainty (1/omega).

Cold-start (no observations yet) returns 0.5 — neither inflate nor suppress
the view. This avoids the V4 issue where every agent is treated identically
via a static base_view_scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AgentReliability:
    """EWMA-tracked per-agent reliability backed by realized hit ratios."""

    halflife: int = 20
    initial_score: float = 0.5
    min_score: float = 0.1
    max_score: float = 1.5
    _scores: dict[str, float] = field(default_factory=dict)
    _samples: dict[str, int] = field(default_factory=dict)

    @property
    def decay(self) -> float:
        return 0.5 ** (1.0 / max(self.halflife, 1))

    def score(self, agent_name: str) -> float:
        return float(self._scores.get(agent_name, self.initial_score))

    def update(self, agent_name: str, predicted_direction: float, realized_return: float) -> float:
        """Update reliability with a single observation.

        predicted_direction: signed direction the agent took (+1 / -1 / scaled).
        realized_return: realized return over the prediction horizon.
        The instantaneous hit metric is sign(pred) == sign(realized); we EWMA
        the smoothed value of |realized| if the directions match (and negative
        otherwise) so that confident-correct moves boost reliability faster
        than weak-correct moves.
        """
        hit_signal = float(np.sign(predicted_direction)) * float(np.sign(realized_return))
        magnitude = float(np.clip(abs(realized_return), 0.0, 0.10))
        instantaneous = hit_signal * (0.5 + 5.0 * magnitude)
        instantaneous = float(np.clip(instantaneous + self.initial_score, 0.0, 2.0))
        prior = self._scores.get(agent_name, self.initial_score)
        decay = self.decay
        new_score = decay * prior + (1.0 - decay) * instantaneous
        new_score = float(np.clip(new_score, self.min_score, self.max_score))
        self._scores[agent_name] = new_score
        self._samples[agent_name] = self._samples.get(agent_name, 0) + 1
        return new_score

    def bulk_update(self, observations: list[tuple[str, float, float]]) -> None:
        for agent_name, direction, realized in observations:
            self.update(agent_name, direction, realized)

    def snapshot(self) -> dict[str, float]:
        return dict(self._scores)
