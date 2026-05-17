"""Gymnasium portfolio environment for V7 target-weight refinement.

The environment consumes PIT alpha predictions and a market panel, then
lets an RL policy propose daily weight deltas. Deltas are projected back
into A-share-safe portfolio constraints before reward calculation. It is
for paper/backtest research only and never creates order intents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # optional dependency, validated in __init__
    import gymnasium as gym
except Exception:  # pragma: no cover - optional dependency
    gym = None


@dataclass(frozen=True)
class PortfolioEnvConfig:
    top_n: int = 80
    max_delta: float = 0.05
    max_weight_per_name: float = 0.10
    max_gross: float = 1.0
    max_turnover: float = 0.40
    cost_bps: float = 12.0
    drawdown_lambda: float = 2.0
    drawdown_limit: float = 0.20
    kill_switch_drawdown: float = 0.30
    initial_nav: float = 1.0


class PortfolioEnv(gym.Env if gym is not None else object):
    """Fixed-universe RL environment with alpha/regime/weight/age obs."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
        config: PortfolioEnvConfig | None = None,
    ) -> None:
        if gym is None:  # pragma: no cover - optional dependency
            raise ImportError("PortfolioEnv requires gymnasium")
        try:
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PortfolioEnv requires gymnasium") from exc
        self.config = config or PortfolioEnvConfig()
        self.predictions = _prepare_predictions(predictions)
        self.market = _prepare_market(market_panel)
        self.dates = sorted(set(self.predictions["trade_date"]).intersection(set(self.market["trade_date"])))
        if len(self.dates) < 3:
            raise ValueError("PortfolioEnv requires at least three overlapping trade dates")
        self.symbols = _select_symbols(self.predictions, self.config.top_n)
        self.n = len(self.symbols)
        if self.n <= 0:
            raise ValueError("PortfolioEnv resolved an empty symbol universe")
        obs_size = self.n * 4 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-float(self.config.max_delta),
            high=float(self.config.max_delta),
            shape=(self.n,),
            dtype=np.float32,
        )
        self._index = 0
        self._weights = np.zeros(self.n, dtype=np.float32)
        self._age = np.zeros(self.n, dtype=np.float32)
        self._nav = float(self.config.initial_nav)
        self._peak_nav = float(self.config.initial_nav)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super_reset = getattr(super(), "reset", None)
        if callable(super_reset):
            try:
                super_reset(seed=seed)
            except TypeError:
                pass
        self._index = 0
        self._weights = np.zeros(self.n, dtype=np.float32)
        self._age = np.zeros(self.n, dtype=np.float32)
        self._nav = float(self.config.initial_nav)
        self._peak_nav = float(self.config.initial_nav)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        prev_weights = self._weights.copy()
        target = _project_weights(
            prev_weights + np.clip(action, -self.config.max_delta, self.config.max_delta),
            previous=prev_weights,
            max_weight=float(self.config.max_weight_per_name),
            max_gross=float(self.config.max_gross),
            max_turnover=float(self.config.max_turnover),
        )
        date = self.dates[self._index]
        next_date = self.dates[self._index + 1]
        returns = self._forward_returns(date, next_date)
        turnover = float(np.abs(target - prev_weights).sum())
        cost = turnover * float(self.config.cost_bps) / 10_000.0
        pnl = float(np.dot(target, returns))
        self._nav *= 1.0 + pnl - cost
        self._peak_nav = max(self._peak_nav, self._nav)
        drawdown = 1.0 - self._nav / max(self._peak_nav, 1e-12)
        dd_penalty = float(self.config.drawdown_lambda) * max(0.0, drawdown - float(self.config.drawdown_limit))
        kill_penalty = 1.0 if drawdown > float(self.config.kill_switch_drawdown) else 0.0
        reward = pnl - cost - dd_penalty - kill_penalty
        self._weights = target.astype(np.float32)
        self._age = np.where(np.abs(self._weights) > 1e-6, self._age + 1.0, 0.0).astype(np.float32)
        self._index += 1
        terminated = self._index >= len(self.dates) - 2 or kill_penalty > 0
        info = {
            "trade_date": str(next_date.date()),
            "pnl": pnl,
            "cost": cost,
            "turnover": turnover,
            "drawdown": drawdown,
            "nav": self._nav,
            "kill_switch_penalty": kill_penalty,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def _obs(self) -> np.ndarray:
        date = self.dates[self._index]
        alpha = self._alpha_for(date)
        regime = np.array([float(np.nanmean(alpha)), float(np.nanstd(alpha))], dtype=np.float32)
        return np.concatenate([alpha, regime, self._weights, self._age, np.sign(alpha)]).astype(np.float32)

    def _alpha_for(self, date: pd.Timestamp) -> np.ndarray:
        day = self.predictions[self.predictions["trade_date"] == date].set_index("symbol")
        values = day["prediction"].reindex(self.symbols).fillna(0.0).to_numpy(dtype=np.float32)
        std = float(np.nanstd(values))
        if std > 1e-9:
            values = (values - float(np.nanmean(values))) / std
        return np.nan_to_num(values).astype(np.float32)

    def _forward_returns(self, date: pd.Timestamp, next_date: pd.Timestamp) -> np.ndarray:
        today = self.market[self.market["trade_date"] == date].set_index("symbol")["close"].reindex(self.symbols)
        nxt = self.market[self.market["trade_date"] == next_date].set_index("symbol")["close"].reindex(self.symbols)
        returns = (nxt.astype(float) / today.astype(float) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return returns.to_numpy(dtype=np.float32)


def _prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("PortfolioEnv requires non-empty predictions")
    data = frame.copy()
    if "prediction" not in data.columns:
        alpha_columns = [c for c in data.columns if c.startswith("alpha_")]
        if not alpha_columns:
            raise ValueError("predictions must include prediction or alpha_* columns")
        data["prediction"] = data[alpha_columns[0]]
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    return data.dropna(subset=["trade_date", "symbol", "prediction"]).reset_index(drop=True)


def _prepare_market(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("PortfolioEnv requires non-empty market_panel")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    return data.dropna(subset=["trade_date", "symbol", "close"]).reset_index(drop=True)


def _select_symbols(predictions: pd.DataFrame, top_n: int) -> list[str]:
    score = predictions.groupby("symbol")["prediction"].apply(lambda s: float(s.abs().mean()))
    return score.sort_values(ascending=False).head(int(top_n)).index.astype(str).tolist()


def _project_weights(
    raw: np.ndarray,
    *,
    previous: np.ndarray,
    max_weight: float,
    max_gross: float,
    max_turnover: float,
) -> np.ndarray:
    target = np.clip(np.nan_to_num(raw.astype(float)), -max_weight, max_weight)
    gross = float(np.abs(target).sum())
    if gross > max_gross:
        target = target * (max_gross / max(gross, 1e-12))
    turnover = float(np.abs(target - previous).sum())
    if turnover > max_turnover:
        scale = max_turnover / max(turnover, 1e-12)
        target = previous + (target - previous) * scale
    return target.astype(np.float32)


__all__ = ["PortfolioEnv", "PortfolioEnvConfig"]
