"""PIT portfolio RL environment under the canonical executable A-share clock.

This environment is deliberately *research only*.  A policy action is formed from
information available at ``close(T)``.  The action can only become a position at
the next global market session ``T+1`` and its one-session reward is therefore the
mark-to-market return from ``close(T+1)`` to ``close(T+2)``.  This is the same
signal/execution clock used by governed executable labels and the strict A-share
simulator; using ``close(T)->close(T+1)`` would reward the policy for a return it
could not have owned.

The original ``PortfolioEnv`` is not used here.  It was rejected because its
fixed top-80 universe was selected using the whole evaluation window.  This v3
environment instead receives the deterministic, point-in-time hold-band book.
A zero action follows that passive book exactly and therefore earns exactly zero
incremental reward.

Execution-state rules are applied on mapped ``T+1`` rather than on the signal
session.  Required flags are explicit and unknown values fail closed:

* suspended -> no increase and no decrease;
* limit-up or ST -> no increase;
* limit-down -> no decrease.

The environment never emits order intents.  Any candidate policy must still be
exported as signal-dated target weights and re-simulated through the strict
position-carrying A-share simulator before it can even pass a research screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.executable_labels import (
    build_executable_forward_returns,
    canonical_market_sessions,
    market_session_schedule_sha256,
)

try:  # optional dependency, validated in __init__
    import gymnasium as gym
except Exception:  # pragma: no cover - optional dependency
    gym = None


RL_REWARD_SEMANTICS = "signal_t_close_entry_t1_close_reward_t1_to_t2_close_v1"
_REQUIRED_EXECUTION_FLAGS = ("is_limit_up", "is_limit_down", "is_suspended", "is_st")


@dataclass(frozen=True)
class PITPortfolioEnvConfig:
    max_book: int = 60
    max_tilt: float = 0.8
    max_cash_tilt: float = 0.3
    min_gross: float = 0.5
    max_gross: float = 1.0
    cost_bps: float = 12.0
    reward_scale: float = 100.0


class PITPortfolioEnv(gym.Env if gym is not None else object):
    """Signal-dated PIT hold-band universe with executable-clock value-add reward."""

    metadata = {"render_modes": []}
    N_SLOT_FEATURES = 5  # alpha_z, ret_5d, age_norm, prev_minus_passive, in_book

    def __init__(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
        market_sessions: Iterable[object],
        config: PITPortfolioEnvConfig | None = None,
    ) -> None:
        if gym is None:  # pragma: no cover - optional dependency
            raise ImportError("PITPortfolioEnv requires gymnasium")
        from gymnasium import spaces

        self.config = config or PITPortfolioEnvConfig()
        self.market_sessions = canonical_market_sessions(market_sessions)
        self.market_session_schedule_sha256 = market_session_schedule_sha256(
            self.market_sessions
        )
        self._build_caches(book_weights, predictions, market_panel)

        n = self.config.max_book
        obs_size = n * self.N_SLOT_FEATURES + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n + 1,), dtype=np.float32
        )
        self._t = 0
        self._prev_w: dict[str, float] = {}
        self._prev_w_passive: dict[str, float] = {}
        self._nav = 1.0
        self._nav_passive = 1.0

    # ------------------------------------------------------------------ setup
    def _build_caches(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
    ) -> None:
        bw = book_weights.copy()
        bw.index = pd.to_datetime(bw.index, errors="coerce").normalize()
        if bw.index.isna().any() or bw.index.duplicated().any():
            raise ValueError("PITPortfolioEnv requires unique valid signal dates in book_weights")
        bw = bw.sort_index()
        if bw.empty:
            raise ValueError("PITPortfolioEnv requires a non-empty signal-dated book")

        panel = market_panel.copy()
        missing = sorted(
            {"symbol", "trade_date", "close", *_REQUIRED_EXECUTION_FLAGS}
            - set(panel.columns)
        )
        if missing:
            raise ValueError(
                "PITPortfolioEnv requires explicit execution-state columns; "
                f"missing={missing}"
            )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
        panel["symbol"] = panel["symbol"].astype(str)
        panel = panel.dropna(subset=["trade_date", "symbol"]).sort_values(
            ["symbol", "trade_date"]
        )
        if panel.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("PITPortfolioEnv requires unique symbol/trade_date market rows")

        built = build_executable_forward_returns(
            panel,
            horizons=(1,),
            price_column="close",
            entry_delay_sessions=1,
            market_sessions=self.market_sessions,
        )
        labelled = built.frame
        executable = labelled.pivot(
            index="trade_date",
            columns="symbol",
            values="forward_executable_return_1d",
        ).sort_index()

        px = panel.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        ret5 = px / px.shift(5) - 1.0

        preds = predictions.copy()
        required_pred = {"symbol", "trade_date"}
        missing_pred = sorted(required_pred - set(preds.columns))
        if missing_pred:
            raise ValueError(f"PITPortfolioEnv predictions missing columns: {missing_pred}")
        preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce").dt.normalize()
        preds["symbol"] = preds["symbol"].astype(str)
        score_col = "alpha_score" if "alpha_score" in preds.columns else "prediction"
        if score_col not in preds.columns:
            raise ValueError("PITPortfolioEnv predictions require alpha_score or prediction")
        if preds.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("PITPortfolioEnv predictions require unique symbol/trade_date rows")
        alpha = preds.pivot(index="trade_date", columns="symbol", values=score_col)

        # PIT regime: only history through close(T) is observable when the action is formed.
        bench = px.pct_change(fill_method=None).mean(axis=1)
        cum = (1.0 + bench.fillna(0.0)).cumprod()
        trail = cum / cum.shift(60) - 1.0

        # Entry-state flags are execution constraints, not observation features.
        state = panel.set_index(["symbol", "trade_date"])[list(_REQUIRED_EXECUTION_FLAGS)]

        positions = self.market_sessions.get_indexer(bw.index)
        if bool((positions < 0).any()):
            bad = bw.index[positions < 0].strftime("%Y-%m-%d").tolist()[:5]
            raise ValueError(f"book signal dates absent from market_sessions: {bad}")
        # Need exact T+1 entry and T+2 reward end.  A terminal signal without T+2
        # is right-censored and excluded by the market calendar, not by outcomes.
        dates = [
            pd.Timestamp(date)
            for date, pos in zip(bw.index, positions)
            if int(pos) + 2 < len(self.market_sessions)
        ]
        if len(dates) < 3:
            raise ValueError("PITPortfolioEnv requires at least 3 signal dates with T+1/T+2 sessions")

        n = self.config.max_book
        T = len(dates)
        self.dates = dates
        self.execution_dates: list[pd.Timestamp] = []
        self.reward_end_dates: list[pd.Timestamp] = []
        self.slot_symbols: list[list[str]] = []
        self.slot_ret = np.zeros((T, n), dtype=np.float64)
        self.slot_alpha = np.zeros((T, n), dtype=np.float32)
        self.slot_ret5 = np.zeros((T, n), dtype=np.float32)
        self.slot_age = np.zeros((T, n), dtype=np.float32)
        self.slot_no_increase = np.zeros((T, n), dtype=bool)
        self.slot_no_decrease = np.zeros((T, n), dtype=bool)
        self.slot_frozen = np.zeros((T, n), dtype=bool)
        self.slot_in_book = np.zeros((T, n), dtype=np.float32)
        self.passive_w = np.zeros((T, n), dtype=np.float64)
        self.regime_vec = np.zeros((T, 2), dtype=np.float32)

        age_track: dict[str, int] = {}
        pos_of = {session: idx for idx, session in enumerate(self.market_sessions)}
        for ti, d in enumerate(dates):
            pos = pos_of[d]
            execution_date = pd.Timestamp(self.market_sessions[pos + 1])
            reward_end_date = pd.Timestamp(self.market_sessions[pos + 2])
            self.execution_dates.append(execution_date)
            self.reward_end_dates.append(reward_end_date)

            row = bw.loc[d]
            held = pd.to_numeric(row, errors="coerce").fillna(0.0)
            held = held[held > 0]
            syms = list(held.index.astype(str))
            if not syms:
                raise ValueError(f"PITPortfolioEnv empty passive book on signal date {d.date()}")

            for s in syms:
                age_track[s] = age_track.get(s, 0) + 1
            for s in list(age_track):
                if s not in syms:
                    age_track.pop(s)

            a_row = alpha.loc[d] if d in alpha.index else pd.Series(dtype=float)
            raw_alpha = pd.to_numeric(a_row.reindex(syms), errors="coerce")
            if raw_alpha.isna().any():
                bad = raw_alpha[raw_alpha.isna()].index.astype(str).tolist()[:5]
                raise ValueError(
                    f"missing signal-date alpha for held names on {d.date()}: {bad}"
                )
            std = float(raw_alpha.std(ddof=0))
            a_z = (
                (raw_alpha - raw_alpha.mean()) / std
                if std > 1e-9
                else raw_alpha * 0.0
            )
            order = a_z.sort_values(ascending=False).index.tolist()[:n]
            self.slot_symbols.append(order)
            k = len(order)

            fwd_row = executable.loc[d] if d in executable.index else pd.Series(dtype=float)
            future_returns = pd.to_numeric(fwd_row.reindex(order), errors="coerce")
            if future_returns.isna().any():
                bad = future_returns[future_returns.isna()].index.astype(str).tolist()[:5]
                raise ValueError(
                    "missing exact T+1->T+2 executable reward for held names on "
                    f"{d.date()}: {bad}; do not shift to a later per-symbol row"
                )

            r5_row = ret5.loc[d] if d in ret5.index else pd.Series(dtype=float)
            self.slot_ret[ti, :k] = future_returns.to_numpy(dtype=np.float64)
            self.slot_alpha[ti, :k] = a_z.reindex(order).to_numpy(dtype=np.float32)
            self.slot_ret5[ti, :k] = np.nan_to_num(
                pd.to_numeric(r5_row.reindex(order), errors="coerce").to_numpy(dtype=np.float32),
                nan=0.0,
            )
            self.slot_age[ti, :k] = np.array(
                [min(age_track.get(s, 1), 60) / 60.0 for s in order], dtype=np.float32
            )
            self.slot_in_book[ti, :k] = 1.0
            self.passive_w[ti, :k] = held.reindex(order).to_numpy(dtype=np.float64)

            keys = pd.MultiIndex.from_arrays(
                [np.asarray(order, dtype=object), np.repeat(execution_date, k)],
                names=["symbol", "trade_date"],
            )
            entry_state = state.reindex(keys)
            lu = _nullable_bool(entry_state["is_limit_up"])
            ld = _nullable_bool(entry_state["is_limit_down"])
            su = _nullable_bool(entry_state["is_suspended"])
            st = _nullable_bool(entry_state["is_st"])
            self.slot_no_increase[ti, :k] = (
                lu.fillna(True) | st.fillna(True) | su.fillna(True)
            ).to_numpy(dtype=bool)
            self.slot_no_decrease[ti, :k] = (
                ld.fillna(True) | su.fillna(True)
            ).to_numpy(dtype=bool)
            self.slot_frozen[ti, :k] = su.fillna(True).to_numpy(dtype=bool)

            tr = float(trail.get(d, np.nan))
            self.regime_vec[ti] = (
                float(np.isfinite(tr) and tr > 0.05),
                float(np.isfinite(tr) and tr < -0.05),
            )

    # ------------------------------------------------------------------ gym api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super_reset = getattr(super(), "reset", None)
        if callable(super_reset):
            try:
                super_reset(seed=seed)
            except TypeError:
                pass
        self._t = 0
        self._prev_w = {}
        self._prev_w_passive = {}
        self._nav = 1.0
        self._nav_passive = 1.0
        return self._obs(), {}

    def step(self, action):
        cfg = self.config
        t = self._t
        n = cfg.max_book
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (n + 1,):
            raise ValueError(f"action shape must be {(n + 1,)}, got {a.shape}")
        in_book = self.slot_in_book[t].astype(bool)
        passive = self.passive_w[t]
        syms = self.slot_symbols[t]

        w = passive * (1.0 + cfg.max_tilt * a[:n])
        w = np.where(in_book, np.maximum(w, 0.0), 0.0)
        passive_gross = float(passive.sum())
        gross = float(
            np.clip(
                passive_gross * (1.0 + cfg.max_cash_tilt * a[n]),
                cfg.min_gross,
                min(cfg.max_gross, 1.0),
            )
        )
        total = float(w.sum())
        if total > 1e-12:
            w = w * (gross / total)

        prev_vec = np.array(
            [self._prev_w.get(s, 0.0) for s in syms] + [0.0] * (n - len(syms))
        )
        prev_b_vec = np.array(
            [self._prev_w_passive.get(s, 0.0) for s in syms]
            + [0.0] * (n - len(syms))
        )
        w = _apply_execution_constraints(
            w,
            prev_vec,
            self.slot_no_increase[t],
            self.slot_no_decrease[t],
            self.slot_frozen[t],
        )
        w_b = _apply_execution_constraints(
            passive,
            prev_b_vec,
            self.slot_no_increase[t],
            self.slot_no_decrease[t],
            self.slot_frozen[t],
        )

        r = self.slot_ret[t]
        ret_p = float(np.dot(w, r))
        ret_b = float(np.dot(w_b, r))
        cur = {s: float(w[i]) for i, s in enumerate(syms)}
        cur_b = {s: float(w_b[i]) for i, s in enumerate(syms)}
        to_p = _sym_turnover(self._prev_w, cur)
        to_b = _sym_turnover(self._prev_w_passive, cur_b)
        cost_p = to_p * cfg.cost_bps / 1e4
        cost_b = to_b * cfg.cost_bps / 1e4

        net_p = ret_p - cost_p
        net_b = ret_b - cost_b
        reward = (net_p - net_b) * cfg.reward_scale
        self._nav *= 1.0 + net_p
        self._nav_passive *= 1.0 + net_b
        self._prev_w = cur
        self._prev_w_passive = cur_b
        self._t += 1
        terminated = self._t >= len(self.dates)
        info = {
            "trade_date": str(self.dates[t].date()),
            "signal_date": str(self.dates[t].date()),
            "execution_date": str(self.execution_dates[t].date()),
            "reward_end_date": str(self.reward_end_dates[t].date()),
            "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            "reward_semantics": RL_REWARD_SEMANTICS,
            "market_session_schedule_sha256": self.market_session_schedule_sha256,
            "weights": cur,
            "net_policy": net_p,
            "net_passive": net_b,
            "value_add": net_p - net_b,
            "turnover_policy": to_p,
            "turnover_passive": to_b,
            "nav": self._nav,
            "nav_passive": self._nav_passive,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    # ---------------------------------------------------------------- guards
    def book_dispersion_report(self, eps: float = 1e-6) -> dict:
        """Report whether within-book alpha dispersion permits name selection."""
        stds: list[float] = []
        for t in range(len(self.dates)):
            in_book = self.slot_in_book[t].astype(bool)
            values = self.slot_alpha[t][in_book]
            if values.size > 1:
                stds.append(float(np.std(values)))
        arr = np.asarray(stds, dtype=float)
        n = int(arr.size)
        flat = int(np.sum(arr < eps)) if n else 0
        flat_frac = float(flat / n) if n else 1.0
        return {
            "n_dates": n,
            "mean_within_book_alpha_std": float(arr.mean()) if n else 0.0,
            "median_within_book_alpha_std": float(np.median(arr)) if n else 0.0,
            "flat_date_fraction": flat_frac,
            "env_can_select": bool(n > 0 and flat_frac < 0.5),
            "reward_semantics": RL_REWARD_SEMANTICS,
            "market_session_schedule_sha256": self.market_session_schedule_sha256,
        }

    def _obs(self) -> np.ndarray:
        cfg = self.config
        n = cfg.max_book
        t = min(self._t, len(self.dates) - 1)
        syms = self.slot_symbols[t]
        prev_vec = np.array(
            [self._prev_w.get(s, 0.0) for s in syms] + [0.0] * (n - len(syms)),
            dtype=np.float32,
        )
        feats = np.concatenate(
            [
                self.slot_alpha[t],
                self.slot_ret5[t],
                self.slot_age[t],
                prev_vec - self.passive_w[t].astype(np.float32),
                self.slot_in_book[t],
            ]
        )
        n_book = float(self.slot_in_book[t].sum())
        globals_ = np.array(
            [
                self.regime_vec[t][0],
                self.regime_vec[t][1],
                n_book / max(1, n),
                1.0 - float(sum(self._prev_w.values())),
                t / max(1, len(self.dates) - 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feats, globals_]).astype(np.float32)


def _nullable_bool(series: pd.Series) -> pd.Series:
    """Parse bool-like execution state while retaining unknown as nullable NA."""
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.isin([0, 1])
    out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(int).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    out.loc[text.isin({"true", "t", "yes", "y", "1"})] = True
    out.loc[text.isin({"false", "f", "no", "n", "0"})] = False
    return out


def _apply_execution_constraints(
    desired: np.ndarray,
    previous: np.ndarray,
    no_increase: np.ndarray,
    no_decrease: np.ndarray,
    frozen: np.ndarray,
) -> np.ndarray:
    constrained = np.asarray(desired, dtype=np.float64).copy()
    constrained = np.where(no_increase, np.minimum(constrained, previous), constrained)
    constrained = np.where(no_decrease, np.maximum(constrained, previous), constrained)
    constrained = np.where(frozen, previous, constrained)
    return constrained


def _sym_turnover(prev: dict[str, float], cur: dict[str, float]) -> float:
    keys = set(prev) | set(cur)
    return float(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


__all__ = [
    "RL_REWARD_SEMANTICS",
    "PITPortfolioEnv",
    "PITPortfolioEnvConfig",
]
