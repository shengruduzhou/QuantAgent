"""PIT portfolio environment v2 — fixes the 2026-06-12 RL rejection root causes.

The first PortfolioEnv was rejected because 100% of its claimed edge was
universe lookahead: it picked a fixed top-80 by mean |prediction| over the
WHOLE eval window, rewarded raw PnL minus a global benchmark, and filled
frictionlessly. A do-nothing policy inherited the universe drift.

This environment is built so that **a do-nothing policy earns exactly zero
reward**:

* Universe at step t = the deterministic hold-band book as of t (point in
  time by construction — the book is what the live loop would hold).
* Reward = (policy book return − policy turnover cost)
         − (passive equal-weight book return − passive turnover cost).
  The passive book IS the deployed baseline, so reward == policy value-add.
* Tradability flags constrain the action: a name limit-up-sealed at signal
  time cannot be increased; suspended / limit-down names are frozen.

The env is for training only; the enable-gate is the exported daily weights
re-simulated through ``run_strict_backtest_v8`` against the same passive
book (scripts/rl_pit_train_eval.py). Research only — no order intents.
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
class PITPortfolioEnvConfig:
    max_book: int = 60
    max_tilt: float = 0.8              # per-name weight multiplier range (1 ± tilt·a)
    max_cash_tilt: float = 0.3         # gross exposure action range
    min_gross: float = 0.5
    max_gross: float = 1.0
    cost_bps: float = 12.0
    reward_scale: float = 100.0


class PITPortfolioEnv(gym.Env if gym is not None else object):
    """Hold-band-book universe, value-add reward, flags-constrained actions."""

    metadata = {"render_modes": []}

    N_SLOT_FEATURES = 5  # alpha_z, ret_5d, age_norm, prev_minus_eqw, in_book

    def __init__(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
        config: PITPortfolioEnvConfig | None = None,
    ) -> None:
        if gym is None:  # pragma: no cover - optional dependency
            raise ImportError("PITPortfolioEnv requires gymnasium")
        from gymnasium import spaces

        self.config = config or PITPortfolioEnvConfig()
        self._build_caches(book_weights, predictions, market_panel)

        n = self.config.max_book
        obs_size = n * self.N_SLOT_FEATURES + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n + 1,), dtype=np.float32)
        self._t = 0
        self._prev_w: dict[str, float] = {}
        self._prev_w_passive: dict[str, float] = {}
        self._nav = 1.0
        self._nav_passive = 1.0

    # ------------------------------------------------------------------ setup
    def _build_caches(self, book_weights: pd.DataFrame, predictions: pd.DataFrame,
                      market_panel: pd.DataFrame) -> None:
        bw = book_weights.copy()
        bw.index = pd.to_datetime(bw.index)
        bw = bw.sort_index()

        panel = market_panel.copy()
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        px = panel.pivot_table(index="trade_date", columns="symbol", values="close",
                               aggfunc="last").sort_index()
        fwd = px.shift(-1) / px - 1.0          # close(t) -> close(t+1)
        ret5 = px / px.shift(5) - 1.0          # trailing, known at t

        flag = {}
        for col in ("is_limit_up", "is_limit_down", "is_suspended"):
            if col in panel.columns:
                flag[col] = (
                    panel.pivot_table(index="trade_date", columns="symbol",
                                      values=col, aggfunc="last")
                    .astype("float32").fillna(0.0).astype(bool)
                )
            else:
                flag[col] = pd.DataFrame(False, index=px.index, columns=px.columns)

        preds = predictions.copy()
        preds["trade_date"] = pd.to_datetime(preds["trade_date"])
        score_col = "alpha_score" if "alpha_score" in preds.columns else "prediction"
        alpha = preds.pivot_table(index="trade_date", columns="symbol",
                                  values=score_col, aggfunc="last")

        # PIT regime from the panel's own equal-weight benchmark
        bench = px.pct_change(fill_method=None).mean(axis=1)
        cum = (1 + bench.fillna(0)).cumprod().shift(1)
        trail = (cum / cum.shift(60) - 1.0)

        dates = [d for d in bw.index if d in fwd.index and np.isfinite(
            fwd.loc[d].dropna().mean() if d in fwd.index else np.nan)]
        dates = [d for d in dates if d in fwd.index][:-1] if dates else []
        if len(dates) < 3:
            raise ValueError("PITPortfolioEnv requires at least 3 book dates with forward returns")

        n = self.config.max_book
        T = len(dates)
        self.dates = dates
        self.slot_symbols: list[list[str]] = []
        self.slot_ret = np.zeros((T, n), dtype=np.float64)
        self.slot_alpha = np.zeros((T, n), dtype=np.float32)
        self.slot_ret5 = np.zeros((T, n), dtype=np.float32)
        self.slot_age = np.zeros((T, n), dtype=np.float32)
        self.slot_no_increase = np.zeros((T, n), dtype=bool)
        self.slot_frozen = np.zeros((T, n), dtype=bool)
        self.slot_in_book = np.zeros((T, n), dtype=np.float32)
        self.passive_w = np.zeros((T, n), dtype=np.float64)
        self.regime_vec = np.zeros((T, 2), dtype=np.float32)

        age_track: dict[str, int] = {}
        for ti, d in enumerate(dates):
            row = bw.loc[d]
            held = row[row > 0]
            syms = list(held.index.astype(str))
            # age bookkeeping
            for s in syms:
                age_track[s] = age_track.get(s, 0) + 1
            for s in list(age_track):
                if s not in syms:
                    age_track.pop(s)
            # rank slots by alpha (stable layout helps the MLP)
            a_row = alpha.loc[d] if d in alpha.index else pd.Series(dtype=float)
            a_vals = a_row.reindex(syms).fillna(0.0)
            std = float(a_vals.std(ddof=0))
            a_z = ((a_vals - a_vals.mean()) / std if std > 1e-9 else a_vals * 0.0)
            order = a_z.sort_values(ascending=False).index.tolist()[:n]
            self.slot_symbols.append(order)
            k = len(order)
            fwd_row = fwd.loc[d] if d in fwd.index else pd.Series(dtype=float)
            r5_row = ret5.loc[d] if d in ret5.index else pd.Series(dtype=float)
            lu = flag["is_limit_up"].loc[d] if d in flag["is_limit_up"].index else pd.Series(dtype=bool)
            ld = flag["is_limit_down"].loc[d] if d in flag["is_limit_down"].index else pd.Series(dtype=bool)
            su = flag["is_suspended"].loc[d] if d in flag["is_suspended"].index else pd.Series(dtype=bool)
            self.slot_ret[ti, :k] = np.nan_to_num(
                fwd_row.reindex(order).to_numpy(dtype=np.float64), nan=0.0)
            self.slot_alpha[ti, :k] = a_z.reindex(order).to_numpy(dtype=np.float32)
            self.slot_ret5[ti, :k] = np.nan_to_num(
                r5_row.reindex(order).to_numpy(dtype=np.float32), nan=0.0)
            self.slot_age[ti, :k] = np.array([min(age_track.get(s, 1), 60) / 60.0
                                              for s in order], dtype=np.float32)
            self.slot_no_increase[ti, :k] = lu.reindex(order).fillna(False).to_numpy(dtype=bool)
            self.slot_frozen[ti, :k] = (ld.reindex(order).fillna(False)
                                        | su.reindex(order).fillna(False)).to_numpy(dtype=bool)
            self.slot_in_book[ti, :k] = 1.0
            self.passive_w[ti, :k] = held.reindex(order).to_numpy(dtype=np.float64)
            tr = float(trail.get(d, np.nan))
            self.regime_vec[ti] = (float(np.isfinite(tr) and tr > 0.05),
                                   float(np.isfinite(tr) and tr < -0.05))

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
        in_book = self.slot_in_book[t].astype(bool)
        passive = self.passive_w[t]
        syms = self.slot_symbols[t]

        # --- policy target: tilt the passive book, adjust gross, re-normalize
        w = passive * (1.0 + cfg.max_tilt * a[:n])
        w = np.where(in_book, np.maximum(w, 0.0), 0.0)
        passive_gross = float(passive.sum())
        gross = float(np.clip(passive_gross * (1.0 + cfg.max_cash_tilt * a[n]),
                              cfg.min_gross, min(cfg.max_gross, 1.0)))
        s = float(w.sum())
        if s > 1e-12:
            w = w * (gross / s)
        # --- tradability constraints vs yesterday's book. The SAME constraint
        # path is applied to the passive benchmark so a zero action stays
        # exactly equal to it (value-add reward is identically 0).
        prev_vec = np.array([self._prev_w.get(s_, 0.0) for s_ in syms]
                            + [0.0] * (n - len(syms)))
        prev_b_vec = np.array([self._prev_w_passive.get(s_, 0.0) for s_ in syms]
                              + [0.0] * (n - len(syms)))
        w = np.where(self.slot_no_increase[t], np.minimum(w, prev_vec), w)
        w = np.where(self.slot_frozen[t], prev_vec, w)
        w_b = np.where(self.slot_no_increase[t], np.minimum(passive, prev_b_vec), passive)
        w_b = np.where(self.slot_frozen[t], prev_b_vec, w_b)

        # --- returns & costs for policy and passive books (symbol-level turnover)
        r = self.slot_ret[t]
        ret_p = float(np.dot(w, r))
        ret_b = float(np.dot(w_b, r))
        cur = {s_: float(w[i]) for i, s_ in enumerate(syms)}
        cur_b = {s_: float(w_b[i]) for i, s_ in enumerate(syms)}
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
        """Diagnostic: can the policy add value via *name selection* at all?

        The env z-scores alpha within the held book each date. If that
        within-book dispersion is ~0 on most dates, name tilts are inert and
        ANY positive value-add must come from the gross/cash-exposure action —
        i.e. a leverage/regime bet, not stock-selection skill. A high
        ``flat_date_fraction`` means a positive value-add should be treated as
        an artifact (the "+39pp env-flat" failure mode), not as alpha. The RL
        enable gate consumes ``env_can_select`` so a flat-env "win" is not
        promoted.
        """
        stds: list[float] = []
        for t in range(len(self.dates)):
            in_book = self.slot_in_book[t].astype(bool)
            a = self.slot_alpha[t][in_book]
            if a.size > 1:
                stds.append(float(np.std(a)))
        arr = np.asarray(stds, dtype=float)
        n = int(arr.size)
        flat = int(np.sum(arr < eps)) if n else 0
        flat_frac = float(flat / n) if n else 1.0
        return {
            "n_dates": n,
            "mean_within_book_alpha_std": float(arr.mean()) if n else 0.0,
            "median_within_book_alpha_std": float(np.median(arr)) if n else 0.0,
            "flat_date_fraction": flat_frac,
            # The env can express stock selection only if most dates have
            # within-book alpha dispersion to tilt on.
            "env_can_select": bool(n > 0 and flat_frac < 0.5),
        }

    def _obs(self) -> np.ndarray:
        cfg = self.config
        n = cfg.max_book
        t = min(self._t, len(self.dates) - 1)
        syms = self.slot_symbols[t]
        prev_vec = np.array([self._prev_w.get(s_, 0.0) for s_ in syms]
                            + [0.0] * (n - len(syms)), dtype=np.float32)
        feats = np.concatenate([
            self.slot_alpha[t],
            self.slot_ret5[t],
            self.slot_age[t],
            prev_vec - self.passive_w[t].astype(np.float32),
            self.slot_in_book[t],
        ])
        n_book = float(self.slot_in_book[t].sum())
        globals_ = np.array([
            self.regime_vec[t][0], self.regime_vec[t][1],
            n_book / max(1, n),
            1.0 - float(sum(self._prev_w.values())),
            t / max(1, len(self.dates) - 1),
        ], dtype=np.float32)
        return np.concatenate([feats, globals_]).astype(np.float32)


def _sym_turnover(prev: dict[str, float], cur: dict[str, float]) -> float:
    keys = set(prev) | set(cur)
    return float(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


__all__ = ["PITPortfolioEnv", "PITPortfolioEnvConfig"]
