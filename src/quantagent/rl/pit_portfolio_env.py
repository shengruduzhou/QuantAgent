"""PIT portfolio environment with an executable T -> T+1 -> T+2 clock.

The environment follows the repository's canonical execution semantics:

* Observation / action time: signal close T.
* Execution constraints: the next market session T+1.
* Reward holding interval: close(T+1) -> close(T+2).
* A training cutoff censors any transition whose *reward end* exceeds the
  cutoff; signal-date truncation alone is not sufficient.
* Tradability is fail-closed on the execution session. A missing bar is accepted
  only when ``session_gaps`` proves ``SUSPENDED``; its valuation then carries
  the last traded close rather than fabricating a bar.
* Limit-up is a no-increase constraint, limit-down is a no-decrease constraint,
  and suspension freezes the position, matching strict A-share broker semantics.
* Universe at each step contains the current passive target plus prior/carried
  names so an untradable exit cannot disappear from accounting.
* Reward is policy value-add versus the same constrained passive book, so a zero
  action earns exactly zero reward.
* Optional risk terms (``drawdown_lambda``, ``volatility_lambda``) price excess
  max drawdown and excess downside semi-variance *against that same passive
  book*. Both default to ``0.0``. The deprecated ``PortfolioEnv`` carried a
  ``drawdown_lambda=2.0`` penalty which was dropped, undocumented, when the
  reward clock was corrected; the reward then encoded only half of the stated
  objective ("highest excess return, smallest drawdown") and was not isomorphic
  to the ``AGENTS.md`` max-drawdown acceptance gate. Turning a term on is an
  explicit caller decision, and must be justified by measurement -- see
  ``docs/audits/round22/11_rl_reward.md``.

This remains research infrastructure. Exported weights require strict replay,
quarantine governance, statistical validation, and forward shadow evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # optional dependency, validated in __init__
    import gymnasium as gym
except Exception:  # pragma: no cover - optional dependency
    gym = None


RL_REWARD_CLOCK_SEMANTICS = (
    "signal_t_close_execute_t_plus_1_close_reward_t_plus_1_to_t_plus_2_close_v1"
)
_REQUIRED_EXECUTION_FLAGS = ("is_limit_up", "is_limit_down", "is_suspended")


@dataclass(frozen=True)
class PITPortfolioEnvConfig:
    """Environment knobs.

    ``drawdown_lambda`` and ``volatility_lambda`` default to ``0.0``, which
    reproduces the pre-round-22 reward **exactly** -- turning a risk penalty on
    is a decision a caller makes, never one this default makes for them. See
    ``PITPortfolioEnv.step`` for what each term costs and in what units.
    """

    max_book: int = 60
    max_tilt: float = 0.8
    max_cash_tilt: float = 0.3
    min_gross: float = 0.5
    max_gross: float = 1.0
    cost_bps: float = 12.0
    reward_scale: float = 100.0
    reward_end_date_limit: str | None = None
    #: Price, in units of cumulative excess return, of one unit of *excess max
    #: drawdown* over the passive book. ``1.0`` means "a percentage point of
    #: excess max drawdown cancels a percentage point of cumulative value-add".
    drawdown_lambda: float = 0.0
    #: Price, in units of cumulative excess return, of one unit of *excess
    #: downside semi-variance* over the passive book (units: 1/return).
    volatility_lambda: float = 0.0


class PITPortfolioEnv(gym.Env if gym is not None else object):
    """Hold-band universe with execution-clock-correct value-add reward."""

    metadata = {"render_modes": []}
    N_SLOT_FEATURES = 5

    def __init__(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
        config: PITPortfolioEnvConfig | None = None,
        *,
        session_gaps: pd.DataFrame | None = None,
    ) -> None:
        if gym is None:  # pragma: no cover - optional dependency
            raise ImportError("PITPortfolioEnv requires gymnasium")
        from gymnasium import spaces

        self.config = config or PITPortfolioEnvConfig()
        if self.config.max_book <= 0:
            raise ValueError("PITPortfolioEnv max_book must be positive")
        for name in ("drawdown_lambda", "volatility_lambda"):
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value < 0.0:
                # A negative lambda would *pay* the agent for running deeper
                # drawdowns than the passive book. That is never a knob anyone
                # wants; accepting it silently would produce a policy whose
                # training objective is the opposite of the acceptance gate.
                raise ValueError(
                    f"PITPortfolioEnv {name} must be finite and >= 0, got {value!r}"
                )
        self._build_caches(
            book_weights,
            predictions,
            market_panel,
            session_gaps=session_gaps,
        )

        n = self.config.max_book
        obs_size = n * self.N_SLOT_FEATURES + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n + 1,),
            dtype=np.float32,
        )
        self._t = 0
        self._prev_w: dict[str, float] = {}
        self._prev_w_passive: dict[str, float] = {}
        self._nav = 1.0
        self._nav_passive = 1.0
        self._peak = 1.0
        self._peak_passive = 1.0
        self._max_drawdown = 0.0
        self._max_drawdown_passive = 0.0

    def _build_caches(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
        *,
        session_gaps: pd.DataFrame | None,
    ) -> None:
        if book_weights is None or book_weights.empty:
            raise ValueError("PITPortfolioEnv requires non-empty book weights")
        if predictions is None or predictions.empty:
            raise ValueError("PITPortfolioEnv requires non-empty predictions")
        if market_panel is None or market_panel.empty:
            raise ValueError("PITPortfolioEnv requires non-empty market panel")

        bw = book_weights.copy()
        bw.index = pd.to_datetime(bw.index, errors="coerce")
        if bw.index.isna().any():
            raise ValueError("PITPortfolioEnv book contains invalid signal dates")
        bw.index = pd.DatetimeIndex(bw.index).normalize()
        if bw.index.duplicated().any():
            raise ValueError("PITPortfolioEnv book contains duplicate signal dates")
        bw = bw.sort_index()
        bw.columns = bw.columns.astype(str)
        bw = bw.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(bw.to_numpy(dtype=float)).all():
            raise ValueError("PITPortfolioEnv book contains non-finite weights")
        if (bw < -1e-12).any().any():
            raise ValueError("PITPortfolioEnv does not support negative book weights")

        panel = market_panel.copy()
        required_panel = {"trade_date", "symbol", "close", *_REQUIRED_EXECUTION_FLAGS}
        missing_panel = sorted(required_panel - set(panel.columns))
        if missing_panel:
            raise ValueError(
                f"PITPortfolioEnv market panel missing required columns: {missing_panel}"
            )
        panel["trade_date"] = pd.to_datetime(
            panel["trade_date"], errors="coerce"
        ).dt.normalize()
        panel["symbol"] = panel["symbol"].astype(str)
        if panel["trade_date"].isna().any() or panel["symbol"].eq("").any():
            raise ValueError("PITPortfolioEnv market panel has invalid date/symbol rows")
        duplicated = panel.duplicated(["trade_date", "symbol"], keep=False)
        if bool(duplicated.any()):
            examples = (
                panel.loc[duplicated, ["trade_date", "symbol"]]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"PITPortfolioEnv market panel has duplicate execution rows: {examples}"
            )
        panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
        panel_index = panel.set_index(["trade_date", "symbol"]).sort_index()
        px = panel.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        ret5 = px / px.shift(5) - 1.0
        gap_index = _prepare_session_gaps(session_gaps)

        preds = predictions.copy()
        if "trade_date" not in preds.columns or "symbol" not in preds.columns:
            raise ValueError("PITPortfolioEnv predictions require trade_date and symbol")
        score_col = "alpha_score" if "alpha_score" in preds.columns else "prediction"
        if score_col not in preds.columns:
            raise ValueError(
                "PITPortfolioEnv predictions require alpha_score or prediction"
            )
        preds["trade_date"] = pd.to_datetime(
            preds["trade_date"], errors="coerce"
        ).dt.normalize()
        preds["symbol"] = preds["symbol"].astype(str)
        if preds["trade_date"].isna().any():
            raise ValueError("PITPortfolioEnv predictions contain invalid trade dates")
        if preds.duplicated(["trade_date", "symbol"]).any():
            raise ValueError(
                "PITPortfolioEnv predictions contain duplicate trade_date/symbol rows"
            )
        preds[score_col] = pd.to_numeric(preds[score_col], errors="coerce")
        alpha = preds.pivot(
            index="trade_date", columns="symbol", values=score_col
        ).sort_index()

        bench = px.pct_change(fill_method=None).mean(axis=1)
        cum = (1 + bench.fillna(0)).cumprod().shift(1)
        trail = cum / cum.shift(60) - 1.0

        sessions = pd.DatetimeIndex(px.index).sort_values().unique()
        session_position = {
            pd.Timestamp(session): index for index, session in enumerate(sessions)
        }
        reward_limit = (
            pd.Timestamp(self.config.reward_end_date_limit).normalize()
            if self.config.reward_end_date_limit
            else None
        )
        triplets: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        for signal_date in bw.index:
            signal = pd.Timestamp(signal_date)
            if signal not in session_position:
                raise ValueError(
                    f"PITPortfolioEnv signal date {signal.date()} is absent from market sessions"
                )
            position = session_position[signal]
            if position + 2 >= len(sessions):
                continue
            execution_date = pd.Timestamp(sessions[position + 1])
            reward_end_date = pd.Timestamp(sessions[position + 2])
            if reward_limit is not None and reward_end_date > reward_limit:
                continue
            triplets.append((signal, execution_date, reward_end_date))
        if len(triplets) < 3:
            raise ValueError(
                "PITPortfolioEnv requires at least 3 signal dates with proven T+1 "
                "execution and T+2 reward sessions inside the configured boundary"
            )

        dates = [item[0] for item in triplets]
        execution_dates = [item[1] for item in triplets]
        reward_end_dates = [item[2] for item in triplets]
        n = self.config.max_book
        T = len(dates)

        self.reward_clock_semantics = RL_REWARD_CLOCK_SEMANTICS
        self.dates = dates
        self.execution_dates = execution_dates
        self.reward_end_dates = reward_end_dates
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
        previous_target_symbols: set[str] = set()
        carried_exit_symbols: set[str] = set()
        for ti, (signal_date, execution_date, reward_end_date) in enumerate(triplets):
            row = bw.loc[signal_date]
            held = row[row > 1e-12]
            current_symbols = list(held.index.astype(str))
            current_set = set(current_symbols)
            if len(current_symbols) > n:
                raise ValueError(
                    f"PITPortfolioEnv signal {signal_date.date()} has "
                    f"{len(current_symbols)} target names > max_book={n}"
                )

            for symbol in current_symbols:
                age_track[symbol] = age_track.get(symbol, 0) + 1

            if signal_date not in alpha.index:
                raise ValueError(
                    f"PITPortfolioEnv missing predictions on signal date "
                    f"{signal_date.date()}"
                )
            alpha_row = alpha.loc[signal_date]
            alpha_values = alpha_row.reindex(current_symbols)
            missing_alpha = alpha_values[
                ~np.isfinite(alpha_values.to_numpy(dtype=float))
            ].index.tolist()
            if missing_alpha:
                raise ValueError(
                    "PITPortfolioEnv missing finite alpha for target names on "
                    f"{signal_date.date()}: {missing_alpha[:10]}"
                )
            alpha_values = alpha_values.astype(float)
            std = float(alpha_values.std(ddof=0))
            alpha_z = (
                (alpha_values - alpha_values.mean()) / std
                if std > 1e-9
                else alpha_values * 0.0
            )

            ranked_current = alpha_z.sort_values(
                ascending=False, kind="mergesort"
            ).index.tolist()
            prior_exposure_candidates = previous_target_symbols | carried_exit_symbols
            liquidation_only = sorted(prior_exposure_candidates - current_set)
            order = ranked_current + liquidation_only
            if len(order) > n:
                raise ValueError(
                    "PITPortfolioEnv transition universe exceeds max_book; "
                    f"signal={signal_date.date()}, required={len(order)}, max_book={n}. "
                    "Increase max_book instead of silently dropping exits."
                )
            self.slot_symbols.append(order)
            k = len(order)

            exec_prices = pd.Series(
                {
                    symbol: _proven_close(
                        symbol,
                        execution_date,
                        px=px,
                        gap_index=gap_index,
                    )
                    for symbol in order
                },
                dtype=float,
            )
            end_prices = pd.Series(
                {
                    symbol: _proven_close(
                        symbol,
                        reward_end_date,
                        px=px,
                        gap_index=gap_index,
                    )
                    for symbol in order
                },
                dtype=float,
            )
            reward_returns = end_prices / exec_prices - 1.0

            no_increase = np.zeros(k, dtype=bool)
            no_decrease = np.zeros(k, dtype=bool)
            frozen = np.zeros(k, dtype=bool)
            for i, symbol in enumerate(order):
                flags = _execution_flags(
                    symbol,
                    execution_date,
                    panel_index=panel_index,
                    gap_index=gap_index,
                )
                no_increase[i] = flags["is_limit_up"]
                no_decrease[i] = flags["is_limit_down"]
                frozen[i] = flags["is_suspended"]

            r5_row = (
                ret5.loc[signal_date]
                if signal_date in ret5.index
                else pd.Series(dtype=float)
            )
            r5_values = r5_row.reindex(order)
            self.slot_ret[ti, :k] = reward_returns.reindex(order).to_numpy(dtype=np.float64)
            self.slot_alpha[ti, :k] = (
                alpha_z.reindex(order).fillna(0.0).to_numpy(dtype=np.float32)
            )
            self.slot_ret5[ti, :k] = np.nan_to_num(
                r5_values.to_numpy(dtype=np.float32), nan=0.0
            )
            self.slot_age[ti, :k] = np.asarray(
                [
                    min(age_track.get(symbol, 0), 60) / 60.0
                    if symbol in current_set
                    else 0.0
                    for symbol in order
                ],
                dtype=np.float32,
            )
            self.slot_no_increase[ti, :k] = no_increase
            self.slot_no_decrease[ti, :k] = no_decrease
            self.slot_frozen[ti, :k] = frozen
            self.slot_in_book[ti, :k] = np.asarray(
                [symbol in current_set for symbol in order],
                dtype=np.float32,
            )
            self.passive_w[ti, :k] = (
                held.reindex(order).fillna(0.0).to_numpy(dtype=np.float64)
            )
            regime = float(trail.get(signal_date, np.nan))
            self.regime_vec[ti] = (
                float(np.isfinite(regime) and regime > 0.05),
                float(np.isfinite(regime) and regime < -0.05),
            )

            carried_exit_symbols = {
                symbol
                for i, symbol in enumerate(order)
                if symbol not in current_set
                and (bool(frozen[i]) or bool(no_decrease[i]))
            }
            for symbol in list(age_track):
                if symbol not in current_set:
                    age_track.pop(symbol)
            previous_target_symbols = current_set

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
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
        self._peak = 1.0
        self._peak_passive = 1.0
        self._max_drawdown = 0.0
        self._max_drawdown_passive = 0.0
        return self._obs(), {}

    def step(self, action):
        cfg = self.config
        t = self._t
        n = cfg.max_book
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (n + 1,):
            raise ValueError(
                f"PITPortfolioEnv action shape {a.shape} != expected {(n + 1,)}"
            )

        in_book = self.slot_in_book[t].astype(bool)
        passive = self.passive_w[t]
        syms = self.slot_symbols[t]
        w = passive * (1.0 + cfg.max_tilt * a[:n])
        w = np.where(in_book, np.maximum(w, 0.0), 0.0)
        passive_gross = float(passive.sum())
        # The gross floor must never lift a book the agent did not ask to lift.
        # Clipping the tilted gross at a fixed ``min_gross`` meant that whenever
        # the passive book was itself below that floor, a zero action produced
        # ``w = passive * (min_gross / passive_gross) != passive`` — the
        # environment levered the book up on its own and then credited the
        # resulting difference to the agent as value-add. The floor therefore
        # binds only down to the passive gross: with ``a[n] == 0`` the tilted
        # gross is exactly ``passive_gross`` and lands inside the interval, so
        # zero action is zero reward by construction.
        gross_floor = min(cfg.min_gross, passive_gross)
        gross_ceiling = min(cfg.max_gross, 1.0)
        gross = float(
            np.clip(
                passive_gross * (1.0 + cfg.max_cash_tilt * a[n]),
                gross_floor,
                max(gross_ceiling, gross_floor),
            )
        )
        weight_sum = float(w.sum())
        if weight_sum > 1e-12:
            w = w * (gross / weight_sum)

        prev_vec = np.array(
            [self._prev_w.get(symbol, 0.0) for symbol in syms]
            + [0.0] * (n - len(syms)),
            dtype=np.float64,
        )
        prev_passive_vec = np.array(
            [self._prev_w_passive.get(symbol, 0.0) for symbol in syms]
            + [0.0] * (n - len(syms)),
            dtype=np.float64,
        )
        w = _apply_execution_constraints(
            w,
            prev_vec,
            no_increase=self.slot_no_increase[t],
            no_decrease=self.slot_no_decrease[t],
            frozen=self.slot_frozen[t],
        )
        w_passive = _apply_execution_constraints(
            passive,
            prev_passive_vec,
            no_increase=self.slot_no_increase[t],
            no_decrease=self.slot_no_decrease[t],
            frozen=self.slot_frozen[t],
        )

        returns = self.slot_ret[t]
        ret_policy = float(np.dot(w, returns))
        ret_passive = float(np.dot(w_passive, returns))
        current = {symbol: float(w[i]) for i, symbol in enumerate(syms)}
        current_passive = {
            symbol: float(w_passive[i]) for i, symbol in enumerate(syms)
        }
        turnover_policy = _sym_turnover(self._prev_w, current)
        turnover_passive = _sym_turnover(self._prev_w_passive, current_passive)
        cost_policy = turnover_policy * cfg.cost_bps / 1e4
        cost_passive = turnover_passive * cfg.cost_bps / 1e4

        net_policy = ret_policy - cost_policy
        net_passive = ret_passive - cost_passive
        value_add = net_policy - net_passive

        self._nav *= 1.0 + net_policy
        self._nav_passive *= 1.0 + net_passive

        # --- risk terms (round 22) ------------------------------------------
        # Both terms are *differences against the same constrained passive
        # book* that the return term already differences against. That is not
        # cosmetic: an absolute risk penalty would charge the agent for the
        # passive book's own drawdown, which it did not choose and cannot
        # avoid, and -- more damagingly -- it would break "zero action earns
        # exactly zero", the construction that makes this environment immune to
        # the env-flat trap (round 21 DEF-036). Under a zero action the policy
        # weights are bit-identical to the passive weights, so both NAV paths,
        # both peaks and both drawdown paths coincide and every term below is
        # identically 0.0.
        #
        # Drawdown term. With MDD_t = max_{s<=t} (1 - NAV_s / peak_s), the
        # per-step increment dMDD_t >= 0 telescopes over the episode to MDD_T.
        # The episode return is therefore
        #     kappa * [ sum_t value_add_t
        #               - lambda_dd * (MDD_policy,T - MDD_passive,T) - ... ]
        # i.e. *isomorphic to the acceptance gate*: cumulative excess return
        # priced against excess max drawdown (AGENTS.md "max drawdown below a
        # configured threshold").
        #
        # Note on potential-based shaping (Ng, Harada & Russell 1999): with
        # Phi(s) = -lambda * MDD(s) and gamma = 1, -lambda * dMDD_t is
        # algebraically a potential-based shaping term, whose theorem would
        # make it policy-*invariant* -- i.e. inert, a penalty that can never
        # change what the agent does. The theorem's escape hatch is its
        # episodic precondition Phi(terminal) = 0. This term deliberately
        # violates it: the telescoped sum is -lambda * MDD_T, which depends on
        # the policy. Enforcing Phi(terminal) = 0 here would cancel the penalty
        # exactly and yield a decorative guard. See docs/audits/round22.
        self._peak = max(self._peak, self._nav)
        self._peak_passive = max(self._peak_passive, self._nav_passive)
        drawdown_policy = 1.0 - self._nav / self._peak
        drawdown_passive = 1.0 - self._nav_passive / self._peak_passive
        previous_mdd = self._max_drawdown
        previous_mdd_passive = self._max_drawdown_passive
        self._max_drawdown = max(previous_mdd, drawdown_policy)
        self._max_drawdown_passive = max(previous_mdd_passive, drawdown_passive)
        drawdown_penalty = cfg.drawdown_lambda * (
            (self._max_drawdown - previous_mdd)
            - (self._max_drawdown_passive - previous_mdd_passive)
        )

        # Downside term. Per-step contribution to the Sortino-style downside
        # semi-variance sigma_down^2 = (1/T) sum_t max(0, -R_t)^2 used by the
        # risk-aware reward literature (arXiv 2506.04358), again differenced
        # against the passive book so upside dispersion is never punished.
        downside_policy = min(net_policy, 0.0) ** 2
        downside_passive = min(net_passive, 0.0) ** 2
        volatility_penalty = cfg.volatility_lambda * (
            downside_policy - downside_passive
        )

        risk_penalty = drawdown_penalty + volatility_penalty
        reward = (value_add - risk_penalty) * cfg.reward_scale

        self._prev_w = current
        self._prev_w_passive = current_passive
        self._t += 1
        terminated = self._t >= len(self.dates)
        info = {
            "trade_date": str(self.dates[t].date()),
            "signal_date": str(self.dates[t].date()),
            "execution_date": str(self.execution_dates[t].date()),
            "reward_end_date": str(self.reward_end_dates[t].date()),
            "reward_clock_semantics": self.reward_clock_semantics,
            "weights": current,
            "net_policy": net_policy,
            "net_passive": net_passive,
            "value_add": value_add,
            "turnover_policy": turnover_policy,
            "turnover_passive": turnover_passive,
            "nav": self._nav,
            "nav_passive": self._nav_passive,
            "drawdown_policy": drawdown_policy,
            "drawdown_passive": drawdown_passive,
            "max_drawdown_policy": self._max_drawdown,
            "max_drawdown_passive": self._max_drawdown_passive,
            "drawdown_penalty": drawdown_penalty,
            "volatility_penalty": volatility_penalty,
            "risk_penalty": risk_penalty,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def book_dispersion_report(self, eps: float = 1e-6) -> dict:
        stds: list[float] = []
        for t in range(len(self.dates)):
            in_book = self.slot_in_book[t].astype(bool)
            alpha = self.slot_alpha[t][in_book]
            if alpha.size > 1:
                stds.append(float(np.std(alpha)))
        arr = np.asarray(stds, dtype=float)
        count = int(arr.size)
        flat = int(np.sum(arr < eps)) if count else 0
        flat_fraction = float(flat / count) if count else 1.0
        return {
            "n_dates": count,
            "mean_within_book_alpha_std": float(arr.mean()) if count else 0.0,
            "median_within_book_alpha_std": (
                float(np.median(arr)) if count else 0.0
            ),
            "flat_date_fraction": flat_fraction,
            "env_can_select": bool(count > 0 and flat_fraction < 0.5),
            "reward_clock_semantics": self.reward_clock_semantics,
            "reward_end_date_limit": self.config.reward_end_date_limit,
        }

    def _obs(self) -> np.ndarray:
        cfg = self.config
        n = cfg.max_book
        t = min(self._t, len(self.dates) - 1)
        syms = self.slot_symbols[t]
        prev_vec = np.array(
            [self._prev_w.get(symbol, 0.0) for symbol in syms]
            + [0.0] * (n - len(syms)),
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


def _prepare_session_gaps(
    session_gaps: pd.DataFrame | None,
) -> dict[tuple[pd.Timestamp, str], str]:
    if session_gaps is None or session_gaps.empty:
        return {}
    required = {"trade_date", "symbol", "classification"}
    missing = sorted(required - set(session_gaps.columns))
    if missing:
        raise ValueError(
            f"PITPortfolioEnv session gaps missing required columns: {missing}"
        )
    gaps = session_gaps.copy()
    gaps["trade_date"] = pd.to_datetime(
        gaps["trade_date"], errors="coerce"
    ).dt.normalize()
    gaps["symbol"] = gaps["symbol"].astype(str)
    gaps["classification"] = gaps["classification"].astype(str).str.upper()
    if gaps["trade_date"].isna().any():
        raise ValueError("PITPortfolioEnv session gaps contain invalid trade dates")
    duplicated = gaps.duplicated(["trade_date", "symbol"], keep=False)
    if bool(duplicated.any()):
        raise ValueError("PITPortfolioEnv session gaps contain duplicate date/symbol rows")
    return {
        (pd.Timestamp(row.trade_date), str(row.symbol)): str(row.classification)
        for row in gaps.itertuples()
    }


def _gap_classification(
    gap_index: dict[tuple[pd.Timestamp, str], str],
    *,
    symbol: str,
    trade_date: pd.Timestamp,
) -> str | None:
    return gap_index.get((pd.Timestamp(trade_date).normalize(), str(symbol)))


def _proven_close(
    symbol: str,
    trade_date: pd.Timestamp,
    *,
    px: pd.DataFrame,
    gap_index: dict[tuple[pd.Timestamp, str], str],
) -> float:
    date = pd.Timestamp(trade_date).normalize()
    if date in px.index and symbol in px.columns:
        value = px.at[date, symbol]
        if _finite_positive(value):
            return float(value)

    classification = _gap_classification(
        gap_index,
        symbol=symbol,
        trade_date=date,
    )
    if classification != "SUSPENDED":
        reason = classification or "NO_GAP_EVIDENCE"
        raise ValueError(
            "PITPortfolioEnv missing/non-positive close without proven suspension "
            f"for {symbol} on {date.date()}: {reason}"
        )
    if symbol not in px.columns:
        raise ValueError(
            f"PITPortfolioEnv suspended symbol {symbol} has no prior price history"
        )
    prior = pd.to_numeric(px.loc[px.index < date, symbol], errors="coerce")
    finite = prior[np.isfinite(prior.to_numpy(dtype=float))]
    finite = finite[finite > 0]
    if finite.empty:
        raise ValueError(
            f"PITPortfolioEnv suspended symbol {symbol} has no prior traded close"
        )
    return float(finite.iloc[-1])


def _execution_flags(
    symbol: str,
    trade_date: pd.Timestamp,
    *,
    panel_index: pd.DataFrame,
    gap_index: dict[tuple[pd.Timestamp, str], str],
) -> dict[str, bool]:
    date = pd.Timestamp(trade_date).normalize()
    key = (date, str(symbol))
    if key in panel_index.index:
        row = panel_index.loc[key]
        return {
            name: _strict_flag(
                row[name],
                name=name,
                symbol=symbol,
                trade_date=date,
            )
            for name in _REQUIRED_EXECUTION_FLAGS
        }

    classification = _gap_classification(
        gap_index,
        symbol=symbol,
        trade_date=date,
    )
    if classification == "SUSPENDED":
        return {
            "is_limit_up": False,
            "is_limit_down": False,
            "is_suspended": True,
        }
    raise ValueError(
        "PITPortfolioEnv missing execution bar without proven SUSPENDED gap for "
        f"{symbol} on {date.date()}: {classification or 'NO_GAP_EVIDENCE'}"
    )


def _apply_execution_constraints(
    desired: np.ndarray,
    previous: np.ndarray,
    *,
    no_increase: np.ndarray,
    no_decrease: np.ndarray,
    frozen: np.ndarray,
) -> np.ndarray:
    result = np.asarray(desired, dtype=np.float64).copy()
    result = np.where(no_increase, np.minimum(result, previous), result)
    result = np.where(no_decrease, np.maximum(result, previous), result)
    result = np.where(frozen, previous, result)
    return result


def _finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def _strict_flag(
    value: object,
    *,
    name: str,
    symbol: str,
    trade_date: pd.Timestamp,
) -> bool:
    if pd.isna(value):
        raise ValueError(
            f"PITPortfolioEnv missing {name} for {symbol} on {trade_date.date()}"
        )
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError(
            f"PITPortfolioEnv invalid {name}={value!r} for {symbol} "
            f"on {trade_date.date()}"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"PITPortfolioEnv invalid {name}={value!r} for {symbol} "
            f"on {trade_date.date()}"
        ) from exc
    if np.isfinite(numeric) and numeric in {0.0, 1.0}:
        return bool(int(numeric))
    raise ValueError(
        f"PITPortfolioEnv invalid {name}={value!r} for {symbol} "
        f"on {trade_date.date()}"
    )


def _sym_turnover(prev: dict[str, float], cur: dict[str, float]) -> float:
    keys = set(prev) | set(cur)
    return float(sum(abs(cur.get(key, 0.0) - prev.get(key, 0.0)) for key in keys))


__all__ = [
    "RL_REWARD_CLOCK_SEMANTICS",
    "PITPortfolioEnv",
    "PITPortfolioEnvConfig",
]
