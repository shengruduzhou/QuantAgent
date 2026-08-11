"""PIT portfolio environment with an executable T -> T+1 -> T+2 clock.

The first PortfolioEnv was rejected because its apparent edge came from universe
lookahead.  The second version removed that bias, but its reward clock still
preceded the repository's canonical execution clock: a signal produced at the
T close was rewarded on close(T)->close(T+1) even though strict execution does
not establish the position until the next market-session close.

This environment makes the execution contract explicit:

* Observation / action time: signal close T.
* Execution constraints: the next market session T+1.
* Reward holding interval: close(T+1) -> close(T+2).
* Tradability is fail-closed on the execution session. Missing/ambiguous flags,
  prices, prediction rows, or required liquidation slots invalidate the env.
* Universe at each step contains the current passive target plus names held by
  the previous target so an untradable exit cannot disappear from accounting.
* Reward remains policy value-add versus the same constrained passive book, so
  a zero action earns exactly zero reward.

The environment is training/research infrastructure only.  Exported weights
must still be re-simulated through ``run_strict_backtest_v8`` and pass the
repository's statistical and risk governance before any promotion.
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
    max_book: int = 60
    max_tilt: float = 0.8
    max_cash_tilt: float = 0.3
    min_gross: float = 0.5
    max_gross: float = 1.0
    cost_bps: float = 12.0
    reward_scale: float = 100.0


class PITPortfolioEnv(gym.Env if gym is not None else object):
    """Hold-band-book universe with execution-clock-correct value-add reward."""

    metadata = {"render_modes": []}

    N_SLOT_FEATURES = 5  # alpha_z, ret_5d, age_norm, prev_minus_target, in_book

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
        if self.config.max_book <= 0:
            raise ValueError("PITPortfolioEnv max_book must be positive")
        self._build_caches(book_weights, predictions, market_panel)

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

    # ------------------------------------------------------------------ setup
    def _build_caches(
        self,
        book_weights: pd.DataFrame,
        predictions: pd.DataFrame,
        market_panel: pd.DataFrame,
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

        # PIT regime: shifted cumulative benchmark never sees T+1 or later.
        bench = px.pct_change(fill_method=None).mean(axis=1)
        cum = (1 + bench.fillna(0)).cumprod().shift(1)
        trail = cum / cum.shift(60) - 1.0

        sessions = pd.DatetimeIndex(px.index).sort_values().unique()
        session_position = {
            pd.Timestamp(session): index for index, session in enumerate(sessions)
        }
        triplets: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        for signal_date in bw.index:
            signal = pd.Timestamp(signal_date)
            if signal not in session_position:
                raise ValueError(
                    f"PITPortfolioEnv signal date {signal.date()} is absent from market sessions"
                )
            position = session_position[signal]
            # The final two signal sessions are right-censored: they cannot prove
            # both an execution session and a post-execution reward session.
            if position + 2 >= len(sessions):
                continue
            triplets.append(
                (
                    signal,
                    pd.Timestamp(sessions[position + 1]),
                    pd.Timestamp(sessions[position + 2]),
                )
            )
        if len(triplets) < 3:
            raise ValueError(
                "PITPortfolioEnv requires at least 3 signal dates with proven T+1 "
                "execution and T+2 reward sessions"
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

            exec_prices = px.loc[execution_date].reindex(order)
            end_prices = px.loc[reward_end_date].reindex(order)
            bad_price_symbols = [
                symbol
                for symbol in order
                if not _finite_positive(exec_prices.get(symbol))
                or not _finite_positive(end_prices.get(symbol))
            ]
            if bad_price_symbols:
                raise ValueError(
                    "PITPortfolioEnv missing/non-positive reward price for "
                    f"{bad_price_symbols[:10]} on execution={execution_date.date()} "
                    f"or reward_end={reward_end_date.date()}"
                )
            reward_returns = end_prices.astype(float) / exec_prices.astype(float) - 1.0

            no_increase = np.zeros(k, dtype=bool)
            frozen = np.zeros(k, dtype=bool)
            for i, symbol in enumerate(order):
                key = (execution_date, symbol)
                if key not in panel_index.index:
                    raise ValueError(
                        "PITPortfolioEnv missing execution row for "
                        f"{symbol} on {execution_date.date()}"
                    )
                execution_row = panel_index.loc[key]
                is_limit_up = _strict_flag(
                    execution_row["is_limit_up"],
                    name="is_limit_up",
                    symbol=symbol,
                    trade_date=execution_date,
                )
                is_limit_down = _strict_flag(
                    execution_row["is_limit_down"],
                    name="is_limit_down",
                    symbol=symbol,
                    trade_date=execution_date,
                )
                is_suspended = _strict_flag(
                    execution_row["is_suspended"],
                    name="is_suspended",
                    symbol=symbol,
                    trade_date=execution_date,
                )
                no_increase[i] = is_limit_up
                frozen[i] = is_limit_down or is_suspended

            r5_row = (
                ret5.loc[signal_date]
                if signal_date in ret5.index
                else pd.Series(dtype=float)
            )
            r5_values = r5_row.reindex(order)
            # Trailing-return features are observations, not economic evidence.
            # Missing history is represented as zero while execution/reward data
            # above remain strictly fail-closed.
            self.slot_ret[ti, :k] = reward_returns.to_numpy(dtype=np.float64)
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

            # A suspended / limit-down exit can remain in the account for more
            # than one session. Carry those liquidation-only names forward until
            # an execution session actually permits the exit; never let them
            # disappear merely because they left the target book.
            carried_exit_symbols = {
                symbol
                for i, symbol in enumerate(order)
                if symbol not in current_set and bool(frozen[i])
            }
            for symbol in list(age_track):
                if symbol not in current_set:
                    age_track.pop(symbol)
            previous_target_symbols = current_set

    # ------------------------------------------------------------------ gym api
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

        # Signal-T target: tilt only names in the passive target. Previous names
        # outside the new target are liquidation-only slots and begin at zero.
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
        weight_sum = float(w.sum())
        if weight_sum > 1e-12:
            w = w * (gross / weight_sum)

        # Execute on T+1. The same execution constraints are applied to policy
        # and passive books, preserving zero-action == zero-value-add.
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
        w = np.where(self.slot_no_increase[t], np.minimum(w, prev_vec), w)
        w = np.where(self.slot_frozen[t], prev_vec, w)
        w_passive = np.where(
            self.slot_no_increase[t],
            np.minimum(passive, prev_passive_vec),
            passive,
        )
        w_passive = np.where(
            self.slot_frozen[t], prev_passive_vec, w_passive
        )

        # Reward begins only after the T+1 close execution: T+1 -> T+2.
        returns = self.slot_ret[t]
        ret_policy = float(np.dot(w, returns))
        ret_passive = float(np.dot(w_passive, returns))
        current = {symbol: float(w[i]) for i, symbol in enumerate(syms)}
        current_passive = {
            symbol: float(w_passive[i]) for i, symbol in enumerate(syms)
        }
        turnover_policy = _sym_turnover(self._prev_w, current)
        turnover_passive = _sym_turnover(
            self._prev_w_passive, current_passive
        )
        cost_policy = turnover_policy * cfg.cost_bps / 1e4
        cost_passive = turnover_passive * cfg.cost_bps / 1e4

        net_policy = ret_policy - cost_policy
        net_passive = ret_passive - cost_passive
        value_add = net_policy - net_passive
        reward = value_add * cfg.reward_scale

        self._nav *= 1.0 + net_policy
        self._nav_passive *= 1.0 + net_passive
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
        }
        return self._obs(), float(reward), bool(terminated), False, info

    # ---------------------------------------------------------------- guards
    def book_dispersion_report(self, eps: float = 1e-6) -> dict:
        """Report whether the policy can express within-book stock selection."""

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
    return float(
        sum(abs(cur.get(key, 0.0) - prev.get(key, 0.0)) for key in keys)
    )


__all__ = [
    "RL_REWARD_CLOCK_SEMANTICS",
    "PITPortfolioEnv",
    "PITPortfolioEnvConfig",
]
