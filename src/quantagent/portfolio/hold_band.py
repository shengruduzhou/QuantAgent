"""Hold-band target-weight builder (turnover-controlled top-K selection).

A hold-band can be used in two distinct contexts and the index meaning must not
be conflated:

* ``delay_days == 0`` -> rows are **signal dated**.  This is the only form that
  may be passed to QuantAgent's strict A-share simulator, which itself maps a
  T-close signal to the next global market session.
* ``delay_days > 0`` -> rows are **execution dated**.  This is useful for
  forward/paper planning files (for example, materialising tomorrow's target),
  but feeding such a matrix into the strict simulator would apply the delay a
  second time.

The returned DataFrame therefore carries ``target_index_semantics`` and
``hold_band_delay_days`` attrs.  The strict simulator rejects execution-dated or
mixed-semantic matrices rather than silently producing a double-T+1 backtest.

The selection rule itself remains unchanged:

  * a name ENTERS only while ranked <= ``entry_rank`` (and there is room),
  * a held name EXITS only when its rank falls below ``exit_rank``,
  * the book holds at most ``n_hold`` names, equal-weighted.

Eligibility: names flagged ST / suspended / limit-up-sealed at signal time are
excluded from both entry and rank maps.  Execution-session tradability is still
owned by the simulator/broker and must not be inferred from this signal-time
filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGNAL_DATE_SEMANTICS = "signal_date"
EXECUTION_DATE_SEMANTICS = "execution_date"
MIXED_DATE_SEMANTICS = "mixed"


@dataclass(frozen=True)
class HoldBandConfig:
    n_hold: int = 50
    entry_rank: int = 30
    exit_rank: int = 150
    delay_days: int = 1
    score_column: str = "alpha_score"

    def __post_init__(self) -> None:
        if int(self.delay_days) < 0:
            raise ValueError("delay_days must be >= 0")


def build_hold_band_weights(
    predictions: pd.DataFrame,
    *,
    config: HoldBandConfig | None = None,
    trade_dates: list[pd.Timestamp] | None = None,
    eligibility_columns: tuple[str, ...] = ("is_st", "is_suspended", "is_limit_up"),
) -> pd.DataFrame:
    """Build an equal-weight hold-band target-weight matrix.

    Parameters
    ----------
    predictions
        Long frame with ``symbol``, ``trade_date``, the score column and
        (optionally pre-joined) boolean eligibility columns.
    trade_dates
        Full trading calendar used only when ``delay_days > 0`` materialises a
        future execution-dated planning matrix.  For a strict backtest set
        ``delay_days=0`` so the output index remains the original signal date;
        the strict simulator owns the sole T -> T+1 mapping.
    """
    cfg = config or HoldBandConfig()
    return _build_weights(
        predictions,
        lambda _d: cfg,
        cfg.score_column,
        trade_dates,
        eligibility_columns,
    )


def build_regime_hold_band_weights(
    predictions: pd.DataFrame,
    *,
    config_map: dict[str, HoldBandConfig],
    regime_by_date: pd.Series,
    default_regime: str = "sideways",
    trade_dates: list[pd.Timestamp] | None = None,
    eligibility_columns: tuple[str, ...] = ("is_st", "is_suspended", "is_limit_up"),
) -> pd.DataFrame:
    """Hold-band weights with regime-conditional band parameters.

    ``regime_by_date`` maps trade_date -> regime label and must itself be PIT
    safe.  If regime configs use different ``delay_days`` values, the returned
    frame is labelled ``mixed`` and is intentionally rejected by the strict
    simulator.
    """
    if default_regime not in config_map:
        raise ValueError(f"config_map missing default regime '{default_regime}'")
    regimes = regime_by_date.copy()
    regimes.index = pd.to_datetime(regimes.index)
    score_column = config_map[default_regime].score_column

    def cfg_for(date: pd.Timestamp) -> HoldBandConfig:
        return config_map.get(
            str(regimes.get(date, default_regime)),
            config_map[default_regime],
        )

    return _build_weights(
        predictions,
        cfg_for,
        score_column,
        trade_dates,
        eligibility_columns,
    )


def _build_weights(
    predictions: pd.DataFrame,
    cfg_for_date,
    score_column: str,
    trade_dates: list[pd.Timestamp] | None,
    eligibility_columns: tuple[str, ...],
) -> pd.DataFrame:
    data = predictions.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol", score_column])

    blocked = pd.Series(False, index=data.index)
    for col in eligibility_columns:
        if col in data.columns:
            blocked |= data[col].fillna(False).astype(bool)
    data = data[~blocked]

    data["_rank"] = data.groupby("trade_date")[score_column].rank(
        ascending=False,
        method="first",
    )

    calendar = (
        pd.DatetimeIndex(sorted(set(trade_dates)))
        if trade_dates is not None
        else pd.DatetimeIndex(sorted(data["trade_date"].unique()))
    )
    position = {d: i for i, d in enumerate(calendar)}

    held: list[str] = []
    rows: dict[pd.Timestamp, pd.Series] = {}
    delays_used: set[int] = set()
    for date, group in data.groupby("trade_date"):
        cfg = cfg_for_date(date)
        if cfg.entry_rank > cfg.exit_rank:
            raise ValueError("entry_rank must be <= exit_rank")
        delay = int(cfg.delay_days)
        if delay < 0:
            raise ValueError("delay_days must be >= 0")
        delays_used.add(delay)

        rank_map = dict(zip(group["symbol"].astype(str), group["_rank"]))
        held = [s for s in held if rank_map.get(s, np.inf) <= cfg.exit_rank]
        if len(held) < cfg.n_hold:
            for sym in group.sort_values("_rank")["symbol"].astype(str):
                if len(held) >= cfg.n_hold:
                    break
                if sym not in held and rank_map[sym] <= cfg.entry_rank:
                    held.append(sym)
        idx = position.get(date)
        if idx is None or not held:
            continue
        output_idx = idx + delay
        if output_idx >= len(calendar):
            continue
        rows[calendar[output_idx]] = pd.Series(1.0 / len(held), index=list(held))

    weights = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    weights.index.name = "trade_date"
    if not delays_used:
        semantics = SIGNAL_DATE_SEMANTICS
        delays = (0,)
    elif delays_used == {0}:
        semantics = SIGNAL_DATE_SEMANTICS
        delays = (0,)
    elif len(delays_used) == 1:
        semantics = EXECUTION_DATE_SEMANTICS
        delays = tuple(sorted(delays_used))
    else:
        semantics = MIXED_DATE_SEMANTICS
        delays = tuple(sorted(delays_used))
    weights.attrs["target_index_semantics"] = semantics
    weights.attrs["hold_band_delay_days"] = delays
    return weights


def turnover_stats(weights: pd.DataFrame) -> dict[str, float]:
    """One-sided daily turnover of a target-weight matrix."""
    if weights.empty or len(weights) < 2:
        return {"mean_daily_turnover": 0.0, "max_daily_turnover": 0.0}
    delta = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    return {
        "mean_daily_turnover": float(delta.mean()),
        "max_daily_turnover": float(delta.max()),
    }


__all__ = [
    "SIGNAL_DATE_SEMANTICS",
    "EXECUTION_DATE_SEMANTICS",
    "MIXED_DATE_SEMANTICS",
    "HoldBandConfig",
    "build_hold_band_weights",
    "build_regime_hold_band_weights",
    "turnover_stats",
]
