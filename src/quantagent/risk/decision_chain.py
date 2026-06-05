"""Daily 15-gate decision chain — composite scores → vetted target weights.

The v8 spec section 7 and 8 require that every candidate produced by
the ensemble pass through a 15-gate risk filter *before* being
converted into an order intent. This module implements the chain as
a pure function over a daily candidate frame, the silver market
panel (for ST / suspension / limit-state flags), and the sector_map.

Inputs
------
* ``composite``    : long-form ``trade_date / symbol / composite_score``
                     (typically the output of
                     :func:`ensemble.blend_optimizer.write_blended_composite`).
* ``market_panel`` : silver panel rows for those (date, symbol)s with
                     ``open/high/low/close/volume/amount`` and the
                     state-flag columns (``is_suspended``, ``is_st``,
                     ``is_limit_up``, ``is_limit_down``).
* ``sector_map``   : symbol → sector_level_1 mapping.
* ``sector_pool``  : optional silver/sector_pool with per-sector tier
                     (core / watch / short_term / excluded). If
                     supplied, sector_pool_top_n trims the candidate
                     universe to the top-N best-IC sectors first.

Output
------
* ``DecisionChainResult``
    * ``target_weights``   : wide-form ``trade_date × symbol`` weight
                             frame, equal-weight over the surviving
                             top-K per date.
    * ``decision_traces``  : long-form ``trade_date / symbol`` rows
                             with the per-gate verdict and the final
                             ``accepted`` flag.
    * ``risk_events``      : list of dicts conforming to the v8
                             ``risk_events.json`` schema.
    * ``summary``          : per-gate rejection counts plus survival
                             rate.

The chain does **not** mutate any input frame. It is the only safe
place to translate model scores into target weights for the v8
backtest / paper / live paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.risk.kill_switch import KillSwitch
from quantagent.risk.risk_limits import V6RiskLimits


GATE_NAMES: tuple[str, ...] = (
    "kill_switch",
    "data_quality",
    "model_confidence_decay",
    "sector_pool_top_n",
    "is_suspended",
    "is_st",
    "limit_up_no_buy",
    "consecutive_limit_up_cap",
    "liquidity_floor",
    "single_name_concentration",
    "sector_concentration",
    "drawdown_brake",
    "order_rate_cap",
    "capital_outflow_spike",
    "old_dealer_block",
    "model_score_floor",
)


@dataclass(frozen=True)
class DecisionChainConfig:
    """Behaviour knobs for the 15-gate chain."""

    top_k: int = 30
    candidate_pool_size: int = 0        # 0 = one-stage; else take top-N by model
                                        # score then trend-rank down to top_k
    max_name_weight: float = 0.05
    max_sector_weight: float = 0.30
    max_orders_per_day: int = 200
    max_consecutive_limit_up: int = 2
    min_avg_amount_yuan: float = 5e7
    liquidity_window: int = 20          # trailing days for the PIT liquidity gate
    sector_pool_top_n: int = 0          # 0 disables sector pool filter
    model_score_min: float = float("-inf")
    drawdown_kill_level: float = 0.20
    data_quality_min: float = 0.85
    model_drift_max: float = 0.30
    require_known_sector: bool = False
    long_only: bool = True
    # ── two-stage selection + limit-up execution realism ──────────────
    trend_rank_in_pool: bool = True     # within the candidate pool, order by
                                        # trend/swing quality (not today's gain)
    ma_window: int = 20                 # MA window for trend quality
    max_extension_over_ma: float = 1.20  # close/MA above this = over-extended
    block_one_word_limit_up: bool = True  # 一字涨停 (no intraday range) unfillable
    limit_up_position_cap: float = 0.05   # regular 涨停 allowed only at small size
    allow_limit_up_small_position: bool = True  # False → hard-block all limit-up
    old_dealer_risk_max: float = 0.70  # 老庄股/弱板块弱趋势 hard block
    # ── market-regime gross-exposure scaling (牛市满仓 / 熊市空仓) ─────
    regime_position_scaling: bool = False  # scale the day's gross by market regime
    regime_index_ma_window: int = 60      # market index trend window
    regime_breadth_ma_window: int = 20    # per-stock MA for breadth (% above MA)
    # gross-exposure multiplier per regime; crisis → 0.0 means hold cash (空仓)
    regime_scale_bull: float = 1.00
    regime_scale_neutral_up: float = 0.70
    regime_scale_neutral_down: float = 0.40
    regime_scale_bear: float = 0.15
    regime_scale_crisis: float = 0.00


@dataclass
class DecisionChainState:
    """Cross-date state carried forward through the chain."""

    cumulative_drawdown: float = 0.0
    consecutive_limit_up: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionChainResult:
    target_weights: pd.DataFrame   # wide trade_date × symbol
    decision_traces: pd.DataFrame  # long per (date, symbol)
    risk_events: list[dict]
    summary: dict[str, object]


def _ensure_state_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in out.columns:
            out[col] = False
        else:
            out[col] = out[col].fillna(False).astype(bool)
    if "amount" not in out.columns:
        out["amount"] = np.nan
    return out


def _resolve_sector_pool_universe(
    sector_pool: pd.DataFrame | None,
    sector_map: pd.DataFrame | None,
    *,
    top_n: int,
) -> set[str] | None:
    """If sector_pool is provided + top_n>0, return the symbol whitelist."""
    if sector_pool is None or sector_pool.empty or top_n <= 0:
        return None
    if sector_map is None or sector_map.empty:
        return None
    # rank by ic_ir if present else ic_mean
    pool = sector_pool.copy()
    if "ic_ir" in pool.columns:
        pool = pool.sort_values("ic_ir", ascending=False)
    elif "ic_mean" in pool.columns:
        pool = pool.sort_values("ic_mean", ascending=False)
    pool = pool[pool["pool_tier"].isin(("core", "watch", "short_term"))]
    keep_sectors = pool["sector_level_1"].head(top_n).astype(str).tolist()
    sm = sector_map.copy()
    if "sector_level_1" not in sm.columns:
        return None
    sm["sector_level_1"] = sm["sector_level_1"].astype(str)
    return set(sm[sm["sector_level_1"].isin(keep_sectors)]["symbol"].astype(str))


def _compute_avg_amount(panel: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Trailing ``window``-day rolling avg amount per symbol (liquidity gate).

    PIT-correct: ``rolling`` over chronologically-sorted rows uses only the
    trailing window up to and including each date — never future data — so
    the tradeable-universe decision at date ``t`` is exactly what an
    operator could compute at ``t``.
    """
    panel = panel.sort_values(["symbol", "trade_date"])
    panel["avg_amount_20d"] = (
        panel.groupby("symbol")["amount"].transform(
            lambda s: s.rolling(int(window), min_periods=5).mean()
        )
    )
    return panel


def _compute_trend_quality(panel: pd.DataFrame, *, ma_window: int = 20,
                           max_extension: float = 1.20) -> pd.DataFrame:
    """Per-(symbol,date) trend/swing quality + one-word-limit-up flag (PIT).

    Trend quality rewards healthy uptrends and *penalises chasing* — the v8
    spec's 低吸不追高 philosophy. Components (all from trailing data):

    * ``above_ma``      close ≥ trailing MA(ma_window)          → trend intact
    * ``ma_slope_pos``  MA today > MA 5 bars ago                → uptrend
    * ``not_extended``  close ≤ ``max_extension`` × MA          → not chasing
    * ``pullback``      close within 8 % of MA (低吸 to support) → bonus
    * ``calm_today``    |today's return| ≤ 7 %                   → not a spike

    ``trend_quality`` is their sum (0-5). Used to order the candidate pool so
    the gate loop fills the best-trend names first rather than whichever
    spiked hardest today.

    ``is_one_word_limit_up``: a 一字板 — limit-up with essentially no intraday
    range (high≈low≈close), i.e. unfillable. Detected from OHLC alone.
    """
    panel = panel.sort_values(["symbol", "trade_date"]).copy()
    g = panel.groupby("symbol", sort=False)
    close = pd.to_numeric(panel["close"], errors="coerce")
    panel["_ma"] = g["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(int(ma_window), min_periods=5).mean()
    )
    panel["_ma_prev"] = g["_ma"].transform(lambda s: s.shift(5))
    panel["_ret_1d"] = g["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").pct_change()
    )
    ma = panel["_ma"]
    dist = close / ma
    above_ma = (close >= ma).fillna(False)
    slope_pos = (panel["_ma"] > panel["_ma_prev"]).fillna(False)
    not_extended = (dist <= float(max_extension)).fillna(False)
    pullback = (dist.between(0.92, 1.08)).fillna(False)
    calm_today = (panel["_ret_1d"].abs() <= 0.07).fillna(True)
    panel["trend_quality"] = (
        above_ma.astype(int) + slope_pos.astype(int) + not_extended.astype(int)
        + pullback.astype(int) + calm_today.astype(int)
    ).astype(float)
    panel["dist_to_ma"] = dist

    # one-word limit-up: limit-up flag + (high - low) negligible vs close
    high = pd.to_numeric(panel.get("high"), errors="coerce")
    low = pd.to_numeric(panel.get("low"), errors="coerce")
    rng = (high - low).abs() / close.replace(0.0, np.nan)
    is_lu = panel.get("is_limit_up", False)
    if not isinstance(is_lu, pd.Series):
        is_lu = pd.Series(bool(is_lu), index=panel.index)
    panel["is_one_word_limit_up"] = (is_lu.fillna(False).astype(bool) & (rng.fillna(1.0) < 0.005))
    panel = panel.drop(columns=["_ma", "_ma_prev", "_ret_1d"])
    return panel


def _compute_market_regime(
    panel: pd.DataFrame,
    *,
    config: "DecisionChainConfig",
) -> pd.DataFrame:
    """Per-date market regime + gross-exposure scale (牛市满仓 / 熊市空仓).

    Two PIT signals, both computable at each date from trailing data:

    * **trend** — an equal-weight market index (cumulative mean daily
      return) versus its trailing ``regime_index_ma_window`` MA. Above →
      uptrend, below → downtrend.
    * **breadth** — fraction of names trading above their own trailing MA
      (``dist_to_ma >= 1``). Wide participation → healthy.

    Regime → gross scale:

    | trend        | breadth      | regime        | scale            |
    |--------------|--------------|---------------|------------------|
    | up           | ≥ 0.55       | bull          | 1.00             |
    | up           | 0.40–0.55    | neutral_up    | 0.70             |
    | down         | ≥ 0.40       | neutral_down  | 0.40             |
    | down         | 0.25–0.40    | bear          | 0.15             |
    | down + <MA20 | < 0.25       | crisis        | 0.00 (空仓)       |

    Returns a frame indexed by ``trade_date`` with ``regime`` and
    ``position_scale`` columns.
    """
    df = panel.sort_values(["symbol", "trade_date"]).copy()
    df["_ret1d"] = df.groupby("symbol", sort=False)["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").pct_change()
    )
    mkt_ret = df.groupby("trade_date")["_ret1d"].mean()
    index = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    idx_ma = index.rolling(int(config.regime_index_ma_window), min_periods=10).mean()
    idx_ma_short = index.rolling(int(config.regime_breadth_ma_window), min_periods=5).mean()
    if "dist_to_ma" in df.columns:
        breadth = df.groupby("trade_date")["dist_to_ma"].apply(lambda s: (s >= 1.0).mean())
    else:
        breadth = pd.Series(0.5, index=index.index)

    out = pd.DataFrame({"index": index, "idx_ma": idx_ma,
                        "idx_ma_short": idx_ma_short, "breadth": breadth})
    regimes = []
    scales = []
    for _, r in out.iterrows():
        up = bool(r["index"] >= r["idx_ma"]) if pd.notna(r["idx_ma"]) else True
        below_short = bool(r["index"] < r["idx_ma_short"]) if pd.notna(r["idx_ma_short"]) else False
        b = float(r["breadth"]) if pd.notna(r["breadth"]) else 0.5
        if up and b >= 0.55:
            reg, sc = "bull", config.regime_scale_bull
        elif up and b >= 0.40:
            reg, sc = "neutral_up", config.regime_scale_neutral_up
        elif (not up) and b < 0.25 and below_short:
            reg, sc = "crisis", config.regime_scale_crisis
        elif (not up) and b < 0.40:
            reg, sc = "bear", config.regime_scale_bear
        else:
            reg, sc = "neutral_down", config.regime_scale_neutral_down
        regimes.append(reg)
        scales.append(float(sc))
    out["regime"] = regimes
    out["position_scale"] = scales
    return out[["regime", "position_scale", "breadth"]]


def _run_one_day(
    *,
    date: pd.Timestamp,
    day_panel: pd.DataFrame,
    day_candidates: pd.DataFrame,
    config: DecisionChainConfig,
    state: DecisionChainState,
    sector_map: pd.DataFrame | None,
    sector_whitelist: set[str] | None,
    kill_switch: KillSwitch,
    data_quality_score: float,
    model_drift_score: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the 15-gate chain for one trade_date. Returns (trace_df, events)."""
    trace_rows: list[dict] = []
    events: list[dict] = []
    panel_idx = day_panel.set_index("symbol")
    sector_idx = (
        sector_map.set_index("symbol")["sector_level_1"]
        if sector_map is not None and "sector_level_1" in (sector_map.columns if sector_map is not None else [])
        else None
    )
    ranked = day_candidates.copy().sort_values("composite_score", ascending=False)
    ranked["initial_rank"] = np.arange(len(ranked))

    # ── Two-stage selection: take the model's top-N candidate pool, then
    # re-order WITHIN the pool by a multi-factor quality score (trend/swing +
    # fundamental + news) so the gate loop fills the healthiest names first
    # rather than whichever spiked hardest today (不追当天涨幅).
    if config.candidate_pool_size and config.candidate_pool_size > 0:
        ranked = ranked.head(int(config.candidate_pool_size)).copy()
        if config.trend_rank_in_pool and panel_idx is not None:
            def _col_lookup(col: str) -> "pd.Series | None":
                s = panel_idx.get(col)
                return s if s is not None else None

            def _map(series, sym):
                if series is None or sym not in series.index:
                    return 0.0
                v = series.loc[sym]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                return float(v) if pd.notna(v) else 0.0

            tq = _col_lookup("trend_quality")
            fq = _col_lookup("fundamental_quality")   # optional, PIT-joined by caller
            nq = _col_lookup("news_score")            # optional sentiment hook
            pq = _col_lookup("core_policy_score")
            sr = _col_lookup("sector_resonance_score")
            db = _col_lookup("dip_buy_flow_score")
            od = _col_lookup("old_dealer_risk_score")
            if any(x is not None for x in (tq, fq, nq, pq, sr, db, od)):
                ranked["_pool_score"] = ranked["symbol"].map(
                    lambda s: (
                        _map(tq, s)
                        + _map(fq, s)
                        + _map(nq, s)
                        + _map(pq, s)
                        + _map(sr, s)
                        + _map(db, s)
                        - _map(od, s)
                    )
                )
                # Multi-factor quality first, model score as tie-break.
                ranked = ranked.sort_values(
                    ["_pool_score", "composite_score"], ascending=[False, False]
                ).reset_index(drop=True)

    # ── global pre-checks (cheap; apply to the entire daily batch) ──
    if kill_switch.triggered:
        for _, row in ranked.iterrows():
            trace_rows.append({
                "trade_date": date, "symbol": row["symbol"],
                "composite_score": float(row["composite_score"]),
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "kill_switch", "accepted": False,
            })
        events.append({
            "trade_date": str(date), "event_type": "kill_switch_triggered",
            "n_candidates": int(len(ranked)),
        })
        return pd.DataFrame(trace_rows), events

    if data_quality_score < config.data_quality_min:
        for _, row in ranked.iterrows():
            trace_rows.append({
                "trade_date": date, "symbol": row["symbol"],
                "composite_score": float(row["composite_score"]),
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "data_quality", "accepted": False,
            })
        events.append({
            "trade_date": str(date), "event_type": "data_quality_below_threshold",
            "value": data_quality_score,
        })
        return pd.DataFrame(trace_rows), events

    if model_drift_score > config.model_drift_max:
        for _, row in ranked.iterrows():
            trace_rows.append({
                "trade_date": date, "symbol": row["symbol"],
                "composite_score": float(row["composite_score"]),
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "model_confidence_decay", "accepted": False,
            })
        events.append({
            "trade_date": str(date), "event_type": "model_drift_above_threshold",
            "value": model_drift_score,
        })
        return pd.DataFrame(trace_rows), events

    if state.cumulative_drawdown > config.drawdown_kill_level:
        for _, row in ranked.iterrows():
            trace_rows.append({
                "trade_date": date, "symbol": row["symbol"],
                "composite_score": float(row["composite_score"]),
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "drawdown_brake", "accepted": False,
            })
        events.append({
            "trade_date": str(date), "event_type": "drawdown_brake_active",
            "value": state.cumulative_drawdown,
        })
        return pd.DataFrame(trace_rows), events

    # ── per-name gates ──
    accepted_so_far: list[str] = []
    sector_load: dict[str, float] = {}
    # Equal-weight target — clamped to the per-name hard cap so a small
    # top_k cannot blow through the single-name limit by construction.
    weight_per_name = min(1.0 / float(config.top_k), config.max_name_weight)
    order_budget = config.max_orders_per_day

    for _, row in ranked.iterrows():
        symbol = str(row["symbol"])
        score = float(row["composite_score"])

        # Gate: sector_pool_top_n (cheap)
        if sector_whitelist is not None and symbol not in sector_whitelist:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "sector_pool_top_n", "accepted": False,
            })
            continue

        # Gate: panel coverage required for everything below
        if symbol not in panel_idx.index:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "panel_missing", "accepted": False,
            })
            continue
        p = panel_idx.loc[symbol]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]

        if bool(p.get("is_suspended", False)):
            events.append({"trade_date": str(date), "symbol": symbol,
                           "event_type": "is_suspended"})
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "is_suspended", "accepted": False,
            })
            continue

        if bool(p.get("is_st", False)):
            events.append({"trade_date": str(date), "symbol": symbol,
                           "event_type": "is_st"})
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "is_st", "accepted": False,
            })
            continue

        # ── Limit-up execution realism ──────────────────────────────
        #   一字板 (one-word, no intraday range) → unfillable, hard block.
        #   普通涨停 → fillable but only a small position (cap), per the
        #   operator's rule "涨停成交只允许小仓".
        is_lu = bool(p.get("is_limit_up", False))
        is_one_word = bool(p.get("is_one_word_limit_up", False))
        limit_up_cap_applies = False
        if is_lu:
            if is_one_word and config.block_one_word_limit_up:
                events.append({"trade_date": str(date), "symbol": symbol,
                               "event_type": "one_word_limit_up_no_buy"})
                trace_rows.append({
                    "trade_date": date, "symbol": symbol, "composite_score": score,
                    "initial_rank": int(row["initial_rank"]),
                    "rejected_gate": "one_word_limit_up_no_buy", "accepted": False,
                })
                continue
            if not config.allow_limit_up_small_position:
                events.append({"trade_date": str(date), "symbol": symbol,
                               "event_type": "limit_up_no_buy"})
                trace_rows.append({
                    "trade_date": date, "symbol": symbol, "composite_score": score,
                    "initial_rank": int(row["initial_rank"]),
                    "rejected_gate": "limit_up_no_buy", "accepted": False,
                })
                continue
            limit_up_cap_applies = True  # regular limit-up: small position only

        prior_limit_up_streak = state.consecutive_limit_up.get(symbol, 0)
        if prior_limit_up_streak >= config.max_consecutive_limit_up:
            events.append({"trade_date": str(date), "symbol": symbol,
                           "event_type": "consecutive_limit_up_cap",
                           "streak": prior_limit_up_streak})
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "consecutive_limit_up_cap", "accepted": False,
            })
            continue

        avg_amount = float(p.get("avg_amount_20d", np.nan))
        if pd.notna(avg_amount) and avg_amount < config.min_avg_amount_yuan:
            events.append({"trade_date": str(date), "symbol": symbol,
                           "event_type": "liquidity_floor",
                           "avg_amount_20d": avg_amount})
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "liquidity_floor", "accepted": False,
            })
            continue

        old_dealer_risk = p.get("old_dealer_risk_score", row.get("old_dealer_risk_score", np.nan))
        old_dealer_flag = p.get("old_dealer_block", row.get("old_dealer_block", False))
        old_dealer_block = bool(old_dealer_flag) if pd.notna(old_dealer_flag) else False
        if pd.notna(old_dealer_risk):
            old_dealer_block = old_dealer_block or float(old_dealer_risk) >= config.old_dealer_risk_max
        if old_dealer_block:
            events.append({
                "trade_date": str(date),
                "symbol": symbol,
                "event_type": "old_dealer_block",
                "old_dealer_risk_score": float(old_dealer_risk) if pd.notna(old_dealer_risk) else np.nan,
            })
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "old_dealer_block", "accepted": False,
                "old_dealer_risk_score": float(old_dealer_risk) if pd.notna(old_dealer_risk) else np.nan,
            })
            continue

        # Gate: model_score_floor
        if score < config.model_score_min:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "model_score_floor", "accepted": False,
            })
            continue

        # Gate: sector concentration (cumulative load check)
        sector_name = "unknown"
        sector_known = False
        if sector_idx is not None and symbol in sector_idx.index:
            val = sector_idx.loc[symbol]
            if pd.notna(val):
                sector_name = str(val)
                sector_known = True
        if config.require_known_sector and not sector_known:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "sector_concentration", "accepted": False,
            })
            continue
        # Per-name target weight: equal-weight, but a regular limit-up name
        # is capped to a small position (涨停成交只允许小仓).
        name_weight = weight_per_name
        if limit_up_cap_applies:
            name_weight = min(weight_per_name, config.limit_up_position_cap)

        # When sector membership is unknown, treat each name as its own
        # bucket so the cap does not collapse the whole portfolio into
        # one synthetic "unknown" sector.
        bucket_key = sector_name if sector_known else f"__unknown__{symbol}"
        prospective_load = sector_load.get(bucket_key, 0.0) + name_weight
        if sector_known and prospective_load > config.max_sector_weight:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "sector_concentration", "accepted": False,
                "sector": sector_name, "prospective_load": prospective_load,
            })
            continue

        # Gate: single_name_concentration — already clamped above; this
        # branch only fires if a per-name override would push over.
        if name_weight > config.max_name_weight + 1e-9:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "single_name_concentration", "accepted": False,
                "proposed_weight": name_weight,
            })
            continue

        # Gate: order_rate_cap (per-day orders)
        if order_budget <= 0:
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "order_rate_cap", "accepted": False,
            })
            continue

        # Gate: capital_outflow_spike — placeholder: only available when
        # the panel carries a `daily_net_outflow_z` column. When the
        # field is absent we leave the gate open.
        outflow_z = p.get("daily_net_outflow_z", np.nan)
        if pd.notna(outflow_z) and float(outflow_z) > 3.0:
            events.append({"trade_date": str(date), "symbol": symbol,
                           "event_type": "capital_outflow_spike", "z": float(outflow_z)})
            trace_rows.append({
                "trade_date": date, "symbol": symbol, "composite_score": score,
                "initial_rank": int(row["initial_rank"]),
                "rejected_gate": "capital_outflow_spike", "accepted": False,
            })
            continue

        # Accepted
        accepted_so_far.append(symbol)
        sector_load[bucket_key] = prospective_load
        order_budget -= 1
        trace_rows.append({
            "trade_date": date, "symbol": symbol, "composite_score": score,
            "initial_rank": int(row["initial_rank"]),
            "rejected_gate": None, "accepted": True,
            "sector": sector_name, "weight": name_weight,
            "limit_up_capped": bool(limit_up_cap_applies),
        })
        if len(accepted_so_far) >= config.top_k:
            break

    # ── update streak state from today's limit-up flags ──
    new_streaks: dict[str, int] = {}
    for symbol, row in panel_idx.iterrows():
        sym = str(symbol)
        if bool(row.get("is_limit_up", False)) if isinstance(row, pd.Series) else bool(row["is_limit_up"]):
            new_streaks[sym] = state.consecutive_limit_up.get(sym, 0) + 1
    state.consecutive_limit_up = new_streaks

    return pd.DataFrame(trace_rows), events


def run_decision_chain(
    composite: pd.DataFrame,
    *,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    sector_pool: pd.DataFrame | None = None,
    config: DecisionChainConfig | None = None,
    kill_switch: KillSwitch | None = None,
    data_quality_score: float = 1.0,
    model_drift_score: float = 0.0,
) -> DecisionChainResult:
    """Apply the 15-gate chain to a multi-date composite frame.

    The composite frame must have columns ``trade_date``, ``symbol``,
    ``composite_score``. The output is suitable for direct passing into
    :func:`run_strict_backtest_v8`.
    """
    config = config or DecisionChainConfig()
    kill_switch = kill_switch or KillSwitch()
    panel = _ensure_state_flags(market_panel.copy())
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if "avg_amount_20d" not in panel.columns:
        panel = _compute_avg_amount(panel, window=config.liquidity_window)
    if not {"trend_quality", "dist_to_ma", "is_one_word_limit_up"}.issubset(panel.columns):
        panel = _compute_trend_quality(
            panel, ma_window=config.ma_window, max_extension=config.max_extension_over_ma,
        )
    composite = composite.copy()
    composite["trade_date"] = pd.to_datetime(composite["trade_date"], errors="coerce")

    regime_table = (
        _compute_market_regime(panel, config=config)
        if config.regime_position_scaling else None
    )

    sector_whitelist = _resolve_sector_pool_universe(
        sector_pool, sector_map, top_n=config.sector_pool_top_n,
    )

    all_traces: list[pd.DataFrame] = []
    all_events: list[dict] = []
    state = DecisionChainState()
    target_rows: dict[pd.Timestamp, dict[str, float]] = {}

    for date, day_candidates in composite.groupby("trade_date", sort=True):
        day_panel = panel[panel["trade_date"] == date]
        if day_panel.empty:
            continue
        traces, events = _run_one_day(
            date=date,
            day_panel=day_panel,
            day_candidates=day_candidates,
            config=config,
            state=state,
            sector_map=sector_map,
            sector_whitelist=sector_whitelist,
            kill_switch=kill_switch,
            data_quality_score=data_quality_score,
            model_drift_score=model_drift_score,
        )
        # Market-regime gross-exposure scale (牛市满仓 / 熊市空仓).
        regime_scale = 1.0
        regime_label = None
        if regime_table is not None and date in regime_table.index:
            regime_scale = float(regime_table.loc[date, "position_scale"])
            regime_label = str(regime_table.loc[date, "regime"])
            if regime_scale < 1.0:
                all_events.append({
                    "trade_date": str(date), "event_type": "regime_position_scale",
                    "regime": regime_label, "scale": regime_scale,
                })
        all_traces.append(traces)
        all_events.extend(events)
        if not traces.empty:
            accepted = traces[traces["accepted"]]
            if not accepted.empty and regime_scale > 0.0:
                target_rows[date] = {
                    str(r["symbol"]): float(r.get("weight", 1.0 / config.top_k)) * regime_scale
                    for _, r in accepted.iterrows()
                }
            # regime_scale == 0 → 空仓 (hold cash, no positions that day)

    if all_traces:
        decision_traces = pd.concat(all_traces, ignore_index=True)
    else:
        decision_traces = pd.DataFrame(columns=[
            "trade_date", "symbol", "composite_score",
            "initial_rank", "rejected_gate", "accepted",
        ])

    # Build wide target weights frame
    symbol_universe = sorted({
        sym for d in target_rows for sym in target_rows[d]
    })
    target_weights = pd.DataFrame(
        index=sorted(target_rows.keys()),
        columns=symbol_universe,
        dtype=float,
    ).fillna(0.0)
    for d, weights in target_rows.items():
        for sym, w in weights.items():
            target_weights.loc[d, sym] = w

    # Summary
    if not decision_traces.empty:
        gate_counts = decision_traces["rejected_gate"].fillna("accepted").value_counts().to_dict()
    else:
        gate_counts = {}
    summary = {
        "n_candidates": int(len(decision_traces)),
        "n_accepted": int(decision_traces["accepted"].sum()) if not decision_traces.empty else 0,
        "n_dates": int(len(target_rows)),
        "gate_counts": gate_counts,
        "config": {
            "top_k": config.top_k,
            "max_name_weight": config.max_name_weight,
            "max_sector_weight": config.max_sector_weight,
            "max_consecutive_limit_up": config.max_consecutive_limit_up,
            "min_avg_amount_yuan": config.min_avg_amount_yuan,
            "sector_pool_top_n": config.sector_pool_top_n,
        },
    }

    return DecisionChainResult(
        target_weights=target_weights,
        decision_traces=decision_traces,
        risk_events=all_events,
        summary=summary,
    )


__all__ = [
    "DecisionChainConfig",
    "DecisionChainResult",
    "GATE_NAMES",
    "run_decision_chain",
]
