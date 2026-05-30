"""V7 prediction → target-weights optimiser.

Translates per-symbol alpha predictions into a tradable
``target_weights`` panel through a constrained, deterministic optimiser
that respects A-share microstructure:

* long-only by default (long-short opt-in via ``long_short=True``)
* max single-name weight cap
* max sector exposure cap
* turnover cap vs the previous day's weights
* liquidity cap (per-symbol max weight from rolling amount * participation)
* ST / suspension / limit constraints
* 100-share lot rounding pre-check via min-trade-amount

Constraints are applied in this order: tradability filter → liquidity
cap → top-K selection → softmax over alpha → sector/single-name cap
projection → turnover cap → renormalisation. The optimiser writes both
the final ``target_weights`` frame and a diagnostics payload (rejected
symbols, applied caps, sector exposures) so callers can audit every
decision.

The implementation never assumes ``cvxpy`` is installed; everything is
plain numpy. If a future user wants tighter constraints they can swap
in cvxpy by editing this module — the API surface is stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_top_k import DynamicTopKConfig, resolve_dynamic_top_k
from .timing_gate import TimingGateConfig, apply_timing_gate
from .position_age_tracker import PositionAgeTracker


@dataclass(frozen=True)
class V7TargetWeightsConfig:
    optimizer_backend: str = "auto"
    objective: str = "max_expected_alpha"
    long_short: bool = False
    # selection_mode: "top_k" (legacy: nlargest by prediction) or
    # "ai_threshold" (filter by prediction > alpha_threshold AND
    # confidence >= confidence_floor, with safety bounds).
    selection_mode: str = "ai_threshold"
    alpha_threshold: float = 0.0
    confidence_floor: float = 0.55
    selection_top_k_min: int = 5
    selection_top_k_max: int = 100
    top_k: int = 30
    top_k_ratio: float | None = 0.10
    min_selection_pressure: float = 3.0
    fail_if_top_k_covers_universe: bool = True
    max_weight_per_name: float = 0.10
    max_sector_weight: float = 0.30
    max_turnover: float = 0.50
    cost_bps: float = 12.0
    liquidity_participation: float = 0.05
    min_amount_yuan: float = 0.0
    min_universe: int = 1
    cash_floor: float = 0.0
    weighting: str = "rank"  # equal | rank | softmax
    alpha_temperature: float = 1.0
    capital_yuan: float = 1_000_000.0
    horizon_column: str | None = None
    block_st: bool = True
    block_suspended: bool = True
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    # Phase 3 dynamic / lifecycle controls.
    dynamic_top_k_enabled: bool = False
    top_k_min: int = 8
    top_k_max: int = 50
    timing_gate_enabled: bool = False
    holding_period_mode: str = "off"  # off | soft | hard
    holding_period_max_delta: float = 0.02  # |Δw| ceiling for locked names
    shrink_on_small_universe: bool = False
    # ((capital_threshold, participation_rate)) ladder — first row whose
    # threshold is ≥ capital_yuan wins. Set to () to disable.
    capital_tier_overrides: tuple[tuple[float, float], ...] = ()


# Tradability constraint table: (market_column, config_attribute, audit_reason).
# This is the single source of truth for which configuration flag gates
# which market-panel column. Keep it in sync with ``V7TargetWeightsConfig``.
_TRADABILITY_CONSTRAINTS: tuple[tuple[str, str, str], ...] = (
    ("is_suspended", "block_suspended", "suspended"),
    ("is_st", "block_st", "st"),
    ("is_limit_up", "block_limit_up_buy", "limit_up_buy_block"),
    ("is_limit_down", "block_limit_down_sell", "limit_down_sell_block"),
)


@dataclass(frozen=True)
class V7TargetWeightsResult:
    target_weights: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


_TRADABILITY_FLAGS: tuple[str, ...] = tuple(column for column, *_ in _TRADABILITY_CONSTRAINTS)
_SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "max_expected_alpha",
    "min_turnover",
    "max_information_ratio_proxy",
)


def _effective_participation_rate(config: V7TargetWeightsConfig) -> float:
    """Derive the effective liquidity participation rate from the
    capital-tier ladder. Returns ``config.liquidity_participation``
    unchanged when no override matches.
    """

    if not config.capital_tier_overrides:
        return float(config.liquidity_participation)
    ladder = sorted(config.capital_tier_overrides, key=lambda row: float(row[0]))
    rate = float(config.liquidity_participation)
    for threshold, candidate in ladder:
        if float(config.capital_yuan) >= float(threshold):
            rate = float(candidate)
    return rate


def build_v7_target_weights(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    config: V7TargetWeightsConfig | None = None,
    *,
    theme_signals: pd.DataFrame | None = None,
    timing_plan: pd.DataFrame | None = None,
    position_state_path: Path | None = None,
) -> V7TargetWeightsResult:
    """Convert per-symbol predictions into a daily target-weights panel.

    The optional keyword arguments enable Phase 3 dynamic behaviour:

    * ``theme_signals`` — frame with ``trade_date``, ``symbol``,
      ``lifecycle_stage``, ``policy_strength``, ``confidence``,
      ``expected_horizon_days``. Drives dynamic ``top_k`` and the
      holding-period lock.
    * ``timing_plan`` — frame from
      :func:`agents.technical_timing_agent.compute_technical_timing`
      (or any compatible producer). When ``timing_gate_enabled`` is
      ``True`` the optimiser uses it to gate new opens and force closes.
    * ``position_state_path`` — parquet path the position-age tracker
      persists to. State survives walk-forward fold boundaries, so the
      holding-period constraint actually binds.
    """

    config = config or V7TargetWeightsConfig()
    if config.objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported optimizer objective: {config.objective}; supported: {_SUPPORTED_OBJECTIVES}")
    if predictions is None or predictions.empty:
        return V7TargetWeightsResult(pd.DataFrame(), {"status": "no_predictions"})
    if market_panel is None or market_panel.empty:
        return V7TargetWeightsResult(pd.DataFrame(), {"status": "no_market_panel"})

    preds = predictions.copy()
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce")
    preds = preds.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)
    if "prediction" not in preds.columns:
        alpha_columns = [c for c in preds.columns if c.startswith("alpha_")]
        if not alpha_columns:
            raise ValueError("predictions frame must include 'prediction' or 'alpha_*' columns")
        column = config.horizon_column or alpha_columns[0]
        preds = preds.rename(columns={column: "prediction"})
        prediction_source = column
    elif "risk_adjusted_prediction" in preds.columns:
        preds["prediction"] = pd.to_numeric(preds["risk_adjusted_prediction"], errors="coerce")
        prediction_source = "risk_adjusted_prediction"
    else:
        prediction_source = "prediction"

    market = market_panel.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market = market.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)

    sector_lookup: dict[str, str] = {}
    if sector_map is not None and not sector_map.empty:
        sector_lookup = (
            sector_map.dropna(subset=["symbol"])
            .groupby("symbol")["industry"]
            .last()
            .astype(str)
            .to_dict()
        )

    by_date_weights: list[pd.DataFrame] = []
    diagnostics: dict[str, list[dict[str, object]]] = {
        "rejected": [],
        "exposures": [],
        "daily_selection": [],
        "alpha_distribution_by_date": [],
        "warnings": [],
        "dynamic_top_k_decisions": [],
        "timing_gate_summary": [],
        "holding_period_locks": [],
    }
    previous_weights: pd.Series | None = None

    effective_participation = _effective_participation_rate(config)

    theme_frame: pd.DataFrame | None = None
    if theme_signals is not None and not theme_signals.empty:
        theme_frame = theme_signals.copy()
        theme_frame["trade_date"] = pd.to_datetime(theme_frame["trade_date"], errors="coerce")
        theme_frame = theme_frame.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)

    timing_gate_decisions: pd.DataFrame | None = None
    timing_gate_diag: dict[str, object] | None = None
    if config.timing_gate_enabled and timing_plan is not None and not timing_plan.empty:
        gate_cfg = TimingGateConfig(enabled=True, require_in_entry_zone=True, enforce_invalidation=True)
        gate_result = apply_timing_gate(market_panel, timing_plan, gate_cfg)
        timing_gate_decisions = gate_result.decisions
        timing_gate_diag = gate_result.diagnostics

    age_tracker: PositionAgeTracker | None = None
    if config.holding_period_mode != "off":
        age_tracker = (
            PositionAgeTracker.from_state(Path(position_state_path))
            if position_state_path is not None
            else PositionAgeTracker()
        )

    dynamic_topk_cfg = DynamicTopKConfig(
        top_k_min=int(config.top_k_min),
        top_k_max=int(config.top_k_max),
        base_top_k=int(config.top_k),
    )

    for date, day in preds.groupby("trade_date", sort=True):
        day_market = market[market["trade_date"] == date]
        merged = day.merge(day_market, on=["symbol", "trade_date"], how="left", suffixes=("", "_mkt"))
        rejected: list[dict[str, object]] = []
        keep_mask = pd.Series(True, index=merged.index)
        for column, config_attr, reason in _TRADABILITY_CONSTRAINTS:
            if column not in merged.columns:
                continue
            if not getattr(config, config_attr, False):
                continue
            blocked = merged[column].fillna(False).astype(bool)
            for symbol in merged.loc[blocked, "symbol"]:
                rejected.append({"trade_date": str(date), "symbol": str(symbol), "reason": reason})
            keep_mask = keep_mask & ~blocked
        if "amount" in merged.columns and config.min_amount_yuan > 0:
            illiquid = merged["amount"].fillna(0.0) < config.min_amount_yuan
            for symbol in merged.loc[illiquid, "symbol"]:
                rejected.append({"trade_date": str(date), "symbol": str(symbol), "reason": "illiquid"})
            keep_mask = keep_mask & ~illiquid
        eligible = merged[keep_mask].copy()
        if eligible.empty or len(eligible) < config.min_universe:
            diagnostics["rejected"].extend(rejected)
            continue
        # Liquidity cap (max weight by participation in amount), with capital tiering.
        if "amount" in eligible.columns and config.capital_yuan > 0:
            amount = pd.to_numeric(eligible["amount"], errors="coerce")
            if amount.notna().sum() == 0:
                eligible["liquidity_cap"] = config.max_weight_per_name
                diagnostics["warnings"].append(
                    {
                        "trade_date": str(date),
                        "warning": "liquidity_amount_all_missing_cap_disabled",
                        "eligible_count": int(len(eligible)),
                    }
                )
            else:
                cap = (amount.fillna(np.nan) * effective_participation) / max(1.0, config.capital_yuan)
                cap = cap.where(amount.notna(), config.max_weight_per_name)
                missing_amount_count = int(amount.isna().sum())
                if missing_amount_count:
                    diagnostics["warnings"].append(
                        {
                            "trade_date": str(date),
                            "warning": "liquidity_amount_partial_missing_default_name_cap",
                            "missing_amount_count": missing_amount_count,
                            "eligible_count": int(len(eligible)),
                        }
                    )
                eligible["liquidity_cap"] = cap.clip(lower=0.0, upper=config.max_weight_per_name)
        else:
            eligible["liquidity_cap"] = config.max_weight_per_name

        # Apply timing gate (Phase 3.3): drop names whose previous-day
        # close fell outside the entry zone; mark force_close on
        # invalidation breaches. Strictly post-A-share-filter so it
        # cannot bypass the suspension/ST/limit gates.
        force_close_symbols: set[str] = set()
        if timing_gate_decisions is not None and not timing_gate_decisions.empty:
            gate_today = timing_gate_decisions[timing_gate_decisions["trade_date"] == date]
            if not gate_today.empty:
                blocked = set(gate_today.loc[~gate_today["allow_open"].astype(bool), "symbol"].astype(str))
                force_close_symbols = set(gate_today.loc[gate_today["force_close"].astype(bool), "symbol"].astype(str))
                if blocked:
                    held = set(previous_weights.index.astype(str)) if previous_weights is not None else set()
                    new_only_blocked = blocked - held
                    if new_only_blocked:
                        eligible = eligible[~eligible["symbol"].astype(str).isin(new_only_blocked)]
                diagnostics["timing_gate_summary"].append(
                    {
                        "trade_date": str(date),
                        "blocked_new_entries": int(len(blocked)),
                        "force_close": int(len(force_close_symbols)),
                    }
                )

        if eligible.empty:
            diagnostics["rejected"].extend(rejected)
            continue

        # Selection: either AI-threshold (prediction & confidence gates with
        # min/max safety bounds) or legacy top-K (nlargest by prediction).
        alpha = eligible["prediction"].astype(float)
        if config.selection_mode == "ai_threshold":
            pool = eligible.copy()
            pool = pool[pool["prediction"].astype(float) > float(config.alpha_threshold)]
            if "confidence" in pool.columns and float(config.confidence_floor) > 0:
                pool = pool[pool["confidence"].astype(float) >= float(config.confidence_floor)]
            min_n = max(1, int(config.selection_top_k_min))
            max_n = max(min_n, int(config.selection_top_k_max))
            fallback_to_min = False
            capped_at_max = False
            if len(pool) < min_n:
                pool = eligible.nlargest(min(min_n, len(eligible)), "prediction")
                fallback_to_min = True
            if len(pool) > max_n:
                pool = pool.nlargest(max_n, "prediction")
                capped_at_max = True
            longs = pool
            shorts = pd.DataFrame(columns=eligible.columns)
            selected_count = int(len(longs))
            effective_top_k = selected_count
            scaled = _selection_weights(longs["prediction"].to_numpy(dtype=float), config.weighting, config.alpha_temperature)
            weights = pd.Series(scaled, index=pd.Index(longs["symbol"].to_numpy(), name="symbol"))
            diagnostics.setdefault("optimizer_backend", []).append({
                "trade_date": str(date),
                "backend": "ai_threshold",
                "alpha_threshold": float(config.alpha_threshold),
                "confidence_floor": float(config.confidence_floor),
                "selected_count": selected_count,
                "eligible_count": int(len(eligible)),
                "fallback_to_min": fallback_to_min,
                "capped_at_max": capped_at_max,
            })
        else:
            if config.dynamic_top_k_enabled:
                theme_today = (
                    theme_frame[theme_frame["trade_date"] == date]
                    if theme_frame is not None
                    else None
                )
                decision = resolve_dynamic_top_k(
                    eligible_count=len(eligible),
                    predictions_for_date=eligible["prediction"],
                    theme_signals_for_date=theme_today,
                    config=dynamic_topk_cfg,
                )
                effective_top_k = int(max(1, decision.top_k))
                diagnostics["dynamic_top_k_decisions"].append({"trade_date": str(date), **decision.diagnostics, "top_k": effective_top_k, "contributions": decision.contributions})
            else:
                effective_top_k = _effective_top_k(len(eligible), config)
            if effective_top_k >= len(eligible):
                if config.dynamic_top_k_enabled or config.shrink_on_small_universe:
                    effective_top_k = max(1, len(eligible) - 1)
                elif config.fail_if_top_k_covers_universe:
                    raise ValueError(
                        "top_k selection covers the eligible universe; increase the universe or lower --top-k/--top-k-ratio"
                    )
            if config.long_short:
                top_k = min(effective_top_k, len(eligible) // 2 or 1)
                longs = eligible.nlargest(top_k, "prediction")
                shorts = eligible.nsmallest(top_k, "prediction")
                selected_count = int(pd.concat([longs[["symbol"]], shorts[["symbol"]]]).drop_duplicates("symbol").shape[0])
                longs_w = _selection_weights(longs["prediction"].to_numpy(dtype=float), config.weighting, config.alpha_temperature)
                shorts_w = -_selection_weights(-shorts["prediction"].to_numpy(dtype=float), config.weighting, config.alpha_temperature)
                weights = pd.Series(
                    np.concatenate([longs_w, shorts_w]),
                    index=pd.Index(np.concatenate([longs["symbol"].to_numpy(), shorts["symbol"].to_numpy()]), name="symbol"),
                )
            else:
                top_k = effective_top_k
                longs = eligible.nlargest(top_k, "prediction")
                shorts = pd.DataFrame(columns=eligible.columns)
                selected_count = int(len(longs))
                cvx_weights, cvx_note = _try_cvxpy_long_only(longs, previous_weights, sector_lookup, config)
                if cvx_weights is not None:
                    weights = cvx_weights
                    diagnostics.setdefault("optimizer_backend", []).append({"trade_date": str(date), "backend": "cvxpy", "note": cvx_note})
                else:
                    scaled = _selection_weights(longs["prediction"].to_numpy(dtype=float), config.weighting, config.alpha_temperature)
                    weights = pd.Series(scaled, index=pd.Index(longs["symbol"].to_numpy(), name="symbol"))
                    diagnostics.setdefault("optimizer_backend", []).append({"trade_date": str(date), "backend": "deterministic", "note": cvx_note})
        selection_pressure = float(len(eligible) / max(selected_count, 1))
        whether_selection_is_real = bool(selection_pressure >= config.min_selection_pressure and selected_count < len(eligible))
        selected_symbols = set(longs["symbol"].astype(str))
        if config.long_short and not shorts.empty:
            selected_symbols |= set(shorts["symbol"].astype(str))
        selected_alpha = eligible[eligible["symbol"].astype(str).isin(selected_symbols)]["prediction"].astype(float)
        unselected_alpha = eligible[~eligible["symbol"].astype(str).isin(selected_symbols)]["prediction"].astype(float)
        alpha_stats = _alpha_distribution_stats(selected_alpha, unselected_alpha)
        if alpha_stats["selected_vs_unselected_alpha_spread"] <= 0:
            diagnostics["warnings"].append(
                {
                    "trade_date": str(date),
                    "warning": "selected_alpha_not_above_unselected_alpha",
                    "selected_vs_unselected_alpha_spread": alpha_stats["selected_vs_unselected_alpha_spread"],
                }
            )
        diagnostics["daily_selection"].append(
            {
                "trade_date": str(date),
                "eligible_count": int(len(eligible)),
                "selected_count": selected_count,
                "effective_top_k": int(effective_top_k),
                "selection_pressure": selection_pressure,
                "top_k_ratio": config.top_k_ratio,
                "whether_selection_is_real": whether_selection_is_real,
            }
        )
        diagnostics["alpha_distribution_by_date"].append({"trade_date": str(date), **alpha_stats})
        if selection_pressure < config.min_selection_pressure and config.selection_mode != "ai_threshold":
            raise ValueError(
                f"selection_pressure={selection_pressure:.3f} is below min_selection_pressure={config.min_selection_pressure:.3f}"
            )
        # Apply per-name cap.
        caps = longs.set_index("symbol")["liquidity_cap"] if not config.long_short else pd.concat(
            [longs.set_index("symbol")["liquidity_cap"], shorts.set_index("symbol")["liquidity_cap"]]
        )
        weights = weights.clip(lower=-config.max_weight_per_name, upper=config.max_weight_per_name)
        weights = weights.clip(lower=-caps.reindex(weights.index).fillna(config.max_weight_per_name),
                                upper=caps.reindex(weights.index).fillna(config.max_weight_per_name))

        # Apply sector cap by iterative scaling.
        if sector_lookup:
            sector_series = weights.index.to_series().map(sector_lookup).fillna("__unknown__")
            for _ in range(5):
                exposures = weights.abs().groupby(sector_series).sum()
                breaches = exposures[exposures > config.max_sector_weight]
                if breaches.empty:
                    break
                for sector, value in breaches.items():
                    scale = config.max_sector_weight / max(value, 1e-9)
                    mask = (sector_series == sector).reindex(weights.index).fillna(False).to_numpy()
                    weights.loc[mask] = weights.loc[mask] * scale

        # Re-normalise (long-only sums to 1; long-short sums to 0 but gross capped).
        gross = float(weights.abs().sum())
        if gross > 0:
            target_gross = 1.0
            weights = weights * (target_gross / gross)
        # Final hard ceiling after renormalisation. Renormalisation can
        # push a single name back above the per-name / liquidity cap, so
        # we clip again and then re-distribute residual weight across the
        # remaining un-capped names.
        per_name_cap = caps.reindex(weights.index).fillna(config.max_weight_per_name)
        for _ in range(5):
            clipped = weights.clip(lower=-per_name_cap, upper=per_name_cap)
            spillover = float(weights.abs().sum() - clipped.abs().sum())
            weights = clipped
            if spillover <= 1e-9:
                break
            slack = per_name_cap - weights.abs()
            slack[slack < 0] = 0.0
            slack_total = float(slack.sum())
            if slack_total <= 0:
                break
            redistribute = slack / slack_total * spillover
            sign = np.sign(weights.replace(0.0, 1.0))
            weights = weights + redistribute * sign

        # Holding-period lock (Phase 3.4): force ``|Δw| ≤ holding_period_max_delta``
        # for names whose age < expected_horizon, unless timing-gate
        # marks them force_close.
        locked_symbols: list[str] = []
        if age_tracker is not None and previous_weights is not None:
            prev_aligned = previous_weights.reindex(weights.index).fillna(0.0)
            for symbol in list(weights.index.astype(str)):
                if symbol in force_close_symbols:
                    weights.loc[symbol] = 0.0
                    continue
                if age_tracker.is_locked(symbol, date, force_close=False):
                    prev_w = float(prev_aligned.get(symbol, 0.0))
                    delta = float(weights.loc[symbol] - prev_w)
                    if abs(delta) > config.holding_period_max_delta:
                        direction = 1.0 if delta >= 0 else -1.0
                        weights.loc[symbol] = prev_w + direction * config.holding_period_max_delta
                        locked_symbols.append(symbol)
        if locked_symbols:
            diagnostics["holding_period_locks"].append(
                {"trade_date": str(date), "locked_symbols": locked_symbols}
            )

        # Apply turnover cap vs previous weights.
        if previous_weights is not None and config.max_turnover > 0:
            blended = _apply_turnover_cap(weights, previous_weights, config.max_turnover)
            weights = blended
        previous_weights = weights.copy()

        if age_tracker is not None:
            expected_horizons: dict[str, int | None] = {}
            if theme_frame is not None:
                today_theme = theme_frame[theme_frame["trade_date"] == date]
                if not today_theme.empty and "expected_horizon_days" in today_theme.columns:
                    for sym, eh in zip(today_theme["symbol"], today_theme["expected_horizon_days"]):
                        if pd.notna(eh):
                            expected_horizons[str(sym)] = int(eh)
            age_tracker.record_session(date, weights.to_dict(), expected_horizons)

        exposures_report: dict[str, float] = {}
        if sector_lookup:
            exposures_report = (
                weights.abs()
                .groupby(weights.index.to_series().map(sector_lookup).fillna("__unknown__"))
                .sum()
                .astype(float)
                .to_dict()
            )
        diagnostics["exposures"].append({"trade_date": str(date), "sector_gross": exposures_report})
        diagnostics["rejected"].extend(rejected)

        by_date_weights.append(
            pd.DataFrame(
                [
                    {"trade_date": date, "symbol": symbol, "weight": float(value)}
                    for symbol, value in weights.items()
                ]
            )
        )

    if not by_date_weights:
        return V7TargetWeightsResult(pd.DataFrame(), {"status": "all_dates_rejected", **diagnostics})

    long_format = pd.concat(by_date_weights, ignore_index=True)
    pivot = long_format.pivot_table(index="trade_date", columns="symbol", values="weight", aggfunc="last").fillna(0.0)
    diagnostics_payload = {
        "status": "passed",
        "raw_symbol_count": int(preds["symbol"].nunique()),
        "market_panel_symbol_count": int(market["symbol"].nunique()),
        "prediction_symbol_count": int(preds["symbol"].nunique()),
        "eligible_symbol_count_by_date": {
            item["trade_date"]: item["eligible_count"] for item in diagnostics["daily_selection"]
        },
        "selected_symbol_count_by_date": {
            item["trade_date"]: item["selected_count"] for item in diagnostics["daily_selection"]
        },
        "selection_pressure_mean": float(np.mean([item["selection_pressure"] for item in diagnostics["daily_selection"]]))
        if diagnostics["daily_selection"]
        else 0.0,
        "selection_pressure_min": float(np.min([item["selection_pressure"] for item in diagnostics["daily_selection"]]))
        if diagnostics["daily_selection"]
        else 0.0,
        "selected_count_mean": float(np.mean([item["selected_count"] for item in diagnostics["daily_selection"]]))
        if diagnostics["daily_selection"]
        else 0.0,
        "selected_count_min": int(np.min([item["selected_count"] for item in diagnostics["daily_selection"]]))
        if diagnostics["daily_selection"]
        else 0,
        "selected_count_max": int(np.max([item["selected_count"] for item in diagnostics["daily_selection"]]))
        if diagnostics["daily_selection"]
        else 0,
        "whether_selection_is_real": bool(
            diagnostics["daily_selection"]
            and all(bool(item["whether_selection_is_real"]) for item in diagnostics["daily_selection"])
        ),
        "prediction_source": prediction_source,
        "dates": int(pivot.shape[0]),
        "symbol_count": int(pivot.shape[1]),
        "average_gross_exposure": float(pivot.abs().sum(axis=1).mean()),
        "average_turnover": float(pivot.diff().abs().sum(axis=1).mean()),
        "supported_objectives": list(_SUPPORTED_OBJECTIVES),
        "constraint_surface": {
            "sector_cap": config.max_sector_weight,
            "single_name_cap": config.max_weight_per_name,
            "liquidity_cap": config.liquidity_participation,
            "cash_floor": config.cash_floor,
            "long_short": config.long_short,
            "max_turnover": config.max_turnover,
            "weighting": config.weighting,
        },
        "config": asdict(config),
        **diagnostics,
    }
    pivot.index.name = "trade_date"

    if age_tracker is not None:
        persisted = age_tracker.persist()
        if persisted is not None:
            diagnostics_payload["position_state_path"] = str(persisted)
        diagnostics_payload["position_state_rows"] = int(len(age_tracker.snapshot()))
    if timing_gate_diag is not None:
        diagnostics_payload["timing_gate"] = timing_gate_diag
    diagnostics_payload["effective_participation_rate"] = float(effective_participation)

    return V7TargetWeightsResult(pivot.reset_index(), diagnostics_payload)


def _softmax_weights(values: np.ndarray, temperature: float) -> np.ndarray:
    if values.size == 0:
        return values
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = safe / max(temperature, 1e-6)
    scaled -= scaled.max()
    expo = np.exp(scaled)
    total = expo.sum()
    if total <= 0:
        return np.ones_like(safe) / len(safe)
    return expo / total


def _selection_weights(values: np.ndarray, weighting: str, temperature: float) -> np.ndarray:
    safe = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if safe.size == 0:
        return safe
    mode = str(weighting).lower()
    if mode == "equal":
        return np.ones_like(safe) / len(safe)
    if mode == "softmax":
        return _softmax_weights(safe, temperature)
    ranks = pd.Series(safe).rank(method="average").to_numpy(dtype=float)
    total = float(ranks.sum())
    if total <= 0:
        return np.ones_like(safe) / len(safe)
    return ranks / total


def _effective_top_k(eligible_count: int, config: V7TargetWeightsConfig) -> int:
    if eligible_count <= 0:
        return 0
    if config.top_k_ratio is None:
        ratio_cap = int(config.top_k)
    else:
        ratio_cap = int(np.floor(float(eligible_count) * float(config.top_k_ratio)))
    return max(1, min(int(config.top_k), max(1, ratio_cap), int(eligible_count)))


def _alpha_distribution_stats(selected_alpha: pd.Series, unselected_alpha: pd.Series) -> dict[str, float]:
    def stats(prefix: str, values: pd.Series) -> dict[str, float]:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return {f"{prefix}_alpha_min": 0.0, f"{prefix}_alpha_mean": 0.0, f"{prefix}_alpha_max": 0.0}
        return {
            f"{prefix}_alpha_min": float(clean.min()),
            f"{prefix}_alpha_mean": float(clean.mean()),
            f"{prefix}_alpha_max": float(clean.max()),
        }

    payload = {**stats("selected", selected_alpha), **stats("unselected", unselected_alpha)}
    payload["selected_vs_unselected_alpha_spread"] = (
        payload["selected_alpha_mean"] - payload["unselected_alpha_mean"]
    )
    return payload


def _apply_turnover_cap(target: pd.Series, previous: pd.Series, cap: float) -> pd.Series:
    aligned_prev = previous.reindex(target.index).fillna(0.0)
    delta = target - aligned_prev
    turnover = float(delta.abs().sum())
    if turnover <= cap:
        return target
    scale = cap / max(turnover, 1e-9)
    return aligned_prev + delta * scale


def _try_cvxpy_long_only(
    longs: pd.DataFrame,
    previous_weights: pd.Series | None,
    sector_lookup: dict[str, str],
    config: V7TargetWeightsConfig,
) -> tuple[pd.Series | None, str]:
    if config.optimizer_backend == "deterministic" or config.long_short:
        return None, "deterministic_requested"
    try:
        import cvxpy as cp  # type: ignore
    except Exception:
        if config.optimizer_backend == "cvxpy":
            raise RuntimeError("optimizer_backend='cvxpy' requires cvxpy; install quantagent[optimization]")
        return None, "cvxpy_unavailable_deterministic_fallback"
    symbols = longs["symbol"].astype(str).tolist()
    if not symbols:
        return None, "empty_universe"
    alpha = longs["prediction"].astype(float).to_numpy()
    caps = longs.set_index("symbol")["liquidity_cap"].reindex(symbols).fillna(config.max_weight_per_name).to_numpy(dtype=float)
    w = cp.Variable(len(symbols))
    prev = (
        previous_weights.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
        if previous_weights is not None
        else np.zeros(len(symbols), dtype=float)
    )
    if config.objective == "min_turnover":
        objective = cp.Minimize(cp.norm1(w - prev) - 1e-4 * (alpha @ w))
    elif config.objective == "max_information_ratio_proxy":
        scaled_alpha = alpha / (np.nanstd(alpha) + 1e-9)
        objective = cp.Maximize(scaled_alpha @ w - float(config.cost_bps) / 10_000.0 * cp.norm1(w - prev))
    else:
        objective = cp.Maximize(alpha @ w - float(config.cost_bps) / 10_000.0 * cp.norm1(w - prev))
    constraints = [w >= 0, w <= caps, cp.sum(w) <= 1.0]
    if config.cash_floor > 0:
        constraints.append(cp.sum(w) <= 1.0 - float(config.cash_floor))
    if sector_lookup:
        sector_series = pd.Series(symbols, index=symbols).map(sector_lookup).fillna("__unknown__")
        for sector in sorted(set(sector_series)):
            idx = [i for i, symbol in enumerate(symbols) if sector_series.loc[symbol] == sector]
            constraints.append(cp.sum(w[idx]) <= config.max_sector_weight)
    if previous_weights is not None and config.max_turnover > 0:
        constraints.append(cp.norm1(w - prev) <= config.max_turnover)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(verbose=False)
    if w.value is None:
        if config.optimizer_backend == "cvxpy":
            raise RuntimeError(f"cvxpy optimizer failed: {problem.status}")
        return None, f"cvxpy_failed:{problem.status}"
    weights = pd.Series(np.asarray(w.value, dtype=float), index=pd.Index(symbols, name="symbol")).clip(lower=0.0)
    gross = float(weights.sum())
    if gross <= 0:
        return None, "cvxpy_zero_solution"
    return weights / gross, f"objective={config.objective};status={problem.status}"


def write_v7_target_weights(result: V7TargetWeightsResult, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.target_weights
    if output_path.suffix == ".parquet":
        try:
            frame.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            output_path = output_path.with_suffix(".csv")
    frame.to_csv(output_path, index=False)
    return output_path


__all__ = [
    "V7TargetWeightsConfig",
    "V7TargetWeightsResult",
    "build_v7_target_weights",
    "write_v7_target_weights",
]
