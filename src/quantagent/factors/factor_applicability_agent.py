from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.factors.evaluation import capacity_proxy, forward_return_labels, information_coefficient, quantile_group_backtest
from quantagent.v7.schemas import FactorApplicability, MarketRegime, ThematicUniverseMember


@dataclass(frozen=True)
class FactorApplicabilityConfig:
    horizons: tuple[int, ...] = (1, 5, 20, 60, 120, 126)
    min_rank_icir: float = 0.05
    min_hit_rate: float = 0.50
    max_crowding_score: float = 0.80
    min_sample_count: int = 30
    min_symbol_count: int = 5
    min_date_count: int = 8
    walk_forward_splits: int = 3
    min_walk_forward_pass_rate: float = 0.50
    embargo_days: int = 5
    amount_column: str = "amount"
    price_column: str = "close"


def validate_factor_applicability(
    factor_frame: pd.DataFrame,
    factor_columns: list[str],
    universe_members: list[ThematicUniverseMember],
    market_regime: MarketRegime,
    config: FactorApplicabilityConfig | None = None,
) -> list[FactorApplicability]:
    """Validate factors by theme, sector, universe bucket, regime, and horizon."""
    config = config or FactorApplicabilityConfig()
    if factor_frame.empty or not factor_columns:
        return []
    data = forward_return_labels(factor_frame, horizons=config.horizons, price_column=config.price_column)
    member_frame = _member_frame(universe_members)
    if not member_frame.empty:
        data = data.merge(member_frame, on="symbol", how="left", suffixes=("", "_member"))
        for column in ("theme", "sector", "chain_node", "watchlist_status"):
            member_column = f"{column}_member"
            if member_column in data.columns:
                if column in data.columns:
                    data[column] = data[column].combine_first(data[member_column])
                else:
                    data[column] = data[member_column]
                data = data.drop(columns=[member_column])
    results: list[FactorApplicability] = []
    for factor in factor_columns:
        if factor not in data.columns:
            continue
        for horizon in config.horizons:
            return_column = f"forward_return_{horizon}d"
            if return_column not in data.columns:
                continue
            required_columns = ["trade_date", "symbol", factor, return_column, "theme", "sector", "chain_node", "watchlist_status"]
            required_columns = [column for column in required_columns if column in data.columns]
            if config.amount_column in data.columns:
                required_columns.append(config.amount_column)
            clean = data[required_columns].replace([np.inf, -np.inf], np.nan)
            clean = clean.dropna(subset=[factor, return_column])
            if clean.empty:
                continue
            global_metrics = _factor_metrics(clean, factor, return_column, config)
            sample_count, symbol_count, date_count = _validation_counts(clean)
            walk_forward_pass_rate = _walk_forward_pass_rate(clean, factor, return_column, config)
            theme_scores = _slice_scores(clean, "theme", factor, return_column, config)
            sector_scores = _slice_scores(clean, "sector", factor, return_column, config)
            universe_scores = _slice_scores(clean, "watchlist_status", factor, return_column, config)
            applicable_themes = tuple(name for name, metrics in theme_scores.items() if _is_applicable(metrics, config))
            applicable_sectors = tuple(name for name, metrics in sector_scores.items() if _is_applicable(metrics, config))
            applicable_universe = tuple(name for name, metrics in universe_scores.items() if _is_applicable(metrics, config))
            sample_ok = sample_count >= config.min_sample_count and symbol_count >= config.min_symbol_count and date_count >= config.min_date_count
            walk_ok = walk_forward_pass_rate >= config.min_walk_forward_pass_rate
            stage = _lifecycle_stage(
                global_metrics,
                bool(applicable_themes or applicable_sectors or applicable_universe),
                sample_ok,
                walk_ok,
                config,
            )
            results.append(
                FactorApplicability(
                    factor_name=factor,
                    factor_category=_factor_category(factor),
                    applicable_universe=applicable_universe,
                    applicable_sector=applicable_sectors,
                    applicable_theme=applicable_themes,
                    applicable_market_regime=(market_regime,) if stage != "invalidated" else (),
                    horizon_days=horizon,
                    decay_half_life=max(1.0, horizon / 2.0),
                    rank_ic=global_metrics["rank_ic"],
                    rank_icir=global_metrics["rank_icir"],
                    hit_rate=global_metrics["hit_rate"],
                    turnover=global_metrics["turnover"],
                    capacity=global_metrics["capacity"],
                    crowding_score=global_metrics["crowding_score"],
                    factor_lifecycle_stage=stage,
                    last_validated_at=str(pd.Timestamp(clean["trade_date"].max()).date()),
                    invalidation_condition=_invalidation_condition(factor, horizon),
                    validation_sample_count=sample_count,
                    validation_symbol_count=symbol_count,
                    validation_date_count=date_count,
                    walk_forward_pass_rate=walk_forward_pass_rate,
                    validation_method=f"walk_forward_pit_embargo_{config.embargo_days}d",
                )
            )
    return results


def _member_frame(members: list[ThematicUniverseMember]) -> pd.DataFrame:
    if not members:
        return pd.DataFrame(columns=["symbol", "theme", "sector", "chain_node", "watchlist_status"])
    return pd.DataFrame(
        [
            {
                "symbol": member.symbol,
                "theme": member.theme,
                # Sector and chain_node are different concepts. Earlier
                # versions overloaded ``sector`` with ``chain_node`` which
                # broke the per-sector factor applicability slice — every
                # company looked like it belonged to its chain node bucket.
                "sector": member.sector or "unknown",
                "chain_node": member.chain_node,
                "watchlist_status": member.watchlist_status.value,
            }
            for member in members
        ]
    ).drop_duplicates("symbol")


def _factor_metrics(frame: pd.DataFrame, factor: str, return_column: str, config: FactorApplicabilityConfig) -> dict[str, float]:
    ic = information_coefficient(frame, factor, return_column)
    groups = quantile_group_backtest(frame, factor, return_column, quantiles=5)
    cap = capacity_proxy(frame, factor, amount_column=config.amount_column) if config.amount_column in frame.columns else None
    hit_rate = _hit_rate(frame, factor, return_column)
    crowding = _crowding_score(frame, factor)
    return {
        "rank_ic": _finite(ic.summary.mean_rank_ic),
        "rank_icir": _finite(ic.summary.rank_icir),
        "hit_rate": hit_rate,
        "turnover": _finite(float(groups.turnover.mean()) if not groups.turnover.empty else np.nan),
        "capacity": _finite(cap.capacity_rmb if cap is not None else np.nan),
        "crowding_score": crowding,
    }


def _validation_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    return (
        int(len(frame)),
        int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        int(pd.to_datetime(frame["trade_date"]).nunique()) if "trade_date" in frame.columns else 0,
    )


def _walk_forward_pass_rate(frame: pd.DataFrame, factor: str, return_column: str, config: FactorApplicabilityConfig) -> float:
    if "trade_date" not in frame.columns:
        return 0.0
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    unique_dates = sorted(data["trade_date"].dropna().unique())
    min_dates = config.min_date_count + config.embargo_days + 1
    if len(unique_dates) < min_dates:
        return 0.0
    splits = max(1, min(config.walk_forward_splits, len(unique_dates) - config.embargo_days - 1))
    pass_flags: list[bool] = []
    for split_index in range(1, splits + 1):
        cut = int(len(unique_dates) * split_index / (splits + 1))
        train_dates = unique_dates[:cut]
        test_dates = unique_dates[min(cut + config.embargo_days, len(unique_dates) - 1) :]
        if len(train_dates) < config.min_date_count or not test_dates:
            continue
        train = data[data["trade_date"].isin(train_dates)]
        test = data[data["trade_date"].isin(test_dates)]
        train_metrics = _factor_metrics(train, factor, return_column, config)
        test_metrics = _factor_metrics(test, factor, return_column, config)
        pass_flags.append(_is_applicable(train_metrics, config) and _is_applicable(test_metrics, config))
    return float(np.mean(pass_flags)) if pass_flags else 0.0


def _slice_scores(frame: pd.DataFrame, column: str, factor: str, return_column: str, config: FactorApplicabilityConfig) -> dict[str, dict[str, float]]:
    if column not in frame.columns:
        return {}
    scores: dict[str, dict[str, float]] = {}
    for value, group in frame.dropna(subset=[column]).groupby(column, sort=True):
        if len(group) < 6 or group["symbol"].nunique() < 3:
            continue
        scores[str(value)] = _factor_metrics(group, factor, return_column, config)
    return scores


def _is_applicable(metrics: dict[str, float], config: FactorApplicabilityConfig) -> bool:
    return (
        metrics["rank_icir"] >= config.min_rank_icir
        and metrics["hit_rate"] >= config.min_hit_rate
        and metrics["crowding_score"] <= config.max_crowding_score
    )


def _lifecycle_stage(
    metrics: dict[str, float],
    has_applicable_slice: bool,
    sample_ok: bool,
    walk_ok: bool,
    config: FactorApplicabilityConfig,
) -> str:
    if not sample_ok:
        return "insufficient_sample"
    if not walk_ok:
        return "validation"
    if metrics["crowding_score"] > config.max_crowding_score:
        return "crowded"
    if metrics["rank_icir"] < -0.05 or metrics["hit_rate"] < 0.45:
        return "invalidated"
    if has_applicable_slice and metrics["rank_icir"] >= config.min_rank_icir and metrics["hit_rate"] >= config.min_hit_rate:
        return "production"
    if metrics["rank_icir"] >= 0.0:
        return "validation"
    return "decaying"


def _hit_rate(frame: pd.DataFrame, factor: str, return_column: str) -> float:
    values = []
    for _, group in frame.groupby("trade_date", sort=False):
        clean = group[[factor, return_column]].dropna()
        if len(clean) < 3:
            continue
        top = clean[clean[factor] >= clean[factor].quantile(0.8)][return_column].mean()
        bottom = clean[clean[factor] <= clean[factor].quantile(0.2)][return_column].mean()
        if np.isfinite(top) and np.isfinite(bottom):
            values.append(float(top > bottom))
    return float(np.mean(values)) if values else 0.0


def _crowding_score(frame: pd.DataFrame, factor: str) -> float:
    clean = frame[factor].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 1.0
    concentration = clean.rank(pct=True).sub(0.5).abs().mean() * 2.0
    return float(np.clip(concentration, 0.0, 1.0))


def _factor_category(factor: str) -> str:
    text = factor.lower()
    if "momentum" in text or "ret_" in text:
        return "momentum"
    if "value" in text or "valuation" in text or "pe" in text or "pb" in text:
        return "value"
    if "quality" in text or "roe" in text or "cash" in text:
        return "quality"
    if "flow" in text or "amount" in text:
        return "fund_flow"
    if "sentiment" in text or "news" in text:
        return "sentiment"
    if "theme" in text or "policy" in text:
        return "policy"
    if "vol" in text:
        return "volatility"
    return "technical_timing"


def _invalidation_condition(factor: str, horizon: int) -> str:
    return f"{factor} invalid when sliced RankICIR<0 or hit_rate<0.45 for horizon={horizon}d over latest walk-forward window"


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
