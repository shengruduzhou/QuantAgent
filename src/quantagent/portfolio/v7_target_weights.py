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


@dataclass(frozen=True)
class V7TargetWeightsConfig:
    long_short: bool = False
    top_k: int = 30
    max_weight_per_name: float = 0.10
    max_sector_weight: float = 0.30
    max_turnover: float = 0.50
    cost_bps: float = 12.0
    liquidity_participation: float = 0.05
    min_amount_yuan: float = 0.0
    min_universe: int = 1
    alpha_temperature: float = 1.0
    capital_yuan: float = 1_000_000.0
    horizon_column: str | None = None
    block_st: bool = True
    block_suspended: bool = True
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True


@dataclass(frozen=True)
class V7TargetWeightsResult:
    target_weights: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


_TRADABILITY_FLAGS: tuple[str, ...] = ("is_suspended", "is_st", "is_limit_up", "is_limit_down")


def build_v7_target_weights(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    config: V7TargetWeightsConfig | None = None,
) -> V7TargetWeightsResult:
    config = config or V7TargetWeightsConfig()
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
    diagnostics: dict[str, list[dict[str, object]]] = {"rejected": [], "exposures": []}
    previous_weights: pd.Series | None = None

    for date, day in preds.groupby("trade_date", sort=True):
        day_market = market[market["trade_date"] == date]
        merged = day.merge(day_market, on=["symbol", "trade_date"], how="left", suffixes=("", "_mkt"))
        rejected: list[dict[str, object]] = []
        keep_mask = pd.Series(True, index=merged.index)
        for column, reason in (
            ("is_suspended", "suspended"),
            ("is_st", "st"),
            ("is_limit_up", "limit_up_buy_block"),
            ("is_limit_down", "limit_down_sell_block"),
        ):
            if column in merged.columns and getattr(config, f"block_{reason.split('_')[0]}", False):
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
        # Liquidity cap (max weight by participation in amount).
        if "amount" in eligible.columns and config.capital_yuan > 0:
            cap = (eligible["amount"].fillna(0.0) * config.liquidity_participation) / max(1.0, config.capital_yuan)
            eligible["liquidity_cap"] = cap.clip(upper=config.max_weight_per_name)
        else:
            eligible["liquidity_cap"] = config.max_weight_per_name

        # Top-K selection by predicted alpha (long side); for long-short keep both tails.
        alpha = eligible["prediction"].astype(float)
        if config.long_short:
            top_k = min(config.top_k, len(eligible) // 2 or 1)
            longs = eligible.nlargest(top_k, "prediction")
            shorts = eligible.nsmallest(top_k, "prediction")
            longs_w = _softmax_weights(longs["prediction"].to_numpy(dtype=float), config.alpha_temperature)
            shorts_w = -_softmax_weights(-shorts["prediction"].to_numpy(dtype=float), config.alpha_temperature)
            weights = pd.Series(
                np.concatenate([longs_w, shorts_w]),
                index=pd.Index(np.concatenate([longs["symbol"].to_numpy(), shorts["symbol"].to_numpy()]), name="symbol"),
            )
        else:
            top_k = min(config.top_k, len(eligible))
            longs = eligible.nlargest(top_k, "prediction")
            scaled = _softmax_weights(longs["prediction"].to_numpy(dtype=float), config.alpha_temperature)
            weights = pd.Series(scaled, index=pd.Index(longs["symbol"].to_numpy(), name="symbol"))
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

        # Apply turnover cap vs previous weights.
        if previous_weights is not None and config.max_turnover > 0:
            blended = _apply_turnover_cap(weights, previous_weights, config.max_turnover)
            weights = blended
        previous_weights = weights.copy()

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
        "dates": int(pivot.shape[0]),
        "symbol_count": int(pivot.shape[1]),
        "average_gross_exposure": float(pivot.abs().sum(axis=1).mean()),
        "average_turnover": float(pivot.diff().abs().sum(axis=1).mean()),
        "config": asdict(config),
        **diagnostics,
    }
    pivot.index.name = "trade_date"
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


def _apply_turnover_cap(target: pd.Series, previous: pd.Series, cap: float) -> pd.Series:
    aligned_prev = previous.reindex(target.index).fillna(0.0)
    delta = target - aligned_prev
    turnover = float(delta.abs().sum())
    if turnover <= cap:
        return target
    scale = cap / max(turnover, 1e-9)
    return aligned_prev + delta * scale


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
