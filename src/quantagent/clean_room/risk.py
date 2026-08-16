"""Risk controls that either fire or say they could not.

Every limit here returns the book it produced AND the reason each name was
removed. A control that silently drops a name is indistinguishable from one that
was never configured, which is how the legacy `require_amount_above` floor sat
disabled in production without anyone noticing.

The house rule this package inherits: an unmeasurable limit fails CLOSED and is
recorded as `*_unavailable`, never waved through and never reported as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    top_k: int = 50
    max_name_weight: float = 0.05
    max_gross: float = 1.0
    #: Minimum 20-day average turnover in CNY for a name to be tradable.
    min_adv_cny: float = 50_000_000.0
    #: Cap position at this share of the name's ADV, so the book cannot assume
    #: it is the entire market in an illiquid name.
    max_adv_participation: float = 0.05
    #: De-risk when the strategy's own drawdown breaches this.
    drawdown_derisk_at: float = 0.10
    drawdown_derisk_to: float = 0.50
    #: Hard stop: flatten entirely beyond this.
    drawdown_kill_at: float = 0.20


@dataclass
class RiskDecision:
    weights: pd.Series
    rejects: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    gross: float = 0.0
    derisk_factor: float = 1.0


def apply_risk_controls(
    scores: pd.Series,
    day: pd.DataFrame,
    config: RiskConfig,
    *,
    current_drawdown: float,
    nav: float,
) -> RiskDecision:
    """Turn a cross-sectional score into a risk-controlled long-only book.

    ``day`` must carry, per symbol: ``adv20`` (CNY), ``tradable`` (bool).
    """
    rejects: dict[str, str] = {}
    notes: list[str] = []

    frame = day.set_index("symbol")
    ranked = scores.dropna().sort_values(ascending=False)

    # 1. tradability -- fail closed. A missing flag is NOT "tradable".
    if "tradable" not in frame.columns:
        notes.append("tradability_unavailable_all_names_blocked")
        return RiskDecision(pd.Series(dtype=float), rejects, notes, 0.0, 0.0)
    tradable = frame["tradable"].reindex(ranked.index)
    unknown = tradable.isna()
    for sym in ranked.index[unknown]:
        rejects[sym] = "tradability_unknown"
    for sym in ranked.index[(~unknown) & (~tradable.fillna(False).astype(bool))]:
        rejects[sym] = "untradable"
    ranked = ranked[(~unknown) & tradable.fillna(False).astype(bool)]

    # 2. liquidity floor -- an unmeasurable ADV is not a passing ADV.
    adv = frame["adv20"].reindex(ranked.index)
    adv_missing = adv.isna()
    for sym in ranked.index[adv_missing]:
        rejects[sym] = "adv_unavailable"
    below = (~adv_missing) & (adv < config.min_adv_cny)
    for sym in ranked.index[below]:
        rejects[sym] = "below_min_adv"
    ranked = ranked[(~adv_missing) & (~below)]

    if ranked.empty:
        notes.append("no_eligible_names")
        return RiskDecision(pd.Series(dtype=float), rejects, notes, 0.0, 0.0)

    # 3. select top-k, equal weight, then cap per name
    chosen = ranked.head(config.top_k)
    weight = min(config.max_name_weight, config.max_gross / max(len(chosen), 1))
    weights = pd.Series(weight, index=chosen.index, dtype=float)

    # 4. capacity: never take more of a name than max_adv_participation
    adv_sel = frame["adv20"].reindex(weights.index)
    cap = (adv_sel * config.max_adv_participation / max(nav, 1.0)).clip(upper=1.0)
    capped = weights > cap
    if capped.any():
        notes.append(f"capacity_capped:{int(capped.sum())}")
    weights = pd.concat([weights, cap], axis=1).min(axis=1)

    # 5. drawdown response -- scale, then kill.
    derisk = 1.0
    if current_drawdown <= -abs(config.drawdown_kill_at):
        notes.append("drawdown_kill_switch")
        return RiskDecision(pd.Series(dtype=float), rejects, notes, 0.0, 0.0)
    if current_drawdown <= -abs(config.drawdown_derisk_at):
        derisk = config.drawdown_derisk_to
        notes.append(f"drawdown_derisk:{derisk}")
    weights = weights * derisk

    gross = float(weights.sum())
    if gross > config.max_gross:
        weights = weights * (config.max_gross / gross)
        gross = float(weights.sum())

    return RiskDecision(weights, rejects, notes, gross, derisk)


def daily_risk_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach ADV and a tradability flag derived only from observable bars.

    `tradable` is NaN -- not False, and not True -- when the inputs needed to
    judge it are absent, so the caller's fail-closed branch can tell "known
    untradable" apart from "never measured".
    """
    frame = panel.sort_values(["symbol", "trade_date"]).copy()
    g = frame.groupby("symbol", sort=False)
    frame["adv20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    prev_close = g["close"].shift(1)

    limit = np.where(frame["symbol"].str.contains(r"sh68|sz30", regex=True), 0.20, 0.10)
    move = frame["close"] / prev_close - 1.0
    at_limit_up = move >= (limit - 0.005)
    at_limit_down = move <= -(limit - 0.005)
    no_trade = frame["volume"].fillna(0) <= 0

    tradable = ~(at_limit_up | at_limit_down | no_trade)
    # Where prev_close is unknown the limit test is undefined -> unknown, not True.
    tradable = tradable.where(prev_close.notna() & frame["volume"].notna())
    frame["tradable"] = tradable
    return frame
