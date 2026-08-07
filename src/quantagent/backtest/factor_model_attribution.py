"""Alpha net of the risk exposures that already have names.

``quantagent.backtest.beta_decomposition`` regresses a strategy on one
benchmark and calls the intercept Jensen alpha. That is the 1964 CAPM test, and
Fama & French (1992) is the paper that says it is not enough: over 1963-1990
"the relation between market β and average return is flat, even when β is the
only explanatory variable", while size and book-to-market between them capture
the cross-section. Carhart (1997) then showed that what looked like persistent
manager skill was one-year momentum — common factors and expenses "almost
completely explain persistence in equity mutual funds' mean and risk-adjusted
returns".

The consequence for this repository is concrete rather than academic. Its books
are concentrated (top-30 style), tilted small by construction, and built on a
factor library whose strongest members are momentum and reversal. A positive
CAPM alpha on such a book is the expected result whether or not the model has
found anything: it is what a small-cap momentum tilt produces against a
broad-index beta. Only the intercept net of SMB, HML and UMD carries the claim
that something new was found.

Three nested levels are reported, each with a Newey-West t-statistic because
daily strategy returns are autocorrelated and an OLS standard error would
overstate significance:

===============  =========================================================
``capm``         ``r - rf = α + β·MKT``
``ff3``          ``+ s·SMB + h·HML``            (Fama & French 1992/1993)
``carhart4``     ``+ m·UMD``                    (Carhart 1997)
===============  =========================================================

Missing factors are reported as ``unavailable``, never as zero. A-share market
equity needs shares outstanding, and this repository's security master carries
only a current snapshot of ``total_shares``/``float_shares``; multiplying that
by a 2019 price is not the 2019 market cap. Callers that have a genuine
point-in-time share count say so through ``share_count_status``; callers that do
not get an SMB marked approximate, and a Carhart alpha that says on its face
which of its controls is soft. Substituting a liquidity proxy for size and
calling the result Carhart alpha would defeat the entire purpose of computing
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import newey_west_t_stat

#: A-share trading days per year, matching ``beta_decomposition.ANN``.
ANN = 244

FactorStatus = Literal["constructed", "approximate", "unavailable"]
ShareCountStatus = Literal["point_in_time", "current_snapshot", "absent"]

MARKET = "MKT"
SIZE = "SMB"
VALUE = "HML"
MOMENTUM = "UMD"


@dataclass(frozen=True)
class StyleFactorSet:
    """Daily style-factor returns plus how each one came to exist."""

    returns: pd.DataFrame
    status: dict[str, FactorStatus]
    notes: dict[str, str] = field(default_factory=dict)
    share_count_status: ShareCountStatus = "absent"

    def available(self, names: Sequence[str]) -> bool:
        return all(
            self.status.get(name) in {"constructed", "approximate"} and name in self.returns.columns
            for name in names
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": dict(self.status),
            "notes": dict(self.notes),
            "shareCountStatus": self.share_count_status,
            "observations": int(len(self.returns)),
            "start": str(self.returns.index.min().date()) if len(self.returns) else None,
            "end": str(self.returns.index.max().date()) if len(self.returns) else None,
        }


@dataclass(frozen=True)
class AttributionLevel:
    """One nested regression: intercept, loadings and how sure we are."""

    name: str
    #: ``measured`` when the regression ran; ``unavailable`` when a required
    #: factor could not be built. Never silently degrades to a lower level.
    status: Literal["measured", "unavailable"]
    alpha_annual: float | None = None
    alpha_t_stat: float | None = None
    loadings: dict[str, float] = field(default_factory=dict)
    r_squared: float | None = None
    observations: int = 0
    missing_factors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "alphaAnnual": None if self.alpha_annual is None else round(self.alpha_annual, 6),
            "alphaTStat": None if self.alpha_t_stat is None else round(self.alpha_t_stat, 4),
            "loadings": {key: round(float(value), 4) for key, value in self.loadings.items()},
            "rSquared": None if self.r_squared is None else round(self.r_squared, 4),
            "observations": int(self.observations),
            "missingFactors": list(self.missing_factors),
        }


@dataclass(frozen=True)
class AttributionReport:
    """CAPM / FF3 / Carhart side by side, with the honest verdict."""

    levels: dict[str, AttributionLevel]
    factor_set: StyleFactorSet
    #: The strictest level that actually ran. Claims about "new alpha" must cite
    #: this one, not whichever level looks best.
    strictest_measured: str
    #: True only when the strictest measured level has a positive alpha whose
    #: Newey-West t-statistic clears ``t_threshold``.
    survives_style_controls: bool
    t_threshold: float

    def alpha_decay(self) -> dict[str, float | None]:
        """How much of the raw CAPM alpha each added control absorbs."""
        capm = self.levels.get("capm")
        base = capm.alpha_annual if capm and capm.alpha_annual is not None else None
        out: dict[str, float | None] = {}
        for name in ("ff3", "carhart4"):
            level = self.levels.get(name)
            if base is None or level is None or level.alpha_annual is None:
                out[name] = None
            else:
                out[name] = round(float(level.alpha_annual - base), 6)
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "levels": {name: level.as_dict() for name, level in self.levels.items()},
            "factorSet": self.factor_set.as_dict(),
            "strictestMeasured": self.strictest_measured,
            "survivesStyleControls": bool(self.survives_style_controls),
            "tThreshold": self.t_threshold,
            "alphaAbsorbedVsCapm": self.alpha_decay(),
        }


def _annualise_daily_alpha(alpha_daily: float) -> float:
    if not np.isfinite(alpha_daily) or alpha_daily <= -1.0:
        return float("nan")
    return float((1.0 + alpha_daily) ** ANN - 1.0)


def _long_short_return(
    frame: pd.DataFrame,
    *,
    characteristic: str,
    return_column: str,
    quantile: float,
    high_minus_low: bool,
    lag_days: int = 1,
) -> pd.Series:
    """Daily return of a top-minus-bottom quantile spread on ``characteristic``.

    A simplification of Fama & French's 2x3 double sort, chosen deliberately:
    the double sort needs a stable size breakpoint that this data cannot supply
    point-in-time, and a single-sort spread is the weaker but honest version of
    the same control. It is documented as such rather than presented as FF's
    construction.

    ``lag_days`` is not optional in spirit. Price-based characteristics —
    book-to-market and market equity both — contain ``close(t)``, while
    ``return_1d(t)`` is the return *into* ``close(t)``. Sorting and paying on the
    same row therefore puts today's losers in the high-B/M leg by construction
    and manufactures an enormous negative value premium out of nothing. Measured
    on this repository's own panel, the unlagged sort produced an HML of
    -30.8%/yr (Sharpe -2.8) over 2021-2026 and -43.3%/yr on the liquid subset,
    negative in every single calendar year — a stability that should be read as
    a mechanical artifact rather than a risk premium. Fama & French sort on
    characteristics known before the return window opens, and so does this.
    """
    work = frame.sort_values(["symbol", "trade_date"])
    if lag_days > 0:
        lagged_name = f"{characteristic}__lag{lag_days}"
        work = work.assign(
            **{
                lagged_name: work.groupby("symbol", sort=False)[characteristic].shift(
                    lag_days
                )
            }
        )
        characteristic = lagged_name
    values: dict[pd.Timestamp, float] = {}
    for date, group in work.groupby("trade_date", sort=True):
        usable = group[[characteristic, return_column]].dropna()
        if len(usable) < 20:
            continue
        ranks = usable[characteristic].rank(pct=True)
        high = usable.loc[ranks >= 1.0 - quantile, return_column]
        low = usable.loc[ranks <= quantile, return_column]
        if high.empty or low.empty:
            continue
        spread = float(high.mean() - low.mean())
        values[pd.Timestamp(date)] = spread if high_minus_low else -spread
    return pd.Series(values, dtype=float).sort_index()


def build_ashare_style_factors(
    panel: pd.DataFrame,
    *,
    return_column: str = "return_1d",
    book_to_market_column: str | None = "book_yield",
    market_equity_column: str | None = None,
    shares_outstanding: Mapping[str, float] | None = None,
    share_count_status: ShareCountStatus = "absent",
    momentum_lookback: int = 244,
    momentum_skip: int = 20,
    quantile: float = 0.30,
    risk_free_daily: float = 0.0,
) -> StyleFactorSet:
    """Build MKT / SMB / HML / UMD daily returns from a market panel.

    Parameters
    ----------
    panel
        Long panel with ``trade_date``, ``symbol``, ``close`` and
        ``return_column``. Extra columns are used when present.
    market_equity_column
        A genuine point-in-time market-equity column, if the caller has one.
        Takes precedence over ``shares_outstanding``.
    shares_outstanding
        ``symbol -> share count``. Combined with ``close`` this yields a market
        equity series. ``share_count_status`` declares whether that count is
        point-in-time; when it is ``current_snapshot`` the resulting SMB is
        marked ``approximate`` and every report built on it says so, because a
        present-day share count applied to a historical price is not that day's
        market cap.
    """
    required = {"trade_date", "symbol", return_column}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel is missing required columns: {sorted(missing)}")

    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["symbol"] = work["symbol"].astype(str)
    work = work.dropna(subset=["trade_date", "symbol"])
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")

    status: dict[str, FactorStatus] = {}
    notes: dict[str, str] = {}
    columns: dict[str, pd.Series] = {}

    # --- MKT: equal-weight universe excess return -------------------------
    market = work.groupby("trade_date", sort=True)[return_column].mean() - risk_free_daily
    columns[MARKET] = market
    status[MARKET] = "constructed"
    notes[MARKET] = "equal-weight universe return minus the supplied risk-free rate"

    # --- SMB: small minus big --------------------------------------------
    equity: pd.Series | None = None
    resolved_share_status: ShareCountStatus = share_count_status
    if market_equity_column and market_equity_column in work.columns:
        equity = pd.to_numeric(work[market_equity_column], errors="coerce")
        status[SIZE] = "constructed"
        notes[SIZE] = f"market equity from column '{market_equity_column}'"
        resolved_share_status = "point_in_time"
    elif shares_outstanding and "close" in work.columns:
        counts = work["symbol"].map({str(k): float(v) for k, v in shares_outstanding.items()})
        equity = pd.to_numeric(work["close"], errors="coerce") * counts
        if share_count_status == "point_in_time":
            status[SIZE] = "constructed"
            notes[SIZE] = "market equity = close x point-in-time shares outstanding"
        else:
            status[SIZE] = "approximate"
            notes[SIZE] = (
                "market equity = close x a CURRENT-SNAPSHOT share count; issuance, "
                "buybacks and splits between the trade date and today are not "
                "reflected, and this panel's close is not consistently adjusted, so "
                "SMB loadings are indicative rather than exact"
            )
    else:
        status[SIZE] = "unavailable"
        notes[SIZE] = (
            "no market-equity column and no share count supplied; size cannot be "
            "constructed from prices alone and is NOT proxied by turnover"
        )

    if equity is not None:
        sized = work.assign(_me=equity)
        columns[SIZE] = _long_short_return(
            sized,
            characteristic="_me",
            return_column=return_column,
            quantile=quantile,
            high_minus_low=False,  # small minus big
        )

    # --- HML: high minus low book-to-market -------------------------------
    bm_column: str | None = None
    if book_to_market_column and book_to_market_column in work.columns:
        bm_column = book_to_market_column
    elif "pb" in work.columns:
        pb = pd.to_numeric(work["pb"], errors="coerce")
        work["_bm_from_pb"] = np.where(pb > 0, 1.0 / pb, np.nan)
        bm_column = "_bm_from_pb"
        notes[VALUE] = "book-to-market derived as 1/pb"
    if bm_column is not None:
        columns[VALUE] = _long_short_return(
            work,
            characteristic=bm_column,
            return_column=return_column,
            quantile=quantile,
            high_minus_low=True,
        )
        status[VALUE] = "constructed"
        notes.setdefault(VALUE, f"book-to-market from column '{bm_column}'")
    else:
        status[VALUE] = "unavailable"
        notes[VALUE] = "no book-to-market or price-to-book column in the panel"

    # --- UMD: momentum, skipping the most recent month --------------------
    # Built by compounding ``return_column``, never by taking a ratio of raw
    # closes. On this repository's own gold panel the two disagree: the stored
    # ``return_1d`` respects the A-share ±10% price limit, while
    # ``close.pct_change()`` reaches +85% because the close series carries
    # unadjusted split and rights-issue steps. A 244-day ratio of such closes
    # would read a 2-for-1 split as -50% momentum and sort the stock into the
    # loser leg on a corporate action.
    ordered = work.sort_values(["symbol", "trade_date"]).copy()
    grouped_returns = ordered.groupby("symbol", sort=False)[return_column]
    log_wealth = grouped_returns.transform(
        lambda series: np.log1p(series.fillna(0.0)).cumsum()
    )
    ordered["_log_wealth"] = log_wealth
    shifted = ordered.groupby("symbol", sort=False)["_log_wealth"]
    lagged = shifted.shift(momentum_skip)
    base = shifted.shift(momentum_lookback)
    ordered["_umd"] = np.expm1(lagged - base)
    columns[MOMENTUM] = _long_short_return(
        ordered,
        characteristic="_umd",
        return_column=return_column,
        quantile=quantile,
        high_minus_low=True,
    )
    status[MOMENTUM] = "constructed"
    notes[MOMENTUM] = (
        f"{momentum_lookback}-day compounded return skipping the most recent "
        f"{momentum_skip} days, top-minus-bottom {quantile:.0%}, built from "
        f"'{return_column}' rather than a raw close ratio"
    )

    frame = pd.DataFrame(columns).sort_index()
    return StyleFactorSet(
        returns=frame,
        status=status,
        notes=notes,
        share_count_status=resolved_share_status,
    )


def _regress(
    strategy_excess: pd.Series,
    factors: pd.DataFrame,
) -> tuple[float, float, dict[str, float], float, int]:
    """OLS with a Newey-West t-stat on the intercept.

    Returns ``(alpha_daily, alpha_t, loadings, r2, n)``.
    """
    aligned = pd.concat([strategy_excess.rename("_y"), factors], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan"), float("nan"), {}, float("nan"), len(aligned)
    y = aligned["_y"].to_numpy(dtype=float)
    x = aligned.drop(columns=["_y"])
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return float("nan"), float("nan"), {}, float("nan"), len(aligned)
    fitted = design @ beta
    residual = y - fitted
    ss_res = float((residual**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    alpha_daily = float(beta[0])
    # The intercept's t-statistic is the t-statistic of the mean residual once
    # the factor exposures are removed, so a HAC estimator on that series is
    # the right standard error for autocorrelated daily strategy returns.
    alpha_t = newey_west_t_stat(pd.Series(residual + alpha_daily, index=aligned.index))
    loadings = {name: float(value) for name, value in zip(x.columns, beta[1:])}
    return alpha_daily, float(alpha_t), loadings, r2, len(aligned)


def attribute_strategy_returns(
    strategy_daily_returns: pd.Series,
    factor_set: StyleFactorSet,
    *,
    risk_free_daily: float = 0.0,
    t_threshold: float = 2.0,
) -> AttributionReport:
    """Run the CAPM / FF3 / Carhart ladder and report where alpha survives.

    ``t_threshold`` defaults to 2.0, the conventional bar. Harvey, Liu & Zhu
    (2016) argue a newly claimed factor should clear roughly 3.0 once the
    multiple testing behind the whole factor zoo is accounted for; callers
    promoting a *new* signal rather than measuring an existing book should pass
    that instead. It is a parameter and not a constant precisely so the choice
    is recorded per call rather than assumed.
    """
    strategy = pd.to_numeric(pd.Series(strategy_daily_returns), errors="coerce").dropna()
    strategy.index = pd.to_datetime(strategy.index, errors="coerce")
    strategy = strategy[~strategy.index.isna()].sort_index()
    excess = strategy - risk_free_daily

    ladder: list[tuple[str, tuple[str, ...]]] = [
        ("capm", (MARKET,)),
        ("ff3", (MARKET, SIZE, VALUE)),
        ("carhart4", (MARKET, SIZE, VALUE, MOMENTUM)),
    ]
    levels: dict[str, AttributionLevel] = {}
    strictest = "none"
    for name, required in ladder:
        absent = tuple(
            factor for factor in required if not factor_set.available((factor,))
        )
        if absent:
            levels[name] = AttributionLevel(
                name=name, status="unavailable", missing_factors=absent
            )
            continue
        alpha_daily, alpha_t, loadings, r2, n = _regress(
            excess, factor_set.returns[list(required)]
        )
        if not np.isfinite(alpha_daily):
            levels[name] = AttributionLevel(
                name=name, status="unavailable", missing_factors=("insufficient_overlap",)
            )
            continue
        levels[name] = AttributionLevel(
            name=name,
            status="measured",
            alpha_annual=_annualise_daily_alpha(alpha_daily),
            alpha_t_stat=alpha_t,
            loadings=loadings,
            r_squared=r2,
            observations=n,
        )
        strictest = name

    survives = False
    if strictest != "none":
        level = levels[strictest]
        survives = bool(
            level.alpha_annual is not None
            and level.alpha_annual > 0.0
            and level.alpha_t_stat is not None
            and np.isfinite(level.alpha_t_stat)
            and level.alpha_t_stat >= t_threshold
        )
    return AttributionReport(
        levels=levels,
        factor_set=factor_set,
        strictest_measured=strictest,
        survives_style_controls=survives,
        t_threshold=float(t_threshold),
    )


__all__ = [
    "ANN",
    "AttributionLevel",
    "AttributionReport",
    "MARKET",
    "MOMENTUM",
    "SIZE",
    "StyleFactorSet",
    "VALUE",
    "attribute_strategy_returns",
    "build_ashare_style_factors",
]
