"""Per-response volume-unit truth for A-share daily bars.

Why this module exists
----------------------
A-share daily-bar vendors disagree on the volume unit (shares vs 手 = 100-share
lots), and the disagreement is not even stable *inside* one vendor:

* Tencent's native daily volume is lots on the main boards / ChiNext but
  **shares on STAR** (akshare PR #7328, and measured: sh688981 2024-01-02
  13,459,147 == Sina shares, ratio 1.000000 on 58/58 sessions).
* akshare >= 1.18.69 multiplies Tencent volume by 100 itself, except for any
  symbol starting ``sh688`` / ``sz000`` / ``sh000`` / ``sz399``. The ``sz000``
  exclusion was meant for indices but also catches every 000xxx **stock**, so
  000001.SZ comes back in lots under a docstring that promises shares.
* akshare 1.18.60 returned Tencent volume in a column it called ``amount``
  and no CNY turnover at all.

Every static unit table written so far has been wrong for at least one of these
shapes (DEF-032 was the first; the STAR and ``sz000`` cases are the second and
third). So the unit is decided **per response, from the response itself**:

    implied VWAP = amount / (volume * scale)   must lie inside [low, high]

Candidate scales differ by 100x while a daily [low, high] range spans at most
~1.6x (a +/-30% limit board), so at most one candidate can fit any given row.
Exactly one scale must fit the bulk of the testable rows; otherwise the
response is ``UNIT_AMBIGUOUS`` and must not be written as ``OK``.

A response with no CNY turnover (Tencent's raw ``fqkline`` endpoint, akshare
1.18.60's Tencent shape) cannot be verified at all. It may still carry a
board-level *prior* scale so that the number is the best available guess, but
it is labelled unverified and never certified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from quantagent.data.ashare.contracts import QUALITY_UNIT_AMBIGUOUS

__all__ = [
    "QUALITY_UNIT_AMBIGUOUS",
    "VOLUME_SCALE_CANDIDATES",
    "VolumeScaleVerdict",
    "implied_vwap_in_band",
    "infer_volume_scale",
    "tencent_native_volume_prior",
]

#: Multipliers from a vendor's raw volume to shares. 1 = already shares,
#: 100 = 手 (lots).
VOLUME_SCALE_CANDIDATES: tuple[float, ...] = (1.0, 100.0)

#: A-share price tick (CNY). Added to the band so a one-price (limit-locked)
#: session whose VWAP rounds a cent outside [low, high] is not misread.
DEFAULT_PRICE_TICK = 0.01
#: Relative slack for vendor rounding of amount (Tencent publishes 万元) and of
#: lot-rounded volume. The candidates are 100x apart, so this cannot make two
#: scales fit the same row.
DEFAULT_RELATIVE_TOLERANCE = 0.02
#: "The bulk of rows": the share of testable rows the winning scale must fit.
DEFAULT_MIN_FIT = 0.95

STATUS_VERIFIED = "verified"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNVERIFIABLE = "unverifiable"

UNIT_NAME_BY_SCALE: dict[float, str] = {1.0: "shares", 100.0: "lots_100_shares"}


@dataclass(frozen=True)
class VolumeScaleVerdict:
    """What one response proves about its own volume unit."""

    status: str
    #: Multiplier actually applied to the raw volume (verified scale, or the
    #: declared fallback/prior when unverified; ``None`` = nothing applied).
    scale: float | None
    basis: str
    testable_rows: int = 0
    fit_rates: tuple[tuple[float, float], ...] = ()
    #: Per-row in-band flag under the applied scale (NaN where untestable).
    row_in_band: pd.Series | None = field(default=None, compare=False, repr=False)

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    @property
    def raw_unit_name(self) -> str:
        if self.scale is None:
            return "unavailable"
        name = UNIT_NAME_BY_SCALE.get(float(self.scale), f"x{self.scale:g}")
        return name if self.verified else f"{name}_unverified"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "scale": self.scale,
            "basis": self.basis,
            "testable_rows": self.testable_rows,
            "fit_rates": {f"{s:g}": round(r, 6) for s, r in self.fit_rates},
        }


def _numeric(values: object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").astype("float64")
    return pd.Series(pd.to_numeric(np.asarray(values, dtype=object), errors="coerce"),
                     index=index, dtype="float64")


def implied_vwap_in_band(
    volume_shares: pd.Series,
    amount: pd.Series,
    low: pd.Series,
    high: pd.Series,
    *,
    tick: float = DEFAULT_PRICE_TICK,
    rel_tol: float = DEFAULT_RELATIVE_TOLERANCE,
) -> pd.Series:
    """Per-row: does ``amount / volume`` sit inside ``[low, high]``?

    Returns a float series: 1.0 inside, 0.0 outside, NaN where the row carries
    no testable turnover (missing/zero volume or amount, or an invalid range).
    """
    volume_shares = _numeric(volume_shares)
    amount = _numeric(amount, volume_shares.index)
    low = _numeric(low, volume_shares.index)
    high = _numeric(high, volume_shares.index)
    testable = (
        volume_shares.gt(0) & amount.gt(0) & low.gt(0) & high.gt(0) & high.ge(low)
    )
    vwap = amount / volume_shares.where(testable)
    lower = low * (1.0 - rel_tol) - tick
    upper = high * (1.0 + rel_tol) + tick
    inside = (vwap >= lower) & (vwap <= upper)
    return inside.astype("float64").where(testable)


def infer_volume_scale(
    raw_volume: pd.Series,
    amount: pd.Series | None,
    low: pd.Series,
    high: pd.Series,
    *,
    candidates: Sequence[float] = VOLUME_SCALE_CANDIDATES,
    tick: float = DEFAULT_PRICE_TICK,
    rel_tol: float = DEFAULT_RELATIVE_TOLERANCE,
    min_fit: float = DEFAULT_MIN_FIT,
    fallback_scale: float | None = None,
    fallback_reason: str = "",
) -> VolumeScaleVerdict:
    """Decide the raw-volume -> shares multiplier from the response itself.

    ``fallback_scale`` is what an *unverified* response is scaled by (a
    documented vendor unit or a board-level prior). It never upgrades the
    verdict: a response that cannot prove its unit stays unverified/ambiguous.
    """
    raw_volume = _numeric(raw_volume)
    index = raw_volume.index
    low = _numeric(low, index)
    high = _numeric(high, index)
    amount_series = (
        _numeric(amount, index) if amount is not None else pd.Series(np.nan, index=index)
    )
    fallback_note = (
        f"fallback_scale={fallback_scale:g}({fallback_reason or 'declared'})"
        if fallback_scale is not None
        else "no_fallback_scale"
    )
    if raw_volume.notna().sum() == 0:
        return VolumeScaleVerdict(
            STATUS_UNVERIFIABLE, None, "no_volume_values", 0, (), None
        )
    if amount_series.notna().sum() == 0:
        row_flags = pd.Series(np.nan, index=index)
        return VolumeScaleVerdict(
            STATUS_UNVERIFIABLE,
            fallback_scale,
            f"unverifiable:no_cny_amount:{fallback_note}",
            0,
            (),
            row_flags,
        )

    flags_by_scale: dict[float, pd.Series] = {}
    fits: list[tuple[float, float]] = []
    testable_rows = 0
    for scale in candidates:
        flags = implied_vwap_in_band(
            raw_volume * float(scale), amount_series, low, high, tick=tick, rel_tol=rel_tol
        )
        flags_by_scale[float(scale)] = flags
        testable_rows = int(flags.notna().sum())
        rate = float(flags.mean()) if testable_rows else 0.0
        fits.append((float(scale), rate))
    fit_text = ",".join(f"fit[x{s:g}]={r:.4f}" for s, r in fits)
    if testable_rows == 0:
        return VolumeScaleVerdict(
            STATUS_UNVERIFIABLE,
            fallback_scale,
            f"unverifiable:no_positive_turnover_rows:{fallback_note}",
            0,
            tuple(fits),
            pd.Series(np.nan, index=index),
        )
    winners = [scale for scale, rate in fits if rate >= min_fit]
    if len(winners) == 1:
        scale = winners[0]
        return VolumeScaleVerdict(
            STATUS_VERIFIED,
            scale,
            f"implied_vwap_in_range:scale=x{scale:g}:{fit_text}:rows={testable_rows}",
            testable_rows,
            tuple(fits),
            flags_by_scale[scale],
        )
    applied = fallback_scale
    flags = flags_by_scale.get(float(applied)) if applied is not None else None
    return VolumeScaleVerdict(
        STATUS_AMBIGUOUS,
        applied,
        f"ambiguous:{len(winners)}_scales_fit_bulk(min_fit={min_fit:g}):{fit_text}:"
        f"rows={testable_rows}:{fallback_note}",
        testable_rows,
        tuple(fits),
        flags if flags is not None else pd.Series(np.nan, index=index),
    )


def tencent_native_volume_prior(symbol: str) -> tuple[float | None, str]:
    """Board-level prior for Tencent's *native* daily volume unit.

    Used only to scale a response that cannot prove its own unit (no CNY
    turnover); such rows remain unverified. Source: akshare PR #7328
    ("主板/创业板为手, 科创板为股") and R5's measurement on sh688981.
    BSE is not served by Tencent's daily endpoint (akshare #6205/#7201), so
    there is no measured prior and nothing is guessed.
    """
    from quantagent.data.ashare.symbols import BOARD_BSE, BOARD_STAR, SymbolError, identify

    try:
        ident = identify(symbol)
    except SymbolError:
        return None, "unidentifiable_symbol"
    if ident.board == BOARD_STAR:
        return 1.0, "tencent_star_native_shares"
    if ident.board == BOARD_BSE:
        return None, "tencent_bse_unit_unmeasured"
    return 100.0, "tencent_lots"
