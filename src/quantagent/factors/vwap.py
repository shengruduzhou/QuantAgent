"""VWAP derivation that keeps "no trading" and "no turnover data" apart.

``vwap = amount / volume`` is undefined in two very different situations, and
collapsing them is a measurement error that survives every internal consistency
check:

* **volume == 0** -- the name genuinely did not trade. VWAP does not exist, and
  substituting the close is the conventional, defensible reference price.
* **amount is missing** -- the name traded, but the source published no turnover.
  The average price is *unknown*. Substituting the close asserts a measurement
  that was never made, and it is not a harmless one: any factor of the form
  ``close / vwap - 1`` then evaluates to exactly 0 for every affected row, which
  reads downstream as a real, measured, zero-valued factor rather than as
  missing data.

This distinction became load-bearing once the AKShare Tencent adapter began
reporting ``amount`` as unavailable (that source publishes volume but no CNY
turnover). Tencent is the failover, so it engages whenever EastMoney is
unreachable -- i.e. exactly when nobody is watching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vwap_or_unknown(
    frame: pd.DataFrame,
    *,
    amount_column: str = "amount",
    volume_column: str = "volume",
    close_column: str = "close",
) -> pd.Series:
    """Return VWAP, NaN where turnover is unavailable, close where nothing traded.

    Args:
        frame: daily bars carrying amount (CNY), volume (shares) and close.

    Returns:
        A float Series aligned to ``frame.index``.
    """
    amount = pd.to_numeric(frame[amount_column], errors="coerce")
    volume = pd.to_numeric(frame[volume_column], errors="coerce")
    close = pd.to_numeric(frame[close_column], errors="coerce")

    vwap = amount / volume.replace(0.0, np.nan)

    # Only a genuine absence of trading earns the close as a stand-in. Rows whose
    # turnover is simply unpublished stay NaN and propagate as unknown.
    no_trading = volume.fillna(0.0) <= 0.0
    return vwap.mask(no_trading, close)
