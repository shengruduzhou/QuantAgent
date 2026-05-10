from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierConfig:
    """pt_sl: profit-taking and stop-loss multiples of sigma."""

    pt_sl: tuple[float, float] = (2.0, 1.0)
    max_holding_days: int = 10
    min_return: float = 0.0


def daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """Lopez de Prado AFML section 3.1 EWMA vol of log returns."""
    log_ret = np.log(close).diff()
    return log_ret.ewm(span=span, adjust=False).std()


def triple_barrier_labels(
    close: pd.Series,
    sigma: pd.Series,
    side: pd.Series | None = None,
    config: BarrierConfig | None = None,
) -> pd.DataFrame:
    """First-touch barrier labeling for a single symbol.

    Returns columns: t1, ret, label, barrier ('pt'/'sl'/'vt').
    """
    config = config or BarrierConfig()
    close = close.dropna().sort_index()
    sigma = sigma.reindex(close.index)
    if side is None:
        side = pd.Series(1.0, index=close.index)
    out = []
    n = len(close)
    for i, (t0, p0) in enumerate(close.items()):
        s = float(sigma.iloc[i]) if not np.isnan(sigma.iloc[i]) else np.nan
        sign = float(side.iloc[i])
        end = min(i + config.max_holding_days, n - 1)
        if end <= i or np.isnan(s) or s <= 0:
            out.append((t0, close.index[end], np.nan, 0, "vt"))
            continue
        window = close.iloc[i + 1 : end + 1]
        path_ret = sign * (np.log(window) - np.log(p0))
        upper = config.pt_sl[0] * s
        lower = -config.pt_sl[1] * s
        hit_pt = path_ret[path_ret >= upper]
        hit_sl = path_ret[path_ret <= lower]
        first_pt = hit_pt.index[0] if not hit_pt.empty else None
        first_sl = hit_sl.index[0] if not hit_sl.empty else None
        candidates = [c for c in (first_pt, first_sl) if c is not None]
        if candidates:
            t1 = min(candidates)
            barrier = "pt" if t1 == first_pt else "sl"
        else:
            t1 = close.index[end]
            barrier = "vt"
        ret = float(path_ret.loc[t1])
        if barrier == "vt" and abs(ret) < config.min_return:
            label = 0
        else:
            label = int(np.sign(ret))
        out.append((t0, t1, ret, label, barrier))
    return pd.DataFrame(out, columns=["t0", "t1", "ret", "label", "barrier"]).set_index("t0")


def sample_weights_by_uniqueness(events: pd.DataFrame, close_index: pd.Index) -> pd.Series:
    """AFML section 4.4 average uniqueness weight: 1 / mean(num concurrent events)."""
    num_co = pd.Series(0, index=close_index, dtype=float)
    for t0, row in events.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            continue
        num_co.loc[t0:t1] += 1.0
    weights = pd.Series(np.nan, index=events.index, dtype=float)
    for t0, row in events.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            continue
        seg = num_co.loc[t0:t1]
        if (seg > 0).any():
            weights.loc[t0] = float((1.0 / seg.where(seg > 0, np.nan)).mean())
    return weights.fillna(weights.mean() if not weights.empty else 1.0)


def meta_label(
    primary_signal: pd.Series,
    primary_label: pd.Series,
    secondary_score: pd.Series,
    threshold: float = 0.5,
) -> pd.Series:
    """Meta-labeling: primary picks side, secondary decides size 0/1."""
    aligned_secondary = secondary_score.reindex(primary_signal.index).fillna(0.0)
    take = (aligned_secondary >= threshold).astype(float)
    return primary_signal.sign() * take * primary_label.abs()
