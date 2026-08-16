"""PIT-safe dataset: features knowable at close(T), labels earned after it.

The alignment rule, stated once and enforced everywhere below:

    features  use bars up to and including close(T)
    execution happens at close(T+1)
    label     is the return close(T+1) -> close(T+1+h)

So a feature and the return it predicts never share a bar. This is the same
contract the PIT RL environment uses, and it is the contract the legacy
event-driven engine violates by marking NAV a bar early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("data/raw/ashare_daily_full/panel_all.parquet")

#: Features are all trailing transforms of bars <= T. Adding anything here that
#: peeks at T+1 breaks the whole package's guarantee.
FEATURES = (
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d",
    "turn_z_20d", "amihud_20d",
    "px_pos_60d", "gap_from_ma20",
)


@dataclass(frozen=True)
class CleanRoomDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    features: tuple[str, ...]
    label: str
    train_span: tuple[str, str]
    test_span: tuple[str, str]
    horizon: int
    #: Names dropped for insufficient coverage, recorded rather than silently lost.
    dropped_symbols: int
    coverage_threshold: float


def _panel(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["symbol", "trade_date", "open", "high", "low",
                       "close", "volume", "amount"]
    )
    frame = frame[frame["trade_date"] >= start]
    # A non-positive close is not a price. Dropping is safe here (these are
    # suspension rows the feed encodes as 0.0); keeping them would let a return
    # of -100% or +inf enter a feature.
    frame = frame[(frame["close"] > 0) & frame["close"].notna()]
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("symbol", sort=False)
    close = frame["close"]

    frame["ret_1d"] = g["close"].pct_change(1)
    frame["ret_5d"] = g["close"].pct_change(5)
    frame["ret_20d"] = g["close"].pct_change(20)
    frame["ret_60d"] = g["close"].pct_change(60)

    r1 = frame["ret_1d"]
    frame["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
    frame["vol_60d"] = g["ret_1d"].transform(lambda s: s.rolling(60, min_periods=40).std())

    # Turnover z-score: amount is genuine CNY here (Sina), verified at ingest.
    amt = frame["amount"]
    mu = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    sd = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).std())
    frame["turn_z_20d"] = (amt - mu) / sd.replace(0.0, np.nan)

    # Amihud illiquidity: |return| per unit of turnover.
    frame["amihud_20d"] = g.apply(
        lambda d: (d["ret_1d"].abs() / (d["amount"].replace(0.0, np.nan) / 1e8))
        .rolling(20, min_periods=15).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    hi = g["close"].transform(lambda s: s.rolling(60, min_periods=40).max())
    lo = g["close"].transform(lambda s: s.rolling(60, min_periods=40).min())
    frame["px_pos_60d"] = (close - lo) / (hi - lo).replace(0.0, np.nan)

    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    frame["gap_from_ma20"] = close / ma20.replace(0.0, np.nan) - 1.0
    return frame


def _add_label(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Executable label: close(T+1) -> close(T+1+h).

    Not close(T)->close(T+h). A signal formed at close(T) cannot be traded at
    close(T), so a label starting at T credits a fill nobody could have got --
    the exact defect that made the legacy PPO reward unachievable.
    """
    g = frame.groupby("symbol", sort=False)["close"]
    entry = g.shift(-1)
    exit_ = g.shift(-(1 + horizon))
    frame[f"fwd_exec_return_{horizon}d"] = exit_ / entry - 1.0
    frame["entry_close"] = entry
    return frame


def build_dataset(
    *,
    panel_path: Path = PANEL,
    test_years: int = 1,
    train_years: int = 3,
    horizon: int = 5,
    min_coverage: float = 0.8,
    embargo_days: int = 10,
) -> CleanRoomDataset:
    """Build train/test with a hard embargo between them.

    ``embargo_days`` removes the last ``horizon + embargo`` training sessions so
    no training label can reach into the test window. Without it the final
    training rows carry returns realised after the split, which is leakage that
    looks like skill.
    """
    warmup = pd.DateOffset(days=180)  # rolling windows need history before train
    end = pd.read_parquet(panel_path, columns=["trade_date"])["trade_date"].max()
    test_start = end - pd.DateOffset(years=test_years)
    train_start = test_start - pd.DateOffset(years=train_years)

    frame = _panel(panel_path, train_start - warmup)

    sessions = frame["trade_date"].nunique()
    counts = frame.groupby("symbol")["trade_date"].nunique()
    keep = counts[counts >= min_coverage * sessions].index
    dropped = int(counts.shape[0] - len(keep))
    frame = frame[frame["symbol"].isin(keep)].copy()

    frame = _add_features(frame)
    frame = _add_label(frame, horizon)
    frame = frame[frame["trade_date"] >= train_start]

    label = f"fwd_exec_return_{horizon}d"
    # Purge: a training row whose label window touches the test window is leakage.
    purge_until = test_start - pd.Timedelta(days=embargo_days + horizon * 2)
    train = frame[frame["trade_date"] < purge_until].dropna(subset=[label, *FEATURES])
    test = frame[frame["trade_date"] >= test_start].copy()

    return CleanRoomDataset(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        features=FEATURES,
        label=label,
        train_span=(str(train["trade_date"].min().date()), str(train["trade_date"].max().date())),
        test_span=(str(test["trade_date"].min().date()), str(test["trade_date"].max().date())),
        horizon=horizon,
        dropped_symbols=dropped,
        coverage_threshold=min_coverage,
    )
