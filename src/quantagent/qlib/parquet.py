"""PIT-aware QuantAgent -> Qlib StaticDataLoader Parquet bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from quantagent.data.v7_auto_range import to_qlib_instrument


_FUTURE_FEATURE_PATTERN = re.compile(
    r"(^|[_\W])(forward|future|lead|label|target)([_\W]|$)", re.IGNORECASE
)


@dataclass(frozen=True)
class QlibParquetManifest:
    output_path: str
    rows: int
    instruments: int
    start_time: str
    end_time: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    symbol_column: str
    time_column: str
    point_in_time_policy: str = "Qlib datetime equals QuantAgent feature available_at"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalise_columns(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    return result


def _assert_safe_features(feature_columns: tuple[str, ...]) -> None:
    suspicious = [name for name in feature_columns if _FUTURE_FEATURE_PATTERN.search(name)]
    if suspicious:
        raise ValueError(
            "Qlib feature set contains label/future-like columns and is blocked: "
            + ", ".join(suspicious)
        )


def build_qlib_static_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    label_columns: Iterable[str] = (),
    symbol_column: str = "symbol",
    time_column: str = "available_at",
) -> pd.DataFrame:
    """Return a StaticDataLoader-compatible frame.

    Feature timestamps are deliberately indexed by ``available_at`` rather than
    source ``trade_date`` by default.  Future outcomes may be present only in the
    top-level ``label`` column group; they can never be admitted as features.
    """
    if frame is None or frame.empty:
        raise ValueError("cannot build a Qlib dataset from an empty frame")

    features = _normalise_columns(feature_columns)
    labels = _normalise_columns(label_columns)
    if not features:
        raise ValueError("feature_columns must contain at least one column")
    _assert_safe_features(features)
    overlap = sorted(set(features).intersection(labels))
    if overlap:
        raise ValueError(f"feature/label columns overlap: {overlap}")

    required = {symbol_column, time_column, *features, *labels}
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"missing Qlib bridge columns: {missing}")

    source = frame.loc[:, [symbol_column, time_column, *features, *labels]].copy()
    timestamps = pd.to_datetime(source[time_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{time_column} contains missing or invalid timestamps")

    instruments = source[symbol_column].astype(str).map(to_qlib_instrument)
    if instruments.isna().any() or (instruments.str.len() == 0).any():
        raise ValueError(f"{symbol_column} contains invalid instruments")

    index = pd.MultiIndex.from_arrays(
        [timestamps, instruments],
        names=["datetime", "instrument"],
    )
    if index.has_duplicates:
        duplicates = int(index.duplicated(keep=False).sum())
        raise ValueError(
            "Qlib StaticDataLoader index must be unique; "
            f"found {duplicates} duplicate datetime/instrument rows"
        )

    values = source.loc[:, [*features, *labels]].copy()
    values.index = index
    values.columns = pd.MultiIndex.from_tuples(
        [*(("feature", name) for name in features), *(("label", name) for name in labels)]
    )
    return values.sort_index()


def write_qlib_static_parquet(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    feature_columns: Iterable[str],
    label_columns: Iterable[str] = (),
    symbol_column: str = "symbol",
    time_column: str = "available_at",
) -> QlibParquetManifest:
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".parquet":
        raise ValueError("Qlib StaticDataLoader output must use a .parquet suffix")
    qlib_frame = build_qlib_static_frame(
        frame,
        feature_columns=feature_columns,
        label_columns=label_columns,
        symbol_column=symbol_column,
        time_column=time_column,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    qlib_frame.to_parquet(output, engine="pyarrow")
    datetimes = qlib_frame.index.get_level_values("datetime")
    instruments = qlib_frame.index.get_level_values("instrument")
    features = _normalise_columns(feature_columns)
    labels = _normalise_columns(label_columns)
    return QlibParquetManifest(
        output_path=str(output),
        rows=int(len(qlib_frame)),
        instruments=int(instruments.nunique()),
        start_time=pd.Timestamp(datetimes.min()).isoformat(),
        end_time=pd.Timestamp(datetimes.max()).isoformat(),
        feature_columns=features,
        label_columns=labels,
        symbol_column=symbol_column,
        time_column=time_column,
    )
