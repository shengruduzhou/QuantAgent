from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

FactorCompute = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass
class FactorMeta:
    name: str
    category: str
    horizon_days: int
    required_columns: tuple[str, ...]
    direction: int
    description: str
    source: str
    compute: FactorCompute | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")


@dataclass(frozen=True)
class FactorOutput:
    name: str
    frame: pd.DataFrame
    meta: FactorMeta

    @property
    def values(self) -> pd.Series:
        return self.frame["factor_value"]


class FactorRegistry:
    """Typed registry for point-in-time factor compute functions."""

    def __init__(self) -> None:
        self._factors: dict[str, FactorMeta] = {}

    def register(self, meta: FactorMeta) -> Callable[[FactorCompute], FactorCompute]:
        def decorator(func: FactorCompute) -> FactorCompute:
            if meta.name in self._factors:
                raise ValueError(f"Factor already registered: {meta.name}")
            meta.compute = func
            self._factors[meta.name] = meta
            return func

        return decorator

    def add(self, meta: FactorMeta, func: FactorCompute) -> None:
        self.register(meta)(func)

    def get(self, name: str) -> FactorMeta:
        try:
            return self._factors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown factor: {name}") from exc

    def names(self, category: str | None = None) -> list[str]:
        if category is None:
            return sorted(self._factors)
        return sorted(name for name, meta in self._factors.items() if meta.category == category)

    def metas(self) -> Mapping[str, FactorMeta]:
        return dict(self._factors)

    def compute(self, name: str, frame: pd.DataFrame) -> FactorOutput:
        meta = self.get(name)
        if meta.compute is None:
            raise ValueError(f"Factor has no compute function: {name}")
        _validate_panel(frame, meta.required_columns)
        prepared = _prepare_panel(frame)
        result = meta.compute(prepared)
        factor_frame = _coerce_factor_frame(result, frame, prepared, meta.name)
        return FactorOutput(name=meta.name, frame=factor_frame, meta=meta)

    def batch_compute(
        self,
        frame: pd.DataFrame,
        names: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        selected = list(names) if names is not None else self.names()
        if categories is not None:
            category_set = set(categories)
            selected = [name for name in selected if self.get(name).category in category_set]
        outputs = [self.compute(name, frame).frame for name in selected]
        if not outputs:
            return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
        return pd.concat(outputs, ignore_index=True).sort_values(
            ["factor_name", "trade_date", "symbol"]
        ).reset_index(drop=True)


def _validate_panel(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    base = {"trade_date", "symbol"}
    missing = (base | set(required_columns)).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    return data.sort_values(["symbol", "trade_date"]).reset_index(drop=False).rename(
        columns={"index": "_original_index"}
    )


def _coerce_factor_frame(
    result: pd.Series | pd.DataFrame,
    source_frame: pd.DataFrame,
    prepared_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    base = source_frame[["trade_date", "symbol"]].copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"])

    if isinstance(result, pd.Series):
        values = _align_values(result, source_frame, prepared_frame)
        out = base.copy()
        out["factor_name"] = factor_name
        out["factor_value"] = values.to_numpy(dtype=float)
        return out

    if {"trade_date", "symbol", "factor_value"}.issubset(result.columns):
        out = result[["trade_date", "symbol", "factor_value"]].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out["factor_name"] = factor_name
        return out[["trade_date", "symbol", "factor_name", "factor_value"]]

    if factor_name in result.columns:
        values = _align_values(result[factor_name], source_frame, prepared_frame)
    elif "factor_value" in result.columns:
        values = _align_values(result["factor_value"], source_frame, prepared_frame)
    else:
        raise ValueError("Factor compute result must be a Series or contain factor_value")
    out = base.copy()
    out["factor_name"] = factor_name
    out["factor_value"] = values.to_numpy(dtype=float)
    return out


def _align_values(
    values: pd.Series,
    source_frame: pd.DataFrame,
    prepared_frame: pd.DataFrame,
) -> pd.Series:
    if values.index.equals(source_frame.index):
        return values.reindex(source_frame.index)
    if len(values) == len(prepared_frame):
        restored = pd.Series(values.to_numpy(dtype=float), index=prepared_frame["_original_index"])
        return restored.reindex(source_frame.index)
    return values.reindex(source_frame.index)


default_registry = FactorRegistry()
