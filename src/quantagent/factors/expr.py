"""Symbolic factor expression DSL inspired by Alpha101 / AKQuant.

Factors are expressed as composable :class:`Expr` objects — for example
``Rank(TsMean(Close, 5))`` — and evaluated against a wide pandas frame
keyed on ``(symbol, trade_date)``. Operators always preserve the index
of the input frame and emit a numeric ``pd.Series`` so factors can be
chained or composed without manual alignment.

Key guarantees:

* All time-series operators (``TsMean``, ``Delay`` etc.) are computed
  per ``symbol`` group and respect the ascending ``trade_date`` sort.
* The DSL has **zero look-ahead**: every operator uses only data
  observed up to and including the current row (``Delay`` shifts
  values forward in time but never backward).
* ``Rank`` is cross-sectional per ``trade_date`` so it's safe to
  combine ranks across symbols on the same day.

A small registry (:func:`register_factor`, :func:`build_factor_frame`)
lets callers declare a library of named factor expressions and
materialise them as a tidy long-format DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


class Expr:
    """Base class for factor-expression nodes."""

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    # Operator overloads for ergonomic composition.
    def __add__(self, other: "Expr | float") -> "Add":
        return Add(self, _wrap(other))

    def __radd__(self, other: float) -> "Add":
        return Add(_wrap(other), self)

    def __sub__(self, other: "Expr | float") -> "Sub":
        return Sub(self, _wrap(other))

    def __mul__(self, other: "Expr | float") -> "Mul":
        return Mul(self, _wrap(other))

    def __rmul__(self, other: float) -> "Mul":
        return Mul(_wrap(other), self)

    def __truediv__(self, other: "Expr | float") -> "Div":
        return Div(self, _wrap(other))

    def __neg__(self) -> "Mul":
        return Mul(self, _wrap(-1.0))


def _wrap(value: "Expr | float | int") -> Expr:
    if isinstance(value, Expr):
        return value
    return Constant(float(value))


@dataclass(frozen=True)
class Column(Expr):
    name: str

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if self.name not in frame.columns:
            raise KeyError(f"factor expression references missing column '{self.name}'")
        return pd.to_numeric(frame[self.name], errors="coerce")


@dataclass(frozen=True)
class Constant(Expr):
    value: float

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=frame.index, dtype=float)


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(frame).add(self.right.evaluate(frame), fill_value=0.0)


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(frame).sub(self.right.evaluate(frame), fill_value=0.0)


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(frame).mul(self.right.evaluate(frame), fill_value=1.0)


@dataclass(frozen=True)
class Div(Expr):
    numerator: Expr
    denominator: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        denom = self.denominator.evaluate(frame).replace(0.0, np.nan)
        return self.numerator.evaluate(frame) / denom


@dataclass(frozen=True)
class Delay(Expr):
    """``Delay(x, k)`` shifts ``x`` forward by ``k`` rows per symbol."""

    expr: Expr
    periods: int = 1

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        values = self.expr.evaluate(frame)
        sorted_frame = _ensure_sorted(frame)
        symbols = sorted_frame["symbol"].to_numpy()
        ordered_values = values.reindex(sorted_frame.index)
        shifted = ordered_values.groupby(symbols, sort=False).shift(self.periods)
        return shifted.reindex(values.index)


@dataclass(frozen=True)
class Delta(Expr):
    expr: Expr
    periods: int = 1

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        current = self.expr.evaluate(frame)
        shifted = Delay(self.expr, self.periods).evaluate(frame)
        return current - shifted


@dataclass(frozen=True)
class _RollingReduction(Expr):
    expr: Expr
    window: int
    op: str

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        values = self.expr.evaluate(frame)
        sorted_frame = _ensure_sorted(frame)
        ordered_values = values.reindex(sorted_frame.index)
        grouped = ordered_values.groupby(sorted_frame["symbol"].to_numpy(), sort=False)
        rolled = grouped.rolling(window=self.window, min_periods=self.window)
        if self.op == "mean":
            result = rolled.mean()
        elif self.op == "std":
            result = rolled.std()
        elif self.op == "sum":
            result = rolled.sum()
        elif self.op == "max":
            result = rolled.max()
        elif self.op == "min":
            result = rolled.min()
        else:  # pragma: no cover - protected by constructors
            raise ValueError(f"unsupported rolling op: {self.op}")
        result = result.reset_index(level=0, drop=True)
        return result.reindex(values.index)


def TsMean(expr: Expr, window: int) -> Expr:
    return _RollingReduction(expr, window, "mean")


def TsStd(expr: Expr, window: int) -> Expr:
    return _RollingReduction(expr, window, "std")


def TsSum(expr: Expr, window: int) -> Expr:
    return _RollingReduction(expr, window, "sum")


def TsMax(expr: Expr, window: int) -> Expr:
    return _RollingReduction(expr, window, "max")


def TsMin(expr: Expr, window: int) -> Expr:
    return _RollingReduction(expr, window, "min")


@dataclass(frozen=True)
class TsRank(Expr):
    expr: Expr
    window: int

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        values = self.expr.evaluate(frame)
        sorted_frame = _ensure_sorted(frame)
        ordered_values = values.reindex(sorted_frame.index)
        grouped = ordered_values.groupby(sorted_frame["symbol"].to_numpy(), sort=False)
        ranked = grouped.rolling(window=self.window, min_periods=self.window).apply(
            lambda block: pd.Series(block).rank(method="average").iloc[-1] / len(block),
            raw=True,
        )
        ranked = ranked.reset_index(level=0, drop=True)
        return ranked.reindex(values.index)


@dataclass(frozen=True)
class Rank(Expr):
    """Cross-sectional rank per ``trade_date``; output in (0, 1]."""

    expr: Expr
    method: str = "average"
    pct: bool = True

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        values = self.expr.evaluate(frame)
        ranks = values.groupby(frame["trade_date"], sort=False).rank(method=self.method, pct=self.pct)
        return ranks


@dataclass(frozen=True)
class Returns(Expr):
    """``Returns(close, k)`` = close / Delay(close, k) - 1."""

    expr: Expr
    periods: int = 1

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        current = self.expr.evaluate(frame)
        delayed = Delay(self.expr, self.periods).evaluate(frame)
        return (current / delayed.replace(0.0, np.nan)) - 1.0


@dataclass(frozen=True)
class Abs(Expr):
    expr: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return self.expr.evaluate(frame).abs()


@dataclass(frozen=True)
class Sign(Expr):
    expr: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return np.sign(self.expr.evaluate(frame))


@dataclass(frozen=True)
class Log(Expr):
    expr: Expr

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        return np.log(self.expr.evaluate(frame).clip(lower=1e-12))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = ("symbol", "trade_date")


def _ensure_sorted(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [k for k in _REQUIRED_KEYS if k not in frame.columns]
    if missing:
        raise KeyError(f"factor evaluation requires {_REQUIRED_KEYS}; missing {missing}")
    return frame.assign(
        trade_date=pd.to_datetime(frame["trade_date"], errors="coerce")
    ).sort_values(["symbol", "trade_date"], kind="mergesort")


# Conventional column aliases for ergonomic factor authoring.
Open = Column("open")
High = Column("high")
Low = Column("low")
Close = Column("close")
Volume = Column("volume")
Amount = Column("amount")
Vwap = Div(Amount, Volume)


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    expr: Expr
    description: str = ""


@dataclass
class FactorRegistry:
    factors: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, name: str, expr: Expr, description: str = "") -> FactorDefinition:
        definition = FactorDefinition(name=name, expr=expr, description=description)
        self.factors[name] = definition
        return definition

    def evaluate_all(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Evaluate every registered factor and return a tidy long-format frame."""
        if not self.factors:
            return pd.DataFrame(columns=["symbol", "trade_date", "factor_name", "factor_value"])
        rows: list[pd.DataFrame] = []
        base = frame[["symbol", "trade_date"]].copy()
        for name, definition in self.factors.items():
            values = definition.expr.evaluate(frame)
            piece = base.copy()
            piece["factor_name"] = name
            piece["factor_value"] = values.values
            rows.append(piece)
        return pd.concat(rows, ignore_index=True, sort=False)

    def evaluate_wide(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Evaluate every registered factor as a wide ``factor_*`` frame."""
        wide = frame[["symbol", "trade_date"]].copy()
        for name, definition in self.factors.items():
            wide[f"factor_{name}"] = definition.expr.evaluate(frame).values
        return wide


_DEFAULT_REGISTRY = FactorRegistry()


def register_factor(name: str, expr: Expr, description: str = "") -> FactorDefinition:
    """Register a named factor in the module-level default registry."""
    return _DEFAULT_REGISTRY.register(name, expr, description=description)


def default_registry() -> FactorRegistry:
    return _DEFAULT_REGISTRY


def build_factor_frame(
    frame: pd.DataFrame,
    factors: dict[str, Expr] | FactorRegistry | None = None,
    long_format: bool = False,
) -> pd.DataFrame:
    """Evaluate a collection of expressions over ``frame``.

    ``factors`` may be a ``FactorRegistry``, a ``dict`` of name → ``Expr``,
    or ``None`` (which falls back to the module-level default registry).
    Use ``long_format=True`` to emit the AKQuant-style
    ``symbol, trade_date, factor_name, factor_value`` layout.
    """
    if factors is None:
        registry = _DEFAULT_REGISTRY
    elif isinstance(factors, FactorRegistry):
        registry = factors
    else:
        registry = FactorRegistry()
        for name, expr in factors.items():
            registry.register(name, expr)
    return registry.evaluate_all(frame) if long_format else registry.evaluate_wide(frame)


# A small handful of canonical Alpha101-style examples for tests and docs.
def _seed_default_registry() -> None:
    if _DEFAULT_REGISTRY.factors:
        return
    _DEFAULT_REGISTRY.register(
        "momentum_5",
        Rank(TsMean(Returns(Close, 1), 5)),
        description="Rank of 5-day mean daily return.",
    )
    _DEFAULT_REGISTRY.register(
        "reversal_1",
        -Rank(Returns(Close, 1)),
        description="Negative rank of 1-day return (1-day reversal).",
    )
    _DEFAULT_REGISTRY.register(
        "volatility_20",
        Rank(TsStd(Returns(Close, 1), 20)),
        description="Rank of trailing 20-day daily-return volatility.",
    )
    _DEFAULT_REGISTRY.register(
        "volume_z_5",
        Div(Sub(Volume, TsMean(Volume, 5)), TsStd(Volume, 5)),
        description="Trailing 5-day volume z-score.",
    )


_seed_default_registry()


__all__ = [
    "Expr",
    "Column",
    "Constant",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Delay",
    "Delta",
    "TsMean",
    "TsStd",
    "TsSum",
    "TsMax",
    "TsMin",
    "TsRank",
    "Rank",
    "Returns",
    "Abs",
    "Sign",
    "Log",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Amount",
    "Vwap",
    "FactorDefinition",
    "FactorRegistry",
    "register_factor",
    "default_registry",
    "build_factor_frame",
]
