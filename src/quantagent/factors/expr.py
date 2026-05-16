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

    def manifest_entry(self, backend: str, fallback: str | None = None) -> dict[str, object]:
        return {
            "factor_name": self.name,
            "expression": repr(self.expr),
            "description": self.description,
            "lookback": _expr_lookback(self.expr),
            "required_columns": sorted(_expr_required_columns(self.expr)),
            "backend": backend,
            "fallback": fallback,
            "no_lookahead_check": True,
        }


@dataclass
class FactorRegistry:
    factors: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, name: str, expr: Expr, description: str = "") -> FactorDefinition:
        definition = FactorDefinition(name=name, expr=expr, description=description)
        self.factors[name] = definition
        return definition

    def evaluate_all(self, frame: pd.DataFrame, backend: str = "pandas") -> pd.DataFrame:
        """Evaluate every registered factor and return a tidy long-format frame."""
        if not self.factors:
            return pd.DataFrame(columns=["symbol", "trade_date", "factor_name", "factor_value"])
        rows: list[pd.DataFrame] = []
        base = frame[["symbol", "trade_date"]].copy()
        for name, definition in self.factors.items():
            values = evaluate_expr(definition.expr, frame, backend=backend)
            piece = base.copy()
            piece["factor_name"] = name
            piece["factor_value"] = values.values
            rows.append(piece)
        return pd.concat(rows, ignore_index=True, sort=False)

    def evaluate_wide(self, frame: pd.DataFrame, backend: str = "pandas") -> pd.DataFrame:
        """Evaluate every registered factor as a wide ``factor_*`` frame."""
        wide = frame[["symbol", "trade_date"]].copy()
        for name, definition in self.factors.items():
            wide[f"factor_{name}"] = evaluate_expr(definition.expr, frame, backend=backend).values
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
    backend: str = "pandas",
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
    return registry.evaluate_all(frame, backend=backend) if long_format else registry.evaluate_wide(frame, backend=backend)


def build_factor_manifest(registry: FactorRegistry | None = None, backend: str = "pandas") -> list[dict[str, object]]:
    """Return serialisable factor metadata for the requested backend."""
    selected = registry or _DEFAULT_REGISTRY
    entries: list[dict[str, object]] = []
    for definition in selected.factors.values():
        fallback = "pandas_reference_for_ts_rank" if backend == "polars" and _expr_uses(definition.expr, TsRank) else None
        entries.append(definition.manifest_entry(backend=backend, fallback=fallback))
    return entries


def _expr_required_columns(expr: Expr) -> set[str]:
    if isinstance(expr, Column):
        return {expr.name}
    if isinstance(expr, (Constant,)):
        return set()
    if isinstance(expr, (Add, Sub, Mul)):
        return _expr_required_columns(expr.left) | _expr_required_columns(expr.right)
    if isinstance(expr, Div):
        return _expr_required_columns(expr.numerator) | _expr_required_columns(expr.denominator)
    if isinstance(expr, (Delay, Delta, Returns, Abs, Sign, Log, Rank, TsRank, _RollingReduction)):
        return _expr_required_columns(expr.expr)
    return set()


def _expr_lookback(expr: Expr) -> int:
    if isinstance(expr, (Delay, Delta, Returns)):
        return int(expr.periods) + _expr_lookback(expr.expr)
    if isinstance(expr, (TsRank, _RollingReduction)):
        return int(expr.window) + _expr_lookback(expr.expr)
    if isinstance(expr, (Add, Sub, Mul)):
        return max(_expr_lookback(expr.left), _expr_lookback(expr.right))
    if isinstance(expr, Div):
        return max(_expr_lookback(expr.numerator), _expr_lookback(expr.denominator))
    if isinstance(expr, (Abs, Sign, Log, Rank)):
        return _expr_lookback(expr.expr)
    return 0


def _expr_uses(expr: Expr, cls: type[Expr]) -> bool:
    if isinstance(expr, cls):
        return True
    children: list[Expr] = []
    if isinstance(expr, (Add, Sub, Mul)):
        children = [expr.left, expr.right]
    elif isinstance(expr, Div):
        children = [expr.numerator, expr.denominator]
    elif isinstance(expr, (Delay, Delta, Returns, Abs, Sign, Log, Rank, TsRank, _RollingReduction)):
        children = [expr.expr]
    return any(_expr_uses(child, cls) for child in children)


def evaluate_expr(expr: Expr, frame: pd.DataFrame, backend: str = "pandas") -> pd.Series:
    """Evaluate one expression using the requested backend."""
    if backend == "pandas":
        return expr.evaluate(frame)
    if backend == "polars":
        return _evaluate_polars(expr, frame)
    raise ValueError("factor backend must be pandas or polars")


def _evaluate_polars(expr: Expr, frame: pd.DataFrame) -> pd.Series:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("backend='polars' requires polars; install quantagent[research]") from exc
    _ensure_sorted(frame)
    base = frame.reset_index(drop=True).copy()
    base["__row_id"] = np.arange(len(base))
    pl_frame = pl.from_pandas(base)
    result = _polars_value_frame(expr, pl_frame, pl)
    aligned = result.sort("__row_id").to_pandas()["factor_value"]
    return pd.Series(aligned.to_numpy(dtype=float), index=frame.index, dtype=float)


def _polars_value_frame(expr: Expr, pl_frame, pl):
    if isinstance(expr, Column):
        if expr.name not in pl_frame.columns:
            raise KeyError(f"factor expression references missing column '{expr.name}'")
        return pl_frame.select("__row_id", pl.col(expr.name).cast(pl.Float64, strict=False).alias("factor_value"))
    if isinstance(expr, Constant):
        return pl_frame.select("__row_id", pl.lit(float(expr.value)).alias("factor_value"))
    if isinstance(expr, Add):
        return _polars_binary(expr.left, expr.right, pl_frame, pl, "add")
    if isinstance(expr, Sub):
        return _polars_binary(expr.left, expr.right, pl_frame, pl, "sub")
    if isinstance(expr, Mul):
        return _polars_binary(expr.left, expr.right, pl_frame, pl, "mul")
    if isinstance(expr, Div):
        return _polars_binary(expr.numerator, expr.denominator, pl_frame, pl, "div")
    if isinstance(expr, Delay):
        return _polars_grouped(expr.expr, pl_frame, pl, pl.col("factor_value").shift(expr.periods))
    if isinstance(expr, Delta):
        current = _polars_value_frame(expr.expr, pl_frame, pl).rename({"factor_value": "left"})
        shifted = _polars_grouped(expr.expr, pl_frame, pl, pl.col("factor_value").shift(expr.periods)).rename({"factor_value": "right"})
        return current.join(shifted, on="__row_id").select("__row_id", (pl.col("left") - pl.col("right")).alias("factor_value"))
    if isinstance(expr, Returns):
        current = _polars_value_frame(expr.expr, pl_frame, pl).rename({"factor_value": "left"})
        shifted = _polars_grouped(expr.expr, pl_frame, pl, pl.col("factor_value").shift(expr.periods)).rename({"factor_value": "right"})
        return current.join(shifted, on="__row_id").select(
            "__row_id",
            ((pl.col("left") / pl.when(pl.col("right") == 0.0).then(None).otherwise(pl.col("right"))) - 1.0).alias("factor_value"),
        )
    if isinstance(expr, _RollingReduction):
        rolling_expr = {
            "mean": pl.col("factor_value").rolling_mean(window_size=expr.window, min_periods=expr.window),
            "std": pl.col("factor_value").rolling_std(window_size=expr.window, min_periods=expr.window),
            "sum": pl.col("factor_value").rolling_sum(window_size=expr.window, min_periods=expr.window),
            "max": pl.col("factor_value").rolling_max(window_size=expr.window, min_periods=expr.window),
            "min": pl.col("factor_value").rolling_min(window_size=expr.window, min_periods=expr.window),
        }[expr.op]
        return _polars_grouped(expr.expr, pl_frame, pl, rolling_expr)
    if isinstance(expr, TsRank):
        # Keep the backend surface deterministic across Polars versions;
        # TsRank is evaluated with the pandas reference while other core
        # rolling reductions use native Polars expressions.
        values = expr.evaluate(pl_frame.to_pandas())
        return pl.DataFrame({"__row_id": pl_frame["__row_id"], "factor_value": values.to_numpy(dtype=float)})
    if isinstance(expr, Rank):
        values = _polars_value_frame(expr.expr, pl_frame, pl)
        keyed = pl_frame.select("__row_id", "trade_date").join(values, on="__row_id")
        return keyed.with_columns(
            pl.col("factor_value").rank(method=expr.method).over("trade_date").alias("__rank"),
            pl.len().over("trade_date").alias("__count"),
        ).select("__row_id", (pl.col("__rank") / pl.col("__count")).alias("factor_value"))
    if isinstance(expr, Abs):
        values = _polars_value_frame(expr.expr, pl_frame, pl)
        return values.select("__row_id", pl.col("factor_value").abs().alias("factor_value"))
    if isinstance(expr, Sign):
        values = _polars_value_frame(expr.expr, pl_frame, pl)
        return values.select("__row_id", pl.col("factor_value").sign().alias("factor_value"))
    if isinstance(expr, Log):
        values = _polars_value_frame(expr.expr, pl_frame, pl)
        return values.select("__row_id", pl.col("factor_value").clip(1e-12, None).log().alias("factor_value"))
    raise TypeError(f"unsupported polars factor expression: {type(expr).__name__}")


def _polars_binary(left: Expr, right: Expr, pl_frame, pl, op: str):
    ldf = _polars_value_frame(left, pl_frame, pl).rename({"factor_value": "left"})
    rdf = _polars_value_frame(right, pl_frame, pl).rename({"factor_value": "right"})
    joined = ldf.join(rdf, on="__row_id")
    if op == "add":
        expr = pl.col("left") + pl.col("right")
    elif op == "sub":
        expr = pl.col("left") - pl.col("right")
    elif op == "mul":
        expr = pl.col("left") * pl.col("right")
    elif op == "div":
        expr = pl.col("left") / pl.when(pl.col("right") == 0.0).then(None).otherwise(pl.col("right"))
    else:  # pragma: no cover
        raise ValueError(op)
    return joined.select("__row_id", expr.alias("factor_value"))


def _polars_grouped(inner: Expr, pl_frame, pl, value_expr):
    values = _polars_value_frame(inner, pl_frame, pl)
    keyed = pl_frame.select("__row_id", "symbol", "trade_date").join(values, on="__row_id")
    return keyed.sort(["symbol", "trade_date", "__row_id"]).with_columns(
        value_expr.over("symbol").alias("factor_value")
    ).select("__row_id", "factor_value")


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
    "build_factor_manifest",
    "evaluate_expr",
]
