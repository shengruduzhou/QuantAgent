"""Configuration helpers for QuantAgent V7.

The most important entry point here is :mod:`quantagent.config.paths`, which
resolves the canonical storage layout for the large real-data assets (Qlib
panels, AkShare snapshots, model checkpoints, predictions, target weights,
backtest reports, logs). By default everything lives under ``E:\\Project\\QuantAgent\\runtime\\``
on Windows and ``~/AI_quant`` on POSIX systems; both can be overridden
with a single environment variable ``QUANTAGENT_HOME``.

Keeping path resolution centralised lets the CLI, bootstrap modules and
training scripts agree on where data lives without each command re-deriving
that layout.
"""

from quantagent.config.paths import (
    DEFAULT_QUANT_HOME_ENV,
    QuantPaths,
    quant_paths,
    resolve_quant_home,
)

__all__ = [
    "DEFAULT_QUANT_HOME_ENV",
    "QuantPaths",
    "quant_paths",
    "resolve_quant_home",
]
