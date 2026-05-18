"""QuantAgent core package."""

from __future__ import annotations

import os


# Some research workstations carry optional pandas acceleration wheels compiled
# against an older NumPy ABI.  Disabling them keeps CLI imports usable while the
# canonical pandas/numpy stack still handles the DataFrame work.
os.environ.setdefault("USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

__all__ = ["__version__"]

__version__ = "0.3.0"
