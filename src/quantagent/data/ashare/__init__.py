"""A-share data foundation: canonical identity, contracts, providers, routing.

This package is the single place where market data enters QuantAgent's
full-universe (U0) pipeline. Its guarantees:

* every row carries provider provenance and a point-in-time ``available_at``;
* no code path fabricates or interpolates market data — a provider that cannot
  answer produces a classified failure, never an empty-looking success;
* vendor units are normalised at the boundary and declared in the contract;
* when a symbol's history changes provider, the seam is recorded as an explicit
  :class:`~quantagent.data.ashare.contracts.SourceBoundary`.
"""

from quantagent.data.ashare.contracts import CONTRACTS, DatasetContract, SourceBoundary
from quantagent.data.ashare.symbols import (
    ALL_BOARDS,
    SecurityIdentity,
    SymbolError,
    board_of,
    canonical_symbol,
    classify_code,
    identify,
)

__all__ = [
    "ALL_BOARDS",
    "CONTRACTS",
    "DatasetContract",
    "SecurityIdentity",
    "SourceBoundary",
    "SymbolError",
    "board_of",
    "canonical_symbol",
    "classify_code",
    "identify",
]
