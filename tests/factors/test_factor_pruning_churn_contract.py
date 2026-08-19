"""Decile churn must be measured by symbol, not by row position.

Round 24, and this pins a bug I wrote and had to catch. The turnover term in
`scripts/prune_and_cost_price_factors.py` compared decile membership between
consecutive sessions to price the cost of holding a factor. The first version
compared *row indices*.

The same symbol occupies a different row on every session, so the intersection
between one day's decile and the next was empty by construction, and every
factor reported a churn of exactly 1.0 -- a complete daily reshuffle of roughly
580 names. That is not a plausible number for any real factor, which is the only
reason it got caught: the arithmetic was fine and nothing raised.

The cost of the mistake was not cosmetic. At churn 1.0 the cost term was ~3x
too large and the count of factors with a positive net decile spread read 35;
measured by symbol it is 65. The bug would have retired 30 factors that pay for
their own trading.

The lesson is the repository's recurring one in a new place: a measurement that
produces a plausible-looking number while measuring the wrong quantity survives
every internal consistency check, because it is internally consistent.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path("scripts/prune_and_cost_price_factors.py")


def test_script_exists() -> None:
    assert SCRIPT.exists()


def test_churn_is_computed_over_symbol_identity() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "symbol_codes" in source, "churn must key on symbol identity"
    # The membership sets that get intersected must be built from symbol codes.
    assert re.search(r"top\s*=\s*set\(symbol_codes\[", source)
    assert re.search(r"bot\s*=\s*set\(symbol_codes\[", source)


def test_returns_still_use_row_positions() -> None:
    """Returns are per-row; only the identity comparison is per-symbol."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "label[top_rows]" in source
    assert "label[bot_rows]" in source


def test_the_row_index_trap_is_documented_where_it_happened() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "different row index every session" in source


def test_cost_charges_both_legs() -> None:
    """A long-short spread trades two books, so churn is paid twice."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "2.0 * churn" in source


def test_evaluation_stays_inside_the_tradable_domain() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "entry_feasible" in source
