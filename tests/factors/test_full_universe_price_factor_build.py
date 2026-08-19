"""The certified panel's price-factor build must stay batch-free and aligned.

Round 23. The certified full-universe panel carried fifteen features and zero
Alpha101 / GTJA-191 columns, which is why the workstation read as a
technical-analysis system. `scripts/build_full_universe_price_factors.py`
closes the half of that gap which is not blocked by data.

Two computation rules are load-bearing, and both encode measured defects rather
than theory:

* **No symbol batching.** Alpha101 is full of cross-sectional `rank` operators,
  so a symbol subset makes every rank subset-local. That silently corrupted 22
  alpha columns across a whole sleeve when a `--batch-symbols 300` flag existed,
  and the columns looked normal afterwards.
* **No history truncation.** The longest lookback here is 250 sessions.
  Measured at 145d against 390d of warmup, 58 of 101 alphas differed by more
  than 1e-9 and alpha072 differed by 396.8 -- with no error raised.

The artifacts live under `runtime/` and are gitignored, so the artifact-level
assertions skip when they are absent rather than failing a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT = Path("scripts/build_full_universe_price_factors.py")
GOLD = Path("runtime/data/gold/full_universe")
MANIFESTS = Path("runtime/data/gold/manifests")


def test_build_script_exists() -> None:
    assert SCRIPT.exists()


def test_script_refuses_to_offer_a_production_symbol_batch() -> None:
    """A symbol-batching flag is the exact shape of the v8.8 rank corruption."""
    source = SCRIPT.read_text(encoding="utf-8")
    # Look for a registered FLAG, not any mention: the module docstring cites
    # `--batch-symbols 300` by name when explaining the defect it must avoid,
    # and losing that explanation would be worse than the naive check is worth.
    assert 'add_argument("--batch-symbols"' not in source
    assert "add_argument('--batch-symbols'" not in source
    # The one subsetting option that exists is explicitly labelled a probe and
    # writes to a separate filename, so its output cannot be mistaken for the
    # production artifact.
    assert "--symbol-limit" in source
    assert "_probe" in source
    assert "SCALE PROBE ONLY" in source


def test_script_documents_both_computation_rules() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Never batch by symbol" in source
    assert "Never truncate per-symbol history" in source


@pytest.mark.parametrize("family,expected_total", [("alpha101", 101), ("gtja191", 64)])
def test_manifest_records_usable_count_and_lookahead_result(family, expected_total) -> None:
    path = MANIFESTS / f"full_universe_factors_{family}.json"
    if not path.exists():
        pytest.skip(f"{path} not built in this checkout")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    extra = manifest["extra"]

    assert manifest["row_count"] == 10_917_401
    assert manifest["column_count"] == expected_total + 2  # + symbol, trade_date
    assert extra["adjustment_basis"].startswith("hfq")
    assert extra["lookahead_check"].startswith("PASS")
    # Usable count must exclude the structurally-NaN placeholders rather than
    # advertising the raw column count as available signal.
    assert extra["usable_factor_count"] == expected_total - len(extra["all_nan_columns"])


def test_all_nan_columns_are_explained_not_merely_listed() -> None:
    """A column that is always NaN must say why, or it reads as a broken factor."""
    path = MANIFESTS / "full_universe_factors_alpha101.json"
    if not path.exists():
        pytest.skip("alpha101 manifest not built in this checkout")
    extra = json.loads(path.read_text(encoding="utf-8"))["extra"]

    assert extra["all_nan_columns"], "alpha101 ships IndClass/cap placeholders"
    assert extra["all_nan_reason"], "an always-NaN column without a reason is a bug report"
    assert "IndClass" in extra["all_nan_reason"]


def test_factor_artifacts_align_row_for_row_with_the_panel() -> None:
    panel = GOLD / "dataset.parquet"
    if not panel.exists():
        pytest.skip("certified panel not present in this checkout")
    pq = pytest.importorskip("pyarrow.parquet")

    expected = pq.ParquetFile(panel).metadata.num_rows
    for family in ("alpha101", "gtja191"):
        artifact = GOLD / f"factors_{family}.parquet"
        if not artifact.exists():
            pytest.skip(f"{artifact} not built in this checkout")
        assert pq.ParquetFile(artifact).metadata.num_rows == expected, (
            f"{family} row count diverged from the panel it must join to"
        )
