"""Regression lock for the delay-1 EXECUTABLE label convention.

EVALUATOR_VALIDITY_AUDIT_IC016.md (2026-07-10) verified empirically that the
production training labels are

    forward_return_{h}d = close(t+1+h) / close(t+1) - 1        (delay-1)

with entry-infeasible rows dropped (is_st(t) | is_suspended(t) |
is_suspended(t+1) | is_limit_up(t+1)) — per scripts/build_executable_labels_
dataset.py. NOTE: quantagent.data.v7_label_builder documents the OLDER
close(t)->close(t+h) convention; the production artifact does NOT use it.
These tests pin the executable convention so a silent rebuild that reverts to
same-day labels (re-introducing sealed-limit-up phantom alpha) fails loudly.

1. Hermetic: run the builder on a synthetic panel, assert values + filters.
2. Artifact spot-check: sample the production dataset (skipped if absent).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts/build_executable_labels_dataset.py"
PROD = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"


def _synthetic_frames(tmp_path: Path) -> tuple[Path, Path]:
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows, flags = [], []
    rng = np.random.default_rng(3)
    for sym in ("AAA.SZ", "BBB.SZ"):
        px = 10.0 + np.cumsum(rng.normal(0, 0.1, len(dates)))
        for i, d in enumerate(dates):
            rows.append({"symbol": sym, "trade_date": d, "close": round(float(px[i]), 4),
                         "feat_x": float(i)})
            flags.append({"symbol": sym, "trade_date": d,
                          "is_st": False,
                          # BBB suspended on days 15-17
                          "is_suspended": sym == "BBB.SZ" and i in (15, 16, 17),
                          # AAA limit-up on day 10
                          "is_limit_up": sym == "AAA.SZ" and i == 10,
                          "is_limit_down": False})
    inp, pan = tmp_path / "input.parquet", tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(inp, index=False)
    pd.DataFrame(flags).to_parquet(pan, index=False)
    return inp, pan


def test_builder_emits_delay1_executable_labels(tmp_path):
    inp, pan = _synthetic_frames(tmp_path)
    out = tmp_path / "out.parquet"
    r = subprocess.run(
        [sys.executable, str(BUILDER), "--input", str(inp), "--panel", str(pan),
         "--output", str(out), "--horizons", "1,5"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    assert not df.duplicated(["symbol", "trade_date"]).any()

    src = pd.read_parquet(inp).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = src.groupby("symbol", sort=False)
    for h in (1, 5):
        src[f"want_{h}"] = g["close"].shift(-(1 + h)) / g["close"].shift(-1) - 1.0
        src[f"want_end_{h}"] = g["trade_date"].shift(-(1 + h))
    m = df.merge(src, on=["symbol", "trade_date"], suffixes=("", "_src"))
    for h in (1, 5):
        got, want = m[f"forward_return_{h}d"], m[f"want_{h}"]
        ok = got.notna() & want.notna()
        assert ok.any()
        # float32 storage => compare at float32 resolution
        assert np.allclose(got[ok], want[ok], atol=1e-6), f"h={h} not delay-1 executable"
        e_ok = m[f"label_end_{h}d"].notna() & m[f"want_end_{h}"].notna()
        assert (pd.to_datetime(m.loc[e_ok, f"label_end_{h}d"]).values
                == m.loc[e_ok, f"want_end_{h}"].values).all()

    # entry filter: day 9 of AAA dropped (limit_up at t+1=day10);
    # days 14-17 of BBB dropped (suspended t or t+1)
    dates = pd.bdate_range("2024-01-02", periods=30)
    aaa = set(pd.to_datetime(df[df["symbol"] == "AAA.SZ"]["trade_date"]))
    bbb = set(pd.to_datetime(df[df["symbol"] == "BBB.SZ"]["trade_date"]))
    assert dates[9] not in aaa
    assert dates[10] in aaa  # limit-up day itself is a valid entry the NEXT day
    for i in (14, 15, 16, 17):
        assert dates[i] not in bbb
    assert dates[13] in bbb


@pytest.mark.skipif(not PROD.exists(), reason="production dataset not on disk")
def test_production_dataset_labels_are_delay1():
    cols = ["symbol", "trade_date", "close", "forward_return_1d"]
    head = pd.read_parquet(PROD, columns=["symbol"]).drop_duplicates()
    syms = head["symbol"].sample(5, random_state=11).tolist()
    df = pd.read_parquet(PROD, columns=cols, filters=[("symbol", "in", syms)])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    want = g["close"].shift(-2) / g["close"].shift(-1) - 1.0
    ok = df["forward_return_1d"].notna() & want.notna()
    # entry-filtered rows break the row-shift arithmetic for their neighbours;
    # the convention still holds for the overwhelming majority
    match = np.isclose(df.loc[ok, "forward_return_1d"], want[ok], atol=1e-6).mean()
    assert match >= 0.95, f"production labels no longer delay-1 executable (match={match:.3f})"
