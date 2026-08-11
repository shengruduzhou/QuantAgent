"""Regression lock for the canonical executable global-session label clock.

Production labels are signal dated.  A signal formed at close(T) can enter only
on the next *global exchange session* T+1 and a horizon-h outcome ends on global
session T+1+h.  Missing symbol bars never redefine that clock to "the next row
for this symbol".

These tests deliberately provide an independent calendar artifact.  They also
pin fail-closed execution flags and the historic ``label_end_{h}d`` compatibility
columns used by purged/WFA consumers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts/build_executable_labels_dataset.py"


def _synthetic_frames(tmp_path: Path) -> tuple[Path, Path, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2024-01-02", periods=30)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(3)
    for sym in ("AAA.SZ", "BBB.SZ"):
        px = 10.0 + np.cumsum(rng.normal(0, 0.1, len(sessions)))
        for i, date in enumerate(sessions):
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": date,
                    "close": round(float(px[i]), 4),
                    "feat_x": float(i),
                    "is_st": False,
                    "is_suspended": sym == "BBB.SZ" and i in (15, 16, 17),
                    "is_limit_up": sym == "AAA.SZ" and i == 10,
                    "is_limit_down": False,
                }
            )
    inp = tmp_path / "input.parquet"
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame(rows).to_parquet(inp, index=False)
    pd.DataFrame({"trade_date": sessions, "is_trading_day": True}).to_parquet(
        calendar, index=False
    )
    return inp, calendar, sessions


def _expected_exact_return(
    source: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    source = source.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    price = source.set_index(["symbol", "trade_date"])["close"]
    position = {session: i for i, session in enumerate(sessions)}
    values: list[float] = []
    ends: list[pd.Timestamp | pd.NaT] = []
    for row in source.itertuples(index=False):
        pos = position[pd.Timestamp(row.trade_date)]
        if pos + 1 + horizon >= len(sessions):
            values.append(np.nan)
            ends.append(pd.NaT)
            continue
        entry_date = sessions[pos + 1]
        end_date = sessions[pos + 1 + horizon]
        entry = price.get((str(row.symbol), entry_date), np.nan)
        end = price.get((str(row.symbol), end_date), np.nan)
        values.append(float(end / entry - 1.0) if np.isfinite(entry) and np.isfinite(end) and entry > 0 else np.nan)
        ends.append(end_date)
    return pd.Series(values, index=source.index, dtype=float), pd.Series(ends, index=source.index)


def test_builder_emits_global_session_executable_labels(tmp_path):
    inp, calendar, sessions = _synthetic_frames(tmp_path)
    out = tmp_path / "out.parquet"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--input",
            str(inp),
            "--market-calendar",
            str(calendar),
            "--output",
            str(out),
            "--horizons",
            "1,5",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    df = pd.read_parquet(out).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    src = pd.read_parquet(inp).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    assert len(df) == len(src)
    assert not df.duplicated(["symbol", "trade_date"]).any()

    for horizon in (1, 5):
        want, want_end = _expected_exact_return(src, sessions, horizon=horizon)
        got = pd.to_numeric(df[f"forward_return_{horizon}d"], errors="coerce")

        # Tradability is a mask on outcomes, not a row deletion.  Where the row
        # is executable, the numeric outcome must match the exact global clock.
        valid = df["_execution_tradable"].astype(bool) & got.notna() & want.notna()
        assert valid.any()
        assert np.allclose(got[valid], want[valid], atol=1e-12)

        got_end = pd.to_datetime(df[f"label_end_{horizon}d"])
        canonical_end = pd.to_datetime(df[f"factor_label_end_{horizon}d"])
        assert got_end.equals(canonical_end)
        end_valid = got_end.notna() & want_end.notna()
        assert (got_end[end_valid].to_numpy() == pd.to_datetime(want_end[end_valid]).to_numpy()).all()

    # AAA signal day 9 maps to entry day 10, which is limit-up -> fail closed.
    aaa9 = df[(df["symbol"] == "AAA.SZ") & (df["trade_date"] == sessions[9])].iloc[0]
    assert not bool(aaa9["_entry_tradable"])
    assert pd.isna(aaa9["forward_return_1d"])

    # AAA signal day 10 itself can execute on global session day 11.
    aaa10 = df[(df["symbol"] == "AAA.SZ") & (df["trade_date"] == sessions[10])].iloc[0]
    assert bool(aaa10["_entry_tradable"])

    # BBB suspension on either signal or mapped entry session fails closed.
    for i in (14, 15, 16, 17):
        row = df[(df["symbol"] == "BBB.SZ") & (df["trade_date"] == sessions[i])].iloc[0]
        assert not bool(row["_execution_tradable"])
        assert pd.isna(row["forward_return_1d"])
    bbb13 = df[(df["symbol"] == "BBB.SZ") & (df["trade_date"] == sessions[13])].iloc[0]
    assert bool(bbb13["_execution_tradable"])

    manifest = json.loads(out.with_suffix(".labels.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "quantagent.training-executable-labels.v3_global_sessions"
    assert manifest["market_calendar"]["session_count"] == len(sessions)
    assert manifest["market_calendar"]["market_session_schedule_sha256"] == manifest["market_session_schedule_sha256"]


def test_missing_exact_symbol_bar_does_not_shift_execution_clock(tmp_path):
    inp, calendar, sessions = _synthetic_frames(tmp_path)
    source = pd.read_parquet(inp)
    # Remove AAA's exact T+1 bar for a signal at T=session[4].  AAA still has a
    # later bar, so a per-symbol row-shift implementation would incorrectly use it.
    source = source[
        ~((source["symbol"] == "AAA.SZ") & (pd.to_datetime(source["trade_date"]) == sessions[5]))
    ]
    source.to_parquet(inp, index=False)
    out = tmp_path / "missing_bar.parquet"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--input",
            str(inp),
            "--market-calendar",
            str(calendar),
            "--output",
            str(out),
            "--horizons",
            "1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_parquet(out)
    row = df[(df["symbol"] == "AAA.SZ") & (df["trade_date"] == sessions[4])].iloc[0]
    assert pd.Timestamp(row["factor_label_entry_date"]) == sessions[5]
    assert not bool(row["factor_label_entry_observed"])
    assert not bool(row["_entry_tradable"])
    assert pd.isna(row["forward_return_1d"])
