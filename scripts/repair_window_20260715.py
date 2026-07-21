#!/usr/bin/env python3
"""Targeted repair pass for the 2026-07-03..2026-07-14 panel append (day-1
shadow catch-up left 1,431 rate-limited symbols). Reuses the proven
repair_fresh_window_20260704 machinery with window constants overridden —
same retry/backoff, surgical insert, flag-rebuild semantics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repair_fresh_window_20260704 as rep  # noqa: E402

rep.WIN_START = pd.Timestamp("2026-07-03")
rep.WIN_END = pd.Timestamp("2026-07-14")
rep.SEED_DATE = pd.Timestamp("2026-07-02")  # last fully-covered pre-append day

if __name__ == "__main__":
    raise SystemExit(rep.main())
