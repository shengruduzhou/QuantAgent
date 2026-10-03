"""Clean-room backtest, isolated from the legacy stack by construction.

This package deliberately imports NOTHING from ``quantagent.backtest``,
``quantagent.ensemble``, ``quantagent.training`` or ``quantagent.rl``. Those
paths carry defects measured in this repository that silently manufacture
performance, and re-using any of them would re-import the thing this rebuild
exists to escape:

* ``backtest/engine.py`` marked NAV under ``dates[i]`` while filling at
  ``dates[i+1]``, so the reported drawdown/vol/Sharpe path belonged to a
  portfolio that was not held (DEF-038, since closed: ``NAV(t)`` now contains
  exactly the fills with ``fill_date <= t``). This engine remains the
  clock-correct reference that ``tests/backtest/test_clean_room_engine.py`` pins.
* ``backtest/strict_v8.py`` reported ``max_drawdown=0.0``/``sharpe=0.0`` for
  backtests that never ran (fixed, but the metric layer is a separate copy).
* Tradability flags were fabricated as ``False`` at seven call sites, so
  suspended and limit-locked names filled freely.
* The factor-promotion gate discarded the label horizon, promoting ~21% of pure
  noise at h=20.

The rule for this package: every number it reports must be traceable to a bar
that existed, at a time it was knowable, for a name that could actually be
traded. Where that cannot be established, it reports ``None``/NaN rather than a
plausible default -- the failure mode that produced every defect above.
"""

from quantagent.clean_room.engine import CleanRoomResult, run_backtest

# ``dataset.py`` and ``risk.py`` were deleted in round 29: nothing imported them
# except this package's own re-export.
__all__ = [
    "CleanRoomResult",
    "run_backtest",
]
