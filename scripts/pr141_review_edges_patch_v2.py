from __future__ import annotations

from pathlib import Path

import pr141_review_edges_patch as patch


patch.main()

# The source patch is unchanged; correct two focused-test wiring mistakes in the
# generated worktree before pytest runs.
p = Path("tests/paper/test_daily_loop_pending_execution.py")
text = p.read_text(encoding="utf-8")
old = 'with pytest.raises(PaperAccountStateRefused, match="no-target predictions must belong exactly"):'
new = 'with pytest.raises(daily_loop.PaperAccountStateRefused, match="no-target predictions must belong exactly"):'
if text.count(old) != 1:
    raise RuntimeError(f"no-target exception anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

p = Path("tests/test_v7_target_initial_weights.py")
text = p.read_text(encoding="utf-8")
old = '''        config=_config(\n            holding_period_mode="hard",\n            position_state_path=str(state_path),\n            max_turnover=1.0,\n        ),\n        timing_plan=timing,\n        initial_weights=pd.Series(dtype=float),\n'''
new = '''        config=_config(\n            holding_period_mode="hard",\n            max_turnover=1.0,\n        ),\n        timing_plan=timing,\n        position_state_path=state_path,\n        initial_weights=pd.Series(dtype=float),\n'''
if text.count(old) != 1:
    raise RuntimeError(f"position-state argument anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
