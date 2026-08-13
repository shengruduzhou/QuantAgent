from __future__ import annotations

from pathlib import Path

import pr141_legacy_assurance_patch as patch


_original_replace_once = patch.replace_once


def _replace_with_bind_specific_rule(path: str, old: str, new: str) -> None:
    if (
        path == "src/quantagent/paper/continuous_execution.py"
        and old.startswith("    operational_state = recover(\n")
        and "_assert_recovered_account_consistent" in old
    ):
        target = Path(path)
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 2:
            raise RuntimeError(
                f"{path}: expected bind/reconcile duplicate anchor count 2, found {count}"
            )
        # bind_legacy_terminal_account is defined before reconcile_indeterminate_account;
        # only the former needs the assurance-classification variable.
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    _original_replace_once(path, old, new)


patch.replace_once = _replace_with_bind_specific_rule
patch.main()
