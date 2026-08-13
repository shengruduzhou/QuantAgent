from __future__ import annotations

from pathlib import Path

import pr141_final_atomic_patch as patch


patch.main()

# Source verifier uses pathlib.Path for the marker-bound summary.
p = Path("src/quantagent/paper/continuous_execution.py")
text = p.read_text(encoding="utf-8")
if "from pathlib import Path\n" not in text:
    anchor = "from dataclasses import dataclass, replace\n"
    if text.count(anchor) != 1:
        raise RuntimeError(f"continuous Path import anchor count={text.count(anchor)}")
    text = text.replace(anchor, anchor + "from pathlib import Path\n", 1)
p.write_text(text, encoding="utf-8")

# Focused missing-summary regression also constructs Path directly.
p = Path("tests/paper/test_continuous_pending_execution.py")
text = p.read_text(encoding="utf-8")
if "from pathlib import Path\n" not in text:
    anchor = "from dataclasses import replace\n"
    if text.count(anchor) != 1:
        raise RuntimeError(f"test Path import anchor count={text.count(anchor)}")
    text = text.replace(anchor, anchor + "from pathlib import Path\n", 1)
p.write_text(text, encoding="utf-8")

# Direct marker unit test now stages the required primary summary first.
p = Path("tests/paper/test_daily_decision_and_legacy_binding.py")
text = p.read_text(encoding="utf-8")
old = '''    prefix = build_canonical_prefix_index(config.canonical_ledger_path)\n    _freeze_daily_decision(\n        config,\n        "2026-08-11",\n        decision_kind="no_target",\n        paper_account_identity_sha256="f" * 64,\n        account_evidence={\n            "account_state_sha256": "a" * 64,\n            "canonical_records": prefix.record_count,\n            "canonical_head_hash": prefix.current_head,\n        },\n    )\n'''
new = '''    prefix = build_canonical_prefix_index(config.canonical_ledger_path)\n    summary_path = daily_loop._write_daily_summary(\n        Path(config.output_root) / "2026-08-11",\n        {\n            "daily_decision_commit_protocol": daily_loop.DAILY_SUMMARY_COMMIT_PROTOCOL,\n            "status": "no_target_generated",\n        },\n    )\n    summary_sha = daily_loop._file_sha256(summary_path)\n    _freeze_daily_decision(\n        config,\n        "2026-08-11",\n        decision_kind="no_target",\n        paper_account_identity_sha256="f" * 64,\n        account_evidence={\n            "account_state_sha256": "a" * 64,\n            "canonical_records": prefix.record_count,\n            "canonical_head_hash": prefix.current_head,\n        },\n        daily_summary_path=summary_path,\n        daily_summary_sha256=summary_sha,\n    )\n'''
if text.count(old) != 1:
    raise RuntimeError(f"direct freeze test anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
