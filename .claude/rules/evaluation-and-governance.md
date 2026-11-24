---
paths:
  - "configs/quarantined_windows.json"
  - "src/quantagent/backtest/**"
  - "src/quantagent/quant_math/**"
  - "src/quantagent/factors/evaluation*.py"
  - "src/quantagent/research/**"
  - "src/quantagent/fusion/**"
  - "scripts/baseline_protocol.py"
  - "docs/GOVERNANCE.md"
  - "docs/research/**"
---

# Evaluation and governance rules

- `configs/quarantined_windows.json` changes only after PIT rebuild + QC + a
  pre-registration, and with explicit user approval. Never as an override.
- Thresholds in `docs/GOVERNANCE.md` and pre-registrations are never relaxed after a
  result is seen. A failing gate is a research conclusion (`rejected`, exit 3), not a bug.
- Changes to costs, impact law, metrics conventions, label conventions or rank-IC
  semantics alter canonical numbers: pre-register, re-baseline, get `math-stat-auditor`
  review; cost/impact-law changes also need user approval.
- DSR uses the family's cumulative trial count; `n_trials` is never user-settable.
- Missing measurements are `unknown`, never `pass`; no `fillna(0)` / default on a
  reported-number path without an explicit, tested reason.
