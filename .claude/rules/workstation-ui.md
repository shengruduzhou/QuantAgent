---
paths:
  - "apps/quant-ui/**"
  - "services/quant_api/**"
---

# Workstation rules

- VNext only (`apps/quant-ui/src/vnext`); semantic tokens from
  `src/vnext/styles/tokens.css`; charts through the `EChart` adapter.
- Every displayed number traces to an artifact; missing stays missing with an
  explanation and next action. No mock metrics, no trading recommendations.
- After frontend changes: `npx tsc --noEmit && npx vitest run && npx vite build`
  (the API serves the built `dist/`).
- New executable capability = allowlisted command in
  `services/quant_api/services/jobs.py` with path bounds, credential provider and tests.
  No arbitrary shell. Credentials never in URL/argv/log/runtime/localStorage.
