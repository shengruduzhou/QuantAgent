# QuantAgent Institutional Workbench — Design QA

## Reference and comparison

- Primary reference: `docs/quant_ui_design_concepts/02-research-workbench.png` (`1487 × 1058`).
- Supporting references: `01-institutional-command-grid.png` for dense metrics/exposure/blotter patterns and `03-signal-observatory.png` for model health, selection funnel, risk, and evidence patterns.
- Implementation state: `/strategy`, cloud-browser viewport `1365 × 936`, VNext `dawn` theme.
- Comparison method: the primary reference and the implementation screenshot were inspected together in one visual comparison pass after the final CSS and interaction changes.

The implementation intentionally keeps the reference's compact shell, permanent module rail, command bar, workspace tabs, dense metric strip, central operating canvas, right-side inspector, and persistent operations dock. Strategy configuration replaces the reference's stock replay chart because the page's primary job is governed research configuration, but the information hierarchy and operator workflow remain equivalent.

## Visible QA findings

| Severity | Finding | Resolution | Result |
|---|---|---|---|
| P0 | Strategy form allowed a `primaryHorizon` outside the declared horizon set. | Replaced the arbitrary numeric control with a value constrained to the active horizon set; legacy drafts normalize before validation. | Passed |
| P0 | API errors exposed an entire FastAPI input payload and obscured the actionable field. | API client now strips `input`, extracts field paths, and displays compact repair guidance. | Passed |
| P0 | Strategy API fields did not all reach the formal CLI signature. | Restored the full API → job allowlist → CLI contract and added signature/serialization tests. | Passed |
| P1 | Dashboard was almost blank when Quant API was unavailable. | Added an honest recovery workstation with actions for tmux startup, Runtime, tasks, and connectors. | Passed |
| P1 | Day/dawn themes inherited fixed dark colors on legacy pages. | Added semantic token aliases and theme-aware charts, code blocks, controls, status bars, and T+1 states. | Passed |
| P1 | Task-tab ellipsis actions were effectively hidden and the menu could overflow. | Made the trigger discoverable, bounded the menu, added a header, Escape/outside-click handling, and explicit pin/duplicate/split/close actions. | Passed |
| P1 | Settings could say `API ready` while the API request had failed; connectors could spin indefinitely. | Status now reports `API unavailable`, and connectors render a truthful unavailable state. | Passed |
| P1 | Training terminology made the V7 pipeline and V8 deep model look like competing versions. | Training page now identifies `V8 深度模型 · GPU` and keeps `train-v8-deep` as the command identifier; strategy copy separates the V7 pipeline/baseline role. | Passed |
| P2 | Small labels and mixed theme contrast reduced legibility. | Raised critical text floors, strengthened muted text, and normalized input, table, console, and action colors. | Passed |
| P2 | T+1 evidence canvas used excessive empty space and fixed chart colors. | Reduced empty height and made axes, tooltips, grids, and A-share red-up/green-down semantics theme-aware. | Passed |

## Interaction coverage

- Global shell: module rail, command search, workspace tabs, visible tab menu, pin/duplicate/split/close controls, theme menu, compact density, operations dock.
- Dashboard: API-offline recovery state and four operator recovery actions.
- Strategy: factor library, constrained horizons, Top-K search, automatic/fixed/off fundamental modes, selection ablation, screening order, T+1 mode, locked GPU mode, Pareto objective weights, validation/save/launch/cancel controls, Human Gate, council and telemetry states.
- Factor lab: catalog filters, full-factor evaluation entry, discovery entry, 12-stage evidence lifecycle, inspector empty state.
- Training lab: experiment/run navigator, configuration summary/editor, GPU-required state, metrics tabs, validation/start/cancel/clone controls, lineage and console states.
- Task center: tmux command, job templates, allowlisted JSON, launch feedback, queue, stop, delete, connection vault, API-unavailable state.
- T+1: inventory/fill evidence chain, failure thresholds, next actions, no-artifact state.
- Resources: CPU/RAM/Runtime facts, GPU telemetry graph contract, and explicit fail-closed unavailable state.

## Theme and accessibility checks

- Verified `night`, `dawn`, and `day` theme selection and persisted `data-theme`.
- Day and dawn states retain readable headings, body text, code, warnings, badges, and form controls.
- Primary controls remain keyboard reachable; tab menus expose `aria-expanded`, close on Escape/outside click, and keep destructive actions visually distinct.
- A-share market semantics use red for positive moves and green for negative moves where direction is encoded.

## Console and environment

- No application-origin uncaught errors were observed during the final page traversal.
- Cloud preview cannot reach the local Quant API and therefore correctly renders `API ERROR` / unavailable states. The only browser-console errors were emitted by the browser metadata extension, not by QuantAgent.
- Backend actions were validated separately through real FastAPI/CLI/component tests; unavailable preview data was never replaced with fabricated metrics.

## Final verification

- Python full suite: `1920 passed, 51 skipped`.
- Frontend final check: `17` files and `43` tests passed.
- TypeScript and the final Vite production build passed.
