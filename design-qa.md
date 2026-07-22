# QuantAgent Institutional Workstation — Design QA

## Scope and sources

- Product baseline: the existing Decision Dashboard and Training Lab anatomy in this repository.
- User references: the supplied Decision Dashboard, Training Lab and Model Registry captures retained in the private review session; they are intentionally not copied into the public repository.
- Reported defects also include the supplied factor-page empty canvas and VN.PY parity table text-collision captures from the same review session.
- Captured implementation state: Factor Intelligence Studio in the cloud browser at 1363 × 936, night theme, expanded rail, open Operations Dock. The verified final tab remains open for inspection.

## Visual contract

- Shared page anatomy: Workbench header → no more than six source-backed metrics → primary canvas → evidence/operations inspector → global Operations Dock.
- Shared semantic colors: information blue, research cyan/violet, verified mint, warning amber and risk coral. Night, dawn and day use the same meanings.
- Dense surfaces use bounded columns, truncation, expandable detail and inspectors instead of allowing table text to overlap.
- Empty states state the missing artifact, the safety implication and a real next action. No fabricated data fills gaps.
- The visible system remains research/paper-only; LLM, network, registry promotion, training and execution remain distinct gates.

## Browser verification

| Area | State and interaction checked | Result |
|---|---|---|
| Factor Intelligence | Empty and populated anatomy; six metrics; five utility filters; 12-stage evidence-to-registry chain; discovery drawer; explicit LLM then network confirmation; append-only human review | Passed |
| Theme | Deep Space, Dawn and Day menu options update `data-theme` without changing semantic status meaning | Passed |
| Backtest | One active experiment uses radio semantics; comparison is separated and capped at four; NAV/drawdown chart has slider and concise dates | Passed |
| Chart Workstation | Wheel zoom contract, pointer-pan configuration, slider, range buttons, left/right controls and Home/End/arrow keys; visible window survives React re-render in component regression | Passed |
| Data Lab | Catalog, provider jobs, quarantine import/export, exact duplicate/date coverage and DataRecorder tick/depth controls; all large paths remain server-side | Passed |
| Selection and T+1 | Actionable evidence chain replaces blank canvas; no fabricated values | Passed |
| Risk | Six-metric strip and evidence panels render with missing event pages; previous undefined-page crash fixed | Passed |
| VN.PY parity | Summary columns plus inspector prevent the reported Current gap / Next action collision | Passed |
| Help | All links are internal QuantAgent routes; external link count is zero | Passed |
| Console | Fresh final browser session contains zero application-level warning/error entries; browser-extension metadata noise excluded | Passed |

## Comparison and iteration history

1. Round 1 — matched the Decision Dashboard / Training Lab density and hierarchy, then introduced the shared Workbench components and semantic chart adapter.
2. Round 2 — replaced the factor page's unused space with the governed discovery cockpit; converted selection and T+1 empty areas into evidence-driven actions; split dense parity detail into an inspector.
3. Round 3 — fixed the Risk event-page crash, restored the legacy truthfulness copy required by regression tests, exercised all three themes and verified K-line controls in the browser.

## Automated verification

- Frontend: 31 tests passed; TypeScript passed; Vite production build passed.
- Quant UI backend: 52 tests passed.
- Full repository: 1305 tests passed, 17 environment-dependent tests skipped.
- Python bytecode compilation and `git diff --check` passed.

final result: passed
