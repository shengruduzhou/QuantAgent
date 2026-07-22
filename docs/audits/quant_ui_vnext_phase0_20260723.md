# QuantAgent Institutional Workstation VNext — Phase 0 Audit

Date: 2026-07-23  
Scope: `apps/quant-ui`, `services/quant_api`, current runtime contracts and real browser interaction  
Status: Option A implemented; automated and browser verification completed for the VNext slice

## Current UI critique

The current UI is a capable route-based research console, but it is not yet a composable institutional workstation.

| Finding | Evidence | Product impact | VNext decision |
| --- | --- | --- | --- |
| Navigation repeats | Top menu, left navigation and route tabs expose overlapping destinations | More chrome than working context | Keep one command bar and one grouped rail; tabs become workspace state |
| Tabs are route history | `useWorkspaceLayout` stores at most eight path strings | No instances, pinning, reorder, split, dirty state or restore | Introduce typed workspace tabs and closed-tab history |
| Dashboard is a module collage | KPI deck, NAV, risk radar, funnel, health, trades and contributors compete | The next action is unclear | Four decision states, one primary canvas and one actionable queue |
| Risk radar is primary | Large radar consumes precise decision space | Hard to compare current value with limits | Replace with limit rows and violation actions |
| Operations are detached | Activity opens a jobs drawer only | Training/backtest/index state lacks persistent operational context | Add a bottom Operations Dock backed by Jobs + WebSocket |
| Search is route-first | Command palette searches modules and recognizes only stock-code format | Cannot open models, runs, backtests or artifacts directly | Group typed entity results and commands |
| Styling is layered, not canonical | Eleven root CSS files; 8,264 lines; `styles.css` alone is 3,962 lines | Specificity drift and expensive regressions | New scoped token/shell/dashboard modules; legacy CSS stays isolated |
| Data trust is mostly correct | API-backed data, explicit empty/error states, no production mock fallback | Good foundation | Preserve fail-loud behavior and provenance language |
| Runtime catalog exists | Typed artifacts, lineage, cleanup and provider workflows exist | Strong base for Data/Runtime workstation | Keep and progressively wrap in typed inspectors |
| Training is model-observability first | Model page exposes metrics and artifacts but not live task lifecycle | Cannot complete Validate → Start → Inspect → Cancel/Resume | Build Training Lab over `/models`, `/jobs`, logs and events |

## Page keep / migrate / retire matrix

| Current route | Decision | VNext destination | Exit condition for legacy page |
| --- | --- | --- | --- |
| `/` Dashboard | Replace behind flag | Decision Dashboard | Four state blocks, primary canvas and queues pass QA |
| `/stock-replay` | Keep and migrate | Chart Workstation | Split panes, linked ranges and event lanes verified |
| `/selection` | Keep | Prediction / Selection Lab | Shared run/model context connected |
| `/factors` | Keep | Factor Lab | Schema-driven factor lifecycle available |
| `/models` | Keep as registry | Model Registry | Training operations move to `/training` |
| `/backtests` | Keep and migrate | Backtest Workstation | Config → run → results loop is real |
| `/t-plus-one` | Keep | T+1 Analysis | Remains research/paper only |
| `/risk` | Keep and migrate | Risk Manager | Rule/threshold/violation workflow replaces radar-first view |
| `/runtime` | Keep and migrate | Data Lab / Runtime Inspector | Type-specific inspectors and task actions exist |
| `/reports` | Keep | Evidence Center / Reports | Evidence context is shared |
| `/settings` | Split later | Task Center / System / Settings | Each concern gets a dedicated workstation |
| `/parity` | Keep as governance | Capability Registry | No retirement planned |
| `/help` | Keep | Product Help | Must remain internal |

## Duplicate navigation and ineffective interaction inventory

1. Top `数据/研究/帮助` duplicates Module Rail destinations.
2. Workspace tab close is real, but the tab itself carries no instance or layout state.
3. `还原布局` resets only rail/drawer/tab paths; it does not restore panel layouts.
4. Command Palette groups no entities and cannot open a known Run, Model, Backtest or Artifact.
5. Activity drawer has no Logs/Alerts/Events/Resources views.
6. Dashboard risk radar has no direct violation action.
7. Several large dashboard regions are informational only, despite interactive-looking borders.
8. Density is fixed; no Compact/Comfortable contract.

## Shell concepts

Scoring weights: readability 30%, task efficiency 30%, information density 20%, extensibility 20%.

| Concept | Structure | Readability | Efficiency | Density | Extensibility | Weighted score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A — Command Rail + Workspace + Dock | Single command bar; expandable grouped rail; typed tabs; optional split; bottom operations dock | 9.1 | 9.3 | 8.6 | 9.4 | **9.14** |
| B — Command Canvas | Palette-first, no persistent rail; maximized canvas; floating context tray | 9.4 | 8.1 | 9.1 | 8.3 | 8.72 |
| C — Mission Control | Persistent navigator, central workspace and right inspector; dock below | 8.0 | 8.8 | 9.4 | 9.0 | 8.69 |

Selected: **A — Command Rail + Workspace + Dock**.

Rationale: it preserves vn.py's modular application mental model without cloning its Qt window chrome, gives long-running jobs a persistent home, and can expand to split workspaces without permanently consuming inspector width.

## Current baseline

- Frontend TypeScript: 7,544 lines.
- Quant API Python: 6,461 lines.
- Root frontend CSS: 8,264 lines across 11 files.
- Last verified build before VNext: CSS 121.89 kB raw / 25.82 kB gzip; main JS 387.29 kB raw / 119.18 kB gzip; lazy ECharts chunk 608.95 kB raw / 206.51 kB gzip.
- Current repository clone contains no configured `~/AI_quant` runtime and no GPU binary, so browser verification must preserve explicit unavailable states. No real training is started from this environment.

## vn.py alignment used for the slice

- MainEngine/EventEngine remain service and event boundaries rather than UI-owned state.
- Modular applications map to the grouped rail and command launcher.
- Monitor-style dense rows map to reusable queue, job and risk-limit lists.
- Chart navigation keeps wheel zoom, drag pan, keyboard movement and selected-range behavior.
- DataRecorder/DataManager workflows remain explicit data operations with persisted outputs, not browser-side file access.

## Phase 1 acceptance target

- VNext is default but `?ui=legacy` and `localStorage.quantagent.ui.version=legacy` preserve the current shell and dashboard.
- One command bar, one rail and typed workspace tabs.
- Tabs reorder, pin, duplicate, close others, reopen and create right/bottom split state.
- Operations Dock exposes real jobs and WebSocket state.
- Dashboard shows four decision states, one primary canvas and actionable queues.
- No primary risk radar.
- Real API empty/error/stale states remain explicit.

## Implemented VNext slice

- Feature flag defaults to VNext while `?ui=legacy` and `localStorage.quantagent.ui.version=legacy` retain a safe legacy fallback.
- Global entity/command search opens typed factor, model, run, backtest, artifact and stock contexts.
- The module rail is grouped by workflow; workspace tabs support pin, reorder, duplicate, reopen, close-others and right/bottom split state.
- The Operations Dock keeps tasks, logs, alerts, events and resources in the workspace and exposes cancellation without nested interactive elements.
- Dashboard uses four decision states, one switchable primary decision canvas and an actionable queue.
- Training Lab implements Validate → Arm → Start → Inspect → Cancel, plus collapsible form/YAML/diff configuration, persisted job context, focused overview/loss/RankIC views, raw/smoothed metrics, progress/ETA/throughput/gradient/GPU telemetry and model comparison.
- The workstation has three persisted visual modes: low-light Night, mixed-brightness Dawn and high-ambient-light Day. Shared design tokens and chart palettes prevent a superficial CSS inversion.
- Backtest context is single-select; Factor Lab exposes useful/rejected/unknown lifecycle filtering; Help remains an internal QuantAgent route.
- K-line interaction follows the vn.py chart mental model: wheel zoom, drag pan, left/right navigation, up/down zoom and Home/End range actions.
- Runtime/Data Ops now includes quarantine import/export, server-side exact duplicate/date coverage, TickFlow daily/minute/tick/depth forms and cancellable forward recording.

## Verification evidence

- Frontend `npm run check`: TypeScript, 34 Vitest tests and production Vite build pass.
- Quant UI Python suite: 47 tests pass, including data transfer, coverage, provider safety and progress regressions.
- Browser QA covered Dashboard, all three visual modes, the collapsed/expanded Training Lab, focused training charts, Backtest, Chart, internal Help, Operations Dock, VN.PY capability registry and Runtime/Data Ops at the workstation viewport.
- Document-level horizontal overflow was checked after toolbar/rail fixes; K-line controls and backtest single selection were operated, not only rendered.
- Authenticated real-provider smoke remains an operator-environment verification because credentials and entitlements are intentionally not present in the repository. The allowlisted execution path is implemented and fails loudly without them.

Final production bundle snapshot: CSS 171.64 kB raw / 33.53 kB gzip; main JS 496.02 kB raw / 143.69 kB gzip; lazy ECharts chunk 608.98 kB raw / 206.53 kB gzip. The chart engine remains isolated from initial module chunks; a 100k/500k visible-range benchmark is retained as an explicit performance follow-up.
