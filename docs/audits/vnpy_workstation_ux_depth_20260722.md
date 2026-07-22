# QuantAgent × VeighNa Workstation UX Depth Audit

Date: 2026-07-22  
QuantAgent base: `main@c47f35546965c1018e8d585913bfb5335fec94b1`  
Active branch: `feat/vnpy-workstation-ux-depth-20260722`  
Draft PR: `#13`  
VeighNa source authority: official `vnpy/vnpy` 4.4.0 and official module documentation

## 1. Scope and non-claim

This slice addresses browser defects and workflow ambiguity reproduced from eight user screenshots. It deepens the existing Web workstation but **does not claim full vn.py parity**.

A capability is not complete because:

- a navigation item exists;
- a table renders;
- a status badge says partial or ready;
- a button exists without a backend command;
- mock or stale data can fill a chart;
- a research-only projection resembles a trading monitor.

The following remain required for a verified capability:

1. real backend behavior;
2. validated command/API schema;
3. explicit lifecycle and state ownership;
4. correct runtime/database mutation where applicable;
5. typed progress/event feedback;
6. error, empty, stale and disconnected states;
7. automated tests;
8. browser verification;
9. auditable evidence;
10. no fabricated result or bypass of safety controls.

## 2. Screenshot findings and implemented response

### 2.1 Capability Registry text collision

Observed:

- `Current gap` and `Next action` were full prose columns inside a fixed-width six-column table;
- long English text crossed column boundaries;
- the inspector repeated the same content while the matrix attempted to display it in full;
- narrowing the browser made the table unreadable.

Implemented:

- replace the six-column long-text table with a five-column summary matrix;
- combine gap and next action into one bounded `补全路径` cell;
- line-limit summary text and expose the complete content in the persistent inspector;
- preserve category, capability, status and canonical QuantAgent mapping as dedicated columns;
- add quick filters for `not_audited`, `missing`, `partial` and `planned`;
- keep full source, adoption decision, tests, evidence and limitations in the inspector;
- add responsive inspector stacking below the matrix at narrower widths.

The matrix remains a governance view, not a substitute for implementation.

### 2.2 Main workstation appearance and density

Observed:

- the shell was functional but inconsistent in information hierarchy;
- several areas had large unused regions while neighboring regions were compressed;
- primary, secondary and safety actions were not always visually distinct;
- external documentation was acting as the Help experience.

Implemented:

- add an internal Help Center using the existing terminal visual system;
- add workflow cards for experiments, full-universe training, cleanup and capability completion;
- introduce one restrained cyan/blue workstation refinement layer rather than a parallel theme;
- retain compact borders, small status text, monospaced identifiers and dense panels;
- add explicit safety blocks for research/paper mode and controlled operations;
- avoid large decorative gradients, excessive glow and card-only dashboard layouts.

### 2.3 Help opened the VeighNa website

Observed:

- the top Help action directly opened an external vn.py documentation page;
- QuantAgent therefore lacked product-specific operation guidance;
- shortcuts, data deletion policy and training boundaries were not discoverable in-product.

Implemented:

- Help now routes to `/help` inside QuantAgent;
- the page documents global search, table navigation, K-line gestures, data cleanup and training scope;
- official VeighNa documentation, source and portal are secondary external references only;
- automated routing and page-content tests verify this boundary.

### 2.4 Backtest experiment multi-selection

Observed:

- up to five experiments could be checked;
- only the first selected run drove metric cards, NAV and capability detail;
- the UI therefore implied multi-run comparison while operating as a single-run inspector.

Implemented:

- replace checkboxes with one radio-selected active experiment;
- one selected ID owns all metrics, NAV, capabilities, artifact path and export;
- row click, keyboard Enter and Space select the active experiment;
- state clearly that future multi-run comparison belongs in a separate Compare workspace with aligned metrics and horizons;
- add a component test asserting exactly one active experiment.

This follows the operational pattern of selecting one strategy/configuration for a backtest result workspace. It does not prohibit a future explicit comparison module.

### 2.5 K-line drag and zoom conflict

Observed:

- mouse wheel both zoomed and moved the chart;
- left-button drag did not perform the expected horizontal pan;
- keyboard movement by one bar was too fine for ordinary review;
- selected-trade look-ahead could prevent a nominal left move from changing the visible right edge.

Implemented:

- wheel performs zoom only;
- left-button drag performs pan only;
- wheel-pan is disabled;
- Left/Right move the visible window by exactly five bars;
- PageUp/PageDown move it by twenty bars;
- Home shows all history and End moves to the latest bar;
- visible-window movement is calculated from the current right edge, accounting for the chart's 20% look-ahead;
- the interaction contract is visible above the chart;
- tests cover dataZoom settings, Home/End and exact keyboard movement.

### 2.6 Factor page empty space and usefulness classification

Observed:

- the sidebar showed a long undifferentiated factor list;
- selected/rejected factors were not clearly separated;
- empty chart areas did not consistently explain which artifact was missing;
- the user had already filtered useless factors, but the UI did not expose that governance state.

Implemented:

- add filters for all, active, useful candidate, excluded and unevaluated factors;
- classify using declared lifecycle plus actual training/selection/timing/risk usage;
- do not invent an IC threshold in the browser to label a factor useful;
- show counts for every utility class;
- show the selected factor's utility status and pipeline usage;
- add explicit empty states for decay, regime and independent backtest artifacts;
- state that excluded factors do not enter training merely because they exist in the registry;
- link to controlled full-universe training and Runtime Cleanup.

Known limitation:

- the canonical accepted/rejected assignment remains a backend/runtime artifact;
- formula-level Alpha101 and Alpha158 equivalence is not verified by this UI classification;
- lifecycle metadata quality still needs repository-wide audit.

### 2.7 Full-universe training from the Web

Existing verified path:

- `train-v8-deep` is an allowlisted JobManager command;
- `symbols` and `symbols_file` are optional;
- when both are absent, the CLI trains on all symbols contained in the dataset;
- the backend validates command IDs, parameters, input paths and Runtime-only output paths.

Implemented product entry:

- `/settings?job=train&universe=all` opens the training job template;
- no symbol filter is included;
- `horizon_class` uses the canonical `short_5d` value;
- `feature_policy: judgment` requests the accepted-factor assignment;
- the page explains GPU/high-cost implications;
- launch still requires explicit user confirmation;
- no free-form shell command is accepted;
- a test asserts that `symbols` and `symbols_file` are absent.

Failure policy:

- unavailable data, GPU, model, judgment assignment or invalid path must fail the job;
- the UI must not silently substitute a smaller universe or all factors and report success.

### 2.8 Deletion from the Web

Existing verified path:

- Runtime Cleanup analyzes backend-approved candidates;
- canonical data, manifests and the primary model registry are protected;
- deletion requires a fresh analysis and confirmation text `DELETE`;
- the server revalidates candidates before deletion;
- execution produces an audit record.

Implemented product entry:

- `/runtime?view=cleanup` opens the Cleanup workspace directly;
- factor and Help workflows deep-link to the existing cleanup contract;
- no general path textbox or arbitrary recursive delete command was added.

## 3. VeighNa source-alignment decisions

### 3.1 MainWindow

Adopt:

- compact workstation navigation;
- explicit module/workspace context;
- persistent layout and restore action;
- dense monitors rather than oversized dashboard cards;
- operational menu actions.

Adapt for Web:

- router-backed workspaces and tabs instead of Qt dock widgets;
- local persisted layout schema instead of `QSettings`;
- internal Help Center instead of forwarding the main Help action.

Still missing:

- multi-pane drag/drop docking;
- saved named layouts;
- linked monitor-to-trading-widget actions;
- module lifecycle status from a backend plugin registry.

### 3.2 BaseMonitor

Already introduced:

- canonical Web `MonitorTable` foundation;
- stable sort;
- keyboard row selection;
- column resize and local persistence;
- auto-fit/reset;
- CSV export;
- explicit empty state.

Still missing:

- keyed tick/order/trade/position/account updates;
- temporary sort suspension during high-frequency updates;
- virtualization and large-row benchmark;
- persisted column order and visibility;
- stale/disconnected/replay-gap status;
- migration of all legacy tables.

### 3.3 ChartWidget and ChartWizard

Adopted:

- linked price and volume panes;
- crosshair and OHLCV tooltip;
- right-side price/volume axes;
- wheel zoom;
- drag pan;
- keyboard navigation;
- jump to latest;
- layered indicators and event markers.

Still missing:

- tick/minute incremental append;
- visible-range Y recalculation benchmark;
- resizable panes;
- brush selection;
- viewport-to-table synchronization;
- named indicator/layout presets;
- image/data export;
- 100k/500k point browser benchmark.

### 3.4 CtaBacktester

Adopted:

- one active experiment context;
- selected configuration controls result metrics and charts;
- explicit artifact capability and unavailable states.

Still missing:

- validated Web strategy-parameter form;
- background parameter optimization;
- optimization progress/cancellation;
- strategy code reload lifecycle;
- explicit multi-run Compare workspace.

### 3.5 DataManager

Existing QuantAgent strengths:

- PIT datasets and manifests;
- Runtime Catalog;
- preview, quality, trust, validation and lineage;
- protected cleanup with audit.

Still missing:

- provider/exchange/symbol/interval/date query form;
- download/update/import job adapters;
- progress and cancellation;
- duplicate/range coverage analysis;
- post-write manifest generation and catalog refresh;
- DataRecorder subscription lifecycle.

### 3.6 RiskManager

Existing QuantAgent strengths:

- canonical A-share RiskGate;
- KillSwitch;
- T+1, lot, price-limit, suspension/ST and liquidity constraints;
- persisted risk events and Risk Center.

Still missing:

- discoverable rule plugin registry;
- per-rule enable/disable and validated parameters;
- configuration persistence/audit;
- event-specific tick/order/trade/timer callbacks;
- paper pre-order interception event projection;
- notification acknowledgement.

RiskGate remains canonical. A vn.py-style adapter must not bypass or duplicate it.

### 3.7 WebTrader and RPC

Current boundary:

- FastAPI and WebSocket support research jobs and UI state;
- there is no verified execution-capable WebTrader path.

Required before any paper/live command transport claim:

- separate process boundary;
- authenticated sessions and RBAC;
- typed RPC command/event contracts;
- correlation ID and idempotency;
- durable sequence/replay;
- heartbeat/reconnect;
- backpressure;
- full command and response audit.

Live trading remains disabled.

## 4. Automated validation

The PR quality gates include:

1. Node 22.12.0 and reproducible `npm ci`;
2. TypeScript `tsc --noEmit` with uploaded diagnostics;
3. canonical MonitorTable tests;
4. focused workstation UX tests for Help, chart, backtest, factor and training interactions;
5. full Vitest suite;
6. Vite production build;
7. Python compile and repository test suite.

Browser verification remains required for:

- 1920, 1440 and 1280 pixel parity layouts;
- actual ECharts wheel/drag behavior;
- no overlap in Capability Registry cells;
- radio selection and factor filters;
- full-universe template review without launch;
- Runtime Cleanup deep link;
- console and network error inspection.

## 5. Prioritized continuation plan

1. **DataManager query/download/update vertical slice**  
   Validated provider command → JobManager → progress/cancel → PIT and quality validation → manifest → Runtime Catalog refresh → Web form.

2. **RiskManager rule registry vertical slice**  
   Typed rules → settings persistence → paper interception evidence → real-time events → Risk Center configuration UI.

3. **BaseMonitor real-time projections**  
   Orders, trades, positions, accounts, logs and risk events with keyed updates and explicit connection state.

4. **Paper OMS snapshot and recovery**  
   Canonical order/trade/position/account state, reconciliation, restart recovery and audit.

5. **Chart visible-range synchronization and performance**  
   Viewport contract, table/stat filters, brush, incremental append and 100k/500k benchmark.

6. **ScriptTrader and RPC process boundary**  
   Governed script commands first; authenticated RPC only when process isolation is required.

7. **Formula-level Alpha audit**  
   Alpha101/Alpha158 definitions, dependency columns, PIT behavior, numeric equivalence and accepted/rejected factor assignment.

## 6. Safety invariants

- live trading disabled by default;
- QMT remains dry-run;
- agents and models cannot submit orders directly;
- OrderManager owns order intents;
- all paper/future live orders pass RiskGate and KillSwitch;
- A-share execution constraints remain canonical;
- Web jobs accept allowlisted structured commands only;
- outputs remain within Runtime;
- cleanup accepts approved candidates only;
- checkpoints and runtime artifacts are never silently rewritten;
- mock, stale, failed and unverified states remain explicit.
