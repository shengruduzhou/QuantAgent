# QuantAgent × VeighNa UI and Interaction Source Alignment

Date: 2026-07-22  
QuantAgent base: `main` after PRs #8–#11  
VeighNa authority: `vnpy/vnpy` 4.4.0, commit/tag `0e8e5ba`

## Scope

This slice fixes confirmed frontend quality-gate failures and applies a bounded set of UI/interaction changes derived from the official VeighNa source. It does not claim complete vn.py parity.

Changed areas:

- frontend test and build gates;
- K-line keyboard navigation and ECharts typing;
- semantic market/terminal color tokens;
- parity registry keyboard inspection and filter feedback;
- workstation menu actions and default-layout restore;
- local startup guidance shown by the API-offline banner;
- GitHub Actions frontend validation;
- removal of developer-specific npm cache configuration.

Not changed:

- training, CUDA processes, checkpoints or models;
- backtest, OrderManager, RiskGate or KillSwitch semantics;
- runtime artifacts;
- paper/live order routing;
- credentials or gateway configuration.

## Official source traversal

### 1. Main window and workspace lifecycle

Source: `vnpy/trader/ui/mainwindow.py` at `0e8e5ba`.

Observed behavior:

- the trading widget and tick/order/active-order/trade/log/account/position monitors are dock widgets;
- related monitors are tabified;
- tick and position rows can update the trading widget through direct interaction;
- applications are discovered from `MainEngine.get_all_apps()` and loaded dynamically;
- window geometry and dock state are persisted with `QSettings`;
- a default layout can be restored;
- the toolbar is fixed and deliberately compact.

QuantAgent decision:

- retain the existing Web workstation tabs and launcher as the canonical shell;
- add an explicit default-layout restore action instead of introducing a second layout manager;
- make the terminal menu operational rather than rendering non-interactive text labels;
- keep multi-dock/split-pane support as a separate vertical slice because it requires layout schema, persistence, drag/drop tests and browser verification.

### 2. Monitor tables

Source: `vnpy/trader/ui/widget.py` at `0e8e5ba`.

Observed behavior:

- `BaseMonitor` subscribes to typed events and updates keyed rows in place;
- sorting is temporarily disabled during updates to prevent row/cell corruption;
- monitors use alternating rows, hidden vertical headers and explicit no-edit behavior;
- right-click actions resize columns and export CSV;
- column state is persisted;
- direction, bid/ask and PnL cells use semantic presentation rather than arbitrary per-page colors.

QuantAgent decision:

- introduce semantic terminal/market tokens now;
- keep current tables operational while planning one canonical Web `MonitorTable` adapter for sorting, virtualization, column persistence, CSV export and event-driven keyed updates;
- do not create page-specific table frameworks.

### 3. Chart behavior

Source: `vnpy/chart/widget.py` at `0e8e5ba`.

Observed behavior:

- plots share a linked X axis;
- chart items are clipped to view and downsampled;
- the right-side axis is used consistently;
- visible-range Y bounds are recalculated automatically;
- Left/Right moves the viewport and cursor;
- Up/Down zooms in and out;
- the mouse wheel zooms;
- `move_to_right()` jumps to the latest data;
- the crosshair spans panes and exposes axis labels and item information.

QuantAgent changes in this slice:

- the chart host is keyboard-focusable and explicitly interactive;
- Left/Right shifts the current anchor;
- Up/Down changes the visible range preset;
- `End` jumps to the latest bar;
- `Home` shows all history;
- reset/latest controls are visible;
- linked price/volume axes, crosshair, wheel zoom and pan remain in the existing ECharts adapter;
- chart options now use explicit series types and the production build runs TypeScript before Vite.

Remaining chart gaps:

- brush selection;
- synchronized visible-range trade/stat filtering;
- resizable panes;
- true incremental bar append without replacing the option;
- saved indicator/layout presets;
- image/data export;
- large-point performance baselines;
- announcement/evidence/model-regime overlays;
- browser visual regression.

### 4. MainEngine and OMS

Source: `vnpy/trader/engine.py` at `0e8e5ba`.

Observed behavior:

- `MainEngine` owns gateway, engine and app registries;
- the event engine starts before functional engines;
- Log, OMS, email and WeChat engines have explicit lifecycle ownership;
- `OmsEngine` provides canonical caches for tick/order/trade/position/account/contract/quote objects;
- gateway commands are routed through one engine and logged;
- shutdown stops the event engine before closing engines and gateways.

QuantAgent decision:

- do not clone `MainEngine` inside the API process;
- continue evolving the existing `ServiceContainer`, `JobManager`, `EventBroker`, `RuntimeIndexer`, OrderManager and paper execution adapters behind explicit ownership contracts;
- the next orchestration slice must define a narrow application/plugin registry rather than a universal manager.

### 5. WebTrader process boundary

Official documentation: `WebTrader - Web服务器模块`.

Observed behavior:

- FastAPI provides REST requests and WebSocket event push;
- the trading/strategy process and Web service process are independent;
- RPC bridges commands from Web to the trading process;
- process health, reconnect and bidirectional communication are first-class concerns.

QuantAgent decision:

- the current in-process EventBroker is sufficient only for research/job UI events;
- paper/live execution must remain isolated from the Web process;
- durable sequence, replay, authentication, idempotency and backpressure remain mandatory before any execution-capable WebTrader parity claim.

### 6. DataManager

Source: `vnpy/vnpy_datamanager`, `vnpy_datamanager/ui/widget.py`.

Observed behavior:

- interval/exchange/symbol data are organized as a tree with counts and time ranges;
- the selected dataset can be viewed in a dense table;
- import, export, update, download and delete are explicit actions;
- destructive deletion requires confirmation;
- update/download show progress and support cancellation.

QuantAgent decision:

- extend the existing Runtime/Data workspace rather than build another data manager;
- implement provider/symbol/exchange/interval/date forms as validated jobs;
- outputs must produce PIT/data-quality manifests and refresh the canonical Runtime Catalog;
- destructive operations remain cleanup plans with confirmation and audit evidence.

## Confirmed bugs fixed

1. `VnpyParityPage.test.tsx` used an unscoped text query even though the same gap appears in the matrix and inspector. The test now scopes assertions to the labelled detail inspector.
2. `CandlestickChart.tsx` used an unsupported ECharts `triggerOn` literal and inferred mark-point label positions as generic strings. Series and mark-point data now use explicit ECharts option types.
3. `vite build` previously succeeded while `tsc --noEmit` failed. The production build now runs typecheck first.
4. Node `22.6.0` is below `@vitejs/plugin-react`'s supported `22.12.0` floor. Local development is pinned with `.nvmrc`; GitHub Actions uses Node `22.12.0`.
5. The API-offline banner previously showed a command without saying it must run from the repository root. The banner now states the required context.
6. Top-level `视图 / 数据 / 研究 / 帮助` labels looked interactive but were inert. They are now real actions.
7. `.npmrc` hard-coded `/home/shanhefu/QuantAgent/apps/quant-ui/.npm-cache`, which broke `npm ci` on GitHub-hosted runners. The developer-specific absolute cache path was removed and CI caching is owned by `actions/setup-node`.
8. The repository CI previously ran only Python. A separate frontend job now executes dependency installation, typecheck, unit/component tests and production build.

## Semantic color policy

VeighNa desktop uses semantic cell colors for long/short, bid/ask and PnL. QuantAgent keeps the previously selected blue-up/red-down market palette, but separates the following meanings in a single token source:

- market up/down;
- normal buy/sell;
- intraday T buy/sell;
- bid/ask;
- warning/risk;
- verified/stale/unavailable;
- focus and selected state.

Legacy `--bg`, `--surface`, `--blue`, `--green`, `--red` and `--amber` variables are routed through the semantic source so new modules do not introduce a parallel theme.

## Required validation

Backend:

```bash
cd /home/shanhefu/QuantAgent
source .venv/bin/activate
python -m pytest -q tests/quant_ui
python -m compileall -q services/quant_api src/quantagent
```

Frontend:

```bash
cd /home/shanhefu/QuantAgent/apps/quant-ui
nvm use
npm ci
npm run check
```

Browser:

- restore the default layout;
- open Data and Research from the terminal menu;
- open `/parity`, filter, clear filters and navigate capability rows with Up/Down;
- open `/stock-replay`, focus the chart, use Left/Right/Up/Down/Home/End;
- test wheel zoom, drag pan, signal selection and latest/reset controls;
- inspect console, failed network requests, focus visibility, text overlap and chart overlap.

## Next vertical slices

1. Canonical Web `MonitorTable`: keyed event updates, stable sorting, column persistence, resize/export, virtualization and explicit stale/disconnected states.
2. Runtime/DataManager query-download loop: validated job, typed progress, cancellation, PIT/data-quality manifest and catalog refresh.
3. Chart visible-range synchronization: chart viewport to trade/stat tables, brush selection and incremental updates.
4. Paper OMS snapshot/recovery: orders, trades, positions, account, reconciliation and restart recovery.
5. WebTrader process boundary: authenticated RPC, heartbeat, sequence/replay, idempotency and backpressure.
