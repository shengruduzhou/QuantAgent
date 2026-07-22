# QuantAgent × VeighNa DataManager control loop

Date: 2026-07-22  
Scope: `data.datamanager-query-download-update`  
Branch: `agent/datamanager-control-loop`

## Outcome

This slice turns the existing Runtime/Data workspace into an operational data control loop without introducing a second catalog, scheduler, storage root or command runner.

The implemented path is:

`provider capability → explicit query form → allowlisted JobManager command → existing PIT/schema validation → Runtime-only output → cache invalidation → canonical RuntimeIndexer rescan`

## Official source alignment

The implementation was checked against:

- VeighNa DataManager documentation: download, view, import, export, delete, update and data-range inspection;
- `vnpy/vnpy_datamanager` `ManagerEngine`: database overview/load/delete and datafeed-backed bar/tick download;
- `vnpy/vnpy_datamanager` `ManagerWidget`: interval/exchange/symbol hierarchy, explicit download fields and update progress;
- `vnpy/vnpy` `BaseDatafeed`: `HistoryRequest`-based bar/tick query boundary;
- `vnpy/vnpy` `BaseDatabase`: save/load/delete/overview abstraction.

QuantAgent adapts these boundaries instead of copying Qt or creating a `MainEngine` clone. Its PIT artifacts, provider adapters, existing CLI commands, `JobManager`, `RuntimeIndexer` and protected cleanup remain canonical.

## Implemented

### Backend

- Adds a provider capability registry for AkShare market bars, local Qlib history, TuShare PIT fundamentals and the canonical Runtime catalog.
- Reports package/configuration state without returning credential values.
- Adds a dedicated `/api/jobs/data` route backed by the existing allowlisted `JobManager`.
- Allows only three existing governed CLI commands:
  - `build-akshare-market-panel-v7`
  - `build-market-panel-v7`
  - `build-fundamentals-v7`
- Requires an explicit symbol list or project-scoped symbol file for provider downloads.
- Requires network permission to be explicit for network providers.
- Keeps all Web-created outputs inside Runtime.
- Adds job cancellation for queued/running work and terminates only the tracked child process.
- Invalidates the existing RuntimeIndexer cache after a successful data job.

### Web workstation

- Adds a Data Ops tab inside Runtime/DataManager.
- Shows provider availability and missing configuration explicitly.
- Uses structured symbol/date/path fields rather than accepting a shell command.
- Separates network confirmation from job submission.
- Shows only data jobs in the local queue and exposes cancellation for active work.
- Links to the existing full-universe training template and protected cleanup workflow.

## Safety properties

- Live trading remains disabled.
- No broker/order path changes.
- No arbitrary shell, URL, environment value or filesystem path is accepted.
- Credentials stay in backend environment variables.
- AkShare download cannot silently become a full-market crawl; symbols are explicit.
- Provider failures do not fall back to mock data.
- Output paths are revalidated by the backend.
- Deletion remains limited to backend-approved cleanup candidates.

## Automated coverage

- Provider registry and credential non-disclosure.
- Rejection of arbitrary data commands.
- Rejection of downloads without an explicit universe.
- Queued job cancellation and terminal-state protection.
- Frontend launch contract for AkShare with explicit symbols, network approval and Runtime output.

## 2026-07-23 completion update

The VNext continuation closes the product gaps that were intentionally left in the first control-loop slice:

- Runtime-only quarantine discovery plus streamed import/export with exact import de-duplication, filters and SHA-256 manifests;
- chunked date/symbol coverage inspection, with optional disk-bounded exact duplicate scanning for large TickFlow files;
- parsed JSON, batch/total and legacy `[current/total]` provider progress;
- TickFlow-specific daily, minute, quote-snapshot and Level-2 forms;
- bounded, cancellable forward DataRecorder jobs for TickFlow quotes and depth snapshots;
- browser verification of Data Ops layout, exclusive controls, overflow and disabled/unavailable states.

`data.data_manager` is now `implemented` in the capability registry. It is not marked `verified` because authenticated provider smoke/scale runs still require the operator's TickFlow/TuShare entitlements and network confirmation. Remaining engineering work is limited to gateway-neutral recorder restart orchestration, exchange-calendar confirmation of candidate date gaps, Windows process-group escalation and 100k/500k throughput baselines. These are recorded as verification/scale follow-ups, not as missing UI workflows.
