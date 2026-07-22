# ADR-002: VNext-only Workstation Cutover

Status: accepted  
Date: 2026-07-23

## Decision

`InstitutionalShell` is the only Quant UI product entry. Remove the parallel `AppShell`, legacy
Dashboard, legacy command palette/workspace state, `?ui=legacy` resolver and the monolithic V5
shell stylesheet. Existing domain pages remain shared inside `WorkspaceRoutes`; they are active
workflows, not legacy entry points.

The supported integrated launcher is:

```bash
./scripts/run_quant_ui.sh --runtime /path/to/runtime --host 127.0.0.1 --port 8000
```

It builds the frontend before starting the single FastAPI process. `python -m services.quant_api`
remains the API/service entry and now exposes matching `--runtime`, `--host`, `--port`, `--reload`
and `--log-level` options.

## Evidence gate

- VNext shell, themes, workspace tabs, split pane, command palette and internal Help passed component tests.
- Shared backtest, factor, chart, Runtime/DataManager and settings workflows remain imported by VNext routes.
- Production frontend build and Quant UI backend tests must pass before merge.
- Browser smoke verification must cover startup, default Dashboard, Help and a shared domain route.

## Consequences

- Stale localStorage or `?ui=legacy` cannot restore obsolete UI code.
- Rollback is performed through Git history, not a permanent second application shell.
- Frontend changes require a new build; `--skip-build` is reserved for an unchanged verified bundle.
