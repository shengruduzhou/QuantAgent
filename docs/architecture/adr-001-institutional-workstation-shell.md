# ADR-001: Institutional Workstation Shell

Status: accepted  
Date: 2026-07-23

## Decision

Build a new, scoped `vnext` shell beside the legacy `AppShell`. VNext owns the Global Command Bar, grouped Module Rail, typed Workspace Tabs, split state, shared context and Operations Dock. Existing domain pages remain reusable while they are progressively replaced.

## Why

Incremental CSS changes to the old shell cannot provide tab instances, split workspaces or canonical shared context without increasing coupling. A separate shell lets the product migrate route by route and provides a safe `legacy` escape hatch.

## Constraints

- No browser-side runtime file reads.
- No mock data in product paths.
- No live trading controls.
- New styles are modular and scoped to `.vnext-shell`.
- Existing pages retire only after equivalent VNext workflows pass browser QA.

## Feature flag

Resolution order:

1. `?ui=vnext|legacy`
2. `localStorage.quantagent.ui.version`
3. `VITE_WORKSTATION_VNEXT`
4. default `vnext`

The query override is persisted so users can recover from a bad layout without rebuilding.
