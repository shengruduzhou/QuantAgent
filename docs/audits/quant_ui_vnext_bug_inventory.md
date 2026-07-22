# Quant UI VNext Bug Inventory

Updated: 2026-07-23

| ID | Reproduce | Expected | Actual / root cause | Fix / regression evidence | State |
| --- | --- | --- | --- | --- | --- |
| UI-001 | Open Dashboard at 1366px | Clear primary decision and no clipped lower content | Equal-weight modules overfill viewport | Four decision states + one primary canvas + action queue; 1363px browser capture has no document overflow | Fixed / browser verified |
| UI-002 | Compare header/rail/tabs | Each surface has a unique responsibility | Route actions repeat in three locations | One global command bar, grouped rail and workspace tabs | Fixed / browser verified |
| UI-003 | Open multiple routes, inspect tabs | Reorder, pin, duplicate and split | Stored as path strings only | Typed workspace store with migration, reopen, pin, duplicate, reorder and right/bottom split tests | Fixed / automated verified |
| UI-004 | Inspect Dashboard risk | Precise current/threshold/violation comparison | Radar is the dominant risk view | Risk limit rows and direct Risk Workstation actions replace the primary radar | Fixed / browser verified |
| UI-005 | Open activity during job | Tasks, logs, alerts, events and connection state remain visible | Jobs-only right drawer | Bottom Operations Dock with independent cancel controls and non-nested buttons | Fixed / browser verified |
| UI-006 | Search model/run/artifact | Grouped entity result opens exact context | Module-only filtering | `/api/search` typed groups + global entity palette and exact query routes | Fixed / automated verified |
| UI-007 | Audit CSS overrides | One token source and bounded modules | 8,264 lines of layered root CSS | New work is scoped to `vnext/styles` tokens/shell/dashboard/training; legacy styles remain isolated for gradual retirement | Mitigated |
| UI-008 | Open at workstation width | No overflow or accidental crop | Not continuously verified | 1363×936 full-page browser QA on Dashboard, Training, Backtest, Chart, Help and Operations Dock; `scrollWidth === clientWidth` | Fixed for verified viewport |
| UI-009 | Disconnect WebSocket | Visible recovery and REST state retained | Hook reconnects but shell hides state unless drawer opens | Always-visible connection status plus Dock details | Fixed / automated verified |
| UI-010 | Refresh active training task | Restore current run from persisted jobs | Model page is artifact-centric | Training Lab navigator, inspector, console, validation/arm/start/cancel and model comparison | Fixed / browser verified |
| UI-011 | Open VN.PY alignment matrix | Long gap/action text never overlaps columns | Free-form text expanded across fixed table cells | Fixed-layout five-column summary, ellipsis/clamping and sticky right inspector | Fixed / browser verified |
| UI-012 | Select backtest context | Exactly one experiment controls result pages | Checkbox table allowed simultaneous active contexts | Native radio selection and query-backed active run; regression test asserts one checked radio and zero checkboxes | Fixed / automated + browser verified |
| UI-013 | Use K-line mouse/keyboard controls | Wheel zooms, drag pans, arrows navigate, Home/End select ranges | Gesture behavior and toolbar wrapping were inconsistent | ECharts inside-zoom/pan contract, focusable keyboard surface, range buttons and wrapped toolbar with no horizontal scroll | Fixed / browser verified |
| UI-014 | Inspect factor registry | Useful/rejected/unknown factors are explicit | Unqualified factor list leaves empty decision space | Lifecycle/usefulness filters, rejection reasons and direct detail context | Fixed / browser verified |
| DATA-001 | Open Runtime/Data Ops | Large TickFlow files stay server-side and all major workflows are present | Import/export, exact coverage, recorder and provider forms were absent | Quarantine import/export, exact duplicate/date scan, TickFlow daily/minute/tick/depth forms and cancellable recorder jobs | Fixed / automated + browser verified |
