# Prototype Instructions

Run the local server yourself and open the preview in the in-app browser. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

## Confirmed Quant UI direction

- Use `docs/quant_ui_design_concepts/02-research-workbench.png` as the primary shell and Stock Replay reference.
- Integrate the dense trade blotter and exposure modules from concept 1.
- Integrate the risk radar, system health, transparent selection funnel, and top-contributor modules from concept 3.
- All intraday overlay language must say `T+1 合规做 T` or `T+1 Analysis`; never label the feature `T+0`.
- The product is a full-interactivity research terminal backed by real QuantAgent API/runtime data. Missing artifacts must show explicit empty or unavailable states.
- The Command Center must read as a professional quant cockpit: one dominant portfolio narrative, explicit secondary risk/health layers, restrained electric-blue/cyan accents, and A-share red-up/green-down market semantics. Avoid equal-weight KPI walls, tiny terminal copy, decorative border overload, and low-contrast metadata.
- VNext separates Dashboard from Workstation: Dashboard answers system/data/model/portfolio/risk/next-action questions, while configuration and execution live in dedicated workstations.
- VNext uses one Global Command Bar, one grouped Module Rail, true context-carrying Workspace Tabs, and a collapsible Operations Dock. Do not duplicate those actions in a second menu.
- Risk radar is never the primary risk decision view. Prefer rules, limits, threshold bars, violations, and actionable queues.
- The VNext shell passed browser and regression QA and is now the only product entry. Do not reintroduce a parallel legacy shell, `?ui=legacy` escape hatch, or monolithic workstation stylesheet; domain pages may remain shared behind the canonical VNext shell.
