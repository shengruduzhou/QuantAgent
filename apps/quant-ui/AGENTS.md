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
