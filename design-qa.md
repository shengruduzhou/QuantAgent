# Quant UI Design QA

## Evidence

- Source visual truth path: `docs/quant_ui_design_concepts/03-signal-observatory.png`
- Implementation screenshot path: `runtime/reports/quant_ui/qa/model-lab-final-1440x1024.png`
- Full-view comparison evidence: `runtime/reports/quant_ui/qa/model-lab-side-by-side-1800x760.png`
- Mobile evidence:
  - `runtime/reports/quant_ui/qa/model-lab-mobile-390x844.png`
  - `runtime/reports/quant_ui/qa/stock-replay-mobile-390x844.png`
- Viewport: desktop 1440 × 1024；mobile 390 × 844
- State: real runtime snapshot；Deep Alpha model selected；no fabricated fallback data

## Full-view comparison

Source and implementation were rendered together in one 1800 × 760 browser frame.
The implementation preserves the source visual language rather than reproducing
unrelated content:

- compact icon navigation、thin cyan/blue borders and restrained dark navy surfaces；
- dense top-level filters and global operational status；
- left asset catalog + central analysis workspace；
- high-information cards、charts、tables and explicit model/risk states；
- cyan/green for available/positive state，amber/red for warning/failure state；
- no decorative gradients、hero illustration、emoji or fake financial imagery。

The source is a portfolio signal observatory while the implementation is a model
observability terminal，so content blocks intentionally differ. Composition、
density、hierarchy and terminal tokens remain aligned.

## Focused region comparison

Focused inspection used the direct 1440 × 1024 implementation capture because
catalog labels、metric cards、chart axes and capability text are too small in the
combined frame.

### Fonts and typography

- Inter Variable + Noto Sans SC provide compact bilingual UI text；JetBrains Mono
  is limited to paths、IDs and numeric values。
- Heading、metric and metadata weights form a clear hierarchy without oversized
  display text。
- Long model versions and paths use controlled truncation or wrapping。

### Spacing and layout rhythm

- Desktop uses a stable 58 px icon rail、8 px panel gaps and compact 7–18 px
  internal padding，matching the source terminal density。
- Model catalog、hero、metrics、tabs and 2-column analysis panels align to a
  consistent grid。
- Mobile collapses to one analysis column and converts catalog/tabs to contained
  horizontal scrolling；document-level horizontal overflow is false。

### Colors and visual tokens

- Background/surface/border hierarchy remains within dark blue-black tokens。
- Cyan is reserved for active/observability state，green for ready/positive，
  amber for warning/manual review and red for destructive actions。
- Contrast and focus rings remain visible without broad neon glow。

### Image quality and assets

- The product has no photographic or illustrative asset requirement。
- Visible icons use the Phosphor icon family；no handcrafted SVG、CSS art、
  placeholder image or emoji substitutes were introduced。
- ECharts output remains sharp at the tested desktop and mobile viewports。

### Copy and content

- App-specific copy consistently uses research-only、PIT、T+1 and no-live-order
  language。
- Missing SHAP、feature importance、trade rationale or prediction artifact is
  labeled unavailable；the UI does not invent values。

### Interactions and accessibility

- Model family filter updates both catalog and selected detail。
- Model comparison supports up to 6 selections and prioritizes persisted
  performance metrics。
- Tabs、artifact inventory、Runtime cleanup、job templates and navigation work。
- `Ctrl/Cmd + K` opens the command palette；arrow keys and Enter work；stock codes
  route to Stock Replay。
- Focus-visible and reduced-motion rules are present。

## Findings

No actionable P0/P1/P2 findings remain.

Residual P3:

- Dense metric bar labels may truncate on narrow panels；the complete metric name
  remains available in the full metrics table。
- Feature importance panels are empty for models without a persisted importance
  artifact；this is correct source-backed behavior rather than a visual defect。

## Patches made since the previous QA pass

- Added unified visibility for Deep FT、registered alpha、RL policy、T+1 joblib
  bundles and generic model binaries。
- Added performance-first model comparison、checkpoint metadata and linked
  artifact size。
- Fixed family-filter selection and defaulted the first view to a rich Deep Alpha
  model rather than a sparse generic artifact。
- Added command palette keyboard navigation and direct stock-code routing。
- Added explicit feedback when a requested stock has no standard trade in the
  selected backtest。
- Changed missing selection decision chains from HTTP 404 to an empty,
  recoverable response。
- Added real mobile layouts and removed the previous 1000 px forced canvas；
  verified `scrollWidth === innerWidth` at 390 px。
- Added audited Runtime cleanup and protected current QA captures from immediate
  reclassification as stale。

## Implementation checklist

- [x] Desktop visual hierarchy and density
- [x] Fonts、spacing、colors、icons and copy review
- [x] Real model data and missing-data states
- [x] Filters、tabs、comparison and command palette
- [x] Runtime cleanup confirmation flow
- [x] Desktop and mobile viewport resilience
- [x] Side-by-side source/implementation comparison

final result: passed
