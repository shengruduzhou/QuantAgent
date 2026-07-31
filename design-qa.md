# QuantAgent ATLAS Workstation — Design QA

## Scope of this pass

This pass replaced the workstation's ad-hoc colour handling with a single
semantic foundation, added the platform's own visual identity, and shipped two
new first-class modules (Alpha Foundry, Decision Council) built entirely from
that foundation.

Reference material remains `docs/quant_ui_design_concepts/*.png` for density and
information hierarchy. The visual identity is **not** taken from any external
product: it is defined in `apps/quant-ui/src/vnext/styles/tokens.css` and
documented in `docs/research/institutional_quant_reference_architecture.md`.

## Design system

| Layer | File | Rule |
|---|---|---|
| Tokens | `vnext/styles/tokens.css` | The only place a colour, size, or duration is defined. All three themes declare the same token names. |
| Primitives | `vnext/styles/foundation.css` | Seven reusable primitives: surface, eyebrow, figure, chip, grid, meter, empty state. |
| Charts | `vnext/theme.ts` | `useVNextChartPalette()` mirrors the `--viz-*` ramp so charts and CSS never drift. |
| Identity | `vnext/styles/shell.css` (identity layer) | Brand-gradient mark, signal rail on the active module, hairline of brand light under the command bar. |

### Colour semantics — three non-overlapping families

| Family | Palette | Used for |
|---|---|---|
| `ui.*` | azure / violet / cyan / amber | Interaction, agent reasoning, live telemetry, attention |
| `status.*` | emerald / amber / crimson | Governance verdicts — always accompanied by a text label |
| `market.*` | **red = up, green = down** | A-share price and return cells only |

Status green and market green are different colours with different meanings.
They are kept apart by context: status colours appear only on labelled chips and
rails, market colours only on numeric price/return cells. This is checked by the
council UI test, which asserts a verdict chip renders its label text.

## Fixed in this pass

| Severity | Finding | Resolution |
|---|---|---|
| P0 | Deflated Sharpe was computed from annualised Sharpe ratios where the estimator expects per-period ones. A strategy with annualised Sharpe 3.78 scored DSR ≈ 0.00002, so the robustness term was dead and could not distinguish any candidate. | `deflated_sharpe_probability` now converts to per-period before calling, with the reasoning recorded in the docstring. Robustness now separates a real signal (0.99) from noise (0.38). A regression test asserts the term is non-trivial so it cannot silently die again. |
| P0 | A job reported `running` before its process was registered, so `pause` and `cancel` could be rejected on a job the operator could see running. | Added a `starting` state; status becomes `running` only after `Popen` succeeds and the process is registered. |
| P1 | `theme.css` defined market and state colours at `:root` with fixed dark values, so day and dawn themes inherited dark-only colours on legacy pages. | Every cross-cutting token (`--market-*`, `--state-*`, `--iw-*`, `--ux-*`, `--focus`) is now declared inside `.vnext-shell` for all three themes. |
| P1 | Chart series colours were chosen per page, so two charts on one screen could use unrelated palettes. | `VNextChartPalette` gained a `series` ramp plus named accents, mirroring `--viz-1…8`. |
| P2 | No shared primitives existed, so each new page reinvented panels, chips, and tables. | `foundation.css` primitives; both new modules use them exclusively. |

## Honesty affordances (verified by test)

These are design decisions with test coverage, not just copy:

- **Controls are visually marked.** Unfitted baselines render with a dashed chip
  and muted row text so they cannot be mistaken for a result.
  (`marks control candidates so they cannot be read as a result`)
- **Trial count is shown before launch** and is derived from the enumerated
  search space; the launch payload contains no `n_trials`.
  (`launching a search posts the governed command without a trial count`)
- **Missing evidence renders as "无估计" / "证据不足"**, never as a pass.
  (`a missing PBO estimate is reported as no evidence, never as a pass`)
- **Overrides preserve what they replaced.** The original verdict stays on
  screen beside the override, with author and timestamp.
  (`an override keeps the original verdict visible next to it`)
- **Empty states explain the missing artifact and the next action**, and never
  fill space with placeholder numbers.
  (`shows an actionable empty state instead of fabricated metrics`)

## Accessibility and theming

- All three themes (`night`, `dawn`, `day`) declare the full token set; no page
  relies on a token that only exists in one theme.
- Focus rings are a single token-driven rule applied to buttons, links, inputs,
  selects, textareas and `[tabindex]` elements.
- Wide content (candidate ledger, fold tables) scrolls inside
  `.atlas-scroll-x`; the page body never scrolls horizontally.
- `prefers-reduced-motion` disables the job-state pulse and all transitions.
- The 10px floor on `small` is retained for dense metadata legibility.

## Verification

- Backend: `178 passed` across `tests/quant_ui/` and `tests/test_fusion_search.py`
  (49 of them new in this pass), plus the full suite run separately.
- Frontend: `19 files, 61 tests passed`; TypeScript clean; Vite production build clean.
- No fabricated metrics were introduced: every number the new modules render
  comes from a persisted artifact, and both modules render an explicit
  unavailable state when the Quant API is unreachable.

## Known limits

- Drawdown in the Alpha Foundry is measured on a rebalance-frequency NAV, not a
  daily one, because the panel supplies forward returns rather than daily marks.
  The UI states this; it is a lower bound on daily-marked drawdown.
- `CouncilThresholds` are this platform's research bars, not an industry
  standard. They are visible to the operator and actually enforced, but must not
  be cited as an external benchmark.
- `selection_governance.py` passes annualised Sharpes to
  `deflated_sharpe_ratio`, the same units mismatch corrected in the fusion path.
  It was left unchanged in this pass to avoid altering an existing governance
  result without a separate review.
