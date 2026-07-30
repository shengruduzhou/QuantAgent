# Quant Operations Control Loop QA

Date: 2026-07-30

## Acceptance map

1. **All-factor experimentation** — `all_reviewed` is the union of reviewed Alpha101, Alpha181, CICC A-share 80, and basic factors. The Factor Lab can launch a chronological calibration/holdout screen and persists summary, correlation, and selected-factor artifacts.
2. **Stock selection as a strategy candidate** — each portfolio experiment can compare `none` with PIT fundamental selection. A missing fundamental input fails only that candidate, not the unscreened baseline.
3. **Durable strategy process** — `scripts/run_quant_ui_tmux.sh` starts the API, log tail, and GPU monitor in a named tmux session and reports whether it created or reattached to a session.
4. **Executable queue coverage** — the all-factor evaluation command is allowlisted and exposed in Factor Lab; strategy launches use the existing allowlisted training pipeline.
5. **Runtime launch feedback** — Data Manager renders submitting/success/error state, while the operations dock receives queued/running/terminal status and bounded logs.
6. **Objective triangle** — the engine generates a Pareto frontier for excess return, annual return, and drawdown. Operator weights rank non-dominated candidates; they do not claim simultaneous maxima.
7. **Tab overflow menu** — the ellipsis menu is rendered outside the clipped tab strip and exposes close/close-others actions.
8. **Top-K search** — the same OOS and cost assumptions are applied to every configured Top-K candidate. Selection uses an early OOS window and final acceptance uses a later untouched holdout.
9. **V7/V8 naming** — the UI labels `V8 Deep GPU Model · FT-Transformer` and `V7 Classical Baseline · Ridge`; the V7 command name denotes the stable orchestration contract, not the model family.
10. **GPU enforcement and telemetry** — the default deep strategy requires GPU and fails closed without it. The UI polls bounded resource telemetry; tmux retains the requested `watch -n 0.1 nvidia-smi` pane.
11. **Theme contrast** — day and dawn muted text tokens were raised and training/factor hard-coded colors were replaced with semantic theme tokens.
12. **T+1 Do-T** — candidates support daily ATR swing timing and an existing minute-level overlay. Minute mode requires minute data and keeps T+1 constraints explicit.
13. **Failed task purge** — purge removes the job record, log, and only outputs proven to have been created by that job.
14. **Factor validity** — holdout RankIC, ICIR, finite coverage, monotonicity, decay readiness, and correlation pruning form an auditable factor gate.
15. **Running task stop and purge** — purge first terminates the process, closes the purge/start race, then removes owned traces. Shared Runtime inputs and outputs owned by other jobs are preserved.

## Verification

- Python: `1913 passed, 51 skipped`.
- Frontend: `15` files and `37` tests passed; TypeScript and production build passed.
- Service smoke: `/health`, `/api/system/resources`, `/api/jobs`, and SPA `/` returned HTTP success from a clean temporary Runtime.
- Shell: both UI launch scripts pass `bash -n`.
- Browser: the managed cloud browser refused local-loopback navigation with `ERR_BLOCKED_BY_CLIENT`; no visual-pass claim is made.
- Environment: the QA container has neither `tmux` nor `nvidia-smi`; real GPU telemetry and tmux persistence must be rechecked on the deployment host.
