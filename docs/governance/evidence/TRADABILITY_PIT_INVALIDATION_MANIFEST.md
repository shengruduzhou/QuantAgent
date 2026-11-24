# Historical ST / Tradability PIT Invalidation Manifest

**Declared:** 2026-08-12  
**Finding:** `TICKFLOW-ST-PIT-001` / GitHub issue #86  
**Status:** **PROMOTION BLOCKED UNTIL REBUILT**

## Decision

Historical evidence derived from a current TickFlow instrument-name snapshot is not point-in-time evidence. A symbol's current `ST` / `*ST` name cannot establish its risk-warning state on an earlier trade date. Any historical row stamped with `is_st_provenance = current_snapshot_broadcast`, or any historical row lacking explicit dated ST provenance, is quarantined for economic promotion, model selection, factor validation, strict backtest claims, FRESH holdout claims, and live-model trust evidence.

This is a temporal-integrity invalidation, not a claim that the underlying OHLCV bars are wrong. The affected artifacts may be retained for forensic comparison but must not be used to establish strategy performance or production eligibility until the tradability fields are rebuilt from dated PIT evidence.

## Known affected artifacts

The following repository/runtime contracts are known to have consumed or documented snapshot-broadcast ST state and therefore require rebuild or explicit re-certification:

| Artifact / path | Known issue | Required disposition |
|---|---|---|
| `runtime/data/v7/silver/market_panel/market_panel.parquet` | historical `is_st` rows may be `current_snapshot_broadcast` while `point_in_time_valid=true` | rebuild affected historical tradability rows from dated PIT ST evidence; persist `is_st_provenance=dated_pit_st_evidence` and `st_evidence_sha256` |
| `FRESH_HOLDOUT_FREEZE_MANIFEST.md` window `2026-05-19..2026-07-02` | manifest explicitly records `current_snapshot_broadcast` ST provenance | prior FRESH QC/performance eligibility is invalid for promotion until the window is rebuilt and re-QC'd without reading strategy performance during repair |
| `runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet` | may have tradability flags copied from the affected panel | rebuild flags only after the panel passes the dated-PIT provenance gate |
| `runtime/data/v7/gold/training_dataset/training_dataset_alpha181_governed_v85.parquet` | same | same |
| `runtime/data/v7/gold/training_dataset/training_dataset_alpha181_selected_v85.parquet` | same | same |
| downstream models / backtests whose manifests bind any pre-rebuild artifact above | inherited temporal contamination | rerun/reissue evidence from rebuilt inputs; do not carry forward old headline CAGR/Sharpe/IC/DSR/SPA as trusted evidence |

The list is conservative and may expand if lineage inspection finds additional consumers. Absence from this table is not proof of safety; source lineage must demonstrate dated historical ST coverage.

## Replacement evidence contract

A historical tradability row is promotion-eligible only when all of the following hold:

1. `is_st` is resolved by `(symbol, trade_date)` or an explicit non-overlapping effective interval.
2. The ST evidence has an `available_at` timestamp no later than the 09:25 Asia/Shanghai pre-open boundary for the relevant session (or earlier).
3. Missing ST state is **unknown/blocking**, never silently `False`.
4. The resulting row carries `is_st_provenance = dated_pit_st_evidence`.
5. The exact evidence file is bound by a valid 64-hex `st_evidence_sha256`.
6. Price-limit flags are recomputed from that historical ST state plus the date-versioned exchange rule source; they are not copied from a snapshot-based panel.
7. Any gold/training rewrite has 100% `(symbol, trade_date)` coverage from a PIT-certified panel.

`src/quantagent/data/providers/st_pit.py` is the supplier-neutral implementation of this contract. `TickflowProvider.tradability()` now fails closed without such evidence; `current_snapshot_tradability()` is explicitly non-PIT and current-monitoring-only.

## Rebuild procedure

1. Acquire or construct a dated historical ST/risk-warning table. Accepted local evidence representations are exact daily rows (`symbol, trade_date, is_st, available_at`) or explicit intervals (`symbol, start_date, end_date, is_st, available_at`).
2. Run the PIT-safe panel updater/repair path with `--historical-st-path` (or `QUANTAGENT_HISTORICAL_ST_PATH`).
3. Verify affected panel rows have no null tradability flags, no `current_snapshot_broadcast`, `point_in_time_valid=true`, and valid evidence SHA bindings.
4. Re-run market-panel QC without inspecting strategy performance.
5. Rebuild gold tradability flags using `scripts/fix_training_dataset_tradability_flags.py`; the script now refuses incomplete/non-PIT panels and missing joins.
6. Re-run dependent factor lifecycle/backtest/statistical evidence from the rebuilt artifacts.
7. Issue a new dated manifest with exact file SHA256s and source lineage. Only that new evidence may supersede this invalidation.

## What remains valid

Repository code, exchange rule definitions, raw OHLCV that independently passes its own provenance checks, and unrelated paper-execution evidence are not invalidated merely by this finding. Performance or factor evidence that depended on the contaminated historical tradability state is invalidated until rebuilt.

## Non-live statement

Resolving this manifest is necessary but not sufficient for live trading. Authoritative exchange calendar certification, production pre-trade risk, venue reconciliation, end-to-end restart semantics, and one-shot fresh acceptance remain separate gates.
