# VALUATION + FUNDAMENTAL TRAINING-SET INTEGRATION — H-020 (pre-registered)

**Status: REGISTERED 2026-07-08 (candidate feature set frozen here, before any build/train run).**

## 0. Why (the gap, from inspection)

The production training dataset `…_v89_plus7clean.parquet` (327 cols, 6.78M rows) carries
technical (alpha101/181 + gtja191), macro (yields/shibor/M0-M2/CPI/PPI), flow (north/margin),
index & commodity/overseas proxies, and tradability flags — but **zero firm-level valuation or
fundamental values**. It only has placeholder flags `missing_fundamentals`, `missing_valuation`,
`missing_disclosures`. Meanwhile:

- **The model already expects fundamentals.** `training/horizon_models.py::select_features` matches
  LONG-horizon features by name pattern including `roe, roa, gross_margin, net_margin, revenue_yoy,
  net_income_yoy, debt_to_asset, inventory_turnover, valuation_percentile, …`. Those columns are
  absent, so the LONG sleeve trains on technical/macro only. **Architecture gap = data gap.**
- **A rich PIT-safe fundamentals panel already exists on disk:**
  `silver/fundamentals/metrics_panel.parquet` — 3654 symbols, 2007-2026, with `announce_date` AND
  `available_at` (= announce_date + 1d), `period_end`, and columns `eps_basic, eps_diluted, bps,
  ocfps, roe, roe_diluted, net_margin, gross_margin, revenue_yoy, net_income_yoy,
  debt_to_asset_ratio, inventory_turnover, operating_cash_to_revenue`. Coverage ≈ full training
  universe (3638 syms).
- **The valuation silver dir is empty**, but every PIT-safe *input* to build PB/PE/PCF exists
  (per-share metrics + daily close). `enrich_panel_fundamentals.py` already builds `pb=close/bps`
  (deliberately skipping `pe_ttm` — needs TTM EPS de-cumulation — and `turnover_rate` — needs
  shares outstanding, which we do NOT have).

**Verified data facts (2026-07-08):** `eps_basic`/`ocfps` are YTD-cumulative (000001.SZ 2024:
0.66→1.23→1.94→2.15, resets 2025Q1) ⇒ TTM needs de-cumulation. `bps` is point-in-time ⇒
`PB=close/bps` is directly PIT-safe. No shares-outstanding on disk ⇒ **PS, EV/EBITDA, market-cap,
turnover_rate are NOT buildable without fabrication and are excluded** (honesty over coverage).

## 1. Reuse decisions (no duplicate engines)

| Need | Reuse | New (minimal) |
|------|-------|---------------|
| PIT fundamentals source | `silver/fundamentals/metrics_panel.parquet` (as-is, no rebuild) | — |
| PIT backward as-of merge | `merge_asof` pattern proven in `enrich_panel_fundamentals.py` | — |
| Valuation math home | extend `src/quantagent/fundamental/financial_features.py` (the designated "PIT financial features" module) | `build_valuation_ttm_features()` fn (TTM de-cumulation + ratios + percentiles) |
| Training-set merge harness | `scripts/augment_training_dataset.py` (row-count-invariant merge) | extend `FINCOLS` + point at plus7clean |
| Trainer feature pickup | `horizon_models.select_features` name patterns (already whitelist these names) | — |
| WF eval / costs / T+1 | corrected `strict_v8` simulator (post-INC-E1), `baseline_protocol`, `exp008_walkforward_eval` | — |

Net new code ≈ one function + one thin materialization script; everything else is extension.

## 2. Frozen feature set (PIT-safe; all via `available_at ≤ trade_date` backward join)

**Direct fundamentals (as-reported):** `roe, roe_diluted, net_margin, gross_margin, revenue_yoy,
net_income_yoy, debt_to_asset, inventory_turnover, operating_cash_to_revenue`.

**Per-share valuation (need per-share metrics only, NO shares):**
- `pb = close / bps` (point-in-time)
- `eps_ttm` = trailing-4-quarter de-cumulated EPS (A-share method: `annual(prev FY) + YTD(current) −
  YTD(prior-year same quarter)`; Q1 uses cumulative directly)
- `pe_ttm = close / eps_ttm` (NaN when `eps_ttm ≤ 0` — negative-earnings PE is meaningless)
- `ocfps_ttm` (same de-cumulation), `pcf = close / ocfps_ttm` (NaN when ≤ 0)
- `earnings_yield = eps_ttm / close`, `ocf_yield = ocfps_ttm / close` (well-defined for negatives)

**Valuation percentiles (per `trade_date`, cross-sectional; cheap ⇒ high):**
- `valuation_percentile` = mean of pct-rank(`earnings_yield`) and pct-rank(`book_yield=1/pb`)
  (name matches the trainer pattern)
- own-history: `pb_own_pctile_2y` = pct-rank of `pb` vs its own trailing 504-td window
  (re-rating / compression signal)

**Composites (cross-sectional z-scores, per date):**
- `quality_composite` = z(roe)+z(net_margin)+z(gross_margin)+z(operating_cash_to_revenue)−z(debt_to_asset)
- `growth_composite` = z(revenue_yoy)+z(net_income_yoy)

**Missingness:** set existing `missing_fundamentals` / `missing_valuation` truthfully so the model
can learn the coverage gate rather than seeing silent zeros.

**Explicitly EXCLUDED (no clean PIT data — will NOT fabricate):** PS, EV/EBITDA, dividend yield
(no per-share dividend), forward/analyst estimates (no timestamped consensus on disk),
turnover_rate & market_cap (no shares). These are separate future data-acquisition tickets.

## 3. Build → merge → dataset (CPU-only, PIT-safe)

1. `build_valuation_fundamental_features.py` (thin): read metrics_panel + market_panel close →
   compute the §2 block per (symbol, trade_date) → write
   `runtime/data/v7/silver/valuation/val_fund_features.parquet`.
2. Extend `augment_training_dataset.py`: backward-safe left-merge the block onto plus7clean on
   (symbol, trade_date) → `…_v89_plus7clean_fund.parquet`. **Assert row count unchanged (no
   fan-out).** Emit a new `feature_schema.json` (feature_version `plus7clean_fund`, fresh
   schema_hash) so the schema-parity gate stays armed.

## 4. Leakage audit (mandatory gates before any training)

- G-PIT-1: for every merged row, `available_at ≤ trade_date` (assert, count violations = 0).
- G-PIT-2: a value on date *t* uses only statements with `announce_date < t` (available_at = ann+1).
- G-PIT-3: TTM de-cumulation spot-check vs a hand-computed symbol (000001.SZ 2025Q3 TTM EPS).
- G-PIT-4: cross-sectional percentiles on date *t* use only date-*t* rows (no pooling across dates).
- G-PIT-5: row-count invariant on merge; no future forward-fill past the next `available_at`.
- G-PIT-6: quarantine guard armed; no fresh-holdout (2026-05-19+) contact in any build/eval.

## 5. What this unblocks (separate pre-registrations, NOT in this ticket)

- **H-021 (GPU retrain + WF ablation):** retrain the 3 sleeves on `plus7clean_fund`, walk-forward
  under the CORRECTED simulator, ablation **technical-only vs +fundamental vs +valuation vs full**,
  ranked by OOS cost-adjusted CAGR; nested WF; PBO/DSR; 25/50 bps cost survival; capacity. Needs
  GPU pre-registration (budget/gates) per the mission — will be written after §3–§4 pass.
- **H-022 (T+1 做T):** reuse `backtest/tplus1_engine.py`; test inventory-based 做T lift vs no-T
  baseline. Separate ticket.

## 6. Acceptance for THIS ticket (H-020, data-engineering only)

ACCEPT when: the val_fund block builds, all §4 PIT gates pass, coverage ≥ 90% of training rows
for the core block (pb/roe/margins) with honest missingness flags, and the merged dataset passes
the row-count invariant + schema emit. This ticket does **not** by itself change any model or
production; it produces a training input for H-021. No CAGR claim is made here.

## 7. Results / verdict
*(filled after the build+audit run)*
