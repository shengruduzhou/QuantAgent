# V7 PIT Data Contract / PIT 数据契约

## Principles

- `available_at` 表示信息可见日，不能等同于 `report_period`。
- Close-derived market features 必须使用 next trading row availability，避免 same-day close leakage。
- Financial rows 必须同时包含 `report_period`、`ann_date`、`available_at`。
- Policy、announcement、news 原文必须保留 `source / published_at / available_at / raw_hash / confidence` 并进入 evidence layer。
- Production real-data path 不能使用 synthetic fallback。

## Market Panel

Required columns:

- `symbol`
- `trade_date`
- `open`, `high`, `low`, `close`
- `volume`, `amount`
- `available_at`

Optional tradability columns:

- `is_suspended`
- `is_st`
- `is_limit_up`
- `is_limit_down`

Missing optional tradability flags are reported in manifests and diagnostics.

## Financial Statements

Required PIT columns:

- `symbol`
- `report_period`
- `ann_date`
- `available_at`
- `source`

Statement merge must use `pit_wide_merge_statements`:

- statement type prefixes are applied, e.g. `income_revenue`, `balance_total_assets`, `cashflow_operating_cash_flow`, `indicator_*`。
- merge key is PIT-safe。
- duplicated `(symbol, report_period, available_at)` raises。

## Manifests

Every silver/gold artifact must have a manifest under:

```text
E:\Project\QuantAgent\runtime\data\v7\manifests\<dataset>.json
```

Manifest fields include provider/vendor, source paths, generated_at, row_count, date_range, symbols, schema report, PIT violations, duplicate rate, warnings and content hash.

## Training Dataset PIT Invariants

Tests must cover:

- no `available_at > trade_date` in training features。
- no label leakage into feature columns。
- no raw forward labels in inference schema。
- no same-day close-derived leakage。

Any failed PIT invariant blocks training or readiness reporting.
