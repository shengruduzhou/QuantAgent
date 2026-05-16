# V7 PIT Data Contract / V7 PIT 数据契约

## 原则 / Principles

V7 的任何 production / realdata 数据必须满足 Point-in-Time safety：
- 行情 close / volume / amount 及 close-derived 技术特征不能以 `trade_date` 为可见日期。`build_market_features` 把 `available_at` 设置为 **下一交易日**，避免 same-day close lookahead。
- 财务 / 公告 / 估值类行必须带 `report_period`、`ann_date`、`available_at`。如果 `ann_date` 缺失，按 vendor 文档 + `available_lag_days` 估计并打 `point_in_time_valid=false`，不能进入 strict PIT cache。
- `available_at` 永远是“信息可见日”，不能等于 `report_period`。

## Schema

### Market Panel (silver/market_panel)
| column | type | notes |
| ------ | ---- | ----- |
| symbol | str | A 股代码（600519.SH 等） |
| trade_date | date | 交易日 |
| open / high / low / close | float | 后复权或不复权由 provider 标注 |
| volume / amount | float |  |
| available_at | date | close-derived features 设为下一交易日 |
| is_suspended / is_st / is_limit_up / is_limit_down | bool | optional，缺失会在 manifest warnings 中标记 |

### Financial Statements (silver/fundamentals)
| column | type | notes |
| ------ | ---- | ----- |
| symbol | str |  |
| report_period | date | 财报截止日 |
| ann_date | date | 公告日 |
| available_at | date | 可见日 = ann_date + available_lag_days |
| statement_type | str | income / balance_sheet / cashflow / financial_indicator |
| 字段名 | float | 归一化后的英文列名 |
| currency / unit | str | 可选 |
| source | str | provider 名 |
| source_url | str | 可选 |
| fetch_time | str | ISO8601 UTC |
| raw_hash | str | sha256(归一化前原始行) |
| point_in_time_valid | bool |  |

### Disclosures (silver/disclosures)
保留公告原文、announcement category、ann_date、available_at、raw_hash。

### Training Dataset (gold/training_dataset)
- 通过 `build-training-dataset-v7` 由 silver 输入 + multi-horizon 标签 as-of join 产出。
- `available_at <= trade_date` 是强约束。
- Feature schema（`<output>.feature_schema.json`）显式列出 `feature_columns`、`label_columns`、`entity_columns`、`timestamp_columns`、`forbidden_columns`。
- Forbidden 列包括 `open/high/low/close/volume/amount` 与所有 `forward_return_*` / `label_end_*`。

## Manifests

每个 silver/gold artifact 都伴随一份 `data/v7/manifests/<dataset>.json`，由 `quantagent.data.manifest.build_manifest_for_frame` 生成：

```
dataset_name, vendor, fetch_time, start_date, end_date,
symbols, universe, raw_paths, output_paths,
row_count, column_count, schema_version,
content_hashes (sha256 per output file),
missing_columns, duplicate_row_count, duplicate_rate,
pit_violation_count, warnings,
quality_status (passed | warning | failed),
extra
```

`quality_status="failed"` 的 dataset 不允许进入下游训练或回测。

## Quality Gates

`src/quantagent/data/v7_quality_gates.py` 提供 blocking checks：
- 行数 / symbol 数 / 日期数 minimum。
- PIT 违反计数必须为 0。
- mock/synthetic source 自动标记为非 production-ready。
- 单因子主导率不能超过阈值（避免 overfit）。

任何 gate 失败都会 raise，禁止静默通过。
