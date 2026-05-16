# V7 Training Dataset Schema / V7 训练数据集 Schema

## 入口 / Entry

```powershell
quantagent build-training-dataset-v7 \
  --market-panel data/v7/silver/market_panel/market_panel.parquet \
  --labels data/v7/labels.parquet \
  --fundamentals-root data/v7/silver/fundamentals \
  --valuation data/v7/silver/valuation/valuation.parquet \
  --disclosures data/v7/silver/disclosures/disclosures.parquet \
  --horizons 1,5,20,60,120,126 \
  --output data/v7/gold/training_dataset/training_dataset.parquet
```

实现：`src/quantagent/data/dataset_builder/v7_training_dataset.py`。

## Join 语义 / Join Semantics

- 市场面板先经过 `build_market_features`，得到带有 `available_at = next trading row` 的 PIT 特征。
- 财务 / 估值 / 披露 frame 必须含 `symbol + available_at`，as-of join 时使用 `direction="backward"`，保证 `feature_available_at <= trade_date`。
- 没有匹配到的源会写入 missingness flag：`missing_fundamentals` / `missing_valuation` / `missing_disclosures`。
- 标签来自 `build-labels-v7` 输出（`forward_return_{h}d` + `label_end_{h}d`）。
- inner-join market features × labels 后丢掉 `available_at` 缺失的行。

## 输出 / Outputs

- `training_dataset.parquet` —— 训练用 DataFrame。
- `training_dataset.feature_schema.json` —— feature/label/entity/timestamp/forbidden 列定义、horizons、source name、available_at policy。
- `data/v7/manifests/training_dataset.json` —— 行数、列数、PIT 违反计数、duplicate rate、文件 sha256、warnings、quality status。

如果 `--output` 指向 lake 外的路径，manifest 会写到 `<output>.manifest.json`。

## 列定义 / Columns

- Entity：`symbol`
- Timestamp：`trade_date`、`available_at`
- Labels：`forward_return_{h}d`、`label_end_{h}d`
- Forbidden（不可用于 inference）：`open / high / low / close / volume / amount` + 所有 `forward_return_*` / `label_end_*`
- Features：所有数值型 / bool 列减去 forbidden / label / entity / timestamp

## 数据质量 / Quality

- min_rows / min_symbols / min_dates 默认 100 / 2 / 5，可由 CLI 调整。
- `enforce_quality_gates=true`（默认）让 gate failure 直接 raise。
- `source_name` 默认 `realdata`；用于 mock-data 验证时显式传 `mock` 来跳过 production gate。
- `allow_synthetic_fallback=false` 永远；显式传 true 会被 builder 拒绝。
