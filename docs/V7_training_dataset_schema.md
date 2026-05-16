# V7 Training Dataset Schema / 训练集 Schema

## Entry

```powershell
quantagent build-training-dataset-v7 `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\AI量化\data\v7\labels.parquet `
  --fundamentals-root E:\AI量化\data\v7\raw\akshare\fundamentals `
  --valuation E:\AI量化\data\v7\silver\valuation\valuation.parquet `
  --disclosures E:\AI量化\data\v7\silver\disclosures\disclosures.parquet `
  --horizons 1,5,20,60,120,126
```

默认输出：

- `E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet`
- `training_dataset.feature_schema.json`
- `E:\AI量化\data\v7\manifests\training_dataset.json`

## Join Semantics

- Market panel 先经过 `build_market_features`，close-derived features 使用 next trading row `available_at`。
- Financial / valuation / disclosure frames 必须带 `symbol + available_at`，as-of join 使用 backward direction。
- `feature_available_at <= trade_date` 是硬约束。
- Labels 来自 `build-labels-v7`：`forward_return_{h}d` 和 `label_end_{h}d`。
- Missing source 会写 missingness flag，例如 `missing_fundamentals`。

## Forbidden Inference Columns

Feature schema 明确列出：

- entity: `symbol`
- timestamp: `trade_date`, `available_at`
- labels: `forward_return_*`, `label_end_*`
- forbidden raw market columns: `open`, `high`, `low`, `close`, `volume`, `amount`

Inference feature columns不得包含 labels、label_end、raw forward labels 或 same-day close-derived leakage。

## Strict Checks

`V7TrainingDatasetConfig.strict_mode=True` 默认开启，以下情况会 raise：

- training dataset 为空。
- 缺少 `symbol / trade_date / available_at`。
- 没有可训练 features。
- 缺少任一 configured horizon label。
- label columns 泄漏到 feature columns。
- `(trade_date, symbol)` 重复。
- `available_at > trade_date`。
- source metadata 包含 `mock / synthetic / demo` 且用于 real-data path。

## Feature Groups

当前 feature groups 覆盖：

- market technical features
- Alpha101-style factor columns
- valuation factors
- financial statement ratios
- growth / profitability / leverage / liquidity ratios
- sector / industry neutralized features
- calendar / regime features

缺列时会通过 schema report 和 missingness summary 暴露，real-data path 不会静默造数。
