# V4 数据层 / Data and Feature Store

V4 数据层目标是 point-in-time consistency。FeatureStore 统一处理 OHLCV、fundamentals、fund flow、events、sector exposure 和 universe snapshots。

## 核心组件 / Components

- `PITJoiner`：按 announcement time 和 event time 做 as-of join，避免 future leakage。
- `EventStore`：标准化 symbol、event_time、event_type、sentiment_score、policy_exposure、confidence。
- `UniverseBuilder`：支持 custom/CSI placeholders、tradability、liquidity、ST、新股和停牌过滤。
- `FeatureStore`：复用 Alpha101、CICC-like factors、sector rotation、fund flow synthetic features。

## 输入输出 / Inputs and Outputs

输入最小字段：

```text
trade_date,symbol,open,high,low,close,volume,amount
```

输出包含 `feature_version`、`asof_time`、technical features、factor columns、event features 和 labels。

## 测试 / Tests

```powershell
python -m pytest tests/test_feature_store_v4.py
```
