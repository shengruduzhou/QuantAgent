# CICC A-share Factor Library / 中金风格 A 股因子库

本页说明 CICC-like high-frequency 和 daily-compatible 因子在 V4 中的位置。它们作为 FeatureStore 的可选 factor sources，不要求外部 API。

## 因子组 / Factor Groups

V4 synthetic flow 会复用 `compute_cicc_high_freq_factors`，包括 last 30min return、daily Amihud、close-volume correlation、turnover concentration、money-flow strength 等 daily-compatible factors。

## 点时一致 / Point-in-time

这些因子只使用当前及历史 OHLCV/minute aggregate 信息。FeatureStore 输出会按 `trade_date, symbol` 稳定排序，并附带 `feature_version` 和 `asof_time`。

## 测试 / Tests

```powershell
python -m pytest tests/test_cicc_factors.py tests/test_feature_store_v4.py
```
