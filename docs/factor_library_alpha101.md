# Alpha101 Factor Library / Alpha101 因子库

Alpha101 是 V3 已有的重要 factor baseline。V4 不重写它，而是在 `FeatureStore` 中通过 registry-based computation 接入。

## 使用方式 / Usage

`FeatureStoreConfig(enable_alpha101=True)` 会调用 `compute_alpha101`，并把 long-form factor frame pivot 成 wide feature columns，例如 `alpha001`、`alpha029`。

## 治理 / Governance

Alpha101 因子可以进入 `FactorDAG` 和 `FactorLifecycleReport`，用 IC、Rank IC、ICIR、turnover、capacity、crowding 和 correlation 来判断 active/degraded/retired/watch。

## 测试 / Tests

```powershell
python -m pytest tests/test_alpha101.py tests/test_factor_lifecycle_v4.py
```
