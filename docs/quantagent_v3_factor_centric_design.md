# V3 Factor-Centric Design / 因子中心设计

V3 把 factor 作为 research first-class object。V4 保留这个原则，并新增 factor DAG 与 lifecycle governance，用来判断 factor group 是否进入 portfolio gate。

## 因子生产 / Factor Production

因子从 raw OHLCV、fund flow、sector exposure、fundamentals 和 event features 中生成。每个 `FactorMeta` 现在包含 group、frequency、lookback、PIT safety、expected direction、capacity proxy、crowding proxy、owner 和 version。

## 生命周期 / Lifecycle

`FactorLifecycleReport` 包含 rolling IC、Rank IC、ICIR、positive ratio、Newey-West t-stat、decay、monotonicity、turnover、capacity、crowding、correlation 和 live drift。

推荐状态 / Recommended status：

```text
active | degraded | retired | watch
```

## 测试 / Tests

```powershell
python -m pytest tests/test_factor_lifecycle_v4.py tests/test_factor_evaluation.py
```
