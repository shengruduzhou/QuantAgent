# QuantAgent SOTA Upgrade v3 / V3 能力回顾

V3 已经提供 AFML labels、PurgedCV、HRP/HERC、iTransformer、conformal prediction、A-share T+1 backtest、Agent BL views 和 QMT stub。V4 不删除这些能力，而是通过 wrappers 和 adapters 把它们接入生产闭环。

## 已保留能力 / Preserved SOTA

- `quant_math/triple_barrier.py`：triple-barrier labels。
- `quant_math/purged_cv.py`：PurgedKFold、CPCV、PBO。
- `quant_math/hrp.py`：HRP/HERC portfolio baselines。
- `models/itransformer.py`：iTransformer 与 PatchTST-style heads。
- `agents/bl_views.py`：Agent views 到 Black-Litterman posterior。
- `backtest/engine.py`：A-share event-driven backtester。

## V4 升级方式 / V4 Upgrade Path

V4 在 V3 上新增 `AshareRuleEngine`、`FeatureStore`、`FactorDAG`、`FactorLifecycleReport`、`V4MultiTowerModel`、`AgentRouter`、`solve_v4_portfolio`、`QMTGateway(dry-run)` 和 CLI services。

## 安全约束 / Safety

V3 的 QMT stub 在 V4 中被强化为 dry-run default。任何 live submit 都必须显式配置、通过 kill switch 和 reconciliation。

## 测试 / Tests

```powershell
python -m pytest tests/test_quant_math_sota.py tests/test_backtest_v4.py tests/test_qmt_gateway_v4.py
```
