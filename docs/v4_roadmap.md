# V4 Roadmap / 路线图

V4 当前目标是最小可落地闭环，而不是完整生产交易平台。后续迭代应在不破坏安全边界的前提下逐步增强。

## 已完成 / Done

- A-share rule engine with board awareness。
- Point-in-time FeatureStore synthetic flow。
- Factor DAG and lifecycle governance。
- V4 multi-tower model and composite loss。
- Structured Agent views and BL adapter。
- Portfolio optimizer wrapper and V4 backtester。
- QMT dry-run gateway、audit、reconciliation、kill switch。
- CLI services for synthetic pipeline。

## 下一步 / Next Steps

- 接入真实但 point-in-time clean 的 A 股数据源。
- 扩展 sector/style risk model 和 capacity model。
- 增加 walk-forward training runner 的持久化 checkpoint。
- 增加 paper trading dashboard 和 daily risk report。
- 在长周期 paper trading 后再评估 live QMT gating。

## 验收 / Acceptance

```powershell
python -m pytest tests/
```
