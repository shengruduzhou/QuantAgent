# V4 QMT 执行准备 / QMT Execution Preparation

V4 QMTGateway 是 safe interface，不是默认 live trading 通道。`QMTGateway()` 默认 `dry_run=True`，不会导入或调用真实 `xtquant`。

## 安全路径 / Safe Path

```text
target_weights
-> OrderManager
-> order intents with metadata
-> RiskGate / KillSwitch
-> reconciliation
-> QMTGateway dry-run audit
```

每个 order intent 记录 signal_id、model_version、feature_version、strategy_version、risk_check_result、timestamp。

## Live Trading Gate / 实盘门槛

只有在 `live_trading_enabled=true` 且 `dry_run=false` 时，才允许考虑 live QMT wiring。仍然必须通过 kill switch、risk gate、reconciliation 和 audit。

## 测试 / Tests

```powershell
python -m pytest tests/test_qmt_gateway_v4.py
```
