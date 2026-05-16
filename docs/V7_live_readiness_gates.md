# V7 Live Readiness Gates / V7 实盘准备门槛

## 默认安全 / Safety Defaults

- `live_trading_enabled=false`
- `dry_run=true`
- `virtual_broker_only=true`
- `agents_can_emit_orders=false`
- Optimizer 只输出 `target_weights`
- 只有 `OrderManager` 把 target weights 转为 order intents
- QMT submit 之前必须通过 risk gate、kill switch、execution constraint simulation、reconciliation、audit replay

任何配置改成 live 之前必须有完整的 sign-off path。仓库中默认配置永远保持 dry-run。

## Acceptance Gates / 验收门槛

`src/quantagent/data/v7_quality_gates.py:evaluate_model_acceptance_gates` 是单一来源。Production-ready 必须全部通过：

| Gate | 阈值（默认） |
| ---- | ----------- |
| rank IC mean | > 0 |
| rank IC stability | > 0 |
| turnover-adjusted net return | > 0 |
| max drawdown | <= 25% |
| single factor dominance | <= 60% |
| adverse regime passed | True |
| paper trading report exists | required |
| uses_mock_or_synthetic | False |

CLI：

```powershell
quantagent v7-live-readiness-report \
  --metrics E:\AI量化\models\v7_alpha\metrics.json \
  --paper-report E:\AI量化\reports\v7\paper_trade_report.json
quantagent evaluate-alpha-v7 \
  --metrics E:\AI量化\models\v7_alpha\metrics.json \
  --paper-report E:\AI量化\reports\v7\paper_trade_report.json
```

输出 JSON 包含 `passed`、`failures`、`metrics`、`safety_defaults`。`passed=true` **不会** 自动开启实盘——仍然需要单独的 production toggle 与人工 sign-off。

## Execution / Backtest Gates

`walk-forward-backtest-v7` 与 `paper-trade-v7` 共享 `simulate_ashare_target_weights`，覆盖：

- T+1 sell restriction
- limit-up buy block / limit-down sell block
- suspension block
- ST 限制
- lot size 100
- volume participation cap
- slippage 曲线
- VirtualBroker cost model
- partial fills
- failed order audit
- reconciliation report（位于 `execution/reconciliation.py`）
- kill switch audit（位于 `execution/risk_kill_switch.py`、`risk/kill_switch.py`）

## Stock Pool Gate

如果 `apply_stock_pool_gate` 在 production 模式下返回空 universe，pipeline **不会** 静默回退到 `universe_members`：
- `gate_failure_reason="empty_after_stock_pool_hard_gate"`
- multi-horizon alpha = `{}`
- portfolio plan target_weights = `{}`
- risk report `risk_passed=false`

参见 `tests/test_v7_real_data_ready_components.py` / `tests/test_v7_exposure_and_pool_gate.py`。

## No Profit Guarantee

V7 不提供任何收益保证。所有 metrics、backtest report、paper-trade report 仅供研究和风险评估使用。任何 live execution 都必须经独立审查。
