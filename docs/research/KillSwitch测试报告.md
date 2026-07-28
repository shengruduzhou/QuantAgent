# Kill Switch 测试报告

- **生成时间**：2026-07-29（UTC）
- **源提交**：`17ada19996fc5f1fceae06bac60ff5ab05d4a922`
- **实现**：`src/quantagent/paper/risk.py`
- **测试**：`tests/paper/test_risk_engine.py` —— 36 passed

## 1. 作用域

四级：`ORDER` / `STRATEGY` / `PORTFOLIO` / `GLOBAL`。

| 行为 | 测试 |
| --- | --- |
| **GLOBAL 熔断压制其下所有作用域** | `test_global_halts_every_scope` |
| STRATEGY 作用域彼此隔离 | `test_strategy_scope_is_isolated` |
| 未知作用域被拒 | `test_unknown_scope_rejected` |
| 熔断后新订单被拒 | `test_triggered_switch_blocks_new_orders` |

## 2. 触发条件（实测）

| 触发器 | 触发的作用域 | 测试 |
| --- | --- | --- |
| 单日亏损超限 | PORTFOLIO | `test_daily_loss_triggers_kill_switch` |
| 回撤超限 | PORTFOLIO | `test_drawdown_triggers_kill_switch` |
| **账本链校验失败** | **GLOBAL** | `test_broken_ledger_triggers_global_kill` |
| **对账失败** | **GLOBAL** | `test_reconciliation_failure_triggers_global_kill` |
| 重复拒单 | 由 operational 检查报出 | `test_repeated_rejections` |
| 陈旧行情 / 心跳 | 由 pre-trade / operational 报出 | ✅ |
| 模型或数据集未批准 | pre-trade 拒单 | ✅ |

账本或对账失败触发**全局**熔断的理由：此时系统对"自己做过什么"的记录已不可信，
继续交易毫无意义。

## 3. 熔断是**闩锁**的

一旦触发即保持触发。清除需要**显式人工确认**：

```python
switch.clear(SCOPE_GLOBAL)                          # 抛 RiskRejection
switch.clear(SCOPE_GLOBAL, human_confirmation=True) # 才生效
```

理由写在代码里：自动重置会**彻底废掉这个控制**。

## 4. 风控拒绝是终局的（本报告最重要的一条）

- `RiskEngine.enforce()` **没有** force / override 参数——有测试用
  `inspect.signature` 断言参数集恰为 `{self, decision}`；
- `RiskDecision.to_dict()` 显式输出 `"override_available": false`；
- 拒绝抛 `RiskRejection`，消息写明"**cannot be overridden by a strategy
  component or a vote**"。

要防的失败模式很具体：**某个策略组件把检查它的组件说服了**。因此这条路径
**在代码里不存在**，而不是靠约定。

## 5. 全部决策入账本

每次风控判定（通过或拒绝）都写入哈希链账本
（`RISK_APPROVED` / `RISK_REJECTED`），因此**拒单理由跨重启存活**，
且账本链在写入后仍校验通过（`test_decisions_are_written_to_the_ledger`）。
