# 多 Agent 决策与否决测试报告

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **测试文件**：`tests/test_governance_protocol.py`
- **结果**：**29 passed**

## 1. 测试设计原则

每条测试对应一种**治理系统在现实中失效的方式**，而不是对应一个函数：

- 没有证据的批准；
- 多数票压过唯一真正检查了数据的 agent；
- 未做的检查被当成通过；
- 崩溃的否决方被当成同意；
- 实盘指令混进研究流程。

## 2. 硬否决测试

### 2.1 数据质量否决压过全部批准（核心用例）

```
9 个 agent APPROVE，confidence = 0.99
data_quality_forensics BLOCK，confidence = 0.10
  hard_blockers = ["tick semantics are UNKNOWN_SEMANTICS"]
```

**结果**：`BLOCKED`，`approved = False`。

断言同时检查 `mean_confidence > 0.8`——即**高平均置信度确实存在，且确实没有起
任何作用**。这正是"高置信度不能推翻硬阻塞"的直接验证。

### 2.2 其余否决

| 测试 | 场景 | 结果 |
| --- | --- | --- |
| `test_risk_block_cannot_be_overridden` | 风控以"最大回撤 22.1% 超过 20% 限额"BLOCK | `BLOCKED` |
| `test_compliance_block_stops_the_decision` | 合规以"授权原始数据进入 Git 树"BLOCK | `BLOCKED` |
| `test_advisory_agent_block_is_downgraded_to_a_rejection` | 选股 agent（无否决权）BLOCK | 降级为 `REJECTED`，`blockers` 为空，并记录"without veto authority" |
| `test_veto_holders_are_the_declared_four_plus_domain_agents` | 否决权名单 | 含 DataQuality/Risk/Compliance/Governance；不含 StockSelection |

## 3. 证据门测试

| 测试 | 场景 | 结果 |
| --- | --- | --- |
| `test_approval_without_evidence_is_downgraded_not_counted` | backtest 以 0.99 置信度 APPROVE 但不引用任何产物 | `NEEDS_EVIDENCE`；该 agent **不计入** approvals；记入 `downgraded_envelopes` |
| `test_missing_mandatory_agent_is_not_an_implicit_pass` | 风控完全缺席 | `NEEDS_EVIDENCE`，`missing_mandatory` 含 risk，说明含 "absent check is not a passed check" |
| `test_all_approvals_with_evidence_are_approved` | 全部带证据批准 | `APPROVED` |
| `test_intraday_decision_requires_the_microstructure_agent` | tick 决策但缺微观结构 agent | `NEEDS_EVIDENCE` |
| `test_daily_decision_does_not_require_microstructure` | 纯日线决策 | `APPROVED`，微观结构 agent 未被咨询 |

## 4. 信封有效性测试

| 测试 | 断言 |
| --- | --- |
| `test_unknown_verdict_is_refused` | 非法判定（如 `"LGTM"`）构造即抛 `EnvelopeError` |
| `test_confidence_must_be_a_probability` | 置信度须在 [0,1] |
| `test_approve_without_evidence_is_structurally_invalid` | 缺 hashes 与 quantitative_evidence 均被指出 |
| `test_block_must_name_its_blockers` | BLOCK 必须列出 hard_blockers |

## 5. 实盘拒绝测试

```python
@pytest.mark.parametrize("action", [
    "enable live trading for the L1 book",
    "submit order to broker for 600000.SH",
    "开启实盘交易",
])
```

三种表述（含中文）均在**咨询任何 agent 之前**抛出 `LiveTradingAttempt`。
另有 `test_live_trading_refusal_is_audited` 断言拒绝事件写入审计日志
（`LIVE_TRADING_REFUSED`）。

## 6. 审计日志完整性测试

| 测试 | 场景 | 结果 |
| --- | --- | --- |
| `test_entries_chain_and_verify` | 正常追加 5 条 | `verify().valid = True` |
| `test_editing_an_entry_breaks_verification` | 篡改第 2 条的 payload | `valid = False` |
| `test_deleting_an_entry_breaks_verification` | 删除第 2 条 | `valid = False` |
| `test_protocol_persists_every_envelope_and_the_decision` | 完整决策流程 | 每个信封 1 条 + 决策 1 条，链验证通过 |
| `test_log_lives_outside_git` | 断言 `.gitignore` 含 `runtime/` | 通过 |

## 7. 角色声明测试

| 测试 | 断言 |
| --- | --- |
| `test_every_role_declares_scope_and_failure_behaviour` | 每个角色都有职责、读范围、输出 schema、失败行为 |
| `test_veto_holders_fail_closed` | **所有否决权持有者必须 FAIL_CLOSED**（崩溃不等于同意） |
| `test_data_acquisition_cannot_write_validation_thresholds` | 取数 agent 的写范围不含 validation |
| `test_execution_agent_is_paper_only` | 允许工具含 paper_broker，且无任何含 "live" 的工具 |

## 8. 汇总

```
29 passed
```

**未覆盖的部分（诚实声明）**：本次测试验证的是**协议机制**，不是 LLM agent 的
实际判断质量。真实 agent 是否会正确识别"tick 语义未知"仍取决于其实现；协议保证的
是——**一旦它这么判定，这个判定不会被投票推翻**。
