# 多 Agent 量化架构设计

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **实现**：`src/quantagent/governance/{envelopes,agents,protocol,audit}.py`

## 1. 设计目标

本架构要解决的**不是**"多个模型如何协作"，而是**"系统如何拒绝自欺"**。

三条承诺：

1. 一次批准必须以**内容哈希**指明它读过哪些产物；
2. 守护某一类风险的 agent 可以**单独终止**决策，任何数量的高置信度批准都无法推翻；
3. **未做的检查永远不等于通过的检查**。

## 2. Agent 不是提示词，是能力边界

每个角色声明：输入、允许工具、读范围、写范围、输出 schema、证据要求、
**否决权等级**、**失败行为**。这些声明由协议层强制执行，因此"风控可以否决"是系统
属性，而不是一句模型可能遵守也可能不遵守的指令。

| Agent | 权限 | 失败行为 | 否决域 |
| --- | --- | --- | --- |
| `orchestrator_governance` | **VETO** | FAIL_CLOSED | 禁评窗泄漏、实盘交易、证据缺失 |
| `data_acquisition` | ADVISORY | FAIL_CLOSED | — |
| `data_quality_forensics` | **VETO** | FAIL_CLOSED | 语义未知、完整性失败、对账未验证 |
| `market_microstructure` | **VETO** | FAIL_CLOSED | 保真度夸大 |
| `stock_selection` | ADVISORY | FAIL_OPEN | — |
| `strategy_research` | ADVISORY | FAIL_OPEN | — |
| `backtest` | **VETO** | FAIL_CLOSED | 执行假设无效、泄漏 |
| `risk` | **VETO** | FAIL_CLOSED | 限额突破、容量、回撤、操作风险 |
| `trading_execution` | ADVISORY | FAIL_CLOSED | — |
| `independent_challenger` | ADVISORY | FAIL_CLOSED | — |
| `compliance_data_rights` | **VETO** | FAIL_CLOSED | 授权、再分发、原始数据入库 |

**所有持否决权的 agent 一律 FAIL_CLOSED**（有测试强制）。理由：若崩溃等于中立，
那么让数据质量检查崩溃就成了通过它的一种手段。

关键的职责隔离：`data_acquisition` 的写范围**不含**任何 validation 路径——
取数的 agent 无权决定数据是否合格（有测试断言）。

## 3. 决策信封

```json
{
  "decision_id": "...", "agent": "...", "hypothesis_or_action": "...",
  "input_artifact_hashes": {}, "data_scope": {}, "method": "...",
  "quantitative_evidence": {}, "assumptions": [], "known_limitations": [],
  "hard_blockers": [], "verdict": "APPROVE | REJECT | BLOCK | NEEDS_EVIDENCE",
  "confidence": 0.0, "output_artifacts": []
}
```

结构性有效性规则：

- `APPROVE` 若缺 `input_artifact_hashes` → 无效（"说不出读过什么的批准不是证据"）；
- `APPROVE` 若缺 `quantitative_evidence` → 无效（"判定须基于测量，不是印象"）；
- `APPROVE` 若缺 `method` → 无效；
- `BLOCK` 若未列 `hard_blockers` → 无效。

**置信度只作为给人看的元数据，协议从不用它决定结果**——不做加权、不做平均、
不设阈值。

## 4. 审批顺序

```
data_acquisition → data_quality_forensics → [market_microstructure]
→ stock_selection → strategy_research → backtest → risk
→ independent_challenger → compliance_data_rights → orchestrator_governance
```

`market_microstructure` **条件参与**：仅当决策涉及日内 / tick / Level-2 时必需。
把它强加到纯日线决策上只会让它变成橡皮图章。

**强制参与**（无论范围）：`data_quality_forensics`、`risk`、
`compliance_data_rights`、`orchestrator_governance`。缺席任一 → `NEEDS_EVIDENCE`。

## 5. 结果判定（严格优先级，非投票）

1. **结构性有效性** —— 无证据的 `APPROVE` 被**降级**为 `NEEDS_EVIDENCE`（不是丢弃，
   也不是让整个决策失败：它要求补齐缺失的证据），并记入 `downgraded_envelopes`；
2. **硬否决** —— 任一持否决权 agent 的 `BLOCK` → `BLOCKED`，**不查置信度**；
   无否决权者的 `BLOCK` 降级为普通 `REJECT` 并记录说明；
3. **强制覆盖** —— 缺席的强制 agent → `NEEDS_EVIDENCE`，说明为
   "an absent check is not a passed check"；
4. **其余** —— 有 `REJECT` → `REJECTED`；有证据缺口 → `NEEDS_EVIDENCE`；
   全部必需 agent 批准 → `APPROVED`。

即使结果明确，只要批准与否定并存，`disagreements` 仍会记录分歧双方。

## 6. 实盘交易的前置拒绝

`DecisionProtocol.decide()` 在**咨询任何 agent 之前**匹配动作文本中的实盘标记
（`live trading`、`real account`、`实盘`、`order_send` 等），命中即抛
`LiveTradingAttempt`，并把拒绝本身写入审计日志。

即：本任务下不存在任何"经过审批后可以开实盘"的路径。

## 7. 审计日志

- 位置：`runtime/governance/`（**Git 之外**——能被 rebase 改写的治理记录不是治理
  记录）；
- 格式：JSONL，逐条带 `prev_hash` / `entry_hash` 的哈希链；
- 篡改或删除任意一行都会使该行之后的验证失败（两种情形均有测试）；
- **诚实声明**：这是**防篡改可检测（tamper-evident）**，不是**防篡改（tamper-proof）**。
  能写该文件的人可以重写整条链。文档与代码注释均如实说明，不夸大为"不可变"。

## 8. 与既有模块的关系

本包**不替代**已有组件，二者层级不同：

- `quantagent.risk.decision_chain` —— 组合层的 15 道日频闸门（把打分变成目标权重）；
- `quantagent.agents.*` —— LLM 分析师视角与证据抽取；
- `quantagent.governance.*` —— **研究/部署决策**层的证据门与否决权（本次新增）。
