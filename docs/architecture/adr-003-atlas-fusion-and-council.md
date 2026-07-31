# ADR-003 — ATLAS：因子融合层与决策议事会

- 状态 / Status: Accepted
- 日期 / Date: 2026-07-31
- 关联 / Related: [ADR-001](adr-001-institutional-workstation-shell.md)、
  [ADR-002](adr-002-vnext-only-cutover.md)、
  [机构量化参考架构](../research/institutional_quant_reference_architecture.md)

## 背景 / Context

工作站已有数据、因子、训练、回测、风控、治理模块，但平台的首要目标——
**用机器学习/优化发现有效的因子融合策略，并用可靠回测与风控验证**——
在 Web 上没有落点：

- `quantagent.optimization.ga_weight_optimizer`（多目标 GA + purged walk-forward）与
  `quantagent.ensemble.blend_optimizer`（混合权重搜索）只能在命令行使用；
- 试验次数没有被系统性计数，Deflated Sharpe 因此无法可信计算；
- 多 Agent 审查只存在于策略校验内部，且是只读的，没有证据展开、没有推翻审计。

## 决策 / Decision

### 1. 新增 L2 融合层：`quantagent.fusion`

搜索协议固定为五步，顺序本身就是保证：

1. 枚举候选**方案**（scheme），此刻确定 `n_trials`；
2. 构造 purged / embargoed 扩张式滚动折；
3. 每折每方案：**只在训练段拟合，只在样本外段评估**；
4. 汇总候选指标，并从样本外收益矩阵估计 PBO（CSCV）；
5. 用步骤 1 的 `n_trials` 计算稳健性，取 Pareto 前沿，再按偏好排序。

关键约束：

- **`n_trials` 不可声明**。CLI、governed command、API、UI 都没有该参数。
  对照方案（等权、随机单纯形、单因子）同样计数——删掉对照来抬高 DSR 是造假。
- **不产出单一最优**。四目标（最大超额 / 最大年化 / 最小回撤 / 样本外稳健）
  相互冲突，输出非受支配集；偏好权重只排序，不影响生成与前沿归属。
- **因子先做逐日横截面排名归一再融合**。原始 alpha 列量纲相差数量级，
  在原始尺度上的权重编码的是量纲而不是信念。
- **回撤按调仓频率净值计算**。面板只有前瞻收益，日频净值不存在；
  这是日频回撤的下界，UI 必须这样说明。

### 2. 新增 L5 议事会：`services.quant_api.services.council`

七个角色，各自只在声明的 `vetoScope` 内否决，只消费结构化证据。三条硬规则：

1. 每条裁决附带它读取的字段（`evidence`），可被复核而不是被信任；
2. **证据缺失记 `unknown`，不记 `pass`**；`unknown` 不阻塞研究，但不算放行；
3. 人工推翻必须带 author 与理由，写入 append-only 日志，原裁决并列保留。

### 3. 任务状态机新增 `starting` 与 `paused`

原实现在 `Popen` 之前就把状态置为 `running`，存在一个窗口：操作者看到
"running" 却因为进程尚未登记而无法暂停。现在状态在进程登记后才变为
`running`；`pause`/`resume` 用 SIGSTOP/SIGCONT，是调度控制而非资源释放。

## 后果 / Consequences

**正面**

- 平台首要目标在 Web 上可配置、可启动、可暂停、可监控、可比较、可治理。
- 统计诚实性成为结构性质而不是纪律要求：试验次数无法被人为压低。
- 议事会把"为什么不能晋级"变成可点开的证据，而不是一个红色徽章。

**代价**

- 搜索成本随 `n_trials × folds` 线性增长；增加对照会让结论更保守也更慢。
- 暂停的任务仍占用内存与显存，只让出 CPU/GPU 时间片。
- 议事会阈值（`CouncilThresholds`）是平台自定的研究口径，不是行业标准；
  它们对操作者可见且会被实际检查，但不应被引用为外部基准。

## 备选方案 / Alternatives considered

- **直接复用 `optimize-ga-weights-v8`**：可以更快上线，但它不计数试验、
  不产出前沿、不估计 PBO，等于把统计诚实性留给使用者自觉。已作为
  `genetic` 方案被包进新协议中，而不是被取代。
- **让议事会用 LLM 生成意见**：会引入不可复核的自然语言判断。
  当前实现全部是确定性检查；LLM 若要接入，应作为额外的"挑战者"角色
  产出可被结构化检查复核的假设，而不是替代任何角色的否决权。
