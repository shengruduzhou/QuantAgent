# ACCEPTANCE_RULES — 改进验收规则（Phase 2 定稿，即刻生效）

> 一个改进 = 对生产链任意环节（数据/因子/模型/blend/book/风控）的一次候选变更。
> **全部门槛为 AND 关系；任何一条不过 = REJECT；不存在"CAGR 高到可以豁免"。**

## R0. 对照基线（beat what）

- 临时 reference（至 walk-forward 重基线完成）：
  - **v8.9 rankfix k50 全 OOS 混窗 +17.25%**（`clean_oos`²，单看）与
  - **plus7 3-sleeve 先验 blend holdout k10 +8.3%**（`clean_oos`¹）构成的信任包络；
  - 以及 **等权全A benchmark**（同窗机械计算）。
- 改进声明的形式必须是：**walk-forward 逐折分布优于同协议下的基线分布**（中位数提升且亏损折占比不升），而非单窗点值比较。
- **禁止**以 38.6%（`contaminated_holdout`）作对照或目标。

## R1. 选择卫生（selection hygiene）

1. 候选的一切选择只用 TRAIN + SEARCH 分区（≤2025-08-31）；触碰 QUARANTINE 窗即 REJECT（评测器守卫 + census 复查）。
2. FRESH 窗只允许预注册配置一次性评测（见 EVALUATION_PROTOCOL_V2 §2）。
3. 搜索必须**预先声明**：候选网格、试验数 N、选择指标 —— 写入 `HYPOTHESIS_REGISTRY.md` 后才允许跑。
4. 累计试验数入账：同族第 k 轮搜索的 N 累加计算 PBO/DSR。

## R2. 统计稳健门

| 指标 | 门槛 |
|---|---|
| PBO（CSCV, S=8） | ≤ 0.25 |
| Deflated Sharpe Ratio（按累计 N 校正） | 显著（p<0.05 等价） |
| walk-forward 亏损折占比 | 不高于基线，且 ≤ 40% |
| 邻域稳定性 | 赢家参数 ±1 格邻域中位表现 ≥ 赢家的 60% |
| 子窗一致性 | SEARCH 窗三分段方向一致（同号超额） |

## R3. 执行现实门（variant-C 强制）

1. 一切收益数字出自 `baseline_protocol.py` variant **C**（T+1、eligible ranking、涨跌停/停牌/ST、lot、成交量参与≤10%、成本+slippage）。代理回测（如 fwd1d-成本近似）只可用于搜索内部预筛，不得出现在验收表。
2. 成本敏感：slippage 8→15bps 时年化衰减 ≤ 40%；15bps 下仍为正超额（vs 等权全A）。
3. **日换手上限 0.35**（单边，占组合比）。现 regime 赢家 0.81–0.86/日直接不合格。豁免需给出成本后逐折证明 + 明示容量牺牲。
4. 平均持有期 ≥ 3 个交易日（与换手门互为印证）。

## R4. 风险与容量门

| 维度 | 门槛 |
|---|---|
| walk-forward 折内最大回撤 | 中位 ≤ 20%，最差折 ≤ 30% |
| 单行业权重 | ≤ 30%（decision chain 现有约束不得放松） |
| 单票权重 | ≤ 10%（k=10 等权时天然=10%，不得更集中） |
| top-K | ≥ 10；k<10 的高集中书需额外容量证明 |
| 容量估算 | 以 20 日 ADV 的 10% 参与率反推可容纳资金 ≥ 目标资金（¥1M 基准，报告 ¥10M 情形） |
| beta/alpha 分解 | 报告 Jensen alpha（`backtest/beta_decomposition.py`）；纯 beta 放大不算改进 |

## R5. 资源门（服务器安全）

| 资源 | 门槛 |
|---|---|
| CPU RAM 峰值 | ≤ 48 GiB（62 GiB 机器留余量），大表必须列裁剪/分块/lazy |
| GPU VRAM 峰值 | ≤ 20 GiB（3090 24G），记录 `torch.cuda.max_memory_allocated()` |
| 新增磁盘 | 单实验 ≤ 5 GB，且实验结束清理中间物（磁盘仅剩 186 GB） |
| 运行时长 | 声明上限并遵守；无界搜索禁止 |
| OOM 行为 | 训练类任务须能捕获 CUDA OOM 并存诊断退出，不得拖垮机器 |

## R6. 台账（无台账 = 无实验）

每个实验在 `EXPERIMENT_LEDGER.md` 登记：git hash、数据集 schema_hash、TRAIN/SEARCH/评测窗、完整命令、改动文件、假设、全部指标（含失败）、RAM/VRAM 峰值、运行时长、ACCEPT/REJECT + 理由。失败实验**必须**入账。

## R7. 最终判定流程

```
候选 → R1 选择卫生 → R3 variant-C 评测(SEARCH/WF) → R2 统计门 → R4 风险容量门
     → R5 资源门 → 台账 → [等 FRESH 窗 ≥120 交易日] → 预注册一次性终评 → ACCEPT
```
中途任何一步失败即止损；"先合入后补验证"不允许。

## R8. 不可协商项（复述硬规则）

不看未来数据；不在 final holdout 上调参；不用合成回填数据；不绕过 DSL 写自由因子 Python；不静默改 feature schema；不隐藏失败实验；不无界搜索；不因 OOM 风险裸跑；不产生实盘指令。
