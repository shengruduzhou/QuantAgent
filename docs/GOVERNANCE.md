# 研究治理 / Research Governance

本文件合并了原 `ACCEPTANCE_RULES.md`、`BASELINE_TRUST_CLASSIFICATION.md`、
`EVALUATION_PROTOCOL_V2.md`（Round 31 起它们不再单独存在；原文可用
`git show 2c4e296:<文件名>` 查看）。隔离窗的原始证据保存在
[`docs/governance/evidence/`](governance/evidence/)。

**核心立场：没有通过本文件全部闸门的数字，不是业绩，只是研究线索。**

## 1. 时间分区与隔离窗 / Time partitions

| 分区 | 窗口 | 用途 | 规则 |
|---|---|---|---|
| TRAIN | ≤ 2024-06-30 | 模型拟合 | embargo ≥ 最长标签跨度 |
| SEARCH / DEV | 2024-07 → 2025-08-29 | 选择、调参、预注册实验 | 每次搜索登记试验数 N |
| **QUARANTINE（已烧毁）** | **2025-09-01 → 2026-05-18** | 禁区 | ≥35 次直接评测 + ≥5 次选择决策消费过；禁止新评测与选择 |
| **FRESH（冻结）** | **2026-05-19 →** | 唯一的新 final holdout | 只允许预注册配置各评测一次；首读需 ≥120 交易日且 PIT 重建通过 |

- 单一事实来源：`configs/quarantined_windows.json`，由
  `quantagent.backtest.quarantine` 强制；访问留痕到
  `runtime/state/holdout_access_log.jsonl`。
- 解除方式只有**修改配置**（在 PIT 重建、重新 QC、预注册之后），永远不是 override。
- 诊断例外：`--allow-quarantined "<原因>"` 留痕，产出数字不得参与选择。
- **当前状态（2026-11-25）**：FRESH 窗的数据管道自 2026-08-05 起停摆，原因是沪市没有带日期的 ST 证据；
  ENG_ST31 预注册验证通过后，3,273 个种子已全部有证据（见 DATA.md），catchup 从下一次 06:00 运行起恢复补齐。
  窗口仍是冻结的最终 holdout：补齐数据不等于读取或评估它。
  见 [ROADMAP](ROADMAP.md) P0-1。

## 2. 信任分级 / Trust classes

任何数字被引用时都必须带 **{窗口, 试验数 N, 信任分级, artifact 路径}** 四元组。
分级取**最坏适用类**：

| 分级 | 含义 |
|---|---|
| `clean_oos` | 配置先验固定，样本外**首次**评测，结果未反哺选择 |
| `walk_forward_oos` | purged/embargoed 走式样本外的折间聚合 |
| `searched_validation` | 在 validation 窗上多候选搜索后的赢家读数 |
| `contaminated_holdout` | 数字的产生或采纳用过 holdout 信息（含"看多个读数再挑"） |
| `in_sample` / `unknown` | 训练窗内 / lineage 不足 |
| `research_certified_panel_st_incomplete` | 认证 r29 面板 + 规范评估器，但沪/北 ST 状态未知（已披露） |
| `unverified_legacy_panel_lookahead_st` | 旧 v7 银层面板：今日 ST 名单广播到全部历史 = 前视 |

### 历史数字的现状（Round 31 复核）

| 声明 | 原分级 | 现状 |
|---|---|---|
| 2-sleeve FT blend +38.6% CAGR | `contaminated_holdout` | **作废**（另：训练面板被 Round 29 证伪） |
| v8.9 rankfix k50 +17.25%、3-sleeve 先验 +8.3% | `clean_oos`¹² | **作废**：执行时钟修复后不可复现；面板为前视 ST |
| EXP-015/019/024 等 walk-forward 结论 | `walk_forward_oos` | **作废**（同一旧面板）；负结论方向仍可参考 |
| 透明量价 top-50 周频：−11.73%/年 vs 等权 +6.89% | `research_certified_panel_st_incomplete` | **有效**：当前最可信的经济证据（负） |
| R29 E1：LightGBM rank-IC 0.1088 > ridge 0.0962（t=3.51，毛） | 预注册 walk-forward | **有效**（毛 IC，未计成本；2023–24 未复现，见 RESEARCH.md） |
| RL 相对确定性账簿 IQM −2.35% ⇒ DO NOT ENABLE | 预注册 | **有效** |

**结论：仓库内不存在任何通过成本、容量与统计闸门的可交易 edge。**

## 3. 验收规则 / Acceptance rules（全部为 AND）

- **R0 对照基线**：认证 r29 执行面板上、规范评估器（`scripts/baseline_protocol.py`
  variant C）计算的**可交易宇宙等权**。旧 v8.9 数字不得作对照。改进必须表现为
  walk-forward 逐折分布优于同协议基线（中位数提升且亏损折占比不升），不是单窗点值。
- **R1 选择卫生**：选择只用 TRAIN + SEARCH；触碰 QUARANTINE 即 REJECT；搜索须**预先声明**
  候选网格、N、选择指标；同族 N 累计（见 RESEARCH.md 试验台账）。
- **R2 统计门**：PBO（CSCV, S=8）≤ 0.25 且不得单独作通过依据；DSR（按累计 N）≥ 0.95；
  亏损折占比 ≤ 40%；邻域稳定（±1 格中位 ≥ 赢家 60%）；子窗方向一致。
  对重叠标签的 IC t 值，以非重叠子样本（或 Hansen–Hodrick）计算，筛选用 |t| ≥ 3。
  实现（`governance_metrics`、`lifecycle`）对**预先定向**的因子取单侧 t ≥ 3（与均值 IC 门一致）：
  未定向的强负因子会被拒绝而不是翻转；方向须在评估前声明。
- **R3 执行现实门**：一切收益数字出自 variant C（T+1、涨跌停/停牌/ST、整手、参与率 ≤10%、
  成本 + 滑点、分红送转与红利税）；滑点 8→15 bps 年化衰减 ≤ 40% 且仍跑赢基线；
  日换手 ≤ 0.35；平均持有 ≥ 3 日。
- **R4 风险容量门**：折内最大回撤中位 ≤ 20%、最差 ≤ 30%；单行业 ≤ 30%；单票 ≤ 10%；
  top-K ≥ 10；以 20 日 ADV 的 10% 参与率反推容量 ≥ 目标资金；报告 Jensen alpha。
- **R5 资源门**：CPU RAM ≤ 48 GiB、VRAM ≤ 20 GiB、单实验磁盘 ≤ 5 GB、声明运行时长、可捕获 OOM。
- **R6 台账**：每个实验（含失败）登记 git hash、schema_hash、窗口、命令、全部指标、资源、裁决。
- **R7 流程**：候选 → R1 → R3 → R2 → R4 → R5 → 台账 → FRESH 窗预注册一次性终评 → ACCEPT。
  "先合入后补验证"不允许。
- **R8 不可协商**：不看未来；不在 holdout 上调参；不用合成数据充当结果；不绕过 DSL 写自由因子；
  不静默改 feature schema；不隐藏失败实验；不无界搜索；不产生实盘指令。

## 4. Forward paper 协议

1. 冻结模型每日打分 → target_weights 追加写入 append-only 账本（含配置与 git hash）。
2. 不回填、不重算历史；模型或配置变更 = 新 track，旧序列保留。
3. 讨论 live-readiness 的必要条件：FRESH 窗首读通过 + forward paper ≥ 3 个月方向一致。
   当前系统为 **RESEARCH / NOT LIVE READY**，实盘路径终态为 `LIVE_DISABLED`。

## 5. 委员会与人工门 / Council and human gate

Decision Council v2（11 岗、四阶段）由 `services.quant_api.services.council` 单一 registry
固定，是**确定性规则**，不调用 LLM。证据缺失记 `unknown`，永远不记 `pass`；人工推翻需
author + 理由，写入 append-only `council_overrides.jsonl`。细节见
[ADR-004](architecture/adr-004-council-v2-company-protocol.md)。
