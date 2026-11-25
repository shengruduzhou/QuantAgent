# 研究记录 / Research Record

研究结论的**唯一摘要页**。完整台账（每个实验与假设，含失败）保存在
[`docs/research/ledgers/`](research/ledgers/)：`EXPERIMENT_LEDGER.md`、
`HYPOTHESIS_REGISTRY.md`、因子候选与 PBO/DSR 结果 CSV。引用规则见
[GOVERNANCE.md](GOVERNANCE.md) §2。

## 1. 现在能确定的事 / What the evidence supports（Round 31，数据 ≤ 2025-08-29）

| 结论 | 证据 | 信任 |
|---|---|---|
| 日频量价有**毛**截面信号：认证 r29 面板上组合 rank-IC 0.06–0.11 | R29 E1（158 去重量价因子，389 OOS 日，5 日标签）；Alpha101 隔离窗干净筛选（82 个可算、|IC|>0.02 共 28 个，最佳 alpha040 0.067） | 预注册 / 隔离窗干净，**毛收益口径** |
| **不存在**成本后可交易的 long-only edge | 透明低波+反转 top-50 周频：−11.73%/年，Sharpe −0.59，最大回撤 74.3%，PSR 0.04；等权 +6.89%。R7 价值/质量 +4.33%、低波 +2.52%，均跑输等权 | `research_certified_panel_st_incomplete` |
| 非线性的增益很小且不稳 | E1：LightGBM 0.1088 vs ridge 0.0962（t 3.51，全期）；Round 31 复核：治理路径"线性胜出"是**原始收益 L2 目标**造成的伪影，换成截面 rank 目标后 ridge 0.073、E1 设定 GBM 0.080（+0.007，t 0.88）；该窗口仅约 34% 功效 | 描述性复现（5 次试验，已入账） |
| RL 不增加价值 | 44 次预注册运行，PPO 相对确定性账簿超额 IQM −2.35%（95% CI [−2.78, −1.94]），20/20 输 | 预注册 ⇒ **DO NOT ENABLE** |
| LLM/多 agent 推理没有可测价值 | 无任何实验显示增益；"+7 LLM 因子"在 H-029 前于前向推理中静默为 NaN；Council v2 是确定性规则 | — |
| 做T（1 分钟 OHLCV）、打板 | 做T 无可实现 edge；打板逆向选择约 −2%/板 | 历史结论，方向可信 |
| 基本面 / regime 元模型 | H-020~H-024 按预注册闸门全部拒绝；瓶颈在容量/组合/执行 | 历史结论（旧面板），方向参考 |

**"生产模型 = FT-Transformer sleeves"不成立**：`configs/production_blend.json` 自身记录
`likely_overfit`（PBO 0.886，27 次试验）与 `contaminated_holdout`；其训练集建在被证伪的
v7 银层面板上，且该数据集把 2026-05-31 的 231 个 ST 名字从全部历史中剔除（前视宇宙，
Round 31 D-01）。它只能被视为**历史、未验证**的产物，在认证数据上重训前不得使用。

## 2. 统计工具的校准 / Calibration of the machinery（Round 31 合成实验）

| 工具 | 结果 | 处置 |
|---|---|---|
| DSR / PSR | 零假设下 P(DSR>0.95) ≤ 2%；单位处理正确 | 保留 |
| SPA | 名义 5% 下实际 2%，功效 95% | 保留 |
| PBO | 零假设均值 0.48；阈值 0.50 让一半噪声族通过，0.25 通过 12.5% | 默认收紧到 0.25，且不得单独作通过依据 |
| HAC t（重叠标签 IC） | 持续性因子在名义 5% 下拒绝 9–11% | 用非重叠子样本 t；筛选 |t| ≥ 3 |
| RD-Agent 接受门 | 默认参数接受 **48% 纯噪声因子** | 已收紧默认值（见 HISTORY Round 31）；ROADMAP P1-1 |
| 标签口径 | delay-0（`close(t+h)/close(t)`）使短周期 IC 虚增最多 53% | 认证数据与 `_exec_` 数据集用 delay-1；标签产物带口径戳，消费者拒绝无戳/混用（ROADMAP P1-2） |

## 3. 多重检验台账 / Trial count

同族累计试验数是 DSR 的输入。Cycle 32 逐项按产物复核（界值一律取上界）：H-001~H-027 全库累计 **117**
（`HYPOTHESIS_REGISTRY.md:313`，H-027 按含条件 arm 的上限 15 计）+ R29 E1 **18**（6 arm a/s/b/c/d/e × run1 与确定性
run2，外加涨停封板敏感性重评分 6，见 `git show 2c4e296:docs/audits/round29/roles/R3_factor_findings.md`）+ R29 治理化
`audit-nonlinear-factors` run1 **6**（`runtime/reports/nonlinearity/round29_r29_panel/run1_20261003`）+ Round 31 描述性复现
X1 **5**（`runtime/archives/audits/round31/audit/quant/notes.md`）⇒ **非线性/融合族 = 146**（旧估算 134 把 E1 只算了一次）。
**台账（Cycle 32 起）**：唯一来源是 `quantagent.research.trial_ledger`
（`$QUANTAGENT_HOME/state/trial_ledger.jsonl`）。`family_count(family)` = 声明的历史基线（附出处）+ 已入账试验；
基线：`nonlinear_fusion` = 146（即上述逐项计数），`rdagent_factor_synthesis` = 153（runtime/ 下 7 份
`rd_agent_task_feedback.json` 的候选数之和，下界）。`audit-nonlinear-factors`、`search-factor-fusion`、
`synthesize-factors-v7`（RD-Agent）每次运行把每个试验（含对照/失败/拒绝）写入台账，DSR 的 N =
累计数（只能比单次运行的 N 大）；产物 `promotion_gate.json` / `trial_ledger.json` 记录所用 N 与台账头哈希。
RD-Agent 接受门本身是非重叠 IC t 值（不算 DSR），累计 N 供下游 DSR/选择使用。引用数字时 N 取台账值。

## 4. 预注册 / Pre-registrations

- [ENG_RISK30](research/preregistrations/ENG_RISK30_preregistration.md)：回撤阶梯 +
  波动率目标（`--exposure-policy`，默认关闭），真实数据检验尚未运行。
- R3_E2（治理化非线性设定消融）：分支 `agent/round30-fusion-e2`，**运行前必须修订**——
  其 CONFIRMED 规则以原始目标 ridge 为对照，会因目标而非模型类别"确认"非线性；
  需加入 rank 目标线性对照、把选择折延长到 2019、接入持久试验台账。
- ENG_ST30（公告重建沪/北 ST 登记）：2026-11-24 执行；DEV 全过、HOLDOUT M5b 0.924 未过 ⇒ 沪/北十年历史保持 BLOCKED_BY_DATA（见 DATA.md 与预注册文件）。
- [ENG_ST31](research/preregistrations/ENG_ST31_preregistration.md)（前向锚定沪/北 ST 登记）：2026-11-25 一次验证通过
  （深市 V1–V5 全过，19/19 变更同日）⇒ 沪/北 2026-07-27 起发布；每日深市监控失守即撤下。
