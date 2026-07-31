# 机构量化参考架构与 QuantAgent 自研设计 / Institutional Reference & Original Architecture

> **信息边界声明 / Disclosure boundary.** 本文只使用公开可得材料：官方网站、
> 技术博客、公开演讲、学术论文、监管与媒体报道。**没有**任何机构的非公开
> 资料、内部文档或商业秘密。凡是从公开证据出发的推断，一律标注
> `[推断/INFERRED]`，与 `[公开/PUBLIC]` 事实严格分开。任何机构的内部
> 工作流细节都**不可**当作事实引用。本文的目的不是模仿，而是抽取
> first-principles 设计约束，用于 QuantAgent 的**自研**架构。

---

## 1. 公开证据摘要 / Public evidence digest

### 1.1 研究基础设施规模 / Research infrastructure scale

| 机构 | 公开事实 `[公开]` | 设计含义 |
|---|---|---|
| Two Sigma | 官方材料称每日运行 100,000+ 次仿真；存储规模 600+ PB；低延迟特征构建/alpha 生成框架以 Rust 重写 | 研究平台的第一性能指标是**每日可完成的独立实验数**，而不是单次回测速度 |
| High-Flyer 幻方 | "萤火二号" 约 10,000 张 A100，2021 投产；自研 hfai.nn 算子库、hfreduce 通信、3FS 文件系统；公开宣称集群利用率与读带宽指标 | 算力是**共享可抢占资源**；平台必须把 GPU 当作带配额与调度的一等公民 |
| Ubiquant 九坤 | 公开报道自建 GPU 集群与 AI Lab / 数据实验室 / 交易执行实验室三分结构 | 数据、算法、执行**分离建制**，各自有独立的验收口径 |
| Jane Street | 官方技术页说明以 OCaml 覆盖研究工具→交易系统→基础设施→会计系统；自建基于状态机复制的分布式框架 | **同一套类型化契约贯穿研究到生产**，避免研究/生产两套语义 |
| Renaissance | 公开访谈材料提到 Medallion 依赖较短周期信号、持有期以日计；执行系统与信号发现同等重要 | 执行成本模型不是事后 TCA，而是**选股环节的一等约束** |

### 1.2 组织与风险 / Organization and risk

`[公开]` 多经理平台（Citadel / Millennium / Point72 / Balyasny）的公开报道一致
描述 "pod" 结构：数百个独立小组、按策略/资产类别划分、风险按组计量、资本按
表现动态分配。公开报道提到 Millennium 有回撤触发的资本削减与终止阈值
（媒体常引用 5% 削半 / 7.5% 终止；这属于**媒体报道口径**，不是官方披露，
本平台不把它当作行业标准数值使用）。Citadel 公开材料强调风险团队按策略
专业化配置，而不是通用风控通才。

**设计含义**：风险不是全局单一阈值，而是**按研究单元分层**的、可以独立
触发降级/停用的契约。这与 QuantAgent 已有的 readiness tier 思路一致。

### 1.3 研究方法学 / Research methodology

`[公开]` 学术与业界共识（Bailey & López de Prado 等）：

- **Deflated Sharpe Ratio (DSR)** — 对选择偏差、试验次数、非正态性做修正。
- **Probability of Backtest Overfitting (PBO)** — 通过 CSCV/CPCV 组合切分估计
  "样本内冠军在样本外落入下半区" 的概率。
- **Purged / embargoed walk-forward** — 消除 forward label 重叠导致的泄漏。
- **多重检验控制** — 试验次数必须计数并进入统计口径。

`[公开]` 市场冲击的平方根律（square-root law）与 Almgren–Chriss 框架：冲击成本
随成交量的约 1/2 次幂增长，且对机构规模订单，冲击成本通常比显式费用高一个
数量级。

`[公开]` Barra 类多因子风险模型：把组合风险拆成 factor（行业/风格/国家）与
specific（特异）两部分，用于风险归因、约束优化与压力测试。

`[公开]` WorldQuant BRAIN 的公开工作流：表达式构建 → 仿真 → 按 Sharpe/fitness
筛选 → **自相关检查** → 提交进入 alpha 池。核心机制是"**入池前必须先过
相关性闸门**"。

### 1.4 A 股特有约束 / A-share specific constraints

`[公开]` A 股 T+1 制度下不存在直接日内回转；市场上的"做 T"是**用底仓换取
日内可卖库存**的变通做法（半仓滚动、三成底仓滚动等）。这意味着做 T 的
研究口径必须显式建模**可卖库存约束**，否则回测会凭空创造 T+0 能力。

---

## 2. 从公开证据抽取的设计约束 / First-principles constraints

把上面的公开证据压缩成对本平台有约束力的 8 条原则：

1. **实验吞吐 > 单次速度**。研究平台的一等指标是每日可完成、可比较、可审计的
   独立实验数量，因此每个实验必须自带 manifest、hash 与 lineage。
2. **试验次数必须被计数**。任何"最优"结论都要带 `n_trials`，并据此计算 DSR/PBO。
   不计数的搜索等于不可信的搜索。
3. **成本与冲击是选股约束，不是事后报表**。容量、换手、冲击进入目标函数。
4. **相关性闸门在入池前**。新因子/新策略先与现有池做相关性检查再谈收益。
5. **风险按单元分层**。每个研究单元有独立的降级/停用契约，不共享一个全局开关。
6. **研究与生产共享同一份契约**。同一个 manifest 既驱动研究运行，也驱动纸面
   交易；不允许出现"研究里能跑、生产里语义不同"的第二套定义。
7. **算力是带配额的共享资源**。GPU 必须可见、可排队、可抢占、可取消。
8. **A 股制度约束是硬编码的物理定律**。T+1、涨跌停、停牌、ST、最小交易单位、
   可卖库存——在回测引擎里是不可绕过的，不是可选开关。

---

## 3. QuantAgent 自研架构 / Original architecture

> 以下是**本平台自己的设计**。不复制任何机构的产品形态、命名或界面。

### 3.1 代号与设计立场 / Codename and stance

平台内部代号 **ATLAS**（A-share Alpha Laboratory & Supervision）。
设计立场三句话：

- **证据优先于结论**：每个数字必须能点开看到它的来源产物与 hash。
- **闸门优先于自由**：能力越强的动作，门槛越显式（人工 Gate、否决权、审计）。
- **诚实优先于美观**：缺数据就显示缺数据并给出下一步，绝不用占位数字填充。

### 3.2 六层架构 / Six-layer architecture

```
L6  监督层 Supervision   readiness tier · 人工 Gate · 审计回放 · kill switch
L5  决策层 Decision      多 Agent 议事会（角色化否决权 + 结构化证据）
L4  组合层 Portfolio     目标权重 · 约束优化 · 风险归因 · 容量/冲击
L3  验证层 Validation    purged walk-forward · PBO/DSR/SPA · 压力测试
L2  融合层 Fusion        ★ 因子融合搜索：GA / 混合权重 / régime 条件化
L1  信号层 Signal        因子库 · 因子发现 · IC/衰减/相关性闸门
L0  数据层 Data          U0 全宇宙 · PIT 元数据 · provenance · 隔离区
```

**L2 是本平台的核心差异化层**，也是用户第一目标（"通过机器学习/优化发现
有效的因子融合策略"）的落点。它已有的 Python 能力：

- `quantagent.optimization.ga_weight_optimizer` — 多目标 GA + purged walk-forward。
- `quantagent.optimization.multi_objective_loss` — 15 项分量损失（净收益 / Sharpe /
  Calmar / 最大回撤 / 换手 / 尾部风险 / regime 一致性 / 交易成本 / 集中度 /
  流动性 / ST 暴露 / 未成交 等）。
- `quantagent.ensemble.blend_optimizer` — 周期/袖套混合权重搜索（含超额收益目标）。
- `quantagent.quant_math.performance` / `purged_cv` — DSR / PBO / CSCV。

**缺口**：这一层此前没有 Web 入口，用户无法配置、启动、监控、比较融合搜索。
本次升级把它提升为一等模块 **Alpha Foundry / 因子融合工场**。

### 3.3 选优口径 / Selection criteria（用户第 3 条要求）

候选融合策略按四个目标排序，**全部在早期 OOS 上计算，最终 holdout 只验收一次**：

| 目标 | 计算口径 | 方向 |
|---|---|---|
| 最大超额收益 excess return | 组合年化 − 基准年化（默认 000300.SH） | ↑ |
| 最小回撤 max drawdown | 净值序列最大回撤（绝对值） | ↓ |
| 最大年化 annualized return | 几何年化（复利，非算术均值） | ↑ |
| 样本外稳健性 OOS robustness | 折间一致性 × (1 − PBO) × DSR 概率，见 §3.4 | ↑ |

四目标相互冲突，因此**不产出单一"最优"，而产出 Pareto 前沿**；操作者在前沿上
按偏好权重选择，权重只影响排序，不影响候选生成。

### 3.4 稳健性评分 / Robustness score

```
robustness = w_c · fold_consistency          # 折间符号一致 + 离散度惩罚
           + w_p · (1 − PBO)                 # 组合对称交叉验证过拟合概率
           + w_d · DSR_probability           # 试验次数修正后的显著性
           + w_r · regime_consistency        # 最差 regime 的表现下限
```

`n_trials` 由搜索本身自动累加并写入 manifest，**操作者不能手动设置**——
这是防止 DSR 被人为美化的关键。

### 3.5 多 Agent 议事会 / Decision council（用户第 2 条要求）

不是"聊天机器人"，而是**角色化的结构性审查**。每个角色：

- 有明确职责域与**否决范围**（veto scope），只能否决自己域内的问题；
- 只消费**结构化证据**（产物路径 + hash + 指标），不消费自然语言臆测；
- 输出 `status / finding / evidence / nextAction`，可被人工**挑战与推翻**，
  但推翻必须留下 `override` 审计记录（谁、何时、理由）。

角色集合（跨模块统一，不再局限于策略校验）：

| Agent | 职责域 | 否决范围 |
|---|---|---|
| `data_quality` | PIT 完整性、provenance、复权口径、隔离区 | 输入数据不可信 → 否决整条链 |
| `factor_integrity` | IC/衰减/相关性/冗余、因子泄漏 | 因子入池 |
| `model_validation` | 训练协议、折切分、泄漏、收敛证据 | 模型晋级 |
| `fusion_search` | 试验计数、PBO/DSR、前沿合法性 | 融合候选晋级 |
| `portfolio_risk` | 约束满足、集中度、行业暴露、容量 | 目标权重发布 |
| `execution_realism` | T+1、涨跌停、停牌、冲击、可卖库存 | 回测可实现性 |
| `governance` | readiness tier、人工 Gate、审计链完整 | 任何 live 意图 |

**一票否决 = 阻塞晋级，不是阻塞研究**。研究可以继续跑，但产物被打上
`BLOCKED_BY_<agent>` 标记且不能进入下一层。

### 3.6 做 T / T+1 日内研究口径（用户第 5 条要求）

A 股没有 T+0，因此做 T 研究的**唯一合法口径**是：

1. **可卖库存约束**：当日可卖数量 ≤ 昨日收盘持仓；当日买入部分 T+1 才可卖。
2. **底仓成本归属**：做 T 的收益必须相对"持有底仓不动"的基线计算，
   否则会把底仓的 beta 收益错记成做 T 的 alpha。
3. **成本敏感性是主结论**：做 T 的边际收益极小，必须在 maker/taker、滑点、
   冲击的**成本曲面**上报告，而不是单点成本。
4. **默认结论是 NO_TRADE**：引擎是 do-no-harm 覆盖层——只有在成本后 EV 显著
   为正时才产生动作。仓库既有的结论（`docs/` 与 EXPERIMENT_LEDGER）显示
   1 分钟 OHLCV 上无可实现 edge，这个负结论必须在 UI 上**保留可见**，
   不允许被新界面掩盖。

### 3.7 运行模式 / Operating modes

| 模式 | 允许的动作 | 闸门 |
|---|---|---|
| `RESEARCH` | 数据构建、因子发现、训练、融合搜索、回测 | 人工 Gate（每次配置变更后重新校验） |
| `VALIDATION` | 压力测试、walk-forward 复核、稳健性验收 | fusion_search + model_validation 无否决 |
| `PAPER` | 纸面交易、对账、审计回放 | portfolio_risk + execution_realism 无否决 |
| `LIVE_DISABLED` | **拒绝一切实盘意图**（终端态，无覆盖路径） | 不可解除 |

---

## 4. 与本平台既有实现的对齐 / Alignment with existing implementation

| 参考约束 | 本平台既有能力 | 本次补齐 |
|---|---|---|
| 实验吞吐与 lineage | `DataManifest` / runtime indexer / job manifest | 融合搜索 run 的 manifest 与比较视图 |
| 试验次数计数 | GA/blend 内部有折结构 | **`n_trials` 自动累加并进入 DSR** |
| 成本进入目标 | `multi_objective_loss` 15 分量 | Web 端可视化与可配置权重 |
| 相关性闸门 | 因子评估 `max_pairwise_correlation` | 融合候选之间的相关性闸门 |
| 分层风险 | readiness tier / risk gate | Agent 角色化否决 |
| 研究=生产契约 | `quantagent.strategy.v1` manifest | 融合 run 复用同一 manifest 谱系 |
| 算力配额 | `require_gpu` fail-closed | 训练/搜索队列的暂停与恢复 |
| A 股物理定律 | `ashare_rules` / `tplus1_engine` | 做 T 可卖库存在 UI 上显式呈现 |

---

## 5. 参考来源 / Sources

- Two Sigma — 官方 About / Engineering 页面：<https://www.twosigma.com/about-us/>，<https://www.twosigma.com/careers/engineering/>
- Jane Street — Technology / Performance Engineering：<https://www.janestreet.com/technology/>；OCaml 成功案例：<https://ocaml.org/success-stories/large-scale-trading-system>
- High-Flyer 幻方 — 技术博客与萤火文档：<https://www.high-flyer.cn/blog/>，<https://doc.hfai.high-flyer.cn/>
- Ubiquant 九坤 — 官网：<https://www.ubiquant.com/>；Wikipedia：<https://en.wikipedia.org/wiki/Ubiquant>
- Bailey & López de Prado — *The Deflated Sharpe Ratio*：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Bailey, Borwein, López de Prado, Zhu — *The Probability of Backtest Overfitting*：<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- MSCI Barra USE4 方法论说明：<https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf>
- 市场冲击平方根律 / 最优执行：<https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf>
- WorldQuant BRAIN 公开介绍：<https://www.worldquant.com/brain/>
- 多经理平台 pod 结构公开报道：<https://capitalgains.thediff.co/p/multimanagerpodhedge-fund-101>
- A 股日内回转（做 T）公开资料：<https://www.myquant.cn/docs/python_strategyies/108>

> 再次声明：以上均为公开来源。本平台不主张、也不使用任何机构的非公开信息；
> 媒体口径的数值（如某些回撤阈值）在本平台内不作为标准或默认值使用。
