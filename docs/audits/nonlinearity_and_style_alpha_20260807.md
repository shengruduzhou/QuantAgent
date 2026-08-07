# 多因子非线性与风险归因审计 — 2026-08-07

审计对象：QuantAgent 的多因子融合、非线性交互、horizon/sleeve 混合、factor
weighting/selection、ML 模型、ensemble、regime conditioning、CAPM/FF/Carhart 归因、
训练/验证/holdout 隔离、PBO/DSR/SPA 多重检验控制。

参考文献：Sharpe (1964)、Black & Scholes (1973)、Fama & French (1992)、
Carhart (1997)、Kakushadze *101 Formulaic Alphas* (2016)、Gu, Kelly & Xiu
*Empirical Asset Pricing via Machine Learning* (RFS 2020)、Microsoft Qlib
(2020)，以及 Harvey, Liu & Zhu (2016) 关于因子多重检验的后续研究。

---

## 0. 术语：五个被混为一谈的概念

审计中反复出现的核心混淆，先定义清楚。代码里现在由
`quantagent.models.interactions.ModelClass` 显式声明，不再靠函数名推断。

| 类别 | 形式 | 是否含交互 |
|---|---|---|
| `linear_additive` | `Σ_j w_j x_j` | 否 |
| `rank_weighted_additive` | `Σ_j w_j rank(x_j)` | 否 |
| `factor_nonlinear_transform` | `Σ_j f_j(x_j)`，`f_j` 非线性 | 否 |
| `factor_interaction` | 含 `x_i x_j`（`i≠j`） | 是 |
| `regime_interaction` | 含 `x_j × m_t` | 是 |
| `nonlinear_learner` | 树 / 神经网络 | 是（隐式） |
| `nonlinear_objective` | Huber / IC loss / 回撤惩罚 | 与函数形式无关 |
| `ensemble` | 多个已拟合模型的组合 | 组合加性模型仍是加性模型 |

**这不是术语洁癖，而是 GKX 论文里最锐利的实证结果。** 他们的 GLM 把每个预测变量
做二阶样条展开——真正的 `factor_nonlinear_transform`——月度 OOS R² 只有 **0.19%**，
对比 elastic net 的 0.11%，几乎没有增量。而树和神经网络（与 GLM 的差别恰恰在于
允许**变量之间**交互）拿到 **0.33%–0.40%**。原文直述：GLM "uses spline functions of
individual features, but includes no interaction among features"，这正是它无法超越
线性方法的原因。

**单因子的曲率很便宜；因子之间的条件化才是收益来源。**

同样值得记住 GKX 的 Diebold-Mariano 配对检验表：树/神经网络 vs PCR/PLS 的差距大多
**不显著**。诚实的先验是"交互带来的增益真实但很小、且难以确立"，而不是"更复杂的模型
默认更好"。

---

## 1. 发现的主要问题

按严重度排序。带 **[MEASURED]** 的均在本仓库真实数据上量化过。

### P0-1 线性基线被前处理缺陷废掉 **[MEASURED]**

`training/v7_experiment.py::_fit_linear` 在**未标准化**的设计矩阵上直接解正规方程。
面板里 `amount_mean_20d`（元，量级 1e8）与 `momentum_20d`（秩尺度，量级 1）相差约 10
个数量级，`np.linalg.pinv` 的奇异值截断把小尺度方向整个丢弃。

合成面板（唯一真实信号在小尺度列）实测：

| | rank IC | 信号列系数 |
|---|---|---|
| 修复前 ridge | **0.0117** | −7e−24（被抹平） |
| 修复后 ridge | **0.2705** | 恢复 |

`elastic_net` 更糟：固定 `lr = 0.05` 的近端梯度在同一设计矩阵上远超收敛所需的 `1/L`
上界，**几次迭代内溢出为 NaN**。NaN 预测流入 `_rank_ic` → `dropna()` → 空序列 →
指标层填 0.0。**一次彻底损坏的拟合，被记录为"假设被检验且无效"。**

> 这是整份审计里最重要的一条：线性基线是仓库里**每一个"树模型更好"结论的对照组**。
> 对照组被前处理 bug 削弱，就等于自动为"非线性有效"制造证据。

### P0-2 `interaction_search` 不搜索任何交互

`ensemble/strict_factor_search.py` 的配置项名为 `interaction_search`，trial stage 标记为
`interaction_beam`。但它做的是**因子子集**的 beam search，每个子集用
`build_factor_composite` 打分——即符号翻转后中心化截面秩的**等权平均**。

全代码库 grep 确认：**不存在任何位置构造 `x_i × x_j`**。

往子集里加一个因子改变的是求和项的成员，不是它们之间的乘积。按 §0 的分类，这是
`rank_weighted_additive`，与 GKX 中"买不到东西"的那一类同构。

### P0-3 regime 从不进入模型系数

regime 在三处出现，**全部在打分之后**：

1. `_lookup_regime_multiplier` — 仓位敞口乘数；
2. `regime_sleeve_blend` / `EnsembleWeights` — sleeve 混合权重；
3. `filter_factor_frame_by_regime` — 按日期**子集化**样本。

三者都无法让模型学到"某因子在熊市里换号"。regime 权重是一个**正数**缩放：可以把因子
调小，永远不能把它调转。

对比 GKX 的基线特征集（式 21）：`z_{i,t} = x_t ⊗ c_{i,t}`——特征本身就是宏观状态向量与
个股特征的 Kronecker 积，共 94×(8+1)+74 = 920 维，**连线性模型都能表达 regime 条件化**。
仓库此前没有任何等价构造。

### P0-4 风险归因只有 CAPM **[MEASURED]**

`backtest/beta_decomposition.py` 只对单一基准回归，把截距称作 Jensen alpha，并据此把策略
标为 `production_candidate`。没有 SMB / HML / UMD。

这正是 FF (1992) 要反驳的东西：1963–1990 年 β 与平均收益的关系是**平的**，而 size 与
book-to-market 联合解释了截面。Carhart (1997) 补上动量：看似的基金经理技能，是一年期动量。

**在本仓库真实面板上构建 A 股风格因子后（2021-01 至 2026-05，3387 只/日）：**

| 因子 | 年化 | 波动 | Sharpe |
|---|---|---|---|
| MKT | +9.54% | 22.02% | +0.52 |
| SMB | **+19.60%** | 17.16% | +1.13 |
| HML | **+14.33%** | 12.89% | +1.10 |
| UMD | +0.74% | 10.59% | +0.12 |

记忆中记录的旗舰结论是 "v8.9 size30 CAGR +56.8%、beta 0.91、**Jensen alpha +12.9%**
vs 全 A ⇒ 真 alpha 非纯 beta"。在 SMB 年化 +19.6%、HML 年化 +14.3% 的窗口里，一个
集中持股 30 只、天然偏小盘的组合拿到 +12.9% 的 **CAPM** alpha，**与"纯风格暴露"完全
一致**。该结论在 Carhart 归因跑出来之前不成立。

（UMD 近似为零，与 A 股动量效应弱、反转效应强的既有认识一致。）

### P0-5 horizon 混合的声明权重不是实际权重 **[MEASURED]**

`portfolio/multi_horizon_blender.py` 计算 `Σ_h w_h · pred_h / Σ_h w_h`，**直接加总原始
预测**。但每个 horizon 预测的是**不同的标签**：gold 面板上 `forward_return_1d` 的截面标准差
约 0.022，`forward_return_120d` 约 0.237，差一个数量级。

下游只对结果取 `nlargest`，所以离散度就是全部贡献。实测声明权重 vs 实际份额：

| horizon | 声明 | 实际 | 比值 |
|---|---|---|---|
| 1d | 10% | **1.8%** | 0.18× |
| 5d | 20% | 8.4% | 0.42× |
| 20d | 30% | 25.4% | 0.85× |
| 60d | 25% | 35.2% | 1.41× |
| 120d | 15% | **29.3%** | 1.95× |

后果：`balanced` 预设实为长周期主导；`short_tactical` 远不如它读起来那么"tactical"；
而 `DECAY` lifecycle override——其**全部目的**就是把权重压到 1d/5d 让组合快速退出衰退
标的——实际只交付了它声明的短周期影响力的约四分之一。

### P1-6 selection / holdout 接缝没有 purge

`cli/v7_train.py::_split_portfolio_selection_holdout` 让 `selection_end` 与
`holdout_start` **相邻**。但预测带 H 日前瞻标签（最长 120 日），最后一个 selection 日的
标签在 holdout 内部才兑现；模拟组合也跨接缝持仓。管线里**其他每一个** train/test 边界
都 purge 了标签期限，唯独最终 holdout 的接缝没有。

### P1-7 `strict_factor_search` 的选择偏差无任何核算

该搜索在**同一窗口**上跑数百次 `_eval`，无任何折分割，返回**样本内最大值**作为
`best_factors` / `best_score`。`n_trials` 被记录但从不进入 DSR/PBO。

对比：`fusion/` 与 `research/selection_governance.py` 在这方面做得**很好**（见 §4），
两套标准并存于同一仓库。

### P1-8 `single_factor_dominance` 闸门度量了错误的量

该闸门（阈值 0.60）比较**系数绝对值**的占比，但读的是原始尺度系数。以元为单位的因子
系数天然渺小，以秩为单位的天然巨大——闸门在比较苹果和橘子。

### P2-9 标签未做截面去均值

标签是原始 `forward_return_Nd`。对一个纯截面排序模型，这让模型花费容量去拟合市场时序
成分。Qlib 的默认 handler 用 `CSRankNorm`/`CSZScoreNorm`；GKX 用超额收益。
**未修改**（见 §7）。

### P2-10 面板 `close` 列复权不一致 **[MEASURED]**

`close.pct_change()` 在 2025 年样本上最高达 **+85%**，而数据集自带的 `return_1d` 正确
受 ±10% 涨跌停约束（相关性 0.92，不是 1.0）。任何从 `close` 计算多日动量/市值的代码都会
把送转股读成 −50% 动量。已在新模块中规避并写入注释。

---

## 2. 根因

三条贯穿全部发现的模式：

**(a) 尺度不变性被反复假定，但从未成立。** P0-1（ridge/enet 惩罚项）、P0-5（horizon
混合）、P1-8（dominance 闸门）是同一个错误的三个实例：在一个把不同量纲的量相加或
比较的地方，忘记先把它们放到同一尺度上。截面秩归一化是本仓库已有的正确工具
（`fusion/evaluation.py` 里就有），只是没有用在这三处。

**(b) 命名先于实现，然后没人回去改。** `interaction_search` 大概率始于一个真要做交互的
意图，最后落地成子集搜索，名字留了下来。名字随后成为"我们做了非线性"的证据。

**(c) 严格性在仓库内分布极不均匀。** `research/selection_governance.py` 的 PBO/DSR/SPA
实现是我在本次审计中读到的最扎实的部分之一（DSR 单位、`n_trials` 与样本离散度分离、
拒绝用中位数填充退化候选——每一处都注释了错误做法为什么反向）。而 `strict_factor_search`
在同一仓库里做纯样本内 argmax。**加固消费者而不加固生产者，等于没有加固**——这与记忆中
DEF-025 的教训完全同形。

---

## 3. 实际修改内容

### 新增模块

**`src/quantagent/models/interactions.py`**
- `ModelClass` 分类枚举（§0），带 `represents_interaction` / `is_additive` 属性；
- `cross_sectional_rank_normalise` — 按日截面秩 → `[-1,1]`（GKX 约定）。用
  `2(r−1)/(n−1)−1` 而非 `2(rank_pct−0.5)`：后者中心在 `1/n` 而非 0，该偏移在加性混合中
  无害，在交互中**有害**——带偏移 `c` 时 `(x_i+c)(x_j+c)` 含 `c(x_i+x_j)` 的纯主效应，
  40 只标的的日期上污染达每个母因子的 2.5%；
- `select_interaction_pairs` — **仅在训练段**选 `x_i x_j`。排序依据不是乘积的原始 IC
  （两个有效因子的乘积几乎必然有效，那只是主效应换个写法），而是**按日回归掉两个母因子
  之后的残差 IC**，再加符号稳定性下限。闭式 2×2 正规方程分组求解（原逐日 `lstsq` 循环
  在真实面板上要跑几小时——**诚实的选择规则如果太贵跑不动，就会被更便宜的不诚实规则替代**）；
- `pairwise_interaction_features` / `regime_interaction_features` — 物化 `x_i x_j` 与
  `x_j ⊗ 1[regime=s]`（GKX 式 21 的离散形式），后者默认丢弃一个参照态以避免完全共线；
- `describe_feature_block` — 从**列名**推断模型类别，而不是从标签。

**`src/quantagent/research/model_comparison.py`**
统一协议下的六臂对照：`linear_baseline`（对照，保留）、`linear_pair_interaction`、
`linear_regime_interaction`、`linear_all_interaction`、`gbm`、`ensemble_stack`（OOF 堆叠）。
- 同一 purged walk-forward 折、同一特征、同一成本、同一评估；
- 四个维度全报：**prediction**（rank IC / ICIR / R²_oos 对零基准，GKX 约定）、
  **economic**（top-K 税后净收益 / Sharpe / Calmar）、**robustness**（折间一致性 / PBO /
  DSR，`n_trials` = 实跑臂数）、**trading**（换手 / 成本拖累 / 选中标的不可交易比例）；
- **增量检验**：与基线**配对**的逐日 IC 差 + Newey-West t 统计量（配对消掉市场状态这个
  巨大的共同成分——GKX 把 Diebold-Mariano 改造成截面平均误差检验正是为此）；
- 五态判决：`production_accepted` / `hypothesis_rejected` / `pipeline_failed` /
  `data_invalid` / `model_invalid`；
- **holdout 纪律**：尾部 `holdout_folds` 折被评分并报告，但不参与任何选择。

**`src/quantagent/backtest/factor_model_attribution.py`**
- `build_ashare_style_factors` — 从面板构建 MKT/SMB/HML/UMD；
- `attribute_strategy_returns` — CAPM → FF3 → Carhart4 嵌套阶梯，截距用 **Newey-West**
  HAC t 统计量（日频策略收益自相关，OLS 标准误会高估显著性）；
- **缺失因子报 `unavailable`，绝不报 0，绝不用流动性代理冒充 size。** 本仓库 security
  master 只有**当期快照**的 `float_shares`，乘以 2021 年的收盘价不是 2021 年的市值——
  这种情况下 SMB 标为 `approximate`，且该标记一路传播到最终判决。
  （记忆中反复出现的缺陷模式正是"缺失测量被合理默认值静默替换"，此处刻意不犯。）
- `t_threshold` 默认 2.0 但**是参数**：Harvey-Liu-Zhu (2016) 认为在整个 factor zoo 的多重
  检验之下，新因子应清过 ~3.0。推广**新信号**（而非度量既有组合）的调用方应显式传 3.0。

**`scripts/audit_nonlinearity_and_style_alpha.py`** — 把两问串成一条可复现命令，退出码
映射到 `research/verdict.py` 的语义（0 接受 / 3 研究否决 / 4 配置阻塞 / 1 工程失败）。

### 修复

| 文件 | 修改 |
|---|---|
| `training/v7_experiment.py` | `_fit_linear` 改为在标准化空间求解，再把 scaler 折回**原始空间系数**（保持 `v7_predictor` 的产物契约不变）；elastic net 改用 Gram 矩阵导出的 Lipschitz 步长 + 收敛判据 + glmnet `lambda_max` 相对惩罚 + 目标标准化；非有限系数与**全零退化拟合**改为显式抛错（"model invalid" ≠ "hypothesis rejected"）；新增 `standardised_coef` 供 dominance 闸门使用 |
| `training/v7_experiment.py` | `_aggregate_metrics` 接受 `dominance_coefficients`，`single_factor_dominance` 改读标准化系数；未收敛拟合记入 `non_converged_linear_fits` |
| `portfolio/multi_horizon_blender.py` | 加权前按 `(trade_date, horizon)` 秩归一化（`scale_normalisation` 默认 `cross_sectional_rank`，`"none"` 保留以复现旧结果）；诊断新增 `model_class` / `weights_are_realised` / `min_cross_section_size` |
| `ensemble/strict_factor_search.py` | `interaction_search`→`subset_beam_search`、`max_interaction_size`→`max_subset_size`（旧名保留为只读属性 + 产物旧键保留）；stage `interaction_beam`→`subset_beam`；产物新增 `model_class` 与 `selection.oos_validated=False` + 选择偏差警告 |
| `backtest/beta_decomposition.py` | `classify_strategy` 接受 `style_attribution`；有 FF3/Carhart 时用其 alpha 替代 CAPM alpha；**仅有 CAPM 的 `production_candidate` 降级为 `research_signal`**，alpha 被风格吸收的降级为 `style_exposure` |
| `cli/v7_train.py` | selection/holdout 接缝按预测中实际出现的最大 horizon purge；证据新增 `purgeDays`/`purgeStart`/`purgeEnd` |
| `cli/v8_gated.py` | CLI 参数改名并保留旧 flag 别名 |
| `fusion/search.py` | summary 显式声明 `modelClass: rank_weighted_additive` / `representsInteraction: false` |

---

## 4. 为什么这样修改

- **保留而非替换基线。** `linear_baseline` 是对照臂，不是待淘汰的旧模型。GKX 的全部结论
  是一个**比较**，没有线性那一列，0.40% 这个数字毫无意义。
- **修复对照组优先于增加候选。** 在 P0-1 修好之前，任何"树 vs 线性"的比较都是无效的。
- **交互显式化而非只加一个树。** 显式 `x_i x_j` 列让"交互是否有用"与"树是否有用"成为两个
  可分离的问题；只加 LightGBM 无法区分增益来自交互、非线性变换，还是别的什么。
- **不因为要过闸门而降低标准。** 反例见 §5：我自己的第一版验收规则被负对照抓到了漏洞。
- **不静默替换缺失测量。** SMB 宁可 `unavailable` 也不用换手率冒充。

---

## 5. 负对照抓到我自己的缺陷（留档）

三个合成 DGP 对照：植入交互 / 纯加性 / 纯噪声。第一版验收规则只要求
`t ≥ 2` 且 `Δ净收益 > 0`。**纯加性 DGP 上 `ensemble_stack` 通过了**：ΔIC = +0.0012，
t = +2.52。

统计显著但经济上毫无意义——400 个配对交易日足以让 +0.001 的 IC 差通过任何 t 检验。
这正是"用统计显著性冒充增益"的教科书案例，而且是我自己写的规则。

**修复：显著性与规模是两个独立问题，都要回答。** 新增 `min_ic_delta`（默认 0.005，约为
截面 IC 典型量级 0.02–0.05 的十分之一）与 `min_net_return_delta`（默认 1pp/年）。
复杂度必须付得起租金。

修复后三对照全部正确：

| DGP | 判决 | champion |
|---|---|---|
| 植入交互 | `production_accepted` | `linear_pair_interaction` |
| 纯加性 | `hypothesis_rejected` | `linear_baseline`（GBM 显著更差，t = −3.46） |
| 纯噪声 | `hypothesis_rejected` | `linear_baseline` |

另有两处我自己写的缺陷被测试抓到并修复，一并留档：

1. **HML 排序未滞后。** `book_yield = 1/pb` 含 `close(t)`，而 `return_1d(t)` 是**进入**
   `close(t)` 的收益。同行排序+结算 ⇒ 当日下跌者被机械地放进高 B/M 腿。实测产出
   HML **−30.8%/年（Sharpe −2.8）**，且 2021–2026 **六个日历年全负**——这种稳定性本身
   就该被读成机械假象而非风险溢价。滞后一日后翻转为 **+14.3%/年（Sharpe +1.1）**。
   已加回归测试：在纯噪声收益上 HML 的 t 必须 < 3。
2. **UMD 用原始 `close` 比值。** 见 P2-10，改为对 `return_1d` 复利。

---

## 6. 与文献的对应关系

| 文献 | 审计中的作用 |
|---|---|
| **Sharpe (1964)** | `beta_decomposition` 此前实现的正是单因子 CAPM 检验——它是**起点**，不是终点 |
| **Fama & French (1992)** | β 与收益关系为平；size + B/M 捕获截面 ⇒ P0-4 的直接依据，也是把 CAPM-only 提名降级的依据 |
| **Carhart (1997)** | 看似的持续技能实为一年期动量 ⇒ 归因阶梯必须到 4 因子；本仓库因子库以动量/反转为主，这一层不可省 |
| **Kakushadze (2016)** | 101 alphas 平均持有期 **0.6–6.4 天**、平均两两相关 15.9%、原文做**行业中性化**。仓库在 5–20 日 horizon、12bps 成本、T+1 与涨跌停约束下使用它们——这是一个**与原始设计不同的用法**，应作为独立假设检验而非默认继承其有效性 |
| **Gu, Kelly & Xiu (2020)** | 全文最核心的对照依据：GLM(0.19%) vs 树/NN(0.33–0.40%) 确立"非线性变换 ≠ 交互"；`z = x_t ⊗ c_{i,t}` 是 regime 交互的实现范本；截面秩 →`[-1,1]` 是归一化约定；R² 对零基准而非历史均值；三段不相交时序切分 + 测试段不参与调参；DM 检验的截面化改造 |
| **Qlib (2020)** | 分层架构（Data Server / Model / Portfolio / Executor）、rolling 重训、`CSRankNorm`/`CSZScoreNorm` 标签归一化——后者是 P2-9 未修项的参考做法 |
| **Harvey, Liu & Zhu (2016)** | 新因子应清 t ≈ 3.0 的多重检验门槛 ⇒ `attribute_strategy_returns` 的 `t_threshold` 设为参数而非常量 |
| **Black & Scholes (1973)** | 与本次审计范围（截面股票 alpha）无直接接口。仓库内相关面在期权/波动率定价，本次未涉及；**不为凑齐引用而强行对应** |

---

## 7. 尚未解决 / 需要更多真实数据

1. **P2-9 标签截面归一化未改。** 全局改变标签约定会使此前所有实验不可比。正确做法是把它
   作为 `model_comparison` 的一个额外臂来检验，而不是直接改产线。
2. **SMB 只到 `approximate`。** 需要**逐期**股本数据（当前 master 只有快照）。在此之前
   FF3/Carhart 的 size 载荷是指示性的，判决中已带标记传播。
3. **风格因子未与外部基准对账。** 本文的 MKT/SMB/HML/UMD 是自建单排序价差，非 FF 的 2×3
   双排序（缺少可 PIT 的 size 断点）。数值合理且逐年可解释，但未与 Barra/CNE5 或公开
   A 股因子库交叉验证——离线环境无法完成。
4. **`strict_factor_search` 仍是样本内 argmax。** 本次只加了选择偏差声明；把它接入
   walk-forward + `selection_governance` 是独立的一块工作。
5. **101 alphas 在 A 股中长 horizon 上的适用性未单独检验。** 见 §6，这是一个明确的、
   尚未回答的问题。
6. **真实面板上的六臂对照结论见 §8。**

---

## 8. 真实数据结论

（本节由 `scripts/audit_nonlinearity_and_style_alpha.py` 在
`training_dataset_alpha181_exec_v89_plus7clean_fund.parquet` 上生成，
2021-01 至 2026-05，4,386,209 行 / 1,295 交易日 / 3,427 标的 / 28 因子。）

> **状态：运行中。** 结果落盘于 `runtime/reports/nonlinearity_audit/`。
> 在该运行完成之前，本审计**不对"非线性在本仓库真实数据上是否带来增量 alpha"作任何
> 断言**——合成对照只证明了检验装置本身有效（能接受真交互、能拒绝加性与噪声），
> 不构成对真实数据的结论。
