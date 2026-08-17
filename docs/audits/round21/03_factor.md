# Round 21 — R3 因子专家 + 策略专家审计报告

角色 / Role: R3 (因子有效性、生命周期、非线性融合)
日期 / Date: 2026-08-18
基线 / Baseline: `b56ae57` (main, clean)
性质 / Nature: **只读审计**。未修改 `src/`、`apps/`，未 commit/push。

裁决词典：`PASS`（实测通过）/ `FAIL`（实测失败）/ `unknown`（无证据，**不等于通过**）/
`BLOCKED_BY_DATA`（缺数据能力，fail-closed）。

---

## 0. 网页调研摘要 / External research (9 sources)

| # | 来源 | 与本仓相关的可执行结论 |
|---|---|---|
| 1 | akquant `textbook/14_factor` | 因子按**研究路径**组织（趋势/反转/量价/复合），核心是**表达式引擎语义**（TS / CS / EL 三类算子）与逐层调试；强调"停牌或缺失日期先做交易日对齐"再做窗口算子。本仓 `factors/expr.py` 与 `factors/operators.py` 是同构实现。 |
| 2 | akquant `guide/factor` | `FactorEngine.run()` / `run_batch()` 双接口；验证点 = 窗口定义、列名映射、时区、warmup 期 NaN 传播。 |
| 3 | akquant `guide/cross_section_checklist` | 六阶段清单（设计/数据/执行/风控/验证/上线前）。关键项："明确信号时点与成交时点关系"、"评分前校验窗口长度，跳过历史不足样本"、universe 来源版本化、冻结数据快照、`reject_reason` 监控、滚动窗口时序验证、换手/集中度/滑点敏感性。**该清单不覆盖 IC 计算、多重检验校正、幸存者偏差** —— 这三项在本仓是自建能力，无外部对标。 |
| 4 | akquant `textbook/12_ml` | 特征必须严格滞后（t 特征只能用 t-1 及更早）；防泄漏三件套 = **Purged K-Fold + embargo**、**walk-forward（expanding/rolling）**、结构性保障；非线性组合推荐树集成（XGBoost/LightGBM/RF）。 |
| 5 | microsoft/qlib | Alpha158 / Alpha360 因子集；DataHandler processor 管线；model zoo（LightGBM/XGBoost/CatBoost/MLP/LSTM/GRU/ALSTM/GAT/TFT/TabNet/TRA/HIST…）。 |
| 6 | RD-Agent(Q) 论文 + repo | 循环 = **提出假设 → 设计实验 → 代码实现 → 真实回测反馈 → 学习改进**；Research/Development 两阶段；宣称"2× 年化、少 70% 因子"。**反馈必须来自真实回测**，这是与本仓 `factor_synthesis.py` 对照的基准。 |
| 7 | 搜索 `nonlinear factor combination gradient boosting` | GBDT 天然捕捉非线性与交互；交互统计量（H-statistic 类）在无交互时为 0，交互越强值越大 —— 可作为交互检测的对照口径。 |
| 8 | 搜索 `factor decay half-life / lifecycle` | **"edge 是易腐品"**；因子半衰期是可度量指标，用于推导每个因子的最优调仓周期（动量 ~3 月、投资因子 ~1 月）。运营口径 = **监控每个因子的 rolling IC、rolling 显著性、估值价差**，区分"拥挤但真实"与"数据挖掘幻象"，并让替换速度快于衰减速度。 |
| 9 | 搜索 `Alpha101 A股 IC ICIR` | A 股实践口径：日 IC / 近 22 日 IC 均值 / 累计 IC；分层（5 或 10 层）看单调性；换手率与收益匹配；常用筛选阈值 `IC>0.05 且 ICIR>0.8`。 |

**调研 → 审计口径**：本报告用第 3/4/8/9 条作为评判标尺 —— 时点关系、purge+embargo、
rolling IC 衰减监控与退役、IC/ICIR/分层/换手四件套。

---

## 1. 因子清单与分类 / Factor inventory (Q1)

### 1.1 实测口径（不是 grep，是 import 后计数）

复现命令：

```bash
AI_quant_venv/bin/python3 -c "
import collections
from quantagent.factors.registry import default_registry
import quantagent.factors.technical_indicators, quantagent.factors.cicc_high_freq, quantagent.factors.cicc_ashare80
m = default_registry.metas()
print(len(m), collections.Counter(v.category for v in m.values()))"
```

实测输出：`default_registry` **195 个**因子，分类计数：

| category | 数量 | 入口文件 |
|---|---|---|
| `alpha101` | 101 | `src/quantagent/factors/alpha101.py` |
| `cicc_ashare80` | 80 | `src/quantagent/factors/cicc_ashare80.py` |
| `cicc_high_freq` | 9 | `src/quantagent/factors/cicc_high_freq.py` |
| `technical_indicators` | **5** | `src/quantagent/factors/technical_indicators.py` |

未进 `default_registry`、但可用的额外因子库：

| 库 | 数量 | 入口文件 | 备注 |
|---|---|---|---|
| GTJA-191 | **64**（`gtja191_names()` 实测） | `src/quantagent/factors/gtja191.py` | 国泰君安短周期价量 191 的 tranche 1，含 `SMA(X,n,m)` 递归平滑精确实现 |
| Alpha181 复合集 | 181 = 101 + 80 | `src/quantagent/factors/alpha181.py` | V7 训练管线的固定特征集 |
| 分时量价 | 16 | `src/quantagent/factors/intraday_volume_price.py` | 1 分钟 bar → 日频截面 |
| CICC 高频（含仅日内） | 9 + 5 | `src/quantagent/factors/cicc_high_freq.py` | `INTRADAY_ONLY_FACTORS` 需分钟数据 |
| 长周期 | — | `src/quantagent/factors/long_horizon_factors.py` | 120/252 日趋势、政策链、宏观 |
| 行业轮动 | — | `src/quantagent/factors/sector_rotation.py` | SW1 行业面板 |
| V7 特征组 | 8 组 / 约 70 列 | `src/quantagent/data/v7_feature_groups.py` | short/medium/long/fundamental/valuation/risk/liquidity/regime |

### 1.2 按用户要求的五类归口

| 类别 | 数量（可计数部分） | 入口文件 | 裁决 |
|---|---|---|---|
| **量价 / Price-volume** | 101 (alpha101) + 80 (cicc80) + 64 (gtja191) + 16 (intraday) ≈ **261** | `alpha101.py` / `cicc_ashare80.py` / `gtja191.py` / `intraday_volume_price.py` | 主体 |
| **传统 TA** | **5**（Bollinger %b、Bollinger bandwidth、RSI-14、MACD hist、MACD hist norm） | `technical_indicators.py:44-209` | **占 `default_registry` 的 2.6%** |
| **基本面 / Fundamental** | 10 (`FUNDAMENTAL_FEATURES`) + 9 (`VALUATION_FEATURES`) + 9 (`RISK_FEATURES`) = 28 列 | `data/v7_feature_groups.py:71-104`；`fundamental/{dupont,peg,statements,market_valuation}.py` | 存在，但走 PIT join 而非 registry |
| **事件 / 预期** | 未计数（`earnings_revision_score` 1 列 + 公告/主题链路） | `concept/announcements.py`、`themes/policy_parser.py`、`credibility/news_credibility_agent.py` | 覆盖薄，见 §1.3 |
| **另类 / Alternative** | `news_sentiment_score`、`fund_flow_5d`、概念/主题强度 | `ashare/fund_flow.py`、`themes/theme_extractor.py`、`agents/flow_agent.py` | 存在但多为 LIVE 筛选，PIT 受限 |
| **微观结构 / Microstructure** | 9 (cicc_high_freq) + 16 (intraday) + Amihud/VWAP 偏离/主力净买压 | `cicc_high_freq.py`、`intraday_volume_price.py`、`execution/intraday_features.py` | **存在**，但无 L2（见 §1.3） |

### 1.3 落盘数据集的实测列构成（决定性证据）

因子**库**里有什么，不等于训练**面板**里有什么。实测两个落盘数据集：

```bash
AI_quant_venv/bin/python3 -c "
import pyarrow.parquet as pq
for p in ['runtime/data/gold/full_universe/dataset.parquet',
          'runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet']:
    f=pq.ParquetFile(p); c=f.schema_arrow.names
    print(p, f.metadata.num_rows, len(c),
          'alpha:',len([x for x in c if x.startswith('alpha')]),
          'gtja:',len([x for x in c if x.startswith('gtja')]))"
```

| 数据集 | 行数 | 列数 | alpha101 | gtja191 | 裁决 |
|---|---|---|---|---|---|
| `runtime/data/gold/full_universe/dataset.parquet`（**唯一持有 `FULL_UNIVERSE_GOLD_READY` 证书**，5790 符号、2016-01→2026-07、hash `10b63ba024dd7428`） | 10,917,401 | 41 | **0** | **0** | **仅 15 个特征** |
| `runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet` | 6,781,038 | 348 | 156 | 58 | 因子丰富 |

**认证全宇宙 gold 的 15 个特征全文**（来自其 `manifest.json` 的 `feature_columns`）：
`ret_1d, ret_5d, ret_20d, ret_60d, px_to_ma_5, px_to_ma_20, px_to_ma_60, vol_20d,
vol_60d, turnover_20d, volume_ratio_5_20, amihud_20d, high_low_range_20d, gap_open, intraday_range`

→ 其中 `px_to_ma_{5,20,60}` 是教科书均线 TA（3/15 = 20%）；其余是收益/波动/换手/
Amihud 的**基础统计量**；**基本面 0、事件 0、另类 0、微观结构 1（amihud_20d）**。

**因子丰富数据集的类别分解**（348 列实测）：

| 类别 | 列数 | 示例 |
|---|---|---|
| 量价（Alpha101） | 156 | `alpha001..` |
| 量价（GTJA-191） | 58 | `gtja001..` |
| LLM 发现因子 | 9 | `llm_momentum_volume_shock_interaction_007`、`llm_illiquidity_premium_volume_001` |
| 基本面 + 估值 | 21 | `pb, pe_ttm, pcf, earnings_yield, ocf_yield, book_yield, valuation_percentile, pb_own_pctile_2y, quality_composite, growth_composite, eps_ttm, ocfps_ttm, roe, roe_diluted, net_margin, gross_margin, revenue_yoy, net_income_yoy, debt_to_asset, inventory_turnover, operating_cash_to_revenue` |
| 宏观 | 22 | `macro_yield_10y, macro_shibor_*, macro_repo_dr007, macro_afre, macro_m1, macro_cpi_yoy, macro_ppi_yoy` |
| 另类 / 资金流 | 4 | `flow_north_hgt, flow_north_sgt, flow_north_total, flow_margin_sh`（北向 + 融资） |
| 跨资产 / 指数 | 22 | `idx_csi300_close/ret5, idx_shfe_copper_*, idx_ine_crude_*, idx_ten_year_treasury_*` |
| 可交易性 | 4 | `is_st, is_suspended, is_limit_up, is_limit_down` |
| 事件 / 预期 | **0** | — |

### 1.4 对"仅有传统 TA、无统计套利或高频 alpha、评分 1.0/5.0"的裁决

**字面指控不成立 / REJECTED**：
- 传统 TA（RSI/MACD/布林）在 `default_registry` 中是 **5/195 = 2.6%**。
- 主体是 WorldQuant Alpha101（101）+ CICC A 股 80（80）+ 国君 GTJA-191（64），
  这是**统计套利/截面 alpha 血统**，不是形态学 TA。
- 真实微观结构因子存在：Amihud 非流动性（日频 + 1 分钟）、量价相关、lead-lag 相关、
  换手集中度、主力净买压、VWAP 偏离、分时量能分布。
- 基本面（21 列）、宏观（22 列）、北向/融资资金流（4 列）、LLM 发现因子（9 列）
  在 348 列数据集中**实测存在**。

**但两条实质批评成立**：

1. **【FIND-R3-01 / P0】认证宇宙与丰富因子集不相交。** 唯一持有
   `FULL_UNIVERSE_GOLD_READY` 证书的面板（5790 符号）只有 15 个特征、
   20% 是均线 TA、基本面/事件/另类全为 0；而 348 列的丰富面板
   **不持有该证书**、符号更少（6.78M vs 10.92M 行）。
   → 用户看到的"系统只有 TA"是**看认证产物**得出的结论，在那个产物上**是对的**。
2. **事件 / 预期类因子 = 0（两个数据集皆是）。** `earnings_revision_score` 与
   `sector_rotation_score` 出现在 `data/v7_feature_groups.py:51-52` 的
   `MEDIUM_TERM_FEATURES` 声明里，但 `_medium_term_features`
   （`data/v7_feature_groups.py:183-205`）只产出 momentum_20d / momentum_60d /
   volatility_20d / reversal_5d / amount_mean_20d / volume_mean_20d / liquidity_20d
   七列 —— **这两个声明列没有任何 builder 会生成它们**（见 FIND-R3-02）。
3. **Level-2 / 逐笔订单流 = BLOCKED_BY_DATA**（A 股 L2 零公开供应商；腾讯"分笔"
   实为 3 秒聚合）。本仓"微观结构"上限是**分钟 OHLCV 派生量**，不是订单流 alpha。
   这是**数据约束**，不是实现缺陷，应记 BLOCKED 而非 FAIL。

**诚实评分建议**（替代 1.0/5.0）：量价 4/5、基本面 3/5、宏观/跨资产 3.5/5、
另类 2/5、微观结构 2.5/5（受 L2 缺位上限约束）、**事件预期 0.5/5**、
**认证产物可用因子广度 1.5/5**（这一项支持用户的直觉）。

---

## 2. 多维度非线性融合 / Nonlinear fusion (Q4 — 用户最看重)

### 2.1 `src/quantagent/fusion/` 到底做了什么 —— **纯线性加权**

`AGENTS.md:169` 声明"融合搜索唯一入口 = `quantagent.fusion` + governed command
`search-factor-fusion`"。实测该模块的全部 7 个方案：

`src/quantagent/fusion/schemes.py:31-47` 的 `BlendScheme` 枚举 =
`equal` / `ic_weighted` / `ic_ir_weighted` / `inverse_volatility` /
`random_simplex` / `single_factor` / `genetic`。

每个方案 `derive_weights()`（`schemes.py:158-216`）返回的都是一个
**长度 = 因子数的权重向量**。打分在 `fusion/evaluation.py:149`：

```python
work["score"] = centred.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
```

即 `score = Σ wᵢ · rank_centred(factorᵢ)` —— **一个矩阵-向量点积**。

**裁决：`quantagent.fusion` 中没有任何交互项、没有任何树模型、没有任何神经网络。
它是 100% 线性的截面加权求和。** `genetic` 方案（`schemes.py:115-155`）用多目标 GA
搜索的仍然是**权重向量**，不是函数形式 —— GA ≠ 非线性。

### 2.2 真正的非线性能力在哪里 —— 存在，且质量高

`src/quantagent/models/interactions.py`（471 行）是本仓**唯一**真正的非线性因子构造，
且论证扎实（模块 docstring 直接引 Gu, Kelly & Xiu 2020：GLM 单因子样条
月度 OOS R² 0.19% vs 弹性网 0.11%≈无增益，而树/神经网络 0.33–0.40%，
差别正在于**是否允许因子之间交互**）。提供两种构造：

- `pairwise_interaction_features`（`interactions.py:344`）：`xᵢ·xⱼ`，rank 归一化后相乘。
- `regime_interaction_features`（`interactions.py:374`）：`xⱼ ⊗ mₜ`，即
  Gu-Kelly-Xiu 式的特征 ⊗ 宏观状态 Kronecker 积。

**配对选择方法值得记功**（`interactions.py:255` `select_interaction_pairs`）：
不按乘积原始 IC 排序（那会把主效应重新选一遍），而是**先把两个父因子截面回归投影掉，
再对残差乘积算 IC** —— 只有交互本身携带父因子没有的信息时才非零。

消费者 `src/quantagent/research/model_comparison.py` 定义 6 条 arm
（`model_comparison.py:485-500`）：`linear_baseline`(加性) /
`linear_pair_interaction` / `linear_regime_interaction` / `linear_all_interaction` /
`gbm`(LightGBM 非线性学习器) / `ensemble_stack`。
`_stack_blocks`（`model_comparison.py:771`）明确拒绝静默丢块 —— 注释记录了历史缺陷：
`if name in blocks` 曾让无法表达自身假设的 arm **静默退化成 baseline 并报出逐位相同的 fold IC**。

### 2.3 【FIND-R3-03 / P0】非线性能力**没有接入受治理路径**

实测两条独立证据：

1. **调用图**：`grep -rn "pairwise_interaction_features\|regime_interaction_features\|select_interaction_pairs" src/ services/`
   的**唯一**生产侧命中是 `src/quantagent/research/model_comparison.py`。
   `src/quantagent/fusion/` 下 **0 处**引用 `models.interactions`。
2. **受治理命令白名单**：`services/quant_api/services/jobs.py` 共 47 个 allowlisted 命令
   （实测枚举）。其中与融合/模型相关的只有 `search-factor-fusion`（线性）、
   `synthesize-factors-v7`、`evaluate-factor-library-v7`、`train-v8-deep`。
   **`audit-nonlinear-factors`（`src/quantagent/cli/nonlinear.py:21`）不在白名单内。**

**失败场景（可复现）**：操作员在工作站上想回答"多因子非线性混合比线性加权好多少"。
他能启动的唯一融合任务是 `search-factor-fusion`；该任务枚举 4 个拟合方案 + 3 类对照，
**全部是线性加权**，产出 Pareto 前沿与 DSR。报告里不会出现任何交互项或树模型，
也不会有任何提示说明"非线性未被搜索"。**结论"最优融合 = ic_ir_weighted"
在读者看来是在包含非线性的空间中得出的，实际上非线性从未进入候选集。**

复现：
```bash
AI_quant_venv/bin/python3 -c "
import re; src=open('services/quant_api/services/jobs.py').read()
keys=re.findall(r'^    \"([a-z0-9\-]+)\": \{', src, re.M)
print('audit-nonlinear-factors' in keys, len(keys))"     # -> False 47
grep -rn "models.interactions" src/quantagent/fusion/ | wc -l   # -> 0
```

**修法建议**（主角色实施）：
- (a) 把 `audit-nonlinear-factors` 登记进 `jobs.py` allowlist（含 path_inputs/outputs、
  无 `n_trials` 参数），使非线性可从工作站发起；**或**
- (b) 在 `fusion/schemes.py` 增加 `pair_interaction` / `regime_interaction` / `gbm`
  三个 scheme，让它们**与线性方案在同一 `build_scheme_specs()` 枚举内竞争**，
  从而自动计入 `n_trials` 并进入同一 DSR 收缩项与同一 Pareto 前沿；
- (c) 最低限度：`search-factor-fusion` 的产出必须显式声明
  `"searchedModelClasses": ["rank_weighted_additive"]`，使"没搜非线性"这件事
  在报告里**可见**而不是靠读者推断。
  三选一即可闭合，但 **(c) 是不可省略的诚实性下限**。

### 2.4 试验次数是否诚实进入 DSR —— **PASS**

`fusion/schemes.py:219-274` `build_scheme_specs()` 枚举全部候选（含 8 个随机对照 +
6 个单因子对照），docstring 明写"Callers must not filter it after the fact to make the
deflated Sharpe ratio look better"。`services/quant_api/services/jobs.py` 的
`search-factor-fusion` 条目注释：`n_trials` is intentionally absent: it is derived
from the enumerated search space so an operator cannot deflate their own Sharpe ratio` —
**API/CLI 均不暴露 `n_trials`**，与 `AGENTS.md:170-171` 铁律一致。

裁决：**PASS**（对**线性**搜索空间而言）。但注意 §2.3 的推论：
DSR 惩罚的是"搜了 18 个线性方案"，**不包含**任何非线性试验，
因为非线性根本没被搜索 —— 这不是造假，但意味着一旦按 (b) 接入非线性，
`n_trials` 必须相应增长，DSR 会变严格。

---
