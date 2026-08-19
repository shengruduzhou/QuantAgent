# Round 23 — R11 强化学习：风险项对照实验 / RL Risk-Term Ablation

- 角色 / Role: R11 — RL 奖励设计与实测
- 基线 / Baseline: `b313153` (main tip)
- 分支 / Branch: `agent/round23-rl-ablation`
- 范围 / Scope: **仅** `src/quantagent/rl/**` 与 `tests/rl/**`
- 日期 / Date: 2026-08-19

## 0. 本轮要补的证据 / The gap being closed

Round 22 实现了 `drawdown_lambda` / `volatility_lambda`（默认 `0.0`，已并入 main），
但对照实验没跑完。主角色在合并说明里写明：

> 本次合并提供的是**能力**，不是**证据**。

用户标准原话：「**一定要保证有用才能使用**」。本轮唯一任务是把
「风险项到底有没有用」跑出答案。

---

## 1. 先解决阻塞：被动账本每天换手一次

### 1.1 为什么这是阻塞而不是瑕疵

RL 奖励是**对被动账本的超额**。被动账本因此不是配角，它是基准，它的交易行为
被烘焙进环境报出的每一个数字。Round 22 用的
`train_ppo.equal_weight_book_from_predictions` 每个 session **从零重建** top-k 名单。

在 12 bps 成本下，一个每天换手 ~90%（双边 1.8）的基准每 session 自己就要付
**21.3 bps**。策略和基准都在以远快于任何回撤项能影响的速度流血 ⇒
各 arm 会按「少交易」被排序，实验会**静默地变成一个关于交易成本的研究**，
而不是关于回撤的研究。

### 1.2 修复前后实测（本轮数据集，690 个 session，`top_k=30`）

由 `quantagent.rl.reward_ablation_experiment.book_turnover_report` 计算，
双边口径（整本换掉 = 2.0），排除第一个 session（那是建仓不是调仓）。

| 指标 | 修复前 `daily_topk_book` | 修复后 `hold_band_book_from_predictions` |
|---|---:|---:|
| 换手 **中位数** | **1.8000** | **0.0000** |
| 换手 均值 | 1.7740 | 0.1935 |
| 换手 p90 | 2.0000 | 0.9333 |
| 每 session 新进名字（中位） | 27 / 30 | 0 |
| **持有期中位数（session）** | **1** | **10** |
| 持有期均值（session） | 1.13 | 9.98 |
| 账本 gross 中位 | 1.000 | 1.000 |

修复前中位换手 1.80、p90 2.00 —— **确认了上一轮被打断前那条「每天换手一次」的
观察**（上一轮记的是 2.0，本轮在这份面板上实测中位 1.80、p90 2.00：
最坏的日子确实是整本换掉）。

均值换手降低 **9.17×**（1.7740 → 0.1935）。以 12 bps 计，基准自身的成本从
**21.3 bps/session** 降到 **2.3 bps/session**。

修复后的构造（`src/quantagent/rl/books.py`）：
`top_k=30`、`exit_rank=90`（出场带比入场带宽 3×）、`min_hold_sessions=10`、
`rebalance_every=5`。非调仓日账本原样结转 ⇒ 换手**恰好** 0，这就是中位数为 0 的原因。
唯一的强制出场是「当天没有有限 alpha」——环境要求每个目标名字有有限 alpha，
带着一个没打分的名字会直接 raise。

（环境内部的 `turnover_passive` 与账本自身的换手不同，因为涨跌停/停牌约束会
按住账本想动的仓位：评估窗口 208 个 transition 上实测 env 口径均值 **0.20449**。）

### 1.3 修复的实现与钉住它的测试

`src/quantagent/rl/books.py` — `hold_band_book_from_predictions`：

- 只在调仓 session（每 `rebalance_every` 个）做增减；其余 session **原样结转**
  （不是重新推导出相同权重 —— 后者在浮点上不逐位相等，而环境按 `sum |dw|` 收费，
  30 个名字上 1e-17 的抖动在 690 个 session 上会累积成真实成本且不会出现在任何摘要里）。
- 出场带 `exit_rank` 必须 ≥ `top_k`，否则直接 raise：一条不比入场带宽的「带」不是带，
  它会在第一次被超越时就换人，把每日churn 换个名字重演一遍。
- 最小持有期 `min_hold_sessions` 保护刚入场的名字。
- 唯一强制出场 = 当天没有有限 alpha；释放的权重**不再分配**给幸存者（否则为一个
  缺失分数向所有幸存者收换手费），账本 gross 下沉、到下一次调仓恢复。

`tests/rl/test_hold_band_book.py`（6 条）把上述性质写成断言，其中
`test_hold_band_book_turns_over_far_less_than_the_daily_rebuild` 在同一个
「排名每天整体轮转」的构造上同时算两种账本，**对修复前的构造是失败的**
（daily rebuild 中位换手实测 2.0）。
`test_book_rows_do_not_depend_on_later_sessions` 钉住无前视：截断输入后，
存活的行逐字节相同。

---

## 2. 实验设置 / Experiment setup

### 2.1 数据来源（全部为真实产物，无合成数据）

| 角色 | 产物 | 事实 |
|---|---|---|
| 行情面板 | `runtime/data/v7/silver/market_panel/market_panel.parquet` | 2022-10-01→2025-09-30 切片 2,655,061 行 / 727 sessions；带真实 `is_limit_up` / `is_limit_down` / `is_suspended` |
| 预测（alpha） | `runtime/stage6_classical_walkforward/wf/walkforward_predictions.parquet` 的 `alpha_5d` | `lightgbm-csrank@cov099`，**6 折 purged walk-forward，每折 120 sessions 全部为样本外**；2022-10-31→2025-08-29 取 690 signal sessions / 3,427 symbols |
| 停牌证据 | `runtime/data/u0/panel/session_gaps.parquet` | 用于证明缺失 bar 属 `SUSPENDED`；环境对未证明的缺失 **fail-closed** |

**评测窗口纪律**：`configs/quarantined_windows.json` 的两个禁评窗
（烧毁 holdout 2025-09-01→2026-05-18、冻结新鲜窗 2026-05-19+）**均未被读取**。
实验窗口在 2025-08-29 处停止。

**宇宙剔除**：149 / 3,427 个 symbol（4.3%）在窗口内某个 session 缺 bar 且
`session_gaps` 未判为 `SUSPENDED`。环境按设计对它们 fail-closed。这些名字被
从候选宇宙**整体剔除**（而不是放松守卫）。剔除用到了全窗口信息，因此在
**宇宙定义**上是一个轻度前视；它对被动账本与所有 arm **完全同等地**施加，
且数量被打印出来以免它悄悄变大。剩余宇宙 3,278 symbols。

### 2.2 训练 / 评测切分

| | 窗口 | sessions |
|---|---|---|
| 训练 | 2022-10-31 → 2024-10-23 | 480（奖励时钟被 `reward_end_date_limit=2024-10-23` 截断，故训练 transition 数为 478） |
| 评测 | 2024-10-24 → 2025-08-27（signal） | 208 |

训练段的奖励时钟被显式截断在切分点（最后一个训练 transition 的 reward end = 2024-10-23，
第一个评测 signal = 2024-10-24），因此**没有任何训练 transition 能读到评测段的收益**。
仅截断 signal date 是不够的：奖励区间是 `close(T+1) → close(T+2)`。

**同一条规则也把评测段挡在禁评窗之外**：评测的奖励时钟同样截断在 2025-08-29。
不截断的话，最后两个 signal（08-28 / 08-29）的奖励区间会落到 2025-09-01 与 2025-09-02，
**正好踩进烧毁 holdout 的第一、二天**。截断后评测 transition 从 210 降到 **208**。
（这是本轮实际改出来的一个洞：第一次跑出来的 `steps=210` 就是踩进去的版本，已作废重跑。）

### 2.3 同一把尺子（反循环论证）

所有 arm —— 包括带惩罚训练出来的 —— 都在
`drawdown_lambda = volatility_lambda = 0` 的评估环境里打分。
用带惩罚的指标去评带惩罚的策略等于假定结论。
`tests/rl/test_reward_ablation_harness.py::test_the_yardstick_never_carries_the_arm_s_risk_lambdas`
把这条写成断言：两个只在训练 λ 上不同的对照 arm 必须给出**逐位相同**的评测指标。

### 2.4 λ 档位不是「小到没影响」——先量出惩罚项的量级

「处理组和对照组没差别」有两种可能：项没用，或者 **λ 数值上可忽略**。
在写裁决之前先把这两者分开。用同一个随机策略走完 478 个训练 transition，
分别记录奖励各分量（`src/quantagent/rl/pit_portfolio_env.py` 的 info 流）：

| 训练 λ | Σ value_add | Σ 惩罚（望远镜求和） | Σ&#124;惩罚&#124; / Σ&#124;value_add&#124; |
|---|---:|---:|---:|
| `λ_dd=1` | −0.284381 | +0.062342 | **0.1197** |
| `λ_dd=2` | −0.284381 | +0.124683 | **0.2393** |
| `λ_vol=25` | −0.284381 | +0.021604 | **0.3104** |
| `λ_vol=100` | −0.284381 | +0.086418 | **1.2415** |

⇒ 本轮的 λ 网格从「占收益信号 12%」一直到「压过收益信号（124%）」。
**任何「没差别」都不能用「λ 太小」来解释。**
（`λ_dd` 的望远镜和 = λ × 超额 MDD_T，随机策略实测超额 MDD = 6.23pp，
与 §2.5 表里 random arm 的 `excess_max_drawdown` 同源。）

### 2.5 裁决规则（在看到处理组数字**之前**写下）

写这一节时，已知的只有两个对照组（`zero` / `random`）与一次 400k 的单 seed 试跑；
6 个训练 arm 的结果尚未产出。规则先定，避免事后挑一个好看的指标当结论。

1. 若**没有任何**训练 arm 的 `cumulative_value_add` 均值超过 `zero` 对照（构造上恰为 0）
   一个 seed 标准差以上 ⇒ 在这套设置下 RL 本身不产生超额，
   **风险项无从推荐**，裁决 `不启用`。
2. 若 `old_reward_lambda0` 胜过 `zero`，且某个 λ arm 在**逐 seed 配对**比较上
   于 Calmar 优于 `old_reward_lambda0` 且符号在各 seed 上一致 ⇒ 裁决 `启用`，给出该 λ 与代价。
3. 其余情形 ⇒ `证据不足`，并写清还缺什么。

两个问题必须分开回答，不能混成一句：

- **(a) 机制**：惩罚项有没有把回撤往它声称的方向推？（看 `max_drawdown` 与 `mean_gross` 的配对差）
- **(b) 价值**：这值不值得在生产里打开？（看它相对 `zero` 对照的绝对位置）

(a) 为「是」而 (b) 为「否」是完全可能的，也**必须如实这样写**。

---

## 3. 顺带挖出的缺陷：仓库自带的 hold band 在全宇宙上不产生持有期

写 `src/quantagent/rl/books.py` 之前先查了仓库是否已有同类实现 —— **有**：
`src/quantagent/portfolio/hold_band.py`，自述「turnover-controlled top-K selection」，
且**已经接在 `scripts/rl_pit_train_eval.py` 上**。于是先在**同一份 690 session 数据**上
实测它，而不是假定它不合用：

| 账本 | 换手中位 | 换手均值 | **持有期中位（session）** |
|---|---:|---:|---:|
| `hold_band` n50/e30/x150（**模块自带默认值**） | 1.4737 | 1.4580 | **1** |
| `hold_band` n30/e30/x90 | 1.6000 | 1.5438 | **1** |
| `hold_band` n30/e30/x300 | 1.1333 | 1.1334 | **1** |
| `hold_band` n30/e30/x900 | 0.5333 | 0.6292 | 2 |
| `hold_band` n30/e30/x3000 | 0.0000 | 0.0484 | 19 |
| `rl.books` k30/x90/minhold**0**/rebal**1** | 1.6000 | 1.5438 | 1 |
| `rl.books` k30/x90/minhold5/rebal1 | 0.4000 | 0.3955 | 5 |
| `rl.books` k30/x90/minhold10/rebal1 | 0.2000 | 0.2088 | 10 |
| `rl.books` k30/x90/minhold10/rebal5（本轮采用） | 0.0000 | 0.1935 | 10 |

两条结论：

1. **`rl.books` 在 `min_hold_sessions=0, rebalance_every=1` 时逐位复现
   `build_hold_band_weights`**（x90 两者 1.6000/1.5438 完全相同，x300、x900 同理）。
   它是那条规则的**忠实超集**，不是另起炉灶的第二套选股规则 ——
   `tests/rl/test_hold_band_book.py::test_it_reproduces_the_repository_hold_band_when_the_new_knobs_are_off`
   把这条钉住；一旦两者分叉，RL 基准就不再能和生产账本比较。

2. **排名带（rank band）在 ~3,253 只的宇宙上根本无法产生持有期。**
   一个名字要活下去就得留在 3,253 名里的前 `exit_rank` 名内，而日频截面 alpha 不会
   让同一批名字长期待在头部：直到把带宽开到覆盖 **~92% 的宇宙**（`exit_rank=3000`）
   它才开始起作用 —— 而那时它已经不再选股了。
   在**模块自己的默认值**下持有期中位数是 **1 个 session**。

⇒ **`HoldBandConfig` 的换手控制在全 A 宇宙上是装饰性的**（Round 20 DEF-030 的同一形状：
一个读起来像守卫、实测下不约束任何东西的旋钮）。真正能造出持有期的是
**基于时间的最小持有期**，而 `HoldBandConfig` 没有这个字段。

**本轮没有修它**：`src/quantagent/portfolio/` 不在本角色的写入范围内，
而且它被生产账本消费 —— 改换手就是改策略，需要它自己的证据链（`ACCEPTANCE_RULES.md`）。
**转交主角色路由**。本轮只在 `src/quantagent/rl/books.py` 里补上 RL 基准需要的那部分，
并在 docstring 里写明「更好的归宿是 `portfolio.hold_band` 本身」。
