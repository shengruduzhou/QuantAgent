# Round 24 — R11 强化学习：风险项裁决 / RL Risk-Term Verdict

- 角色 / Role: R11 — RL 奖励设计与实测
- 基线 / Baseline: `3d4ecd8` (main tip)
- 分支 / Branch: `agent/round24-rl-verdict`
- 范围 / Scope: **仅** `src/quantagent/rl/**` 与 `tests/rl/**`（本轮无生产代码改动）
- 日期 / Date: 2026-08-20

## 0. 本轮的唯一任务 / The one thing this round does

Round 22 实现了 `drawdown_lambda` / `volatility_lambda`（默认 `0.0`）。
Round 23 清掉了阻塞（被动账本换手中位 1.8 → 0）并**在看到任何处理组数字之前
写下了裁决规则**，但被 session limit 打断，主结果表没有落盘。

本轮**不新增任何设计**，只做一件事：把 8 个 arm × 5 seed 跑完，
按 Round 23 §2.5 那条**预注册**规则宣判。

**状态：RUNNING —— 本文件按 arm 增量落盘。**

---

## 1. 复现与产物 / Reproduction

### 1.1 输入（全部为真实产物；无合成数据）

| 角色 | 产物 | sha256（源文件） |
|---|---|---|
| 行情面板 | `runtime/data/v7/silver/market_panel/market_panel.parquet` | `03b7625482494a56…` |
| 预测（alpha） | `runtime/stage6_classical_walkforward/wf/walkforward_predictions.parquet` | `6a882452f97dc7df…` |
| 停牌证据 | `runtime/data/u0/panel/session_gaps.parquet` | `776663fb967527fd…` |

切片（2022-10-01→2025-09-30 面板 / 2022-10-31→2025-08-29 预测）由
`scratchpad/rl24/prep.py` 重新生成，并与 Round 23 的切片**逐字节比对**：

| 切片 | sha256 | 与 Round 23 相同 |
|---|---|---|
| `panel_wf.parquet` | `d5fa578d8a16d096…` | **是** |
| `preds_wf.parquet` | `9ce3ab5036532bbc…` | **是** |
| `gaps_wf.parquet` | `871d10953519b7b2…` | **是** |

面板 2,655,061 行 / 727 sessions；预测 2,339,803 行 / 690 signal sessions / 3,427 symbols。
`alpha_5d` 来自 `lightgbm-csrank@cov099` 的 6 折 purged walk-forward，**每折 120 sessions 全部样本外**。

### 1.2 为什么本轮全部重跑，而不是引用 Round 23 已落盘的数字

Round 23 的 6 个训练 arm 是在 `4dccec3` 上跑的，那时 `mean_gross` **还没加进
`EpisodeMetrics`**（`bde4b5d` 才加）。本轮任务书要求用 gross 检查退化解，
而已训练模型没有被持久化 ⇒ 无法只补评估。

因此本轮在 **main tip `3d4ecd8` 的已提交代码**上把 8 个 arm × 5 seed 全部重跑。
`bde4b5d` 对 env 的改动只是往 info 里多发一个 `weights_passive` 键，
对策略、奖励、随机数流**无影响**（`git show bde4b5d` 可核对），
所以重跑同时构成一次**独立复现检查**：同 `(arm, seed)` 的数字应与 Round 23 一致。
该检查的实测结果见 §4.4。

### 1.3 命令

```bash
PYTHONPATH=<worktree>/src OMP_NUM_THREADS=2 \
AI_quant_venv/bin/python3 -m quantagent.rl.reward_ablation_experiment \
  --panel  <sc>/panel_wf.parquet \
  --predictions <sc>/preds_wf.parquet \
  --session-gaps <sc>/gaps_wf.parquet \
  --start 2022-10-31 --end 2025-08-29 \
  --top-k 30 --exit-rank 90 --min-hold-sessions 10 --rebalance-every 5 \
  --train-sessions 480 --timesteps 400000 --n-envs 4 --device cpu \
  --arms <arm> --seeds 1729,20260819,7,13,42 \
  --results <sc>/results_<tag>.jsonl
```

6 个训练 arm 按 arm 拆成 6 个并行进程（各自的 `--results` 是**可续跑**的）。
`--eval-reward-end-limit` 未显式给出 ⇒ 取 `--end` = 2025-08-29，
这是**必要的**：不截断的话最后两个 signal 的奖励区间会落进
`configs/quarantined_windows.json` 的烧毁 holdout 首日。评测 transition 因此为 **208** 而非 210。

---

## 2. 实验设置 / Setup

| | 窗口 | 数量 |
|---|---|---|
| 训练 | 2022-10-31 → 2024-10-23 | 480 book sessions（奖励时钟截断在 2024-10-23 ⇒ 478 transitions） |
| 评测 | 2024-10-24 → 2025-08-29 | 210 book sessions ⇒ **208 transitions** |

- **宇宙**：3,427 → **3,278** symbols。149 个（4.3%）在窗口内某 session 缺 bar 且
  `session_gaps` 未判为 `SUSPENDED`，环境按设计 fail-closed，故整体剔除
  （轻度前视，位于**宇宙定义**上，对被动账本与全部 arm 同等施加）。
- **被动账本**：`hold_band_book_from_predictions(top_k=30, exit_rank=90,
  min_hold_sessions=10, rebalance_every=5)`；账本自身换手中位 **0.0000**、均值
  **0.19352**、持有期中位 **10** session。修复前的 `daily_topk_book` 中位 **1.8000**、
  持有期中位 **1**。
- **规模**：每 arm 每 seed **400,000** PPO timesteps、`MlpPolicy` 默认超参、
  `n_envs=4`（`DummyVecEnv`）、CPU。5 个 seed：`1729, 20260819, 7, 13, 42`。
- **同一把尺子**：所有 arm（含带惩罚训练的）都在
  `drawdown_lambda = volatility_lambda = 0` 的评估环境里打分。
  `tests/rl/test_reward_ablation_harness.py::test_the_yardstick_never_carries_the_arm_s_risk_lambdas`
  把这条钉成断言。
- **测试**：`tests/rl/` 104 passed（本轮未改代码，仅确认基线绿）。

### 2.1 预注册裁决规则（Round 23 §2.5，原文照抄）

> 1. 若**没有任何**训练 arm 的 `cumulative_value_add` 均值超过 `zero` 对照
>    （构造上恰为 0）一个 seed 标准差以上 ⇒ 在这套设置下 RL 本身不产生超额，
>    **风险项无从推荐**，裁决 `不启用`。
> 2. 若 `old_reward_lambda0` 胜过 `zero`，且某个 λ arm 在**逐 seed 配对**比较上
>    于 Calmar 优于 `old_reward_lambda0` 且符号在各 seed 上一致 ⇒ 裁决 `启用`。
> 3. 其余情形 ⇒ `证据不足`。

并要求把两个问题分开回答：**(a) 机制**——惩罚项有没有把回撤往它声称的方向推；
**(b) 价值**——值不值得在生产里打开。

#### 规则里的歧义，以及本轮选的解释

规则 1 说「超过 `zero` 一个 seed 标准差以上」，但 `zero` 对照的 seed 标准差
**构造上恰为 0**（它没有随机性）。「一个 seed 标准差」有两种读法：

- **读法 A（保守，本轮采用）**：用**该训练 arm 自己**跨 seed 的标准差，
  即要求 `mean(value_add) > 0 + std(value_add)`。
- 读法 B（宽松）：`zero` 的 std 为 0 ⇒ 门槛退化为 `mean(value_add) > 0`。

本轮**采用读法 A**，并**同时报告读法 B 的结果**（§4.2），
以证明结论不依赖于挑哪个解释。

---

## 3. 结果 / Results

### 3.1 先决检查 A：λ 网格不是「小到没影响」

「处理组和对照组没差别」有两种可能：项没用，或者 **λ 数值上可忽略**。
在写裁决之前先把这两者分开。用同一个随机策略（`default_rng(1729)`）走完 478 个训练
transition，逐分量记录（`scratchpad/rl24/penalty_scale.py`，与 Round 23 §2.4 同法重跑）：

| 训练 λ | Σ value_add | Σ 惩罚（望远镜求和） | Σ&#124;惩罚&#124; / Σ&#124;value_add&#124; |
|---|---:|---:|---:|
| `λ_dd=0.5` | −0.284381 | +0.031171 | **0.0598** |
| `λ_dd=1` | −0.284381 | +0.062342 | **0.1197** |
| `λ_dd=2` | −0.284381 | +0.124683 | **0.2393** |
| `λ_vol=25` | −0.284381 | +0.021604 | **0.3104** |
| `λ_vol=100` | −0.284381 | +0.086418 | **1.2415** |

⇒ 本轮 λ 网格从「占收益信号 6%」一直到「压过收益信号（124%）」。
**任何「没差别」都不能用「λ 太小」来解释。**
（`Σ value_add` 与 Round 23 逐位相同：`−0.2843807233463903`。）

### 3.2 先决检查 B：两个惩罚项的**稀疏度**差约 10 倍

同一条随机轨迹、478 个训练 transition：

| 惩罚项 | 非零 step | 占比 |
|---|---:|---:|
| `drawdown_penalty`（λ_dd 任意 > 0） | **26 / 478** | **5.44%** |
| `volatility_penalty`（λ_vol 任意 > 0） | **249 / 478** | **52.09%** |

回撤项按定义只在刷新历史最大回撤（或被动账本刷新）的那一刻付费：
`ΔMDD_t = MDD_t − MDD_{t−1}` 在其余时刻恒为 0（本轮实测 **452 个 step 恰好为 0.0**，
15 个严格为正、11 个严格为负 —— 负值来自被动账本刷新得更快时的 credit）。
PPO 还要靠优势估计把这个信号分摊回导致回撤的那些动作上。
下行半方差项则在**每个亏损日**都开火。

> **更正 Round 23**：Round 23 §4.1 记「23 / 478 (4.8%)」。本轮同一条轨迹、
> 同一个 λ、**惩罚求和逐位相同**（`0.062341543537462574`）却数出 26 个非零 step。
> 差别不在数据而在计数口径 —— Round 23 漏掉了 11 个 credit 里的一部分。
> 正确的数是 **26 / 478 = 5.44%**（`scratchpad/rl24/penalty_sign.py`）。
> 结论方向不变（仍然比波动项稀疏近 10 倍），但数字以本轮为准。

### 3.3 先决检查 C：`zero` 对照在构造上仍恰为 0

| 指标 | `zero` arm（5 seed 全部） |
|---|---|
| `cumulative_value_add` | **0.000000**（std = 0） |
| `excess_max_drawdown` | **0.000000**（std = 0） |
| `mean_gross` | **0.977244** = `mean_gross_passive`（std = 1.24e−16） |
| `mean_turnover` | 0.204487 = `mean_turnover_passive` |
| `nav` / `max_drawdown` | 1.444285 / 0.186281 |

Round 21 DEF-036 修好的「零动作恰好 0」在真实数据上依旧成立。
被动账本的 env 内换手是 **0.2045**（账本自身是 0.1935；差额来自涨跌停/停牌
把账本想动的仓位按住）。

### 3.4 主结果表 / Main table

（**待填**——6 个训练 arm 运行中，按 arm 增量落盘。）

---

## 4. 裁决 / Verdict

（**待填**。）
