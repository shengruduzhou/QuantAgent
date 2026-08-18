# Round 22 — 主角色整合裁决 / Main-Role Integration

- 日期 / Date: 2026-08-19
- 基线 / Baseline: `5957011` → 本轮起点
- 角色 / Role: Main（CIO，唯一合并权）

## 组织形式

沿用第 21 轮验证过的**分批 3 个** agent（第 20/21 轮各 11 个和 8 个全被 session
limit 打死；分批 3 个则 3/3 全部交付）。本轮三个角色各自在**独立 worktree**
里直接改代码并提交到自己的分支，主角色 review 后合并 —— 这是上一轮 R8 验证可行的模式。

| Role | 分支 | 领域（互不相交） |
|---|---|---|
| R1 回测 | `agent/round22-fill-semantics` | `backtest/`、`quant_math/ashare.py`、`quant_math/transaction_cost.py` |
| R9 治理可靠性 | `agent/round22-governance-chain` | `governance/` |
| R11 RL | `agent/round22-rl-reward` | `rl/` |
| Main | 直接在 main | `services/quant_api/`、`apps/quant-ui/` |

---

## 主角色本轮已修

### DEF-042【P1】多资产 playbook 把没打印的交易日填成 0% 收益

来源：R9 A-06。`market_playbooks_v3.time_series_momentum` 用**并集**日期建面板
再 `.ffill()`。三条后果全部朝乐观方向：

1. 策略被记为在**从未打印过的 open 价**上成交，而 `assumptions` 声明的是
   open-to-open marking；
2. 每个缺口交易日变成 **0% 收益**而不是"缺失"；
3. 填充改变了已实现波动率，而权重是**逆波动率**的 ⇒ 仓位按"哪个标的数据最差"移动。

R9 实测同一价格过程：完整观测 **+33.5102%** vs ffill 面板 **+41.1854%**，
**7.67pp 纯属填充**。

修法：**没打印就是不能交易**，面板限制到"每个标的都被观测到"的交易日，
被丢弃的天数作为 `unobservedSessionsDropped` / `sessionsEvaluated` **公布**而非吸收。
`short_term_reversal` 的基准腿是同一缺陷（**DEF-022 原样重现**：填充的基准缺口
= 0% 基准日 ⇒ 超额虚增整个缺失涨跌幅），基准改为留 NaN 并公布
`benchmarkMissingSessions`。

**写测试时的一处自我更正**：初版断言"填充总是压低波动率"。**这是错的** ——
填充同时插入一个平坦日**和**一个翻倍跳空，方向取决于价格路径。测试改为钉住
真正的不变量：填充后的序列含有**市场从未产生过的收益观测**（每个未打印交易日
恰好一个 0%），而观测序列一个都没有。

### DEF-043【P1】工作站默认面板的特征砍半从未被记录

来源：R3b FIND-R3b-02。`jobTemplates.ts` 的注释记录了宇宙从 3,872 扩到 5,790，
**但同一次切换把特征从 348 列砍到 15 列这一半从未被写下来**。

认证面板的 15 个特征：`ret_{1,5,20,60}d`、`px_to_ma_{5,20,60}`、`vol_{20,60}d`、
`turnover_20d`、`volume_ratio_5_20`、`amihud_20d`、`high_low_range_20d`、
`gap_open`、`intraday_range` —— **零 Alpha101、零 GTJA-191、零基本面、零事件、零宏观**，
其中 3 个是均线。**工作站读起来像技术分析系统，是因为在这个产物上它就是。**

`FULL_UNIVERSE_GOLD_READY` 只认证结构就绪：它的 18 项检查
（`build_u0_full_universe_gold.py` + `readiness_tiers.py`）**没有一项涉及特征广度**。

已在**两处默认点**（训练模板与融合搜索表单）写清这个取舍，包括 348 列备选面板
（3,638 符号、**零 STAR 零 BSE**、qfq 而非 hfq）为什么不是直接替代品。
**两个产物互不支配**，默认继续选宇宙广度，但取舍现在是被声明的而不是被继承的。

---

## 本轮排队未做 / Queued, deliberately not done

### Q-01【P1】`OrderManagerConfig.max_participation_rate` 是个没人读的旋钮

R9 A-09。`src/quantagent/execution/order_manager.py:135` 声明它，
`src/quantagent/backtest/ashare_execution_simulator_impl.py:151` **写入**它
（来自 `config.volume_participation_cap`），**而 `order_manager.py` 从不读它**。
调用方以为自己设置了约束，实际什么也没发生。

**为什么本轮不修**：不能简单"接上"。这个字段的语义是"一笔单可以吃掉一根 bar 的多少"，
是**成交定量**概念；而 OMS 不撮合。把它接到 pre-trade 闸门上会重演上一轮
已经论证过要避免的问题——**pre-trade 限额与 venue 成交上限设成同一个数会让部分成交不可达**。
正确的修法是**从 `OrderManagerConfig` 移除**，但那要同时改
`ashare_execution_simulator_impl.py`，本轮属于 R1 的领域。**待 R1 分支合并后统一处理。**

### Q-02【P0】给认证全宇宙面板补上价量因子

R3b 实测：Alpha101 `wall=165.3s`、GTJA-191 `wall=220.3s`，
**合计 6.4 分钟 / 21 GB 内存 / 约 10 GB 磁盘 = 165 个因子**。本机实测
`runtime/data/gold/full_universe/dataset.parquet`（2.27 GB）在位，可用内存 55 GB
⇒ **计算侧不是阻塞**。

真阻塞只有两条，且**只卡基本面那一半**：
1. 基本面抓取宇宙冻结在 3,658（**STAR 613 全部 + BSE 328 全部从未被抓过**），
   覆盖 63.1% 符号 / 79.38% 行，且**是近年更差**（2019 年 93.41% → 2026 年 66.43%）；
2. hfq/qfq 口径不同，必须重算不能搬列。

**价量因子（Alpha101 + GTJA-191）不受这两条阻塞**。这是本仓离"多因子"最近的一步，
但它会产出一个新的认证产物并牵动 readiness 闸门语义，**应当独占一轮**，
不与三个 agent 分支的合并混在一起。

---

## 三个 agent 分支的合并裁决 / Merge adjudication

分批 3 个的结果：**R9 完整交付；R1 与 R11 再次被 session limit 打断**
（reset 1:40am），但**两者的产出都被完整抢救并合并**——因为它们都在自己的
worktree 里留下了可验证的代码，而不是只有一段对话。

| Role | 状态 | 裁决 |
|---|---|---|
| R9 治理哈希链 | 完整（2 commits） | **合并** |
| R1 撮合语义 | 被打断，0 commits，改动在 stash + worktree | **抢救后合并** |
| R11 RL 奖励 | 被打断，1 commit + 未提交的实验台 | **合并，但默认关闭** |

### R9 — 合并

实测（4 进程 × 40 条，**每次 append 前都设 Barrier**，使碰撞不靠调度运气）：

```
修复前  lines=160  distinct_seq=41   forked_prev=40  reachable=0    orphaned=160
修复后  lines=160  distinct_seq=160  forked_prev=0   reachable=160  orphaned=0
```

比第 21 轮报的数字**更糟**：Barrier 让链在第 0 条就分叉，所以
`reachable_from_genesis=0` 而不是 1。**160 条治理记录全部孤儿，且不抛任何异常。**

锁是从 `paper/execution_journal.py` **逐字搬过来**的而不是新造；tail 在**锁内**重读；
fsync 失败则 latch 关闭。**DEF-017 没有被重新引入**：锁内重读是跨进程新鲜度，
而 DEF-017 禁止的是"在你自己的持久化失败之后仍然相信磁盘字节"——latch 就是这条边界。

附带收益：4000 次 append **25.481s → 2.564s**（O(n²) 的 tail 扫描没了）。

**R9 的一条重要观察值得单独记**：A-04 之所以**没有造成生产损失，正是因为 A-05**
——生产环境里根本没人写这个日志。**这次修的是地基，不是正在流血的伤口。**

### R1 — 抢救后合并

它没来得及 commit，但 worktree + stash 里有 367 行源码改动、2 个新测试文件和
一份报告。抢救后 `tests/backtest` + `tests/domain` **339 passed**。

关键是**它没有削弱黄金场景，而是逐行重算了手算表**：
成交价 `10.0020001 → 10.0051`、现金流出 `100,046.00620 → 100,077.01326`、
NAV `999,953.9938 → 999,922.98674`，且 `abs=1e-7` 的精度**保持不变**。
**成本一律上升**，是保守方向。

F-03 带来了本轮最有说服力的实测：**30 个真实标的、2022-01..2026-05、31,650 根 bar**，
close 判据得到 **210** 次涨停 vs open 判据 **31** 次 ⇒ **189 次假阻断**，
外加 **10 根开盘即封板、本来会被错误成交**的 bar。

**仍然缺的**：真实多年面板上的前后对比表 ⇒ 修正的**幅度**未量化，
只知道**符号**（旧数字偏乐观）。这些数字在第 21 轮已因 NAV 时钟原因作废。

### R11 — 合并但默认关闭

奖励新增两个可选风险项（超额最大回撤、超额下行半方差），**都是对同一个受约束
被动账本的差分**——这正是保住「零动作恰好得 0」的原因：零动作下 `w` 与 `w_b`
逐比特相同，两条 NAV、两条 peak、两条 MDD 路径全等 ⇒ 每一项恒等于 0.0
（5 个 gross × 4 组 λ = 20 例，`abs=1e-12`）。**默认 λ=0 逐比特复现旧奖励。**

两个被**评估后否决**的方案（不是没读过）：Differential Sharpe 是**绝对** Sharpe 的
增量，会让被动账本自身的波动给零动作打出非零分，**破坏 DEF-036 刚修好的保证**；
potential-based shaping 的 policy invariance **恰恰是我们不要的性质**，
而该定理反过来被用作判据——补一个终止修正会让惩罚在 episode 上抵消为 0，
成为**装饰性守卫（Round 20 DEF-030 的同一形状）**。

**尚未确立、也正是默认保持关闭的原因**：回答「这有没有用」的对照实验**没跑完**。
agent 在被打断前刚发现它拿到的被动账本**每天换手一次**（中位换手 2.0），
那会让实验变成关于成本而不是关于回撤的。实验台
（`src/quantagent/rl/reward_ablation.py`，325 行）已抢救并入库，
但**实验本身未运行**，因此**不作任何有用性声明**。

---

## 本轮实测汇总

| 项 | 结果 |
|---|---|
| 后端全量 | **3546 passed / 4 skipped / 0 failed**（本轮起点 3461） |
| 前端 typecheck / vitest / build | 干净 / 119 passed / 5.33s |

## 流程事故与新规矩 / Incident

R9 报告了一起**跨 worktree 的 `git stash` 冲突**：`refs/stash` 是
**仓库级共享**的，与分支无关。R9 的 `git stash pop` 弹出了 **RL 角色的工作**，
而它自己的修复被丢掉。R9 把对方的 diff 存成 patch、原样退回 stash 并加了
`RETURNED-BY-R9:` 标记，然后重建了自己的修复。**没有丢失任何东西**，
但这是一次真实的近失事故。

**新规矩：worktree 隔离的 agent 一律禁止使用 `git stash`**，
需要临时保存改动就用文件拷贝。主角色在合并前应先 `git stash list`
并把每条 stash 备份成 patch 文件——本轮正是这么做的，两条 stash 都在确认
内容已进入 main 之后才丢弃。
