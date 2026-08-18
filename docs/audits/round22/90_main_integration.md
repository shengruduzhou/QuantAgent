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
