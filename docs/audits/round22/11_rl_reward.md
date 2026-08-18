# Round 22 — R11 强化学习：风险感知奖励 / Risk-Aware RL Reward

- 角色 / Role: R11 — RL 奖励设计与实测
- 基线 / Baseline: `057f8cf` (main tip)
- 分支 / Branch: `agent/round22-rl-reward`
- 范围 / Scope: **仅** `src/quantagent/rl/**` 与 `tests/rl/**`
- 日期 / Date: 2026-08-19

## 0. 任务缺口 / The gap being closed

Round 21 (`docs/audits/round21/11_rl.md` §2.2, `90_main_integration.md` §2 R11) 记录：

> reward 逐 step 线性于收益差，**无方差项、无 Sharpe 项、无回撤项**。
> 已弃用的 `PortfolioEnv` 反而更丰富（`drawdown_lambda=2.0` + kill-switch 惩罚）。
> **在把奖励时钟修对的同时风险惩罚被一并丢掉了，且无任何文档记录该取舍。**

`AGENTS.md:97` 的验收闸门含「max drawdown 低于配置阈值」。
用户目标原话是「最高超额 + 最小回撤」。
**当前奖励只编码了目标的前半句** ⇒ 策略可以在奖励上赢、在闸门上被否。

---

## 1. 文献调研 / Literature review

执行 4 次 WebSearch + 2 次 WebFetch。

### 1.1 现有奖励其实是文献里的 "alpha reward"，这一点是对的

`CLaC @ FinMMEval 2026 Task 3` ([arXiv 2607.16028](https://arxiv.org/html/2607.16028v1)) 提出
`r_t = log(V_t/V_{t-1}) − log(c_t/c_{t-1})`，并证明三件事：

1. **被动持有恒得 0**（无成本时）—— 与本仓库「零动作 = 恰好 0」构造同源。
2. **逐 step 奖励在 episode 上望远镜求和 = log(终端 alpha)**，使训练信号与评测标准同构。
3. 它是一个 **variance-reducing control variate**：减掉市场项后，
   纯多头无成本情形下奖励方差**降到零**，agent 拿到的是「偏离被动」的干净信号
   而不是被大盘涨跌淹没的噪声。

⇒ **裁决：本仓库 `R = κ·[(⟨w,r⟩−c_p) − (⟨w_b,r⟩−c_b)]` 的相对化形式不该动。**
它已经是文献主流且优于绝对收益奖励。本轮只在其上**加**风险差分项。

### 1.2 Differential Sharpe Ratio (Moody & Saffell, NeurIPS 1998)

[papers.neurips.cc/paper/1551](http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)。
Sharpe 定义在整段 T 上、不能在线用，故构造 O(1) 增量更新一二阶矩的 DSR，
使 cumulative reward 逼近整段 Sharpe。

**为什么本轮不采用 DSR 作为主形式**：
- DSR 是**绝对** Sharpe 的增量，不是相对被动账本的差分。直接接上会**破坏「零动作 = 0」**
  （被动账本自身的波动会给零动作打出非零 DSR）—— 这是本 env 最重要的构造性保证，
  也是 Round 21 DEF-036 刚刚修好的东西，不能为了风险项再毁掉。
- DSR 依赖二阶矩的 EMA，是**非平稳且路径依赖**的量；`arXiv 2506.04358` 也指出
  风险调整型奖励应「减少对不稳定二阶统计量的敏感性」。
- 本仓库验收闸门写的是 **max drawdown**，不是 Sharpe。奖励要和闸门同构，就该直接编码 MDD。

DSR 记为 **未采用但已评估**，理由如上，不是没读过。

### 1.3 Risk-aware reward：显式惩罚下行与回撤是有效的

[arXiv 2506.04358](https://arxiv.org/html/2506.04358v1) 提出复合奖励
`ℛ = w₁·R_ann − w₂·σ_down + w₃·D_ret + w₄·T_ry`，其中
`σ_down = sqrt( (1/T)·Σ_t max(0, −R_t)² )` 是**只对负收益计的下行风险**（Sortino 口径）。
实测：NVIDIA 上 MaxDD 仅 8% 同时峰值收益 +42% vs +38%。
该文**不给固定超参数值**，而是在单纯形上网格搜索 —— 明确把权重当成设计选择。

⇒ 本轮采纳 `σ_down` 的**逐步贡献量** `max(0,−R)²` 作为波动惩罚的函数形式，
但同样做成**对被动账本的差分**。

其余综述（[MDPI 数学 14/8/1334](https://www.mdpi.com/2227-7390/14/8/1334)、
[MDPI JRFM 18/7/347](https://www.mdpi.com/1911-8074/18/7/347)）一致报告：
「variance penalization 改善风险调整后收益，但**过度**的 downside penalization 会适得其反」。
⇒ 这直接支持「λ 必须是可调、默认关闭、并由实测决定」的做法，而不是拍一个 2.0 上去。

### 1.4 Potential-based reward shaping：**判断结果 = 不适用，而且这个结论有用**

Ng, Harada & Russell (1999)「Policy invariance under reward transformations」：
若 `F(s,s') = γ·Φ(s') − Φ(s)`，则最优策略集合**不变**（policy invariance）。

**判断**：这正是本任务**不想要**的性质。
我们要的就是改变最优策略，让它在收益与回撤之间做取舍。
一个严格 policy-invariant 的惩罚项**在定义上不可能**让 agent 少冒风险 ——
它只能加速学习同一个最优解。

但这条定理**不是白读的**，它精确地告诉我们哪种写法会失效：

- 令 `Φ(s) = −λ·MDD(s)`（MDD = 状态里的 running max drawdown），
  则 `γ=1` 时 `F = −λ·(MDD_t − MDD_{t−1}) = −λ·ΔMDD_t`。
  **逐步回撤增量惩罚在代数形式上恰好是一个 potential-based shaping 项。**
- Ng 定理对 episodic 任务有一个前提：**`Φ(terminal) = 0`**。
  若强行满足它（例如在终止时补回 `+λ·MDD_T`），整个惩罚在 episode 上望远镜求和后
  **恰好抵消为 0**，惩罚变成装饰品 —— 一个「只会通过、不会失败」的惩罚项。
- 本轮**故意违反** `Φ(terminal)=0`：望远镜求和给出
  `Σ_t (−λ·ΔMDD_t) = −λ·MDD_T`，是一个**依赖策略**的量，所以它真的有牙。

⇒ **PBRS 在此不适用（applies-as-a-negative-result）**：它被用作判据，
排除了一个会静默失效的实现写法。这条写进设计取舍 §2.4。

**来源 / Sources:**
- [Moody & Saffell, RL for Trading (NeurIPS 1998)](http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
- [Ng, Harada & Russell 1999, policy invariance](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)
- [A Risk-Aware RL Reward for Financial Trading (arXiv 2506.04358)](https://arxiv.org/html/2506.04358v1)
- [CLaC @ FinMMEval 2026 Task 3, alpha reward (arXiv 2607.16028)](https://arxiv.org/html/2607.16028v1)
- [Risk-Sensitive RL for Portfolio Optimization (MDPI Mathematics 14/8/1334)](https://www.mdpi.com/2227-7390/14/8/1334)
- [Risk-Sensitive Deep RL for Portfolio Optimization (MDPI JRFM 18/7/347)](https://www.mdpi.com/1911-8074/18/7/347)

---

## 2. 实现 / What was implemented

改动只在 `src/quantagent/rl/pit_portfolio_env.py`（+106 行）与
新文件 `tests/rl/test_risk_aware_reward.py`。

### 2.1 奖励公式（新）

记 `NAV_t`、`NAV^b_t` 为 policy / passive 的账面净值（env 内已有），
`peak_t = max_{s≤t} NAV_s`，`DD_t = 1 − NAV_t/peak_t`，
`MDD_t = max_{s≤t} DD_s`（同理 `MDD^b`）。
`net = ⟨w,r⟩ − c` 为该 step 的成本后净收益。

```
R_t = κ · [  value_add_t
           − λ_dd  · ( ΔMDD_t   − ΔMDD^b_t )
           − λ_vol · ( min(net_t,0)²  − min(net^b_t,0)² )  ]

value_add_t = net_t − net^b_t
ΔMDD_t      = MDD_t − MDD_{t−1}   ( ≥ 0 )
κ = 100 (reward_scale, 不变)
λ_dd  = drawdown_lambda   默认 0.0
λ_vol = volatility_lambda 默认 0.0
```

### 2.2 望远镜性质 = 与验收闸门同构（这是本设计的核心）

`ΔMDD_t` 在 episode 上望远镜求和：`Σ_t ΔMDD_t = MDD_T − MDD_0 = MDD_T`。
所以整段 episode 的累计奖励是

```
Σ_t R_t = κ · [  Σ_t value_add_t              ← 「最高超额」
               − λ_dd · ( MDD_T − MDD^b_T )   ← 「最小回撤」
               − λ_vol · ( 超额下行半方差 )  ]
```

**这正是 `AGENTS.md:97` 的 max-drawdown 闸门在奖励里的镜像**，
也正是用户「最高超额 + 最小回撤」两句话的直接编码。
`λ_dd` 的单位是可解释的：**λ_dd = 1.0 意味着「1 个百分点的超额最大回撤，
抵消 1 个百分点的累计超额收益」**（Calmar 式的定价）。
测试 `test_drawdown_penalty_telescopes_to_excess_max_drawdown` 逐参数钉住这个恒等式。

### 2.3 三条硬约束的满足方式

| 约束 | 满足方式 | 钉住它的测试 |
|---|---|---|
| 零动作恰好得 0 | 两个风险项都是**对同一受约束被动账本的差分**。零动作下 `w` 与 `w_b` **逐比特相同**（`gross/weight_sum = passive_gross/passive_gross = 1.0` 精确）⇒ 两条 NAV、两条 peak、两条 MDD 路径全等 ⇒ 每一项恒等于 `0.0` | `test_zero_action_still_earns_exactly_zero_with_risk_terms_on`（5 gross × 4 λ 组合 = 20 例，`abs=1e-12`） |
| 默认行为不变 | `λ_dd = λ_vol = 0.0`，`risk_penalty` 恒为 `0.0`，`reward == value_add·100` 精确 | `test_default_config_reproduces_the_previous_reward_exactly` |
| 先读文献再定形式 | §1，4 search + 2 fetch | — |

另加一条**未被要求但必要**的守卫：`λ < 0` 或非有限值**直接 raise**。
负 λ 会**付钱让 agent 跑出比被动账本更深的回撤** —— 训练目标与验收闸门符号相反。
`test_negative_or_non_finite_lambda_is_rejected`。

### 2.4 设计取舍 / Trade-offs（含被否决的方案）

1. **为什么不是 Differential Sharpe（Moody & Saffell）** —— DSR 是**绝对** Sharpe 的
   增量，接上去会让被动账本自身的波动给零动作打出非零分，**破坏 DEF-036 刚修好的
   构造性保证**。见 §1.2。
2. **为什么不是 potential-based shaping** —— policy invariance 恰恰是我们**不要**的
   性质。见 §1.4。同时该定理被用作**判据**：它精确指出「补一个 `Φ(terminal)=0`
   的终止修正」会让惩罚在 episode 上抵消为 0，成为装饰性守卫
   （Round 20 DEF-030 的同一形状）。`test_drawdown_shaping_is_not_policy_invariant`
   把「它不是 policy-invariant」写成断言。
3. **为什么惩罚在 `κ` 内侧** —— `reward = (value_add − risk_penalty)·κ`，
   使 λ 的单位与 `value_add` 同为「收益」，与 `reward_scale` 解耦。
4. **回撤是「更浅」时会给正分（credit）** —— 差分形式是对称的。
   这是有意的：用户目标就是相对被动账本降低回撤。副作用是 agent 可以靠
   `a[n] = −1` 降 gross（最多 −30%）来买回撤，**这正是需要被实测定价的取舍**，
   不是能靠拍一个 λ 解决的。`test_drawdown_penalty_credits_the_shallower_path`。
5. **诚实缺口**：env 的 `NAV` 是 `net = ⟨w,r⟩ − 12bps·turnover` 在连续
   T+1→T+2 区间上的连乘，**不含滑点、整手、现金约束**。
   因此 **env MDD ≠ strict simulator MDD**，闸门用的是后者。
   本轮的风险项优化的是**前者的代理**。这是已知的、未闭合的口径差。
