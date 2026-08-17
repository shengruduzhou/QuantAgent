# Round 21 — R2 风控专家审计报告 / Risk Audit

日期 / Date: 2026-08-18
基线 / Baseline: `b56ae57` (branch `main`)
角色 / Role: R2 风控专家（只读审计，未修改 `src/` 或 `apps/`）
Python: `AI_quant_venv/bin/python3`

> 裁决规则：证据缺失记 `unknown`，**永不记 `pass`**。每条 finding 带 `file:line` +
> 失败场景 + 复现命令。本文件按确认顺序**增量写入**。

---

## 0. 网页调研摘要 / External research summary

调研了 8 个来源（4 fetch + 4 search），提炼出机构级风控的参照系：

| 来源 | 关键结论 | 对本仓的含义 |
|---|---|---|
| [akquant 10_analysis](https://akquant.akfamily.xyz/textbook/10_analysis/) | 风险度量标准集 = MDD + VaR(95/99) + **CVaR/Expected Shortfall** + 下行标准差 + Sharpe/Sortino/**Calmar**；归因用 Brinson + 因子分解 | 用户要"最大年化 + 最小回撤" ⇒ 目标函数就是 **Calmar**，仓库必须至少能算 Calmar 与 CVaR |
| [akquant guide/analysis](https://akquant.akfamily.xyz/guide/analysis/) | 提供 `exposure_df()`（net/gross exposure、leverage）、`attribution_df()`、`capacity_df()`（成交率/换手/容量）、Kelly、Ulcer Index/UPI | **敞口分解与容量分析应是一等产物**，不是事后手算 |
| [akquant 15_live_trading](https://akquant.akfamily.xyz/textbook/15_live_trading/) | "**风控前置 (Pre-trade Risk Check) 是防止乌龙指的最后一道防线**"；四类约束 = 单笔上限 / 保证金占用 / 日内撤单数 / 策略级止损。**明确警告**：`broker_live` 模式下 max daily loss 与策略风险预算**不生效**，只在 paper 生效 ⇒ 必须在平台侧与券商侧**双层**部署 | 与本仓 **F-01** 完全同构，但本仓是**反过来**的（live 才生效、paper 不生效），风险更大 |
| [qmfquant modelEdit](https://qmfquant.com/static/doc/code/modelEdit.html) | 该页是 ML 教程，**没有**风控 API 规范（只提到"保守仓位控制和止损"）⇒ 记 `unknown`，不作为参照 | 不可引用 |
| Search: volatility targeting | 机制 = 按**上期已实现方差的倒数**缩放敞口；估计用 rolling std 或 **EWMA**；实现必须带 **leverage cap（150–200%）+ floor 0%（long-only）**；代价 = 换手与时变杠杆显著上升 | 本仓 long-only、`max_leverage=1.0` ⇒ vol targeting 只能**向下缩**（de-risking），这恰好是"最小回撤"的正确工具 |
| Search: CVaR / Expected Shortfall | CVaR **次可加**（VaR 不是）⇒ 更适合组合风险预算；Basel FRTB 已从 VaR 99% 迁移到 **ES 97.5%**；机构做法 = 设"风险限额"并按**成分对限额的贡献**做组合构建 | 若要上尾部风控，直接上 **CVaR 97.5%** 而非 VaR |
| Search: pre-trade risk / circuit breaker | 风控引擎**内联在 Execution Gateway**，位于 order submission 与 market execution **之间**；违规订单**在触达市场前即被拒绝**并记录拒绝原因回传；SEC 15c3-5 强制 fat-finger（max qty / max value / **price collar**） | 判定标准：风控必须在 **gateway 内联**，而不是"决策链上游的一个建议者" |
| Search: A股 组合风控/行业中性/beta 敞口 | 市场中性 = **Beta 中性**（组合 β→0）+ **行业中性**（行业敞口保持中性）两个分量缺一不可；敞口公式按 `mvi × βiA / CVA` 折算对冲手数 | 本仓有 `beta_exposure_limit` 与 `max_sector_weight` 定义，但接线情况见 §3 |

**参照系结论**：机构做法是"**pre-trade 内联网关 + 确定性拒绝 + 组合级限额（CVaR/beta/行业）+ 动态 sizing（vol target）**"四件套。下文逐条对照本仓实测。

---

## 1. 风控决策链的真实调用图 / Actual call graph

### 1.1 声明的链路（`AGENTS.md:24-26`）

```
Agents ──(evidence/views/constraints/risk flags, 无订单)──┐
                                                          ├─> Optimizer ──> target_weights
                                                          │      (只输出权重，无订单)
                                                          └─> OrderManager ──> order intents
                                                                  │
                                        [risk gate → kill switch → execution constraint
                                         → reconciliation → audit replay]  ← 声明的必经关卡
                                                                  │
                                                                  v
                                                              QMT submit
```

### 1.2 实测的链路（grep 证据）

存在**三条互不相通**的风控实现，且**生产路径走的那条最弱**：

```
路径 A（RiskGate，最完整：beta/行业/style/TE/换手/杠杆/涨跌停/ST/停牌/conformal）
  quantagent.risk.risk_gate.RiskGate.check_target_weights
    └── 唯一调用点: src/quantagent/execution/live_session.py:155
    └── check_order_intents 唯一调用点: src/quantagent/execution/live_session.py:192
        └── LiveTradingSession 的调用者 = ?
            services/quant_api/services/production_readiness.py:22   （只读 readiness 报告）
            tests/execution/test_live_*.py                            （4 个测试文件）
            ⇒ **零生产执行调用点**

路径 B（paper RiskEngine，第二完整：单票/行业/gross/日亏/回撤/参与率/fat-finger/scoped kill switch）
  quantagent.paper.risk.RiskEngine.check_order / check_portfolio / check_operational
    └── 全仓调用点: tests/paper/test_risk_engine.py  （**唯一** import 者）
        ⇒ **零生产调用点**（见 F-02）

路径 C（实际生产路径，最弱：只有 quantity>0 / limit_price>0 / 交易所手数）
  target_weights
    └─> OrderManager.submit  (src/quantagent/execution/order_manager.py:381)
          └─> OrderManager._pretrade_decision  (order_manager.py:493)
                ├─ upstream risk_check_result 若为 "rejected" 则拒 (:499)
                ├─ quantity <= 0 拒 (:513)
                ├─ LIMIT 且 price<=0 拒 (:526)
                ├─ _exchange_quantity_rejection 板块最小手数/增量/上限 (:540)
                └─ if not self._requires_production_pretrade():  (:560)
                        return approved=True   ← **在这里直接放行**
                   （ExecutionConstraintEvaluator 在 :607，位于该 return 之后）
          └─> broker.place_order
                ├── PaperBrokerAdapter -> PaperBroker      （paper 生产）
                └── QMTGateway                              （dry_run=True，LIVE_DISABLED）
```

生产 OrderManager 的实例化点（`grep OrderManager(`）：

- `services/quant_api/services/paper_orders.py:290` — HTTP `/api/paper/orders` 全链路
- `src/quantagent/paper/continuous_execution.py:1193` — 连续执行 loop
- `src/quantagent/backtest/ashare_execution_simulator_impl.py:143` — 回测
- `src/quantagent/reconciliation/composite.py:385` — 对账

以上 **4 个生产实例化点，全部不经过路径 A/B**。

### 1.3 结论

- 用户的指控"**风控被当成决策链中间的一个普通 Agent，没有独立于 Agent 体系的确定性断路器**"：
  **部分成立，且实际情况比指控更严重** —— 风控不是"被 Agent 绕过"，而是**最完整的两套风控引擎根本没接到生产执行路径上**（F-01/F-02）。
- 确定性方面**指控不成立**：`RiskGate`、`KillSwitch`、`RiskEngine`、`ExecutionConstraintEvaluator` 全部是**纯确定性 Python**，无 LLM 调用、无 agent 依赖（见 §2）。问题不是"被 LLM 覆盖"，是"没被调用"。

---

## 2. Kill switch / Circuit breaker 判定

### 2.1 确定性与不可覆盖性 — **成立（这部分是好的）**

| 断言 | 裁决 | 证据 |
|---|---|---|
| 纯确定性代码，无 LLM | **PASS** | `src/quantagent/risk/kill_switch.py` 全文无网络/LLM import；`src/quantagent/paper/risk.py` 同 |
| 不可被 Agent 覆盖 | **PASS** | `paper/risk.py:40` `RiskRejection` "deliberately has no override path"；`risk.py:420-427` `enforce()` 无 override 参数；`risk.py:103` `"override_available": False` |
| 清除需人工确认 | **PASS** | `paper/risk.py:131-140` `clear(human_confirmation=False)` 直接抛 `RiskRejection`；`risk/kill_switch.py:52-76` 持久化实例禁止裸 `release()`，全量清除需 `release_all(confirm=True)` |
| 状态损坏时 fail-closed | **PASS** | `risk/kill_switch.py:133-138` 状态文件不可解析 ⇒ `manual_triggered=True` + `kill_switch_state_unreadable`，**不会绿灯启动**。这是全仓风控里写得最好的一段 |
| 持久化原子写 | **PASS** | `risk/kill_switch.py:150-156` tmp + `fsync` + `os.replace` |
| 优先级最高 | **UNKNOWN → 实为 N/A** | `RiskGate.check_target_weights:193` 与 `check_order_intents:321` 确实**第一个**检查 kill switch；但这两个函数在生产路径上从不被调用（§1.2），故"优先级"无从生效 |

### 2.2 触发后已在途订单如何处理 — **UNKNOWN（无实现，见 F-04）**

全仓 grep 未找到任何"kill switch 触发 ⇒ 撤销在途订单"的代码路径：

- `risk/kill_switch.py` 只有布尔状态，无 `cancel_*` / `flatten` / `liquidate` 出口。
- `paper/risk.py:107-144` `KillSwitch` 同样只是 scoped 布尔字典。
- `paper/broker.py:261` `arm_kill_switch` / `:264` `trigger_kill_switch` — 需核实是否撤单（见 F-04）。
- `RiskGate` 两个检查都只是把 `kill_switch_triggered` 加进 `violations`，即**阻止新单**，对已提交未成交的订单无动作。

⇒ 语义是 **"stop-new-orders"**，不是机构意义上的 **"cancel-on-disconnect / flatten"** 断路器。

---
