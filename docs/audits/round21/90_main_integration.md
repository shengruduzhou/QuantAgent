# Round 21 — 主角色整合裁决 / Main-Role Integration

- 日期 / Date: 2026-08-18
- 基线 / Baseline: `b56ae57` → 交付分支 `agent/round21-multirole-audit`
- 角色 / Role: Main（量化/架构/全栈总负责，唯一有写权限的角色）

---

## 0. 本轮组织实况 / What actually happened

按章程启动了 8 个隔离角色（R1 回测、R2 风控、R3 因子、R4 选股、R5 测试、
R8 UI、R9 debug、R11 RL）。**8 个全部被 session limit 中断**（与 Round 20 同样的
失败模式）。章程里"每确认一条 finding 就立刻落盘"的纪律**救回了 6 份报告共 964 行**：

| 角色 | 报告 | 状态 |
|---|---|---|
| R1 回测 | `01_backtest.md` 266 行 | 5 条 findings 完整，含独立复现 |
| R2 风控 | `02_risk.md` 125 行 | 调用图 + kill-switch 裁决完整，findings 部分 |
| R3 因子 | `03_factor.md` 252 行 | 因子清单 + 非线性裁决完整 |
| R4 选股 | `04_selection.md` 33 行 | 仅调研摘要 |
| R5 测试 | `05_test.md` 144 行 | 环境/导入/依赖完整，全量 pytest 未回填 |
| R11 RL | `11_rl.md` 144 行 | reward 公式 + 可交易性裁决完整 |
| R8 UI / R9 debug | — | **零产出**，由主角色补做（见 §3、§4） |

**流程教训（写给下一轮）**：并发 8 个重型 role agent 会在本账号的 session
预算内被打死。下一轮应**分批 3–4 个**，且必须保留增量落盘纪律 —— 它是本轮
唯一让审计成果不归零的原因。

---

## 1. 已修复并合并 / Fixed in this round

### DEF-033【P0】不可测量的风控约束被当成通过（fail-open）

来源：R2。`ExecutionConstraintEvaluator` 对两条**组合级**限额的处理是静默跳过：

- `constraints.py` 参与率检查：`if not dvol: continue`
- `constraints.py` 换手检查：`if navs:`（空则整段不执行）

而生产下单路径**恰好两个输入都不提供**：`OrderManager` 把
`daily_volume_hint=None` 写死，并把 `query_account_value()` 的任何异常
吞成 `nav = None`。于是 `max_single_stock_participation_rate=0.10` 与
`max_daily_turnover=2.0` 两条限额**在生产路径上结构性不可达**，而报告仍返回
`passed=True`。

**这是本仓反复出现的缺陷形状的又一实例**：缺失的测量被合理默认值静默替换，
且因为内部一致性检查照样通过而长期存活。

修法：新增 `UnmeasuredConstraint` 三态记录 + `report.fully_measured`；
生产 pre-trade 改为要求 `passed and fully_measured`，缺测量即**拒单**；
`OrderManager` 增加 `daily_volume_hints` 与可选 `broker.query_daily_volume`
钩子，使限额真的能触发。新增 8 个测试。

### DEF-034【P1】平方根冲击成本被发布进信任证书但从未收取

来源：R1 F-02。`AShareCostModel` 的 docstring 声明平方根冲击模型，
`trusted_cost_model_config()` 把 `impact_alpha_bps=10.0` 原样写进信任证书，
但全仓无任何调用点传 `participation_rate` ⇒ `impact_cost` 恒为 `0.0`。
**证书声称收了一项从未收取的成本。**

修法：`VirtualBroker` 本来就持有当日成交量，现按 `fill_quantity / day_volume`
算出参与率并收费；`TradeFill.impact_cost`（默认 0.0）把它带进账本现金流。
实测：5% 参与率的买单现在被收 **2.236 bps = 10 × √0.05** ✓ 与声明的模型一致。

> **副作用（必须承认）**：成本上升 ⇒ 任何用 `VirtualBroker` 产生的历史
> 净值/Sharpe 与本次修复后不可比。这与 §2 的"数字作废"结论方向一致。

### DEF-035【P0】`quantagent.clean_room` 整包不可导入

来源：R5。包里有 `__init__.py` / `dataset.py` / `risk.py`，但 `__init__` 导入
`engine.py` —— **该文件从未被提交**，三个模块全部 `ModuleNotFoundError`。
全仓 `grep clean_room` 只有 3 行、全在它自己的 `__init__.py` 里 ⇒ 无测试、
无 CLI、无脚本引用，所以 pytest 全量跑碰不到它，`compileall` 也发现不了
（语法没错，是缺文件）。

严重性在于**它是什么**：这个包的 docstring 明确写着它是为了逃离
`backtest/engine.py` 的 interior-bar NAV 缺陷而建的干净房间回测。
也就是说**本轮最关键的历史遗留项，其替代实现处于"提交了 2/3、整包 import
即崩、无人引用"的状态**。

修法：实现 `engine.py`，把 NAV 时钟钉死为
**`NAV(t) = cash(t) + 当日交易后持股 × close(t)`**；`close(T)` 形成的账本在
`close(T+1)` 成交并**首次**出现在 `NAV(T+1)`。成交价与估值价是同一个观测，
因此隔夜跳空无法被记成瞬时损益。持有但无法定价的标的令 NAV = `None`
（不是静默按零计价）；指标算不出时返回 `None`（不是 0.0）。新增 10 个测试。

### DEF-036【P1】RL 环境在零动作下自行加杠杆并把差额计为 agent 的 value-add

来源：R11 F-RL-03。奖励是对被动账本的差分
`R = κ·[(⟨w,r⟩−c_policy) − (⟨w_b,r⟩−c_passive)]`，零动作应恰好得 0 ——
这是免疫 env-flat 陷阱的构造性保证。但 gross 被 clip 进固定的
`[min_gross, max_gross]`，**当被动账本自身 gross < `min_gross`（0.5）时**，
零动作产出 `w = passive·(0.5/passive_gross) ≠ passive`，环境替 agent 做了一次
加杠杆决策并把结果计入 value-add。低 gross 账本正是"已减仓/部分现金"的常态，
恰恰是 value-add 度量最需要诚实的场景。

修法：gross 下限改为 `min(cfg.min_gross, passive_gross)` —— 只约束 tilt，
不抬高被动基线。零动作时 tilted gross 恰为 `passive_gross`，落在区间内。
新增 7 个测试，并**用 `git stash` 对照验证**：对旧代码 3 failed / 4 passed，
对新代码 7 passed。

### DEF-037【P1】工作站没有任何路径可以问"非线性混合有没有增量"

来源：R3 FIND-R3-03。`quantagent.fusion` 的全部 7 个方案返回的都是
**权重向量**，打分是 `score = Σ wᵢ·rank(xᵢ)` —— 一个矩阵-向量点积。
`genetic` 是对加性模型的非线性**优化器**，不是非线性**模型**。
真正的交互构造在 `models/interactions.py`（质量很高，配对选择先把父因子
投影掉再对残差乘积算 IC，只有交互本身携带新信息时才非零），但它的唯一
生产侧消费者是 `research/model_comparison.py`，而
**`audit-nonlinear-factors` 不在 `jobs.py` 的 47 个受治理命令白名单里**。

修法：把 `audit-nonlinear-factors` 登记进白名单（不暴露 `n_trials`，与
fusion 同一铁律）。新增 5 个测试，其中一条**逐参数比对白名单与 CLI 签名**，
防止未来漂移。

> **对 R3 的一处更正**：R3 建议的"诚实性下限（c）—— 融合产出必须显式声明
> 搜过的模型类"**已经实现**。`fusion/search.py:172-173` 已发布
> `"modelClass": "rank_weighted_additive"` 与 `"representsInteraction": False`。
> 已加测试钉住它。

---

## 2. 已确认但**未**在本轮修复 / Confirmed, deliberately not fixed

### R1 F-01【P0】interior-bar NAV 时钟错位 —— Sharpe 符号翻转

R1 独立复现：一条除一根跳空 bar 外完全平坦的 tape，真实收益 −0.046%，
引擎报 **+11.05%**；`sharpe` 从诚实的 **−7.0993** 翻成 **+7.0993**，
`max_drawdown` 报 **0.0**。亏损策略被报成完美策略。

**为什么本轮不修**：`docs/interior_bar_nav_defect.md` 记录了上一次修复尝试
因打破 `tests/domain/test_composite_replay.py` 的 ledger-replay 不变量而被
回滚。正确的修法必须**先在 `domain/ledger.py` 与 composite replay 路径钉死
约定**（`NAV(t)` 是否包含由 `t−1` 信号调度、在 `t` 执行的成交），两种约定都
可辩护，但引擎与 replay 必须一致。在没有把这个约定写成可执行契约之前动手，
只会重演一次回滚。**不得为了让修复通过而削弱 composite replay 测试。**

本轮的替代交付：`clean_room/engine.py` 现在提供一个**时钟正确的参照实现**
（DEF-035），下一轮可以用它对同一份 target_weights 做差分回归，把错位的
经济后果量化成一个具体数字，再据此选定约定。

同时 R1 指出黄金场景**在构造上看不见这个缺陷**：
`tests/test_golden_backtest_scenarios.py:268` 让每根 bar 都 `open == close`
——正是错位消失的唯一情形。**它们通过不构成证据。**

### R1 F-03【P1】涨跌停判据按 close，成交按 open

判据与撮合价不同源，两个方向都会错，其中"假成交"（一字涨停开盘价买到货）
是**有利方向**的偏差且无法从 reject 日志看出。一字板在代码里没有任何专门判据。

### R1 F-04/F-05【P1】声明滑点 5 bps vs 实际 2 bps；快引擎冲击是线性且被压成近零

`BacktestConfig.cost.slippage_bps = 5.0` 仍被序列化进配置，引擎实际施加的是
`FillModelConfig.slippage_bps = 2.0`。任何读配置声明"本次回测滑点 5 bps"的
报告都是错的（偏乐观）。且 `fill_model` 的冲击是线性、系数 1.0 bps，在
参与率封顶 0.05 下 ≤ 0.05 bps —— **实质为零，容量效应在快引擎里不存在**。

### R2【P0】两套最完整的风控引擎零生产调用点

`RiskGate`（beta/行业/style/TE/换手/杠杆/涨跌停/ST/停牌/conformal）唯一调用点
在 `live_session.py`，而 `LiveTradingSession` 的调用者只有一个只读 readiness
报告和 4 个测试文件。`paper.RiskEngine`（单票/行业/gross/日亏/回撤/参与率/
fat-finger/scoped kill switch）的**唯一 import 者是它自己的测试**。
4 个生产 `OrderManager` 实例化点全部不经过这两条路径。

**对用户指控的裁决**：
- "风控被当成决策链中间的一个普通 Agent、可被绕过" —— **部分成立，且实情更严重**：
  不是被绕过，是**最完整的两套风控根本没接到生产执行路径上**。
- "风控被 LLM 覆盖" —— **不成立**。`RiskGate`、`KillSwitch`、`RiskEngine`、
  `ExecutionConstraintEvaluator` 全是纯确定性 Python，无 LLM、无 agent 依赖。
  问题是"没被调用"，不是"被覆盖"。
- kill switch 本身质量高：状态文件不可解析时 fail-closed（`manual_triggered=True`），
  原子写 + fsync，清除需人工确认且无 override 路径。**但语义是 "stop-new-orders"，
  不是机构意义上的 cancel-on-disconnect / flatten** —— 触发后对已在途订单无动作。

### R3 FIND-R3-01【P0】认证宇宙与丰富因子集不相交

| 数据集 | 行数 | 特征数 | alpha101 | gtja191 |
|---|---|---|---|---|
| `runtime/data/gold/full_universe/dataset.parquet`（**唯一持 `FULL_UNIVERSE_GOLD_READY` 证书**，5790 符号） | 10,917,401 | 41（15 个特征） | **0** | **0** |
| `training_dataset_alpha181_exec_v89_plus7clean_fund.parquet` | 6,781,038 | 348 | 156 | 58 |

认证面板的 15 个特征里 3 个是均线 TA，其余是收益/波动/换手/Amihud 基础统计量，
**基本面 0、事件 0、另类 0**。

**对"系统只有传统 TA、评分 1.0/5.0"的裁决**：字面指控**不成立**——
`default_registry` 里 TA 只占 5/195 = 2.6%，主体是 WorldQuant Alpha101 +
CICC A股 80 + 国君 GTJA-191，这是统计套利/截面 alpha 血统。348 列面板里
基本面 21 列、宏观 22 列、北向/融资 4 列、LLM 发现因子 9 列**实测存在**。
**但用户的直觉在他实际看到的产物上是对的**：唯一持证书的面板确实只有 15 个
基础特征。诚实评分建议：量价 4/5、基本面 3/5、宏观 3.5/5、
**事件预期 0.5/5**、**认证产物可用因子广度 1.5/5**。

事件/预期类因子 **= 0**（两个数据集皆是）：`earnings_revision_score` 与
`sector_rotation_score` 出现在 `v7_feature_groups.py:51-52` 的声明里，
但没有任何 builder 会生成它们。

Level-2 / 逐笔订单流 = **BLOCKED_BY_DATA**（A 股 L2 零公开供应商，腾讯"分笔"
实为 3 秒聚合）。这是数据约束不是实现缺陷，记 BLOCKED 而非 FAIL。

### R11【P1】RL 奖励里没有任何风险项

reward 逐 step 线性于收益差，**无方差项、无 Sharpe 项、无回撤项**。
已弃用的 `PortfolioEnv` 反而更丰富（有 `drawdown_lambda=2.0` 与 kill-switch
惩罚）。**在把奖励时钟修对的同时风险惩罚被一并丢掉了，且无任何文档记录该取舍。**
后果：奖励与 `AGENTS.md` 的验收闸门**不同构** —— 策略可以在奖励上赢、
在 max drawdown 闸门上被否。

### R5【P1】`[all]` extra 名不副实

`lightgbm` / `xgboost` / `gymnasium` / `stable_baselines3` / `optuna` / `deap` /
`catboost` / `matplotlib` 等 12 个包不在 `pyproject.toml` 任何 extra 里，
只在函数体或 `try:` 内导入。其中 `lightgbm` 尤其值得记：`AGENTS.md:68` 把
`train-alpha-v7 --model lightgbm` 列为"real LightGBM, fail-loud if missing"的
真实数据命令，但它不在任何 extra 中 ⇒ 标准安装下该命令必然 fail-loud。
fail-loud 本身正确，问题是**声明与文档不一致**。

---

## 3. 主角色补做：AkShare 数据真相 / AkShare data truth (measured)

用户反复反馈"akshare 的数据还是不太对"。本轮做了实测取证。

### 3.1 单位与口径契约：**实测通过**

用 2026-08-03→08-14 的 600519 交叉验证 Sina 与腾讯：

| 检验 | 实测 |
|---|---|
| 收盘价一致性 | 10/10 日 `close_diff = 0.0`（逐位相同） |
| volume 比值（Sina ÷ 腾讯） | 均值 **100.000022** ⇒ 腾讯是**手**、Sina 是**股**，仓库 `LOTS_TO_SHARES=100` 正确 |
| amount 口径 | 隐含 VWAP = amount/volume **10/10 日落在 [low, high] 内** ⇒ amount 确为 CNY |
| 腾讯日线无 amount 列 | 已按每次响应探测 payload 形状（`sources.py:184`），DEF-032 修法在位 |
| 腾讯 quote f37 | 实测 = 万元，`sources.py:272` 的 `× 10000` 正确 |

⇒ **单位契约这一层不是"数据不对"的原因。**

### 3.2 真正的原因：东方财富端点从本机 100% 不可达

实测 10 个常用 AkShare 接口：

| 接口 | 结果 |
|---|---|
| `stock_zh_a_hist`（东财日线） | **FAIL** `RemoteDisconnected` |
| `stock_zh_a_spot_em`（东财全市场快照） | **FAIL** `RemoteDisconnected` |
| `stock_info_a_code_name`（股票列表） | **FAIL** `ConnectionReset`（32s） |
| `stock_zh_a_hist_min_em`（东财分钟） | **FAIL** `RemoteDisconnected` |
| `stock_individual_info_em` | **FAIL** `JSONDecodeError`（返回非 JSON） |
| `stock_board_industry_name_em` | **FAIL** `RemoteDisconnected` |
| `stock_zh_a_daily`（新浪日线） | **OK** 10 行，含 amount |
| `stock_financial_abstract`（财务摘要） | **OK** 80 行 |
| `stock_zh_index_daily`（指数，新浪） | **OK** 5972 行 |
| `tool_trade_date_hist_sina`（交易日历） | **OK** 8797 行 |

直接绕过 akshare 打 `push2his.eastmoney.com` / `push2.eastmoney.com`：
三种 header 组合（无头、仅 UA、UA+Referer）全部 `RemoteDisconnected`；
`push2.eastmoney.com` 返回 **nginx 502**。⇒ 不是 akshare 的问题，
是**东财侧从本机不可达**（限流/封禁/上游故障）。

**结论**：`*_em` 系列（东财）在本机全线不可用，Sina / 同花顺系列可用。
凡是依赖 `*_em` 的路径都会失败或退化；仓库对此的处理必须 fail-closed
并显式记录 `BLOCKED_BY_NETWORK`，不得静默降级。仓库共有 12 处 `ak.*_em(` 调用。

**下一轮待办**：给 `run_akshare_source_smoke.py` 加一条"端点族可达性"矩阵产物，
把这张表变成每日自动记录的证据，而不是每次靠人肉复现。

---

## 4. 主角色补做：前后端实测 / Full-stack measured

R8 零产出，主角色实跑：

| 检查 | 命令 | 结果 |
|---|---|---|
| TypeScript | `npm run typecheck` | **exit 0，零错误** |
| 前端测试 | `npm test`（vitest 4.1.9） | **100 passed / 1 skipped，26 文件，8.13s** |
| 前端构建 | `npm run build:vite` | **成功 5.40s**；`EChart` chunk 651.57 kB（gzip 219.97 kB）超 500 kB 警告 |
| 后端 API 启动 | `uvicorn services.quant_api.app:create_app --factory` | **启动成功** |

12 个只读端点实测（含返回体前 110 字节）：

| 端点 | 码 | 关键观察 |
|---|---|---|
| `/api/system/overview` | 200 | 4516B |
| `/api/risk/overview` | 200 | **`maxDrawdown: null`** —— 缺失即缺失，未渲染成 0 ✓ |
| `/api/risk/rules` | 200 | 规则带中文描述 |
| `/api/backtests` | 200 | 47.9 kB |
| `/api/data/quarantine` | 200 | **`status:"empty"`** 三态 ✓ |
| `/api/strategies` | 200 | 带 version |
| `/api/paper/account` | 200 | cash 1,000,000 |
| `/api/jobs` | 200 | 66 kB，含 `status:"rejected"`（研究否决，非工程故障）✓ |
| `/api/factors` | 200 | **209 kB** |
| `/api/market/stocks/600519.SH/overview` | 200 | **`status:"unavailable"` + `market_provider_unavailable`** ✓ fail-closed，未编造 |
| `/api/data/coverage` | 422 | 缺必填 query `path`（契约如此，非缺陷） |
| `/api/governance/gates` | 404 | 该路径不存在（主角色猜测的路径，非缺陷） |

⇒ **"产物缺失即缺失、不得渲染成 0 或 pass"这条铁律在 API 层实测成立。**
前端 chunk 体积是唯一的工程债（`EChart` 651 kB 未做 manualChunks 拆分）。

---

## 5. 数字作废清单 / Numbers that must not be quoted

以下数字在本轮之后**不得再被引用**，直到用修正后的引擎重跑：

1. 任何来自 `backtest/engine.py` 的 Sharpe / 年化 / 最大回撤 —— interior-bar
   NAV 错位未修（R1 F-01），Sharpe 可能符号翻转、回撤可能报 0。
2. 任何来自 `VirtualBroker` 的净值 —— 冲击成本从本轮起才真的收（DEF-034），
   与历史不可比。
3. 任何声明"滑点 5 bps"的回测报告 —— 实际施加 2 bps（R1 F-04）。
4. 任何基于快引擎的**容量**结论 —— 冲击实质为零（R1 F-05）。

---

## 6. 下一轮优先级 / Next-round priority

1. **P0** 钉死 NAV 约定（`domain/ledger.py` + composite replay），再修 F-01；
   用 `clean_room/engine.py` 做差分回归量化错位的经济后果。
2. **P0** 把 `RiskGate` 或 `paper.RiskEngine` 接到 4 个生产 `OrderManager`
   实例化点中至少 paper 那条；否则组合级风控在生产上等于不存在。
3. **P0** 让认证全宇宙面板带上基本面/事件因子，或明确降级它的证书措辞 ——
   现状下用户"只有 TA"的直觉在他能看到的产物上是对的。
4. **P1** RL 奖励补回风险项（至少回撤惩罚），使其与验收闸门同构。
5. **P1** 涨跌停判据与撮合价同源 + 一字板独立判据。
6. **P1** kill switch 从 stop-new-orders 升级到 flatten/cancel-on-trigger。
7. **P2** AkShare 端点族可达性矩阵进每日产物。
8. **P2** 前端 `EChart` chunk 拆分。
