# Round 21 — R1 回测专家审计报告 / Backtest Expert Audit

- 日期 / Date: 2026-08-18
- 基线 commit: `b56ae57`
- 角色 / Role: R1 回测专家（只读审计，不修改 `src/`、`apps/`，不 commit）
- Python: `AI_quant_venv/bin/python3`

> 纪律：证据缺失记 `unknown`，永不记 `pass`。所有 finding 带 `file:line` + 失败场景 + 最小复现命令。

---

## 0. 上游调研摘要 / Upstream research digest

调研对象：AKQuant textbook/guide/meta（7 个页面全文读取）+ 2 个 WebSearch 主题。

### 0.1 AKQuant `textbook/04_backtest_engine/`
- **事件驱动**（Rust event loop），不是向量化。口号：`事件驱动回测 = 时间推进 + 状态更新 + 订单撮合 + 风控拦截`。
- 时钟：`T` bar 触发 `on_bar` → 同 bar 提交订单 → **`T+1` bar 成交**，默认 `FillPolicy = NextOpen()`。
  其余可选 `CurrentClose()` / `NextClose()` / `NextAverage()` / `NextHighLowMid()`。
- 费用四层优先级：order-level > strategy-level > run-level > market default。
- 滑点公式：`Final = Exec × (1 ± slippage)`，买加卖减，**只作用一次**。
- T+1：`available_positions` 当日买入为 0，次日释放（`t_plus_one=True`）。
- 整手：100 股倍数，违反直接拒单。
- 成交量约束：`volume_limit_pct`（如 10% bar 量）。
- 冲击成本：平方根律 `Cost ∝ σ × √(Q/V)` 记为设计参考，实际用 `slippage + volume_limit_pct` 近似。

### 0.2 AKQuant `meta/internals/`
- 订单生命周期：Signal → Creation → Submission → Risk Check → Matching → Settlement → Reporting。
- `ExecutionPolicyCore` 三元组：`price_basis`(open/close/ohlc4/hl2) × `bar_offset`(0/1) × `temporal`。
- 限价穿透：买单需 `Bar.Low ≤ limit`，卖单需 `Bar.High ≥ limit`；有 price improvement。
- **滑点与佣金各只在 settlement 施加一次**：`Net PnL = Gross − Commission − Slippage`。
- A 股费用在 `src/market/china.rs`：印花税 + 过户费 + 佣金三项分开。
- 资金不足触发 **Partial Fill 而非 reject**。

### 0.3 AKQuant `textbook/appendix_pitfalls/`
逐条：未来函数（`shift(1)` / 事件驱动天然免疫）、幸存者偏差（需含退市）、复权（qfq）、
T+1 混淆（查 `available_position`）、涨跌停不可成交（`can_buy`/`can_sell` 数据旗标）、
漏成本、过度简化撮合、参数过拟合（找 plateau 不找 peak）、data snooping（DSR 惩罚）、
单条 walk-forward 路径不足以证明（要 CPCV 分布）、随机 K-Fold 泄漏（要 Purged K-Fold + Embargo）、
样本过短（MinTRL）。

### 0.4 AKQuant `textbook/11_optimization/` + `guide/optimization/`
- WFA 是 gold standard；`run_walk_forward(train_period=250, test_period=60, metric=...)`，
  `test_period` **同时是测试窗长度和滚动步长** ⇒ 上游默认是 **rolling 非 anchored**。
- 100 组参数在纯噪声上测 ⇒ ~99.4% 概率找到 95% 显著性（data snooping）。
- OOS Sharpe 比 IS 低 50%+ 即怀疑过拟合。
- CPCV：N 组、C(N,k) 组合、purge + embargo、产出 C(N,k) 条净值 ⇒ Sharpe 概率分布。
- DSR：`E[max SR]_K ≈ E[SR] + σ_SR √(2 ln K)`；**试验次数越多，门槛越高**。
- MinTRL ∝ 1/SR²：SR=0.5 需要 ~50 年；SR=2.0 只需 2–3 年。

### 0.5 AKQuant `guide/analysis/`
- 年化收益 = `(1+总收益)^(1/年数) − 1`；波动率 = `std × √252`；
  Sharpe = `(年化收益 − rf)/年化波动`；Calmar = `年化收益 / |MaxDD|`。
- **上游未记录 turnover 与净成本收益的口径** ⇒ 本仓自己的口径需要独立验证。

### 0.6 AKQuant `guide/testing/`
- 黄金场景（golden tests）+ 基线锁定（equity curve / 成交记录 / 指标逐项比对）；
  具体场景 `stock_t1`（Day1 买 Day2 才可卖）、`futures_margin`、`option_basic`。
- 结论：**任何算法变更都不得意外改变回测结果** —— 与本仓 Round 8/17 的"逐字节钉住"同源。

### 0.7 WebSearch: CPCV 实现陷阱
关键点：purging 的 off-by-one 会**悄悄**重新引入泄漏；train/test 索引处理错误 ⇒ lookahead；
高阶矩（skew/kurtosis）算错 ⇒ PSR 失效。**最危险的是 silent failure：代码不崩、出数、但数是错的。**
（[mlfinlab CPCV](https://www.mlfinlab.com/en/latest/cross_validation/cpcv.html)、
[quantinsti](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)、
[skfolio](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)）

### 0.8 WebSearch: DSR / PBO
- PBO 随试验数 N 增大而趋近 1，**与单个配置是否真有 alpha 无关**；PBO 衡量的是"选择过程"是否
  倾向于挑出 OOS 跑输中位数的配置。
- 高 PBO（如 0.93）不必然等于"策略经济上空洞"，也可能是"plateau 内部无法从 72 季度样本
  分辨出哪个配置最优" ⇒ **PBO 的解读必须区分这两种情形**。
- CPCV 相比传统方法给出更低 PBO 与更好 DSR。
（[Bailey/López de Prado DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)、
[PBO 原文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)、[pypbo](https://github.com/esvhd/pypbo)）

### 0.9 WebSearch: anchored vs rolling
- anchored（扩张窗）用满历史、偏好长记忆；rolling（固定窗）更适应 regime。
- **两者无绝对优劣，但选择必须被记录和辩护，不能被优化**。
- 若 rolling 胜 anchored ⇒ regime 漂移；若 anchored 胜 rolling ⇒ 样本不足。

---

## 1. 逐条裁决表 / Verdict table

（随审计进度增量填充，见下方 findings）

---

## 2. Findings

### F-01 【P0】回测引擎的 NAV 时钟提前一天 —— Sharpe 符号被翻转

- `src/quantagent/backtest/engine.py:164`（`fill_date = dates[i + 1]`）
- `src/quantagent/backtest/engine.py:308-313`（`equity = cash + Σ shares × close(dates[i])`）

**缺陷**：信号在 `dates[i]` 产生，成交发生在 `dates[i+1]` 的 `open`，但成交后的
**账面被盖章在 `dates[i]` 的 close 上**。于是 `NAV(t)` 里含有一个在 `t+1` 才买入的仓位，
成本价是 `open(t+1)`，估值价是 `close(t)`。每次调仓都把整段隔夜跳空
`Δshares × (close(t) − open(t+1))` 当成瞬时损益记入 `t`。
方向是系统性的：**隔夜跳空向下的标的会立刻"赚钱"**。

**实测失败场景**（我独立复现，非引用既有文档）：
一条除一根跳空 bar 外完全平坦的 tape（所有 `close = 10.00`，仅 `open[2] = 9.00`），
持有权重恒为 1.0。真实策略收益 = 只有费用，应当是 −0.046%。

| 指标 | 引擎输出 | 诚实值（同 tape，`open == close`） |
|---|---|---|
| NAV 首日 | 1,110,540（+11.05%） | 999,540（−0.046%） |
| annualized_return | **+8074.16** | −0.0191 |
| **sharpe** | **+7.0993** | **−7.0993** |
| max_drawdown | **0.0** | −0.00046 |
| volatility | 0.7848 | 0.00326 |

**Sharpe 恰好符号翻转，且回撤被报成 0。** 亏损策略被报成完美的赚钱策略。

**最小复现命令**：
```bash
AI_quant_venv/bin/python3 - <<'PY'
import sys; sys.path.insert(0,'src')
import pandas as pd
from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester
D = pd.bdate_range('2024-01-01', periods=6)
opens  = [10.,10.,9.,10.,10.,10.]     # 唯一一根跳空 bar
closes = [10.]*6                       # tape 完全平坦
p = pd.DataFrame([{'trade_date':d,'symbol':'600000.SH','open':o,'high':max(o,c)*1.5,
                   'low':min(o,c)*0.5,'close':c,'pre_close':c,'volume':1e12,'amount':1e12}
                  for d,o,c in zip(D,opens,closes)])
tw = pd.DataFrame({'600000.SH':[0.,1.,1.,1.,1.,1.]}, index=D)
r = EventDrivenBacktester(BacktestConfig()).run(tw, p)
print(r.nav_curve.to_string()); print(r.report['sharpe'], r.report['max_drawdown'])
PY
```

**状态**：仓库已在 `docs/interior_bar_nav_defect.md` 记录为 **OPEN**（`6327294`），
终端 bar 的同类问题已修（`6027705`），**内部 bar 未修**；一次修复尝试因打破
`tests/domain/test_composite_replay.py` 的 ledger-replay 不变量而被回滚。
我的复现独立证实了该文档，并给出更强的量化：**Sharpe 是符号翻转而非仅仅偏高**。

**建议修法**：先在 `src/quantagent/domain/ledger.py` 与 composite replay 路径
**钉死约定**——`NAV(t)` 是否包含在 `t` 执行的成交（该成交由 `t−1` 的信号调度）。
两种约定都可辩护，但引擎与 replay 必须一致。约定钉死后再把
mark-to-market 移到交易块之前。**不得**为了让修复通过而削弱 composite replay 测试。

**注意**：黄金场景**在构造上看不见这个缺陷** —— `tests/test_golden_backtest_scenarios.py:179`
只断言 `trades.iloc[0]`，`:268` 让每根 bar 都 `open == close`（正是错位消失的唯一情形）。
它们通过不构成证据。

---

### F-02 【P1】平方根冲击成本模型存在、被写进信任证书、但从未被调用

- `src/quantagent/execution/cost_model.py:38-66`（`calculate(..., participation_rate=0.0)`；`_impact_cost` 在 `p<=0` 时 `return 0.0`）
- `src/quantagent/execution/virtual_broker.py:45` — `self.cost_model.calculate(order.side, fill.quantity, fill.price)` **不传 `participation_rate`**
- `src/quantagent/backtest/strict_v8.py:164` — 同样不传
- `src/quantagent/backtest/paper_report.py:131` — 同样不传
- `src/quantagent/portfolio/do_t_overlay.py:190-191` — 同样不传
- `src/quantagent/execution/trusted_backtest_semantics.py:31-32` — `trusted_cost_model_config()` = `asdict(AShareCostModel())` ⇒ 向信任证书**发布 `impact_alpha_bps: 10.0`**

**缺陷**：`AShareCostModel` 的 docstring 明确声明"square-root market impact model"，
`trusted_cost_model_config()` 把 `impact_alpha_bps=10.0` 原样写进 trust certificate，
但**全仓库没有任何一处**用非零 `participation_rate` 调用 `calculate`。
`impact_cost` 恒为 `0.0`。信任证书声称计入了一项实际从未收取的成本。

这正是本仓反复出现的形状：**声明与执行分离，且声明一侧自审通过**。

**最小复现命令**：
```bash
grep -rn "cost_model.calculate\|cm.calculate" --include=*.py src/    # 5 处，全部不传 participation_rate
AI_quant_venv/bin/python3 -c "
import sys; sys.path.insert(0,'src')
from quantagent.execution.cost_model import AShareCostModel
from quantagent.execution.broker_base import OrderSide
print(AShareCostModel().calculate(OrderSide.BUY, 100000, 10.0))"
# -> impact_cost: 0.0，尽管 impact_alpha_bps=10.0
```

**建议修法**：`VirtualBroker.submit` 已经从 `_available_volume(symbol)` 拿到了当日
volume，把 `participation_rate = fill.quantity / volume` 传下去即可（成本会真的上升，
这本身就是必须重新回测的理由之一）。或者，若决定不启用冲击模型，
`trusted_cost_model_config()` 必须把 `impact_alpha_bps` 标注为 `not_enforced`
而不是原样发布。**严重度取决于哪条路线：现状是"证书撒谎"，这是 P1。**

---

### F-03 【P1】涨跌停旗标按 close 判定，成交却按 open 撮合 —— 判据与撮合价不同源

- `src/quantagent/quant_math/ashare.py:320-338`（`limit_up_mask`：`close >= prev_close × (1+limit)`）
- `src/quantagent/backtest/engine.py:169-176`（用 `fill_date` 的 close-based 旗标构造 `can_buy`/`can_sell`）
- `src/quantagent/backtest/engine.py:36`（`fill_price_column = "open"`）

**缺陷**：引擎在 `open(t+1)` 成交，却用 `close(t+1)` 是否封板来决定能否成交。两个方向都会错：

1. **假阻断**：标的 `open(t+1)` 正常、盘中走高、`close` 封涨停 ⇒ 开盘本可买入，引擎拒单。
2. **假成交**：标的 `open(t+1)` 一字涨停、盘中开板、`close` 跌回 −5% ⇒ 旗标为 False，
   引擎在**买不到的涨停开盘价**上成交。

第 2 种是有利方向的偏差（在最强势的开盘价拿到货），且**无法从 reject 日志看出**——
它表现为一笔正常成交。**一字板（open==high==low==close==limit）在代码里没有任何专门判据**，
`grep -rn "一字\|one_word\|is_one_word\|open == high" src/quantagent/backtest/` 为空。

**最小复现命令**：
```bash
grep -n "def limit_up_mask" -A 20 src/quantagent/quant_math/ashare.py   # close-based
grep -n "fill_price_column" src/quantagent/backtest/engine.py           # "open"
```

**建议修法**：撮合价与可成交判据必须同源。要么按 `open` 判板
（`open >= prev_close×(1+limit)−tol` 才算开盘封板不可买），要么把 `fill_price_column`
改成 `close` 与判据对齐。同时补一个 `one_word_board` 旗标
（`high == low` 且触板）作为独立的 fail-closed 拒单理由。

---

### F-04 【P1】快引擎声明的滑点（5 bps）与实际收取的滑点（约 2.05 bps）不是同一个数

- `src/quantagent/quant_math/transaction_cost.py:10-19` — `CostModelConfig.slippage_bps = 5.0`
- `src/quantagent/backtest/engine.py:38` — `BacktestConfig.cost: CostModelConfig`（回测配置里可读到 5.0）
- `src/quantagent/backtest/fill_model.py:10-11` — `FillModelConfig.slippage_bps = 2.0`, `impact_bps = 1.0`
- `src/quantagent/backtest/engine.py:265` — 实际成交价来自 `self.fill_model.fill(...)`

**缺陷**：Round 早期修掉了"滑点收两次"（`engine.py:458-462` 注释记录了这次修复，
且我复核确认 `_execute_buy`/`_execute_sell` 确实**不再**再收 `cost.slippage_bps`
——**这一条是 PASS**）。但修复留下了一个新的不一致：`BacktestConfig.cost.slippage_bps`
仍然是 5.0 且仍被序列化进配置，而引擎**实际**施加的是
`FillModelConfig.slippage_bps=2.0 + impact_bps×participation`。
实测（F-01 复现脚本）成交价 `9.0018 / 9.0 − 1 = 2.0 bps`。

任何读 `BacktestConfig` 来声明"本次回测滑点 5 bps"的报告都是错的（实际 2 bps，偏乐观）。

**最小复现命令**：见 F-01 脚本的 `r.trades`，`price = 9.0018` vs reference `9.00`。

**建议修法**：`BacktestConfig` 里删掉 `cost.slippage_bps` 这一路（或让
`FillModelConfig` 从它取值），使"声明的滑点"与"施加的滑点"只有一个来源。

---

### F-05 【P1】`AShareFillModel` 的冲击成本是线性且被 `impact_bps=1.0` 压成接近零

- `src/quantagent/backtest/fill_model.py:40` — `impact = self.config.impact_bps * (filled / max(volume, 1.0))`

**缺陷**：这是**线性**参与率冲击，不是上游（AKQuant `04_backtest_engine`：
`Cost ∝ σ × √(Q/V)`）与本仓自己的 `AShareCostModel`（平方根律）所要求的形状。
且系数极小：在 `volume_cap_ratio=0.10 / participation_rate=0.05` 的封顶下，
participation 最大 0.05 ⇒ `impact ≤ 1.0 × 0.05 = 0.05 bps`。**实质为零。**

结果：**容量效应在快引擎里不存在**。参考 memory 里 EXP-024
"冠军组合活在流动性最低十分位、可辩护容量 10–30M CNY" —— 用这个引擎评估的容量
结论没有冲击成本支撑。

**最小复现命令**：
```bash
AI_quant_venv/bin/python3 -c "
import sys; sys.path.insert(0,'src')
from quantagent.backtest.fill_model import AShareFillModel
m = AShareFillModel()
for v in (1e5, 1e6, 1e9):
    r = m.fill('buy', 100000, 10.0, v)
    print(v, r.filled_quantity, r.fill_price, (r.fill_price/10.0-1)*1e4, 'bps')"
```

**建议修法**：与 F-02 合并为一次修复 —— 让快引擎与 `AShareCostModel` 共用同一个
平方根冲击函数，`impact_alpha_bps` 单一来源。

---

