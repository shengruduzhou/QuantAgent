# 本地 Paper Broker 设计报告

- **生成时间**：2026-07-29（UTC）
- **源提交**：`17ada19996fc5f1fceae06bac60ff5ab05d4a922`
- **实现**：`src/quantagent/paper/{broker,orders,portfolio,ledger,recovery,risk}.py`
- **测试**：`tests/paper/` —— **78 passed**

## 1. 安全定位（先说边界）

**完全本地。** 无连接器、无凭证、无账户 ID、无网络调用。
测试 `test_no_network_or_broker_import_in_the_package` 用 AST 断言整个 paper 包
**不导入** `requests`/`httpx`/`socket`/`urllib`/`xtquant`/`MetaTrader5`——
模拟订单**无法离开进程，因为没有出口**。

类属性写明身份：`is_local_simulation = True`、`has_broker_connection = False`。

## 2. 订单类型：刻意没有市价单

支持：限价买、限价卖、撤单、**可成交限价单（marketable limit）**、TWAP／VWAP／POV
母单。

**不提供无约束市价单。** 缺少限价直接抛异常，理由写在代码里：在 A 股涨跌停板上，
无价格上限的成交会报出**现实中根本拿不到的利润**。因此最"激进"的类型也带一个
不会越过的最差价。

母单是**调度**而非交易所指令：切成带限价的子单，只有子单进入撮合，这样参与率与
冲击建模才诚实。POV 在市场清淡时**会少成交**——这是策略本身的性质，不是缺陷
（有测试 `test_pov_underfills_a_thin_market`）。

## 3. A 股规则（复用既有规则库，不重复实现）

规则来自 `quantagent.backtest.ashare_rules`：

| 规则 | 实现与验证 |
| --- | --- |
| **T+1** | 持仓分 `total` 与 `sellable`，当日买入进 `pending_settlement`；**只有 `settle()` 能晋升**。当日卖出被拒（`test_t_plus_one_blocks_same_day_sell`） |
| 板块最小申报 | 主板/创业板 100 股整数倍；**科创板 200 股起**（`test_star_board_requires_two_hundred_shares`） |
| 零股卖出 | 仅一次性清仓时允许 |
| 涨跌停 | 封涨停**不可买**、封跌停**不可卖**（分方向验证） |
| ST 价格带 | ±5%，越界限价被拒；ST 买入默认按策略阻断 |
| 停牌 | 双向阻断 |
| 交易时段 | 午休不成交；收盘集合竞价可成交 |
| 印花税 | **仅卖出单边**（买入 0，卖出 >0，有对照测试） |
| 佣金 / 过户费 | 双边，含最低 5 元 |
| 参与率上限 | 默认 10% 会话成交量，超出部分不成交 |
| 公司行动 | 送转按比例缩放股数与成本，**账面价值不变**；分红计入现金 |

## 4. 成交价：滑点 + 平方根冲击

```
price = last ± (last × slippage_bps/1e4 + last × k × sqrt(participation))
```

随后被涨跌停与订单限价双重夹逼。
测试 `test_impact_grows_with_participation` 断言：同样 10,000 股，在 20 万成交量
的市场上均价**必须**高于在 5,000 万成交量的市场上。

## 5. 保真度的诚实边界

当前数据是日线，因此：

- 成交按**会话成交量参与**建模；
- **不声称任何排队位置**——代码里没有 queue-position 字段可填。

这与既有的保真度分级一致：只有拿到逐笔委托与成交才可能进入 Level A。

## 6. 状态机

`NEW → ACCEPTED → PARTIALLY_FILLED → FILLED`，以及 `CANCEL_REQUESTED → CANCELLED`、
`REJECTED`。非法跃迁抛 `OrderStateError`——**防止撤单后再来一笔成交**，那正是
悄悄放大模拟账面的方式。

## 7. 成本口径的一个刻意选择

**平均成本包含费用。** 排除费用会低估建仓价，从而美化之后的每一个 P&L 数字。
有测试 `test_average_cost_includes_fees`。

## 8. 测试汇总（78）

Paper Broker 42 条 + 风控 36 条，覆盖：T+1、板块手数、零股、ST 与普通涨跌停、
停牌、涨停拒买、跌停拒卖、部分成交、撤单、现金不足、可卖不足、费用、公司行动、
重启恢复、确定性重放。
