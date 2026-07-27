# MQL5 策略与指标代码审计

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **审计器**：`src/quantagent/mt5/mql5_audit.py`
- **测试**：`tests/test_mt5_bridge_and_mql5_audit.py`

## 1. 审计结果

```
files_audited: 6
counts_by_severity: {}     # 无 ERROR、无 WARN
clean: true
```

受审文件：

| 文件 | 类型 |
| --- | --- |
| `mql5/Include/QuantAgent/Logging.mqh` | 基础设施 |
| `mql5/Include/QuantAgent/AShareGuards.mqh` | A 股护栏 |
| `mql5/Include/QuantAgent/IndicatorHandle.mqh` | 指标句柄生命周期 |
| `mql5/Include/QuantAgent/RiskGuard.mqh` | 风控与熔断 |
| `mql5/Experts/QuantAgent/PaperTrendEA.mq5` | 参考 EA（纸上交易） |
| `mql5/Indicators/QuantAgent/TradeSignImbalance.mq5` | 参考指标 |

## 2. 为什么这个 "clean" 是可信的

**一个从不报警的检查器，其"通过"没有信息量。** 因此测试设计为两段：

**第一段：证明每条规则能抓到坏代码。** 用一段故意写错的 EA 样本，逐条断言规则触发：

| 测试 | 断言触发的规则 |
| --- | --- |
| `test_catches_mql4_style_indicator_value` | `mql4_indicator_value` |
| `test_catches_lowercase_ctrade_call` | `ctrade_lowercase` |
| `test_catches_martingale_doubling` | `martingale_doubling` |
| `test_catches_ignored_copybuffer_result` | `copybuffer_result_ignored` |
| `test_catches_missing_real_account_guard` | `missing_real_account_guard` |
| `test_catches_missing_indicator_release` | `missing_indicator_release` |
| `test_catches_missing_trade_transaction_handler` | `missing_trade_transaction_handler` |
| `test_catches_fx_contract_size` | `fx_contract_size` |
| `test_catches_dom_subscribed_but_never_read` | `dom_subscribed_not_read` |

**第二段：证明不误报。**

- `test_does_not_fire_on_prose_inside_comments`：注释里解释"`trade.buy` 是错的"
  不应被判违规——审计器先把注释与字符串置空再匹配，且保留行号对齐。
- `test_correct_handle_usage_is_not_flagged`：正确的 `int handle = iMA(...)` +
  `CopyBuffer` + `IndicatorRelease` 组合不触发任何规则。

**第三段**：在前两段成立的前提下，才断言本仓库源码干净
（`test_repository_sources_are_clean`）。

## 3. 参考实现的关键设计

### 3.1 指标句柄（`IndicatorHandle.mqh`）

- `QAIndicator` 类：构造接管句柄，析构 `IndicatorRelease`；
- `Read()` 返回**实际拷贝条数**，短读**不补零**——补零会让"指标尚未就绪"与
  "指标值为 0"不可区分；
- `ReadLatest()` 未就绪返回 `false`，**绝不返回假值**。

### 3.2 A 股护栏（`AShareGuards.mqh`）

- `QA_RoundShares()`：买入不足最小单位**返回 0 而非向上取整**。向上取整会让回测
  悄悄下出真实市场会拒绝的单；
- 卖出**仅在一次性清仓时**允许零股，这是交易所真实规则，把"卖出总是自由"建模会
  高估退出的顺畅程度；
- `QA_IsContinuousSession()` 要求调用方**先完成时区换算**，函数**拒绝替调用方猜
  时区**——猜错会让整段午休变成可交易；
- `QA_AssertNotRealAccount()`：`ACCOUNT_TRADE_MODE_REAL` 默认拒绝初始化；
- `QA_IsWhitelistedSymbol()`：只允许 `QA_` 前缀品种，防止误触券商真实品种；
- `QA_TickSourceLabel()`：MQL5 不向 EA 暴露建模模式，故要求运行方**显式声明**，
  函数**拒绝推断**。

### 3.3 风控（`RiskGuard.mqh`）

单日下单笔数 / 名义金额 / 亏损上限、跨日重置、熔断开关、行情新鲜度、时段检查、
品种白名单。**风控失败阻止下单**，不是记日志后继续。

### 3.4 参考 EA（`PaperTrendEA.mq5`）

- `OnInit` 中先拒绝实盘账户，再校验品种白名单与板块；
- 指标全部走句柄 + `CopyBuffer`；`OnDeinit` 释放；
- `OrderSend` 前先 `OrderCheck`；
- 实现 `OnTradeTransaction` 处理异步/部分成交与拒单；
- 测试器中若未声明真实 tick，主动打印警告。

### 3.5 参考指标（`TradeSignImbalance.mq5`）

文件头部**明确声明**其方向来自供应商的 quote-rule 分类，**不是**交易所发布的
主动买卖方向，因此"可用于研究相对变化，不可用于宣称订单流结论"。平盘 bar
**不计入任何一边**，而非随意归属。

## 4. 明确不提供的内容

- **马丁格尔 / 无限网格 / 加倍摊平**：不作为任何默认策略。
- **实盘下单路径**：本任务内所有 EA 均为纸上交易，无真实账户下单。
- **绩效数字**：Strategy Tester 未运行（本机无终端），因此本报告不含任何回测结果。

## 5. 编译状态（明确声明）

> **MetaEditor 是 MQL5 唯一的编译器，不在本机运行。因此以上源码未经编译验证。**

静态审计覆盖"能编译通过但行为错误"这一类；语法错误、包含路径、标准库版本差异
**只有编译器能发现**。审计器输出携带该限制声明，并有测试断言其存在
（`test_audit_states_its_compilation_limitation`）。
