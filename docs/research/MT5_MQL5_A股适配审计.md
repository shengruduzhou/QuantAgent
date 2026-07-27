# MT5 / MQL5 A股适配审计

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **静态审计器**：`src/quantagent/mt5/mql5_audit.py`
- **参考实现**：`mql5/`

## 1. 官方文档要点（事实）

以下均为 2026-07-27 访问官方文档所得。

| 主题 | 官方地址 | 关键事实 |
| --- | --- | --- |
| Python 集成 | `https://www.mql5.com/en/docs/python_metatrader5` | 通过 IPC **直连本地终端**获取数据；安装说明指向 Windows 版 Python |
| Tick 获取 | `.../mt5copyticksrange_py` | `copy_ticks_range(symbol, from, to, flags)` |
| 品种发现 | `.../mt5symbolsget_py` | `symbols_get()` 返回**券商**品种，非交易所名录 |
| 市场深度订阅 | `.../mt5marketbookadd_py` | 订阅成功 ≠ 券商提供深度 |
| 市场深度读取 | `.../mt5marketbookget_py` | 必须实际读取才知道档位数 |
| 自定义品种 | `https://www.mql5.com/en/docs/customsymbols` | `CustomSymbolCreate` / `CustomTicksReplace` / `CustomRatesUpdate` |
| 事件处理 | `.../event_handlers` | `OnTick` / `OnBookEvent` / `OnTradeTransaction` |
| CTrade | `.../standardlibrary/tradeclasses/ctrade` | 方法名首字母大写：`Buy` / `Sell` / `PositionOpen` |
| 指标 | `.../indicators`、`.../series/copybuffer` | **MQL5 指标函数返回句柄，取值必须用 `CopyBuffer`** |
| 生成 tick | `https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation` | 测试器可由 bar 合成 tick |

**中文教学站**（`mt5.me` 系列）按任务要求仅作为**辅助教学材料**，不作为 API 权威。

## 2. MQL4 → MQL5 的高危误用（本审计的核心）

这些错误的共同特征是：**能编译通过，然后静默地做错事**。

### 2.1 指标函数返回句柄，不是数值（最高危）

```cpp
// ❌ MQL4 写法照抄，MQL5 下编译通过但完全错误
double fast = iMA(_Symbol, PERIOD_D1, 10, 0, MODE_EMA, PRICE_CLOSE);
if(fast > slow) { ... }   // 比较的是句柄编号（通常是 0,1,2...），不是价格
```

句柄编号通常是很小的整数，**看起来像一个合理的价格**，因此极难通过肉眼或回测
曲线发现。

```cpp
// ✅ MQL5 正确写法
int handle = iMA(_Symbol, PERIOD_D1, 10, 0, MODE_EMA, PRICE_CLOSE);
double buf[];
if(CopyBuffer(handle, 0, 0, 1, buf) != 1) return;   // 未就绪就退出
double fast = buf[0];
// OnDeinit 中：IndicatorRelease(handle);
```

### 2.2 其余已编码为检测规则的误用

| 规则 ID | 问题 | 严重度 |
| --- | --- | --- |
| `mql4_indicator_value` | 句柄当数值用 | ERROR |
| `copybuffer_result_ignored` | 忽略 `CopyBuffer` 返回值，"未就绪"读成"值为 0" | ERROR |
| `ctrade_lowercase` | `trade.buy(...)`——CTrade 无此小写方法 | ERROR |
| `missing_real_account_guard` | EA 无实盘账户拒绝 | ERROR |
| `martingale_doubling` | `volume *= 2` 类加倍摊平 | ERROR |
| `fx_contract_size` | 合约乘数 100000（外汇默认） | ERROR |
| `missing_indicator_release` | 句柄不释放，反复加载耗尽资源 | WARN |
| `missing_trade_transaction_handler` | 无 `OnTradeTransaction`，漏掉异步/部分成交与拒单 | WARN |
| `missing_order_check` | `OrderSend` 前无 `OrderCheck` | WARN |
| `dom_subscribed_not_read` | 只 `MarketBookAdd` 不 `MarketBookGet` | WARN |
| `fx_lot_assumption` | 0.01/0.1 手——A 股按股数下单 | WARN |
| `martingale_grid_marker` | 出现马丁/网格词汇 | WARN |

## 3. A 股特有的不适配点

MT5 的默认模型来自外汇与 CFD，直接套用到 A 股现货会在以下位置出错：

| MT5 默认 | A 股现实 | 本仓库处置 |
| --- | --- | --- |
| 合约乘数 100,000，手数 0.01 | 按**股**下单，主板 100 股整数倍 | 自定义品种 `contract_size = 1`，`volume_step` 按板块 |
| 可日内平仓 | **T+1**：当日买入次日才可卖 | `QA_RoundShares` + 模拟器 `tradability()` |
| 无涨跌幅 | 主板 ±10%、ST ±5%、创业/科创 ±20%、北交所 ±30% | `ashare_rules.price_limits()` |
| 双边点差成本 | 卖出单边印花税 0.05% + 双边过户费 0.001% | `ashare_rules.trading_costs()` |
| 连续交易 | 午休 11:30–13:00、集合竞价、科创板盘后固定价格 | `QA_IsContinuousSession` + `session_phase` |
| 服务器时间即市场时间 | MT5 给的是**券商服务器时间** | `QA_IsContinuousSession` 要求调用方先换算，**不替其猜时区** |
| 图上有报价即可交易 | 涨停封板买不进、跌停封板卖不出 | 模拟器分方向否决 |

## 4. 参考实现如何避免上述问题

`mql5/Include/QuantAgent/`：

- **`IndicatorHandle.mqh`** —— `QAIndicator` 类持有句柄，析构即 `IndicatorRelease`；
  `Read()` 返回**实际拷贝条数**，短读**不补零**（补零会把"未就绪"伪装成"值为 0"）；
  `ReadLatest()` 未就绪返回 `false`，绝不返回假值。
- **`AShareGuards.mqh`** —— 板块推断、按板块的最小股数与步长、连续竞价判定、
  行情新鲜度、实盘账户拒绝、`QA_` 品种白名单、测试器 tick 来源标注。
  其中 `QA_RoundShares` 对买入**不足最小单位返回 0 而非向上取整**——向上取整会让
  回测悄悄下出真实市场会拒绝的单。
- **`RiskGuard.mqh`** —— 单日下单笔数/名义金额/亏损上限、跨日重置、熔断开关。
  风控失败**阻止**下单，而非记日志后继续。
- **`Logging.mqh`** —— 决策留痕，**拒单理由同样记录**：从不解释拒单原因的 EA
  无法审计。

## 5. 审计结果

对 `mql5/` 下 6 个源文件运行静态审计：

```
files_audited: 6
counts_by_severity: {}       # 无任何 ERROR / WARN
clean: true
```

**该"clean"结论仅在检测规则被证明有效的前提下才有意义。** 因此
`tests/test_mt5_bridge_and_mql5_audit.py` 先对**故意写坏**的样本逐条验证每个规则
会触发，再断言本仓库源码干净。另有一条测试确保审计器**不会**因注释里的说明文字
误报（解释"为什么 `trade.buy` 是错的"的注释本身不应被判违规）。

## 6. 无法验证的部分（明确声明）

**MetaEditor 是 MQL5 唯一的编译器，且不在本机运行。因此 `mql5/` 下的源码
未经编译验证。**

静态审计覆盖的是"能编译通过但行为错误"这一类缺陷；语法错误、包含路径问题、
标准库版本差异等**只有编译器能发现**，本次未覆盖。审计器输出中显式携带该限制
声明（`limitation` 字段），并有测试断言该声明存在。

## 7. 明确不采用的社区实践

- **马丁格尔 / 无限网格 / 加倍摊平**：尾部风险无界，不作为任何默认策略。若需研究，
  须置于隔离的教学分类下并附尾部风险警告。
- **把测试器生成 tick 当作真实 tick 汇报**：MQL5 不向 EA 暴露建模模式，因此
  `QA_TickSourceLabel()` 要求运行方**显式声明**，函数本身**拒绝推断**——推断错误
  正是"生成 tick 被报告成真实 tick"的成因。
- **假定所有券商都提供 DOM**：订阅成功不构成证据，必须实际读到档位。
