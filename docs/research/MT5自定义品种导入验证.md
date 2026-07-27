# MT5 自定义品种导入验证

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **实现模块**：`src/quantagent/mt5/custom_symbol_bridge.py`
- **导出脚本**：`scripts/export_mt5_custom_symbols.py`
- **产物目录**：`runtime/data/mt5_custom_symbols/`

## 1. 单向设计（架构前提）

```
不可变事件日志  ──►  MT5 自定义品种  ──►  图表 / Strategy Tester
                （无反向路径）
```

本模块**没有**任何"从 MT5 读回行情写入日志"的函数。原因：事件一旦进入终端，
MT5 无法区分"外部导入的"与"自己生成的"。因此权威数据只能单向流出。

## 2. 本次导出结果（真实执行）

| 项 | 值 |
| --- | --- |
| 交易日 | 2026-07-24 |
| 导出标的 | **9** |
| tick 总数 | **24,222** |
| 日线总数 | **2,250** |
| 导入后数据类 | `CUSTOM_SYMBOL_REPLAY` |
| 原始数据类 | `SNAPSHOT_DERIVED_TRADE_AGGREGATE` |

各 bundle（节选）：

| 自定义品种 | 规范代码 | 板块 | ticks | bars | 源内容哈希 |
| --- | --- | --- | --- | --- | --- |
| `QA_002156_SZ` | 002156.SZ | SZ_Main | 4,781 | 250 | `3a4e4706ed504236` |
| `QA_002211_SZ` | 002211.SZ | SZ_Main | 208 | 250 | `012d478902e05020` |
| `QA_002759_SZ` | 002759.SZ | SZ_Main(ST) | 4,419 | 250 | `77534b17f69247fc` |
| `QA_300308_SZ` | 300308.SZ | ChiNext | 4,764 | 250 | `8ce5d5627d80cd97` |
| `QA_300555_SZ` | 300555.SZ | ChiNext | 127 | 250 | `7c9894f1d7de0686` |

## 3. 数据类变更被显式记录

导入后一律标记为 `CUSTOM_SYMBOL_REPLAY`，但 manifest **同时保留原始数据类**。
这样任何 Strategy Tester 结果都可回溯到它究竟建立在交易所成交之上，还是建立在
3 秒聚合之上。

## 4. 随行的两条诚实警告

导出过程自动产生（未被平滑掉）：

1. > "origin data is 3-second snapshot aggregates: the terminal will display them
   > as ticks, but intra-bucket sequencing and per-trade size are not present in
   > the underlying data"

2. > "source carries no quotes; bid/ask exported as 0 rather than being
   > synthesised from last price, so spread-based indicators in the terminal
   > will be meaningless"

第二条尤其重要：**本仓库拒绝用最后价合成买卖盘**。合成点差会让终端里所有点差类
指标读到一个由本仓库凭空发明的数字。宁可导出 0 并声明"未提供"。

## 5. A 股语义正确性（非外汇默认）

| 属性 | 设定值 | 理由 |
| --- | --- | --- |
| `contract_size` | **1**（股） | 外汇默认 100,000 会让每一笔仓位规模差 5 个数量级 |
| `currency` | CNY | |
| `digits` / `tick_size` | 2 / 0.01 | A 股报价到分 |
| `volume_min` / `volume_step` | 主板 100/100；**科创板 200/1**；北交所 100/1 | 按板块真实最小申报单位 |
| 报价时段 | 09:15–11:30, 13:00–15:00 | 含集合竞价 |
| 交易时段 | 09:30–11:30, 13:00–15:00 | |
| 科创板额外时段 | **15:05–15:30** | 盘后固定价格交易 |

命名一律 `QA_` 前缀（`QA_600000_SH`），确保在同一终端内**不可能**与券商可交易
品种混淆。

## 6. tick 字段映射

| MT5 字段 | 来源 | 说明 |
| --- | --- | --- |
| `time` / `time_msc` | `exchange_time` | 秒 / 毫秒 |
| `last` | `price` | |
| `volume_real` | `volume_shares` | **股数放在能保留精度的字段** |
| `volume` | 四舍五入的股数 | MT5 整型字段 |
| `bid` / `ask` | 源有则用，无则 **0** | 不合成 |
| `flags` | `LAST\|VOLUME`，按 side 追加 `BUY`/`SELL`，有报价再加 `BID`/`ASK` | |

## 7. 导入校验

`verify_import()` 比对终端侧行数与导出时记录的 manifest 行数。

**措辞刻意保守**：判定值为 `COUNTS_MATCH` 而非 `IMPORT_VERIFIED`——行数一致只
证明没有丢行，**不证明**价格与时间戳在往返中未被改变。这个区别在函数 docstring
中写明。

本机无终端，因此**终端侧校验本次未执行**；`verify_import` 的两条路径（一致/不一致）
均有单元测试覆盖。

## 8. 真实 tick 与生成 tick 的区分

`classify_tester_ticks(modelling_mode)`：

| 建模模式 | 数据类 | 可否称为"真实 tick" |
| --- | --- | --- |
| `EVERY_TICK_BASED_ON_REAL_TICKS` | `CUSTOM_SYMBOL_REPLAY` | ✅ 是（消费我们提供的 tick） |
| `EVERY_TICK` | `GENERATED_TESTER_TICK` | ❌ 否 |
| `ONE_MINUTE_OHLC` | `GENERATED_TESTER_TICK` | ❌ 否 |
| `OPEN_PRICES_ONLY` | `GENERATED_TESTER_TICK` | ❌ 否 |

生成模式的说明文字：

> "the tester synthesised ticks from bars; results describe the tick generator's
> behaviour and must not be reported as tick-level results"

**Strategy Tester 本次未运行**（无终端），因此本报告不含任何回测绩效数字。

## 9. 不可变日志的保护

`RawEventStore.append()` 拒绝写入 `NON_AUTHORITATIVE_CLASSES`，其中就包括
`CUSTOM_SYMBOL_REPLAY` 与 `GENERATED_TESTER_TICK`。即：**从 MT5 出来的东西在架构
上无法回流进权威日志**，有两条测试锁定该行为。

## 10. 复现命令

```bash
python scripts/export_mt5_custom_symbols.py --trade-date 2026-07-24 --output runtime/data/mt5_custom_symbols
```
