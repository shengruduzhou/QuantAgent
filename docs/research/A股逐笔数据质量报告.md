# A股逐笔数据质量报告

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **采集脚本**：`scripts/acquire_ashare_ticks.py`
- **实现模块**：`src/quantagent/data/microstructure/{integrity,reconcile,store}.py`
- **产物**：`runtime/data/market_events/_reports/tick_acquisition_2026-07-24.json`

## 1. 本次采集范围

**交易日**：2026-07-24
**供应商**：腾讯（`stock.gtimg.cn`）
**数据类**：`SNAPSHOT_DERIVED_TRADE_AGGREGATE`（3 秒聚合，非逐笔）

样本选取原则：**一只蓝筹证明不了覆盖**。因此按板块分层，每个板块取当日成交额
最高与最低各一只，另加一只 ST 标的：

| 标的 | 板块 | 选取理由 | 采集行数 |
| --- | --- | --- | --- |
| 603986.SH | 沪主板 | 板块成交额最高 | 4,819 |
| 600365.SH | 沪主板 | 板块成交额最低（流动性下限） | 171 |
| 002156.SZ | 深主板 | 板块成交额最高 | 4,781 |
| 002211.SZ | 深主板 | 板块成交额最低 | 208 |
| 300308.SZ | 创业板 | 板块成交额最高 | 4,764 |
| 300555.SZ | 创业板 | 板块成交额最低 | 127 |
| 688008.SH | 科创板 | 板块成交额最高 | 4,811 |
| 688184.SH | 科创板 | 板块成交额最低 | 122 |
| 002759.SZ | ST | ST 标的（±5% 价格限制） | 4,419 |
| 920238.BJ | 北交所 | 当日唯一有日线的北交所标的 | **0（缺口）** |

合计：**24,222 条事件，9 只标的成功，1 只失败**。

## 2. 完整性检查结果

13 项检查：**PASS 7 / WARN 2 / NOT_RUN 4 / FAIL 0**

### 2.1 通过项（7）

`schema_columns`、`declared_semantics`、`timestamp_monotonicity`、
`manufactured_identifiers`、`price_sanity`、`volume_sanity`、`clock_drift`

其中 `manufactured_identifiers` 专门检测"看起来像生成序列"的标识符：若某个受
契约保护的字段恰好是稠密的 `0..N-1`，判 FAIL。本次通过，即腾讯源确实没有被
补造过交易所标识符。

### 2.2 警告项（2）—— 均为真实且必要的警告

**`side_provenance` = WARN**

```
0 observed sides, 21426 inferred sides — inferred direction is not an observation
```

腾讯的 B/S 标记是其自身分类，不是交易所发布的主动买卖方向。所有方向标注
`side_method = QUOTE_RULE_INFERRED`。若某日出现 `observed_rows > 0`，只可能来自
带订单号的 Level-2 源。**推断方向不是观测方向**，这条警告不应被消除。

**`session_boundaries` = WARN**

```
164 events (0.677%) fall in the post-close window
```

见 §4 的时段分析。这些记录被隔离到 `POST_CLOSE_UNCLASSIFIED` 时段，既不接受进
连续竞价特征，也不当作噪声丢弃。

### 2.3 未运行项（4）—— **不得视为通过**

`duplicate_events`、`sequence_gaps`、`cumulative_monotonicity`、`book_ordering`

原因：公开源**不发布交易所序列号、成交编号或盘口**，因此这四项检查在物理上
无法评估。

这是本仓库的一条硬性纪律：`NOT_RUN ≠ PASS`。`IntegrityReport.usable` 要求
**既无 FAIL 也无 NOT_RUN**，故本数据集 `usable = False`，其直接后果是回测保真度
被自动降级（Level-A 需要"已证明的序列完整性"）。

## 3. 与 U0 日线面板的对账

**这是判断一个 tick 源能否被信任的决定性检验。**

结果：**9/9 个 symbol-day 通过，匹配率 100%**

| 状态 | 数量 |
| --- | --- |
| `MATCH`（六个字段全部一致） | 6 |
| `MATCH_WITHIN_AGGREGATION_LIMITS` | 3 |
| `MISMATCH` | 0 |
| `NO_PANEL_ROW`（无法验证） | 0 |

### 3.1 逐字段结果（600000.SH，2026-07-24）

| 字段 | 面板值 | tick 重建值 | 差异 |
| --- | --- | --- | --- |
| open | 9.08 | 9.08 | 0 |
| high | 9.12 | 9.12 | 0 |
| low | 9.02 | 9.03 | **+0.01** |
| close | 9.04 | 9.04 | 0 |
| volume | 50,675,100 | 50,675,100 | **0** |
| amount | 459,278,100 | 459,278,079 | −21（4.6e-8 相对误差） |

成交量**完全一致**，成交额在 4.59 亿中差 21 元（取整）。

### 3.2 `low` 差异的性质：聚合盲区，而非错误

3 秒桶的 `price` 是该快照的最后价，因此**桶内出现但未收在桶尾的极值不在数据里**。
聚合只会**丢失**极值，不会**制造**极值。

对账器据此设置了独立状态 `MATCH_WITHIN_AGGREGATION_LIMITS`，其判定条件是**双重**的：

1. 不一致字段是 `high` / `low` 的子集；
2. **且**重建区间严格落在面板区间**内部**（重建 high < 面板 high，重建 low > 面板 low）。

若重建 high **高于**面板 high，则判 `MISMATCH`——聚合无法凭空造出更高的价格，那
只能是真错误。该规则有专门测试
（`test_derived_high_above_panel_high_is_a_real_error`）。

对非聚合类（如 `EXCHANGE_TRADE_EVENT`）**不给这个宽容**，同样有测试覆盖。

## 4. 时段边界：对账驱动的两处修正

初次对账失败，暴露了会话分类器的两个真实缺陷。**是对账推动了修正，而不是反过来
放松对账。**

### 4.1 收盘集合竞价成交回报戳在 15:00:03

600000.SH 的缺失量恰为 **486,400 股 / 4,396,839 元**，隐含价 9.0396 ≈ 面板收盘价
9.04。该记录时间戳为 `15:00:03`——集合竞价 14:57–15:00 结束后，结果在**下一个
快照**才发布。

原半开区间 `[14:57, 15:00)` 把它排除在外，导致全日少算恰好一笔集合竞价。修正为
`[14:57, 15:01)`。

### 4.2 早盘收盘价戳在 11:30:00

5 只标的中 5 只的早盘收盘聚合时间戳为 `11:30:00`–`11:30:01`，被原区间
`[09:30, 11:30)` 推入午休，触发 FAIL。修正为 `[09:30, 11:31)`。

### 4.3 盘后记录：新增独立时段而非二选一

15:05–15:23 之间存在 235 条记录（占全日成交量 0.047%），**五个板块都有**，因此
不能简单归为科创板盘后固定价格交易。

这些记录**不在**交易所日线内（排除它们后对账才完全吻合）。其性质——大宗交易回报、
零股清算、还是供应商重发——**未能确定**。

处理方式：新增 `POST_CLOSE_UNCLASSIFIED` 时段，
- 不计入日线重建（`EXCLUDED_FROM_DAILY_BAR`）；
- 完整性检查报 WARN 并给出占比；
- 不静默丢弃，也不当作正常盘中数据。

同时会话分类器改为**按标的板块**判定，使科创板 15:05–15:30 的盘后固定价格交易
能被正确识别为 `AFTER_HOURS_FIXED_PRICE` 而非"不明盘后记录"。

## 5. 不可变日志

事件写入 `runtime/data/market_events/`，Hive 分区：

```
provider=tencent/family=trade_event/exchange=SH/trade_date=2026-07-24/symbol=600000.SH/part-0000-<hash>.parquet
```

本次写入 **9 个分区 / 24,222 行**，每个分区附 `*.receipt.json`（行数、内容哈希、
ingest 序号区间、写入时刻）。

写入期强制的三条不变量：

1. **语义已声明**：`data_class` 必须在分类表内，且不得属于
   `NON_AUTHORITATIVE_CLASSES`（生成 tick、bar 派生 tick、回放、未知语义）；
2. **契约完整**：缺列直接报错，不做 null 填充；
3. **不可变**：同内容重写为幂等；**不同内容**写入已有分区必须显式 `supersede=True`，
   并留下 tombstone，而不是删除历史。

`symbol` 等分区键经过白名单正则校验，`../../etc/passwd` 类输入被拒（有测试）。

## 6. 已知限制（不得在下游被省略）

1. 数据是 **3 秒聚合**，不是逐笔；桶内顺序与单笔规模不可得。
2. 买卖方向是**推断**，不是交易所字段。
3. 无交易所序列号 → **无法证明流的完整性**，只能证明与日线一致。
4. `low` / `high` 存在系统性内缩偏差（3/9 标的可见）。
5. 北交所在该源存在覆盖缺口（920238.BJ 返回 0 行）。
6. 单日样本（2026-07-24），未做跨市场状态（涨跌停潮、极端波动）的稳健性检验。

## 7. 复现命令

```bash
python scripts/acquire_ashare_ticks.py --trade-date 2026-07-24 --cohort board-spread --output runtime/data/market_events
```
