# A股 Tick 与 Level-2 供应商矩阵

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **探测脚本**：`scripts/probe_tick_l2_source_matrix.py`
- **实现模块**：`src/quantagent/data/microstructure/capability.py`、`public_tick_sources.py`
- **产物**：`runtime/data/capabilities/tick_l2/tick_l2_capability_matrix.{json,csv}`

## 1. 核心结论

| 结论 | 证据强度 |
| --- | --- |
| **本机 Level-2 的四个族全部无供应商**（快照 / 逐笔委托 / 逐笔成交 / 委托队列） | 实测 |
| **腾讯"分笔"是唯一仍可用的类 tick 公开源，但它不是逐笔成交** | 实测 |
| **新浪历史分笔接口已下线** | 实测 |
| 东财逐笔接口在轻负载下即返回 HTTP 502 | 实测 |
| QMT/xtdata 号称提供全部 Level-2，但本机无法运行客户端 | 实测（见专题报告） |

矩阵统计（48 个 cell）：

```
SERVING              10
NOT_OFFERED          16
CLIENT_UNAVAILABLE   18
UNAUTHORIZED          3
EMPTY_NO_DATA         1
```

**有 SERVING 供应商的族**：

| 数据族 | 供应商 |
| --- | --- |
| `daily_bars_raw` | tickflow、tencent、sina |
| `daily_bars_adjusted` | tickflow |
| `trade_ticks` | tencent（**注意语义，见 §3**） |
| `level1_quote` | tencent、tickflow |
| `security_master` | tickflow |

**无任何供应商的族**：
`minute_bars`、`level2_snapshot`、`level2_order_events`、`level2_transaction_events`、
`order_queue`、`cancellations`、`large_order_stats`、`auction_data`、`st_history`、
`suspension_history`、`corporate_actions`、`financials`、`index_membership`

> 说明：`corporate_actions` / `suspension_history` 在 U0 层另有已落盘来源
> （sina 复权因子、akshare 停牌快照），本矩阵只统计**本次 tick/L2 探测直接实测**
> 的结果，不把 U0 的既有产物重复计入。

## 2. 状态与权限二维分离

本矩阵的每个 cell 有两个独立轴，**不允许合并成一个"支持"布尔值**：

- `status`：本机此刻真实调用的结果；
- `entitlement`：账户权限等级，与本机能否调用无关。

状态取值及其含义：

| 状态 | 含义 |
| --- | --- |
| `SERVING` | 真实调用返回了真实数据（**唯一**可称"可用"） |
| `EMPTY_NO_DATA` | 调用成功但该标的/日期确实无数据 |
| `UNAUTHORIZED` | 供应商以权限理由拒绝 |
| `CLIENT_UNAVAILABLE` | 客户端/终端在本机根本无法运行 |
| `BLOCKED_BY_ENVIRONMENT` | 网络出口、端口或 DNS 阻断 |
| `THROTTLED` | 被限流，能力未证伪 |
| `NOT_PROBED` | SDK 有此接口但本次未调用 |
| `NOT_OFFERED` | 已确认供应商不提供 |
| `UNKNOWN_SEMANTICS` | 返回了内容但供应商不说明其含义 |

权限等级区分了任务要求的各种"免费"：

`PUBLIC_FREE`（无需账户）、`FREE_ACCOUNT`、`FREE_DELAYED`（延时行情）、
`ENTITLED_PAID`（本仓库持有且有效）、`PAID_NOT_HELD`（供应商售卖但本账户无）、
`BROKER_ACCOUNT_REQUIRED`（需券商资金账户）、`TRIAL_ONLY`、`ENTITLEMENT_UNKNOWN`。

**一个暴露了接口名但不返回授权数据的供应商，不算可用。**

## 3. 重要发现：腾讯"分笔"不是逐笔成交

这是本次探测最重要的语义修正。

### 3.1 实测数据

接口：`https://stock.gtimg.cn/data/index.php?appn=detail&action=data&c=sh600000&p=<page>&d=20260724`

返回格式（2026-07-27 实测原文片段）：

```
v_detail_data_sh600000=[0,"0/09:25:03/9.08/0.01/4655/4226740/B|1/09:30:02/9.08/0.00/1849/1681017/S|..."]
```

字段：`序号/时间/价格/涨跌/成交量(手)/成交额(元)/方向`

### 3.2 两项证据表明这是**3 秒聚合**

对 `600000.SH` 与 `000001.SZ` 的 2026-07-24 全日数据实测：

| 指标 | 600000.SH | 000001.SZ |
| --- | --- | --- |
| 相邻记录间隔恰为 3.0 秒的比例 | 88%（740/840） | 99%（830/840） |
| 其余间隔 | 均为 3 的整数倍（6/9/12 秒） | 同左 |
| `amount ≠ price × volume` 的记录占比 | **48%** | **82%** |

第二项是决定性的：若每条记录是一笔成交，则成交额必然等于价格乘以数量。实测大量
记录不满足，说明**每条记录内含多笔不同价格的成交**，而 `price` 字段是该 3 秒
快照的最后价，不是成交均价。

### 3.3 语义标注

因此该数据被标注为 `SNAPSHOT_DERIVED_TRADE_AGGREGATE`，而**不是** `TRADE_TICK`
或 `EXCHANGE_TRADE_EVENT`。

> **说明**：该分类是本仓库在任务给定的分类表之外**新增**的一项。原分类表没有
> 适合 3 秒聚合的槽位，而把它塞进 `TRADE_TICK` 会正好造成本分类体系要防止的
> 夸大。宁可扩展分类表，也不给真实数据贴错标签。

### 3.4 由此产生的硬性限制

- 买卖方向来自腾讯自己的 B/S/M 分类，**不是交易所发布的主动方向**，故
  `side_method = QUOTE_RULE_INFERRED`（推断，非观测）；
- `M`（中性）不被强行归入买或卖，`side` 置空；
- 无交易所序列号 → `sequence` 恒为空，只有本仓库自有的 `ingest_sequence`；
- 桶内成交先后顺序、单笔规模分布、3 秒以下的延迟结论**均不可得**；
- 回测保真度上限被自动降级（见 `docs/research/A股高频回测模型说明.md`）。

## 4. 新浪历史分笔：已下线

接口：`https://market.finance.sina.com.cn/downxls.php?date=2026-07-24&symbol=sh600000`

实测：**HTTP 200**，响应体 5 字节，内容为 `服务已下线`。

这是一个危险的失败模式：状态码正常，宽松的解析器会把它当成"当天无数据"。适配器
因此显式检测下线告示，返回 `RETRY_PERMANENT` 并置 `ok=False`，而不是返回空帧。
对应回归测试：`test_sina_decommission_notice_is_a_permanent_failure`。

> 注意：这只影响**分笔**接口。新浪的复权因子与退市股日线接口仍在 U0 层正常服务，
> 两者是不同端点。

## 5. 东财逐笔

接口：`https://push2.eastmoney.com/api/qt/stock/details/get`

实测：轻负载下即返回 **HTTP 502**，本次未取得数据（`EMPTY_NO_DATA`）。东财按 IP
与端点限流，失败属预期，已记录而非静默吞掉。该源保留为补充源，不作为主链路。

## 6. 板块覆盖缺口

在 2026-07-24 的采集队列中，北交所标的 `920238.BJ` 返回 **0 行**。即腾讯分笔接口
对北交所的覆盖存在缺口。该事实已记入 `fetch_log`，未被平均成"9/10 成功"掩盖。

## 7. 未在本机验证的合法来源

以下来源**未被证伪**，只是本机无法或未取证，不得写成"不可用"：

| 来源 | 状态 | 阻塞原因 |
| --- | --- | --- |
| QMT / XtData | `CLIENT_UNAVAILABLE` | 客户端仅 Windows，需券商资金账户 |
| MT5 券商源 | `CLIENT_UNAVAILABLE` | 本机无终端 |
| 上交所/深交所/北交所授权数据产品 | `NOT_PROBED` | 需商务授权，非技术阻塞 |
| 交易所授权历史数据服务商 | `NOT_PROBED` | 同上 |
| baostock / pytdx | `BLOCKED_BY_ENVIRONMENT` | 运行时出口仅开放 80/443 |

## 8. 建议优先级（基于证据，非基于名气）

1. **QMT/XtData（需券商权限 + Windows 主机）** —— 唯一在 API 层面覆盖逐笔委托、
   逐笔成交、千档队列的候选，且其 `get_his_st_data` 正好指向 U0 当前唯一的 PIT
   阻塞项。
2. **交易所授权历史数据** —— 合规且可回溯，但需商务流程。
3. **腾讯分笔** —— 已可用，但只能支撑 Level-C 保真度，且须始终标注为 3 秒聚合。
4. **MT5 券商源** —— 仅在证明存在真实交易所 A 股行情后才考虑，目前无证据。

## 9. 事实与推论

| 陈述 | 类别 |
| --- | --- |
| 腾讯分笔为 3 秒聚合（间隔 + 金额两项证据） | **事实（实测）** |
| 新浪 downxls 返回"服务已下线" | **事实（实测）** |
| 本机 Level-2 无任何供应商 | **事实（实测）** |
| QMT 能提供真正的逐笔委托 | **供应商声明 + 源码接口，未经本账户验证** |
| 3 秒聚合无法支撑排队位置估计 | **推论（由聚合性质直接得出）** |
