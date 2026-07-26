# U0 数据层最终状态报告

- 生成时间：2026-07-27
- 源码提交：`agent/u0-assembly-scale` → `59ac825`；`agent/u0-data-closure`（本次）
- 基线：`main` = `699e9d6`（PR #24 已合并）
- 运行环境：Linux，出网仅放行 TCP 80/443；GPU = RTX 3090 24GB（空闲）

本报告只记录**已实测**的结果。未完成的阶段在第 8 节明确列出，不做任何"应该可以"的推断。

## 1. 采集最终状态

上一轮的两个采集 worker 在会话结束时被终止，终止前已把绝大部分宇宙抓完。分区实测数量：

| staging 目录 | 分区数 |
|---|---|
| `runtime/data/u0/bars/tickflow_raw` | 5,054 |
| `runtime/data/v7/full_universe/_staging`（历史 TickFlow） | 1,892 |
| `runtime/data/u0/bars/tencent_raw` | 3,468 |
| `runtime/data/u0/bars/sina_delisted` | 318 |
| **去重后唯一证券** | **5,892 / 5,894 = 99.97%** |

Track F（`catchup_panel_chunked.py`）在本次会话期间持续运行并持有
`.catchup_supervisor.lock`，U0 未与其争抢 vendor 限额。

## 2. 每只证券的最终处置（权威证据）

新增 `scripts/u0_exchange_register_reconcile.py`，用**交易所自己的上市名录**核对
master，而不是相信 vendor 的 instrument 列表。

- 数据源：深交所 `A股列表`（`CATALOGID=1110`），145 页 / **2,892 条**，as_of `2026-07-24`
- 上交所 `commonQuery.do` 本次返回 0 行（端点在本运行环境间歇不可用）；代码对
  **无法读取名录的交易所**标记为 `UNVERIFIED`，不会因缺证据就判定为"未上市"

结果（`runtime/data/u0/master_disposition.parquet`）：

| 处置 | 数量 |
|---|---|
| `LISTED_WITH_HISTORY` | 5,531 |
| `DELISTED_WITH_HISTORY` | 361 |
| `PRE_LISTING_NO_SESSIONS` | 2 |
| 合计 | 5,894 |

- vendor 与交易所**上市日期分歧 = 0**
- 交易所名录中存在但 master 缺失的代码 = **0**
- `PRE_LISTING_NO_SESSIONS` = `001232.SZ`（嘉立创）、`301677.SZ`（欣兴工具）

这两只的判定依据是三重实测证据，不是推测：

1. TickFlow 与腾讯**均返回 EMPTY**（ledger 记录在案）；
2. vendor `listing_date` 为 epoch 哨兵 `1970-01-01`（全 master 仅此 2 行）；
3. 腾讯实时快照 `last_price == prev_close`、**成交量 = 0**、时间戳 `09:00:00`；
4. 深交所 2,892 条上市名录中**不存在**这两个代码。

即：已发行但尚未开始交易。**未生成任何占位 bar**。

## 3. 面板与 provider 优先级

TickFlow 追平后重新装配（`scripts/u0_assemble_panel.py`），高优先级 TickFlow 分区
覆盖了此前的腾讯回退分区：

| 指标 | 数值 |
|---|---|
| 面板 | `runtime/data/u0/panel/daily_bars_raw.parquet` |
| 行数 | **17,829,080** |
| 证券数 | **5,892** |
| 日期范围 | 1990-12-19 → 2026-07-24 |
| 复权口径 | none（raw 未复权），经除权日因子回放取证 |
| `amount` 覆盖率 | **98.75%**（重装配前为 69.2%） |

serving provider 分布：TickFlow 5,054 + 历史 TickFlow 674 + Sina(截断) 151 +
腾讯 13，未覆盖 2。腾讯回退从 1,260 只降到 13 只。

板块覆盖：

| 板块 | 覆盖 / 总数 |
|---|---|
| SH_Main | 1,848 / 1,848 |
| SZ_Main | 1,660 / 1,661 |
| ChiNext | 1,440 / 1,441 |
| STAR | 614 / 614 |
| BSE | 330 / 330 |
| 退市 | 361 / 361 |

隔离（不进面板、保留 provenance）：OHLC 矛盾 8 行、上市前 14 行、退市后 12 行。

## 4. 缺口语义

面板只包含**真实成交日**，不写 NaN 行。存续期内无 bar 的交易日进
`session_gaps.parquet` 并分为三类：

| 分类 | 数量 |
|---|---|
| `SUSPENDED`（命中 vendor 停牌区间） | 7,373 |
| `PROVIDER_HISTORY_TRUNCATED`（Sina 1023 根上限） | 315,697 |
| `MISSING_UNEXPLAINED` | 518,417 |

`MISSING_UNEXPLAINED` 绝大部分落在停复牌快照覆盖窗口（2019-09-04 起）之前——
窗口之外**不声称无停牌**，如实标注。

## 5. PIT 区间表

| 字段 | 状态 | 实测规模 |
|---|---|---|
| 交易日历 | AVAILABLE | 8,797 个交易日，1990-12-19 起 |
| 上市日期 | AVAILABLE | 5,894 / 5,894 |
| 退市日期 | AVAILABLE | 361 条带日期 |
| 复权因子 / 公司行动身份 | AVAILABLE | 71,574 条 / 5,891 只 |
| 分红送转 | AVAILABLE | 57,632 条 / 5,773 只 |
| 停牌区间 | AVAILABLE | 2,157 段 / 2,157 只，快照窗口 2019-09-04 → 2026-07-24 |
| 涨跌停规则 / IPO 特殊规则 | AVAILABLE | 交易所规则确定性推导 |
| **历史 ST 区间** | **BLOCKED_BY_DATA（部分）** | 深交所简称变更登记 906 段 / 651 只；沪市、北交所无带日期登记 |

## 6. 校验与对账

`scripts/u0_validate.py`：**23 PASS / 2 WARN / 0 FAIL / 0 NOT_RUN**

关键实测结果：

- `adjustment_is_raw` **PASS**：22,657 个除权事件，符号一致率 **0.9635**；
  旧面板 frozen_cohort 为 0.5123（`ADJUSTED_OR_MIXED`）
- `corporate_action_agreement` **PASS**：5,772 只、57,626 个除息日，
  **57,491 个与因子跳变精确对齐 = 99.77%**
- `amount_volume_units` **PASS**：20 万抽样行中 **99.94%** 的 `amount/volume`
  隐含 VWAP 落在当日 [low, high] 内；隐含 VWAP / close 中位数 = 1.000，
  证明 `volume` 单位是**股**而非手
- `cross_provider_reconciliation` **PASS**：12 只 TickFlow 服务的证券对独立公开源
  重新抓取，收盘与成交量匹配率 **1.000**
- `intraday_to_daily_reconciliation` **PASS**：800 个交易日，收盘一致率 1.000
- `pit_available_at` **PASS**：1,782 万行零泄漏
- 2 项 WARN：`universe_completeness`（5,892/5,894，即上述 2 只未上市）与
  `suspension_representation`（停牌快照窗口起点限制）

## 7. 证书结果

| 证书 | 结果 |
|---|---|
| `u0_bar_readiness_certificate.json` | **`U0_BAR_READY`**（identity / provider / coverage / quality 全 PASS） |
| `full_universe_readiness_certificate.json` | **`FULL_UNIVERSE_DATA_NOT_READY_PIT`** |
| `u0_strict_pit_certificate.json` | `FULL_UNIVERSE_DATA_NOT_READY_PIT`，`training_permitted = false` |

严格证书的**唯一**剩余阻塞项是 `st_intervals`。覆盖门此前为 FAIL，本次因交易所
名录证据而通过——`covered = expected = 5,892`，`unexplained_uncovered = []`。

覆盖门的放行是**证据驱动**的，不是放宽：

- 只有 `master_disposition.parquet` 存在时才允许从分母中剔除证券；
- 剔除条件是交易所名录中不存在 **且** 所有 provider 都未返回过 bar；
- 缺少该证据文件时，门恢复"每一只 master 证券都必须被覆盖"的严格判定。

三条回归测试锁定该语义（无证据即阻塞、已确认未上市可剔除、剔除一只不得掩盖另一只缺口）。

## 8. 本次**未完成**的工作

以下阶段本次会话没有完成，不做任何完成度上的模糊表述：

1. **历史 ST 区间重建（沪市 / 北交所）** —— 未做公告级重建。仅深交所简称变更
   登记可用。严格证书因此仍为 `NOT_READY_PIT`。
2. **训练切片证书（`FULL_UNIVERSE_TRAINING_DATA_READY`）** —— 未实现。
3. **字段级特征覆盖统计** —— `amount` / `volume` 单位与覆盖率已实测（见第 6 节），
   但未产出按 provider / 板块 / 日期的逐特征覆盖表，也未建立缺失掩码。
4. **U0 代码清理（Phase F）** —— 未执行。
5. **全宇宙训练（Phase G/H/I）** —— **未启动**。已定位关键缺口：
   `train-v8-deep` 消费的是 **gold 训练数据集**（`runtime/data/v7/gold/training_dataset/*.parquet`，
   约 8GB，基于冻结的 3,872 只 qfq 队列），而 U0 面板是 **raw 未复权全宇宙**。
   两者之间**目前没有可执行的连接路径**。要打通需要：
   1. 用 `pit/adjust_factors.parquet` 把 raw 面板转成复权面板（因子已齐备，5,891 只）；
   2. 在复权面板上重算 alpha 特征与可执行标签；
   3. 产出全宇宙 gold 数据集；
   4. 再跑 `train-v8-deep`。
   本次未执行上述任何一步，因此**没有** smoke training 结果，**没有** 训练进程、
   PID、checkpoint 或 epoch 证据。

## 9. 复现命令

```bash
# 交易所名录核对（产出 master_disposition.parquet）
AI_quant_venv/bin/python3 scripts/u0_exchange_register_reconcile.py --allow-network

# 重新装配面板
AI_quant_venv/bin/python3 scripts/u0_assemble_panel.py

# 校验 + 取证 + 三份证书
AI_quant_venv/bin/python3 scripts/u0_validate.py --allow-network
AI_quant_venv/bin/python3 scripts/u0_adjustment_forensics.py
AI_quant_venv/bin/python3 scripts/u0_audit.py
AI_quant_venv/bin/python3 scripts/u0_bar_readiness.py
AI_quant_venv/bin/python3 scripts/u0_pit_readiness.py
```

继续采集（两只未上市证券无需再抓；如需补齐 transient 失败）：

```bash
AI_quant_venv/bin/python3 scripts/u0_acquire_bars.py --allow-network \
    --providers tickflow --staging-name tickflow_raw --max-minutes 600
```

## 10. 外部阻塞

| 项 | 状态 | 原因 |
|---|---|---|
| Level-2 | BLOCKED | TickFlow `depth.get` 未授权；公开源仅五档 L1；QMT 为 Windows 客户端 |
| 深历史分钟线 | BLOCKED | TickFlow 日内未授权；公开源仅滚动窗口 |
| 沪市 / 北交所历史 ST | BLOCKED_BY_DATA | 未找到带日期的简称变更登记；baostock 逐日 `isST` 需 TCP 10030 |
| 上交所上市名录 | 间歇不可用 | `query.sse.com.cn` 本次返回 0 行 |
| baostock / mootdx | BLOCKED_BY_ENVIRONMENT | 运行环境只放行 80/443 |
| TuShare | NO_CREDENTIAL | 未配置 `TUSHARE_TOKEN` |
