# 全宇宙 Gold 构建报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **实现**：`src/quantagent/data/ashare/gold_bridge.py`（本次经 PR #26 恢复）
- **产物**：`runtime/data/gold/full_universe_smoke/`

## 1. 结论

Gold 桥**已恢复且可执行**，但**全量构建未进行**，因为 PIT 未闭环时产物无论如何
不可训练。

冒烟构建（此前实测，代码经本次恢复后一致）：

| 项 | 值 |
| --- | --- |
| 行数 | 140,994 |
| 标的数 | 250（跨全部板块） |
| 日期范围 | 2024-01-02 .. 2026-07-23 |
| 复权方式 | **hfq**（单一口径） |
| 复权因子版本 | `0d0911395d9243ad` |
| 数据集内容哈希 | `16160d22b3b42935` |
| 可训练行 | 136,001 / 140,994 |

## 2. 满足的构建要求

| 要求 | 实现 |
| --- | --- |
| 不混用 raw 与复权价 | API 层**无逐列复权参数**，混用写不出来 |
| 显式复权模式 | 单一 `adjustment_method`，写入 manifest |
| 成交量不随价格缩放 | volume/amount 永不参与价格缩放（实测 618/618 行价格变、量不变） |
| 延迟可执行标签 | `close(t+1+h)/close(t+1)-1` |
| T+1 | 入场价取 t+1 |
| 涨停入场限制 | t+1 封涨停的行被丢弃 |
| 停牌日排除 | t 或 t+1 停牌丢弃 |
| 上市前排除 | `mask_pre_listing` |
| 退市后排除 | `mask_post_delisting` |
| 缺失特征掩码 | 三态掩码（含 `UNKNOWN`） |
| **QMT 可得性掩码** | `has_tick_events` / `has_level2_snapshot` / `has_level2_order_events` / `has_minute_bars` |
| 数据集哈希 | `content_hash` |
| 源提交 | `source_commit` |
| 供应商血缘 | `inputs` + `patch_provenance`（对账打补丁时） |
| 确定性重建命令 | manifest 内 `rebuild_command` |

## 3. 三态掩码为何必要

`UNKNOWN` 与 `FALSE` 必须不同：

- "本交易所没有 ST 登记" ≠ "这只票不是 ST"；
- U0 的 ST 登记只覆盖深交所，故 `mask_is_st` 当前全为 `UNKNOWN`。

冒烟构建的掩码分布：

```
mask_is_suspended    : FALSE 140,994
mask_is_st           : UNKNOWN 140,994    ← 登记不完整
mask_pre_listing     : FALSE 140,994
mask_post_delisting  : FALSE 140,994
mask_seasoning       : FALSE 136,001 / TRUE 4,993
```

## 4. 近期 QMT tick 覆盖**不是**日线基线的前置条件

按任务要求，`has_*` 是可得性指示器而非必需字段。没有 tick 数据的 security-date
行**不会**被当作"订单流为 0"，但也**不阻塞**日线 Gold 构建。

## 5. 训练闸门

**`TRAINING_BLOCKED`**，两条独立阻塞：

```
1. U0 PIT gate withholds training permission:
   FULL_UNIVERSE_DATA_NOT_READY_PIT (blocked fields: ['st_intervals'])
2. gold build reported an incomplete PIT mask
```

构建脚本返回**退出码 2**，串联训练的调用方会停下。

## 6. 未决

- 全量（5,892 只）Gold **未构建**——PIT 解除前无意义；
- QMT 可得性掩码当前全为 `false`（无 QMT 数据）。
