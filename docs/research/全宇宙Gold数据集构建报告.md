# 全宇宙 Gold 数据集构建报告

- **生成时间**：2026-07-27（UTC）
- **源提交**：`0b9adbe6e2450c3dd3238e39e17963d0a74a4f1e`
- **实现**：`src/quantagent/data/ashare/gold_bridge.py`
- **构建脚本**：`scripts/build_full_universe_gold.py`
- **产物**：`runtime/data/gold/full_universe_smoke/`

## 1. 这座桥要防的具体事故

本仓库有过一次**已记录在案**的事故：一个全宇宙面板**混用了 qfq 与 raw 价格，却
把复权方式声明为 "none"**，并且通过了自己的审计——因为当时的闸门是字面量 `True`。

本模块的每一项设计都针对该事故。

## 2. 四条硬性设计

### 2.1 一次构建只有一种复权口径，且混用在 API 层不可达

- `build_gold_dataset` 只接受**一个** `adjustment_method`；
- **没有**逐列复权参数，因此"close 用 qfq、volume 用 raw"写不出来；
- 声明 hfq 却不给因子表 → **抛异常**，而不是发出"声明的口径 ≠ 实际口径"的数据；
- **成交量与成交额永不随价格缩放**——它们是真实成交数量，跟着价格缩放正是当年的
  混用 bug。

**实测验证**（真实 U0 数据）：`000062.SZ` 在窗口内有 **7 个不同的除权因子**
（7.77 → 8.20），复权后收盘价与原始价在**全部 618 行**上都不同，成交量保持不变。

### 2.2 掩码是三态，`UNKNOWN` 不等于 `FALSE`

"本交易所没有 ST 登记"与"这只票不是 ST"**必须**产生不同的列值。U0 的 ST 登记
只覆盖深交所，因此 `st_available=False` 时 `mask_is_st` 全为 `UNKNOWN`。

掩码清单：`mask_is_suspended`、`mask_is_st`、`mask_pre_listing`、
`mask_post_delisting`、`mask_seasoning`。

**次新股锁定期按面板内观测到的交易日计数**，不按自然日——否则节假日会缩短窗口。

### 2.3 可得性是一等列

`has_tick_events`、`has_level2_snapshot`、`has_level2_order_events`、
`has_minute_bars`。

**没有 tick 数据的 security-date 行绝不能被读成"订单流为 0"的行。** 下游特征必须
查这个指示器，而不是直接用 0。

### 2.4 标签是 delay-1 可执行

```
forward_return_{h}d = close(t+1+h) / close(t+1) - 1
```

与 `tests/test_executable_label_convention.py` 锁定的口径一致。

**入场不可行的行被丢弃**，而不是留在一个没人能成交的价格上：

| 丢弃理由 | 说明 |
| --- | --- |
| t 日停牌 | |
| t+1 日停牌 | 无法入场 |
| t 日 ST | |
| **t+1 日封涨停** | 买不进封死的涨停——旧的幻影 alpha 正是从这里来的 |
| t+1 日无收盘价 | 序列尾部 |

## 3. 冒烟构建结果（真实执行）

命令：

```bash
python scripts/build_full_universe_gold.py --max-symbols 250 --start-date 2024-01-01 \
    --output runtime/data/gold/full_universe_smoke
```

| 项 | 值 |
| --- | --- |
| 行数 | **140,994** |
| 标的数 | 250（跨全部板块，非单一板块抽样） |
| 日期范围 | 2024-01-02 .. 2026-07-23 |
| 复权方式 | **hfq** |
| 复权因子版本 | `0d0911395d9243ad` |
| 数据集内容哈希 | `16160d22b3b42935` |
| 标签列 | `forward_return_{1,5,20}d` |
| 可训练行 | 136,001 / 140,994 |

掩码分布：

```
mask_is_suspended    : FALSE 140,994
mask_is_st           : UNKNOWN 140,994      ← ST 登记不完整
mask_pre_listing     : FALSE 140,994
mask_post_delisting  : FALSE 140,994
mask_seasoning       : FALSE 136,001 / TRUE 4,993
```

丢弃统计：

```
suspended_at_t        : 6
suspended_at_t1       : 1
st_at_t               : 0     ← ST 未知，故无法据此排除任何行
entry_price_missing   : 250   ← 每只标的的序列尾部各一行
rows_dropped_total    : 257
```

## 4. 训练闸门：**推导，不是断言**

`certify_training_slice()` 的权威是 U0 的 PIT 证书，而**不是**"构建成功"这件事。
构建成功的数据集不等于可训练的数据集。

**本次判定 = `TRAINING_BLOCKED`**，两条独立阻塞：

```
1. U0 PIT gate withholds training permission:
   FULL_UNIVERSE_DATA_NOT_READY_PIT (blocked fields: ['st_intervals'])
2. gold build reported an incomplete PIT mask
```

脚本据此**返回退出码 2**（已实测），使得任何串联训练的调用方会停下，而不是在
不可用的数据集上继续。

另有两条测试锁定该闸门不会退化：

- `test_missing_certificate_is_not_permission` —— 没有证书**不等于**有许可；
- `test_a_successful_build_alone_does_not_permit_training` —— 即使 U0 放行，
  若构建自身报告 PIT 掩码不完整，仍然阻塞。

## 5. Manifest 内容

`runtime/data/gold/full_universe_smoke/build_manifest.json` 含：
生成时间、源提交、复权方式与因子版本、行/标的/日期范围、特征列与覆盖率、
掩码列与分布、可得性列、标签列与口径说明、丢弃统计、内容哈希、
**可复现命令**、输入文件清单、警告。

## 6. 解除阻塞的唯一路径

`st_intervals` 目前：

```
BLOCKED_BY_DATA — PARTIAL: 906 dated episodes over 651 securities from SZSE;
no dated register for BSE, SSE
```

已识别的候选来源：QMT/xtdata 的 `get_his_st_data` / `download_his_st_data`
（详见 `QMT_XtData权限与数据能力报告.md` §5）。需券商权限 + Windows 主机，
**尚未验证**。

## 7. 尚未执行的部分（诚实声明）

- **全量构建未跑**：本次只跑了 250 只标的的冒烟。全量（5,892 只）在 PIT 解除
  之前没有意义，因为产物无论如何都不可训练。
- **训练未启动**：闸门阻塞，这是正确结果，不是桥的缺陷。
