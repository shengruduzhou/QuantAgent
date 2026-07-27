# U0 PIT 闭环报告

- **生成时间**：2026-07-28（UTC）
- **源提交**：`9787de0727d2caa8c3720891f1af8b4af9b4017d`
- **证据**：`runtime/data/u0/u0_strict_pit_certificate.json`、`u0_bar_readiness_certificate.json`

## 1. 结论：PIT **未**闭环

```
decision            = FULL_UNIVERSE_DATA_NOT_READY_PIT
training_permitted  = false
blocked_pit_fields  = ["st_intervals"]
bar_decision        = U0_BAR_READY
```

**严格 PIT 要求未被弱化，`st_intervals` 未从强制字段中移除。**

## 2. 从产物独立复算的 U0 状态

不引用旧叙述，直接读产物：

| 项 | 值 |
| --- | --- |
| 证券主表 | **5,894** |
| 面板行数 | **17,829,080** |
| 面板标的数 | **5,892** |
| 日期范围 | 1990-12-19 .. 2026-07-24 |
| amount 覆盖率 | **98.75%** |
| BAR 闸门 | **`U0_BAR_READY`** |
| PIT 闸门 | **未通过** |

## 3. 唯一阻塞项

```
st_intervals: BLOCKED_BY_DATA — PARTIAL: 906 dated episodes over 651 securities
              from SZSE; no dated register for BSE, SSE; current state known
              for 333 names
```

只有**深交所**有带日期的简称变更登记；**沪市与北交所没有**。

**部分覆盖算 BLOCKED，不算通过。** 这条判定未被放宽。

## 4. 其余 PIT 字段（均已具备）

| 字段 | 状态 |
| --- | --- |
| listing_date | AVAILABLE（5894/5894） |
| delisting_date | AVAILABLE（361 条带日期退市） |
| trading_calendar | AVAILABLE（8797 个交易日） |
| price_limit_regime | AVAILABLE（交易所规则区间） |
| ipo_special_limit | AVAILABLE |
| corporate_action_identity | AVAILABLE（71,574 条除权因子 / 5,891 只） |
| suspension_intervals | AVAILABLE（2,157 条） |
| **st_intervals** | **BLOCKED_BY_DATA** |

## 5. 本次尝试的闭环路径与结果

**路径：QMT `download_his_st_data` / `get_his_st_data`。**

**结果：未能执行。** 本机 Linux，无 MiniQMT，判定 `PLATFORM_UNAVAILABLE`。

已实现但未能运行的探测（`probe_st_with_controls`）要求阳性对照：若已知曾 ST 的
标的也返回空，则判定数据集不可用，**任何标的都不得被记为 never-ST**。
详见 `QMT历史ST权限与覆盖报告.md`。

## 6. 为什么不实现"训练切片证书"旁路

任务允许：当**选定训练窗内每个 security-date 的 PIT 状态都已知**时，可实现独立的
训练切片证书。

**当前不满足**：沪市与北交所在整个历史区间都没有带日期的 ST 登记，因此不存在一个
"所有 PIT 状态已知"的全宇宙窗口。强行实现只会成为绕过闸门的通道。

因此**未实现**该旁路，训练维持阻塞。

## 7. 未决与后续路径

| 步骤 | 状态 |
| --- | --- |
| Windows 主机 + 券商 QMT 权限 | ❌ 不具备 |
| `get_his_st_data` 阳性对照探测 | ⏸ 待执行 |
| 沪市/北交所 ST 带日期登记重建（官方公告路径） | ⏸ 备选 |
| 严格 PIT 转 `FULL_UNIVERSE_DATA_READY` | ⏸ 阻塞中 |
